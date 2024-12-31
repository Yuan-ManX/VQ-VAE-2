import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn


class Quantize(nn.Module):
    """
    向量量化（Vector Quantization, VQ）模块。

    该模块用于将连续的特征映射到离散的码本中，通过找到最近的码字并进行量化。
    它还实现了码本的更新机制，使其能够在线学习并适应数据分布。
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        """
        初始化向量量化模块。

        参数:
            dim (int): 输入特征的维度。
            n_embed (int): 码本中码字的数量。
            decay (float, 可选): 码本更新的衰减因子，默认为 0.99。
            eps (float, 可选): 用于避免除以零的小常数，默认为 1e-5。
        """
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # 初始化码本，形状为 (dim, n_embed)
        embed = torch.randn(dim, n_embed)
        # 注册码本为缓冲区，不作为模型参数更新
        self.register_buffer("embed", embed)
        # 注册聚类大小缓冲区，初始化为全零
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        # 注册平均嵌入缓冲区，初始化为码本的副本
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        """
        前向传播函数，执行向量量化。

        参数:
            input (torch.Tensor): 输入张量，形状为 (batch_size, ..., dim)。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 量化后的张量、量化误差和嵌入索引。
        """
        # 将输入张量展平为形状 (batch_size * ..., dim)
        flatten = input.reshape(-1, self.dim)
        # 计算每个输入向量与每个码字之间的平方距离
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # 找到最近的码字索引
        # 距离的负值最大，即距离最小
        _, embed_ind = (-dist).max(1)
        # 将嵌入索引转换为 one-hot 编码
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # 将嵌入索引重塑为与输入相同的形状 (batch_size, ..., 1)
        embed_ind = embed_ind.view(*input.shape[:-1])
        # 根据嵌入索引获取量化后的嵌入
        quantize = self.embed_code(embed_ind)

        if self.training:
            # 如果在训练模式下，更新码本
            # 统计每个码字的 one-hot 总和
            embed_onehot_sum = embed_onehot.sum(0)
            # 计算每个码字的平均嵌入
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # 使用分布式通信函数 all_reduce 同步所有进程的码本更新
            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            # 更新聚类大小和平均嵌入
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            # 计算总聚类大小
            n = self.cluster_size.sum()
            # 计算每个码字的归一化大小
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            # 计算归一化的嵌入
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            # 更新码本
            self.embed.data.copy_(embed_normalized)

        # 计算量化误差
        diff = (quantize.detach() - input).pow(2).mean()
        # 计算最终的量化输出
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        """
        根据嵌入索引获取嵌入。

        参数:
            embed_id (torch.Tensor): 嵌入索引，形状为 (batch_size, ..., 1)。

        返回:
            torch.Tensor: 量化后的嵌入，形状为 (batch_size, ..., dim)。
        """
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    """
    残差块（Residual Block）类。

    残差块通过引入跳跃连接（skip connection），使得网络可以更深，同时缓解了梯度消失问题。
    该残差块包含两个卷积层，第一个卷积层改变通道数，第二个卷积层恢复原始通道数。
    """
    def __init__(self, in_channel, channel):
        """
        初始化残差块。

        参数:
            in_channel (int): 输入的通道数。
            channel (int): 第一个卷积层输出的通道数。
        """
        super().__init__()

        # 定义残差块中的卷积层序列
        self.conv = nn.Sequential(
            nn.ReLU(),  # 第一个 ReLU 激活函数
            nn.Conv2d(in_channel, channel, 3, padding=1),  # 3x3 卷积层，改变通道数
            nn.ReLU(inplace=True),  # 第二个 ReLU 激活函数，原地操作节省内存
            nn.Conv2d(channel, in_channel, 1),  # 1x1 卷积层，恢复原始通道数
        )

    def forward(self, input):
        """
        前向传播函数。

        参数:
            input (torch.Tensor): 输入张量，形状为 (batch_size, in_channel, H, W)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, in_channel, H, W)。
        """
        # 通过卷积层序列
        out = self.conv(input)
        # 加上输入，实现跳跃连接
        out += input

        return out


class Encoder(nn.Module):
    """
    编码器类，用于将输入图像映射到潜在空间表示。

    编码器由多个卷积层和残差块组成，逐步降低空间分辨率并增加通道数。
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        """
        初始化编码器。

        参数:
            in_channel (int): 输入图像的通道数。
            channel (int): 初始卷积层的输出通道数。
            n_res_block (int): 残差块的数目。
            n_res_channel (int): 残差块中卷积层的通道数。
            stride (int): 下采样步幅，决定了空间分辨率的降低程度。
        """
        super().__init__()

        # 根据步幅选择不同的卷积层配置
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        # 添加多个残差块
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        # 添加最终的 ReLU 激活函数
        blocks.append(nn.ReLU(inplace=True))

        # 将所有层组合成一个 Sequential 模块
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        前向传播函数。

        参数:
            input (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channel, H, W)。

        返回:
            torch.Tensor: 编码后的特征张量，形状为 (batch_size, channel, H', W')。
        """
        # 通过卷积层和残差块序列
        return self.blocks(input)


class Decoder(nn.Module):
    """
    解码器类，用于将潜在空间表示映射回图像。

    解码器由多个卷积层和残差块组成，逐步提高空间分辨率并减少通道数。
    """
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        """
        初始化解码器。

        参数:
            in_channel (int): 输入特征的通道数。
            out_channel (int): 输出图像的通道数。
            channel (int): 初始卷积层的输出通道数。
            n_res_block (int): 残差块的数目。
            n_res_channel (int): 残差块中卷积层的通道数。
            stride (int): 上采样步幅，决定了空间分辨率的提高程度。
        """
        super().__init__()

        # 初始 3x3 卷积层
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        # 添加多个残差块
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        # 添加最终的 ReLU 激活函数
        blocks.append(nn.ReLU(inplace=True))

        # 根据步幅选择不同的上采样层配置
        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1), # 4x4 转置卷积层，步幅为2
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1 # 4x4 转置卷积层，步幅为2
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1) # 4x4 转置卷积层，步幅为2
            )

        # 将所有层组合成一个 Sequential 模块
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """
        前向传播函数。

        参数:
            input (torch.Tensor): 输入特征张量，形状为 (batch_size, in_channel, H, W)。

        返回:
            torch.Tensor: 解码后的图像张量，形状为 (batch_size, out_channel, H', W')。
        """
        # 通过卷积层和残差块序列
        return self.blocks(input)


class VQVAE(nn.Module):
    """
    矢量量化变分自编码器（Vector Quantized Variational AutoEncoder, VQ-VAE）类。

    VQ-VAE 通过编码器将输入图像映射到潜在空间，并通过矢量量化将连续的特征映射到离散的码本中。
    解码器则从量化后的潜在表示中重建输入图像。
    """
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        """
        初始化 VQ-VAE 模型。

        参数:
            in_channel (int, 可选): 输入图像的通道数，默认为 3。
            channel (int, 可选): 初始卷积层的通道数，默认为 128。
            n_res_block (int, 可选): 残差块的数目，默认为 2。
            n_res_channel (int, 可选): 残差块中卷积层的通道数，默认为 32。
            embed_dim (int, 可选): 嵌入向量的维度，默认为 64。
            n_embed (int, 可选): 码本中码字的数量，默认为 512。
            decay (float, 可选): 码本更新的衰减因子，默认为 0.99。
        """
        super().__init__()

        # 初始化编码器部分
        # 底部编码器
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # 顶部编码器
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)

        # 初始化量化卷积层和量化模块
        # 顶部量化卷积层
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        # 顶部量化模块
        self.quantize_t = Quantize(embed_dim, n_embed)
        # 顶部解码器
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )

        # 初始化底部量化卷积层和量化模块
        # 底部量化卷积层
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        # 底部量化模块
        self.quantize_b = Quantize(embed_dim, n_embed)

        # 初始化上采样层
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        ) # 顶部上采样层

        # 初始化最终解码器
        self.dec = Decoder(
            embed_dim + embed_dim,   # 输入通道数
            in_channel,              # 输出通道数
            channel,                 # 初始卷积层的通道数
            n_res_block,             # 残差块的数目
            n_res_channel,           # 残差块中卷积层的通道数
            stride=4,                # 上采样步幅
        )

    def forward(self, input):
        """
        前向传播函数。

        参数:
            input (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channel, H, W)。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 解码后的图像和量化误差。
        """
        # 编码并量化
        quant_t, quant_b, diff, _, _ = self.encode(input)
        # 解码
        dec = self.decode(quant_t, quant_b)

        # 返回解码后的图像和量化误差
        return dec, diff

    def encode(self, input):
        """
        编码函数，将输入图像编码为量化后的潜在表示。

        参数:
            input (torch.Tensor): 输入图像张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                顶部量化表示、底部量化表示、量化误差、顶部嵌入索引、底部嵌入索引。
        """
        # 底部编码
        enc_b = self.enc_b(input)
        # 顶部编码
        enc_t = self.enc_t(enc_b)

        # 顶部量化卷积
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        # 顶部量化
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        # 转置维度
        quant_t = quant_t.permute(0, 3, 1, 2)
        # 增加维度
        diff_t = diff_t.unsqueeze(0)

        # 顶部解码
        dec_t = self.dec_t(quant_t)
        # 合并顶部解码和底部编码
        enc_b = torch.cat([dec_t, enc_b], 1)
        
        # 底部量化卷积
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        # 底部量化
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        # 转置维度
        quant_b = quant_b.permute(0, 3, 1, 2)
        # 增加维度
        diff_b = diff_b.unsqueeze(0)

        # 返回量化表示和量化误差
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        """
        解码函数，将量化后的潜在表示解码为重建图像。

        参数:
            quant_t (torch.Tensor): 顶部量化表示。
            quant_b (torch.Tensor): 底部量化表示。

        返回:
            torch.Tensor: 解码后的图像。
        """
        # 顶部上采样
        upsample_t = self.upsample_t(quant_t)
        # 合并顶部上采样和底部量化
        quant = torch.cat([upsample_t, quant_b], 1)
        # 最终解码
        dec = self.dec(quant)

        # 返回解码后的图像
        return dec

    def decode_code(self, code_t, code_b):
        """
        通过码字索引解码图像。

        参数:
            code_t (torch.Tensor): 顶部码字索引。
            code_b (torch.Tensor): 底部码字索引。

        返回:
            torch.Tensor: 解码后的图像。
        """
        quant_t = self.quantize_t.embed_code(code_t)  # 顶部嵌入码
        quant_t = quant_t.permute(0, 3, 1, 2)  # 转置维度
        quant_b = self.quantize_b.embed_code(code_b)  # 底部嵌入码
        quant_b = quant_b.permute(0, 3, 1, 2)  # 转置维度

        # 解码
        dec = self.decode(quant_t, quant_b)

        # 返回解码后的图像
        return dec
