import os
import pickle
from collections import namedtuple
import lmdb

import torch
from torch.utils.data import Dataset
from torchvision import datasets


# 定义一个具名元组，用于存储编码行数据，包括顶部图像、底部图像和文件名
CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    """
    自定义的 ImageFolder 数据集类，继承自 PyTorch 的 ImageFolder。

    该类扩展了原始的 ImageFolder 数据集，增加了返回图像文件名和类别名的功能。
    """
    def __getitem__(self, index):
        """
        获取指定索引的样本数据。

        参数:
            index (int): 样本的索引。

        返回:
            Tuple[torch.Tensor, int, str]: 图像样本、目标标签和文件名。
        """
        # 调用父类的 __getitem__ 方法获取样本和目标标签
        sample, target = super().__getitem__(index)
        # 获取样本的路径和文件名
        path, _ = self.samples[index]
        # 分离目录和文件名
        dirs, filename = os.path.split(path)
        # 分离父目录和类别名
        _, class_name = os.path.split(dirs)
        # 构建新的文件名路径，格式为 "类别名/文件名"
        filename = os.path.join(class_name, filename)

        # 返回样本、目标标签和新的文件名
        return sample, target, filename


class LMDBDataset(Dataset):
    """
    LMDB 数据集类，用于从 LMDB 数据库中读取数据。

    该类使用 LMDB 作为后端存储，支持高效地读取大量数据。
    """
    def __init__(self, path):
        """
        初始化 LMDB 数据集。

        参数:
            path (str): LMDB 数据库的路径。
        """
        # 打开 LMDB 环境
        self.env = lmdb.open(
            path,                # 数据库路径
            max_readers=32,      # 最大读取器数量
            readonly=True,       # 以只读模式打开
            lock=False,          # 不使用文件锁
            readahead=False,     # 禁用预读
            meminit=False,       # 不初始化内存
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        """
        返回数据集的长度。

        返回:
            int: 数据集的长度。
        """
        return self.length

    def __getitem__(self, index):
        """
        获取指定索引的样本数据。

        参数:
            index (int): 样本的索引。

        返回:
            Tuple[torch.Tensor, torch.Tensor, str]: 顶部图像、底部图像和文件名。
        """
        with self.env.begin(write=False) as txn:
            # 将索引转换为字符串并编码为 UTF-8 字节
            key = str(index).encode('utf-8')
            # 从数据库中获取对应键的值，并反序列化
            row = pickle.loads(txn.get(key))
            
        # 将顶部和底部图像从 NumPy 数组转换为 PyTorch 张量
        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename
