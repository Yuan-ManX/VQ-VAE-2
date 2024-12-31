from math import cos, pi, floor, sin
from torch.optim import lr_scheduler


class CosineLR(lr_scheduler._LRScheduler):
    """
    余弦退火学习率调度器（Cosine Annealing Learning Rate Scheduler）。

    该调度器在学习率调度过程中采用余弦函数进行退火，使得学习率在每个步长内从最大值下降到最小值，然后重新开始。
    """
    def __init__(self, optimizer, lr_min, lr_max, step_size):
        """
        初始化余弦退火学习率调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_min (float): 学习率的最小值。
            lr_max (float): 学习率的最大值。
            step_size (int): 每个余弦周期的步长。
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        self.iteration = 0

        # 调用父类的初始化方法，-1 表示不设置初始学习率
        super().__init__(optimizer, -1)

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            List[float]: 优化器中每个参数组的学习率列表。
        """
        # 计算当前迭代的学习率，使用余弦函数进行退火
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + cos(self.iteration / self.step_size * pi)
        )
        # 迭代次数加一
        self.iteration += 1

        if self.iteration == self.step_size:
            # 如果达到一个完整的周期，重置迭代次数
            self.iteration = 0

        # 返回学习率列表
        return [lr for base_lr in self.base_lrs]


class PowerLR(lr_scheduler._LRScheduler):
    """
    幂律衰减学习率调度器（Power Law Decay Learning Rate Scheduler）。

    该调度器在学习率调度过程中采用幂律函数进行衰减，使得学习率随着迭代次数的增加而逐渐减小。
    """
    def __init__(self, optimizer, lr_min, lr_max, warmup):
        """
        初始化幂律衰减学习率调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_min (float): 学习率的最小值。
            lr_max (float): 学习率的最大值。
            warmup (int): 热身阶段的步数。
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup = warmup
        self.iteration = 0

        # 调用父类的初始化方法，-1 表示不设置初始学习率
        super().__init__(optimizer, -1)

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            List[float]: 优化器中每个参数组的学习率列表。
        """
        if self.iteration < self.warmup:
            # 在热身阶段，学习率线性增加
            lr = (
                self.lr_min + (self.lr_max - self.lr_min) / self.warmup * self.iteration
            )

        else:
            # 在衰减阶段，学习率按照幂律衰减
            lr = self.lr_max * (self.iteration - self.warmup + 1) ** -0.5

        # 迭代次数加一
        self.iteration += 1

        return [lr for base_lr in self.base_lrs]


class SineLR(lr_scheduler._LRScheduler):
    """
    正弦波动学习率调度器（Sine Wave Learning Rate Scheduler）。

    该调度器在学习率调度过程中采用正弦函数进行波动，使得学习率在每个步长内从最小值波动到最大值，然后重新开始。
    """
    def __init__(self, optimizer, lr_min, lr_max, step_size):
        """
        初始化正弦波动学习率调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_min (float): 学习率的最小值。
            lr_max (float): 学习率的最大值。
            step_size (int): 每个正弦周期的步长。
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        self.iteration = 0

        # 调用父类的初始化方法，-1 表示不设置初始学习率
        super().__init__(optimizer, -1)

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            List[float]: 优化器中每个参数组的学习率列表。
        """
        # 计算当前迭代的学习率，使用正弦函数进行波动
        lr = self.lr_min + (self.lr_max - self.lr_min) * sin(
            self.iteration / self.step_size * pi
        )
        self.iteration += 1

        if self.iteration == self.step_size:
            # 如果达到一个完整的周期，重置迭代次数
            self.iteration = 0

        # 返回学习率列表
        return [lr for base_lr in self.base_lrs]


class LinearLR(lr_scheduler._LRScheduler):
    """
    线性学习率调度器（Linear Learning Rate Scheduler）。

    该调度器在热身阶段保持最大学习率，之后以线性方式降低学习率，直到达到最小学习率。
    """
    def __init__(self, optimizer, lr_min, lr_max, warmup, step_size):
        """
        初始化线性学习率调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_min (float): 学习率的最小值。
            lr_max (float): 学习率的最大值。
            warmup (int): 热身阶段的步数。
            step_size (int): 学习率调度器的总步数。
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        self.warmup = warmup
        self.iteration = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            List[float]: 优化器中每个参数组的学习率列表。
        """
        if self.iteration < self.warmup:
            # 在热身阶段，学习率保持为最大学习率
            lr = self.lr_max

        else:
            # 在线性衰减阶段，学习率线性降低
            lr = self.lr_max + (self.iteration - self.warmup) * (
                self.lr_min - self.lr_max
            ) / (self.step_size - self.warmup)
        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        return [lr for base_lr in self.base_lrs]


class CLR(lr_scheduler._LRScheduler):
    """
    循环学习率调度器（Cyclic Learning Rate Scheduler）。

    该调度器在每个周期内学习率呈三角波形变化，从最小值上升到最大值，再下降到最小值。
    """
    def __init__(self, optimizer, lr_min, lr_max, step_size):
        """
        初始化循环学习率调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_min (float): 学习率的最小值。
            lr_max (float): 学习率的最大值。
            step_size (int): 每个周期的步数。
        """
        self.epoch = 0
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.current_lr = lr_min
        self.step_size = step_size

        super().__init__(optimizer, -1)

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            List[float]: 优化器中每个参数组的学习率列表。
        """
        # 计算当前周期
        cycle = floor(1 + self.epoch / (2 * self.step_size))
        # 计算当前周期内的位置
        x = abs(self.epoch / self.step_size - 2 * cycle + 1)
        # 计算学习率
        lr = self.lr_min + (self.lr_max - self.lr_min) * max(0, 1 - x)
        # 更新当前学习率
        self.current_lr = lr

        self.epoch += 1

        # 返回学习率列表
        return [lr for base_lr in self.base_lrs]


class Warmup(lr_scheduler._LRScheduler):
    """
    热身学习率调度器（Warmup Learning Rate Scheduler）。

    该调度器在学习初期以较快的速度增加学习率，然后在达到最大学习率后以较慢的速度降低学习率。
    """
    def __init__(self, optimizer, model_dim, factor=1, warmup=16000):
        """
        初始化热身学习率调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            model_dim (int): 模型的维度，用于计算学习率。
            factor (float, 可选): 学习率调整因子，默认为 1。
            warmup (int, 可选): 热身阶段的步数，默认为 16000。
        """
        self.optimizer = optimizer
        self.model_dim = model_dim
        self.factor = factor
        self.warmup = warmup
        self.iteration = 0

        # 调用父类的初始化方法，-1 表示不设置初始学习率
        super().__init__(optimizer, -1)

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            List[float]: 优化器中每个参数组的学习率列表。
        """
        # 迭代次数加一
        self.iteration += 1
        # 计算学习率
        lr = (
            self.factor
            * self.model_dim ** (-0.5)
            * min(self.iteration ** (-0.5), self.iteration * self.warmup ** (-1.5))
        )

        # 返回学习率列表
        return [lr for base_lr in self.base_lrs]


class CycleAnnealScheduler:
    """
    循环退火调度器（Cyclic Annealing Scheduler）。

    该调度器在每个周期内学习率和动量呈周期性变化，以帮助模型更好地收敛。
    """
    def __init__(
        self, optimizer, lr_max, lr_divider, cut_point, step_size, momentum=None
    ):
        """
        初始化循环退火调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_max (float): 学习率的最大值。
            lr_divider (float): 学习率的分频器，用于降低学习率。
            cut_point (float): 循环的切割点，表示每个周期内学习率下降的时机。
            step_size (int): 每个周期的步数。
            momentum (tuple, 可选): 动量范围，默认为 None。
        """
        self.lr_max = lr_max
        self.lr_divider = lr_divider
        self.cut_point = step_size // cut_point
        self.step_size = step_size
        self.iteration = 0
        self.cycle_step = int(step_size * (1 - cut_point / 100) / 2)
        self.momentum = momentum
        self.optimizer = optimizer

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            float: 当前学习率。
        """
        if self.iteration > 2 * self.cycle_step:
            # 在第二个下降阶段
            cut = (self.iteration - 2 * self.cycle_step) / (
                self.step_size - 2 * self.cycle_step
            )
            lr = self.lr_max * (1 + (cut * (1 - 100) / 100)) / self.lr_divider

        elif self.iteration > self.cycle_step:
            # 在第一个下降阶段
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        else:
            # 在上升阶段
            cut = self.iteration / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        return lr

    def get_momentum(self):
        """
        计算当前迭代的动量。

        返回:
            float: 当前动量。
        """
        if self.iteration > 2 * self.cycle_step:
            # 在第二个下降阶段，动量为最小值
            momentum = self.momentum[0]

        elif self.iteration > self.cycle_step:
            # 在第一个下降阶段，动量线性增加
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        else:
            # 在上升阶段，动量线性增加
            cut = self.iteration / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        return momentum

    def step(self):
        """
        更新学习率和动量。

        返回:
            float: 当前学习率。
        """
        lr = self.get_lr()

        if self.momentum is not None:
            momentum = self.get_momentum()

        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                group['betas'] = (momentum, group['betas'][1])

        return lr


def anneal_linear(start, end, proportion):
    """
    线性退火函数。

    参数:
        start (float): 起始值。
        end (float): 结束值。
        proportion (float): 当前进度比例（0 到 1 之间）。

    返回:
        float: 退火后的值。
    """
    return start + proportion * (end - start)


def anneal_cos(start, end, proportion):
    """
    余弦退火函数。

    参数:
        start (float): 起始值。
        end (float): 结束值。
        proportion (float): 当前进度比例（0 到 1 之间）。

    返回:
        float: 退火后的值。
    """
    # 计算余弦值并偏移到 [0, 2] 范围
    cos_val = cos(pi * proportion) + 1
    # 计算退火后的值
    return end + (start - end) / 2 * cos_val


class Phase:
    """
    阶段类，用于定义一个学习率或动量调整的阶段。

    每个阶段有起始值、结束值、迭代次数和退火函数。
    """
    def __init__(self, start, end, n_iter, anneal_fn):
        """
        初始化阶段。

        参数:
            start (float): 阶段的起始值。
            end (float): 阶段的结束值。
            n_iter (int): 阶段的迭代次数。
            anneal_fn (callable): 退火函数，用于计算当前值。
        """
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = 0

    def step(self):
        """
        执行一个步骤，计算当前值并增加迭代计数。

        返回:
            float: 当前值。
        """
        self.n += 1  # 增加迭代计数
        # 计算当前值
        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        """
        重置阶段，将迭代计数归零。
        """
        self.n = 0

    @property
    def is_done(self):
        """
        检查阶段是否完成。

        返回:
            bool: 如果迭代计数达到或超过迭代次数，则返回 True，否则返回 False。
        """
        return self.n >= self.n_iter


class CycleScheduler:
    """
    循环调度器类，用于在训练过程中循环调整学习率和动量。

    该调度器包含两个阶段：热身阶段和衰减阶段，每个阶段使用不同的退火函数。
    """
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        momentum=(0.95, 0.85),
        divider=25,
        warmup_proportion=0.3,
        phase=('linear', 'cos'),
    ):
        """
        初始化循环调度器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_max (float): 学习率的最大值。
            n_iter (int): 总迭代次数。
            momentum (tuple, 可选): 动量范围，默认为 (0.95, 0.85)。
            divider (float, 可选): 学习率的分频器，用于计算最小学习率，默认为 25。
            warmup_proportion (float, 可选): 热身阶段的比例，默认为 0.3。
            phase (tuple, 可选): 阶段的退火函数类型，默认为 ('linear', 'cos')。
        """
        self.optimizer = optimizer

        # 计算每个阶段的迭代次数
        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1

        # 计算最小学习率
        lr_min = lr_max / divider

        # 定义阶段映射字典，将字符串映射到退火函数
        phase_map = {'linear': anneal_linear, 'cos': anneal_cos}

        # 初始化学习率阶段
        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, phase_map[phase[0]]),  # 热身阶段
            Phase(lr_max, lr_min / 1e4, phase2, phase_map[phase[1]]),  # 衰减阶段
        ]

        # 动量范围
        self.momentum = momentum

        if momentum is not None:
            # 如果提供了动量范围，则初始化动量阶段
            mom1, mom2 = momentum
            self.momentum_phase = [
                Phase(mom1, mom2, phase1, phase_map[phase[0]]),  # 热身阶段
                Phase(mom2, mom1, phase2, phase_map[phase[1]]),  # 衰减阶段
            ]

        else:
            # 如果没有提供动量范围，则为空列表
            self.momentum_phase = []

        # 当前阶段索引
        self.phase = 0

    def step(self):
        """
        执行一个调度步骤，更新学习率和动量。

        返回:
            Tuple[float, Optional[float]]:: 当前学习率和动量。
        """
        # 计算当前学习率
        lr = self.lr_phase[self.phase].step()

        if self.momentum is not None:
            # 计算当前动量
            momentum = self.momentum_phase[self.phase].step()

        else:
            momentum = None

        # 更新优化器的学习率和动量
        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                if 'betas' in group:
                    group['betas'] = (momentum, group['betas'][1])

                else:
                    group['momentum'] = momentum

        # 如果当前阶段完成，则切换到下一个阶段
        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        # 如果所有阶段完成，则重置所有阶段
        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            for phase in self.momentum_phase:
                phase.reset()

            self.phase = 0

        # 返回当前学习率和动量
        return lr, momentum


class LRFinder(lr_scheduler._LRScheduler):
    """
    学习率查找器（Learning Rate Finder）类。

    该类用于在训练初期逐步增加学习率，以找到合适的学习率范围，从而帮助选择最佳的学习率。
    """
    def __init__(self, optimizer, lr_min, lr_max, step_size, linear=False):
        """
        初始化学习率查找器。

        参数:
            optimizer (Optimizer): 优化器实例。
            lr_min (float): 学习率的最小值。
            lr_max (float): 学习率的最大值。
            step_size (int): 学习率查找的总步数。
            linear (bool, 可选): 是否使用线性增长方式，默认为 False（使用指数增长）。
        """
        # 计算学习率增长的比率
        ratio = lr_max / lr_min
        # 是否使用线性增长
        self.linear = linear
        # 学习率的最小值
        self.lr_min = lr_min
        # 计算学习率增长的倍数
        self.lr_mult = (ratio / step_size) if linear else ratio ** (1 / step_size)
        # 当前迭代次数
        self.iteration = 0
        # 存储学习率的历史记录
        self.lrs = []
        # 存储损失函数值的记录
        self.losses = []

        super().__init__(optimizer, -1)

    def get_lr(self):
        """
        计算当前迭代的学习率。

        返回:
            List[float]: 优化器中每个参数组的学习率列表。
        """
        # 计算当前学习率
        lr = (
            self.lr_mult * self.iteration
            if self.linear
            else self.lr_mult ** self.iteration
        )
        # 根据是否线性增长调整学习率
        lr = self.lr_min + lr if self.linear else self.lr_min * lr

        # 迭代次数加一
        self.iteration += 1
        # 记录当前学习率
        self.lrs.append(lr)

        # 返回学习率列表
        return [lr for base_lr in self.base_lrs]

    def record(self, loss):
        """
        记录当前的损失函数值。

        参数:
            loss (float): 当前迭代的损失函数值。
        """
        self.losses.append(loss)

    def save(self, filename):
        """
        将学习率和对应的损失函数值保存到文件中。

        参数:
            filename (str): 文件名，用于保存学习率和损失函数值。
        """
        with open(filename, 'w') as f:
            # 将学习率和损失函数值写入文件
            for lr, loss in zip(self.lrs, self.losses):
                f.write('{},{}\n'.format(lr, loss))
