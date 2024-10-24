import functools  # 导入 functools 模块，提供高阶函数的功能
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from networks.base_model import BaseModel, init_weights  # 从自定义的模块中导入基类和初始化权重的函数
import sys  # 导入系统模块
from models import get_model  # 从自定义的模块中导入获取模型的函数

class Trainer(BaseModel):  # 定义 Trainer 类，继承自 BaseModel
    def name(self):
        return 'Trainer'  # 返回类的名称

    def __init__(self, opt):  # 构造函数，接收配置选项
        super(Trainer, self).__init__(opt)  # 调用父类的构造函数
        self.opt = opt  # 保存配置选项
        self.model = get_model(opt.arch)  # 根据配置选项获取模型
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)  # 初始化模型最后一层的权重

        if opt.fix_backbone:  # 如果配置选项中要求固定骨干网络
            params = []  # 创建参数列表
            for name, p in self.model.named_parameters():  # 遍历模型的所有参数
                if name == "fc.weight" or name == "fc.bias":  # 如果是最后一层的权重或偏置
                    params.append(p)  # 添加到参数列表中
                else:
                    p.requires_grad = False  # 否则，冻结该参数
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")  # 如果骨干网络未固定，打印警告信息
            import time  # 导入时间模块
            time.sleep(3)  # 暂停 3 秒
            params = self.model.parameters()  # 获取所有参数

        if opt.optim == 'adam':  # 如果选择 Adam 优化器
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)  # 初始化 AdamW 优化器
        elif opt.optim == 'sgd':  # 如果选择 SGD 优化器
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)  # 初始化 SGD 优化器
        else:
            raise ValueError("optim should be [adam, sgd]")  # 如果优化器选择不合法，抛出异常

        self.loss_fn = nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失函数

        self.model.to(opt.gpu_ids[0])  # 将模型转移到指定的 GPU

    def adjust_learning_rate(self, min_lr=1e-6):  # 调整学习率的方法
        for param_group in self.optimizer.param_groups:  # 遍历所有优化器的参数组
            param_group['lr'] /= 10.  # 将学习率降低十倍
            if param_group['lr'] < min_lr:  # 如果学习率低于最小值
                return False  # 返回 False
        return True  # 返回 True

    def set_input(self, input):  # 设置输入数据的方法
        self.input = input[0].to(self.device)  # 将输入数据移动到指定设备
        self.label = input[1].to(self.device).float()  # 将标签移动到指定设备并转换为浮点型

    def forward(self):  # 前向传播方法
        self.output = self.model(self.input)  # 通过模型获取输出
        self.output = self.output.view(-1).unsqueeze(1)  # 调整输出形状

    def get_loss(self):  # 获取损失值的方法
        return self.loss_fn(self.output.squeeze(1), self.label)  # 计算并返回损失

    def optimize_parameters(self):  # 优化参数的方法
        self.forward()  # 执行前向传播
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)  # 计算损失
        self.optimizer.zero_grad()  # 清零梯度
        self.loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
