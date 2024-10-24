from .clip import clip  # 从当前目录导入 clip 模块
from PIL import Image  # 导入 PIL 库中的 Image 模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

# 定义不同模型的通道数
CHANNELS = {
    "RN50": 1024,  # RN50 模型的特征通道数
    "ViT-L/14": 768  # ViT-L/14 模型的特征通道数
}

class CLIPModel(nn.Module):  # 定义 CLIPModel 类，继承自 nn.Module
    def __init__(self, name, num_classes=1):  # 构造函数，接收模型名称和类别数量
        super(CLIPModel, self).__init__()  # 调用父类的构造函数

        self.model, self.preprocess = clip.load(name, device="cpu")  # 加载指定名称的 CLIP 模型和预处理函数，指定设备为 CPU
        # 注意：self.preprocess 在训练过程中不会使用，由 Dataset 类处理
        self.fc = nn.Linear(CHANNELS[name], num_classes)  # 定义全连接层，将特征通道映射到类别数量

    def forward(self, x, return_feature=False):  # 前向传播方法，接收输入和返回特征的标志
        features = self.model.encode_image(x)  # 通过模型编码输入图像，获取特征
        if return_feature:  # 如果需要返回特征
            return features  # 返回特征
        return self.fc(features)  # 否则，将特征通过全连接层并返回输出


class myCLIPModel(nn.Module):  # 定义 CLIPModel 类，继承自 nn.Module
    def __init__(self, name, num_classes=1):  # 构造函数，接收模型名称和类别数量
        super(CLIPModel, self).__init__()  # 调用父类的构造函数

        self.model, self.preprocess = clip.load(name, device="cpu")  # 加载指定名称的 CLIP 模型和预处理函数，指定设备为 CPU
        # 注意：self.preprocess 在训练过程中不会使用，由 Dataset 类处理
        self.fc = nn.Linear(CHANNELS[name], num_classes)  # 定义全连接层，将特征通道映射到类别数量

    def forward(self, x, return_feature=False):  # 前向传播方法，接收输入和返回特征的标志
        features = self.model.encode_image(x)  # 通过模型编码输入图像，获取特征
        if return_feature:  # 如果需要返回特征
            return features  # 返回特征
        return self.fc(features)  # 否则，将特征通过全连接层并返回输出