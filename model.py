# model.py

import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        初始化PneumoniaCNN模型。

        参数:
            num_classes (int): 分类的数量，默认值为2（正常与肺炎）。
        """
        super(PneumoniaCNN, self).__init__()

        # 第一卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三卷积层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四卷积层
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # 假设输入图像大小为224x224
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, 3, 224, 224)。

        返回:
            torch.Tensor: 模型的输出，形状为(batch_size, num_classes)。
        """
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 第三层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # 第四层卷积
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def print_model_summary(model):
    """
    打印模型的结构摘要。

    参数:
        model (torch.nn.Module): 需要打印的模型。
    """
    print(model)

if __name__ == "__main__":
    # 构建模型
    model = PneumoniaCNN(num_classes=2)

    # 打印模型结构
    print_model_summary(model)

    # 计算模型的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
