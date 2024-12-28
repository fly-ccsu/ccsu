# 1.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# 定义数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

def load_data(data_dir, batch_size=32, num_workers=0):
    """
    加载数据集并创建数据加载器。

    参数:
        data_dir (str): 数据集的根目录路径。
        batch_size (int): 每个批次的样本数量。
        num_workers (int): 用于数据加载的子进程数量。Windows上设置为0。

    返回:
        dataloaders (dict): 包含训练、验证和测试数据加载器的字典。
        dataset_sizes (dict): 各数据集的大小。
        class_names (list): 类别名称。
    """
    # 加载数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    # 创建数据加载器
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val', 'test']}

    # 数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # 类别名称
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def imshow(inp, title=None):
    """展示图像张量"""
    inp = inp.numpy().transpose((1, 2, 0))  # 转换维度
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet均值
    std = np.array([0.229, 0.224, 0.225])   # ImageNet标准差
    inp = std * inp + mean  # 反归一化
    inp = np.clip(inp, 0, 1)  # 限制在[0,1]范围内
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)  # 暂停以更新图像

def visualize_sample(dataloaders, class_names):
    """
    可视化一批数据。

    参数:
        dataloaders (dict): 数据加载器字典。
        class_names (list): 类别名称。
    """
    # 获取一批训练数据
    inputs, classes = next(iter(dataloaders['train']))

    # 创建网格
    out = torchvision.utils.make_grid(inputs[:4])

    # 展示图像
    imshow(out, title=[class_names[x] for x in classes[:4]])

if __name__ == '__main__':
    # 定义数据集路径
    data_dir = r'C:\Users\ht2\Desktop\dzy\chest_xray\chest_xray'  # 使用原始字符串

    # 加载数据
    dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size=32, num_workers=0)

    # 打印信息
    print(f"类别: {class_names}")
    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"验证集大小: {dataset_sizes['val']}")
    print(f"测试集大小: {dataset_sizes['test']}")

    # 可视化部分预处理后的图像
    visualize_sample(dataloaders, class_names)
