import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools

from model import PneumoniaCNN  # 假设您的模型定义在 model.py 中

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

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    """
    训练并验证模型。

    参数:
        model (torch.nn.Module): 要训练的模型。
        dataloaders (dict): 数据加载器字典。
        dataset_sizes (dict): 各数据集的大小。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler): 学习率调度器。
        device (torch.device): 训练设备（CPU或GPU）。
        num_epochs (int): 训练的总轮数。

    返回:
        model (torch.nn.Module): 训练好的模型。
        history (dict): 训练和验证的损失与准确率历史。
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch +1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零梯度
                optimizer.zero_grad()

                # 前向传播
                # 只在训练阶段计算梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best Validation Acc: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, num_epochs):
    """
    绘制训练和验证的损失与准确率曲线。

    参数:
        history (dict): 训练和验证的损失与准确率历史。
        num_epochs (int): 训练的总轮数。
    """
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def save_model(model, path):
    """
    保存模型的状态字典。

    参数:
        model (torch.nn.Module): 要保存的模型。
        path (str): 模型保存路径。
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def visualize_confusion_matrix(model, dataloader, class_names, device):
    """
    可视化混淆矩阵。

    参数:
        model (torch.nn.Module): 已训练好的模型。
        dataloader (DataLoader): 测试数据加载器。
        class_names (list): 类别名称。
        device (torch.device): 训练设备（CPU或GPU）。
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵。

    参数:
        cm (array, shape = [n, n]): 混淆矩阵。
        classes (list): 类别名称。
        normalize (bool): 是否标准化。
        title (str): 标题。
        cmap: 色彩图。
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def evaluate_model(model, dataloader, class_names, device):
    """
    在测试集上评估模型，并打印分类报告。

    参数:
        model (torch.nn.Module): 已训练好的模型。
        dataloader (DataLoader): 测试数据加载器。
        class_names (list): 类别名称。
        device (torch.device): 训练设备（CPU或GPU）。
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 定义数据集路径
    data_dir = r'C:\Users\ht2\Desktop\dzy\chest_xray\chest_xray'  # 使用原始字符串

    # 加载数据
    dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size=32, num_workers=0)

    # 打印信息
    print(f"类别: {class_names}")
    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"验证集大小: {dataset_sizes['val']}")
    print(f"测试集大小: {dataset_sizes['test']}")

    # 构建模型
    model = PneumoniaCNN(num_classes=2)
    model = model.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器（仅优化可训练参数）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    num_epochs = 25
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=num_epochs)

    # 绘制训练历史
    plot_history(history, num_epochs)

    # 保存最佳模型
    save_model(model, 'best_pneumonia_model.pth')

    # 在测试集上评估模型
    print("Evaluating on test set:")
    evaluate_model(model, dataloaders['test'], class_names, device)

    # 可视化混淆矩阵
    visualize_confusion_matrix(model, dataloaders['test'], class_names, device)
