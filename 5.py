# 5.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools

from model import PneumoniaCNN  # 确保 model.py 和 5.py 位于同一目录下

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
                                 shuffle=False, num_workers=num_workers)
                   for x in ['train', 'val', 'test']}  # 测试集不需要打乱

    # 数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # 类别名称
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def load_model(model_path, device):
    """
    加载训练好的模型权重。

    参数:
        model_path (str): 模型权重文件路径。
        device (torch.device): 训练设备（CPU或GPU）。

    返回:
        model (torch.nn.Module): 加载了权重的模型。
    """
    model = PneumoniaCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置模型为评估模式
    return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    绘制混淆矩阵。

    参数:
        cm (array, shape = [n, n]): 混淆矩阵。
        classes (list): 类别名称。
        normalize (bool): 是否标准化。
        title (str): 标题。
        cmap: 色彩图。
        save_path (str): 可选，保存图像的路径。
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

    if save_path:
        plt.savefig(save_path)
        print(f'Confusion matrix saved to {save_path}')
    else:
        plt.show()

def plot_roc_curve(y_true, y_scores, classes, save_path=None):
    """
    绘制ROC曲线并计算AUC。

    参数:
        y_true (list): 真实标签。
        y_scores (list): 预测分数（概率）。
        classes (list): 类别名称。
        save_path (str): 可选，保存图像的路径。
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    # 二分类情况下，binarize只会有一个输出
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    y_score = y_scores

    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
        print(f'ROC curve saved to {save_path}')
    else:
        plt.show()

def evaluate_model(model, dataloader, class_names, device, save_confusion_matrix=True, save_roc_curve=True):
    """
    在测试集上评估模型，并生成评价指标和可视化图表。

    参数:
        model (torch.nn.Module): 已训练好的模型。
        dataloader (DataLoader): 测试数据加载器。
        class_names (list): 类别名称。
        device (torch.device): 训练设备（CPU或GPU）。
        save_confusion_matrix (bool): 是否保存混淆矩阵图像。
        save_roc_curve (bool): 是否保存ROC曲线图像。
    """
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)  # 计算概率
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 假设类别1为“PNEUMONIA”

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n", report)

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    if save_confusion_matrix:
        plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix', save_path='confusion_matrix.png')
    else:
        plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')

    # 计算AUC并绘制ROC曲线
    plot_roc_curve(all_labels, all_probs, classes=class_names, save_path='roc_curve.png')

def main():
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

    # 加载最佳模型
    model_path = 'best_pneumonia_model.pth'
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在。请先训练并保存模型。")
        return

    model = load_model(model_path, device)

    # 在测试集上评估模型
    print("Evaluating on test set:")
    evaluate_model(model, dataloaders['test'], class_names, device)

if __name__ == '__main__':
    main()
