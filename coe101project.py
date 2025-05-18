import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import os
from .models import ConvNet3, ConvNet5, ResNet, DenseNet, ViTNet
from data_stats_cal import calculate_dataset_stats
import TransformDataset as TransformDataset
import argparse
import draw_figures as draw_figures

'''
def show_transformed_images(dataset, num_images=5):
    figure = plt.figure(figsize=(20, 8))
    
    for i in range(num_images):
        image, label = dataset[i]
        # 将标准化的图像转回原始范围
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        figure.add_subplot(1, num_images, i+1)
        plt.title(f"Class: {dataset.dataset.dataset.classes[label]}")
        plt.axis("off")
        plt.imshow(image)
    
    plt.show()
'''

def is_valid_file(path):
    """检查是否是有效的图像文件"""
    return os.path.splitext(path)[1].lower() in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

def is_valid_class_dir(directory):
    """检查目录是否为有效的类别目录（排除隐藏文件夹）"""
    return not directory.startswith('.')

    
def evaluate(test_loader, net, device):
    n_correct = 0
    n_total = 0
    class_correct = {}
    class_total = {}
    
    net.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            # 统计每个类别的准确率
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    # 打印每个类别的准确率
    if len(class_correct) > 0:
        print("\n各类别准确率:")
        for label in sorted(class_correct.keys()):
            accuracy = class_correct[label] / class_total[label]
            print(f"类别 {label}: {accuracy:.4f} ({class_correct[label]}/{class_total[label]})")
    
    return n_correct / n_total

def train_epoch(train_loader, net, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    class_predictions = {}  # 跟踪每个类别的预测次数
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 统计每个类别的预测分布
        for pred in predicted.cpu().numpy():
            if pred not in class_predictions:
                class_predictions[pred] = 0
            class_predictions[pred] += 1
    
    train_accuracy = correct / total
    print(f"训练准确率: {train_accuracy:.4f}")
    print(f"预测分布: {class_predictions}")
    
    return running_loss / len(train_loader), train_accuracy

def main(args):
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据路径
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"错误: 找不到数据集目录 {data_dir}")
        return

    epoch_list = []
    train_acc_list = []
    val_acc_list = []
    loss_list = []
    # 加载完整数据集
    try:
        # 过滤掉隐藏目录
        full_dataset = datasets.ImageFolder(
            root=data_dir, 
            transform=None,
            is_valid_file=is_valid_file
        )
        print(f"成功加载数据集，共有 {len(full_dataset)} 个样本")
        
        # 打印类别信息
        print("类别分布:")
        class_counts = {}
        for label in full_dataset.targets:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
            
        for label_idx, count in class_counts.items():
            print(f"类别 '{full_dataset.classes[label_idx]}': {count} 个样本")
            
    except Exception as e:
        print(f"加载数据集出错: {e}")
        return

    # 设置训练集和验证集的比例 (例如: 80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 随机分割数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    # 计算数据集统计信息
    mean, std = calculate_dataset_stats(data_dir)
    

    # 应用不同的转换
    train_dataset_transformed = TransformDataset(train_dataset, mean, std, train=True)
    val_dataset_transformed = TransformDataset(val_dataset, mean, std, train=False)

    # 创建DataLoader
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型并转移到设备
    num_classes = len(full_dataset.classes)
    if args.model_name == 'ConvNet3':
        net = ConvNet3(num_classes=num_classes).to(device)
    elif args.model_name == 'ConvNet5':
        net = ConvNet5(num_classes=num_classes).to(device)
    elif args.model_name == 'ResNet':
        net = ResNet(num_classes=num_classes).to(device)
    elif args.model_name == 'DenseNet':
        net = DenseNet(num_classes=num_classes).to(device)
    elif args.model_name == 'ViTNet':
        net = ViTNet(num_classes=num_classes).to(device)
    else:
        raise ValueError('没有这个模型')
    
    print(f"创建网络: {net}，类别数量: {num_classes}")
    
    
    lr = args.learning_rate
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    # 训练模型
    num_epochs = args.num_epochs
    print("开始训练...")
    
    best_val_acc = 0
    best_model_path = "./weights/{net}_best_model.pth"
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss, train_acc = train_epoch(train_loader, net, criterion, optimizer, device)
        val_accuracy = evaluate(val_loader, net, device)
        epoch_list.append(epoch + 1)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_accuracy)
        loss_list.append(avg_loss)
        # 学习率调整
        scheduler.step(val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(net.state_dict(), best_model_path)
            print(f"保存新的最佳模型，验证准确率: {val_accuracy:.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_accuracy:.4f}")

    print(f"训练完成。最佳验证准确率: {best_val_acc:.4f}")
    
    # 加载最佳模型进行评估
    net.load_state_dict(torch.load(best_model_path))
    final_accuracy = evaluate(val_loader, net, device)
    print(f"最终测试准确率: {final_accuracy:.4f}")
    
    '''# 可视化一些预测结果
    net.eval()
    with torch.no_grad():
        # 获取一批验证数据
        dataiter = iter(val_loader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        
        # 显示图像和预测结果
        plt.figure(figsize=(20, 10))
        for i in range(min(5, images.size(0))):
            # 将图像转回CPU并转换为numpy
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            # 反标准化
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            plt.title(f'真实: {full_dataset.classes[labels[i]]}\n预测: {full_dataset.classes[predicted[i]]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()'''
    
    # 保存最终模型
    torch.save(net.state_dict(), f"./weights/{net}_final_model.pth")
    print("最终模型已保存")
    print("正在画图……")
    draw_figures(epoch_list, train_acc_list, val_acc_list, loss_list, args.model_name)
    print('画图完毕，图像已保存')
    print('程序结束！')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练和评估图像分类模型")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="数据集目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.0015, help="学习率")
    parser.add_argument("--model_name", type=str, default='ConvNet3', choices=['ConvNet3', 'ConvNet5', 'ResNet', 'DenseNet', 'ViTNet'])
    args = parser.parse_args()   
    
    
    multiprocessing.freeze_support()
    main(args)