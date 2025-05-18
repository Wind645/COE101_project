from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch

def calculate_dataset_stats(dataset_path):
    
    temp_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),  # 使用与最终transform相同的大小
        transforms.ToTensor()  # 将图像转换为[0,1]范围的Tensor
    ])
    
    
    dataset = datasets.ImageFolder(root=dataset_path, transform=temp_transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("计算数据集均值和标准差...")
    
    # 计算均值
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean /= len(dataset)
    
    # 计算标准差
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).mean(2).sum(0)
    std = torch.sqrt(std / len(dataset))
    
    print(f"计算完成!\n均值: {mean.tolist()}\n标准差: {std.tolist()}")
    return mean.tolist(), std.tolist()