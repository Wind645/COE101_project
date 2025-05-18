from torchvision import transforms, datasets

# 自定义数据集类来应用不同的转换
class TransformDataset:
    def __init__(self, dataset, mean, std, train=True):
        self.dataset = dataset
        if train:
            self.transform =   transforms.Compose([
            transforms.RandomResizedCrop(64),           # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
            transforms.RandomRotation(15),               # 随机旋转±15度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 随机调整亮度、对比度和饱和度
            transforms.ToTensor(),                       # 转换为Tensor并归一化到[0,1]
            transforms.Normalize(mean=mean, std=std)  # 标准化
        ])
        else:
            # 对验证/测试数据只进行大小调整和归一化
            self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label