import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import ViT
from .Bottleneck import Bottleneck

class ConvNet3(nn.Module):
    '''
    这个模型由三层的卷积层的特征提取器和一个两层的全连接层的分类器组成
    '''
    def __init__(self, num_classes):
        super().__init__()
        # 多尺度卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 多尺度特征融合后的处理
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 分类器 - 假设输入图像为64x64，经过一次池化变为32x32
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*3 * 32 * 32, 512),  # 三个卷积层的通道总和：64*3
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        # 在通道维度上拼接
        x_cat = torch.cat([x1, x2, x3], dim=1)
        
        # 池化减小特征图尺寸
        x_pool = self.pool(x_cat)
        
        # 扁平化
        x_flat = torch.flatten(x_pool, 1)
        
        # 分类
        out = self.classifier(x_flat)
        return out
    def __str__(self):
        return self.__class__.__name__
    

        
class ConvNet5(nn.Module):
    '''
    这个模型由五层的卷积层的特征提取器和一个两层的全连接层的分类器组成
    '''
    def __init__(self, num_classes):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积块: 3->32, 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积块: 32->64, 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三个卷积块: 64->128, 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 扁平化，保留批次维度
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def __str__(self):
        return self.__class__.__name__


class ResNet(nn.Module):
    '''
    这个模型是基于ResNet-50架构设计的网络
    '''
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        # 初始层
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4个阶段
        self.layer1 = self._make_layer(64, 3, stride=1)  # Stage 1：3个Bottleneck块
        self.layer2 = self._make_layer(128, 4, stride=2)  # Stage 2：4个Bottleneck块
        self.layer3 = self._make_layer(256, 6, stride=2)  # Stage 3：6个Bottleneck块
        self.layer4 = self._make_layer(512, 3, stride=2)  # Stage 4：3个Bottleneck块
        
        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        
        # 如果步长不为1或输入通道数不等于输出通道数的4倍（Bottleneck.expansion）
        # 那么需要一个下采样层来改变维度
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )
            
        layers = []
        # 添加第一个Bottleneck块，可能需要下采样
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        
        # 更新输入通道数
        self.in_channels = out_channels * Bottleneck.expansion
        
        # 添加其余的Bottleneck块
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
        
    def __str__(self):
        return "ResNet50"
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixBlock, self).__init__()
        self.channel1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.channel1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.channel2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.channel3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 第四路也用卷积 + 池化，保证输入输出都是 4D
        self.channel4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()
        
    def forward(self, x):
        x1 = self.channel1(x)
        x2 = self.channel2(x)
        x3 = self.channel3(x)
        x4 = self.channel4(x)
        
        # 在通道维度上拼接
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        # 融合
        x_cat = self.fusion(x_cat)
        
        return x_cat
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ConvNetMix(nn.Module):
    '''
    多尺度特征融合
    '''
    def __init__(self, num_classes):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            MixBlock(3, 32),  # 输入通道数为3，输出通道数为32
            MixBlock(32, 64),  # 输入通道数为32，输出通道数为64
            MixBlock(64, 128),  # 输入通道数为64，输出通道数为128
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 扁平化，保留批次维度
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def __str__(self):
        return self.__class__.__name__