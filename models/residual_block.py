import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_features, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU())
        self.layer2 = nn.Sequential(
        nn.Conv2d(64, in_features, 3, 1, 1),
        nn.BatchNorm2d(in_features),
        nn.ReLU())
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = x + x2
        return x3