import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        
        # 加载预训练的ResNet-50模型
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 替换最后一层全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
