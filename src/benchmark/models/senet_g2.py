"""
SENet-G2+ implementation for image retrieval.
Based on SENet backbone with G2+ (Generalized-mean pooling with learnable power) pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    """SENet Bottleneck block"""
    
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class SENet(nn.Module):
    """SENet backbone"""
    
    def __init__(self, block, layers, num_classes=1000, reduction=16):
        super(SENet, self).__init__()
        self.inplanes = 64
        self.reduction = reduction
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=self.reduction))
        
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
        
        return x


class G2Pooling(nn.Module):
    """G2+ (Generalized-mean pooling with learnable power) pooling"""
    
    def __init__(self, p=3.0, eps=1e-6):
        super(G2Pooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
        # Additional learnable parameters for G2+
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # x: [B, C, H, W]
        # Standard GeM pooling
        gem_pooled = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                                 (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        
        # G2+ enhancement: learnable scaling and bias
        enhanced = self.alpha * gem_pooled + self.beta
        
        return enhanced


class SENetG2Model(nn.Module):
    """SENet with G2+ pooling for image retrieval"""
    
    def __init__(self, layers=[3, 4, 6, 3], num_classes=1000, feature_dim=2048, 
                 reduction=16, gem_p=3.0):
        super(SENetG2Model, self).__init__()
        
        # SENet backbone
        self.backbone = SENet(SEBottleneck, layers, reduction=reduction)
        backbone_dim = 512 * SEBottleneck.expansion  # 2048
        
        # G2+ pooling
        self.g2_pool = G2Pooling(p=gem_p)
        
        # Feature projection
        self.feature_proj = nn.Linear(backbone_dim, feature_dim)
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def extract_features(self, x):
        """Extract features from input images"""
        # Backbone features
        features = self.backbone(x)  # [B, C, H, W]
        
        # G2+ pooling
        pooled = self.g2_pool(features)  # [B, C, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, C]
        
        # Feature projection
        features = self.feature_proj(pooled)  # [B, feature_dim]
        
        return features
    
    def forward(self, x, targets=None):
        """Forward pass"""
        # Extract features
        features = self.extract_features(x)
        
        # Classification
        logits = self.classifier(features)
        
        if self.training and targets is not None:
            loss = self.criterion(logits, targets)
            return loss, logits
        else:
            return None, logits
    
    def extract_descriptor(self, x):
        """Extract global descriptor for retrieval"""
        with torch.no_grad():
            features = self.extract_features(x)
            # L2 normalize
            features = F.normalize(features, p=2, dim=1)
        return features


class SENetG2Wrapper(nn.Module):
    """Wrapper for SENet-G2+ model to match the existing interface"""
    
    def __init__(self, num_classes, backbone='senet50', feature_dim=2048, 
                 reduction=16, gem_p=3.0):
        super(SENetG2Wrapper, self).__init__()
        
        # Determine layers based on backbone
        if backbone == 'senet50':
            layers = [3, 4, 6, 3]
        elif backbone == 'senet101':
            layers = [3, 4, 23, 3]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone = SENetG2Model(
            layers=layers,
            num_classes=num_classes,
            feature_dim=feature_dim,
            reduction=reduction,
            gem_p=gem_p
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x, targets=None):
        """Forward pass compatible with existing training loop"""
        if self.training and targets is not None:
            loss, logits = self.backbone(x, targets)
            return loss, logits
        else:
            _, logits = self.backbone(x)
            return None, logits
    
    def extract_global_descriptor(self, x):
        """Extract global descriptor for evaluation"""
        return self.backbone.extract_descriptor(x)


def create_senet_g2_optimizer(args, model):
    """Create optimizer for SENet-G2+ model"""
    # Use SGD optimizer with specific settings for SENet
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    return optimizer


# Model factory function
def get_senet_g2_model(num_classes, backbone='senet50', **kwargs):
    """Factory function to create SENet-G2+ model"""
    return SENetG2Wrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )

