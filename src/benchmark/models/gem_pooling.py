"""
GeM (Generalized Mean) Pooling implementation for image retrieval.
Based on "Fine-tuning CNN Image Retrieval with No Human Annotation" (TPAMI 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeMPooling(nn.Module):
    """Generalized Mean Pooling layer"""
    
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        # x: [B, C, H, W]
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                           (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class GeMModel(nn.Module):
    """GeM-based model for image retrieval"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1000, 
                 feature_dim=2048, gem_p=3.0):
        super(GeMModel, self).__init__()
        
        # Backbone network
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the original classifier and pooling
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # GeM pooling
        self.gem_pool = GeMPooling(p=gem_p)
        
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
        
        # GeM pooling
        pooled = self.gem_pool(features)  # [B, C, 1, 1]
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


class GeMWrapper(nn.Module):
    """Wrapper for GeM model to match the existing interface"""
    
    def __init__(self, num_classes, backbone='resnet50', feature_dim=2048, gem_p=3.0):
        super(GeMWrapper, self).__init__()
        self.backbone = GeMModel(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
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


def create_gem_optimizer(args, model):
    """Create optimizer for GeM model"""
    # Use SGD optimizer as commonly used for GeM
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    return optimizer


# Model factory function
def get_gem_model(num_classes, backbone='resnet50', **kwargs):
    """Factory function to create GeM model"""
    return GeMWrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )

