"""
DELG (DEep Local and Global features) implementation for image retrieval.
Based on "Unifying Deep Local and Global Features for Image Search" (ECCV 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .gem_pooling import GeMPooling


class LocalFeatureExtractor(nn.Module):
    """Local feature extraction module"""
    
    def __init__(self, input_dim, local_dim=1024):
        super(LocalFeatureExtractor, self).__init__()
        self.local_conv = nn.Conv2d(input_dim, local_dim, 1)
        self.attention_conv = nn.Conv2d(input_dim, 1, 1)
    
    def forward(self, x):
        # x: [B, C, H, W]
        local_features = self.local_conv(x)  # [B, local_dim, H, W]
        attention_map = torch.sigmoid(self.attention_conv(x))  # [B, 1, H, W]
        
        # Apply attention
        attended_features = local_features * attention_map
        
        return local_features, attention_map, attended_features


class DELGModel(nn.Module):
    """DELG model for image retrieval"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1000,
                 global_dim=2048, local_dim=1024, gem_p=3.0):
        super(DELGModel, self).__init__()
        
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
        
        # Global feature extraction
        self.gem_pool = GeMPooling(p=gem_p)
        self.global_proj = nn.Linear(backbone_dim, global_dim)
        
        # Local feature extraction
        self.local_extractor = LocalFeatureExtractor(backbone_dim, local_dim)
        
        # Classifier
        self.classifier = nn.Linear(global_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def extract_features(self, x):
        """Extract both global and local features"""
        # Backbone features
        backbone_features = self.backbone(x)  # [B, C, H, W]
        
        # Global features
        global_pooled = self.gem_pool(backbone_features)  # [B, C, 1, 1]
        global_pooled = global_pooled.view(global_pooled.size(0), -1)  # [B, C]
        global_features = self.global_proj(global_pooled)  # [B, global_dim]
        
        # Local features
        local_features, attention_map, attended_local = self.local_extractor(backbone_features)
        
        return {
            'global_features': global_features,
            'local_features': local_features,
            'attention_map': attention_map,
            'attended_local': attended_local
        }
    
    def forward(self, x, targets=None):
        """Forward pass"""
        # Extract features
        features = self.extract_features(x)
        global_features = features['global_features']
        
        # Classification using global features
        logits = self.classifier(global_features)
        
        if self.training and targets is not None:
            loss = self.criterion(logits, targets)
            return loss, logits
        else:
            return None, logits
    
    def extract_descriptor(self, x):
        """Extract global descriptor for retrieval"""
        with torch.no_grad():
            features = self.extract_features(x)
            global_features = features['global_features']
            # L2 normalize
            global_features = F.normalize(global_features, p=2, dim=1)
        return global_features
    
    def extract_local_descriptors(self, x):
        """Extract local descriptors for re-ranking"""
        with torch.no_grad():
            features = self.extract_features(x)
            local_features = features['local_features']
            attention_map = features['attention_map']
            
            # Normalize local features
            B, C, H, W = local_features.shape
            local_features = local_features.view(B, C, -1)  # [B, C, H*W]
            local_features = F.normalize(local_features, p=2, dim=1)
            
            # Apply attention threshold (keep top features)
            attention_flat = attention_map.view(B, -1)  # [B, H*W]
            
        return local_features, attention_flat


class DELGWrapper(nn.Module):
    """Wrapper for DELG model to match the existing interface"""
    
    def __init__(self, num_classes, backbone='resnet50', global_dim=2048, 
                 local_dim=1024, gem_p=3.0):
        super(DELGWrapper, self).__init__()
        self.backbone = DELGModel(
            backbone=backbone,
            num_classes=num_classes,
            global_dim=global_dim,
            local_dim=local_dim,
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
    
    def extract_local_descriptors(self, x):
        """Extract local descriptors for re-ranking"""
        return self.backbone.extract_local_descriptors(x)


def create_delg_optimizer(args, model):
    """Create optimizer for DELG model"""
    # Use Adam optimizer as commonly used for DELG
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer


# Model factory function
def get_delg_model(num_classes, backbone='resnet50', **kwargs):
    """Factory function to create DELG model"""
    return DELGWrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )

