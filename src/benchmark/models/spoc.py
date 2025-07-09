"""
SpoC (Spatial Pyramid of Contexts) implementation for image retrieval.
Based on spatial pyramid pooling with contextual information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer"""
    
    def __init__(self, levels=[1, 2, 4], pool_type='max'):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels
        self.pool_type = pool_type
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        pooled_features = []
        
        for level in self.levels:
            # Calculate kernel size and stride for each level
            kernel_h = H // level
            kernel_w = W // level
            stride_h = H // level
            stride_w = W // level
            
            # Pooling
            if self.pool_type == 'max':
                pooled = F.max_pool2d(x, kernel_size=(kernel_h, kernel_w), 
                                    stride=(stride_h, stride_w))
            elif self.pool_type == 'avg':
                pooled = F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), 
                                    stride=(stride_h, stride_w))
            else:
                raise ValueError(f"Unsupported pool_type: {self.pool_type}")
            
            # Flatten spatial dimensions
            pooled = pooled.view(B, C, -1)  # [B, C, level*level]
            pooled_features.append(pooled)
        
        # Concatenate all levels
        pyramid_features = torch.cat(pooled_features, dim=2)  # [B, C, total_regions]
        
        return pyramid_features


class ContextualAttention(nn.Module):
    """Contextual attention mechanism for SpoC"""
    
    def __init__(self, input_dim, context_dim=512):
        super(ContextualAttention, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Context encoding
        self.context_encoder = nn.Sequential(
            nn.Conv2d(input_dim, context_dim, 3, padding=1),
            nn.BatchNorm2d(context_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_dim, context_dim, 3, padding=1),
            nn.BatchNorm2d(context_dim),
            nn.ReLU(inplace=True)
        )
        
        # Attention computation
        self.attention_conv = nn.Conv2d(context_dim, 1, 1)
        
        # Feature refinement
        self.feature_refine = nn.Conv2d(input_dim + context_dim, input_dim, 1)
    
    def forward(self, x):
        # x: [B, C, H, W]
        
        # Encode context
        context = self.context_encoder(x)  # [B, context_dim, H, W]
        
        # Compute attention weights
        attention_weights = torch.sigmoid(self.attention_conv(context))  # [B, 1, H, W]
        
        # Apply attention to original features
        attended_x = x * attention_weights
        
        # Concatenate original and context features
        combined = torch.cat([attended_x, context], dim=1)  # [B, C+context_dim, H, W]
        
        # Refine features
        refined_features = self.feature_refine(combined)  # [B, C, H, W]
        
        return refined_features, attention_weights


class SpoCModel(nn.Module):
    """SpoC model for image retrieval"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1000,
                 feature_dim=2048, pyramid_levels=[1, 2, 4], context_dim=512,
                 use_context=True):
        super(SpoCModel, self).__init__()
        
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
        
        # Contextual attention (optional)
        self.use_context = use_context
        if use_context:
            self.contextual_attention = ContextualAttention(backbone_dim, context_dim)
        
        # Spatial pyramid pooling
        self.pyramid_levels = pyramid_levels
        self.spp = SpatialPyramidPooling(levels=pyramid_levels, pool_type='max')
        
        # Calculate total regions in pyramid
        total_regions = sum([level * level for level in pyramid_levels])
        
        # Feature aggregation
        self.feature_aggregation = nn.Sequential(
            nn.Conv1d(backbone_dim, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Final feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def extract_features(self, x):
        """Extract features from input images"""
        # Backbone features
        backbone_features = self.backbone(x)  # [B, C, H, W]
        
        # Apply contextual attention if enabled
        if self.use_context:
            refined_features, attention_weights = self.contextual_attention(backbone_features)
        else:
            refined_features = backbone_features
        
        # Spatial pyramid pooling
        pyramid_features = self.spp(refined_features)  # [B, C, total_regions]
        
        # Feature aggregation
        aggregated_features = self.feature_aggregation(pyramid_features)  # [B, feature_dim, 1]
        aggregated_features = aggregated_features.squeeze(2)  # [B, feature_dim]
        
        # Final projection
        features = self.feature_proj(aggregated_features)
        
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


class SpoCWrapper(nn.Module):
    """Wrapper for SpoC model to match the existing interface"""
    
    def __init__(self, num_classes, backbone='resnet50', feature_dim=2048,
                 pyramid_levels=[1, 2, 4], context_dim=512, use_context=True):
        super(SpoCWrapper, self).__init__()
        self.backbone = SpoCModel(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
            pyramid_levels=pyramid_levels,
            context_dim=context_dim,
            use_context=use_context
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


def create_spoc_optimizer(args, model):
    """Create optimizer for SpoC model"""
    # Use SGD optimizer for SpoC
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    return optimizer


# Model factory function
def get_spoc_model(num_classes, backbone='resnet50', **kwargs):
    """Factory function to create SpoC model"""
    return SpoCWrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )

