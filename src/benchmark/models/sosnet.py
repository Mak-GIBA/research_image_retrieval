"""
SoSNet (Second-order Similarity Network) implementation for image retrieval.
Based on second-order similarity learning for robust image retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SecondOrderPooling(nn.Module):
    """Second-order pooling layer"""
    
    def __init__(self, input_dim, output_dim=None, normalize=True):
        super(SecondOrderPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.normalize = normalize
        
        # Dimension reduction if needed
        if self.output_dim != self.input_dim:
            self.proj = nn.Conv2d(input_dim, self.output_dim, 1)
        else:
            self.proj = None
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Apply projection if needed
        if self.proj is not None:
            x = self.proj(x)
            C = self.output_dim
        
        # Reshape to [B, C, N] where N = H*W
        x = x.view(B, C, -1)  # [B, C, H*W]
        
        # Compute second-order statistics (covariance)
        # Center the features
        mean = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        x_centered = x - mean  # [B, C, H*W]
        
        # Compute covariance matrix
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (H * W - 1)  # [B, C, C]
        
        # Flatten upper triangular part (including diagonal)
        indices = torch.triu_indices(C, C, offset=0)
        second_order_features = cov[:, indices[0], indices[1]]  # [B, C*(C+1)/2]
        
        # Normalize if required
        if self.normalize:
            second_order_features = F.normalize(second_order_features, p=2, dim=1)
        
        return second_order_features


class SimilarityAttention(nn.Module):
    """Similarity-based attention mechanism"""
    
    def __init__(self, input_dim, hidden_dim=512):
        super(SimilarityAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape to [B*H*W, C]
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # Compute attention weights
        attention_weights = self.attention_net(x_flat)  # [B*H*W, 1]
        
        # Reshape back to [B, H, W, 1] and then [B, 1, H, W]
        attention_weights = attention_weights.view(B, H, W, 1).permute(0, 3, 1, 2)
        
        # Apply attention
        attended_features = x * attention_weights
        
        return attended_features, attention_weights


class SoSNetModel(nn.Module):
    """SoSNet model for image retrieval"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1000,
                 feature_dim=2048, second_order_dim=512, use_attention=True):
        super(SoSNetModel, self).__init__()
        
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
        
        # Similarity attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.similarity_attention = SimilarityAttention(backbone_dim)
        
        # Second-order pooling
        self.second_order_pool = SecondOrderPooling(
            input_dim=backbone_dim,
            output_dim=second_order_dim,
            normalize=True
        )
        
        # Calculate second-order feature dimension
        so_feature_dim = second_order_dim * (second_order_dim + 1) // 2
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(so_feature_dim, feature_dim),
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
        
        # Apply similarity attention if enabled
        if self.use_attention:
            attended_features, attention_weights = self.similarity_attention(backbone_features)
        else:
            attended_features = backbone_features
        
        # Second-order pooling
        second_order_features = self.second_order_pool(attended_features)
        
        # Feature projection
        features = self.feature_proj(second_order_features)
        
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


class SoSNetWrapper(nn.Module):
    """Wrapper for SoSNet model to match the existing interface"""
    
    def __init__(self, num_classes, backbone='resnet50', feature_dim=2048,
                 second_order_dim=512, use_attention=True):
        super(SoSNetWrapper, self).__init__()
        self.backbone = SoSNetModel(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
            second_order_dim=second_order_dim,
            use_attention=use_attention
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


def create_sosnet_optimizer(args, model):
    """Create optimizer for SoSNet model"""
    # Use Adam optimizer for SoSNet
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer


# Model factory function
def get_sosnet_model(num_classes, backbone='resnet50', **kwargs):
    """Factory function to create SoSNet model"""
    return SoSNetWrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )

