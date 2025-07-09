"""
Token-based image retrieval implementation.
Based on transformer-style token aggregation for image retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for spatial features"""
    
    def __init__(self, d_model, max_len=196):  # 14x14 = 196 for typical feature maps
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


class TokenAggregator(nn.Module):
    """Token-based feature aggregation using transformer"""
    
    def __init__(self, input_dim, hidden_dim=512, num_heads=8, num_layers=2):
        super(TokenAggregator, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Global token (learnable)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x = x.view(B, C, -1).permute(2, 0, 1)  # [H*W, B, C]
        
        # Project to hidden dimension
        x = self.input_proj(x)  # [H*W, B, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Add global token
        global_tokens = self.global_token.expand(-1, B, -1)  # [1, B, hidden_dim]
        x = torch.cat([global_tokens, x], dim=0)  # [1+H*W, B, hidden_dim]
        
        # Apply transformer
        x = self.transformer(x)  # [1+H*W, B, hidden_dim]
        
        # Extract global token
        global_feature = x[0]  # [B, hidden_dim]
        
        # Project back
        global_feature = self.output_proj(global_feature)  # [B, C]
        
        return global_feature


class TokenModel(nn.Module):
    """Token-based model for image retrieval"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1000,
                 feature_dim=2048, hidden_dim=512, num_heads=8, num_layers=2):
        super(TokenModel, self).__init__()
        
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
        
        # Token aggregator
        self.token_aggregator = TokenAggregator(
            input_dim=backbone_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Feature projection
        self.feature_proj = nn.Linear(backbone_dim, feature_dim)
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def extract_features(self, x):
        """Extract features from input images"""
        # Backbone features
        backbone_features = self.backbone(x)  # [B, C, H, W]
        
        # Token aggregation
        aggregated_features = self.token_aggregator(backbone_features)  # [B, C]
        
        # Feature projection
        features = self.feature_proj(aggregated_features)  # [B, feature_dim]
        
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


class TokenWrapper(nn.Module):
    """Wrapper for Token model to match the existing interface"""
    
    def __init__(self, num_classes, backbone='resnet50', feature_dim=2048,
                 hidden_dim=512, num_heads=8, num_layers=2):
        super(TokenWrapper, self).__init__()
        self.backbone = TokenModel(
            backbone=backbone,
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
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


def create_token_optimizer(args, model):
    """Create optimizer for Token model"""
    # Use AdamW optimizer as commonly used for transformer-based models
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer


# Model factory function
def get_token_model(num_classes, backbone='resnet50', **kwargs):
    """Factory function to create Token model"""
    return TokenWrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )

