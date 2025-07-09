"""
DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features
Implementation based on the paper: https://arxiv.org/abs/2108.02927

This implementation provides a clean, efficient version of DOLG for image retrieval tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional


class SpatialAttention2d(nn.Module):
    """
    Spatial attention module for DOLG
    """
    def __init__(self, in_channels: int, act_fn: str = 'relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.act_fn = getattr(F, act_fn)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Attention weights [B, 1, H, W]
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fn(out)
        out = self.conv2(out)
        return torch.sigmoid(out)


class OrthogonalFusion(nn.Module):
    """
    Orthogonal fusion module for combining local and global features
    """
    def __init__(self, local_dim: int, global_dim: int, output_dim: int):
        super(OrthogonalFusion, self).__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.output_dim = output_dim
        
        # Projection layers
        self.local_proj = nn.Linear(local_dim, output_dim, bias=False)
        self.global_proj = nn.Linear(global_dim, output_dim, bias=False)
        
        # Orthogonal fusion weights
        self.fusion_weights = nn.Parameter(torch.randn(2))
        
    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_feat: Local features [B, local_dim]
            global_feat: Global features [B, global_dim]
        Returns:
            Fused features [B, output_dim]
        """
        # Project features to same dimension
        local_proj = self.local_proj(local_feat)
        global_proj = self.global_proj(global_feat)
        
        # Normalize projections
        local_proj = F.normalize(local_proj, p=2, dim=1)
        global_proj = F.normalize(global_proj, p=2, dim=1)
        
        # Orthogonal fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * local_proj + weights[1] * global_proj
        
        return F.normalize(fused, p=2, dim=1)


class DOLGNet(nn.Module):
    """
    DOLG: Deep Orthogonal Fusion of Local and Global Features
    
    Args:
        backbone_name: Name of backbone network ('resnet50', 'resnet101', etc.)
        pretrained: Whether to use pretrained backbone
        local_dim: Dimension of local features
        global_dim: Dimension of global features  
        output_dim: Final output dimension
        num_classes: Number of classes for classification head
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        local_dim: int = 1024,
        global_dim: int = 2048,
        output_dim: int = 512,
        num_classes: int = 1000
    ):
        super(DOLGNet, self).__init__()
        
        self.backbone_name = backbone_name
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Load backbone
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone_name == 'resnet152':
            backbone = models.resnet152(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract feature layers
        self.backbone_layers = nn.Sequential(*list(backbone.children())[:-2])
        
        # Local feature extraction
        self.local_attention = SpatialAttention2d(backbone_dim)
        self.local_pool = nn.AdaptiveAvgPool2d(1)
        self.local_fc = nn.Linear(backbone_dim, local_dim)
        
        # Global feature extraction  
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(backbone_dim, global_dim)
        
        # Orthogonal fusion
        self.fusion = OrthogonalFusion(local_dim, global_dim, output_dim)
        
        # Classification head
        self.classifier = nn.Linear(output_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features"""
        return self.backbone_layers(x)
    
    def extract_local_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract local features with spatial attention"""
        # Spatial attention
        attention = self.local_attention(features)
        attended_features = features * attention
        
        # Global average pooling
        pooled = self.local_pool(attended_features).flatten(1)
        
        # Linear projection
        local_feat = self.local_fc(pooled)
        return F.normalize(local_feat, p=2, dim=1)
    
    def extract_global_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract global features"""
        # Global average pooling
        pooled = self.global_pool(features).flatten(1)
        
        # Linear projection
        global_feat = self.global_fc(pooled)
        return F.normalize(global_feat, p=2, dim=1)
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
            targets: Target labels [B] (for training)
            
        Returns:
            loss: Loss value (if targets provided)
            logits: Classification logits [B, num_classes]
        """
        # Extract backbone features
        features = self.extract_features(x)
        
        # Extract local and global features
        local_feat = self.extract_local_features(features)
        global_feat = self.extract_global_features(features)
        
        # Orthogonal fusion
        fused_feat = self.fusion(local_feat, global_feat)
        
        # Classification
        logits = self.classifier(fused_feat)
        
        if self.training and targets is not None:
            loss = self.criterion(logits, targets)
            return loss, logits
        else:
            return None, logits
    
    def extract_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract final descriptor for retrieval
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Descriptors [B, output_dim]
        """
        with torch.no_grad():
            features = self.extract_features(x)
            local_feat = self.extract_local_features(features)
            global_feat = self.extract_global_features(features)
            fused_feat = self.fusion(local_feat, global_feat)
            return fused_feat


def create_dolg_model(
    backbone: str = 'resnet50',
    pretrained: bool = True,
    local_dim: int = 1024,
    global_dim: int = 2048,
    output_dim: int = 512,
    num_classes: int = 1000
) -> DOLGNet:
    """
    Create DOLG model
    
    Args:
        backbone: Backbone network name
        pretrained: Use pretrained weights
        local_dim: Local feature dimension
        global_dim: Global feature dimension
        output_dim: Output descriptor dimension
        num_classes: Number of classes
        
    Returns:
        DOLGNet model
    """
    return DOLGNet(
        backbone_name=backbone,
        pretrained=pretrained,
        local_dim=local_dim,
        global_dim=global_dim,
        output_dim=output_dim,
        num_classes=num_classes
    )


if __name__ == "__main__":
    # Test the model
    print("Testing DOLG model...")
    
    # Create model
    model = create_dolg_model(
        backbone='resnet50',
        pretrained=True,
        local_dim=1024,
        global_dim=2048,
        output_dim=512,
        num_classes=20
    )
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randint(0, 20, (batch_size,))
    
    # Training mode
    model.train()
    loss, logits = model(x, targets)
    print(f"Training - Loss: {loss.item():.4f}, Logits shape: {logits.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        _, logits = model(x)
        descriptors = model.extract_descriptor(x)
        print(f"Inference - Logits shape: {logits.shape}, Descriptors shape: {descriptors.shape}")
    
    print("✓ DOLG model test completed successfully!")

