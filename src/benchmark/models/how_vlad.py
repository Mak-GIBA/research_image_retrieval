"""
HOW-VLAD and HOW-ASMK implementation for image retrieval.
Based on "Learning and aggregating deep local descriptors for instance-level recognition" (ECCV 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.cluster import KMeans


class VLADPooling(nn.Module):
    """VLAD (Vector of Locally Aggregated Descriptors) pooling layer"""
    
    def __init__(self, num_clusters=64, dim=128, alpha=100.0):
        super(VLADPooling, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        
        # Learnable cluster centers
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        
        # Initialize centroids
        with torch.no_grad():
            self.centroids.data.copy_(torch.rand(num_clusters, dim))
    
    def forward(self, x):
        # x: [B, N, D] where N is number of local descriptors, D is descriptor dimension
        B, N, D = x.shape
        
        # Compute soft assignment
        # Distance to centroids: [B, N, K]
        distances = torch.cdist(x, self.centroids.unsqueeze(0).expand(B, -1, -1))
        
        # Soft assignment with temperature
        soft_assign = F.softmax(-self.alpha * distances, dim=2)  # [B, N, K]
        
        # VLAD computation
        vlad = torch.zeros(B, self.num_clusters, D, device=x.device)
        
        for k in range(self.num_clusters):
            # Residuals: x - centroid_k
            residuals = x - self.centroids[k].unsqueeze(0).unsqueeze(0)  # [B, N, D]
            
            # Weighted residuals
            weighted_residuals = soft_assign[:, :, k].unsqueeze(2) * residuals  # [B, N, D]
            
            # Sum over local descriptors
            vlad[:, k, :] = weighted_residuals.sum(dim=1)  # [B, D]
        
        # Flatten and normalize
        vlad = vlad.view(B, -1)  # [B, K*D]
        vlad = F.normalize(vlad, p=2, dim=1)
        
        return vlad


class ASMKPooling(nn.Module):
    """ASMK (Aggregated Selective Match Kernel) pooling layer"""
    
    def __init__(self, num_clusters=64, dim=128):
        super(ASMKPooling, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        
        # Learnable cluster centers
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        
        # Learnable weights for selective matching
        self.weights = nn.Parameter(torch.ones(num_clusters))
    
    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        
        # Compute distances to centroids
        distances = torch.cdist(x, self.centroids.unsqueeze(0).expand(B, -1, -1))
        
        # Find nearest centroid for each descriptor
        nearest_centroids = torch.argmin(distances, dim=2)  # [B, N]
        
        # Selective matching: only keep descriptors close to their nearest centroid
        min_distances = torch.gather(distances, 2, nearest_centroids.unsqueeze(2)).squeeze(2)
        
        # Threshold for selective matching (adaptive)
        threshold = min_distances.mean(dim=1, keepdim=True) + min_distances.std(dim=1, keepdim=True)
        mask = min_distances < threshold  # [B, N]
        
        # Aggregate features
        asmk = torch.zeros(B, self.num_clusters, device=x.device)
        
        for b in range(B):
            for n in range(N):
                if mask[b, n]:
                    k = nearest_centroids[b, n]
                    asmk[b, k] += self.weights[k]
        
        # Normalize
        asmk = F.normalize(asmk, p=2, dim=1)
        
        return asmk


class HOWModel(nn.Module):
    """HOW (Hyperdimensional Object Whitening) model base"""
    
    def __init__(self, backbone='resnet50', pretrained=True, num_classes=1000,
                 local_dim=128, pooling_type='vlad', num_clusters=64):
        super(HOWModel, self).__init__()
        
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
        
        # Local descriptor projection
        self.local_proj = nn.Conv2d(backbone_dim, local_dim, 1)
        
        # Pooling layer
        if pooling_type == 'vlad':
            self.pooling = VLADPooling(num_clusters, local_dim)
            pooled_dim = num_clusters * local_dim
        elif pooling_type == 'asmk':
            self.pooling = ASMKPooling(num_clusters, local_dim)
            pooled_dim = num_clusters
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
        
        # Final projection
        self.final_proj = nn.Linear(pooled_dim, 2048)
        
        # Classifier
        self.classifier = nn.Linear(2048, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def extract_local_descriptors(self, x):
        """Extract local descriptors"""
        # Backbone features
        backbone_features = self.backbone(x)  # [B, C, H, W]
        
        # Local descriptors
        local_desc = self.local_proj(backbone_features)  # [B, local_dim, H, W]
        
        # Reshape to [B, N, D] format
        B, D, H, W = local_desc.shape
        local_desc = local_desc.view(B, D, -1).permute(0, 2, 1)  # [B, H*W, D]
        
        # L2 normalize local descriptors
        local_desc = F.normalize(local_desc, p=2, dim=2)
        
        return local_desc
    
    def extract_features(self, x):
        """Extract global features using pooling"""
        # Local descriptors
        local_desc = self.extract_local_descriptors(x)  # [B, N, D]
        
        # Pooling
        pooled_features = self.pooling(local_desc)  # [B, pooled_dim]
        
        # Final projection
        features = self.final_proj(pooled_features)  # [B, 2048]
        
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


class HOWVLADWrapper(nn.Module):
    """Wrapper for HOW-VLAD model"""
    
    def __init__(self, num_classes, backbone='resnet50', local_dim=128, num_clusters=64):
        super(HOWVLADWrapper, self).__init__()
        self.backbone = HOWModel(
            backbone=backbone,
            num_classes=num_classes,
            local_dim=local_dim,
            pooling_type='vlad',
            num_clusters=num_clusters
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


class HOWASMKWrapper(nn.Module):
    """Wrapper for HOW-ASMK model"""
    
    def __init__(self, num_classes, backbone='resnet50', local_dim=128, num_clusters=64):
        super(HOWASMKWrapper, self).__init__()
        self.backbone = HOWModel(
            backbone=backbone,
            num_classes=num_classes,
            local_dim=local_dim,
            pooling_type='asmk',
            num_clusters=num_clusters
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


def create_how_optimizer(args, model):
    """Create optimizer for HOW models"""
    # Use Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer


# Model factory functions
def get_how_vlad_model(num_classes, backbone='resnet50', **kwargs):
    """Factory function to create HOW-VLAD model"""
    return HOWVLADWrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )


def get_how_asmk_model(num_classes, backbone='resnet50', **kwargs):
    """Factory function to create HOW-ASMK model"""
    return HOWASMKWrapper(
        num_classes=num_classes,
        backbone=backbone,
        **kwargs
    )

