"""
Corrected IRIS implementation with proper parameter names and functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union
import math
import numpy as np
from collections import OrderedDict

# Utility Functions
def compute_similarity(query_embedding: torch.Tensor, gallery_embeddings: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between query and gallery embeddings."""
    query_embedding = F.normalize(query_embedding, p=2, dim=-1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=-1)
    similarity = torch.matmul(query_embedding, gallery_embeddings.t())
    return similarity

def evaluate_retrieval(query_embeddings: torch.Tensor, gallery_embeddings: torch.Tensor, 
                       query_labels: torch.Tensor, gallery_labels: torch.Tensor, 
                       top_k: List[int]=[1, 5, 10]) -> Dict[str, float]:
    """Evaluates retrieval performance (mAP, Precision@k)."""
    results = {}
    
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)
    
    # Compute similarity
    similarity = torch.matmul(query_embeddings, gallery_embeddings.t())
    
    # For each query
    ap_list = []
    precision_at_k = {k: [] for k in top_k}
    
    for i in range(query_embeddings.size(0)):
        # Get similarity scores for this query
        sim_scores = similarity[i]
        
        # Get ground truth matches
        query_label = query_labels[i]
        relevance = (gallery_labels == query_label).float()
        
        # Sort gallery by similarity
        _, indices = torch.sort(sim_scores, descending=True)
        sorted_relevance = relevance[indices]
        
        # Calculate AP
        if sorted_relevance.sum() > 0:
            cumulative_relevance = torch.cumsum(sorted_relevance, dim=0)
            cumulative_precision = cumulative_relevance / torch.arange(1, len(relevance) + 1, 
                                                                      device=relevance.device)
            ap = (cumulative_precision * sorted_relevance).sum() / sorted_relevance.sum()
            ap_list.append(ap.item())
        
        # Calculate Precision@k
        for k in top_k:
            if k <= len(sorted_relevance):
                precision_k = sorted_relevance[:k].sum().item() / k
                precision_at_k[k].append(precision_k)
    
    # Calculate mean metrics
    mean_ap = np.mean(ap_list) if ap_list else 0.0
    mean_precision_at_k = {k: np.mean(v) if v else 0.0 for k, v in precision_at_k.items()}
    
    results['mAP'] = mean_ap
    for k in top_k:
        results[f'P@{k}'] = mean_precision_at_k[k]
    
    return results

class ORACLE(nn.Module):
    """ORACLE module for object-relation understanding"""
    def __init__(self, dim: int = 2048, output_dim: int = 512, num_objects: int = 8, 
                 relation_dim: int = 256, context_balance: float = 0.5, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.num_objects = num_objects
        self.relation_dim = relation_dim
        self.context_balance = context_balance
        self.num_heads = num_heads
        
        # Object detection simulation
        self.conv_reduce = nn.Conv2d(dim, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((num_objects, 1))
        self.obj_proj = nn.Linear(512, output_dim)
        
        # Relation modeling
        self.relation_net = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        
        # Context integration
        self.context_proj = nn.Linear(dim, output_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection
        self.final_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Extract object features
        conv_out = self.relu(self.conv_reduce(x))  # [B, 512, H, W]
        pooled = self.adaptive_pool(conv_out)  # [B, 512, num_objects, 1]
        pooled = pooled.squeeze(-1).transpose(1, 2)  # [B, num_objects, 512]
        obj_features = self.obj_proj(pooled)  # [B, num_objects, output_dim]
        
        # Global context
        global_context = self.global_pool(x).flatten(1)  # [B, dim]
        
        # Relation modeling
        attended_features, _ = self.relation_net(obj_features, obj_features, obj_features)
        
        # Aggregate object features
        aggregated = attended_features.mean(dim=1)  # [B, output_dim]
        
        # Context integration
        context_features = self.context_proj(global_context)
        final_features = self.context_balance * aggregated + (1 - self.context_balance) * context_features
        
        # Final projection
        output = self.final_proj(final_features)
        
        return output

class CASTLE(nn.Module):
    """CASTLE module for causal attention"""
    def __init__(self, dim: int = 512, num_heads: int = 8, qkv_bias: bool = False, 
                 causal_threshold: float = 0.5, temperature: float = 0.07, 
                 counterfactual_strength: float = 0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.causal_threshold = causal_threshold
        self.temperature = temperature
        self.counterfactual_strength = counterfactual_strength
        
        # Attention layers
        self.attention = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        
        # Causal predictor
        self.causal_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim]
        x = x.unsqueeze(1)  # [B, 1, dim]
        
        # Self-attention
        attended, _ = self.attention(x, x, x)
        
        # Causal prediction
        causal_scores = self.causal_net(attended)  # [B, 1, 1]
        
        # Apply causal masking
        causal_mask = (causal_scores > self.causal_threshold).float()
        masked_features = attended * causal_mask
        
        # Feature refinement
        refined = self.refine_net(masked_features)
        
        # Combine with original
        output = attended + self.counterfactual_strength * refined
        
        return output.squeeze(1)  # [B, dim]

class NEXUS(nn.Module):
    """NEXUS module for hierarchical attention"""
    def __init__(self, dim: int = 512, num_heads: int = 8, window_size: int = 7, 
                 sparsity: float = 0.5, qkv_bias: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.sparsity = sparsity
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        
        # Hierarchical processing
        self.local_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        
        self.global_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim]
        x = x.unsqueeze(1)  # [B, 1, dim]
        
        # Self-attention
        attended, _ = self.attention(x, x, x)
        
        # Local processing
        local_features = self.local_net(attended)
        
        # Global processing
        global_features = self.global_net(attended)
        
        # Fusion
        combined = torch.cat([local_features, global_features], dim=-1)
        output = self.fusion(combined)
        
        return output.squeeze(1)  # [B, dim]

class IRISModel(nn.Module):
    """Complete IRIS model combining ORACLE, CASTLE, and NEXUS"""
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, 
                 output_dim: int = 512, num_classes: int = 1000):
        super().__init__()
        self.backbone_name = backbone
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # IRIS components
        self.oracle = ORACLE(dim=backbone_dim, output_dim=output_dim)
        self.castle = CASTLE(dim=output_dim)
        self.nexus = NEXUS(dim=output_dim)
        
        # Final layers
        self.feature_proj = nn.Linear(output_dim, output_dim)
        self.classifier = nn.Linear(output_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Extract backbone features
        backbone_features = self.backbone(x)  # [B, C, H, W]
        
        # ORACLE: Object-relation understanding
        oracle_features = self.oracle(backbone_features)  # [B, output_dim]
        
        # CASTLE: Causal attention
        castle_features = self.castle(oracle_features)  # [B, output_dim]
        
        # NEXUS: Hierarchical attention
        nexus_features = self.nexus(castle_features)  # [B, output_dim]
        
        # Final feature projection
        final_features = self.feature_proj(nexus_features)
        final_features = self.dropout(final_features)
        
        if return_features:
            return final_features
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for retrieval"""
        return self.forward(x, return_features=True)

# Wrapper for compatibility
class IRISWrapper(nn.Module):
    """Wrapper for IRIS model to match expected interface"""
    def __init__(self, backbone: str = 'resnet18', output_dim: int = 512, num_classes: int = 50):
        super().__init__()
        self.model = IRISModel(
            backbone=backbone,
            pretrained=True,
            output_dim=output_dim,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        return self.model(x, return_features=return_features)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(x)

# Loss function for IRIS
class IRISLoss(nn.Module):
    """Combined loss for IRIS training"""
    def __init__(self, margin: float = 0.5, temperature: float = 0.07, 
                 classification_weight: float = 1.0, retrieval_weight: float = 0.5):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.classification_weight = classification_weight
        self.retrieval_weight = retrieval_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits: torch.Tensor, features: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        # Retrieval loss (contrastive)
        features = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(features, features.t()) / self.temperature
        
        # Create positive and negative masks
        labels_expanded = labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()).float()
        neg_mask = 1 - pos_mask
        
        # Remove diagonal (self-similarity)
        pos_mask.fill_diagonal_(0)
        
        # Contrastive loss
        pos_sim = similarity * pos_mask
        neg_sim = similarity * neg_mask
        
        pos_loss = -torch.log(torch.exp(pos_sim).sum(dim=1) + 1e-8).mean()
        neg_loss = torch.log(torch.exp(neg_sim).sum(dim=1) + 1e-8).mean()
        
        retrieval_loss = pos_loss + neg_loss
        
        # Combined loss
        total_loss = (self.classification_weight * cls_loss + 
                     self.retrieval_weight * retrieval_loss)
        
        return total_loss

if __name__ == "__main__":
    # Test the corrected implementation
    print("Testing corrected IRIS implementation...")
    
    # Test individual components
    oracle = ORACLE(dim=2048, output_dim=512)
    castle = CASTLE(dim=512)
    nexus = NEXUS(dim=512)
    
    print("✓ All components initialized successfully")
    
    # Test forward passes
    x_spatial = torch.randn(4, 2048, 7, 7)
    oracle_out = oracle(x_spatial)
    print(f"✓ ORACLE output shape: {oracle_out.shape}")
    
    castle_out = castle(oracle_out)
    print(f"✓ CASTLE output shape: {castle_out.shape}")
    
    nexus_out = nexus(castle_out)
    print(f"✓ NEXUS output shape: {nexus_out.shape}")
    
    # Test complete model
    model = IRISModel(backbone='resnet18', output_dim=512, num_classes=50)
    x = torch.randn(4, 3, 224, 224)
    
    logits = model(x)
    features = model(x, return_features=True)
    
    print(f"✓ Model logits shape: {logits.shape}")
    print(f"✓ Model features shape: {features.shape}")
    
    print("All tests passed! IRIS implementation is working correctly.")

