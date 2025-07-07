import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union
import math
import numpy as np
from collections import OrderedDict

# Utility Functions (from iris_implementation.py)
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
        
        # Compute AP
        cumulative_relevance = torch.cumsum(sorted_relevance, dim=0)
        cumulative_precision = cumulative_relevance / torch.arange(1, len(relevance) + 1, 
                                                                  device=relevance.device)
        ap = torch.sum(sorted_relevance * cumulative_precision) / torch.sum(relevance).clamp(min=1)
        ap_list.append(ap.item())
        
        # Compute Precision@k
        for k in top_k:
            if k <= len(indices):
                precision = torch.sum(sorted_relevance[:k]) / k
                precision_at_k[k].append(precision.item())
    
    # Compute mAP
    results["mAP"] = np.mean(ap_list)
    
    # Compute average Precision@k
    for k in top_k:
        results[f"P@{k}"] = np.mean(precision_at_k[k]) if precision_at_k[k] else 0.0
    
    return results

# --- Adaptive Hybrid Feature Learning for Efficient Image Retrieval --- #

class GeM(nn.Module):
    """Generalized Mean Pooling (GeM)"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

class SpatialContextAwareLocalAttention(nn.Module):
    """Spatial Context-Aware Local Attention (SCALA)"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Spatial position encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 49, dim))  # 7x7 spatial positions
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Reshape to sequence format
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add positional encoding
        if x_seq.size(1) == self.pos_embed.size(1):
            x_seq = x_seq + self.pos_embed
        
        # Multi-head attention
        qkv = self.qkv(x_seq).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_out = self.proj(x_out)
        
        # Reshape back to spatial format
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out + x  # Residual connection

class ChannelwiseDilatedConvolution(nn.Module):
    """Channel-wise Dilated Convolution (CDConv)"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Different dilation rates for different channel groups
        self.conv1 = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, 
                              padding=1, dilation=1, groups=1)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, 
                              padding=2, dilation=2, groups=1)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, 
                              padding=3, dilation=3, groups=1)
        self.conv4 = nn.Conv2d(in_channels - 3 * (in_channels // 4), 
                              out_channels - 3 * (out_channels // 4), kernel_size, 
                              padding=4, dilation=4, groups=1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split channels into groups
        x1 = x[:, :self.in_channels//4]
        x2 = x[:, self.in_channels//4:self.in_channels//2]
        x3 = x[:, self.in_channels//2:3*self.in_channels//4]
        x4 = x[:, 3*self.in_channels//4:]
        
        # Apply different dilated convolutions
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        out3 = self.conv3(x3)
        out4 = self.conv4(x4)
        
        # Concatenate outputs
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class EnhancedBackbone(nn.Module):
    """Enhanced backbone with SCALA and CDConv"""
    def __init__(self, backbone_name: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        if backbone_name == 'resnet50':
            backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 2048
        elif backbone_name == 'resnet18':
            backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Add SCALA and CDConv enhancements
        self.scala = SpatialContextAwareLocalAttention(self.feature_dim)
        self.cdconv = ChannelwiseDilatedConvolution(self.feature_dim, self.feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply enhancements
        x = self.scala(x)
        x = self.cdconv(x)
        
        return x

class AdaptiveHybridModel(nn.Module):
    """
    提案手法のメインモデル。
    画像からSC-GeM, Regional-GeM, Scale-GeM特徴を抽出する。
    """
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, output_dim: int = 2048):
        super().__init__()
        self.backbone_name = backbone
        self.output_dim = output_dim
        
        # Enhanced backbone
        self.backbone = EnhancedBackbone(backbone, pretrained)
        self.feature_dim = self.backbone.feature_dim
        
        # GeM pooling layers
        self.gem_pool = GeM()
        
        # Projection layers to ensure consistent dimensions
        self.proj_sc_gem = nn.Linear(self.feature_dim, output_dim)
        self.proj_regional_gem = nn.Linear(self.feature_dim * 4, output_dim)  # 2x2 regions
        self.proj_scale_gem = nn.Linear(self.feature_dim * 2, output_dim)     # 2 scales
        
        # Online token learning for SC-GeM
        self.token_learner = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract spatial features using enhanced backbone
        spatial_features = self.backbone(x)  # [B, C, H, W]
        
        # Generate hybrid feature representations
        sc_gem_feature = self._sc_gem(spatial_features)
        regional_gem_feature = self._regional_gem(spatial_features)
        scale_gem_feature = self._scale_gem(spatial_features)
        
        # Project to consistent dimensions
        sc_gem_feature_proj = self.proj_sc_gem(sc_gem_feature)
        regional_gem_feature_proj = self.proj_regional_gem(regional_gem_feature)
        scale_gem_feature_proj = self.proj_scale_gem(scale_gem_feature)
        
        return sc_gem_feature_proj, regional_gem_feature_proj, scale_gem_feature_proj
    
    def _sc_gem(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial Context-aware Global GeM feature"""
        # Apply GeM pooling
        global_feat = self.gem_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        
        # Apply online token learning for spatial context awareness
        attention_weights = self.token_learner(global_feat)
        enhanced_feat = global_feat * attention_weights
        
        return enhanced_feat
    
    def _regional_gem(self, x: torch.Tensor) -> torch.Tensor:
        """Regional GeM features from 2x2 grid"""
        B, C, H, W = x.shape
        h_half, w_half = H // 2, W // 2
        
        # Extract 4 regions
        regions = []
        regions.append(self.gem_pool(x[:, :, :h_half, :w_half]))      # Top-left
        regions.append(self.gem_pool(x[:, :, :h_half, w_half:]))      # Top-right
        regions.append(self.gem_pool(x[:, :, h_half:, :w_half]))      # Bottom-left
        regions.append(self.gem_pool(x[:, :, h_half:, w_half:]))      # Bottom-right
        
        # Concatenate regional features
        regional_feat = torch.cat(regions, dim=1).squeeze(-1).squeeze(-1)  # [B, 4*C]
        
        return regional_feat
    
    def _scale_gem(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale GeM features"""
        # Original scale
        original_feat = self.gem_pool(x)
        
        # Half scale
        half_scale_x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        half_scale_feat = self.gem_pool(half_scale_x)
        
        # Concatenate scale features
        scale_feat = torch.cat([original_feat, half_scale_feat], dim=1).squeeze(-1).squeeze(-1)  # [B, 2*C]
        
        return scale_feat

class QAFF(nn.Module):
    """
    Query-Adaptive Feature Fusion (QAFF) Module.
    クエリ特徴に基づいて、複数の特徴（SC-GeM, Regional-GeM, Scale-GeM）を動的に融合する。
    """
    def __init__(self, feature_dim: int, num_feature_types: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_feature_types = num_feature_types
        
        # Attention network for weight generation
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, num_feature_types),
            nn.Softmax(dim=-1)
        )
        
        # Feature normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query_feature: torch.Tensor, gallery_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            query_feature: クエリ画像の特徴 [B, feature_dim]
            gallery_features: ギャラリー画像の各特徴のリスト [[B, feature_dim], [B, feature_dim], [B, feature_dim]]
        Returns:
            融合されたギャラリー画像の特徴 [B, feature_dim]
        """
        # Normalize query feature
        query_feature = self.layer_norm(query_feature)
        
        # Generate fusion weights based on query
        weights = self.weight_generator(query_feature)  # [B, num_feature_types]
        
        # Ensure all gallery features have the same dimension
        normalized_gallery_features = []
        for feat in gallery_features:
            normalized_gallery_features.append(self.layer_norm(feat))
        
        # Weighted fusion
        fused_feature = torch.zeros_like(normalized_gallery_features[0])
        for i, gal_feat in enumerate(normalized_gallery_features):
            weight = weights[:, i].unsqueeze(1)  # [B, 1]
            fused_feature += weight * gal_feat
        
        return fused_feature

class AdaptiveHybridRetrieval(nn.Module):
    """効率的な単一ステージ検索のためのラッパーモデル"""
    def __init__(self, model: AdaptiveHybridModel, qaff_module: QAFF):
        super().__init__()
        self.model = model
        self.qaff_module = qaff_module
        
        # Gallery storage
        self.gallery_sc_gem_embeddings = None
        self.gallery_regional_gem_embeddings = None
        self.gallery_scale_gem_embeddings = None
        self.gallery_labels = None
        self.gallery_paths = None
        
    @torch.no_grad()
    def add_to_gallery(self, images: torch.Tensor, labels: torch.Tensor, paths: List[str]):
        """ギャラリーに画像を追加し、特徴を事前に抽出して保存する"""
        self.model.eval()
        
        # Extract features in batches to handle memory efficiently
        batch_size = 32
        all_sc_gems, all_regional_gems, all_scale_gems = [], [], []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            sc_gems, regional_gems, scale_gems = self.model(batch_images)
            all_sc_gems.append(sc_gems)
            all_regional_gems.append(regional_gems)
            all_scale_gems.append(scale_gems)
        
        # Concatenate all batches
        self.gallery_sc_gem_embeddings = torch.cat(all_sc_gems, dim=0)
        self.gallery_regional_gem_embeddings = torch.cat(all_regional_gems, dim=0)
        self.gallery_scale_gem_embeddings = torch.cat(all_scale_gems, dim=0)
        self.gallery_labels = labels
        self.gallery_paths = paths
        
    @torch.no_grad()
    def search(self, query_image: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        単一ステージ検索を実行。
        クエリ画像とギャラリーの特徴を入力として、QAFFを適用し類似度を計算する。
        """
        if self.gallery_sc_gem_embeddings is None:
            raise ValueError("Gallery is empty. Call add_to_gallery first.")
        
        self.model.eval()
        self.qaff_module.eval()
        
        # Extract query features
        query_sc_gem, query_regional_gem, query_scale_gem = self.model(query_image)
        
        # Use SC-GeM as the representative query feature for QAFF guidance
        query_embedding_for_qaff = query_sc_gem  # [1, feature_dim]
        
        # Prepare gallery features list
        gallery_features_list = [
            self.gallery_sc_gem_embeddings,
            self.gallery_regional_gem_embeddings,
            self.gallery_scale_gem_embeddings
        ]
        
        # Expand query embedding to match gallery size for QAFF
        num_gallery = self.gallery_sc_gem_embeddings.shape[0]
        expanded_query_embedding = query_embedding_for_qaff.expand(num_gallery, -1)
        
        # Apply QAFF to fuse gallery features
        fused_gallery_embeddings = self.qaff_module(expanded_query_embedding, gallery_features_list)
        
        # For query, we also need to create a fused representation
        # Use equal weights for query fusion (could be made adaptive too)
        query_features_list = [query_sc_gem, query_regional_gem, query_scale_gem]
        query_weights = torch.ones(1, 3, device=query_sc_gem.device) / 3.0  # Equal weights
        query_fused = torch.zeros_like(query_sc_gem)
        for i, feat in enumerate(query_features_list):
            query_fused += query_weights[:, i].unsqueeze(1) * feat
        
        # Compute similarity between fused query and fused gallery features
        similarity = compute_similarity(query_fused, fused_gallery_embeddings)
        
        # Get top-k results
        scores, indices = torch.topk(similarity, k=min(top_k, len(self.gallery_paths)), dim=-1)
        
        # Get corresponding paths
        retrieved_paths = [self.gallery_paths[i] for i in indices.squeeze(0).tolist()]
        
        return scores, indices, retrieved_paths

# Training utilities
class ContrastiveLoss(nn.Module):
    """Contrastive loss for training the retrieval model"""
    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = 1.0 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Apply InfoNCE-style loss
        # For each sample, compute loss against all others
        losses = []
        for i in range(embeddings.size(0)):
            # Get positive and negative similarities for sample i
            pos_sims = similarity_matrix[i] * positive_mask[i]
            neg_sims = similarity_matrix[i] * negative_mask[i]
            
            # Get actual positive similarities (non-zero)
            pos_sims_actual = pos_sims[pos_sims > 0]
            
            if len(pos_sims_actual) > 0:
                # Compute InfoNCE loss
                numerator = torch.exp(pos_sims_actual).sum()
                denominator = torch.exp(similarity_matrix[i]).sum() - torch.exp(similarity_matrix[i, i])  # Exclude self
                
                if denominator > 0:
                    loss_i = -torch.log(numerator / (denominator + 1e-8))
                    losses.append(loss_i)
        
        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            # Fallback: simple margin-based loss
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        return torch.abs(loss)  # Ensure positive loss

def train_model(model: AdaptiveHybridModel, qaff: QAFF, train_loader, val_loader, 
                num_epochs: int = 100, lr: float = 1e-4, device: str = 'cuda'):
    """Training function for the adaptive hybrid retrieval model"""
    
    # Move models to device
    model = model.to(device)
    qaff = qaff.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(qaff.parameters()),
        lr=lr, weight_decay=1e-4
    )
    
    # Loss function
    criterion = ContrastiveLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        qaff.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Extract features
            sc_gem, regional_gem, scale_gem = model(images)
            
            # Create query-gallery pairs for training
            batch_size = images.size(0)
            query_indices = torch.arange(batch_size, device=device)
            
            # Use SC-GeM as query guidance
            query_features = sc_gem[query_indices]
            gallery_features_list = [sc_gem, regional_gem, scale_gem]
            
            # Apply QAFF
            fused_features = qaff(query_features, gallery_features_list)
            
            # Compute loss
            loss = criterion(fused_features, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        qaff.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                sc_gem, regional_gem, scale_gem = model(images)
                query_features = sc_gem
                gallery_features_list = [sc_gem, regional_gem, scale_gem]
                fused_features = qaff(query_features, gallery_features_list)
                
                loss = criterion(fused_features, labels)
                val_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'qaff_state_dict': qaff.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_adaptive_hybrid_model.pth')
    
    print(f'Training completed. Best validation loss: {best_val_loss:.4f}')

