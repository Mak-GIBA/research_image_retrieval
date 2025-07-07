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

class AdaptiveHybridModel(nn.Module):
    """
    提案手法のメインモデル。
    画像からSC-GeM, Regional-GeM, Scale-GeM特徴を抽出する。
    """
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, output_dim: int = 2048):
        super().__init__()
        self.backbone_name = backbone
        self.output_dim = output_dim # 最終的な融合特徴の次元

        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Identity() # 最後のFC層を削除
            self.feature_dim = 2048 # ResNet50のconv5_xの出力次元
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Identity() # 最後のFC層を削除
            self.feature_dim = 512 # ResNet18のconv5_xの出力次元
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.gem_pool = GeM() # Generalized Mean Pooling

        # 各特徴の次元を最終出力次元に合わせるためのプロジェクション層
        # SC-GeMはGlobal特徴なので、そのままfeature_dim
        # Regional-GeMは2x2で4領域なので feature_dim * 4
        # Scale-GeMは2スケールなので feature_dim * 2
        # 各特徴が最終出力次元の1/3を占めるように仮定し、合計がoutput_dimになるように調整
        # ただし、割り切れない場合があるので、最後のプロジェクション層で調整
        self.proj_sc_gem = nn.Linear(self.feature_dim, output_dim // 3)
        self.proj_regional_gem = nn.Linear(self.feature_dim * 4, output_dim // 3)
        self.proj_scale_gem = nn.Linear(self.feature_dim * 2, output_dim - 2 * (output_dim // 3)) # 残りの次元を割り当てる

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # バックボーンによる特徴抽出
        # ResNetのconv5_x層の出力を空間特徴として利用
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        spatial_features = self.backbone.layer4(x) # [B, C, H, W]

        # ハイブリッド特徴表現の生成
        sc_gem_feature = self.gem_pool(spatial_features).squeeze(-1).squeeze(-1) # Global GeM
        regional_gem_feature = self._regional_gem(spatial_features) # Regional GeM
        scale_gem_feature = self._scale_gem(spatial_features) # Scale GeM

        # 各特徴を最終出力次元の1/3にプロジェクション
        sc_gem_feature_proj = self.proj_sc_gem(sc_gem_feature)
        regional_gem_feature_proj = self.proj_regional_gem(regional_gem_feature)
        scale_gem_feature_proj = self.proj_scale_gem(scale_gem_feature)

        return sc_gem_feature_proj, regional_gem_feature_proj, scale_gem_feature_proj

    def _regional_gem(self, x: torch.Tensor) -> torch.Tensor:
        """空間特徴を複数の領域に分割し、それぞれにGeM Poolingを適用"""
        # 例: 2x2グリッドに分割
        regions = []
        h, w = x.shape[2:]
        h_half, w_half = h // 2, w // 2

        regions.append(self.gem_pool(x[:, :, :h_half, :w_half]))
        regions.append(self.gem_pool(x[:, :, :h_half, w_half:]))
        regions.append(self.gem_pool(x[:, :, h_half:, :w_half]))
        regions.append(self.gem_pool(x[:, :, h_half:, w_half:]))
        
        # 各領域の特徴を結合
        return torch.cat(regions, dim=1).squeeze(-1).squeeze(-1)

    def _scale_gem(self, x: torch.Tensor) -> torch.Tensor:
        """異なるスケールでGeM Poolingを適用"""
        # 例: オリジナルスケールと半分のスケール
        original_scale_feature = self.gem_pool(x)
        
        # 半分のスケールにリサイズしてGeM
        # F.interpolateのalign_cornersはTrueにすると警告が出る場合があるため注意
        half_scale_x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        half_scale_feature = self.gem_pool(half_scale_x)
        
        return torch.cat([original_scale_feature, half_scale_feature], dim=1).squeeze(-1).squeeze(-1)

class QAFF(nn.Module):
    """
    Query-Adaptive Feature Fusion (QAFF) Module.
    クエリ特徴に基づいて、ギャラリー画像の複数の特徴（SC-GeM, Regional-GeM, Scale-GeM）を融合する。
    """
    def __init__(self, feature_dim: int, num_feature_types: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_feature_types = num_feature_types

        # クエリ特徴から融合重みを生成するためのネットワーク
        # クエリ特徴を線形変換し、各特徴タイプに対応する重みを生成
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_feature_types),
            nn.Softmax(dim=-1) # 重みの合計が1になるようにSoftmax
        )

    def forward(self, query_feature: torch.Tensor, gallery_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            query_feature: クエリ画像の特徴 [B, feature_dim]
            gallery_features: ギャラリー画像の各特徴のリスト [ [B, feature_dim], [B, feature_dim], [B, feature_dim] ]
        Returns:
            融合されたギャラリー画像の特徴 [B, feature_dim]
        """
        # クエリ特徴から融合重みを生成
        # query_feature: [B, feature_dim] -> weights: [B, num_feature_types]
        weights = self.weight_generator(query_feature)

        # 重み付け和による特徴融合
        fused_feature = torch.zeros_like(gallery_features[0]) # 最初の特徴の形状をコピー
        for i, gal_feat in enumerate(gallery_features):
            # weights[:, i] は [B] なので、[B, 1] に拡張してブロードキャスト
            fused_feature += weights[:, i].unsqueeze(1) * gal_feat

        return fused_feature

class AdaptiveHybridRetrieval(nn.Module):
    """効率的な単一ステージ検索のためのラッパーモデル"""
    def __init__(self, model: AdaptiveHybridModel, qaff_module: QAFF):
        super().__init__()
        self.model = model # AdaptiveHybridModel
        self.qaff_module = qaff_module # QAFFモジュール

        self.gallery_sc_gem_embeddings = None
        self.gallery_regional_gem_embeddings = None
        self.gallery_scale_gem_embeddings = None
        self.gallery_labels = None
        self.gallery_paths = None

    @torch.no_grad()
    def add_to_gallery(self, images: torch.Tensor, labels: torch.Tensor, paths: List[str]):
        """ギャラリーに画像を追加し、特徴を事前に抽出して保存する"""
        self.model.eval()
        sc_gems, regional_gems, scale_gems = self.model(images)

        self.gallery_sc_gem_embeddings = sc_gems
        self.gallery_regional_gem_embeddings = regional_gems
        self.gallery_scale_gem_embeddings = scale_gems
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

        # クエリ画像から特徴を抽出
        query_sc_gem, query_regional_gem, query_scale_gem = self.model(query_image)
        # ここでは、クエリの特徴としてSC-GeM特徴を代表として使用する
        # もしクエリ側も融合が必要なら、別途QAFFを適用するか、別の融合戦略を考える
        # 今回は、クエリがギャラリーの特徴融合を「ガイド」するという役割なので、
        # クエリ自体は単一の代表特徴（例: SC-GeM）で十分と仮定
        query_embedding_for_qaff = query_sc_gem # [1, feature_dim] (バッチサイズ1を想定)

        # ギャラリーの各特徴タイプをリストにまとめる
        gallery_features_list = [
            self.gallery_sc_gem_embeddings,
            self.gallery_regional_gem_embeddings,
            self.gallery_scale_gem_embeddings
        ]

        # QAFFを適用してギャラリー特徴を融合
        # QAFFはバッチ処理を想定しているので、query_embedding_for_qaffをギャラリーサイズに拡張
        # query_embedding_for_qaff: [1, feature_dim] -> [N_gallery, feature_dim]
        # gallery_features_listの各要素: [N_gallery, feature_dim]
        num_gallery = self.gallery_sc_gem_embeddings.shape[0]
        expanded_query_embedding = query_embedding_for_qaff.expand(num_gallery, -1)

        fused_gallery_embeddings = self.qaff_module(expanded_query_embedding, gallery_features_list)

        # クエリと融合されたギャラリー特徴との類似度を計算
        # ここでは、クエリの代表特徴（query_sc_gem）と融合されたギャラリー特徴の類似度を計算
        similarity = compute_similarity(query_sc_gem, fused_gallery_embeddings)

        # 上位K件を取得
        scores, indices = torch.topk(similarity, k=top_k, dim=-1)

        # 対応するパスを取得
        retrieved_paths = [self.gallery_paths[i] for i in indices.squeeze(0).tolist()]

        return scores, indices, retrieved_paths


