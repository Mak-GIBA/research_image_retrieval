"""
ULTRON: Unifying Local Transformer and Convolution for Large-scale Image Retrieval
MadaCos Loss Function Implementation

論文の式(13), (14), (15)に基づく厳密な実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MadaCosLoss(nn.Module):
    """
    Margin-based Adaptive Cosine Loss (MadaCos)
    
    論文の式(13), (14), (15)に基づく実装:
    - 式(13): s = log((1-t)(1-ρ)/σ) / (1 - median({cos(θi)}_{i=1}^N))
    - 式(14): m = (1/N) Σ cos(θi) - (1/2) log(ρ Σ exp(s·cos(θj)) / (1-ρ))
    - 式(15): L = -log(exp(s·cos(θi - m)) / (exp(s·cos(θi - m)) + Σ exp(s·cos(θj))))
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        rho: float = 0.04,
        t: float = 0.1,
        sigma: float = 0.1,
        margin_type: str = "adaptive"  # "adaptive" or "fixed"
    ):
        super(MadaCosLoss, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.rho = rho
        self.t = t
        self.sigma = sigma
        self.margin_type = margin_type
        
        # 重み行列 (クラス中心)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # 固定マージンの場合
        if margin_type == "fixed":
            self.margin = 0.5
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embed_dim] - 正規化された特徴ベクトル
            labels: [B] - クラスラベル
        
        Returns:
            loss: スカラー損失値
        """
        batch_size = embeddings.size(0)
        
        # 特徴とクラス重みの正規化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # コサイン類似度の計算
        cosine_sim = F.linear(embeddings, weight_normalized)  # [B, num_classes]
        
        # 正解クラスのコサイン類似度
        target_cosine = cosine_sim.gather(1, labels.view(-1, 1)).squeeze(1)  # [B]
        
        # スケール係数の計算 (式13)
        if self.margin_type == "adaptive":
            # 中央値の計算
            median_cosine = torch.median(target_cosine).item()
            scale = math.log((1 - self.t) * (1 - self.rho) / self.sigma) / (1 - median_cosine + 1e-8)
        else:
            scale = 30.0  # 固定スケール
        
        # マージンの計算 (式14)
        if self.margin_type == "adaptive":
            # 平均コサイン類似度
            mean_cosine = torch.mean(target_cosine).item()
            
            # 負例のコサイン類似度の指数和
            mask = torch.ones_like(cosine_sim).scatter_(1, labels.view(-1, 1), 0)  # 正解クラスをマスク
            negative_cosines = cosine_sim * mask
            exp_sum = torch.sum(torch.exp(scale * negative_cosines), dim=1).mean().item()
            
            # 適応的マージン
            margin = mean_cosine - 0.5 * math.log(self.rho * exp_sum / (1 - self.rho) + 1e-8)
            margin = max(0.0, min(margin, 1.0))  # [0, 1]にクランプ
        else:
            margin = self.margin
        
        # マージン適用後のコサイン類似度
        target_cosine_margin = target_cosine - margin
        
        # スケール適用
        scaled_cosine = scale * cosine_sim
        scaled_target = scale * target_cosine_margin
        
        # ソフトマックス損失の計算 (式15)
        # 正解クラスのlogit
        target_logit = scaled_target
        
        # 全クラスのlogit
        all_logits = scaled_cosine
        
        # 正解クラスのlogitを更新
        all_logits.scatter_(1, labels.view(-1, 1), target_logit.view(-1, 1))
        
        # クロスエントロピー損失
        loss = F.cross_entropy(all_logits, labels)
        
        return loss

class ULTRONTrainingLoss(nn.Module):
    """
    ULTRON Training Loss with auxiliary losses
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        rho: float = 0.04,
        aux_weight: float = 0.1
    ):
        super(ULTRONTrainingLoss, self).__init__()
        
        self.madacos_loss = MadaCosLoss(num_classes, embed_dim, rho)
        self.aux_weight = aux_weight
        
        # 補助分類器 (中間特徴用)
        self.aux_classifier = nn.Linear(embed_dim // 2, num_classes)
        
    def forward(
        self,
        final_features: torch.Tensor,
        aux_features: Optional[torch.Tensor],
        labels: torch.Tensor
    ) -> dict:
        """
        Args:
            final_features: [B, embed_dim] - 最終特徴
            aux_features: [B, embed_dim//2] - 中間特徴 (optional)
            labels: [B] - クラスラベル
        
        Returns:
            dict: 損失値の辞書
        """
        # メイン損失
        main_loss = self.madacos_loss(final_features, labels)
        
        total_loss = main_loss
        loss_dict = {"main_loss": main_loss}
        
        # 補助損失
        if aux_features is not None:
            aux_logits = self.aux_classifier(aux_features)
            aux_loss = F.cross_entropy(aux_logits, labels)
            total_loss = total_loss + self.aux_weight * aux_loss
            loss_dict["aux_loss"] = aux_loss
        
        loss_dict["total_loss"] = total_loss
        
        return loss_dict

def test_madacos_loss():
    """MadaCos損失のテスト"""
    print("=== MadaCos Loss テスト ===")
    
    # テスト設定
    batch_size = 32
    num_classes = 1000
    embed_dim = 512
    
    # 損失関数
    madacos_loss = MadaCosLoss(num_classes, embed_dim, rho=0.04)
    training_loss = ULTRONTrainingLoss(num_classes, embed_dim, rho=0.04)
    
    # テストデータ
    embeddings = torch.randn(batch_size, embed_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2正規化
    labels = torch.randint(0, num_classes, (batch_size,))
    aux_features = torch.randn(batch_size, embed_dim // 2)
    
    print(f"入力特徴形状: {embeddings.shape}")
    print(f"ラベル形状: {labels.shape}")
    print(f"補助特徴形状: {aux_features.shape}")
    
    # MadaCos損失テスト
    with torch.no_grad():
        loss_value = madacos_loss(embeddings, labels)
        print(f"MadaCos損失値: {loss_value.item():.4f}")
    
    # 学習損失テスト
    with torch.no_grad():
        loss_dict = training_loss(embeddings, aux_features, labels)
        print(f"メイン損失: {loss_dict['main_loss'].item():.4f}")
        print(f"補助損失: {loss_dict['aux_loss'].item():.4f}")
        print(f"総損失: {loss_dict['total_loss'].item():.4f}")
    
    # 勾配テスト
    embeddings_grad = torch.randn(batch_size, embed_dim, requires_grad=True)
    loss_grad = madacos_loss(embeddings_grad, labels)
    loss_grad.backward()
    
    print(f"勾配計算成功: {embeddings_grad.grad is not None}")
    if embeddings_grad.grad is not None:
        print(f"勾配ノルム: {embeddings_grad.grad.norm().item():.4f}")
    else:
        print("勾配が計算されませんでした")
    
    # パラメータ数
    total_params = sum(p.numel() for p in madacos_loss.parameters())
    print(f"MadaCos パラメータ数: {total_params:,}")
    
    print("✓ MadaCos Loss テスト完了")

if __name__ == "__main__":
    test_madacos_loss()

