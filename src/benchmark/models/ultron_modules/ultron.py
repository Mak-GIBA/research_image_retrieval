"""
ULTRON: Unifying Local Transformer and Convolution for Large-scale Image Retrieval
Main ULTRON Network Implementation

論文のFigure 1に基づく4ステージアーキテクチャの実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

from models.ultron_modules.cdconv import CDConv, CDConvBlock
from scala import SCALABlock

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super(PatchEmbed, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # パッチ埋め込み
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]

        if self.norm is not None:
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        return x

class PatchMerging(nn.Module):
    """
    Patch Merging Layer
    """

    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # パッチマージング (2x2 -> 1)
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2, W/2, 2*C]

        # [B, H/2, W/2, 2*C] -> [B, 2*C, H/2, W/2]
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

class ULTRONStage(nn.Module):
    """
    ULTRON Stage (CDConv or SCALA blocks)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        stage_type: str = "cdconv",  # "cdconv" or "scala"
        window_size: int = 7,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] = None,
        downsample: Optional[nn.Module] = None,
        tau1: float = 0.75,
        tau2: float = 0.50,
        delta1: int = 3,
        delta2: int = 6
    ):
        super(ULTRONStage, self).__init__()

        self.dim = dim
        self.depth = depth
        self.stage_type = stage_type

        if drop_path is None:
            drop_path = [0.0] * depth

        # ブロックの構築
        self.blocks = nn.ModuleList()

        for i in range(depth):
            if stage_type == "cdconv":
                # CDConv Block
                block = CDConvBlock(
                    in_channels=dim,
                    out_channels=dim,
                    tau1=tau1,
                    tau2=tau2,
                    delta1=delta1,
                    delta2=delta2
                )
            elif stage_type == "scala":
                # SCALA Block
                block = SCALABlock(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                )
            else:
                raise ValueError(f"Unknown stage_type: {stage_type}")

            self.blocks.append(block)

        # ダウンサンプリング
        if downsample is not None:
            self.downsample = downsample(dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

class AttentionBasedGlobalPooling(nn.Module):
    """
    Attention-based Global Pooling
    論文の式(9), (10), (11), (12)に基づく実装
    """

    def __init__(self, dim: int, gamma: float = 2.0, alpha: float = 2.0):
        super(AttentionBasedGlobalPooling, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.tensor(gamma))  # GeM pooling parameter
        self.alpha = alpha  # scaling parameter

        # Query descriptor projection
        self.query_proj = nn.Linear(dim, dim)

        # Attention weights
        self.attention_weights = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def gem_pooling(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Generalized Mean (GeM) Pooling
        """
        # クランプしてオーバーフローを防ぐ
        gamma_clamped = torch.clamp(gamma, min=1e-7, max=100.0)

        # GeM pooling
        x_pow = torch.clamp(x, min=1e-7)  # 負の値を避ける
        pooled = F.adaptive_avg_pool2d(x_pow.pow(gamma_clamped), 1)
        pooled = pooled.pow(1.0 / gamma_clamped)

        return pooled.squeeze(-1).squeeze(-1)  # [B, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        式(9): dq = Wq · (1/HW ∑∑ X^(h,w))^(1/γ)
        式(10): A = softmax(K^T dq / √d)
        式(11): fα(x) = {x^(1/α) if α ≠ 1, ln x if α = 1}
        式(12): d = fα^(-1)(Gα(V)A)
        """
        B, C, H, W = x.shape

        # 1. GeM pooling による初期記述子の取得 (式9)
        dq = self.gem_pooling(x, self.gamma)  # [B, C]
        dq = self.query_proj(dq)  # [B, C]

        # 2. 特徴マップをKey, Valueに変換
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        K = x_flat  # [B, HW, C]
        V = x_flat  # [B, HW, C]

        # 3. 注意スコアの計算 (式10)
        attention_scores = torch.matmul(K, dq.unsqueeze(-1)).squeeze(-1)  # [B, HW]
        attention_scores = attention_scores / math.sqrt(self.dim)
        A = F.softmax(attention_scores, dim=-1)  # [B, HW]

        # 4. 重み付き特徴の計算
        weighted_features = torch.matmul(A.unsqueeze(1), V).squeeze(1)  # [B, C]

        # 5. スケーリング関数の適用 (式11, 12)
        if self.alpha != 1.0:
            # fα(x) = x^(1/α), fα^(-1)(x) = x^α
            output = weighted_features.pow(self.alpha)
        else:
            # fα(x) = ln x, fα^(-1)(x) = exp(x)
            output = torch.exp(weighted_features)

        # L2正規化
        output = F.normalize(output, p=2, dim=1)

        return output

class ULTRON(nn.Module):
    """
    ULTRON: Unifying Local Transformer and Convolution
    論文のFigure 1に基づく4ステージアーキテクチャ
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [3, 5, 9, 5],  # ULTRON-S configuration
        window_size: int = 7,
        num_heads: List[int] = [3, 6, 12, 24],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        tau1: float = 0.75,
        tau2: float = 0.50,
        delta1: int = 3,
        delta2: int = 6,
        gem_gamma: float = 2.0,
        gem_alpha: float = 2.0
    ):
        super(ULTRON, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratio = mlp_ratio

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dims[0], patches_resolution[0], patches_resolution[1])
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()

        for i_stage in range(self.num_layers):
            # Stage type: CDConv for stages 0,1 and SCALA for stages 2,3
            stage_type = "cdconv" if i_stage < 2 else "scala"

            # Downsample
            if i_stage > 0:
                # PatchMergingの適切な初期化
                def make_downsample(dim):
                    return PatchMerging(
                        input_resolution=(patches_resolution[0] // (2 ** i_stage),
                                        patches_resolution[1] // (2 ** i_stage)),
                        dim=dim
                    )
                downsample = make_downsample
            else:
                downsample = None

            stage = ULTRONStage(
                dim=embed_dims[i_stage],
                depth=depths[i_stage],
                stage_type=stage_type,
                window_size=window_size,
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                downsample=downsample,
                tau1=tau1,
                tau2=tau2,
                delta1=delta1,
                delta2=delta2
            )
            self.stages.append(stage)

        # Global pooling
        self.global_pool = AttentionBasedGlobalPooling(
            dim=self.num_features, gamma=gem_gamma, alpha=gem_alpha
        )

        # Classifier
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)

        # Add absolute position embedding
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # Forward through stages
        for stage in self.stages:
            x = stage(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.forward_features(x)

        # Global pooling
        x = self.global_pool(x)

        # Classification
        x = self.head(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """特徴抽出（分類層なし）"""
        x = self.forward_features(x)
        x = self.global_pool(x)
        return x

def create_ultron_s(num_classes: int = 1000, **kwargs) -> ULTRON:
    """ULTRON-S model"""
    return ULTRON(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 9, 5],
        num_heads=[2, 4, 10, 16],
        num_classes=num_classes,
        **kwargs
    )

def create_ultron_b(num_classes: int = 1000, **kwargs) -> ULTRON:
    """ULTRON-B model"""
    return ULTRON(
        embed_dims=[96, 192, 384, 768],
        depths=[5, 7, 18, 5],
        num_heads=[3, 6, 12, 24],
        num_classes=num_classes,
        **kwargs
    )

def test_ultron():
    """ULTRON のテスト"""
    print("=== ULTRON テスト ===")

    # テスト設定
    batch_size = 2
    num_classes = 1000
    img_size = 224

    # モデル作成
    model_s = create_ultron_s(num_classes=num_classes)
    model_b = create_ultron_b(num_classes=num_classes)

    # 入力データ
    x = torch.randn(batch_size, 3, img_size, img_size)

    print(f"入力形状: {x.shape}")

    # パラメータ数の計算
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"ULTRON-S パラメータ数: {count_parameters(model_s):,}")
    print(f"ULTRON-B パラメータ数: {count_parameters(model_b):,}")

    # テスト実行
    with torch.no_grad():
        # ULTRON-S テスト
        output_s = model_s(x)
        features_s = model_s.extract_features(x)
        print(f"ULTRON-S 出力形状: {output_s.shape}")
        print(f"ULTRON-S 特徴形状: {features_s.shape}")

        # ULTRON-B テスト
        output_b = model_b(x)
        features_b = model_b.extract_features(x)
        print(f"ULTRON-B 出力形状: {output_b.shape}")
        print(f"ULTRON-B 特徴形状: {features_b.shape}")

    # 勾配テスト
    x_grad = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True)
    output_grad = model_s(x_grad)
    loss = output_grad.sum()
    loss.backward()

    print(f"勾配計算成功: {x_grad.grad is not None}")
    print(f"勾配ノルム: {x_grad.grad.norm().item():.4f}")

    print("✓ ULTRON テスト完了")

if __name__ == "__main__":
    test_ultron()

