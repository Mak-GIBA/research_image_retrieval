import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Dict, Optional, Union

class HierarchicalFeatureExtractor(nn.Module):
    """階層的特徴抽出バックボーン (HFE)
    
    異なる階層から特徴マップを抽出し、様々な抽象度の情報を捉える
    """
    def __init__(self, backbone_name: str = 'resnet18', pretrained: bool = True):
        super(HierarchicalFeatureExtractor, self).__init__()
        
        # バックボーンネットワークの選択
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            self.feature_dims = [64, 128, 256, 512]  # 各階層の特徴次元
            self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
            self.layer1 = backbone.layer1  # 出力サイズ: 64 x H/4 x W/4
            self.layer2 = backbone.layer2  # 出力サイズ: 128 x H/8 x W/8
            self.layer3 = backbone.layer3  # 出力サイズ: 256 x H/16 x W/16
            self.layer4 = backbone.layer4  # 出力サイズ: 512 x H/32 x W/32
        elif backbone_name == 'mobilenetv3':
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            self.feature_dims = [16, 24, 48, 96]
            # MobileNetV3の階層を適切に分割
            self.layer0 = backbone.features[0:2]
            self.layer1 = backbone.features[2:4]
            self.layer2 = backbone.features[4:7]
            self.layer3 = backbone.features[7:13]
            self.layer4 = backbone.features[13:]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: 入力画像テンソル (B x 3 x H x W)
            
        Returns:
            features: 異なる階層からの特徴マップのリスト
        """
        features = []
        
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        features.append(x1)
        
        x2 = self.layer2(x1)
        features.append(x2)
        
        x3 = self.layer3(x2)
        features.append(x3)
        
        x4 = self.layer4(x3)
        features.append(x4)
        
        return features


class AdaptiveCrossScaleAttention(nn.Module):
    """適応型クロススケール注意機構 (ACSA)
    
    異なる階層間の特徴マップ間で注意スコアを計算し、情報を統合
    """
    def __init__(self, feature_dims: List[int], hidden_dim: int = 256, num_heads: int = 8):
        super(AdaptiveCrossScaleAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 各階層の特徴を同じ次元に投影するための変換層
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim, kernel_size=1)
            for dim in feature_dims
        ])
        
        # クエリ、キー、バリュー投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 出力投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 適応型窓サイズ決定ネットワーク
        self.window_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 階層間情報流制御ゲート
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: 異なる階層からの特徴マップのリスト
            
        Returns:
            enhanced_features: 注意機構で強化された特徴マップのリスト
        """
        batch_size = features[0].shape[0]
        enhanced_features = []
        
        # 各特徴マップを同じ次元に投影
        projected_features = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        for i, feat_i in enumerate(projected_features):
            # 特徴マップをフラット化して注意計算の準備
            h_i, w_i = feat_i.shape[2:]
            feat_i_flat = feat_i.flatten(2).permute(0, 2, 1)  # B x (H*W) x C
            
            # 適応型窓サイズの決定
            window_size = self.window_predictor(feat_i)  # B x 1
            window_size = torch.clamp(window_size * 10, min=1, max=5)  # スケーリングして1〜5の範囲に
            
            # クエリ投影
            q = self.q_proj(feat_i_flat)  # B x (H*W) x C
            q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B x num_heads x (H*W) x head_dim
            
            enhanced_feat = feat_i_flat
            
            # 他の階層との注意計算
            for j, feat_j in enumerate(projected_features):
                if i == j:
                    continue
                
                # 特徴マップをフラット化
                h_j, w_j = feat_j.shape[2:]
                feat_j_flat = feat_j.flatten(2).permute(0, 2, 1)  # B x (H*W) x C
                
                # キーとバリュー投影
                k = self.k_proj(feat_j_flat)  # B x (H*W) x C
                v = self.v_proj(feat_j_flat)  # B x (H*W) x C
                
                k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B x num_heads x (H*W) x head_dim
                v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B x num_heads x (H*W) x head_dim
                
                # 注意スコアの計算
                attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # B x num_heads x (H_i*W_i) x (H_j*W_j)
                
                # 適応型窓サイズに基づくマスキング（簡略化のため省略）
                
                # 注意重みの計算
                attn_weights = F.softmax(attn_scores, dim=-1)  # B x num_heads x (H_i*W_i) x (H_j*W_j)
                
                # 注意適用
                context = torch.matmul(attn_weights, v)  # B x num_heads x (H_i*W_i) x head_dim
                context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, h_i * w_i, self.hidden_dim)  # B x (H_i*W_i) x C
                
                # 出力投影
                context = self.out_proj(context)  # B x (H_i*W_i) x C
                
                # ゲート機構による情報流制御
                gate_input = torch.cat([enhanced_feat, context], dim=-1)  # B x (H_i*W_i) x (2*C)
                gate_value = self.gate(gate_input)  # B x (H_i*W_i) x C
                
                # ゲート適用
                enhanced_feat = enhanced_feat + gate_value * context
            
            # 元の形状に戻す
            enhanced_feat = enhanced_feat.permute(0, 2, 1).view(batch_size, self.hidden_dim, h_i, w_i)
            enhanced_features.append(enhanced_feat)
        
        return enhanced_features


class ContextEnhancedConvModule(nn.Module):
    """コンテキスト強化型畳み込みモジュール (CECM)
    
    コンテキスト情報に基づいて動的に畳み込みパラメータを調整
    """
    def __init__(self, in_channels: int, hidden_dim: int = 256, groups: int = 8):
        super(ContextEnhancedConvModule, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.groups = groups
        self.channels_per_group = hidden_dim // groups
        
        # コンテキストプーリング
        self.context_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        
        # 動的カーネル生成ネットワーク
        self.kernel_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, groups * 9)  # 各グループに3x3カーネルを生成
        )
        
        # グループ畳み込み用の入力変換
        self.input_transform = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 出力変換
        self.output_transform = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        # 正規化層
        self.norm = nn.BatchNorm2d(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力特徴マップ (B x C x H x W)
            
        Returns:
            out: コンテキスト強化された特徴マップ
        """
        batch_size, _, height, width = x.shape
        
        # コンテキスト情報の抽出
        context = self.context_pool(x)  # B x hidden_dim x 1 x 1
        context = context.squeeze(-1).squeeze(-1)  # B x hidden_dim
        
        # 動的カーネルの生成
        kernels = self.kernel_generator(context)  # B x (groups * 9)
        kernels = kernels.view(batch_size, self.groups, 3, 3)  # B x groups x 3 x 3
        
        # 入力変換
        x = self.input_transform(x)  # B x hidden_dim x H x W
        
        # グループごとに分割
        x_groups = x.view(batch_size, self.groups, self.channels_per_group, height, width)  # B x groups x channels_per_group x H x W
        
        # 各グループに動的カーネルを適用（簡略化のため、単純な実装）
        out_groups = []
        for g in range(self.groups):
            # カーネルの取得
            kernel = kernels[:, g, :, :].unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 3 x 3
            
            # グループの取得
            x_g = x_groups[:, g]  # B x channels_per_group x H x W
            
            # 各バッチ要素に対して畳み込みを適用（簡略化）
            # 実際の実装ではより効率的な方法が必要
            out_g = F.conv2d(
                x_g.view(1, -1, height, width),
                kernel.view(-1, 1, 3, 3),
                padding=1,
                groups=batch_size * self.channels_per_group
            )
            out_g = out_g.view(batch_size, self.channels_per_group, height, width)
            out_groups.append(out_g)
        
        # グループの結合
        out = torch.cat(out_groups, dim=1)  # B x hidden_dim x H x W
        
        # 出力変換と正規化
        out = self.output_transform(out)
        out = self.norm(out)
        
        # 残差接続
        return x + out


class MultimodalKnowledgeDistillation(nn.Module):
    """マルチモーダル知識蒸留モジュール (MKDM)
    
    大規模マルチモーダルモデルからの知識を蒸留し、セマンティックな理解を強化
    """
    def __init__(self, feature_dim: int, text_dim: int = 512, hidden_dim: int = 256):
        super(MultimodalKnowledgeDistillation, self).__init__()
        
        self.feature_dim = feature_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # 特徴変換
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # テキスト変換
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 共同埋め込み空間への投影
        self.joint_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 温度パラメータ（蒸留用）
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, features: torch.Tensor, text_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: 画像特徴 (B x feature_dim)
            text_features: テキスト特徴 (B x text_dim)、訓練時のみ使用
            
        Returns:
            enhanced_features: 知識蒸留で強化された特徴
            distillation_loss: 蒸留損失（訓練時のみ）
        """
        # 特徴変換
        transformed_features = self.feature_transform(features)  # B x hidden_dim
        
        # 共同埋め込み空間への投影
        enhanced_features = self.joint_projection(transformed_features)  # B x hidden_dim
        
        # 訓練時のみ蒸留損失を計算
        distillation_loss = None
        if text_features is not None:
            # テキスト特徴の変換
            transformed_text = self.text_transform(text_features)  # B x hidden_dim
            
            # コサイン類似度の計算
            sim_matrix = torch.matmul(
                F.normalize(enhanced_features, dim=1),
                F.normalize(transformed_text, dim=1).t()
            ) / self.temperature
            
            # 対角要素が正例、それ以外が負例
            labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
            
            # 対称クロスエントロピー損失
            loss_i2t = F.cross_entropy(sim_matrix, labels)
            loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
            distillation_loss = (loss_i2t + loss_t2i) / 2
        
        return enhanced_features, distillation_loss


class HierarchicalFeatureFusionNetwork(nn.Module):
    """階層的特徴融合ネットワーク (HFFN)
    
    異なる階層からの特徴を効果的に融合
    """
    def __init__(self, feature_dims: List[int], hidden_dim: int = 256):
        super(HierarchicalFeatureFusionNetwork, self).__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # 各階層の特徴変換
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ) for dim in feature_dims
        ])
        
        # 重要度予測ネットワーク
        self.importance_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * len(feature_dims), 256),
            nn.ReLU(),
            nn.Linear(256, len(feature_dims)),
            nn.Softmax(dim=1)
        )
        
        # クロスアテンション
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 異なる階層からの特徴マップのリスト
            
        Returns:
            fused_features: 融合された特徴マップ
        """
        batch_size = features[0].shape[0]
        
        # 各階層の特徴変換
        transformed_features = [transform(feat) for transform, feat in zip(self.transforms, features)]
        
        # 重要度の計算
        pooled_features = [F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in transformed_features]
        pooled_concat = torch.cat(pooled_features, dim=1)  # B x (hidden_dim * num_layers)
        importance_weights = self.importance_predictor(pooled_concat)  # B x num_layers
        
        # 特徴のリサイズと重み付け
        resized_features = []
        target_size = transformed_features[-1].shape[2:]  # 最も深い層のサイズに合わせる
        
        for i, feat in enumerate(transformed_features):
            # 必要に応じてリサイズ
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            # 重み付け
            weight = importance_weights[:, i].view(batch_size, 1, 1, 1)
            weighted_feat = feat * weight
            resized_features.append(weighted_feat)
        
        # 初期融合（加重和）
        initial_fusion = sum(resized_features)  # B x hidden_dim x H x W
        
        # クロスアテンションによる強化
        h, w = initial_fusion.shape[2:]
        flat_fusion = initial_fusion.flatten(2).permute(0, 2, 1)  # B x (H*W) x hidden_dim
        
        # 自己注意適用
        enhanced_fusion, _ = self.cross_attention(flat_fusion, flat_fusion, flat_fusion)
        
        # 元の形状に戻す
        fused_features = enhanced_fusion.permute(0, 2, 1).view(batch_size, self.hidden_dim, h, w)
        
        return fused_features


class GlobalRepresentationGenerator(nn.Module):
    """グローバル表現生成モジュール (GRGM)
    
    階層的特徴から最終的なグローバル表現を生成
    """
    def __init__(self, feature_dim: int, output_dim: int = 512):
        super(GlobalRepresentationGenerator, self).__init__()
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # 注意ベースのプーリング
        self.attention_pool = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 次元削減と正規化
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力特徴マップ (B x C x H x W)
            
        Returns:
            global_repr: グローバル表現 (B x output_dim)
        """
        batch_size = x.shape[0]
        
        # 注意マップの計算
        attention_map = self.attention_pool(x)  # B x 1 x H x W
        
        # 注意ベースのプーリング
        weighted_features = x * attention_map  # B x C x H x W
        pooled_features = weighted_features.sum(dim=(2, 3)) / (attention_map.sum(dim=(2, 3)) + 1e-8)  # B x C
        
        # 次元削減と正規化
        global_repr = self.mlp(pooled_features)  # B x output_dim
        global_repr = F.normalize(global_repr, p=2, dim=1)  # L2正規化
        
        return global_repr


class HAMLET(nn.Module):
    """HAMLET: 階層的適応型マルチモーダル埋め込みTransformer
    
    Transformerと畳み込みの長所を融合し、マルチモーダル情報と階層的特徴表現を活用する画像検索アーキテクチャ
    """
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        hidden_dim: int = 256,
        output_dim: int = 512,
        text_dim: int = 512
    ):
        super(HAMLET, self).__init__()
        
        # 階層的特徴抽出バックボーン
        self.feature_extractor = HierarchicalFeatureExtractor(backbone_name, pretrained)
        feature_dims = self.feature_extractor.feature_dims
        
        # 適応型クロススケール注意機構
        self.cross_scale_attention = AdaptiveCrossScaleAttention(feature_dims, hidden_dim)
        
        # コンテキスト強化型畳み込みモジュール
        self.context_enhanced_convs = nn.ModuleList([
            ContextEnhancedConvModule(hidden_dim, hidden_dim)
            for _ in range(len(feature_dims))
        ])
        
        # 階層的特徴融合ネットワーク
        self.feature_fusion = HierarchicalFeatureFusionNetwork([hidden_dim] * len(feature_dims), hidden_dim)
        
        # グローバル表現生成モジュール
        self.global_repr_generator = GlobalRepresentationGenerator(hidden_dim, output_dim)
        
        # マルチモーダル知識蒸留モジュール
        self.knowledge_distillation = MultimodalKnowledgeDistillation(output_dim, text_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 入力画像 (B x 3 x H x W)
            text_features: テキスト特徴 (B x text_dim)、訓練時のみ使用
            
        Returns:
            global_repr: グローバル表現 (B x output_dim)
            distillation_loss: 蒸留損失（訓練時のみ）
        """
        # 階層的特徴抽出
        hierarchical_features = self.feature_extractor(x)
        
        # 適応型クロススケール注意機構
        enhanced_features = self.cross_scale_attention(hierarchical_features)
        
        # コンテキスト強化型畳み込みモジュール
        context_enhanced_features = [
            conv(feat) for conv, feat in zip(self.context_enhanced_convs, enhanced_features)
        ]
        
        # 階層的特徴融合
        fused_features = self.feature_fusion(context_enhanced_features)
        
        # グローバル表現生成
        global_repr = self.global_repr_generator(fused_features)
        
        # マルチモーダル知識蒸留（訓練時のみ）
        enhanced_global_repr, distillation_loss = self.knowledge_distillation(global_repr, text_features)
        
        return enhanced_global_repr, distillation_loss


# 損失関数の定義
class HAMLETLoss(nn.Module):
    """HAMLETの学習に使用する複合損失関数"""
    def __init__(self, lambda_cls: float = 1.0, lambda_mm: float = 0.5, lambda_hier: float = 0.3):
        super(HAMLETLoss, self).__init__()
        
        self.lambda_cls = lambda_cls  # 分類損失の重み
        self.lambda_mm = lambda_mm  # マルチモーダル一貫性損失の重み
        self.lambda_hier = lambda_hier  # 特徴階層性保存損失の重み
        
        # 分類損失
        self.cls_loss = nn.CrossEntropyLoss()
        
        # 特徴階層性保存損失
        self.hier_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        distillation_loss: Optional[torch.Tensor] = None,
        hierarchical_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            outputs: モデル出力 (B x num_classes)
            targets: 目標ラベル (B)
            distillation_loss: 蒸留損失
            hierarchical_features: 階層的特徴のリスト（階層性保存損失用）
            
        Returns:
            total_loss: 合計損失
        """
        # 分類損失
        cls_loss = self.cls_loss(outputs, targets)
        
        # 合計損失の初期化
        total_loss = self.lambda_cls * cls_loss
        
        # マルチモーダル一貫性損失（利用可能な場合）
        if distillation_loss is not None:
            total_loss += self.lambda_mm * distillation_loss
        
        # 特徴階層性保存損失（利用可能な場合）
        if hierarchical_features is not None and len(hierarchical_features) > 1:
            hier_loss = 0
            for i in range(len(hierarchical_features) - 1):
                # 隣接する階層間の一貫性を促進
                feat_i = F.adaptive_avg_pool2d(hierarchical_features[i], 1).squeeze()
                feat_i_plus_1 = F.adaptive_avg_pool2d(hierarchical_features[i + 1], 1).squeeze()
                
                # 特徴の正規化
                feat_i = F.normalize(feat_i, p=2, dim=1)
                feat_i_plus_1 = F.normalize(feat_i_plus_1, p=2, dim=1)
                
                # MSE損失
                hier_loss += self.hier_loss(feat_i, feat_i_plus_1)
            
            total_loss += self.lambda_hier * hier_loss
        
        return total_loss
