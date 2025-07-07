import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer
import math
from typing import Dict, List, Optional, Tuple, Union

class LLMKnowledgeDistillation(nn.Module):
    """
    LLM知識蒸留モジュール (LKD)
    大規模マルチモーダルLLMの知識を軽量モデルに蒸留し、豊かな表現能力を獲得する
    """
    def __init__(
        self,
        visual_dim: int,
        text_dim: int = 768,
        embed_dim: int = 256,
        temperature: float = 2.0,
        alpha: float = 0.5,
        text_model_name: str = 'distilbert-base-uncased',
        use_text_encoder: bool = True
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.alpha = alpha
        self.use_text_encoder = use_text_encoder
        
        # 視覚特徴の投影
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # テキストエンコーダー
        self.text_encoder = None
        self.text_tokenizer = None
        
        if use_text_encoder:
            try:
                self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
                self.text_tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
                self.text_dim = self.text_encoder.config.hidden_size  # 通常は768
            except Exception as e:
                print(f"テキストモデルの読み込みに失敗しました: {e}")
                print("テキスト機能を無効化します")
                self.use_text_encoder = False
        
        # テキスト特徴の投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 蒸留ヘッド
        self.distill_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """テキストをエンコードして特徴ベクトルを取得"""
        if not self.use_text_encoder or self.text_encoder is None:
            return None
        
        # テキストのトークン化
        if isinstance(text, str):
            text = [text]
        
        inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 入力をモデルのデバイスに移動
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # テキスト特徴の抽出
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            text_features = outputs.last_hidden_state
        
        return text_features
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, torch.Tensor]:
        # 視覚特徴が4次元の場合（[B, C, H, W]）、2次元に変換
        if len(visual_features.shape) == 4:
            visual_features = F.adaptive_avg_pool2d(visual_features, (1, 1)).flatten(1)
        
        # 視覚特徴の投影
        visual_embed = self.visual_proj(visual_features)
        
        # テキスト特徴の取得
        if text_features is None and text is not None and self.use_text_encoder:
            text_features = self.encode_text(text)
        
        # テキスト特徴がない場合は視覚特徴のみを返す
        if text_features is None:
            return {
                'visual_embed': visual_embed,
                'text_embed': None,
                'distill_loss': torch.tensor(0.0, device=visual_features.device)
            }
        
        # テキスト特徴の処理（平均プーリング）
        if len(text_features.shape) == 3:  # [B, L, D]
            text_features = text_features.mean(dim=1)  # [B, D]
        
        # テキスト特徴の投影
        text_embed = self.text_proj(text_features)
        
        # 蒸留ヘッド
        distill_embed = self.distill_head(visual_embed)
        
        # 特徴の正規化
        visual_embed_norm = F.normalize(visual_embed, p=2, dim=1)
        text_embed_norm = F.normalize(text_embed, p=2, dim=1)
        distill_embed_norm = F.normalize(distill_embed, p=2, dim=1)
        
        # 蒸留損失の計算（コサイン類似度ベース）
        sim_matrix = torch.matmul(distill_embed_norm, text_embed_norm.t()) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        distill_loss = F.cross_entropy(sim_matrix, labels)
        
        return {
            'visual_embed': visual_embed,
            'text_embed': text_embed,
            'distill_embed': distill_embed,
            'distill_loss': distill_loss
        }


class AdaptiveSparseEncoder(nn.Module):
    """
    適応型スパース符号化器 (ASE)
    入力に応じて動的にスパース度を調整し、重要な特徴次元を保持しながら冗長性を削減
    """
    def __init__(
        self,
        input_dim: int,
        min_sparsity: float = 0.05,
        max_sparsity: float = 0.3,
        init_sparsity: float = 0.1,
        learn_sparsity: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.init_sparsity = init_sparsity
        self.learn_sparsity = learn_sparsity
        
        # スパース度予測ネットワーク
        if learn_sparsity:
            self.sparsity_predictor = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, 1),
                nn.Sigmoid()
            )
        else:
            self.register_buffer('sparsity', torch.tensor(init_sparsity))
    
    def get_sparsity(self, x: torch.Tensor) -> torch.Tensor:
        """入力に応じたスパース度を取得"""
        if self.learn_sparsity:
            # バッチ内の各サンプルに対してスパース度を予測
            sparsity = self.sparsity_predictor(x)
            # スパース度を指定範囲に制限
            sparsity = self.min_sparsity + (self.max_sparsity - self.min_sparsity) * sparsity
            return sparsity
        else:
            return self.sparsity.expand(x.size(0), 1)
    
    def get_threshold(self, x: torch.Tensor, sparsity: torch.Tensor) -> torch.Tensor:
        """スパース度に基づく閾値を計算"""
        # 各サンプルの絶対値の大きさに基づいてパーセンタイル閾値を計算
        abs_x = torch.abs(x)
        thresholds = []
        
        for i in range(x.size(0)):
            # スパース度に基づいてパーセンタイルを計算
            percentile = 100.0 * (1.0 - sparsity[i].item())
            # パーセンタイル閾値を計算
            k = max(1, int(round(x.size(1) * (1.0 - sparsity[i].item()))))
            threshold = torch.kthvalue(abs_x[i], k).values
            thresholds.append(threshold)
        
        return torch.stack(thresholds).unsqueeze(1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # スパース度の予測
        sparsity = self.get_sparsity(x)
        
        # 閾値の計算
        threshold = self.get_threshold(x, sparsity)
        
        # スパース化
        mask = (torch.abs(x) > threshold).float()
        sparse_x = x * mask
        
        # スパース度の計算（実際の非ゼロ要素の割合）
        actual_sparsity = mask.sum(dim=1) / mask.size(1)
        
        # L1正則化項（スパース性を促進）
        l1_reg = torch.abs(sparse_x).sum(dim=1).mean()
        
        return {
            'sparse_features': sparse_x,
            'mask': mask,
            'sparsity': sparsity,
            'actual_sparsity': actual_sparsity,
            'l1_reg': l1_reg
        }


class SemanticPreservingQuantization(nn.Module):
    """
    セマンティック保存量子化 (SPQ)
    セマンティック情報の損失を最小化しながら、特徴表現を効率的に量子化
    """
    def __init__(
        self,
        input_dim: int,
        min_bits: int = 2,
        max_bits: int = 8,
        semantic_weight: float = 0.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.semantic_weight = semantic_weight
        
        # セマンティック重要度推定器
        self.importance_estimator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # 量子化スケーリング係数（学習可能）
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
    
    def get_bit_allocation(self, importance: torch.Tensor) -> torch.Tensor:
        """重要度に基づいてビット割り当てを決定"""
        # 重要度を[min_bits, max_bits]の範囲にスケーリング
        bits = self.min_bits + (self.max_bits - self.min_bits) * importance
        # 整数に丸める
        bits = torch.round(bits)
        return bits
    
    def quantize(self, x: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """混合精度量子化"""
        # スケーリング係数と零点を適用
        scale = torch.abs(self.scale) + 1e-6  # 正の値を保証
        zero_point = self.zero_point
        
        # 量子化範囲の計算
        qmin = torch.zeros_like(bits)
        qmax = 2.0 ** bits - 1.0
        
        # 量子化
        x_scaled = x / scale + zero_point
        x_clipped = torch.clamp(x_scaled, qmin, qmax)
        x_rounded = torch.round(x_clipped)
        
        # 逆量子化
        x_dequantized = (x_rounded - zero_point) * scale
        
        return x_dequantized
    
    def straight_through_estimator(self, x: torch.Tensor, x_dequantized: torch.Tensor) -> torch.Tensor:
        """Straight-Through Estimator（STE）による勾配の近似"""
        # 順伝播ではx_dequantizedを使用し、逆伝播ではxの勾配をそのまま使用
        return x + (x_dequantized - x).detach()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # セマンティック重要度の推定
        importance = self.importance_estimator(x)
        
        # ビット割り当ての決定
        bits = self.get_bit_allocation(importance)
        
        # 量子化
        x_quantized = self.quantize(x, bits)
        
        # Straight-Through Estimatorによる勾配の近似
        x_ste = self.straight_through_estimator(x, x_quantized)
        
        # 再構成損失
        recon_loss = F.mse_loss(x_quantized, x)
        
        # セマンティック保存損失（重要度の高い次元の誤差を重視）
        semantic_loss = (importance * torch.abs(x_quantized - x)).sum(dim=1).mean()
        
        # 総合損失
        total_loss = recon_loss + self.semantic_weight * semantic_loss
        
        # 平均ビット数
        avg_bits = bits.mean()
        
        return {
            'quantized_features': x_ste,
            'importance': importance,
            'bits': bits,
            'avg_bits': avg_bits,
            'recon_loss': recon_loss,
            'semantic_loss': semantic_loss,
            'quant_loss': total_loss
        }


class IterativeQueryExpansion(nn.Module):
    """
    反復クエリ拡張機構 (IQE)
    初期検索結果に基づいてクエリを反復的に拡張し、検索精度を向上
    """
    def __init__(
        self,
        embed_dim: int,
        top_k: int = 5,
        max_iterations: int = 2,
        gamma: float = 0.6
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.gamma = gamma
        
        # クエリ拡張生成器
        self.expansion_generator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def expand_query(
        self,
        query_features: torch.Tensor,
        database_features: torch.Tensor,
        iteration: int = 0
    ) -> Dict[str, torch.Tensor]:
        """クエリを拡張し、検索結果を改善"""
        # 反復回数が上限に達した場合は元のクエリを返す
        if iteration >= self.max_iterations:
            return {
                'expanded_query': query_features,
                'final_scores': None,
                'iteration': iteration
            }
        
        # 初期検索（コサイン類似度の計算）
        query_norm = F.normalize(query_features, p=2, dim=1)
        database_norm = F.normalize(database_features, p=2, dim=1)
        
        initial_scores = torch.matmul(query_norm, database_norm.t())
        
        # トップK結果の取得
        top_scores, top_indices = torch.topk(initial_scores, k=min(self.top_k, database_features.size(0)), dim=1)
        
        expanded_queries = []
        
        # バッチ内の各クエリに対して拡張を実行
        for i in range(query_features.size(0)):
            # トップK結果の特徴を取得
            top_features = database_features[top_indices[i]]
            
            # トップK結果の平均特徴
            avg_top_features = top_features.mean(dim=0, keepdim=True)
            
            # 元のクエリと平均特徴を結合
            combined = torch.cat([query_features[i:i+1], avg_top_features], dim=1)
            
            # 拡張クエリの生成
            expanded_query = self.expansion_generator(combined)
            expanded_queries.append(expanded_query)
        
        # 拡張クエリの統合
        expanded_queries = torch.cat(expanded_queries, dim=0)
        
        # 拡張クエリでの検索
        expanded_norm = F.normalize(expanded_queries, p=2, dim=1)
        expanded_scores = torch.matmul(expanded_norm, database_norm.t())
        
        # 初期スコアと拡張スコアの統合
        final_scores = self.gamma * initial_scores + (1 - self.gamma) * expanded_scores
        
        # 次の反復に進む（再帰的に呼び出し）
        next_iteration = self.expand_query(expanded_queries, database_features, iteration + 1)
        
        if iteration == 0:
            return {
                'expanded_query': next_iteration['expanded_query'],
                'initial_scores': initial_scores,
                'expanded_scores': expanded_scores,
                'final_scores': final_scores,
                'iteration': iteration
            }
        else:
            return {
                'expanded_query': expanded_queries,
                'final_scores': final_scores,
                'iteration': iteration
            }


class SPARSE(nn.Module):
    """
    Semantic-Preserving Adaptive Representation with Sparse Encoding (SPARSE) モデル
    計算効率と検索精度のバランスを最適化した表現学習フレームワーク
    """
    def __init__(
        self,
        backbone_type: str = 'resnet18',
        pretrained: bool = True,
        embed_dim: int = 256,
        output_dim: int = 128,
        use_lkd: bool = True,
        use_ase: bool = True,
        use_spq: bool = True,
        use_iqe: bool = True,
        text_model_name: str = 'distilbert-base-uncased',
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.use_lkd = use_lkd
        self.use_ase = use_ase
        self.use_spq = use_spq
        self.use_iqe = use_iqe
        
        # バックボーンネットワークの設定
        if backbone_type.startswith('resnet'):
            if backbone_type == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                self.backbone_dim = 512
            elif backbone_type == 'resnet34':
                self.backbone = models.resnet34(pretrained=pretrained)
                self.backbone_dim = 512
            elif backbone_type == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                self.backbone_dim = 2048
            elif backbone_type == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
                self.backbone_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet type: {backbone_type}")
            
            # 最後の全結合層を削除
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        elif backbone_type.startswith('mobilenet'):
            if backbone_type == 'mobilenet_v3_small':
                self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
                self.backbone_dim = 576
            elif backbone_type == 'mobilenet_v3_large':
                self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
                self.backbone_dim = 960
            else:
                raise ValueError(f"Unsupported MobileNet type: {backbone_type}")
            
            # 分類層を削除
            self.backbone = self.backbone.features
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # LLM知識蒸留モジュール (LKD)
        if use_lkd:
            self.lkd = LLMKnowledgeDistillation(
                visual_dim=self.backbone_dim,
                embed_dim=embed_dim,
                text_model_name=text_model_name
            )
        
        # 適応型スパース符号化器 (ASE)
        if use_ase:
            self.ase = AdaptiveSparseEncoder(
                input_dim=embed_dim,
                min_sparsity=0.05,
                max_sparsity=0.3,
                init_sparsity=0.1,
                learn_sparsity=True
            )
        
        # セマンティック保存量子化 (SPQ)
        if use_spq:
            self.spq = SemanticPreservingQuantization(
                input_dim=embed_dim,
                min_bits=2,
                max_bits=8,
                semantic_weight=0.5
            )
        
        # 反復クエリ拡張機構 (IQE)
        if use_iqe:
            self.iqe = IterativeQueryExpansion(
                embed_dim=output_dim,
                top_k=5,
                max_iterations=2,
                gamma=0.6
            )
        
        # 出力投影
        self.output_proj = nn.Linear(embed_dim, output_dim)
    
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        database_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 視覚特徴の抽出
        visual_features = self.backbone(images)
        
        # LLM知識蒸留モジュール (LKD)
        if self.use_lkd:
            lkd_out = self.lkd(visual_features, text=text)
            features = lkd_out['visual_embed']
            distill_loss = lkd_out['distill_loss']
        else:
            # グローバルプーリング
            features = F.adaptive_avg_pool2d(visual_features, (1, 1)).flatten(1)
            # 特徴次元の変換
            features = self.output_proj(features)
            distill_loss = torch.tensor(0.0, device=features.device)
        
        # 適応型スパース符号化器 (ASE)
        if self.use_ase:
            ase_out = self.ase(features)
            features = ase_out['sparse_features']
            sparsity_loss = ase_out['l1_reg']
        else:
            sparsity_loss = torch.tensor(0.0, device=features.device)
        
        # セマンティック保存量子化 (SPQ)
        if self.use_spq:
            spq_out = self.spq(features)
            features = spq_out['quantized_features']
            quant_loss = spq_out['quant_loss']
        else:
            quant_loss = torch.tensor(0.0, device=features.device)
        
        # 出力投影
        output_features = self.output_proj(features)
        
        # 出力特徴の正規化
        output_features = F.normalize(output_features, p=2, dim=1)
        
        # 反復クエリ拡張機構 (IQE)
        iqe_out = None
        if self.use_iqe and database_features is not None:
            iqe_out = self.iqe.expand_query(output_features, database_features)
            expanded_features = iqe_out['expanded_query']
            # 検索フェーズでのみ拡張クエリを使用
            if not self.training:
                output_features = expanded_features
        
        return {
            'output_features': output_features,
            'visual_features': visual_features,
            'sparse_features': features if self.use_ase else None,
            'distill_loss': distill_loss,
            'sparsity_loss': sparsity_loss,
            'quant_loss': quant_loss,
            'iqe_out': iqe_out
        }


class SPARSELoss(nn.Module):
    """
    SPARSEモデルの損失関数
    複数の損失を組み合わせて最適化
    """
    def __init__(
        self,
        distill_weight: float = 0.5,
        sparsity_weight: float = 0.1,
        quant_weight: float = 0.2,
        triplet_weight: float = 0.2,
        margin: float = 0.3
    ):
        super().__init__()
        self.distill_weight = distill_weight
        self.sparsity_weight = sparsity_weight
        self.quant_weight = quant_weight
        self.triplet_weight = triplet_weight
        self.margin = margin
        
        # トリプレット損失
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 各損失の取得
        distill_loss = outputs.get('distill_loss', torch.tensor(0.0, device=outputs['output_features'].device))
        sparsity_loss = outputs.get('sparsity_loss', torch.tensor(0.0, device=outputs['output_features'].device))
        quant_loss = outputs.get('quant_loss', torch.tensor(0.0, device=outputs['output_features'].device))
        
        # トリプレット損失（ラベルがある場合のみ）
        triplet_loss = torch.tensor(0.0, device=outputs['output_features'].device)
        
        if labels is not None:
            # トリプレット損失の計算（簡易実装）
            features = outputs['output_features']
            batch_size = features.size(0)
            
            if batch_size >= 3:  # トリプレットには少なくとも3サンプルが必要
                # 各サンプルに対して同じクラスと異なるクラスのサンプルを選択
                triplet_losses = []
                
                for i in range(batch_size):
                    anchor = features[i].unsqueeze(0)
                    anchor_label = labels[i]
                    
                    # 同じクラスのサンプル（正例）
                    positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                    positive_indices = positive_indices[positive_indices != i]  # 自分自身を除外
                    
                    # 異なるクラスのサンプル（負例）
                    negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
                    
                    if len(positive_indices) > 0 and len(negative_indices) > 0:
                        # ランダムに正例と負例を選択
                        positive_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
                        negative_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
                        
                        positive = features[positive_idx].unsqueeze(0)
                        negative = features[negative_idx].unsqueeze(0)
                        
                        loss = self.triplet_loss(anchor, positive, negative)
                        triplet_losses.append(loss)
                
                if triplet_losses:
                    triplet_loss = torch.stack(triplet_losses).mean()
        
        # 総合損失
        total_loss = (
            self.distill_weight * distill_loss +
            self.sparsity_weight * sparsity_loss +
            self.quant_weight * quant_loss +
            self.triplet_weight * triplet_loss
        )
        
        return {
            'total': total_loss,
            'distill': distill_loss,
            'sparsity': sparsity_loss,
            'quant': quant_loss,
            'triplet': triplet_loss
        }
