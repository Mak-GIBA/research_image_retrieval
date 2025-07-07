# SPECTRUM アーキテクチャ詳細設計

## 1. 全体アーキテクチャ

SPECTRUMは、5つの主要モジュール（CASTLE, PRISM, NEXUS, ORACLE, HARMONY）を統合した次世代バッチ内注意機構です。全体アーキテクチャは以下の構成となります：

```
SPECTRUM
├── バックボーンネットワーク（ResNet18/MobileNetV3など）
├── 階層的特徴抽出器
├── 5つの主要モジュール
│   ├── CASTLE（因果選択的トランスフォーマー学習強化機構）
│   ├── PRISM（プロンプト応答型インタラクティブセマンティックマッピング）
│   ├── NEXUS（神経交差ウィンドウスパース注意機構）
│   ├── ORACLE（オブジェクト関係認識適応型コンテキスト学習強化）
│   └── HARMONY（階層的適応型マルチモーダル調和最適化ネットワーク）
├── 適応型モジュール選択機構
├── 特徴統合器
└── 出力ヘッド（分類/検索）
```

## 2. 各モジュールの詳細設計

### 2.1 CASTLE（因果選択的トランスフォーマー学習強化機構）

#### クラス構造
```python
class CASTLE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        causal_threshold: float = 0.5,
        num_hierarchical_levels: int = 3,
        counterfactual_strength: float = 0.5,
        temperature: float = 0.07
    ):
        # 初期化コード
        
    def compute_causal_mask(self, features: torch.Tensor) -> torch.Tensor:
        # 因果マスクの計算
        
    def build_hierarchical_causal_graph(self, features: torch.Tensor, causal_scores: torch.Tensor) -> List[torch.Tensor]:
        # 階層的因果グラフの構築
        
    def compute_counterfactual_attention(self, features: torch.Tensor, perturbation_strength: float = 0.1) -> torch.Tensor:
        # 反実仮想注意の計算
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 順伝播処理
```

#### データフロー
1. 入力特徴から因果マスクを計算
2. 階層的因果グラフを構築
3. 反実仮想注意スコアを計算
4. 標準的な注意スコアと反実仮想注意スコアを統合
5. 注意機構を適用して出力特徴を生成

### 2.2 PRISM（プロンプト応答型インタラクティブセマンティックマッピング）

#### クラス構造
```python
class PRISM(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        output_dim: int,
        num_heads: int = 8,
        semantic_alignment_weight: float = 0.5,
        kl_divergence_weight: float = 0.1
    ):
        # 初期化コード
        
    def generate_semantic_prompt(self, features: torch.Tensor) -> torch.Tensor:
        # セマンティックプロンプトの生成（実際のLLMの代わりにシミュレーション）
        
    def cross_modal_attention(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # クロスモーダル注意の計算
        
    def interactive_semantic_mapping(self, features: torch.Tensor, semantic_similarity: torch.Tensor) -> torch.Tensor:
        # インタラクティブセマンティックマッピング
        
    def forward(self, visual_features: torch.Tensor, text_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 順伝播処理
```

#### データフロー
1. 視覚特徴からセマンティックプロンプトを生成（またはテキスト特徴を入力として受け取る）
2. 視覚特徴とテキスト特徴間のクロスモーダル注意を計算
3. 意味的類似度に基づいてインタラクティブセマンティックマッピングを実行
4. 視覚特徴と意味特徴を統合して出力特徴を生成

### 2.3 NEXUS（神経交差ウィンドウスパース注意機構）

#### クラス構造
```python
class NEXUS(nn.Module):
    def __init__(
        self,
        dim: int,
        min_window_size: int = 2,
        max_window_size: int = 8,
        num_scales: int = 3,
        sparsity_threshold: float = 0.5
    ):
        # 初期化コード
        
    def adaptive_cross_window(self, x: torch.Tensor) -> torch.Tensor:
        # 適応型クロスウィンドウ構造の適用
        
    def neural_sparse_attention_mask(self, attention_map: torch.Tensor) -> torch.Tensor:
        # 神経スパース注意マスクの生成
        
    def hierarchical_cross_window_fusion(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        # 階層的クロスウィンドウ融合
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 順伝播処理
```

#### データフロー
1. 入力特徴に適応型クロスウィンドウ構造を適用
2. 注意マップに神経スパース注意マスクを適用
3. 異なるスケールのクロスウィンドウ表現を階層的に融合
4. スパース化された注意マップを用いて出力特徴を生成

### 2.4 ORACLE（オブジェクト関係認識適応型コンテキスト学習強化）

#### クラス構造
```python
class ORACLE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_objects: int = 8,
        relation_dim: int = 256,
        context_balance: float = 0.5
    ):
        # 初期化コード
        
    def extract_object_features(self, x: torch.Tensor) -> torch.Tensor:
        # オブジェクト特徴の抽出（実際のオブジェクト検出器の代わりにシミュレーション）
        
    def dual_branch_processing(self, object_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # デュアルブランチ処理（オブジェクト特徴と関係特徴）
        
    def adaptive_context_aggregation(self, object_features: torch.Tensor, relation_scores: torch.Tensor) -> torch.Tensor:
        # 適応型コンテキスト集約
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 順伝播処理
```

#### データフロー
1. 入力特徴からオブジェクト特徴を抽出
2. オブジェクト特徴ブランチと関係特徴ブランチで並列処理
3. ブランチ間で相互情報を伝播
4. 関係強度に基づいて適応型コンテキスト集約を実行
5. 局所的・大域的コンテキストのバランスを取りながら出力特徴を生成

### 2.5 HARMONY（階層的適応型マルチモーダル調和最適化ネットワーク）

#### クラス構造
```python
class HARMONY(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        output_dim: int,
        num_levels: int = 3,
        kd_weight: float = 0.5,
        harmony_weight: float = 0.3,
        reg_weight: float = 0.2
    ):
        # 初期化コード
        
    def multimodal_knowledge_distillation(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        # マルチモーダル知識蒸留
        
    def hierarchical_modality_fusion(self, visual_features_list: List[torch.Tensor], text_features_list: List[torch.Tensor]) -> torch.Tensor:
        # 階層的モダリティ融合
        
    def harmonic_optimization(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 調和的最適化
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor, teacher_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 順伝播処理
```

#### データフロー
1. 視覚特徴とテキスト特徴を入力として受け取る
2. 教師モデル（M-LLM）の特徴が提供されている場合、知識蒸留を実行
3. 異なる抽象度レベルでモダリティ間の情報交換を行う
4. モダリティ間の調和を最適化する損失関数を計算
5. 調和的に融合された出力特徴を生成

## 3. 統合アーキテクチャ（SPECTRUM）

### クラス構造
```python
class SPECTRUM(nn.Module):
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        dim: int = 512,
        num_classes: int = 1000,
        use_castle: bool = True,
        use_prism: bool = True,
        use_nexus: bool = True,
        use_oracle: bool = True,
        use_harmony: bool = True
    ):
        # 初期化コード
        
    def extract_hierarchical_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 階層的特徴の抽出
        
    def adaptive_module_selection(self, features: torch.Tensor, task_embedding: Optional[torch.Tensor] = None) -> Dict[str, float]:
        # 適応型モジュール選択
        
    def integrate_features(self, module_outputs: Dict[str, torch.Tensor], module_weights: Dict[str, float]) -> torch.Tensor:
        # 特徴統合
        
    def forward(
        self, 
        x: torch.Tensor, 
        text_input: Optional[torch.Tensor] = None,
        task_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 順伝播処理
        
    def get_embedding(self, x: torch.Tensor, text_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 画像検索用の埋め込みベクトルを取得
```

### データフロー
1. 入力画像からバックボーンネットワークを用いて特徴を抽出
2. 階層的特徴抽出器を用いて異なるスケールの特徴マップを生成
3. 入力データと検索タスクに応じて適応型モジュール選択を実行
4. 選択されたモジュールを並列に実行
   - CASTLE: 因果選択的注意機構を適用
   - PRISM: セマンティックマッピングを実行
   - NEXUS: クロスウィンドウスパース注意を適用
   - ORACLE: オブジェクト関係認識を実行
   - HARMONY: マルチモーダル調和最適化を実行
5. 各モジュールの出力を重み付けして統合
6. 最終的な特徴表現を生成し、タスクに応じた出力ヘッドに渡す

## 4. 損失関数と学習戦略

### 損失関数
```python
class SPECTRUMLoss(nn.Module):
    def __init__(
        self,
        task_weight: float = 1.0,
        kd_weight: float = 0.5,
        fkd_weight: float = 0.3,
        harmony_weight: float = 0.3,
        reg_weight: float = 0.2
    ):
        # 初期化コード
        
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # 損失計算
```

### 学習戦略
1. **段階的事前学習**
   - 各モジュールを個別に事前学習
   - モジュール間の結合事前学習
   - 全体アーキテクチャの事前学習

2. **タスク固有の微調整**
   - 画像検索タスクに特化した微調整
   - タスク固有の損失関数を用いた最適化

3. **マルチタスク学習**
   - 複数のタスク（分類、検索、マッチングなど）を同時に学習
   - タスク間の知識共有を促進する正則化

## 5. ユーティリティ関数

```python
# バッチ内の画像間の因果関係を可視化
def visualize_causal_relations(causal_scores: torch.Tensor, save_path: str = None) -> None:
    # 可視化コード

# モジュール選択の重みを可視化
def visualize_module_weights(module_weights: Dict[str, float], save_path: str = None) -> None:
    # 可視化コード

# 埋め込みベクトルの類似度を計算
def compute_similarity(query_embedding: torch.Tensor, gallery_embeddings: torch.Tensor) -> torch.Tensor:
    # 類似度計算コード

# 検索精度の評価
def evaluate_retrieval(query_embeddings: torch.Tensor, gallery_embeddings: torch.Tensor, query_labels: torch.Tensor, gallery_labels: torch.Tensor) -> Dict[str, float]:
    # 評価コード
```
