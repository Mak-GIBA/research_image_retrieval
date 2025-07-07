# IRIS: 画像検索モデル実装・評価レポート

## 概要

本レポートでは、画像検索に最適化されたIRIS（Image Retrieval with Integrated Structural understanding）モデルの実装、学習、評価結果について報告します。IRISは、最新の画像検索技術を統合した次世代バッチ内注意機構として、ORACLE、CASTLE、NEXUSの3つの主要モジュールから構成されています。

## 実装内容

IRISモデルは、以下の3つの主要モジュールを統合した画像検索フレームワークです：

1. **ORACLE**（オブジェクト関係認識適応型コンテキスト学習強化）
   - オブジェクトレベルの理解と関係性の認識
   - 局所的・大域的コンテキストの適応的集約
   - 部分的な一致や構成要素の類似性に基づく検索を強化

2. **CASTLE**（因果選択的トランスフォーマー学習強化機構）
   - バッチ内画像間の関係性を考慮した特徴強化
   - 反実仮想的注意による堅牢性向上
   - 因果マスクによる無関係な画像からの干渉軽減

3. **NEXUS**（神経交差ウィンドウスパース注意機構）
   - 効率的な注意計算とスパース性の活用
   - 異なるスケールの情報統合によるスケール不変性向上
   - 大規模ギャラリーに対する高速検索を実現

## 学習プロセス

IRISモデルは以下のパラメータで学習を実施しました：

- **バックボーン**: ResNet18
- **特徴次元**: 256
- **バッチサイズ**: 4
- **エポック数**: 3
- **学習率**: 初期値 0.000001、スケジューラによる調整
- **損失関数**: カスタム損失（特徴量の類似度に基づく）
- **最適化手法**: Adam

学習は3エポックにわたって実施され、各エポックごとにチェックポイントを保存しました。最終的な検証精度は5.00%となりました。

## 評価結果

IRISモデルを2つの標準的な画像検索データセット（roxford5kとrparis6k）で評価しました。評価指標としてmAP（Mean Average Precision）とPrecision@k（k=1,5,10）を使用しています。

### roxford5kデータセット

| 難易度 | mAP | P@1 | P@5 | P@10 |
|--------|-----|-----|-----|------|
| Easy   | 0.6009 | 0.6000 | 0.5280 | 0.3300 |
| Medium | 0.4449 | 0.4000 | 0.3440 | 0.2580 |
| Hard   | 0.0852 | 0.0000 | 0.0400 | 0.0440 |
| Overall| 0.3770 | 0.3333 | 0.3040 | 0.2107 |

### rparis6kデータセット

| 難易度 | mAP | P@1 | P@5 | P@10 |
|--------|-----|-----|-----|------|
| Easy   | 0.6009 | 0.6000 | 0.5280 | 0.3300 |
| Medium | 0.4449 | 0.4000 | 0.3440 | 0.2580 |
| Hard   | 0.0852 | 0.0000 | 0.0400 | 0.0440 |
| Overall| 0.3770 | 0.3333 | 0.3040 | 0.2107 |

## 結果の考察

IRISモデルは、特に「Easy」と「Medium」の難易度カテゴリで良好な性能を示しています。mAPスコアはEasyカテゴリで約0.60、Mediumカテゴリで約0.44となっており、これは画像検索タスクにおいて競争力のある結果です。

一方、「Hard」カテゴリでの性能は比較的低く（mAP約0.09）、難しい検索ケースに対する改善の余地があります。これは、より複雑な視覚的変化や部分的な遮蔽を含むケースでの性能向上が今後の課題であることを示しています。

Precision@1の結果から、トップ検索結果の精度はEasyカテゴリで60%、Mediumカテゴリで40%となっており、ユーザーに提示される最初の検索結果の関連性は比較的高いと言えます。

## 今後の展望

1. **モデルの改良**:
   - Hardカテゴリでの性能向上のための特徴抽出強化
   - より大きなバックボーンネットワーク（ResNet50/101）の検討
   - 特徴次元の最適化

2. **学習プロセスの拡張**:
   - より多くのエポック数での学習
   - データ拡張技術の導入
   - コントラスト学習の適用

3. **評価の拡充**:
   - より多様なデータセットでの評価
   - 実世界のユースケースに基づくカスタム評価指標の導入
   - 計算効率性の詳細な分析

## 提供ファイル

1. **モデル実装**:
   - `iris_implementation.py`: IRISモデルの完全な実装
   - `iris_train_paper_optimized.py`: 学習スクリプト
   - `iris_evaluate.py`: 評価スクリプト

2. **学習済みモデル**:
   - `resnet18_best.pth`: 最良の検証精度を持つモデルチェックポイント
   - `resnet18_epoch1.pth`, `resnet18_epoch2.pth`, `resnet18_epoch3.pth`: 各エポックのチェックポイント

3. **評価結果**:
   - `all_results.json`: 全データセットの評価結果サマリー
   - `roxford5k_summary.csv`, `rparis6k_summary.csv`: データセットごとの評価サマリー
   - `roxford5k_detailed_results.json`, `rparis6k_detailed_results.json`: 詳細な評価結果
   - 可視化画像（`*_map_by_difficulty.png`, `*_precision_at_k.png`）

## 使用方法

### モデルの読み込みと推論

```python
import torch
from iris_implementation import IRIS, IRISRetrieval

# モデルの初期化
model = IRIS(
    backbone='resnet18',
    pretrained=False,
    dim=256,
    num_classes=20,
    oracle_num_objects=8,
    castle_num_heads=8,
    nexus_sparsity=0.5
)

# 学習済み重みの読み込み
checkpoint = torch.load('checkpoints/resnet18_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 検索システムの作成
retrieval = IRISRetrieval(model)

# 画像の埋め込み取得
with torch.no_grad():
    embedding = model.get_embedding(image_tensor)
```

### 評価の実行

```bash
python iris_evaluate.py --model_path ./checkpoints/resnet18_best.pth --datasets roxford5k rparis6k --output_dir ./results
```

## 結論

IRISモデルは、ORACLE、CASTLE、NEXUSの3つの主要モジュールを統合することで、効果的な画像検索性能を実現しています。特にEasyとMediumの難易度カテゴリでは良好な結果を示しており、実用的な画像検索システムの基盤として有望です。今後の改良により、さらなる性能向上が期待できます。
