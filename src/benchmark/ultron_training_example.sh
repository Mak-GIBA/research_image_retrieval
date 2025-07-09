#!/bin/bash

# ULTRON学習用の設定例
# multi_model_train_modified.pyを使用してULTRONモデルを学習するためのスクリプト

echo "=== ULTRON Training Configuration Example ==="

# 基本設定
MODEL="ultron"
BACKBONE="resnet50"
BATCH_SIZE=32  # GPU環境に応じて調整
NUM_EPOCHS=40
EMBED_DIM=512
RHO=0.04

# 学習率設定 (論文設定)
BASE_LR=1e-3  # AdamWフェーズ
FINAL_LR=1e-6
WARMUP_EPOCHS=5

# データ設定
IMSIZE=224  # または512 (論文設定)
SEED=11
SPLIT=None

# その他設定
DEVICE="cuda"
NUM_WORKERS=4
VAL_EPOCH=1
SAVE_FREQ=5
COMMENT="ultron_experiment"

echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Embed Dim: $EMBED_DIM"
echo "Rho: $RHO"
echo "Base LR: $BASE_LR"
echo "Image Size: $IMSIZE"

# Python実行コマンドの例
echo ""
echo "=== Example Python Command ==="
echo "python multi_model_train_modified.py \\"
echo "  --model $MODEL \\"
echo "  --backbone $BACKBONE \\"
echo "  --batch_size $BATCH_SIZE \\"
echo "  --num_epochs $NUM_EPOCHS \\"
echo "  --embed_dim $EMBED_DIM \\"
echo "  --rho $RHO \\"
echo "  --base_lr $BASE_LR \\"
echo "  --final_lr $FINAL_LR \\"
echo "  --warmup_epochs $WARMUP_EPOCHS \\"
echo "  --imsize $IMSIZE \\"
echo "  --seed $SEED \\"
echo "  --device $DEVICE \\"
echo "  --num_workers $NUM_WORKERS \\"
echo "  --val_epoch $VAL_EPOCH \\"
echo "  --save_freq $SAVE_FREQ \\"
echo "  --comment $COMMENT \\"
echo "  --distributed"

python multi_model_train_modified.py \
  --model "$MODEL" \
  --backbone "$BACKBONE" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --embed_dim "$EMBED_DIM" \
  --rho "$RHO" \
  --base_lr "$BASE_LR" \
  --final_lr "$FINAL_LR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --imsize "$IMSIZE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --val_epoch "$VAL_EPOCH" \
  --save_freq "$SAVE_FREQ" \
  --comment "$COMMENT" \
  --distributed


echo ""
echo "=== Notes ==="
echo "1. ULTRONモデルは自動的にAdamW(5epoch) → SGD(35epoch)の最適化戦略を使用します"
echo "2. 論文設定では画像サイズ512x512、バッチサイズ128を推奨しています"
echo "3. 分散学習を使用する場合は --distributed フラグを追加してください"
echo "4. GPU環境に応じてバッチサイズを調整してください"
echo "5. MadaCos損失のρパラメータは論文設定の0.04を使用します"

echo ""
echo "=== SPCA Training Example (for comparison) ==="
echo "python multi_model_train_modified.py \\"
echo "  --model spca \\"
echo "  --backbone resnet50 \\"
echo "  --batch_size 32 \\"
echo "  --num_epochs 40 \\"
echo "  --outputdim 512 \\"
echo "  --codebook_size 1024 \\"
echo "  --tau 0.1 \\"
echo "  --margin 0.5 \\"
echo "  --base_lr 1e-2 \\"
echo "  --final_lr 1e-6 \\"
echo "  --comment spca_experiment"

