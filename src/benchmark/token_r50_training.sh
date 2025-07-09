#!/bin/bash

# Token-based with ResNet-50 学習用の設定例
# Transformer-based Token Aggregation for Image Retrieval

echo "Token-based R50 Training Configuration Example"
echo "=============================================="

# 基本設定
MODEL="token_r50"
BACKBONE="resnet50"
BATCH_SIZE=24  # Transformerは重いのでバッチサイズを小さく
NUM_EPOCHS=100
RESNET_DEPTH=50
EMBED_DIM=512
NUM_HEADS=8
NUM_LAYERS=6

# 学習率設定
BASE_LR=0.001  # Transformerは小さい学習率が推奨
FINAL_LR=1e-6
WARMUP_EPOCHS=10  # Transformerは長いウォームアップが推奨
WARMUP_LR=0.0001

# その他設定
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
COMMENT="token_r50_experiment"

echo "Training Token-based R50 with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "ResNet Depth: $RESNET_DEPTH"
echo "Embed Dim: $EMBED_DIM"
echo "Num Heads: $NUM_HEADS"
echo "Num Layers: $NUM_LAYERS"
echo "Base LR: $BASE_LR"
echo ""

# 実際の学習コマンド
python multi_model_train_modified.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --embed_dim $EMBED_DIM \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --base-lr $BASE_LR \
    --final-lr $FINAL_LR \
    --warmup-epochs $WARMUP_EPOCHS \
    --warmup-lr $WARMUP_LR \
    --weight-decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --comment $COMMENT \
    --distributed \
    --seed 42

