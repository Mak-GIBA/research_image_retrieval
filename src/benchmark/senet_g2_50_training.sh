#!/bin/bash

# SENet-G2+ with 50 layers 学習用の設定例
# Squeeze-and-Excitation Networks with G2+ pooling

echo "SENet-G2+ 50 Training Configuration Example"
echo "==========================================="

# 基本設定
MODEL="senet_g2_50"
BACKBONE="senet50"
BATCH_SIZE=32
NUM_EPOCHS=100
SENET_DEPTH=50
REDUCTION_RATIO=16  # SENet用のreduction ratio
G2_ALPHA=1.0  # G2+ pooling用のalpha

# 学習率設定
BASE_LR=0.01
FINAL_LR=1e-6
WARMUP_EPOCHS=5
WARMUP_LR=0.001

# その他設定
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
COMMENT="senet_g2_50_experiment"

echo "Training SENet-G2+ 50 with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "SENet Depth: $SENET_DEPTH"
echo "Reduction Ratio: $REDUCTION_RATIO"
echo "G2 Alpha: $G2_ALPHA"
echo "Base LR: $BASE_LR"
echo ""

# 実際の学習コマンド
python multi_model_train_modified.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --senet_depth $SENET_DEPTH \
    --reduction_ratio $REDUCTION_RATIO \
    --g2_alpha $G2_ALPHA \
    --base_lr $BASE_LR \
    --final_lr $FINAL_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --warmup_lr $WARMUP_LR \
    --weight_decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --comment $COMMENT \
    --distributed \
    --seed 42

