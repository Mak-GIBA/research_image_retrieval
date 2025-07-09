#!/bin/bash

# GeM pooling with ResNet-50 学習用の設定例
# Generalized Mean Pooling for Image Retrieval

echo "GeM R50 Training Configuration Example"
echo "======================================"

# 基本設定
MODEL="gem_r50"
BACKBONE="resnet50"
BATCH_SIZE=32
NUM_EPOCHS=100
RESNET_DEPTH=50
REDUCTION_DIM=2048

# 学習率設定
BASE_LR=0.01
FINAL_LR=1e-6
WARMUP_EPOCHS=5
WARMUP_LR=0.001

# その他設定
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
COMMENT="gem_r50_experiment"

echo "Training GeM R50 with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "ResNet Depth: $RESNET_DEPTH"
echo "Reduction Dim: $REDUCTION_DIM"
echo "Base LR: $BASE_LR"
echo ""

# 実際の学習コマンド
python multi_model_train_modified.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --resnet_depth $RESNET_DEPTH \
    --reduction_dim $REDUCTION_DIM \
    --base_lr $BASE_LR \
    --final_lr $FINAL_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --warmup_lr $WARMUP_LR \
    --weight_decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --comment $COMMENT \
    --distributed \
    --seed 42

