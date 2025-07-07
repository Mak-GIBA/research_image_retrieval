#!/bin/bash

# DOLGNet学習用の設定例
# DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features

echo "DOLGNet Training Configuration Example"
echo "====================================="

# 基本設定
MODEL="dolgnet"
BACKBONE="resnet50"
BATCH_SIZE=32
NUM_EPOCHS=40
LOCAL_DIM=1024
GLOBAL_DIM=2048
OUTPUT_DIM=512

# 学習率設定 (AdamW推奨)
BASE_LR=0.001
FINAL_LR=1e-6
WARMUP_EPOCHS=5
WARMUP_LR=0.0001

# その他設定
WEIGHT_DECAY=0.0001
COMMENT="dolgnet_experiment"

echo "Training DOLGNet with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Local Dim: $LOCAL_DIM"
echo "Global Dim: $GLOBAL_DIM"
echo "Output Dim: $OUTPUT_DIM"
echo "Base LR: $BASE_LR (AdamW)"
echo ""

# 実際の学習コマンド
python multi_model_train.py \
  --model $MODEL \
  --backbone $BACKBONE \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --local_dim $LOCAL_DIM \
  --global_dim $GLOBAL_DIM \
  --output_dim $OUTPUT_DIM \
  --base_lr $BASE_LR \
  --final_lr $FINAL_LR \
  --warmup_epochs $WARMUP_EPOCHS \
  --warmup_lr $WARMUP_LR \
  --weight_decay $WEIGHT_DECAY \
  --comment $COMMENT

echo ""
echo "DOLGNet training command executed!"
echo ""
echo "Alternative configurations:"
echo ""
echo "# ResNet-101 with larger dimensions:"
echo "python multi_model_train.py --model dolgnet --backbone resnet101 --local_dim 1536 --global_dim 3072 --output_dim 768 --batch_size 16 --num_epochs 40"
echo ""
echo "# Lightweight configuration:"
echo "python multi_model_train.py --model dolgnet --backbone resnet50 --local_dim 512 --global_dim 1024 --output_dim 256 --batch_size 64 --num_epochs 20"
echo ""
echo "# High-performance configuration:"
echo "python multi_model_train.py --model dolgnet --backbone resnet152 --local_dim 2048 --global_dim 4096 --output_dim 1024 --base_lr 0.0005 --batch_size 16 --num_epochs 60"
echo ""
echo "# Fast training for debugging:"
echo "python multi_model_train.py --model dolgnet --backbone resnet50 --local_dim 256 --global_dim 512 --output_dim 128 --batch_size 32 --num_epochs 5 --base_lr 0.01"

