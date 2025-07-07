#!/bin/bash

# CVNet学習用の設定例
# Correlation Verification Network for Image Retrieval

echo "CVNet Training Configuration Example"
echo "===================================="

# 基本設定
MODEL="cvnet"
BACKBONE="resnet50"
BATCH_SIZE=32
NUM_EPOCHS=40
RESNET_DEPTH=50
REDUCTION_DIM=512

# 学習率設定
BASE_LR=0.01
FINAL_LR=1e-6
WARMUP_EPOCHS=5
WARMUP_LR=0.001

# その他設定
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
COMMENT="cvnet_experiment"

echo "Training CVNet with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "ResNet Depth: $RESNET_DEPTH"
echo "Reduction Dim: $REDUCTION_DIM"
echo "Base LR: $BASE_LR"
echo ""

# 実際の学習コマンド
python multi_model_train.py \
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
  --comment $COMMENT

echo ""
echo "CVNet training command executed!"
echo ""
echo "Alternative configurations:"
echo ""
echo "# ResNet-101 with larger reduction dimension:"
echo "python multi_model_train.py --model cvnet --backbone resnet101 --resnet_depth 101 --reduction_dim 1024 --batch_size 16 --num_epochs 40"
echo ""
echo "# Smaller model for faster training:"
echo "python multi_model_train.py --model cvnet --backbone resnet50 --resnet_depth 50 --reduction_dim 256 --batch_size 64 --num_epochs 20"
echo ""
echo "# High-performance configuration:"
echo "python multi_model_train.py --model cvnet --backbone resnet50 --resnet_depth 50 --reduction_dim 512 --base_lr 0.02 --batch_size 32 --num_epochs 60"

