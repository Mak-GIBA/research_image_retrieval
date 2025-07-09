#!/bin/bash

# SoSNet with ResNet-50 学習用の設定例
# Second-order Similarity Network for Image Retrieval

echo "SoSNet R50 Training Configuration Example"
echo "========================================="

# 基本設定
MODEL="sosnet_r50"
BACKBONE="resnet50"
BATCH_SIZE=24  # SoSNetは計算量が多いのでバッチサイズを小さく
NUM_EPOCHS=100
RESNET_DEPTH=50
SOS_DIM=256  # Second-order統計の次元
COVARIANCE_TYPE="full"  # 共分散行列のタイプ

# 学習率設定
BASE_LR=0.01
FINAL_LR=1e-6
WARMUP_EPOCHS=5
WARMUP_LR=0.001

# その他設定
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
COMMENT="sosnet_r50_experiment"

echo "Training SoSNet R50 with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "ResNet Depth: $RESNET_DEPTH"
echo "SoS Dim: $SOS_DIM"
echo "Covariance Type: $COVARIANCE_TYPE"
echo "Base LR: $BASE_LR"
echo ""

# 実際の学習コマンド
python multi_model_train_modified.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --sos_dim $SOS_DIM \
    --covariance_type $COVARIANCE_TYPE \
    --base-lr $BASE_LR \
    --final-lr $FINAL_LR \
    --warmup-epochs $WARMUP_EPOCHS \
    --warmup-lr $WARMUP_LR \
    --weight-decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --comment $COMMENT \
    --distributed \
    --seed 42

