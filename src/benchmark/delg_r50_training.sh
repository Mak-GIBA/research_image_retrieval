#!/bin/bash

# DELG with ResNet-50 学習用の設定例
# Deep Local and Global features for Image Retrieval

echo "DELG R50 Training Configuration Example"
echo "======================================="

# 基本設定
MODEL="delg_r50"
BACKBONE="resnet50"
BATCH_SIZE=32
NUM_EPOCHS=100
RESNET_DEPTH=50
LOCAL_DIM=1024
GLOBAL_DIM=2048

# 学習率設定
BASE_LR=0.01
FINAL_LR=1e-6
WARMUP_EPOCHS=5
WARMUP_LR=0.001

# その他設定
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
COMMENT="delg_r50_experiment"

echo "Training DELG R50 with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "ResNet Depth: $RESNET_DEPTH"
echo "Local Dim: $LOCAL_DIM"
echo "Global Dim: $GLOBAL_DIM"
echo "Base LR: $BASE_LR"
echo ""

# 実際の学習コマンド
python multi_model_train_modified.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --local-dim $LOCAL_DIM \
    --outputdim $GLOBAL_DIM \
    --base-lr $BASE_LR \
    --final-lr $FINAL_LR \
    --warmup-epochs $WARMUP_EPOCHS \
    --warmup-lr $WARMUP_LR \
    --weight-decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --comment $COMMENT \
    --distributed \
    --seed 42

