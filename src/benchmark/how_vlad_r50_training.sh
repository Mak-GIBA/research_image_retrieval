#!/bin/bash

# HOW-VLAD with ResNet-50 学習用の設定例
# HOW + VLAD pooling for Image Retrieval

echo "HOW-VLAD R50 Training Configuration Example"
echo "==========================================="

# 基本設定
MODEL="how_vlad_r50"
BACKBONE="resnet50"
BATCH_SIZE=32
NUM_EPOCHS=100
RESNET_DEPTH=50
NUM_CLUSTERS=64  # VLAD用のクラスタ数
VLAD_DIM=2048

# 学習率設定
BASE_LR=0.01
FINAL_LR=1e-6
WARMUP_EPOCHS=5
WARMUP_LR=0.001

# その他設定
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
COMMENT="how_vlad_r50_experiment"

echo "Training HOW-VLAD R50 with the following parameters:"
echo "Model: $MODEL"
echo "Backbone: $BACKBONE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "ResNet Depth: $RESNET_DEPTH"
echo "Num Clusters: $NUM_CLUSTERS"
echo "VLAD Dim: $VLAD_DIM"
echo "Base LR: $BASE_LR"
echo ""

# 実際の学習コマンド
python multi_model_train_modified.py \
    --model $MODEL \
    --backbone $BACKBONE \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --num_clusters $NUM_CLUSTERS \
    --vlad_dim $VLAD_DIM \
    --base-lr $BASE_LR \
    --final-lr $FINAL_LR \
    --warmup-epochs $WARMUP_EPOCHS \
    --warmup-lr $WARMUP_LR \
    --weight-decay $WEIGHT_DECAY \
    --momentum $MOMENTUM \
    --comment $COMMENT \
    --distributed \
    --seed 42

