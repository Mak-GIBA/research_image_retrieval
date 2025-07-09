# =========================================
#  SPCA 学習
# =========================================
MODEL="spca"               # ここだけ変える
BACKBONE="resnet50"
BATCH_SIZE=32
NUM_EPOCHS=40
OUTPUTDIM=512              # SPCA 固有
CODEBOOK_SIZE=1024         # SPCA 固有
TAU=0.1                    # SPCA 固有
MARGIN=0.5                 # SPCA 固有
BASE_LR=1e-2
FINAL_LR=1e-6
IMSIZE=224
SEED=11
DEVICE="cuda"
NUM_WORKERS=4
VAL_EPOCH=1
SAVE_FREQ=5
COMMENT="spca_experiment"

echo ""
echo "=== SPCA training start ==="
python multi_model_train_modified.py \
  --model "$MODEL" \
  --backbone "$BACKBONE" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --outputdim "$OUTPUTDIM" \
  --codebook_size "$CODEBOOK_SIZE" \
  --tau "$TAU" \
  --margin "$MARGIN" \
  --base_lr "$BASE_LR" \
  --final_lr "$FINAL_LR" \
  --imsize "$IMSIZE" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --val_epoch "$VAL_EPOCH" \
  --save_freq "$SAVE_FREQ" \
  --comment "$COMMENT" \
  --distributed
echo "=== SPCA training done ==="