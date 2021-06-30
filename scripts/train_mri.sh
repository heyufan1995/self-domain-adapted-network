#!/bin/bash
ROOT=/workspace/;
TRAINER=$2;BATCH_SIZE=2;AUG=0;
DATAROOT="$ROOT/Projects/data/mri/JHU/"
results_dir="$ROOT/Projects/test-time/treat_ms/exps/mri_a{$AUG}b{$BATCH_SIZE}"
export CUDA_VISIBLE_DEVICES="$1"
echo "train MRI $TRAINER on GPU $1"
mkdir -p "$results_dir"
cp -rf "$0" "$results_dir"
if [ $TRAINER == "tnet" ]; then
    python3 "$ROOT"/Projects/test-time/treat_ms/code/train.py \
    --epochs=100 \
    --task=syn_t1 \
    --batch-size=$BATCH_SIZE \
    --td=1,64,64,64,64,1 \
    --aprob=$AUG \
    --img_path="$DATAROOT"/t1/train/ \
    --label_path="$DATAROOT"/t2/train/ \
    --vimg_path="$DATAROOT"/t1/val/ \
    --vlabel_path="$DATAROOT"/t2/val/ \
    --trainer=$TRAINER \
    --img_ext=nii.gz \
    --label_ext=nii.gz \
    --results_dir="$results_dir"
else
    python3 "$ROOT"/Projects/test-time/treat_ms/code/train.py \
    --epochs=20 \
    --task=syn_t1 \
    --batch-size=$BATCH_SIZE \
    --td=1,64,64,64,64,1 \
    --aprob=0 \
    --img_path="$DATAROOT"/t1/train/ \
    --label_path="$DATAROOT"/t2/train/ \
    --vimg_path="$DATAROOT"/t1/val \
    --vlabel_path="$DATAROOT"/t2/val \
    --trainer=$TRAINER \
    --img_ext=nii.gz \
    --label_ext=nii.gz \
    --results_dir="$results_dir" \
    --wt=1,0,1,1,1,1 \
    --resume_T=$results_dir/checkpoints/tnet_checkpoint_e15.pth
fi