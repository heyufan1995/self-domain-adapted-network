#!/bin/bash
ROOT=/workspace/;
BATCH_SIZE=2;AUG=0;
SITENAME=(HH)
for SITE in ${SITENAME[@]}
do
    DATAROOT="$ROOT/Projects/data/mri/"
    results_dir="$ROOT/Projects/iacl/test-time/treat_ms/exps/mri_a{$AUG}b{$BATCH_SIZE}"
    checkpoints='checkpoints'
    export CUDA_VISIBLE_DEVICES="$1"
    echo "test MRI $SITE on GPU $1"
    mkdir -p "$results_dir"
    cp -rf "$0" "$results_dir"
    python3 "$ROOT"/Projects/iacl/test-time/treat_ms/code/train.py \
    --tepochs=10 \
    --task=syn_t1 \
    --batch-size=1 \
    --td=1,64,64,64,64,1 \
    --img_path="$DATAROOT"/$SITE/t1/ \
    --label_path="$DATAROOT"/$SITE/t2/ \
    --vimg_path="$DATAROOT"/$SITE/t1/ \
    --vlabel_path="$DATAROOT"/$SITE/t2/ \
    --sub_name="$DATAROOT"/$SITE/IXI_name.txt \
    --img_ext=nii.gz \
    --label_ext=nii.gz \
    --results_dir="$results_dir"/"$SITE" \
    --ss=1 \
    --resume_T=$results_dir/$checkpoints/tnet_checkpoint_e15.pth \
    --resume_AE=$results_dir/$checkpoints/aenet_checkpoint_e20.pth \
    --wo=5 \
    --wt=1,0,1,1,1,1 \
    --seq=1,2,3 \
    -t
done

