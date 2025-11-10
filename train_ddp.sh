#!/bin/bash

# DetSeg3D Multi-GPU Training Script
# Usage: bash test_ddp.sh

echo "🚀 Starting training with torchrun..."

torchrun --nproc_per_node=4 nndet_simple.py \
    --mode train \
    --image_dir /mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/image \
    --label_dir /mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/mask \
    --batch_size 2 \
    --epochs 1000 \
    --num_classes 1 \
    --backbone resnet50
# NOTE: EfficientNetBN-B3는 resnet_fpn_feature_extractor와 호환되지 않음
# 사용하려면 커스텀 FPN 구현 필요 

