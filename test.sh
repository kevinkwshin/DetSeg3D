#!/bin/bash

# ============================================================================
# Test Script for 3D Detection
# ============================================================================
# 
# IMPORTANT: Change these paths to your actual TEST data
# Current paths point to TRAINING data (AMC_train) - 747 images
# 
# For test data, use something like:
#   - AMC_test/hemo/image
#   - AMC_val/hemo/image
# ============================================================================

# Training data (DO NOT USE for testing)
# TRAIN_IMAGE_DIR="/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/image"

# ⚠️ CHANGE THIS to your actual test data directory
TEST_IMAGE_DIR="/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_test/hemo/image"
TEST_LABEL_DIR="/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_test/hemo/mask"

# If you don't have separate test data, you can use validation split from training
# TEST_IMAGE_DIR="./data/val/images"

python nndet_simple.py --mode test \
	--save_predictions \
	--test_image_dir "${TEST_IMAGE_DIR}" \
	--test_label_dir "${TEST_LABEL_DIR}" \
	--checkpoint ./outputs_detection/best_model.pth \
	--score_thresh 0.001  # Use 0.001 for early-stage models, 0.02 for well-trained
