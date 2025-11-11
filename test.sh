#!/bin/bash

# ============================================================================
# 3D Detection: Inference/Testing
# ============================================================================
# Reads all parameters from config.yaml and runs inference
# To modify settings (test data path, score_thresh, etc.), edit config.yaml
# ============================================================================

echo "🔍 Starting inference..."
echo ""

# Run testing (all parameters from config.yaml)
python3 nndet_simple.py --mode test

# For custom config file:
# python3 nndet_simple.py --mode test --config my_config.yaml

echo ""
echo "============================================================"
echo "Testing Notes:"
echo "  - All parameters loaded from config.yaml"
echo "  - Checkpoint: ./outputs_detection/best_model_<backbone>.pth"
echo "  - Test data: Set test_image_dir in config.yaml"
echo ""
echo "Output:"
echo "  - predictions.json            : Box coordinates, scores, classes"
echo "  - predictions_nifti/*.nii.gz  : Binary masks (if save_nifti=true)"
echo "============================================================"
