#!/bin/bash

# ============================================================================
# 3D Detection: Multi-GPU Training with DDP
# ============================================================================
# Reads all parameters from config.yaml and runs training with DDP
# To modify settings (epochs, batch_size, lr, etc.), edit config.yaml
# ============================================================================

echo "🚀 Starting multi-GPU training with torchrun..."
echo ""

# Run training with 4 GPUs (adjust --nproc_per_node for your setup)
torchrun --nproc_per_node=4 nndet_simple.py --mode train

# For custom config file:
# torchrun --nproc_per_node=4 nndet_simple.py --mode train --config my_config.yaml

echo ""
echo "============================================================"
echo "Training Notes:"
echo "  - All parameters loaded from config.yaml"
echo "  - Effective batch size = per_gpu_batch × num_samples × num_gpus"
echo "  - SyncBatchNorm automatically enabled for multi-GPU"
echo "  - Best model saved to: ./outputs_detection/best_model_<backbone>.pth"
echo "============================================================"

