#!/bin/bash

# ============================================================================
# Anchor Shape Optimization & Patch Size Recommendation
# ============================================================================
# This script reads all parameters from config.yaml and performs:
#   1. Dataset analysis (image shapes, lesion sizes)
#   2. K-Means clustering for optimal anchor shapes
#   3. Patch size recommendation for training/validation
#   4. (Optional) Save sample cases showing box merging
#
# To modify settings, edit config.yaml directly.
# ============================================================================

echo "🚀 Running anchor optimization..."
echo ""

# Run optimization (all parameters from config.yaml)
python3 optimize_anchors.py

echo ""
echo "============================================================"
echo "✅ Done! Check the following:"
echo "   - ./eda/optimized_anchors.json  (anchor shapes + patch sizes)"
echo "   - ./eda/anchor_optimization.png (visualization)"
echo "   - ./eda/samples/                (optional: box merging samples)"
echo "============================================================"

