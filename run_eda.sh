#!/bin/bash

echo "🔍 Starting EDA for 3D Detection Dataset..."
echo ""

# Default paths
IMAGE_DIR="/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/image"
LABEL_DIR="/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/mask"
OUTPUT_DIR="./eda"
MIN_SIZE=10
MERGE_DISTANCE_MM=20.0  # Merge lesions within 20mm physical distance

# Run EDA
python eda_dataset.py \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --min_size "$MIN_SIZE" \
    --merge_distance_mm "$MERGE_DISTANCE_MM"

echo ""
echo "✅ EDA complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Check the results: cat ./eda/dataset.json"
echo "   2. Run training: ./train_ddp.sh"
echo "   3. The training will automatically use the anchor shapes from EDA"

