# 3D Detection with MONAI RetinaNet

Simple, function-based 3D object detection for medical images using MONAI RetinaNet.

---

## 🚀 Quick Start

```bash
# 1. Optimize anchors + patch sizes (recommended, run once)
bash run_optimize_anchors.sh
# → Saves optimized_anchors.json with:
#   - anchor_shapes (for RetinaNet)
#   - train_patch_size (auto-loaded during training)

# 2. Train (automatically uses optimized values from step 1)
bash train_ddp.sh

# 3. Test
bash test.sh
```

---

## ⚙️ Configuration (`config.yaml`)

All parameters are managed in `config.yaml`:

```yaml
data:
  image_dir: "/path/to/images"
  label_dir: "/path/to/labels"

model:
  backbone: "resnet101"  # resnet50 or resnet101

training:
  epochs: 1000
  batch_size: 1
  lr: 0.005
  patch_size: [256, 256, 24]
```

**Priority**: CLI args > `config.yaml` > defaults

**Override example**:
```bash
python nndet_simple.py --mode train --batch_size 2  # Override config
```

---

## ✨ Features

- **Unified config**: Central `config.yaml` for all parameters
- **Self-contained**: Single file (`nndet_simple.py`), function-based
- **Multi-GPU**: DDP with `torchrun`, SyncBatchNorm
- **Auto-resume**: Loads `best_model_{backbone}.pth` if exists
- **Anchor optimization**: K-Means clustering (YOLO-style) + **auto patch size recommendation**
- **Smart loading**: Automatically uses optimized anchors & patch sizes from EDA
- **Backbones**: ResNet50, ResNet101
- **Smart cropping**: `RandCropByPosNegLabeld` (50% lesions, 50% background)
- **Debug mode**: Saves validation samples as NIfTI
- **Config verification**: Auto-checks checkpoint compatibility during test
- **NIfTI export**: Save predictions as masks for visualization

---

## 📦 Installation

```bash
pip install torch monai nibabel scikit-learn matplotlib tqdm tensorboard pyyaml
```

---

## 🎯 Usage

### **1️⃣ Configure (`config.yaml`)**

Edit `config.yaml` to set all parameters:

```yaml
data:
  image_dir: "/path/to/train/images"
  label_dir: "/path/to/train/labels"
  test_image_dir: "/path/to/test/images"  # For inference

model:
  backbone: "resnet50"  # or "resnet101"
  num_classes: 1

training:
  epochs: 1000
  batch_size: 1  # per GPU
  lr: 0.005
  patch_size: [256, 256, 24]  # Fallback (auto-optimized if you run step 0)

detection:
  score_thresh_test: 0.3  # Min confidence (0.2-0.4)
  detections_per_img_test: 10  # Max detections (10-20)
```

**Note:** `patch_size` in config.yaml is a fallback value. If you run anchor optimization (step 0), the recommended patch size will be automatically loaded from `./eda/optimized_anchors.json`.

---

### **0️⃣ Optimize (Optional but Recommended)**

Run this **once** before training to optimize anchor shapes and patch sizes for your dataset:

```bash
bash run_optimize_anchors.sh
```

**Output:** `./eda/optimized_anchors.json`
- ✅ Optimized anchor shapes (for RetinaNet)
- ✅ Recommended patch size (auto-loaded during training)
- ✅ Dataset statistics (image/lesion sizes)

**What happens:**
- Analyzes your dataset (image sizes, lesion sizes)
- Uses K-Means to find optimal anchor shapes
- Calculates optimal patch_size based on lesion distribution
- Saves everything to `optimized_anchors.json`

**When training:**
- Training script **automatically** loads these optimized values
- No need to manually edit `config.yaml`!

---

### **2️⃣ Train**

```bash
# Single GPU
python3 nndet_simple.py

# Multi-GPU (4 GPUs with DDP)
bash train_ddp.sh
# or
torchrun --nproc_per_node=4 nndet_simple.py
```

**Training automatically:**
- ✅ Loads **optimized anchor shapes** from `./eda/optimized_anchors.json` (if exists)
- ✅ Loads **optimized patch_size** from `./eda/optimized_anchors.json` (if exists)
- ✅ Saves best model to `./outputs_detection/best_model_{backbone}.pth`
- ✅ Uses SyncBatchNorm for multi-GPU training
- ✅ Logs to TensorBoard: `./outputs_detection/tensorboard/`

**💡 If optimization was not run:**
- Falls back to `config.yaml` values
- Still works, but may be suboptimal for your dataset

---

### **3️⃣ Test/Inference**

```bash
bash test.sh
# or
python3 nndet_simple.py --mode test
```

**Output:**
- `predictions.json`: Bounding boxes, scores, classes
- `predictions_nifti/*.nii.gz`: Binary masks (if `save_nifti: true` in config)

**Configuration Verification:**  
Automatically checks checkpoint compatibility:
- ✅ Backbone, num_classes, anchor_shapes
- ⚠️ Warns if mismatch detected!

---

### **4️⃣ Tuning Detection Parameters**

If you get too many/few detections or duplicates, edit `config.yaml`:

```yaml
detection:
  score_thresh_test: 0.35  # Higher = fewer detections (more strict)
  nms_thresh_test: 0.15    # Lower = stricter NMS (remove more overlaps)
  detections_per_img_test: 5  # Lower = only top-N confident
```

**💡 Tips:**

**Problem: Too many false positives?**
- → Increase `score_thresh_test` (e.g., 0.3 → 0.4)

**Problem: Missing lesions?**
- → Decrease `score_thresh_test` (e.g., 0.3 → 0.2)

**Problem: Duplicate detections (multiple boxes on same lesion)?**
- → Decrease `nms_thresh_test` (e.g., 0.22 → 0.15)
- → NMS removes overlapping boxes with IoU > threshold

**Problem: Scores always low (~0.3)?**
- → Model needs more training epochs
- → Check if anchors are optimized (`bash run_optimize_anchors.sh`)

---

### **Anchor Optimization (K-Means)**

```bash
# Run anchor optimization (reads from config.yaml)
bash run_optimize_anchors.sh
```

**Output:**
- `./eda/optimized_anchors.json` - Anchor shapes + recommended patch sizes (auto-loaded during training)
- `./eda/anchor_optimization.png` - Visualization of lesion distribution
- `./eda/optimal_k_curve.png` - IoU vs number of anchors (if find_optimal_k=true)

**Note:** All parameters are read from `config.yaml`. To change settings (e.g., `merge_distance_mm`, `num_anchors`), edit the config file.

**Visualize Box Merging:**

To save sample cases showing before/after box merging, set in `config.yaml`:
```yaml
anchor:
  save_samples: true
  num_samples: 5
```

Then run:
```bash
bash run_optimize_anchors.sh
```

**Output:** `./eda/samples/sample_XX_*/`
- `image.nii.gz`: Original image
- `mask_original.nii.gz`: Original segmentation
- `boxes_before_merge.nii.gz`: Boxes BEFORE merging (20mm)
- `boxes_after_merge.nii.gz`: Boxes AFTER merging
- `info.txt`: Detailed box information

---

## 📊 Model Configuration

**Architecture:**
- Backbone: ResNet50/101 + FPN (3 layers)
- Detection head: RetinaNet (Focal Loss + Smooth L1)
- Matcher: ATSS (Adaptive Training Sample Selection)
- Anchors: K-Means optimized or default 5 shapes

**Training:**
- Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
- Scheduler: Warmup (10 epochs) + Step decay
- AMP: Enabled by default (mixed precision)
- Data: 80% train, 20% validation

---

## 🔧 Checkpoint Management

Checkpoints are saved with backbone name:
```
outputs_detection/
├── best_model_resnet50.pth
├── best_model_resnet101.pth
└── tensorboard/
```

Each checkpoint contains:
- `model_state_dict`
- `optimizer_state_dict`
- `epoch`, `best_metric`
- `backbone`, `anchor_shapes`

---

## 📈 Output

**Training logs:**
```
Epoch 50/1000
─────────────────────────
📉 Train Loss: 0.1234 | LR: 4.50e-03

📊 Detection Statistics (IoU ≥ 0.1):
   ├─ True Positives:  1234
   ├─ False Positives:  256
   ├─ False Negatives:  800
   ├─ Sensitivity:     60.7%
   └─ Precision:       82.8%

📈 Validation Results:
  ├─ mAP@0.10-0.50: 0.3456
  ├─ AP@0.10:       0.4123
  ├─ Sensitivity:   60.7%
  └─ 🎯 Avg Metric:  0.3456 ← New Best! 🏆
   💾 Saved: best_model_resnet50.pth
```

**Test output:**
```
./outputs_detection/
├── test_predictions.json  # All predictions
└── debug_samples/         # Visual debugging (optional)
    ├── image.nii.gz
    ├── gt_boxes.nii.gz
    ├── pred_boxes.nii.gz
    └── info.txt
```

---

## 🎓 Tips

**For better performance:**
1. ✅ Run `run_optimize_anchors.sh` before training
2. ✅ Use ResNet101 for larger datasets (>500 images)
3. ✅ Adjust `score_thresh`: 0.1 (more detections) ↔ 0.3 (higher precision)
4. ✅ Increase `patch_size` if you have large GPU memory
5. ✅ Use `--debug` for verbose box extraction logs

**For faster training:**
- Reduce `patch_size` (e.g., `192 192 16`)
- Use ResNet50 instead of ResNet101
- Reduce `batch_size` if OOM

---

## 📁 Project Structure

```
DetSeg3D/
├── nndet_simple.py              # Main script (all-in-one)
├── train_ddp.sh                 # Multi-GPU training
├── test.sh                      # Inference
├── optimize_anchors.py          # K-Means anchor optimization
├── run_optimize_anchors.sh      # Anchor optimization script
└── ANCHOR_OPTIMIZATION.md       # Detailed anchor guide
```

---

## 🔗 References

- **MONAI**: https://monai.io/
- **RetinaNet**: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- **ATSS**: [Zhang et al., 2020](https://arxiv.org/abs/1912.02424)
- **Anchor Optimization**: [Redmon et al., 2016](https://arxiv.org/abs/1506.02640)

---

## 📝 License

MIT

---

## 🐛 Troubleshooting

**Q: Architecture mismatch error?**  
A: Checkpoint was trained with different anchors. Re-train or check `anchor_shapes`.

**Q: Zero detections?**  
A: Lower `score_thresh` (e.g., `0.1`) or train longer.

**Q: OOM (Out of Memory)?**  
A: Reduce `batch_size` or `patch_size`.

**Q: DDP hangs?**  
A: Check multi-GPU setup, ensure all GPUs are available.

---

**For detailed anchor optimization guide, see `ANCHOR_OPTIMIZATION.md`**
