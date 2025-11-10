# DetSeg3D: 3D Medical Lesion Detection

Professional 3D object detection for medical imaging based on **MONAI RetinaNet**.

---

## 🎯 Overview

This project implements state-of-the-art **3D RetinaNet** for medical lesion detection using the **MONAI detection module**.

**Key Features:**
- ✅ **MONAI RetinaNet**: Production-ready 3D detection
- ✅ **ResNet50 + FPN backbone**: Multi-scale feature extraction
- ✅ **ATSS Matcher**: Adaptive Training Sample Selection
- ✅ **Anchor-based detection**: Proven robust performance
- ✅ **Auto segmentation-to-box conversion**: Works with segmentation labels
- ✅ **Multi-GPU support**: DistributedDataParallel (DDP) training with `torchrun`
- ✅ **AMP (Mixed Precision)**: Faster training with FP16
- ✅ **Sliding window inference**: Handles large images
- ✅ **COCO metrics**: Standard evaluation (mAP, mAR)

---

## 📐 Architecture

```
Input (1, D, H, W)
    ↓
ResNet50 Backbone
    ↓
Feature Pyramid Network (FPN)
    ├─ P3 (stride=8)
    ├─ P4 (stride=16)
    └─ P5 (stride=32)
    ↓
RetinaNet Heads
    ├─ Classification Head (Focal Loss)
    └─ Box Regression Head (L1 Loss)
    ↓
ATSS Matcher + Hard Negative Mining
    ↓
NMS (Non-Maximum Suppression)
    ↓
Output: Boxes + Confidence Scores
```

### Components

#### 1. **Backbone: ResNet50**
- Pre-downsampling with stride [2,2,1] for 3D medical images
- Residual blocks: [3, 4, 6, 3] (ResNet50)
- Output features from layer 1 and 2 for small lesion detection

#### 2. **Feature Pyramid Network (FPN)**
- Multi-scale features for detecting lesions of various sizes
- Top-down pathway with lateral connections
- Feature map scales: [1, 2, 4]

#### 3. **RetinaNet Detection Heads**
- **Classification head**: Focal loss for class imbalance
- **Box regression head**: Smooth L1 loss
- Anchors: Multiple aspect ratios per location

#### 4. **ATSS Matcher**
- Adaptive Training Sample Selection
- Automatically determines positive/negative samples
- Better than IoU-based matching for small objects

#### 5. **Hard Negative Mining**
- Balances positive/negative samples (ratio: 0.3)
- Focuses on hard negatives
- Reduces false positives

---

## 🚀 Installation

```bash
# Create conda environment
conda create -n detseg3d python=3.10
conda activate detseg3d

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install MONAI with detection support
pip install "monai[all]"

# Install other dependencies
pip install scipy tensorboard tqdm
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- MONAI ≥ 1.3 (with detection module)
- CUDA ≥ 11.0 (for GPU)

---

## 📁 Data Preparation

### Directory Structure

```
your_data/
├── images/
│   ├── case001.nii.gz
│   ├── case002.nii.gz
│   └── ...
└── labels/
    ├── case001.nii.gz  (binary segmentation mask)
    ├── case002.nii.gz
    └── ...
```

### Label Format

- **Segmentation masks**: Binary or multi-class (H, W, D)
- **Automatic box extraction**: Connected components → bounding boxes
- **Coordinate system**: Image coordinates (handled automatically)

**No manual box annotation needed!** The code automatically extracts bounding boxes from segmentation masks.

---

## 🏋️ Training

### Basic Training

```bash
python nndet_simple.py --mode train \
    --image_dir /path/to/images \
    --label_dir /path/to/labels \
    --output_dir ./outputs \
    --batch_size 1 \
    --epochs 100
```

### Multi-GPU Training (Recommended - DDP with torchrun)

```bash
# Use torchrun for efficient DistributedDataParallel training
# --batch_size is PER GPU (total = batch_size × num_gpus)
torchrun --nproc_per_node=4 nndet_simple.py \
    --mode train \
    --image_dir /path/to/images \
    --label_dir /path/to/labels \
    --output_dir ./outputs \
    --batch_size 1 \
    --epochs 100
```

**Benefits of torchrun + DDP:**
- ✅ **All GPUs fully utilized** (unlike DataParallel which underutilizes)
- ✅ **Faster training**: Each GPU runs an independent process
- ✅ **Better gradient sync**: Efficient all-reduce operations
- ✅ **Simple scaling**: Just change `--nproc_per_node`

**Example:** 4 GPUs × `--batch_size 1` = **effective batch size of 4**

### Advanced Options

```bash
python nndet_simple.py --mode train \
    --image_dir /path/to/images \
    --label_dir /path/to/labels \
    --output_dir ./outputs \
    --batch_size 2 \
    --patch_size 192 192 80 \
    --val_patch_size 512 512 208 \
    --lr 1e-2 \
    --num_classes 1 \
    --multi_gpu \
    --amp \
    --val_interval 5 \
    --epochs 100
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | **required** | Path to image directory |
| `--label_dir` | **required** | Path to label directory |
| `--output_dir` | `./outputs_detection` | Output directory for models and logs |
| `--batch_size` | `1` | Batch size **per GPU** |
| `--patch_size` | `[192,192,80]` | Training patch size (D,H,W) |
| `--val_patch_size` | `[512,512,208]` | Validation patch size |
| `--lr` | `1e-2` | Learning rate |
| `--num_classes` | `1` | Number of foreground classes |
| `--multi_gpu` | `False` | (Deprecated) Use `torchrun` instead for DDP |
| `--amp` | `True` | Use automatic mixed precision |
| `--val_interval` | `5` | Validation every N epochs |
| `--epochs` | `100` | Total epochs |

---

## 📊 Validation & Metrics

The code automatically evaluates using **COCO metrics**:

- **mAP (mean Average Precision)**: IoU thresholds [0.1, 0.3, 0.5]
- **mAR (mean Average Recall)**: Max detections = 100
- **Per-class metrics**: If multiple classes

Example output:
```
Validation Results:
  mAP: 0.8543
  mAP@0.1: 0.9234
  mAP@0.3: 0.8765
  mAP@0.5: 0.7631
  mAR: 0.8912
```

---

## 🧪 Inference

Coming soon! Will include:
- Sliding window inference for large volumes
- Box NMS with configurable threshold
- World coordinate conversion
- JSON/CSV export
- Visualization

---

## 🔬 Data Augmentation

The training pipeline includes comprehensive augmentation:

**Spatial:**
- Random zoom (0.8-1.2)
- Random flip (3 axes, prob=0.5 each)
- Random 90° rotation (prob=0.5)

**Intensity:**
- Gaussian noise (prob=0.1)
- Gaussian smooth (prob=0.1)
- Scale intensity (prob=0.15)
- Shift intensity (prob=0.15)
- Adjust contrast (prob=0.3, gamma=0.7-1.5)

**Box handling:**
- Boxes are converted to points before augmentation
- Same transforms applied to images and points
- Points converted back to boxes
- Invalid boxes (outside image) are removed

---

## 🎛️ Hyperparameters

### Model Architecture

```python
# Anchor shapes (adjust for your lesion sizes)
base_anchor_shapes = [[6,8,4], [8,6,5], [10,10,6]]

# FPN returned layers (lower = higher resolution for small lesions)
returned_layers = [1, 2]  # Use layers 1 and 2

# ResNet stride configuration
conv1_t_stride = [2, 2, 1]  # Less downsampling in Z for medical images
```

### Training Configuration

```python
# ATSS matcher
num_candidates = 4          # Number of candidate anchors per GT
center_in_gt = False        # Relaxed matching for small objects

# Hard negative sampler
batch_size_per_image = 64   # Samples per image
positive_fraction = 0.3     # 30% positive, 70% negative
pool_size = 20
min_neg = 16

# NMS parameters
score_thresh = 0.02         # Confidence threshold
nms_thresh = 0.22           # IoU threshold for NMS
detections_per_img = 100    # Max detections per image
```

### Optimizer & Scheduler

```python
# SGD with momentum
optimizer = torch.optim.SGD(
    params,
    lr=1e-2,
    momentum=0.9,
    weight_decay=3e-5,
    nesterov=True
)

# Step LR scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,
    gamma=0.1
)
```

---

## 📈 Monitoring

**TensorBoard:**
```bash
tensorboard --logdir outputs_detection/tensorboard
```

**Metrics tracked:**
- Training: total loss, classification loss, box regression loss, learning rate
- Validation: mAP, mAR (at various IoU thresholds)

---

## 🗂️ Outputs

```
outputs_detection/
├── best_model.pth              # Best model by mAP
├── checkpoint_epoch10.pth      # Checkpoints every 10 epochs
├── checkpoint_epoch20.pth
├── ...
└── tensorboard/                # TensorBoard logs
    └── events.out.tfevents.*
```

**Model checkpoint contains:**
- `model_state_dict`: Network weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Epoch number
- `best_metric`: Best mAP score (for best_model.pth)

---

## 🎓 Reference

This implementation is based on:

**MONAI Detection Module:**
- [MONAI Detection Tutorial](https://github.com/Project-MONAI/tutorials/tree/main/detection)
- [MONAI Documentation](https://docs.monai.io/)

**Papers:**
- **RetinaNet:** [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) (Lin et al., ICCV 2017)
- **ATSS:** [Bridging the Gap Between Anchor-based and Anchor-free Detection](https://arxiv.org/abs/1912.02424) (Zhang et al., CVPR 2020)
- **nnDetection:** [A Self-Configuring Method for Medical Object Detection](https://arxiv.org/abs/2106.00817) (Baumgartner et al., MICCAI 2021)

---

## 🤝 Acknowledgements

- **MONAI Team** for the excellent detection module
- **nnDetection** for design insights and best practices
- **LUNA16 Challenge** for evaluation methodology

---

## 📝 Citation

If you use this code, please cite MONAI and the relevant papers:

```bibtex
@article{cardoso2022monai,
  title={MONAI: An open-source framework for deep learning in healthcare},
  author={Cardoso, M Jorge and others},
  journal={arXiv preprint arXiv:2211.02701},
  year={2022}
}

@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={ICCV},
  year={2017}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 (same as MONAI).

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solution 1:** Reduce patch size
```bash
--patch_size 128 128 64  # Smaller patches
```

**Solution 2:** Reduce batch size
```bash
--batch_size 1  # Already minimal
```

**Solution 3:** Disable AMP (if causing issues)
```bash
--no-amp  # Use FP32 instead of FP16
```

### No boxes detected from labels

**Check 1:** Verify label format (binary mask, 0=background, 1=foreground)

**Check 2:** Adjust minimum size threshold
```python
# In GenerateBoxMaskd class
min_size = 5  # Reduce from 10
```

### Low mAP scores

**Solution 1:** Adjust anchor shapes for your lesion sizes
```python
base_anchor_shapes = [[4,4,2], [6,6,3], [8,8,4]]  # Smaller for tiny lesions
```

**Solution 2:** Use more FPN layers
```python
returned_layers = [0, 1, 2]  # Include layer 0 (highest resolution)
```

**Solution 3:** Train longer
```bash
--epochs 200  # More epochs for convergence
```

---

## 🔮 Future Work

- [ ] Test-time augmentation (TTA)
- [ ] Ensemble inference
- [ ] Segmentation refinement (Stage 2)
- [ ] 3D visualization tools
- [ ] FROC curve evaluation
- [ ] Cross-validation support
- [x] DistributedDataParallel (DDP) for multi-GPU training (`torchrun` support)

---

**Happy detecting! 🎯**
