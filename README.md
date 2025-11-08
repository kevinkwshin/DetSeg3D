# DetSeg3D: End-to-End RoI-based 3D Detection-Segmentation

## 개요

전체 3D volume에서 가볍게 RoI를 탐지하고, 각 RoI를 정밀하게 분할하는 two-stage 모델

```
Input: Full 3D Volume
    ↓
Stage 1: Detection Network (lightweight)
    → RoI proposals + confidence scores
    ↓
Stage 2: Segmentation Network (per RoI)
    → Precise mask per lesion
```

**핵심 아이디어:** 작은 병변도 큰 병변과 동등한 가중치로 학습

---

## Architecture

### Stage 1: Detection Network (CenterNet-style) ⭐

**입력:** 원본 3D volume (no patches)  
**출력:** RoI coordinates + confidence scores

**구조:**
- **MONAI ResNet-style backbone** with residual connections
- Multi-scale feature extraction (3 levels, 8× downsampling)
- Feature fusion layer
- **CenterNet-style anchor-free heads**
  - **Center heatmap** (Gaussian): 병변 중심점 위치
  - **Bounding box size** (3D): (d, h, w) 예측
  - **Center offset** (sub-pixel): 정밀한 중심 좌표

**Training (CenterNet Style):**
1. **GT Generation**: Segmentation label → Connected Components → BBoxes
2. **Gaussian Heatmap**: 각 bbox center에 Gaussian kernel 생성
3. **Size Target**: 각 center point에서 bbox 크기 학습
4. **Offset Target**: Sub-pixel refinement

**특징:**
- ✅ **Proper Detection Training**: Size와 Offset이 실제로 학습됨
- ✅ **Multi-scale Support**: 다양한 크기의 병변 대응
- ✅ **Gaussian Heatmap**: CenterNet 논문과 동일한 방식
- ✅ **Anchor-free**: Anchor 없이 직접 예측
- Parameters: ~2.5M (적절한 균형)

### Stage 2: Segmentation Network (Enhanced)

**입력:** 각 RoI crop (예: 32³ resize)  
**출력:** Binary mask per RoI

**구조:**
- **Enhanced 3D U-Net** with deeper architecture
- 4-level encoder-decoder (32 → 64 → 128 → 256)
- Residual units at each level (2 units)
- Dropout (0.1) for regularization

**특징:**
- 각 RoI 독립적으로 처리 → 크기 무관 동등 가중치
- 더 강력한 feature extraction
- Parallel processing 가능
- 작은 병변 확대 효과
- Parameters: ~5M (정밀한 분할)

---

## Loss Functions

### Stage 1: Detection Loss (CenterNet Style)

```python
# 1. GT Generation (from segmentation label)
batch_bboxes = extract_bboxes_from_label(gt_label)  # Connected components

# 2. Generate Gaussian targets
gt_heatmap = generate_gaussian_heatmap(bboxes)      # Gaussian at centers
gt_size = extract_size_at_centers(bboxes)           # Size at center points
gt_offset = extract_offset_at_centers(bboxes)       # Sub-pixel offset

# 3. Loss computation (CenterNet)
L_det = Focal(pred_heatmap, gt_heatmap) 
      + 0.5 × L1(pred_size, gt_size)[positive_points]
      + 0.1 × L1(pred_offset, gt_offset)[positive_points]
```

**핵심:**
- ✅ **GT BBox**: Segmentation label → Connected components → Real bboxes
- ✅ **Gaussian Heatmap**: 각 bbox center에 Gaussian 생성
- ✅ **Positive Points**: GT center에서만 size/offset loss 계산
- ✅ **Proper Training**: Size와 Offset이 실제 값 학습

### Stage 2: Segmentation Loss

```python
L_seg = (1/N) Σᵢ [Dice(mask_i) + BCE(mask_i)]
```

- **핵심:** 각 RoI별 독립 계산
- 1000px 병변 = 10px 병변 (동등한 가중치)

### Total Loss

```
L_total = L_det + λ × L_seg
```

(λ = 1.0 또는 2.0)

---

## Training Strategy

### End-to-End 학습

1. 전체 volume → Detection network → RoI proposals
2. 각 RoI crop → Segmentation network → masks
3. L_total로 역전파 → 두 stage 동시 최적화

### 학습 팁

- **HU windowing**: [0, 120] → [0, 1] (뇌출혈 최적화)
- Detection threshold: train 0.3, inference 0.1
- RoI sampling: positive + hard negative
- Data augmentation: flip, rotate, scale
- Mixed precision training

---

## 핵심 장점

✅ **강력한 Feature Extraction** - MONAI ResNet-style backbone with residual connections  
✅ **작은 병변에 강함** - RoI별 독립 loss로 크기 편향 제거  
✅ **메모리 효율적** - 전체 volume dense segmentation 불필요  
✅ **전체 맥락 보존** - Patch 방식과 달리 global detection  
✅ **정밀한 분할** - Enhanced U-Net (4-level + residual units)  
✅ **해석 가능** - RoI + confidence + mask 출력  
✅ **고속 학습** - Multi-GPU + fp16 지원

## 성능 최적화

### Mixed Precision (fp16)
- **메모리 사용량 30-50% 감소** → 더 큰 배치 크기 가능
- **학습 속도 2-3배 향상** (RTX 30xx, A100 등 Tensor Core 지원 GPU)
- 정확도 손실 거의 없음

### Multi-GPU
- DataParallel로 모든 가용 GPU를 **training에** 자동 활용
- Training 배치를 GPU들에 분산하여 처리
- Validation은 메모리 안정성을 위해 single GPU 사용
- 예: 4 GPU, batch_size=2 → training 총 배치 = 8

**권장 설정:**

```bash
# 단일 GPU (16GB)
python main.py --mode train --batch_size 2

# 단일 GPU (16GB) + fp16 → 더 큰 배치
python main.py --mode train --batch_size 4 --fp16

# 4x GPU (16GB each) + fp16
python main.py --mode train \
    --batch_size 2 \      # GPU당 2 → 총 8 (training)
    --val_batch_size 4 \  # 4 (validation, single GPU)
    --fp16 \
    --multi_gpu
```

**효과적인 배치 크기:**

| GPU 구성 | batch_size | 총 배치 | 메모리/GPU | 학습 속도 |
|---------|------------|---------|------------|----------|
| 1 GPU | 2 | 2 | ~6GB | 1x |
| 1 GPU + fp16 | 4 | 4 | ~6GB | 1x |
| 4 GPU | 2 | **8** | ~6GB | **4x** |
| 4 GPU + fp16 | 2 | **8** | ~4GB | **4x** |
| 4 GPU + fp16 | 4 | **16** | ~8GB | **4x** |

**Tip:**
- Multi-GPU 사용 시 `--batch_size`는 **GPU 당 크기**로 설정 (training only)
- Validation은 메모리 안정성을 위해 single GPU에서 실행됨
- `--val_batch_size`를 더 크게 설정하여 validation 속도 향상 가능
- fp16으로 메모리 절약 → batch_size 증가 가능

---

## 프로젝트 구조

```
DetSeg3D/
├── main.py               # 메인 학습/테스트 코드 (all-in-one)
├── inference_example.py  # 단일 이미지 추론 예제
├── visualize.py          # 결과 시각화 도구
├── requirements.txt      # 필요 패키지
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 학습

```bash
python main.py --mode train \
    --image_dir /path/to/train/images \
    --label_dir /path/to/train/labels \
    --output_dir ./outputs \
    --epochs 100 \
    --batch_size 2 \
    --val_split 0.2
```

**고급 옵션 (성능 향상):**

```bash
# Mixed precision (fp16) + Multi-GPU (4 GPUs)
python main.py --mode train \
    --image_dir /path/to/train/images \
    --label_dir /path/to/train/labels \
    --epochs 100 \
    --batch_size 2 \
    --val_batch_size 4 \
    --fp16 \
    --multi_gpu
```

**설정 해석:**
- `--batch_size 2`: **GPU 당** training 배치 크기
- 4 GPU 사용 시 → 실제 총 training 배치: 2 × 4 = **8**
- `--val_batch_size 4`: validation 배치 크기 (single GPU 사용)

**기타 옵션:**
- 이미지/레이블 폴더를 지정하면 자동으로 80/20 train/val split
- `.nii`, `.nii.gz`, `.npy` 형식 지원
- **자동 HU windowing**: 입력 이미지를 [0, 120] HU로 클리핑 후 [0, 1]로 정규화
- `--fp16`: Mixed precision training (메모리 절약 + 속도 향상)
- `--multi_gpu`: 모든 가용 GPU를 training에 사용 (validation은 메모리 안정성을 위해 single GPU)

### 테스트

```bash
python main.py --mode test \
    --test_image_dir /path/to/test/images \
    --test_label_dir /path/to/test/labels \
    --output_dir ./outputs
```

- 원본 이미지 크기로 재구성된 전체 segmentation 저장
- 각 샘플별 Dice score 계산
- RoI 정보 및 confidence 저장
- 결과는 `outputs/predictions/` 폴더에 `.npy` 형식으로 저장

### 단일 이미지 추론

```bash
python inference_example.py \
    --model ./outputs/best_model.pth \
    --image /path/to/image.npy \
    --output prediction.npy
```

### 결과 시각화

```bash
python visualize.py \
    --image /path/to/image.npy \
    --label /path/to/label.npy \
    --pred ./outputs/predictions/pred_0000.npy \
    --output result.png
```

## 핵심 기능

### Adaptive RoI Processing ⭐ **NEW!**

**작은 병변 검출에 최적화된 Adaptive ROI 처리:**

| 병변 크기 | 처리 방법 | 해상도 보존 | 장점 |
|----------|----------|-----------|------|
| 작음 (< 64 voxels) | **원본 크기 유지** | **100%** ✅ | 디테일 손실 없음 |
| 중간 (64-128) | **원본 크기 유지** | **100%** ✅ | 정확한 경계 |
| 큼 (> 128) | Aspect ratio 유지하며 축소 | ~70% | 메모리 효율 |

**핵심 특징:**
- ✅ **Anisotropic 이미지 지원**: (160, 160, 16) → 비율 유지
- ✅ **작은 병변 해상도 완전 보존**: 정보 손실 0%
- ✅ **메모리 효율**: 같은 크기끼리 배치 처리
- ✅ **유연성**: 모든 병변 크기에 대응

### RoI → Full Segmentation 재구성

Stage 1에서 추출된 각 RoI의 segmentation 결과를 원본 이미지 크기로 복원:

1. **Detection**: 전체 볼륨에서 RoI 탐지 (좌표 + confidence)
2. **Adaptive Segmentation**: 각 RoI를 **원본 크기 또는 비율 유지하며 resize**
3. **Reconstruction**: 분할된 RoI를 원래 크기로 복원하여 원본 볼륨에 배치
4. **Merge**: 겹치는 영역은 평균값으로 처리

```python
# 코드 예시
outputs = model(image, mode='test')
full_seg = outputs['full_segmentation']  # 원본 크기
roi_info = outputs['roi_info']            # RoI 정보 (bbox, confidence)
```

## 구현 상태

✅ **CenterNet-style Detection** ⭐ (Proper bbox regression with Gaussian heatmap)  
✅ **Enhanced Detection network** (MONAI ResNet-style backbone)  
✅ **Enhanced Segmentation network** (Deeper U-Net with residual units)  
✅ **Adaptive RoI Processing** ⭐ (작은 병변 해상도 100% 보존)  
✅ **GT BBox extraction** (Connected components from segmentation labels)  
✅ End-to-end pipeline  
✅ Data loading (auto train/val split)  
✅ **RoI → Full segmentation 재구성**  
✅ **Evaluation metrics (Dice score)**  
✅ **결과 저장 및 시각화**  
✅ **Multi-GPU support** (DataParallel)  
✅ **Mixed precision training** (fp16)

---

## 빠른 시작 가이드

### 1. 환경 설정

```bash
git clone <repository>
cd DetSeg3D
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터를 다음과 같은 구조로 준비:

```
data/
├── train/
│   ├── images/
│   │   ├── case001.npy
│   │   ├── case002.npy
│   │   └── ...
│   └── labels/
│       ├── case001.npy
│       ├── case002.npy
│       └── ...
└── test/
    ├── images/
    └── labels/
```

### 3. 학습

```bash
# 단일 GPU
python main.py --mode train \
    --image_dir ./data/train/images \
    --label_dir ./data/train/labels \
    --epochs 100 \
    --batch_size 2

# Multi-GPU (4 GPU 예시)
python main.py --mode train \
    --image_dir ./data/train/images \
    --label_dir ./data/train/labels \
    --epochs 100 \
    --batch_size 2 \  # GPU당 2개 → 총 8
    --fp16 \
    --multi_gpu
```

### 4. 테스트

```bash
python main.py --mode test \
    --test_image_dir ./data/test/images \
    --test_label_dir ./data/test/labels
```

### 5. 결과 확인

```bash
# 예측 결과는 ./outputs/predictions/ 에 저장됨
ls ./outputs/predictions/

# 시각화
python visualize.py \
    --image ./data/test/images/case001.npy \
    --label ./data/test/labels/case001.npy \
    --pred ./outputs/predictions/pred_0000.npy \
    --output result.png
```

---

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | 100 | 학습 에포크 수 |
| `--batch_size` | 2 | **GPU 당** training 배치 크기 |
| `--val_batch_size` | None | validation 배치 크기 (single GPU, None이면 batch_size와 동일) |
| `--lr` | 1e-4 | Learning rate |
| `--roi_size` | 32 | RoI 크기 (32³) |
| `--val_split` | 0.2 | 검증 데이터 비율 |
| `--fp16` | False | Mixed precision (fp16) 사용 |
| `--multi_gpu` | False | 모든 가용 GPU를 training에 사용 (validation은 single GPU) |
| `--max_rois` | 64 | 이미지당 최대 RoI 개수 (OOM 방지) |
| `--val_threshold` | 0.1 | Validation/Test용 detection threshold |
| `--roi_batch_size` | 8 | RoI segmentation 처리 시 mini-batch 크기 (OOM 방지) |
| `--val_interval` | 1 | Validation 실행 간격 (epoch 단위) |
| `--small_roi_threshold` | 64 | 이 크기보다 작은 RoI는 원본 크기 유지 (작은 병변 해상도 보존) |
| `--max_roi_size` | 128 | 큰 RoI의 최대 크기 (aspect ratio 유지하며 resize) |

**Multi-GPU 사용 시:**
- 실제 총 training 배치 크기 = `batch_size × GPU 개수`
- 예: `--batch_size 2 --multi_gpu` (4 GPU) → 총 8 samples/batch

**OOM (Out of Memory) 문제 해결:**

**문제:** Validation 시 GPU 0번만 메모리를 과도하게 사용 (48GB/49GB)

Validation은 single GPU (GPU 0)에서 실행되므로, 다음 파라미터로 메모리를 조절하세요:

1. **`--max_rois`를 줄이기** (기본값: 100)
   ```bash
   --max_rois 50  # 이미지당 최대 50개 RoI만 처리
   ```

2. **`--val_threshold`를 높이기** (기본값: 0.1)
   ```bash
   --val_threshold 0.3  # Confidence가 높은 RoI만 선택
   ```

3. **`--roi_batch_size`를 줄이기** (기본값: 32)
   ```bash
   --roi_batch_size 16  # RoI를 16개씩 처리
   ```

4. **`--val_interval`로 validation 빈도 줄이기** (기본값: 1)
   ```bash
   --val_interval 5  # 5 epoch마다 validation 실행
   ```

**권장 조합 (GPU 메모리 부족 시):**
```bash
# 방법 1: 파라미터 조정 (정확도 유지)
python main.py --mode train \
    --max_rois 50 \
    --val_threshold 0.3 \
    --roi_batch_size 16 \
    --batch_size 1 \
    --fp16 --multi_gpu

# 방법 2: Validation 빈도 줄이기 (빠른 학습)
python main.py --mode train \
    --val_interval 5 \
    --batch_size 1 \
    --fp16 --multi_gpu
```

**효과:**
- GPU 0 메모리: 48GB → ~25GB
- Training 속도: 영향 없음
- Validation 속도: 더 빠름