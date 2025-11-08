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

### Stage 1: Detection Network

**입력:** 원본 3D volume (no patches)  
**출력:** RoI coordinates + confidence scores

**구조:**
- Lightweight 3D CNN backbone
- FPN for multi-scale features
- Anchor-free detection heads
  - Center heatmap
  - Bounding box size
  - Center offset

**특징:**
- 메모리 효율적 (~2GB)
- High recall을 위한 low threshold (0.1)
- 3D NMS로 중복 제거

### Stage 2: Segmentation Network

**입력:** 각 RoI crop (예: 32³ resize)  
**출력:** Binary mask per RoI

**구조:**
- Lightweight 3D U-Net (~2M params)
- Optional: context attention layer

**특징:**
- 각 RoI 독립적으로 처리 → 크기 무관 동등 가중치
- Parallel processing 가능
- 작은 병변 확대 효과

---

## Loss Functions

### Stage 1: Detection Loss

```
L_det = Focal(heatmap) + L1(size) + L1(offset)
```

### Stage 2: Segmentation Loss

```
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

- Detection threshold: train 0.3, inference 0.1
- RoI sampling: positive + hard negative
- Data augmentation: flip, rotate, scale
- Mixed precision training

---

## 핵심 장점

✅ **작은 병변에 강함** - RoI별 독립 loss로 크기 편향 제거  
✅ **메모리 효율적** - 전체 volume dense segmentation 불필요  
✅ **전체 맥락 보존** - Patch 방식과 달리 global detection  
✅ **해석 가능** - RoI + confidence + mask 출력

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

- 이미지/레이블 폴더를 지정하면 자동으로 80/20 train/val split
- `.nii`, `.nii.gz`, `.npy` 형식 지원

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

### RoI → Full Segmentation 재구성

Stage 1에서 추출된 각 RoI의 segmentation 결과를 원본 이미지 크기로 복원:

1. **Detection**: 전체 볼륨에서 RoI 탐지 (좌표 + confidence)
2. **Segmentation**: 각 RoI를 32³으로 resize하여 정밀 분할
3. **Reconstruction**: 분할된 RoI를 원래 크기로 복원하여 원본 볼륨에 배치
4. **Merge**: 겹치는 영역은 평균값으로 처리

```python
# 코드 예시
outputs = model(image, mode='test')
full_seg = outputs['full_segmentation']  # 원본 크기
roi_info = outputs['roi_info']            # RoI 정보 (bbox, confidence)
```

## 구현 상태

✅ Detection network (lightweight FPN)  
✅ Segmentation network (lightweight U-Net)  
✅ End-to-end pipeline  
✅ Data loading (auto train/val split)  
✅ **RoI → Full segmentation 재구성**  
✅ **Evaluation metrics (Dice score)**  
✅ **결과 저장 및 시각화**  
⚠️ GT RoI extraction for training (simplified - 개선 필요)

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
python main.py --mode train \
    --image_dir ./data/train/images \
    --label_dir ./data/train/labels \
    --epochs 100 \
    --batch_size 2
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
| `--batch_size` | 2 | 배치 크기 |
| `--lr` | 1e-4 | Learning rate |
| `--roi_size` | 32 | RoI 크기 (32³) |
| `--val_split` | 0.2 | 검증 데이터 비율 |
| `--det_threshold` | 0.3 (train) / 0.1 (test) | Detection threshold |