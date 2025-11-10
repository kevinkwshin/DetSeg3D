# EDA for 3D Detection Dataset

## 목적

데이터셋의 lesion 크기를 분석하여 최적의 anchor shapes를 자동으로 생성합니다.

## 사용 방법

### 1. EDA 실행

```bash
# 기본 경로로 실행 (20mm 이내 lesion 통합)
./run_eda.sh

# 또는 직접 실행
python eda_dataset.py \
    --image_dir /path/to/images \
    --label_dir /path/to/masks \
    --output_dir ./eda \
    --min_size 10 \
    --merge_distance_mm 20.0

# Merge 기능 비활성화
python eda_dataset.py \
    --label_dir /path/to/masks \
    --merge_distance_mm 0
```

### 2. 결과 확인

```bash
# JSON 결과 확인
cat ./eda/dataset.json

# 또는 pretty print
python -m json.tool ./eda/dataset.json
```

### 3. 학습 시 자동 적용

`./train_ddp.sh`를 실행하면 `./eda/dataset.json`이 있을 경우 자동으로 anchor shapes를 로드합니다.

```bash
./train_ddp.sh
```

학습 시작 시 다음과 같은 메시지가 표시됩니다:
```
🎯 Loaded anchor shapes from EDA:
   EDA file: ./eda/dataset.json
   Original sizes (pixels):
      0: [30, 40, 3]
      1: [80, 90, 5]
      2: [120, 140, 8]
      3: [200, 220, 12]
   Feature map anchors (stride=4):
      0: [7, 10, 1]
      1: [20, 22, 1]
      2: [30, 35, 2]
      3: [50, 55, 3]
```

## 출력 결과

`./eda/dataset.json` 파일에는 다음 정보가 포함됩니다:

### 1. Dataset 정보
- 전체 파일 수
- Lesion이 있는 파일 수
- 전체 lesion 개수

### 2. Box 크기 통계
각 차원(width, height, depth)에 대해:
- min, max, mean, median, std
- percentiles (p10, p25, p50, p75, p90, p95)

### 3. Volume 통계
- Lesion의 전체 voxel 수 통계

### 4. Aspect Ratios
- width/height, width/depth, height/depth 평균

### 5. 추천 Anchor Shapes
- Small (p25): 작은 lesion 대응
- Medium (p50): 중간 크기 lesion 대응
- Large (p75): 큰 lesion 대응
- Very Large (p90): 매우 큰 lesion 대응

## 예시 출력

```json
{
  "dataset_info": {
    "num_files": 747,
    "num_files_with_lesions": 747,
    "total_lesions": 2241,
    "min_lesion_size": 10
  },
  "box_sizes": {
    "width": {
      "min": 5,
      "max": 300,
      "mean": 85.3,
      "median": 75.0,
      "percentiles": {
        "p25": 50.0,
        "p50": 75.0,
        "p75": 110.0,
        "p90": 150.0
      }
    },
    ...
  },
  "recommended_anchors": {
    "anchor_shapes": [
      [50, 55, 3],
      [75, 80, 5],
      [110, 120, 8],
      [150, 160, 12]
    ],
    "description": [
      "Small (p25)",
      "Medium (p50)",
      "Large (p75)",
      "Very Large (p90)"
    ]
  }
}
```

## 참고사항

- **Merge Distance**: `--merge_distance_mm` 옵션으로 물리적으로 가까운 lesion들을 하나의 box로 통합합니다 (기본값: 20mm)
  - Morphological closing (dilation → erosion) 사용
  - NIfTI header의 voxel spacing 정보를 사용하여 물리적 거리 계산
  - 0으로 설정하면 merge 기능 비활성화
- **Feature Map Stride**: 기본값은 4입니다 (ResNet FPN의 첫 번째 layer stride)
- **Anchor 개수**: 기본적으로 4개의 anchor shapes 생성 (small, medium, large, very large)
- **최소 크기**: `--min_size` 옵션으로 너무 작은 lesion 제외 (기본값: 10 voxels)

## Troubleshooting

### EDA가 자동으로 로드되지 않는 경우

1. 파일 경로 확인:
```bash
ls -la ./eda/dataset.json
```

2. JSON 형식 확인:
```bash
python -m json.tool ./eda/dataset.json
```

3. 수동으로 anchor shapes 확인:
```python
import json
with open('./eda/dataset.json', 'r') as f:
    data = json.load(f)
print(data['recommended_anchors']['anchor_shapes'])
```

### 기본 anchor shapes 사용하는 경우

EDA 파일이 없으면 다음 기본값을 사용합니다:
```python
base_anchor_shapes = [
    [30, 30, 3],   # Small variant
    [40, 40, 4],   # Median lesion
    [50, 50, 5],   # Large variant
]
```

학습 시 다음 메시지가 표시됩니다:
```
📌 Using default anchor shapes (run eda_dataset.py to optimize)
```

