"""
Inference Process Simulation
입력 이미지: (512, 512, 30) - Brain CT scan
목표: 전체 flow를 단계별로 추적
"""

import torch
import torch.nn.functional as F
import numpy as np

# ============================================================================
# INPUT
# ============================================================================
print("="*80)
print("STEP 1: INPUT IMAGE")
print("="*80)

# 입력 이미지: (512, 512, 30) → (1, 1, 512, 512, 30) for batch
input_shape = (512, 512, 30)
batch_input = torch.randn(1, 1, 512, 512, 30)  # (B, C, D, H, W) - MONAI format
print(f"Input shape: {batch_input.shape}")
print(f"  - Batch: {batch_input.shape[0]}")
print(f"  - Channels: {batch_input.shape[1]}")
print(f"  - Spatial (D, H, W): {batch_input.shape[2:]}")
print()


# ============================================================================
# STEP 2: DETECTION NETWORK
# ============================================================================
print("="*80)
print("STEP 2: DETECTION NETWORK (Stage 1)")
print("="*80)

# Detection network는 stride=8로 downsampling
stride = 8
det_output_shape = (
    batch_input.shape[2] // stride,  # D: 512/8 = 64
    batch_input.shape[3] // stride,  # H: 512/8 = 64  
    batch_input.shape[4] // stride,  # W: 30/8 = 3.75 → 3 (floor)
)
print(f"Detection output spatial size: {det_output_shape}")

# Detection outputs
heatmap = torch.rand(1, 1, *det_output_shape)  # (1, 1, 64, 64, 3)
size = torch.rand(1, 3, *det_output_shape) * 32  # (1, 3, 64, 64, 3)
offset = torch.rand(1, 3, *det_output_shape) * 2 - 1  # (1, 3, 64, 64, 3)

print(f"Heatmap shape: {heatmap.shape}")
print(f"Size shape: {size.shape}")
print(f"Offset shape: {offset.shape}")
print()


# ============================================================================
# STEP 3: ROI EXTRACTION
# ============================================================================
print("="*80)
print("STEP 3: ROI EXTRACTION")
print("="*80)

# Simulate peak detection (threshold = 0.3)
threshold = 0.3
hm_np = heatmap[0, 0].numpy()
peaks = np.where(hm_np > threshold)
print(f"Detected {len(peaks[0])} peaks above threshold {threshold}")

# 시뮬레이션: 3개 ROI 발견했다고 가정
num_rois = 3
print(f"\nAssuming {num_rois} ROIs detected after filtering")
print()

# 각 ROI의 정보 시뮬레이션
roi_infos = []
for i in range(num_rois):
    # Detection space에서의 위치 (64, 64, 3)
    d_det, h_det, w_det = np.random.randint(0, det_output_shape[0]), \
                           np.random.randint(0, det_output_shape[1]), \
                           np.random.randint(0, det_output_shape[2])
    
    # Original space로 변환 (stride=8)
    center_d = d_det * stride
    center_h = h_det * stride
    center_w = w_det * stride
    
    # ROI size (예시)
    sz_d, sz_h, sz_w = np.random.randint(16, 48), \
                       np.random.randint(16, 48), \
                       np.random.randint(8, 24)
    
    # Bounding box
    d_start = max(0, center_d - sz_d//2)
    h_start = max(0, center_h - sz_h//2)
    w_start = max(0, center_w - sz_w//2)
    d_end = min(input_shape[0], d_start + sz_d)
    h_end = min(input_shape[1], h_start + sz_h)
    w_end = min(input_shape[2], w_start + sz_w)
    
    confidence = np.random.uniform(0.5, 0.9)
    
    roi_infos.append({
        'center': (center_d, center_h, center_w),
        'size': (sz_d, sz_h, sz_w),
        'bbox': (d_start, h_start, w_start, d_end, h_end, w_end),
        'confidence': confidence
    })
    
    print(f"ROI {i+1}:")
    print(f"  Center (D,H,W): {roi_infos[i]['center']}")
    print(f"  Size: {roi_infos[i]['size']}")
    print(f"  BBox: {roi_infos[i]['bbox']}")
    print(f"  Confidence: {roi_infos[i]['confidence']:.4f}")
    print()


# ============================================================================
# STEP 4: ROI CROPPING & ADAPTIVE RESIZE
# ============================================================================
print("="*80)
print("STEP 4: ROI CROPPING & ADAPTIVE RESIZE")
print("="*80)

small_roi_threshold = 64
max_roi_size = 128
min_size = 16  # Filter out too small ROIs

roi_crops = []
for i, info in enumerate(roi_infos):
    d_start, h_start, w_start, d_end, h_end, w_end = info['bbox']
    
    # Crop from original image
    roi_crop = batch_input[0:1, :, d_start:d_end, h_start:h_end, w_start:w_end]
    orig_shape = roi_crop.shape[2:]
    
    print(f"\nROI {i+1} Original crop shape: {orig_shape}")
    
    # Adaptive resize logic
    max_dim = max(orig_shape)
    
    if max_dim < small_roi_threshold:
        # Small ROI: pad to multiple of 8
        pad_d = ((orig_shape[0] + 7) // 8) * 8
        pad_h = ((orig_shape[1] + 7) // 8) * 8
        pad_w = ((orig_shape[2] + 7) // 8) * 8
        final_size = (pad_d, pad_h, pad_w)
        strategy = "Small ROI - pad to 8x"
    elif max_dim > max_roi_size:
        # Large ROI: resize with aspect ratio
        scale = max_roi_size / max_dim
        new_d = max(8, ((int(orig_shape[0] * scale) + 7) // 8) * 8)
        new_h = max(8, ((int(orig_shape[1] * scale) + 7) // 8) * 8)
        new_w = max(8, ((int(orig_shape[2] * scale) + 7) // 8) * 8)
        final_size = (new_d, new_h, new_w)
        strategy = "Large ROI - resize"
    else:
        # Medium ROI: pad to multiple of 8
        pad_d = ((orig_shape[0] + 7) // 8) * 8
        pad_h = ((orig_shape[1] + 7) // 8) * 8
        pad_w = ((orig_shape[2] + 7) // 8) * 8
        final_size = (pad_d, pad_h, pad_w)
        strategy = "Medium ROI - pad to 8x"
    
    print(f"  Strategy: {strategy}")
    print(f"  Final size: {final_size}")
    
    # Check if too small
    if final_size[0] < min_size or final_size[1] < min_size or final_size[2] < min_size:
        print(f"  ⚠️  TOO SMALL! Filtered out (min_size={min_size})")
        continue
    
    resized_roi = F.interpolate(roi_crop, size=final_size, mode='trilinear', align_corners=False)
    roi_crops.append(resized_roi)
    info['roi_shape'] = final_size  # Store for reconstruction
    print(f"  ✅ Processed shape: {resized_roi.shape}")

print(f"\n{len(roi_crops)} ROIs ready for segmentation")
print()


# ============================================================================
# STEP 5: SEGMENTATION NETWORK
# ============================================================================
print("="*80)
print("STEP 5: SEGMENTATION NETWORK (Stage 2)")
print("="*80)

num_classes = 2
roi_masks = []

print(f"Processing {len(roi_crops)} ROIs through U-Net...")
for i, roi_crop in enumerate(roi_crops):
    print(f"\nROI {i+1}:")
    print(f"  Input to U-Net: {roi_crop.shape}")
    
    # Simulate U-Net output (multi-class)
    # Output: (1, num_classes, D, H, W)
    mask = torch.randn(1, num_classes, *roi_crop.shape[2:])
    mask = torch.softmax(mask, dim=1)  # Softmax for multi-class
    
    print(f"  Output from U-Net: {mask.shape}")
    print(f"    - Class 0 (background) prob: {mask[0, 0].mean().item():.4f}")
    print(f"    - Class 1 (hemorrhage) prob: {mask[0, 1].mean().item():.4f}")
    
    roi_masks.append(mask)

print()


# ============================================================================
# STEP 6: RECONSTRUCTION TO ORIGINAL SIZE
# ============================================================================
print("="*80)
print("STEP 6: RECONSTRUCTION TO ORIGINAL SIZE")
print("="*80)

# Initialize full segmentation
D, H, W = input_shape
full_seg = torch.zeros(1, num_classes, D, H, W)
full_seg[0, 0] = 1.0  # Background probability = 1.0
count_map = torch.zeros(1, 1, D, H, W)

print(f"Full segmentation initialized: {full_seg.shape}")
print()

for i, (mask, info) in enumerate(zip(roi_masks, [r for r in roi_infos if 'roi_shape' in r])):
    print(f"Reconstructing ROI {i+1}:")
    
    # Get bbox
    d_start, h_start, w_start, d_end, h_end, w_end = info['bbox']
    original_size = (d_end - d_start, h_end - h_start, w_end - w_start)
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Target bbox size: {original_size}")
    
    # Resize mask back to original bbox size
    resized_mask = F.interpolate(mask, size=original_size, mode='trilinear', align_corners=False)
    print(f"  Resized mask: {resized_mask.shape}")
    
    # Place in full segmentation
    full_seg[0, :, d_start:d_end, h_start:h_end, w_start:w_end] += resized_mask[0]
    count_map[0, 0, d_start:d_end, h_start:h_end, w_start:w_end] += 1
    print(f"  Placed at [{d_start}:{d_end}, {h_start}:{h_end}, {w_start}:{w_end}]")
    print()

# Average overlapping regions
count_map = torch.clamp(count_map, min=1)
full_seg = full_seg / count_map

# Re-normalize
full_seg = full_seg / full_seg.sum(dim=1, keepdim=True).clamp(min=1e-6)

print(f"Final full segmentation: {full_seg.shape}")
print(f"  - Class 0 (background) mean prob: {full_seg[0, 0].mean().item():.4f}")
print(f"  - Class 1 (hemorrhage) mean prob: {full_seg[0, 1].mean().item():.4f}")
print()


# ============================================================================
# STEP 7: FINAL PREDICTION
# ============================================================================
print("="*80)
print("STEP 7: FINAL PREDICTION")
print("="*80)

# Argmax to get class labels
pred_class = torch.argmax(full_seg, dim=1, keepdim=True)  # (1, 1, 512, 512, 30)

print(f"Predicted class labels: {pred_class.shape}")
print(f"  - Unique values: {torch.unique(pred_class)}")
print(f"  - Background voxels (class 0): {(pred_class == 0).sum().item()}")
print(f"  - Hemorrhage voxels (class 1): {(pred_class == 1).sum().item()}")
print()

# Get hemorrhage probability map
hemorrhage_prob = full_seg[0, 1]  # (512, 512, 30)
print(f"Hemorrhage probability map: {hemorrhage_prob.shape}")
print(f"  - Min: {hemorrhage_prob.min().item():.4f}")
print(f"  - Max: {hemorrhage_prob.max().item():.4f}")
print(f"  - Mean: {hemorrhage_prob.mean().item():.4f}")
print()


# ============================================================================
# FINAL OUTPUT SUMMARY
# ============================================================================
print("="*80)
print("FINAL OUTPUT SUMMARY")
print("="*80)

output = {
    'full_segmentation': full_seg,  # (1, 2, 512, 512, 30)
    'roi_info': [r for r in roi_infos if 'roi_shape' in r],
    'roi_masks': roi_masks
}

print(f"\n1. full_segmentation:")
print(f"   Shape: {output['full_segmentation'].shape}")
print(f"   Type: Multi-class probabilities (softmax)")
print(f"   Channel 0: Background")
print(f"   Channel 1: Hemorrhage")
print()

print(f"2. roi_info: (List of {len(output['roi_info'])} ROIs)")
for i, roi in enumerate(output['roi_info']):
    print(f"   ROI {i+1}:")
    print(f"     - Center: {roi['center']}")
    print(f"     - Confidence: {roi['confidence']:.4f}")
    print(f"     - Size: {roi['size']}")
print()

print(f"3. roi_masks: (List of {len(output['roi_masks'])} tensors)")
for i, mask in enumerate(output['roi_masks']):
    print(f"   Mask {i+1}: {mask.shape}")
print()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
print("="*80)
print("USAGE EXAMPLES")
print("="*80)

print("\n# Get final binary segmentation:")
print("pred_binary = torch.argmax(full_seg, dim=1)[0]  # (512, 512, 30)")
pred_binary = torch.argmax(full_seg, dim=1)[0]
print(f"Result shape: {pred_binary.shape}")
print()

print("# Get hemorrhage probability:")
print("hemorrhage_prob = full_seg[0, 1]  # (512, 512, 30)")
print(f"Result shape: {hemorrhage_prob.shape}")
print()

print("# High-confidence ROIs only:")
high_conf_rois = [r for r in output['roi_info'] if r['confidence'] > 0.7]
print(f"Found {len(high_conf_rois)} high-confidence ROIs (> 0.7)")
print()


# ============================================================================
# POTENTIAL ISSUES TO DISCUSS
# ============================================================================
print("="*80)
print("⚠️  POTENTIAL ISSUES TO DISCUSS")
print("="*80)

print("\n1. Anisotropic Input (512, 512, 30):")
print("   - D=512, H=512, W=30 ← W dimension is very small!")
print("   - Detection output: (64, 64, 3) ← Only 3 slices in W!")
print("   - 문제: W 방향으로 해상도가 너무 낮음")
print("   - 해결책: Input을 (D, H, W) = (30, 512, 512)로 변경?")
print()

print("2. ROI Size vs Input Anisotropy:")
print("   - ROI 크기가 모든 방향으로 동일하게 추출되는가?")
print("   - Anisotropic spacing을 고려해야 하는가?")
print()

print("3. Detection Stride=8:")
print("   - W=30 → W/8=3.75 → 3 (floor)")
print("   - 정보 손실이 큰가?")
print()

print("4. Small ROI Filtering (min_size=16):")
print("   - W 방향으로 작은 병변이 필터링될 수 있음")
print("   - 적절한가?")
print()

print("="*80)
print("SIMULATION COMPLETE!")
print("="*80)

