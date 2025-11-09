#!/usr/bin/env python3
"""
DetSeg3D: End-to-End RoI-based 3D Detection-Segmentation
MONAI 기반 심플 구현
"""

import os
import argparse
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import monai
from monai.networks.nets import UNet, SegResNet
from monai.networks.blocks import Convolution, ResidualUnit
from monai.losses import DiceLoss, FocalLoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityd, ScaleIntensityRanged, RandFlipd, RandRotate90d, EnsureTyped
)
from monai.data import decollate_batch, MetaTensor, pad_list_data_collate
from monai.metrics import DiceMetric
import nibabel as nib

# ============================================================================
# Stage 1: Detection Network (MONAI-based, Enhanced)
# ============================================================================

class DetectionNetwork(nn.Module):
    """
    3D Detection Network with MONAI ResNet backbone
    More powerful feature extraction for better detection
    """
    
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        # Enhanced backbone: ResNet-style with residual connections
        self.init_conv = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=3,
            strides=1,
            norm='batch',
            act='relu'
        )
        
        # Multi-scale encoder with residual blocks
        self.encoder1 = nn.Sequential(
            ResidualUnit(
                spatial_dims=3,
                in_channels=base_channels,
                out_channels=base_channels,
                strides=1,
                kernel_size=3,
                subunits=2,
                act='relu',
                norm='batch'
            ),
            Convolution(3, base_channels, base_channels*2, strides=2, norm='batch', act='relu')  # /2
        )
        
        self.encoder2 = nn.Sequential(
            ResidualUnit(
                spatial_dims=3,
                in_channels=base_channels*2,
                out_channels=base_channels*2,
                strides=1,
                kernel_size=3,
                subunits=2,
                act='relu',
                norm='batch'
            ),
            Convolution(3, base_channels*2, base_channels*4, strides=2, norm='batch', act='relu')  # /4
        )
        
        self.encoder3 = nn.Sequential(
            ResidualUnit(
                spatial_dims=3,
                in_channels=base_channels*4,
                out_channels=base_channels*4,
                strides=1,
                kernel_size=3,
                subunits=2,
                act='relu',
                norm='batch'
            ),
            Convolution(3, base_channels*4, base_channels*8, strides=2, norm='batch', act='relu')  # /8
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            ResidualUnit(
                spatial_dims=3,
                in_channels=base_channels*8,
                out_channels=base_channels*8,
                strides=1,
                kernel_size=3,
                subunits=2,
                act='relu',
                norm='batch'
            )
        )
        
        # Detection heads (anchor-free)
        self.heatmap_head = nn.Sequential(
            Convolution(3, base_channels*8, base_channels*4, kernel_size=3, norm='batch', act='relu'),
            nn.Conv3d(base_channels*4, 1, kernel_size=1)
        )
        
        self.size_head = nn.Sequential(
            Convolution(3, base_channels*8, base_channels*4, kernel_size=3, norm='batch', act='relu'),
            nn.Conv3d(base_channels*4, 3, kernel_size=1)
        )
        
        self.offset_head = nn.Sequential(
            Convolution(3, base_channels*8, base_channels*4, kernel_size=3, norm='batch', act='relu'),
            nn.Conv3d(base_channels*4, 3, kernel_size=1)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        x = self.init_conv(x)
        x = self.encoder1(x)  # /2
        x = self.encoder2(x)  # /4
        x = self.encoder3(x)  # /8
        
        # Feature fusion
        x = self.feature_fusion(x)
        
        # Detection outputs
        heatmap = torch.sigmoid(self.heatmap_head(x))
        size = F.relu(self.size_head(x))  # positive values
        offset = torch.tanh(self.offset_head(x))  # [-1, 1]
        
        return heatmap, size, offset


# ============================================================================
# Stage 2: Segmentation Network (Enhanced U-Net)
# ============================================================================

class SegmentationNetwork(nn.Module):
    """
    Enhanced 3D Segmentation Network for RoI
    More powerful with deeper U-Net and residual connections
    Multi-class segmentation: Class 0 (background), Class 1 (hemorrhage)
    """
    
    def __init__(self, roi_size=32, num_classes=2):
        super().__init__()
        self.roi_size = roi_size
        self.num_classes = num_classes
        
        # Enhanced 3D U-Net with more depth and residual units
        # Using InstanceNorm for stability with small ROIs (BatchNorm fails on 1x1x1 features)
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,  # Multi-class: background + hemorrhage
            channels=(32, 64, 128, 256),  # Deeper network
            strides=(2, 2, 2),
            num_res_units=2,  # More residual units per level
            norm='instance',  # Changed from 'batch' to 'instance' for small ROI stability
            act='relu',
            dropout=0.1,
        )
        
    def forward(self, roi_crops):
        """
        Args:
            roi_crops: (N_rois, 1, D, H, W) - batch of RoI crops
        Returns:
            probs: (N_rois, num_classes, D, H, W) - softmax probabilities
        """
        if roi_crops.shape[0] == 0:
            # Return zero tensor with correct number of classes
            return torch.zeros(roi_crops.shape[0], self.num_classes, *roi_crops.shape[2:], 
                              device=roi_crops.device, dtype=roi_crops.dtype)
        
        logits = self.unet(roi_crops)
        probs = torch.softmax(logits, dim=1)  # Multi-class softmax
        return probs


# ============================================================================
# RoI Reconstruction
# ============================================================================

def reconstruct_segmentation_from_rois(roi_masks, roi_info, original_shape, num_classes=2):
    """
    RoI segmentation 결과를 원본 이미지 크기로 재구성 (Multi-class)
    
    Args:
        roi_masks: List of (1, num_classes, D, H, W) - RoI segmentation probabilities
        roi_info: List of dicts with 'bbox', 'roi_shape' info
        original_shape: (D, H, W) - 원본 볼륨 크기
        num_classes: Number of classes (default: 2 for background + hemorrhage)
    
    Returns:
        full_seg: (1, num_classes, D, H, W) - 재구성된 전체 segmentation probabilities
    """
    if roi_masks is None or len(roi_info) == 0:
        # Return background probability = 1.0
        full_seg = torch.zeros(1, num_classes, *original_shape)
        full_seg[0, 0] = 1.0  # Background class
        return full_seg
    
    D, H, W = original_shape
    # Get device from first mask
    device = roi_masks[0].device if isinstance(roi_masks, list) else roi_masks.device
    
    # 전체 segmentation map 초기화 (multi-class)
    full_seg = torch.zeros(1, num_classes, D, H, W, device=device)
    full_seg[0, 0] = 1.0  # Initialize with background probability = 1.0
    count_map = torch.zeros(1, 1, D, H, W, device=device)  # 겹침 처리용
    
    for i, info in enumerate(roi_info):
        # Get RoI mask (already in appropriate size from adaptive resize)
        roi_mask = roi_masks[i] if isinstance(roi_masks, list) else roi_masks[i:i+1]
        
        # 원래 bbox 크기로 복원
        d_start, h_start, w_start, d_end, h_end, w_end = info['bbox']
        original_size = (d_end - d_start, h_end - h_start, w_end - w_start)
        
        # Resize to original bbox size (all classes)
        resized_mask = F.interpolate(
            roi_mask, 
            size=original_size,
            mode='trilinear',
            align_corners=False
        )
        
        # 원본 볼륨의 해당 위치에 배치 (all classes)
        full_seg[0, :, d_start:d_end, h_start:h_end, w_start:w_end] += resized_mask[0]
        count_map[0, 0, d_start:d_end, h_start:h_end, w_start:w_end] += 1
    
    # 겹치는 영역은 평균 처리
    count_map = torch.clamp(count_map, min=1)
    full_seg = full_seg / count_map
    
    # Re-normalize to sum to 1 (valid probability distribution)
    full_seg = full_seg / full_seg.sum(dim=1, keepdim=True).clamp(min=1e-6)
    
    return full_seg


# ============================================================================
# End-to-End Model
# ============================================================================

class DetSegModel(nn.Module):
    """Two-Stage Detection + Segmentation (Multi-class)"""
    
    def __init__(self, roi_size=32, det_threshold=0.3, max_rois=64, val_threshold=0.1, roi_batch_size=8,
                 small_roi_threshold=64, max_roi_size=128, num_classes=2, min_roi_depth=8):
        super().__init__()
        self.detection_net = DetectionNetwork()
        self.segmentation_net = SegmentationNetwork(roi_size=roi_size, num_classes=num_classes)
        self.roi_size = roi_size
        self.num_classes = num_classes  # Number of segmentation classes
        self.det_threshold = det_threshold  # Training threshold
        self.max_rois = max_rois  # Maximum RoIs per image
        self.val_threshold = val_threshold  # Validation/test threshold
        self.roi_batch_size = roi_batch_size  # Mini-batch size for RoI processing
        self.small_roi_threshold = small_roi_threshold  # Keep original size if smaller
        self.max_roi_size = max_roi_size  # Max size for large ROIs
        self.min_roi_depth = min_roi_depth  # Minimum depth (W) for anisotropic data
        self.return_simple = False  # For multi-GPU validation
        
    def extract_rois_from_heatmap(self, volume, heatmap, size, offset, threshold=0.3, max_rois=100):
        """
        Extract RoI crops from detection heatmap
        
        Args:
            volume: Input volume (B, C, D, H, W)
            heatmap: Detection heatmap (B, 1, D/8, H/8, W/8)
            size: Predicted bbox sizes (B, 3, D/8, H/8, W/8)
            offset: Predicted offsets (B, 3, D/8, H/8, W/8)
            threshold: Confidence threshold
            max_rois: Maximum number of RoIs per batch (for memory control)
        
        Returns:
            rois: (N, 1, roi_size, roi_size, roi_size) or None
            roi_info: List of roi metadata
        """
        B, C, D, H, W = heatmap.shape
        
        rois = []
        roi_info = []
        
        for b in range(B):
            hm = heatmap[b, 0].detach().cpu().numpy()
            peaks = np.where(hm > threshold)
            
            if len(peaks[0]) == 0:
                continue
            
            # Get all peak confidences
            confidences = hm[peaks]
            
            # Sort by confidence and select top-k
            num_peaks = len(peaks[0])
            if num_peaks > max_rois:
                top_k_indices = np.argsort(confidences)[-max_rois:]  # Top max_rois
            else:
                top_k_indices = np.arange(num_peaks)
            
            for idx in top_k_indices:
                i = int(idx)
                d, h, w = peaks[0][i], peaks[1][i], peaks[2][i]
                
                # Get bbox size
                sz = size[b, :, d, h, w].detach().cpu().numpy() * 8  # scale back (stride=8)
                sz = np.clip(sz, 8, 64).astype(int)
                
                # For anisotropic data: ensure minimum depth (W dimension)
                # Lesions may be only 1 slice thick, but we need context for U-Net
                min_depth = self.min_roi_depth  # Minimum depth for stable U-Net processing
                if sz[2] < min_depth:
                    sz[2] = min_depth
                
                # Get center in original space
                stride = 8
                center = np.array([d, h, w]) * stride + offset[b, :, d, h, w].detach().cpu().numpy() * stride
                center = center.astype(int)
                
                # Crop RoI with adjusted size
                d_start = max(0, int(center[0] - sz[0]//2))
                h_start = max(0, int(center[1] - sz[1]//2))
                w_start = max(0, int(center[2] - sz[2]//2))
                
                d_end = min(volume.shape[2], d_start + sz[0])
                h_end = min(volume.shape[3], h_start + sz[1])
                w_end = min(volume.shape[4], w_start + sz[2])
                
                # Adjust if bbox exceeds volume boundary (especially for depth)
                # Ensure minimum size even at boundaries
                if (d_end - d_start) < sz[0] and d_end < volume.shape[2]:
                    d_end = min(volume.shape[2], d_start + sz[0])
                if (h_end - h_start) < sz[1] and h_end < volume.shape[3]:
                    h_end = min(volume.shape[3], h_start + sz[1])
                if (w_end - w_start) < min_depth and w_end < volume.shape[4]:
                    # If depth is too small, expand the ROI
                    needed = min_depth - (w_end - w_start)
                    # Try to expand equally on both sides
                    expand_start = needed // 2
                    expand_end = needed - expand_start
                    w_start = max(0, w_start - expand_start)
                    w_end = min(volume.shape[4], w_end + expand_end)
                    # If still not enough (at boundary), take what we can
                    if (w_end - w_start) < min_depth:
                        if w_start == 0:
                            w_end = min(volume.shape[4], min_depth)
                        elif w_end == volume.shape[4]:
                            w_start = max(0, volume.shape[4] - min_depth)
                
                roi_crop = volume[b:b+1, :, d_start:d_end, h_start:h_end, w_start:w_end]
                
                # Get original ROI size
                orig_d, orig_h, orig_w = roi_crop.shape[2:]
                
                # Adaptive resize: keep small ROIs original, resize large ones
                max_dim = max(orig_d, orig_h, orig_w)
                
                if max_dim < self.small_roi_threshold:
                    # Small ROI: keep original size but pad to U-Net compatible size (8의 배수)
                    # U-Net has 3 levels of downsampling (stride=2), so need multiples of 2^3=8
                    pad_d = ((orig_d + 7) // 8) * 8
                    pad_h = ((orig_h + 7) // 8) * 8
                    pad_w = ((orig_w + 7) // 8) * 8
                    resized_roi = F.interpolate(roi_crop, size=(pad_d, pad_h, pad_w),
                                                mode='trilinear', align_corners=False)
                    final_size = (pad_d, pad_h, pad_w)
                elif max_dim > self.max_roi_size:
                    # Large ROI: resize with aspect ratio preservation to multiples of 8
                    scale = self.max_roi_size / max_dim
                    new_d = max(8, ((int(orig_d * scale) + 7) // 8) * 8)
                    new_h = max(8, ((int(orig_h * scale) + 7) // 8) * 8)
                    new_w = max(8, ((int(orig_w * scale) + 7) // 8) * 8)
                    resized_roi = F.interpolate(roi_crop, size=(new_d, new_h, new_w),
                                                mode='trilinear', align_corners=False)
                    final_size = (new_d, new_h, new_w)
                else:
                    # Medium ROI: pad to multiples of 8
                    pad_d = ((orig_d + 7) // 8) * 8
                    pad_h = ((orig_h + 7) // 8) * 8
                    pad_w = ((orig_w + 7) // 8) * 8
                    resized_roi = F.interpolate(roi_crop, size=(pad_d, pad_h, pad_w),
                                                mode='trilinear', align_corners=False)
                    final_size = (pad_d, pad_h, pad_w)
                
                # Skip ROIs that are too small (will cause BatchNorm issues)
                # Minimum size to avoid 1x1x1 after downsampling
                min_size = 16  # With 3 levels of stride-2 downsampling: 16/2/2/2 = 2
                if final_size[0] < min_size or final_size[1] < min_size or final_size[2] < min_size:
                    continue  # Skip this ROI
                
                # Convert MetaTensor to regular tensor to avoid metadata conflicts
                if hasattr(resized_roi, 'as_tensor'):
                    resized_roi = resized_roi.as_tensor()
                
                rois.append(resized_roi)
                roi_info.append({
                    'batch': b,
                    'center': center,
                    'size': sz,
                    'confidence': float(hm[d, h, w]),
                    'bbox': (d_start, h_start, w_start, d_end, h_end, w_end),
                    'roi_shape': final_size  # Store resized shape
                })
        
        if len(rois) == 0:
            return None, []
        
        # ROIs now have variable sizes, return as list
        # (can't use torch.cat with different sizes)
        return rois, roi_info
    
    def process_variable_size_rois(self, rois):
        """
        Process ROIs with variable sizes by grouping same-size ones together
        
        Args:
            rois: List of (1, 1, D, H, W) tensors with potentially different sizes
        
        Returns:
            masks: List of (1, 1, D, H, W) masks in same order as input
        """
        if not rois:
            return []
        
        # Group ROIs by shape
        shape_groups = {}
        for idx, roi in enumerate(rois):
            shape = roi.shape[2:]  # (D, H, W)
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append((idx, roi))
        
        # Process each group and collect results
        all_masks = [None] * len(rois)
        
        for shape, group in shape_groups.items():
            indices, roi_tensors = zip(*group)
            
            # Stack same-size ROIs into batch
            batch_rois = torch.cat(roi_tensors, dim=0)  # (N, 1, D, H, W)
            
            # Process in mini-batches if too large
            batch_masks_list = []
            for i in range(0, batch_rois.shape[0], self.roi_batch_size):
                mini_batch = batch_rois[i:i+self.roi_batch_size]
                mini_masks = self.segmentation_net(mini_batch)
                batch_masks_list.append(mini_masks)
            
            batch_masks = torch.cat(batch_masks_list, dim=0)
            
            # Distribute masks back to original positions
            for i, idx in enumerate(indices):
                all_masks[idx] = batch_masks[i:i+1]  # Keep as (1, 1, D, H, W)
        
        return all_masks
    
    def forward(self, x, mode=None, return_loss=False, labels=None):
        # Determine mode from model.training if not specified
        if mode is None:
            mode = 'train' if self.training else 'test'
        
        # Stage 1: Detection
        heatmap, size, offset = self.detection_net(x)
        
        if mode == 'train' and return_loss:
            # Training with loss computation (for DataParallel memory efficiency)
            # Compute loss here to avoid gathering large tensors
            det_loss = detection_loss(heatmap, size, offset, labels)
            
            # Segmentation loss - MUST compute ROIs and segment!
            rois, roi_info = self.extract_rois_from_heatmap(
                x, heatmap, size, offset, self.det_threshold, max_rois=self.max_rois
            )
            
            seg_loss = torch.tensor(0.0, device=x.device)
            if rois is not None and len(rois) > 0:
                # Process ROIs through segmentation network
                masks = self.process_variable_size_rois(rois)
                
                # Compute segmentation loss per ROI (variable sizes!)
                seg_losses = []
                for mask, info in zip(masks, roi_info):
                    b = info['batch']
                    bbox = info['bbox']
                    d_start, h_start, w_start, d_end, h_end, w_end = bbox
                    gt_crop = labels[b:b+1, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # Resize GT to match predicted mask size
                    target_size = info['roi_shape']
                    gt_resized = F.interpolate(gt_crop, size=target_size, mode='trilinear', align_corners=False)
                    gt_resized = (gt_resized > 0.5).float()  # Binarize
                    
                    if hasattr(gt_resized, 'as_tensor'):
                        gt_resized = gt_resized.as_tensor()
                    
                    # Compute loss for this ROI
                    roi_loss = segmentation_loss(mask, gt_resized)
                    seg_losses.append(roi_loss)
                
                # Average all ROI losses
                if len(seg_losses) > 0:
                    seg_loss = torch.stack(seg_losses).mean()
            
            total_loss = det_loss + 2.0 * seg_loss
            return total_loss
            
        elif mode == 'train':
            # Training: extract RoIs and segment
            rois, roi_info = self.extract_rois_from_heatmap(
                x, heatmap, size, offset, self.det_threshold, max_rois=self.max_rois
            )
            
            if rois is not None:
                # Process variable-size ROIs (grouped by size for efficient batch processing)
                masks = self.process_variable_size_rois(rois)
            else:
                masks = None
            
            return {
                'heatmap': heatmap,
                'size': size,
                'offset': offset,
                'rois': rois,
                'masks': masks,
                'roi_info': roi_info
            }
        else:
            # Inference: 전체 segmentation 재구성
            rois, roi_info = self.extract_rois_from_heatmap(
                x, heatmap, size, offset, threshold=self.val_threshold, max_rois=self.max_rois
            )
            
            if rois is not None:
                # Process variable-size ROIs (grouped by size for efficient batch processing)
                roi_masks = self.process_variable_size_rois(rois)
                
                # RoI 결과를 원본 크기로 재구성
                original_shape = x.shape[2:]  # (D, H, W)
                full_segmentation = reconstruct_segmentation_from_rois(
                    roi_masks, roi_info, original_shape, num_classes=self.num_classes
                )
            else:
                # No ROIs detected: return background probability = 1.0
                full_segmentation = torch.zeros(x.shape[0], self.num_classes, *x.shape[2:], 
                                               device=x.device, dtype=x.dtype)
                full_segmentation[:, 0] = 1.0  # Background class
            
            # Simple output for multi-GPU validation (controlled by attribute)
            if self.return_simple:
                return full_segmentation  # Tensor only (DataParallel friendly)
            
            return {
                'full_segmentation': full_segmentation,
                'roi_info': roi_info,
                'roi_masks': roi_masks if rois is not None else None
            }


# ============================================================================
# Dataset
# ============================================================================

class MedicalDataset(Dataset):
    """Simple dataset for 3D medical images"""
    
    def __init__(self, image_dir, label_dir, transforms=None, cache=False):
        self.image_files = sorted(glob(os.path.join(image_dir, "*.nii*")) + 
                                  glob(os.path.join(image_dir, "*.npy")))
        self.label_files = sorted(glob(os.path.join(label_dir, "*.nii*")) + 
                                  glob(os.path.join(label_dir, "*.npy")))
        
        assert len(self.image_files) == len(self.label_files), \
            f"Image/Label count mismatch: {len(self.image_files)} vs {len(self.label_files)}"
        
        self.transforms = transforms
        self.cache = cache
        self._cache = {} if cache else None
        
        print(f"Loaded {len(self.image_files)} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]
        
        data = {
            'image': self.image_files[idx],
            'label': self.label_files[idx]
        }
        
        if self.transforms:
            data = self.transforms(data)
        
        if self.cache:
            self._cache[idx] = data
        
        return data


# ============================================================================
# Loss Functions - CenterNet Style
# ============================================================================

def extract_bboxes_from_label(gt_label):
    """
    Extract bounding boxes from segmentation label using connected components
    
    Args:
        gt_label: (B, 1, D, H, W) segmentation mask
    
    Returns:
        batch_bboxes: List of List[dict] - bboxes for each batch item
    """
    import scipy.ndimage as ndimage
    
    batch_size = gt_label.shape[0]
    batch_bboxes = []
    
    for b in range(batch_size):
        label_np = gt_label[b, 0].cpu().numpy()
        
        # Connected components
        labeled, num_features = ndimage.label(label_np > 0)
        
        bboxes = []
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            coords = np.argwhere(mask)
            
            if len(coords) == 0:
                continue
            
            # Get bbox coordinates
            d_min, h_min, w_min = coords.min(axis=0)
            d_max, h_max, w_max = coords.max(axis=0)
            
            # Center and size
            center_d = (d_min + d_max) / 2.0
            center_h = (h_min + h_max) / 2.0
            center_w = (w_min + w_max) / 2.0
            
            size_d = d_max - d_min + 1
            size_h = h_max - h_min + 1
            size_w = w_max - w_min + 1
            
            bboxes.append({
                'center': np.array([center_d, center_h, center_w], dtype=np.float32),
                'size': np.array([size_d, size_h, size_w], dtype=np.float32),
                'bbox': (d_min, h_min, w_min, d_max, h_max, w_max)
            })
        
        batch_bboxes.append(bboxes)
    
    return batch_bboxes


def gaussian_3d(shape, center, sigma=2.0):
    """
    Generate 3D Gaussian kernel
    
    Args:
        shape: (D, H, W)
        center: (d, h, w) center coordinates
        sigma: Gaussian standard deviation
    
    Returns:
        gaussian: (D, H, W) Gaussian heatmap
    """
    D, H, W = shape
    cd, ch, cw = center
    
    d = np.arange(0, D, dtype=np.float32)
    h = np.arange(0, H, dtype=np.float32)
    w = np.arange(0, W, dtype=np.float32)
    
    d = d[:, np.newaxis, np.newaxis]
    h = h[np.newaxis, :, np.newaxis]
    w = w[np.newaxis, np.newaxis, :]
    
    d0 = cd
    h0 = ch
    w0 = cw
    
    gaussian = np.exp(-((d - d0) ** 2 + (h - h0) ** 2 + (w - w0) ** 2) / (2 * sigma ** 2))
    
    return gaussian


def generate_centerdet_targets(batch_bboxes, output_shape, stride=8):
    """
    Generate CenterNet-style targets (heatmap, size, offset)
    
    Args:
        batch_bboxes: List of List[dict] - bboxes for each batch item
        output_shape: (B, D, H, W) output feature map shape
        stride: Downsampling stride (default 8)
    
    Returns:
        gt_heatmap: (B, 1, D, H, W)
        gt_size: (B, 3, D, H, W)
        gt_offset: (B, 3, D, H, W)
    """
    B, D, H, W = output_shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gt_heatmap = torch.zeros(B, 1, D, H, W, dtype=torch.float32, device=device)
    gt_size = torch.zeros(B, 3, D, H, W, dtype=torch.float32, device=device)
    gt_offset = torch.zeros(B, 3, D, H, W, dtype=torch.float32, device=device)
    
    for b, bboxes in enumerate(batch_bboxes):
        for bbox in bboxes:
            center = bbox['center']  # Original resolution
            size = bbox['size']      # Original resolution
            
            # Convert to heatmap resolution
            center_d = center[0] / stride
            center_h = center[1] / stride
            center_w = center[2] / stride
            
            # Integer center for heatmap
            ct_int_d = int(center_d)
            ct_int_h = int(center_h)
            ct_int_w = int(center_w)
            
            # Skip if out of bounds
            if ct_int_d < 0 or ct_int_d >= D or ct_int_h < 0 or ct_int_h >= H or ct_int_w < 0 or ct_int_w >= W:
                continue
            
            # 1. Gaussian heatmap (CenterNet style)
            radius = max(1, int(np.ceil(min(size) / stride / 4)))  # Adaptive radius
            gaussian = gaussian_3d((D, H, W), (center_d, center_h, center_w), sigma=radius)
            gaussian_tensor = torch.from_numpy(gaussian).to(device)
            
            # Max pooling (multiple objects may overlap)
            gt_heatmap[b, 0] = torch.max(gt_heatmap[b, 0], gaussian_tensor)
            
            # 2. Size target (at center point)
            gt_size[b, 0, ct_int_d, ct_int_h, ct_int_w] = float(size[0] / stride)
            gt_size[b, 1, ct_int_d, ct_int_h, ct_int_w] = float(size[1] / stride)
            gt_size[b, 2, ct_int_d, ct_int_h, ct_int_w] = float(size[2] / stride)
            
            # 3. Offset target (sub-pixel refinement)
            gt_offset[b, 0, ct_int_d, ct_int_h, ct_int_w] = float(center_d - ct_int_d)
            gt_offset[b, 1, ct_int_d, ct_int_h, ct_int_w] = float(center_h - ct_int_h)
            gt_offset[b, 2, ct_int_d, ct_int_h, ct_int_w] = float(center_w - ct_int_w)
    
    return gt_heatmap, gt_size, gt_offset


def detection_loss(pred_heatmap, pred_size, pred_offset, gt_label, focal_alpha=2.0, return_stats=False):
    """
    CenterNet-style Detection Loss
    
    Args:
        pred_heatmap: (B, 1, D/8, H/8, W/8) predicted heatmap
        pred_size: (B, 3, D/8, H/8, W/8) predicted bbox size
        pred_offset: (B, 3, D/8, H/8, W/8) predicted offset
        gt_label: (B, 1, D, H, W) ground truth segmentation
        return_stats: if True, return diagnostic statistics
    
    Returns:
        total_loss: scalar
        stats: dict (if return_stats=True)
    """
    # 1. Extract GT bboxes from segmentation label
    batch_bboxes = extract_bboxes_from_label(gt_label)
    
    # Count GT objects
    num_gt_objects = sum(len(bboxes) for bboxes in batch_bboxes)
    
    # 2. Generate CenterNet targets
    B, _, D, H, W = pred_heatmap.shape
    gt_heatmap, gt_size, gt_offset = generate_centerdet_targets(
        batch_bboxes, (B, D, H, W), stride=8
    )
    
    # 3. Heatmap loss (Focal Loss)
    focal_loss = FocalLoss()(pred_heatmap, gt_heatmap)
    
    # 4. Size and offset loss (only at positive locations)
    # Positive mask: where GT heatmap > 0.5 (center points)
    pos_mask = (gt_heatmap > 0.5).float()
    num_pos = pos_mask.sum().clamp(min=1.0)
    
    # Size loss (L1)
    size_loss = F.l1_loss(pred_size * pos_mask, gt_size * pos_mask, reduction='sum') / num_pos
    
    # Offset loss (L1)
    offset_loss = F.l1_loss(pred_offset * pos_mask, gt_offset * pos_mask, reduction='sum') / num_pos
    
    # Total loss
    total_loss = focal_loss + 0.5 * size_loss + 0.1 * offset_loss
    
    if return_stats:
        stats = {
            'focal_loss': focal_loss.item(),
            'size_loss': size_loss.item(),
            'offset_loss': offset_loss.item(),
            'num_gt_objects': num_gt_objects,
            'heatmap_max': pred_heatmap.max().item(),
            'gt_heatmap_max': gt_heatmap.max().item()
        }
        return total_loss, stats
    
    return total_loss


def segmentation_loss(pred_masks, gt_rois):
    """
    Multi-class Segmentation loss for RoIs
    Per-instance loss (핵심!)
    
    Args:
        pred_masks: (N, num_classes, D, H, W) - softmax probabilities
        gt_rois: (N, 1, D, H, W) - ground truth labels (0 or 1)
    """
    if pred_masks is None or gt_rois is None:
        return torch.tensor(0.0, device=pred_masks.device if pred_masks is not None else 'cpu')
    
    # Multi-class Dice Loss (autocast-safe)
    dice_loss = DiceLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=False,  # Already applied softmax in forward
        reduction='mean'
    )(pred_masks, gt_rois.long())
    
    # Cross-Entropy Loss with autocast disabled (fp32 for stability)
    with torch.amp.autocast('cuda', enabled=False):
        pred_fp32 = pred_masks.float()
        gt_fp32 = gt_rois.long().squeeze(1)  # (N, D, H, W)
        ce_loss = F.cross_entropy(pred_fp32, gt_fp32, reduction='mean')
    
    return dice_loss + ce_loss


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, device, epoch, scaler=None, use_fp16=False):
    model.train()
    total_loss = 0
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if use_fp16 and scaler is not None:
            with torch.amp.autocast('cuda'):
                # Forward with loss computation (memory efficient for DataParallel)
                loss = model(images, mode='train', return_loss=True, labels=labels)
                
                # DataParallel returns vector of losses, need to average
                if loss.dim() > 0:
                    loss = loss.mean()
            
            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal training
            # Forward with loss computation
            loss = model(images, mode='train', return_loss=True, labels=labels)
            
            # DataParallel returns vector of losses, need to average
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Backward
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Debugging: print diagnostic info
        if batch_idx == 0:
            print(f"\n[DEBUG Epoch {epoch}] Input shape: {images.shape} (B, C, D, H, W)")
            print(f"[DEBUG] Spatial dimensions: D={images.shape[2]}, H={images.shape[3]}, W={images.shape[4]}")
            print(f"[DEBUG] Expected detection output: ({images.shape[2]//8}, {images.shape[3]//8}, {images.shape[4]//8})")
        
        if batch_idx % 50 == 0 and batch_idx > 0:
            print(f"\n[DIAG Epoch {epoch}, Iter {batch_idx}] Loss: {loss.item():.4f}, Avg: {total_loss/(batch_idx+1):.4f}")
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, device, use_multi_gpu=True):
    """
    Validation with optional multi-GPU support
    
    Args:
        use_multi_gpu: If True and model is DataParallel, use multi-GPU
                       If False, use single GPU (more stable for debugging)
    """
    if use_multi_gpu and isinstance(model, nn.DataParallel):
        # Multi-GPU validation (faster)
        model.eval()
        model_to_use = model
        # Set return_simple flag for DataParallel compatibility
        if hasattr(model, 'module'):
            model.module.return_simple = True
    else:
        # Single GPU validation (stable)
        if isinstance(model, nn.DataParallel):
            model_to_use = model.module
        else:
            model_to_use = model
        model_to_use.eval()
        model_to_use.return_simple = False
    
    dice_metric = DiceMetric(reduction='mean')
    total_rois = 0
    num_batches = 0
    
    try:
        with torch.no_grad():
            pbar = tqdm(loader, desc="[Val]", leave=False)
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                # Inference
                outputs = model_to_use(images)
                
                # Extract segmentation (handle both tensor and dict)
                if isinstance(outputs, dict):
                    full_seg = outputs['full_segmentation']
                    # Count ROIs if available
                    if 'roi_info' in outputs:
                        total_rois += len(outputs['roi_info'])
                else:
                    full_seg = outputs  # Already tensor
                
                # Multi-class output: (B, num_classes, D, H, W) → (B, 1, D, H, W)
                # Get class with highest probability (argmax)
                if full_seg.shape[1] > 1:  # Multi-class
                    pred_class = torch.argmax(full_seg, dim=1, keepdim=True).float()  # (B, 1, D, H, W)
                else:  # Binary (legacy support)
                    pred_class = (full_seg > 0.5).float()
                
                # Calculate Dice (only for foreground class)
                dice_metric(y_pred=pred_class, y=labels)
                num_batches += 1
                
                # Print diagnostic info for first batch
                if batch_idx == 0 and isinstance(outputs, dict) and 'roi_info' in outputs:
                    print(f"\n[VAL DIAG] First batch: {len(outputs['roi_info'])} ROIs detected")
                    if len(outputs['roi_info']) > 0:
                        print(f"[VAL DIAG] Confidence range: {outputs['roi_info'][0]['confidence']:.4f} - {outputs['roi_info'][-1]['confidence']:.4f}")
    finally:
        # Reset return_simple flag
        if isinstance(model, nn.DataParallel):
            if hasattr(model, 'module'):
                model.module.return_simple = False
        else:
            model.return_simple = False
    
    # Aggregate metric
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    # Print validation statistics
    avg_rois = total_rois / max(num_batches, 1)
    print(f"\n[VAL STATS] Avg ROIs per batch: {avg_rois:.1f}, Total: {total_rois}")
    
    return mean_dice


def test_and_save(model, loader, device, output_dir):
    """Test and save segmentation results"""
    # Use single GPU for testing (batch_size=1)
    if isinstance(model, nn.DataParallel):
        model_eval = model.module
    else:
        model_eval = model
    
    model_eval.eval()
    model_eval.return_simple = False  # Return full dict for RoI info
    dice_metric = DiceMetric(reduction='mean')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    results = []
    
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="[Test]")
        for idx, batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Inference
            outputs = model_eval(images)
            full_seg = outputs['full_segmentation']
            roi_info = outputs['roi_info']
            
            # Multi-class output: (B, num_classes, D, H, W) → (B, 1, D, H, W)
            # Get class with highest probability (argmax)
            if full_seg.shape[1] > 1:  # Multi-class
                pred_class = torch.argmax(full_seg, dim=1, keepdim=True).float()  # (B, 1, D, H, W)
            else:  # Binary (legacy support)
                pred_class = (full_seg > 0.5).float()
            
            # Calculate Dice
            dice_metric(y_pred=pred_class, y=labels)
            dice_score = dice_metric.aggregate().item()
            dice_metric.reset()
            
            # Save prediction (class label: 0 or 1)
            pred_np = pred_class[0, 0].cpu().numpy().astype(np.uint8)
            pred_path = os.path.join(output_dir, 'predictions', f'pred_{idx:04d}.npy')
            np.save(pred_path, pred_np)
            
            # Save info
            result_info = {
                'sample_idx': idx,
                'dice': dice_score,
                'num_rois': len(roi_info),
                'roi_confidences': [info['confidence'] for info in roi_info]
            }
            results.append(result_info)
            
            # Update progress bar
            pbar.set_postfix({'dice': f'{dice_score:.4f}', 'rois': len(roi_info)})
    
    # Save summary
    mean_dice = np.mean([r['dice'] for r in results])
    print(f"\n{'='*50}")
    print(f"Test Results Summary")
    print(f"{'='*50}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Num Samples: {len(results)}")
    print(f"Mean RoIs per sample: {np.mean([r['num_rois'] for r in results]):.1f}")
    
    return mean_dice, results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DetSeg3D Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--image_dir', type=str, required=True, help='이미지 폴더')
    parser.add_argument('--label_dir', type=str, required=True, help='레이블 폴더')
    parser.add_argument('--test_image_dir', type=str, default=None)
    parser.add_argument('--test_label_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size (default: same as batch_size)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--roi_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.2, help='검증 데이터 비율')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true', help='Mixed precision training (fp16)')
    parser.add_argument('--multi_gpu', action='store_true', help='Use all available GPUs')
    parser.add_argument('--max_rois', type=int, default=64, help='Maximum number of RoIs to extract per image')
    parser.add_argument('--val_threshold', type=float, default=0.3, help='Detection threshold for validation/test')
    parser.add_argument('--roi_batch_size', type=int, default=8, help='Mini-batch size for RoI segmentation')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (epochs)')
    parser.add_argument('--small_roi_threshold', type=int, default=64, help='Keep ROIs smaller than this threshold at original size')
    parser.add_argument('--max_roi_size', type=int, default=128, help='Maximum ROI size (resize larger ROIs)')
    parser.add_argument('--min_roi_depth', type=int, default=8, help='Minimum ROI depth (W dimension) for anisotropic data')
    args = parser.parse_args()
    
    # Set validation batch size
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transforms
    train_transforms = Compose([
        LoadImaged(keys=['image', 'label'], image_only=False),
        EnsureChannelFirstd(keys=['image', 'label']),
        # HU windowing: clip to [0, 120] and scale to [0, 1]
        ScaleIntensityRanged(
            keys=['image'],
            a_min=0,
            a_max=120,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
        EnsureTyped(keys=['image', 'label']),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label'], image_only=False),
        EnsureChannelFirstd(keys=['image', 'label']),
        # HU windowing: clip to [0, 120] and scale to [0, 1]
        ScaleIntensityRanged(
            keys=['image'],
            a_min=0,
            a_max=120,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=['image', 'label']),
    ])
    
    if args.mode == 'train':
        # Load dataset
        full_dataset = MedicalDataset(args.image_dir, args.label_dir, train_transforms)
        
        # Split train/val
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Calculate effective batch size for multi-GPU
        num_gpus = torch.cuda.device_count() if args.multi_gpu else 1
        effective_batch_size = args.batch_size * num_gpus if args.multi_gpu else args.batch_size
        effective_val_batch_size = args.val_batch_size * num_gpus if args.multi_gpu else args.val_batch_size
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=True, 
            num_workers=4,
            collate_fn=pad_list_data_collate
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=effective_val_batch_size, 
            shuffle=False, 
            num_workers=2,
            collate_fn=pad_list_data_collate
        )
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Model
        model = DetSegModel(
            roi_size=args.roi_size,
            det_threshold=0.3,
            max_rois=args.max_rois,
            val_threshold=args.val_threshold,
            roi_batch_size=args.roi_batch_size,
            small_roi_threshold=args.small_roi_threshold,
            max_roi_size=args.max_roi_size,
            min_roi_depth=args.min_roi_depth
        ).to(args.device)
        
        print(f"\nModel configuration:")
        print(f"  - RoI size (legacy): {args.roi_size}³")
        print(f"  - Adaptive ROI: small <{args.small_roi_threshold} (original), large >{args.max_roi_size} (resized)")
        print(f"  - Max RoIs per image: {args.max_rois}")
        print(f"  - Val/Test threshold: {args.val_threshold}")
        print(f"  - RoI mini-batch size: {args.roi_batch_size}")
        print(f"  - Min ROI depth (W): {args.min_roi_depth} (for anisotropic data)")
        print(f"  - Validation interval: every {args.val_interval} epoch(s)")
        
        # Multi-GPU
        if args.multi_gpu and torch.cuda.device_count() > 1:
            print(f"\nUsing {num_gpus} GPUs with DataParallel")
            print(f"  - Batch size per GPU: {args.batch_size}")
            print(f"  - Total effective batch size: {effective_batch_size}")
            print(f"  - Validation batch size per GPU: {args.val_batch_size}")
            print(f"  - Total validation batch size: {effective_val_batch_size}")
            model = nn.DataParallel(model)
        else:
            print(f"\nUsing single GPU/CPU")
            print(f"  - Batch size: {args.batch_size}")
        
        # Scale learning rate with batch size (optional but recommended)
        # effective_lr = args.lr * (effective_batch_size / args.batch_size)
        # For now, use base lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Mixed Precision
        scaler = None
        if args.fp16:
            if args.device == 'cuda':
                scaler = torch.amp.GradScaler('cuda')
                print("Using mixed precision training (fp16)")
            else:
                print("Warning: fp16 only works with CUDA. Ignoring --fp16 flag.")
                args.fp16 = False
        
        # Training loop
        best_val = 0.0
        print(f"\n{'='*70}")
        print(f"Starting training for {args.epochs} epochs")
        print(f"{'='*70}\n")
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, args.device, epoch, scaler, args.fp16)
            
            # Validation - run every val_interval epochs
            if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
                # Use single GPU for memory stability
                # (multi-GPU validation can cause OOM when gathering large full_segmentation tensors)
                val_score = validate(model, val_loader, args.device, use_multi_gpu=False)
                
                # Set model back to train mode after validation
                model.train()
                
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Dice: {val_score:.4f}", end="")
                
                # Save checkpoint
                if val_score > best_val:
                    best_val = val_score
                    # Save model (handle DataParallel)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                    print(f" | ★ Best!")
                else:
                    print()
            else:
                # Skip validation, only print training loss
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val: skipped")
            
            scheduler.step()
        
        print(f"\n{'='*70}")
        print(f"Training completed! Best Val Dice: {best_val:.4f}")
        print(f"{'='*70}\n")
    
    else:  # test mode
        assert args.test_image_dir and args.test_label_dir, "Test directories required"
        
        test_dataset = MedicalDataset(args.test_image_dir, args.test_label_dir, val_transforms)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=2,
            collate_fn=pad_list_data_collate
        )
        
        # Load model
        model = DetSegModel(
            roi_size=args.roi_size,
            det_threshold=0.3,
            max_rois=args.max_rois,
            val_threshold=args.val_threshold,
            roi_batch_size=args.roi_batch_size,
            small_roi_threshold=args.small_roi_threshold,
            max_roi_size=args.max_roi_size,
            min_roi_depth=args.min_roi_depth
        ).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
        
        print("Testing and saving predictions...")
        test_score, results = test_and_save(model, test_loader, args.device, args.output_dir)
        
        print(f"\nTest completed! Results saved to {args.output_dir}/predictions/")


if __name__ == '__main__':
    main()

