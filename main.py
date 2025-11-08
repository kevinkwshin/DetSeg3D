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
    """
    
    def __init__(self, roi_size=32):
        super().__init__()
        self.roi_size = roi_size
        
        # Enhanced 3D U-Net with more depth and residual units
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256),  # Deeper network
            strides=(2, 2, 2),
            num_res_units=2,  # More residual units per level
            norm='batch',
            act='relu',
            dropout=0.1,
        )
        
    def forward(self, roi_crops):
        """
        Args:
            roi_crops: (N_rois, 1, D, H, W) - batch of RoI crops
        Returns:
            masks: (N_rois, 1, D, H, W) - segmentation masks
        """
        if roi_crops.shape[0] == 0:
            return torch.zeros_like(roi_crops)
        
        masks = torch.sigmoid(self.unet(roi_crops))
        return masks


# ============================================================================
# RoI Reconstruction
# ============================================================================

def reconstruct_segmentation_from_rois(roi_masks, roi_info, original_shape):
    """
    RoI segmentation 결과를 원본 이미지 크기로 재구성
    
    Args:
        roi_masks: (N_rois, 1, roi_size, roi_size, roi_size) - RoI segmentation masks
        roi_info: List of dicts with 'bbox', 'size' info
        original_shape: (D, H, W) - 원본 볼륨 크기
    
    Returns:
        full_seg: (1, 1, D, H, W) - 재구성된 전체 segmentation
    """
    if roi_masks is None or len(roi_info) == 0:
        return torch.zeros(1, 1, *original_shape)
    
    D, H, W = original_shape
    device = roi_masks.device
    
    # 전체 segmentation map 초기화
    full_seg = torch.zeros(1, 1, D, H, W, device=device)
    count_map = torch.zeros(1, 1, D, H, W, device=device)  # 겹침 처리용
    
    for i, info in enumerate(roi_info):
        # RoI mask를 원래 크기로 resize
        roi_mask = roi_masks[i:i+1]  # (1, 1, roi_size, roi_size, roi_size)
        
        # 원래 bbox 크기로 복원
        d_start, h_start, w_start, d_end, h_end, w_end = info['bbox']
        original_size = (d_end - d_start, h_end - h_start, w_end - w_start)
        
        # Resize to original bbox size
        resized_mask = F.interpolate(
            roi_mask, 
            size=original_size,
            mode='trilinear',
            align_corners=False
        )
        
        # 원본 볼륨의 해당 위치에 배치
        full_seg[0, 0, d_start:d_end, h_start:h_end, w_start:w_end] += resized_mask[0, 0]
        count_map[0, 0, d_start:d_end, h_start:h_end, w_start:w_end] += 1
    
    # 겹치는 영역은 평균 처리
    count_map = torch.clamp(count_map, min=1)
    full_seg = full_seg / count_map
    
    return full_seg


# ============================================================================
# End-to-End Model
# ============================================================================

class DetSegModel(nn.Module):
    """Two-Stage Detection + Segmentation"""
    
    def __init__(self, roi_size=32, det_threshold=0.3, max_rois=100, val_threshold=0.1, roi_batch_size=32):
        super().__init__()
        self.detection_net = DetectionNetwork()
        self.segmentation_net = SegmentationNetwork(roi_size=roi_size)
        self.roi_size = roi_size
        self.det_threshold = det_threshold  # Training threshold
        self.max_rois = max_rois  # Maximum RoIs per image
        self.val_threshold = val_threshold  # Validation/test threshold
        self.roi_batch_size = roi_batch_size  # Mini-batch size for RoI processing
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
                
                # Get center in original space
                stride = 8
                center = np.array([d, h, w]) * stride + offset[b, :, d, h, w].detach().cpu().numpy() * stride
                center = center.astype(int)
                
                # Crop RoI
                d_start = max(0, int(center[0] - sz[0]//2))
                h_start = max(0, int(center[1] - sz[1]//2))
                w_start = max(0, int(center[2] - sz[2]//2))
                
                d_end = min(volume.shape[2], d_start + sz[0])
                h_end = min(volume.shape[3], h_start + sz[1])
                w_end = min(volume.shape[4], w_start + sz[2])
                
                roi_crop = volume[b:b+1, :, d_start:d_end, h_start:h_end, w_start:w_end]
                
                # Resize to fixed size
                roi_crop = F.interpolate(roi_crop, size=(self.roi_size, self.roi_size, self.roi_size),
                                        mode='trilinear', align_corners=False)
                
                # Convert MetaTensor to regular tensor to avoid metadata conflicts
                if hasattr(roi_crop, 'as_tensor'):
                    roi_crop = roi_crop.as_tensor()
                
                rois.append(roi_crop)
                roi_info.append({
                    'batch': b,
                    'center': center,
                    'size': sz,
                    'confidence': float(hm[d, h, w]),
                    'bbox': (d_start, h_start, w_start, d_end, h_end, w_end)
                })
        
        if len(rois) == 0:
            return None, []
        
        rois = torch.cat(rois, dim=0)  # (N, 1, roi_size, roi_size, roi_size)
        return rois, roi_info
    
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
            
            # Segmentation loss (simplified)
            seg_loss = torch.tensor(0.0, device=x.device)
            
            total_loss = det_loss + 2.0 * seg_loss
            return total_loss
            
        elif mode == 'train':
            # Training: extract RoIs and segment
            rois, roi_info = self.extract_rois_from_heatmap(
                x, heatmap, size, offset, self.det_threshold, max_rois=self.max_rois
            )
            
            if rois is not None:
                # Convert MetaTensor to regular Tensor to avoid metadata issues during slicing
                if hasattr(rois, 'as_tensor'):
                    rois = rois.as_tensor()
                
                # Process RoIs in mini-batches for training too
                roi_masks_list = []
                num_rois = rois.shape[0]
                
                for i in range(0, num_rois, self.roi_batch_size):
                    batch_rois = rois[i:i+self.roi_batch_size]
                    batch_masks = self.segmentation_net(batch_rois)
                    roi_masks_list.append(batch_masks)
                
                masks = torch.cat(roi_masks_list, dim=0) if roi_masks_list else None
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
                # Convert MetaTensor to regular Tensor to avoid metadata issues during slicing
                if hasattr(rois, 'as_tensor'):
                    rois = rois.as_tensor()
                
                # Process RoIs in mini-batches to avoid OOM
                roi_masks_list = []
                num_rois = rois.shape[0]
                
                for i in range(0, num_rois, self.roi_batch_size):
                    batch_rois = rois[i:i+self.roi_batch_size]
                    batch_masks = self.segmentation_net(batch_rois)
                    roi_masks_list.append(batch_masks)
                
                roi_masks = torch.cat(roi_masks_list, dim=0)
                
                # RoI 결과를 원본 크기로 재구성
                original_shape = x.shape[2:]  # (D, H, W)
                full_segmentation = reconstruct_segmentation_from_rois(
                    roi_masks, roi_info, original_shape
                )
            else:
                full_segmentation = torch.zeros_like(x)
            
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
# Loss Functions
# ============================================================================

def detection_loss(pred_heatmap, pred_size, pred_offset, gt_label, focal_alpha=2.0):
    """
    Detection loss
    GT generation: create heatmap, size, offset from segmentation label
    """
    # GT heatmap generation (간단한 버전 - 실제로는 Gaussian 사용)
    gt_heatmap = (gt_label > 0).float().max(dim=1, keepdim=True)[0]  # any foreground
    gt_heatmap = F.max_pool3d(gt_heatmap, kernel_size=3, stride=8, padding=1)  # downsample
    
    # Focal loss for heatmap
    focal_loss = FocalLoss()(pred_heatmap, gt_heatmap)
    
    # Size and offset loss (simplified - 실제로는 GT bbox 필요)
    size_loss = F.l1_loss(pred_size, torch.zeros_like(pred_size))
    offset_loss = F.l1_loss(pred_offset, torch.zeros_like(pred_offset))
    
    return focal_loss + 0.1 * size_loss + 0.1 * offset_loss


def segmentation_loss(pred_masks, gt_rois):
    """
    Segmentation loss for RoIs
    Per-instance loss (핵심!)
    """
    if pred_masks is None or gt_rois is None:
        return torch.tensor(0.0)
    
    dice_loss = DiceLoss(sigmoid=False)(pred_masks, gt_rois)
    bce_loss = F.binary_cross_entropy(pred_masks, gt_rois)
    
    return dice_loss + bce_loss


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
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, device, use_multi_gpu=False):
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
    
    try:
        with torch.no_grad():
            pbar = tqdm(loader, desc="[Val]", leave=False)
            for batch in pbar:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                # Inference
                outputs = model_to_use(images)
                
                # Extract segmentation (handle both tensor and dict)
                if isinstance(outputs, dict):
                    full_seg = outputs['full_segmentation']
                else:
                    full_seg = outputs  # Already tensor
                
                # Threshold
                full_seg_binary = (full_seg > 0.5).float()
                
                # Calculate Dice
                dice_metric(y_pred=full_seg_binary, y=labels)
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
            
            # Threshold
            full_seg_binary = (full_seg > 0.5).float()
            
            # Calculate Dice
            dice_metric(y_pred=full_seg_binary, y=labels)
            dice_score = dice_metric.aggregate().item()
            dice_metric.reset()
            
            # Save prediction
            pred_np = full_seg_binary[0, 0].cpu().numpy().astype(np.uint8)
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
    parser.add_argument('--val_batch_size', type=int, default=None, help='Validation batch size (default: same as batch_size)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--roi_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.2, help='검증 데이터 비율')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true', help='Mixed precision training (fp16)')
    parser.add_argument('--multi_gpu', action='store_true', help='Use all available GPUs')
    parser.add_argument('--max_rois', type=int, default=100, help='Maximum number of RoIs to extract per image')
    parser.add_argument('--val_threshold', type=float, default=0.1, help='Detection threshold for validation/test')
    parser.add_argument('--roi_batch_size', type=int, default=32, help='Mini-batch size for RoI segmentation')
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
            roi_batch_size=args.roi_batch_size
        ).to(args.device)
        
        print(f"\nModel configuration:")
        print(f"  - RoI size: {args.roi_size}³")
        print(f"  - Max RoIs per image: {args.max_rois}")
        print(f"  - Val/Test threshold: {args.val_threshold}")
        print(f"  - RoI mini-batch size: {args.roi_batch_size}")
        
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
            
            # Validation - use single GPU for memory stability
            # (multi-GPU validation can cause OOM when gathering large full_segmentation tensors)
            val_score = validate(model, val_loader, args.device, use_multi_gpu=False)
            
            # Set model back to train mode after validation
            model.train()
            
            scheduler.step()
            
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
            roi_batch_size=args.roi_batch_size
        ).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
        
        print("Testing and saving predictions...")
        test_score, results = test_and_save(model, test_loader, args.device, args.output_dir)
        
        print(f"\nTest completed! Results saved to {args.output_dir}/predictions/")


if __name__ == '__main__':
    main()

