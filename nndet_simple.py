#!/usr/bin/env python3
"""
3D Detection with MONAI RetinaNet
Based on MONAI detection module examples
"""

import os
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import DataLoader, Dataset, box_utils, decollate_batch
from monai.data.utils import no_collation
from monai.networks.nets import resnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    RandCropByPosNegLabeld, ScaleIntensityRanged, RandFlipd, RandRotate90d, RandZoomd,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandAdjustContrastd, EnsureTyped,
    DeleteItemsd, MapTransform, CopyItemsd
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    ConvertBoxModed,
    StandardizeEmptyBoxd,
)
from monai.transforms.utility.dictionary import ApplyTransformToPointsd
from monai.transforms.spatial.dictionary import ConvertBoxToPointsd, ConvertPointsToBoxesd
from monai.config import KeysCollection
from monai.utils.type_conversion import convert_data_type
from monai.data.box_utils import clip_boxes_to_image
from monai.apps.detection.transforms.box_ops import convert_box_to_mask
import scipy.ndimage as ndimage
import nibabel as nib


# ============================================================================
# Helper: Extract bboxes from segmentation mask
# ============================================================================

def extract_bboxes_from_mask(mask, min_size=10, merge_distance_mm=20.0, spacing=None, debug=False):
    """
    Extract bounding boxes from binary segmentation mask using connected components
    
    Pipeline:
        1. Binary mask creation
        2. Initial cluster analysis (connected components)
        3. Merge nearby clusters (morphological closing)
        4. Filter by minimum size (remove small merged clusters)
        5. Final cluster analysis
        6. Box extraction
    
    Args:
        mask: (H, W, D) binary mask tensor or array
        min_size: minimum voxel count for valid box AFTER merging
        merge_distance_mm: merge lesions within this physical distance (mm). Set to 0 to disable.
        spacing: (3,) array of voxel spacing in mm [sx, sy, sz]. If None, assumes [1, 1, 1]
        debug: print debug information
        
    Returns:
        boxes: (N, 6) array in xyzxyz format (x1, y1, z1, x2, y2, z2)
        labels: (N,) array of labels (all 0 for foreground)
    """
    # Convert to numpy and ensure 3D
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
    if mask_np.ndim == 4:
        mask_np = mask_np[0]
    
    if debug:
        print(f"\n🔍 extract_bboxes_from_mask DEBUG:")
        print(f"   Mask shape: {mask_np.shape}")
        print(f"   Non-zero voxels: {np.sum(mask_np > 0)}")
        print(f"   Merge distance: {merge_distance_mm} mm")
    
    # Step 1: Binary mask creation
    binary_mask = (mask_np > 0).astype(np.uint8)
    
    # Step 2: Initial cluster analysis (connected components)
    # This identifies individual lesions before merging
    initial_labeled, initial_num = ndimage.label(binary_mask)
    
    if debug:
        print(f"   Initial components (before merge): {initial_num}")
    
    # Step 3: Merge nearby clusters (if requested)
    if merge_distance_mm > 0 and spacing is not None:
        # Calculate dilation distance in voxels for each dimension
        # Use half of merge_distance_mm for dilation (the other half comes from the other lesion)
        spacing_array = np.array(spacing) if spacing is not None else np.array([1.0, 1.0, 1.0])
        dilation_voxels = np.round(merge_distance_mm / (2.0 * spacing_array)).astype(int)
        dilation_voxels = np.maximum(dilation_voxels, 1)  # At least 1 voxel
        
        if debug:
            print(f"   Voxel spacing: {spacing_array}")
            print(f"   Dilation distance (voxels): {dilation_voxels}")
        
        # CRITICAL FIX: Use TRUE anisotropic dilation (different iterations per axis)
        # Depth spacing is usually much larger (e.g., 4.8mm) than X/Y spacing (e.g., 0.4mm)
        # So we need: X=13 iterations, Y=13 iterations, Z=1 iteration
        from scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
        
        # Anisotropic morphological closing: dilate each axis separately
        mask_dilated = binary_mask.copy()
        
        # Dilate along each axis with its specific iteration count
        for axis in range(3):
            iterations = int(dilation_voxels[axis])
            if iterations > 0:
                # Create 1D structuring element for this axis only
                # Shape: [1, 1, 1] but with 3 elements along the target axis
                struct_shape = [1, 1, 1]
                struct_shape[axis] = 3
                struct_1d = np.zeros(struct_shape, dtype=bool)
                # Set center and neighbors along this axis to True
                if axis == 0:  # Height axis
                    struct_1d[0:3, 0, 0] = True
                elif axis == 1:  # Width axis
                    struct_1d[0, 0:3, 0] = True
                else:  # Depth axis
                    struct_1d[0, 0, 0:3] = True
                
                # Apply dilation along this axis
                for _ in range(iterations):
                    mask_dilated = binary_dilation(mask_dilated, structure=struct_1d)
        
        # Erosion: use ANISOTROPIC erosion (same as dilation but reversed)
        # To preserve small lesions, we erode each axis separately with 50% of dilation iterations
        mask_closed = mask_dilated.copy()
        
        for axis in range(3):
            iterations = max(1, int(dilation_voxels[axis] * 0.5))  # 50% of dilation
            if iterations > 0:
                # Create 1D structuring element for this axis only
                struct_shape = [1, 1, 1]
                struct_shape[axis] = 3
                struct_1d = np.zeros(struct_shape, dtype=bool)
                # Set center and neighbors along this axis to True
                if axis == 0:  # Height axis
                    struct_1d[0:3, 0, 0] = True
                elif axis == 1:  # Width axis
                    struct_1d[0, 0:3, 0] = True
                else:  # Depth axis
                    struct_1d[0, 0, 0:3] = True
                
                # Apply erosion along this axis
                for _ in range(iterations):
                    mask_closed = binary_erosion(mask_closed, structure=struct_1d)
        
        if debug:
            erosion_per_axis = [max(1, int(dilation_voxels[i] * 0.5)) for i in range(3)]
            print(f"   Dilation iterations per axis: H={int(dilation_voxels[0])}, W={int(dilation_voxels[1])}, D={int(dilation_voxels[2])}")
            print(f"   Erosion iterations per axis:  H={erosion_per_axis[0]}, W={erosion_per_axis[1]}, D={erosion_per_axis[2]} (50% of dilation)")
            components_before = ndimage.label(binary_mask)[1]
            components_after = ndimage.label(mask_closed)[1]
            nonzero_before = np.sum(binary_mask > 0)
            nonzero_after = np.sum(mask_closed > 0)
            print(f"   Non-zero voxels: {nonzero_before} → {nonzero_after}")
            print(f"   Components: {components_before} → {components_after}")
        
        # Use merged mask for further processing
        binary_mask = mask_closed.astype(np.uint8)
    
    # Step 4 & 5: Final cluster analysis and size filtering
    labeled_mask, num_components = ndimage.label(binary_mask)
    
    if debug:
        print(f"   Final components (after merge): {num_components}")
    
    boxes = []
    labels = []
    
    # Step 6: Box extraction with size filtering
    for component_id in range(1, num_components + 1):
        component_mask = (labeled_mask == component_id)
        voxel_count = component_mask.sum()
        
        # Step 4: Filter by minimum voxel count AFTER merging
        # Small clusters that merged together can now pass the threshold
        if voxel_count < min_size:
            continue
            
        # Get bounding box coordinates
        coords = np.argwhere(component_mask)  # Returns (N, 3) with indices [dim0, dim1, dim2]
        if len(coords) == 0:
            continue
            
        # coords is (N, 3) with order matching np.argwhere output
        # For 3D array with shape (H, W, D):
        #   coords[:, 0] = H indices (Height, Y axis)
        #   coords[:, 1] = W indices (Width, X axis)
        #   coords[:, 2] = D indices (Depth, Z axis)
        h_coords = coords[:, 0]  # Height (Y)
        w_coords = coords[:, 1]  # Width (X)
        d_coords = coords[:, 2]  # Depth (Z)
        
        # MONAI expects boxes as (x1, y1, z1, x2, y2, z2) where x=W, y=H, z=D
        x1 = float(w_coords.min())
        x2 = float(w_coords.max() + 1)  # +1 for exclusive end
        y1 = float(h_coords.min())
        y2 = float(h_coords.max() + 1)
        z1 = float(d_coords.min())
        z2 = float(d_coords.max() + 1)
        
        # Calculate box dimensions for debug output
        width = x2 - x1
        height = y2 - y1
        depth = z2 - z1
        
        if debug and component_id == 1:
            print(f"\n   Component {component_id} (voxels: {voxel_count}):")
            print(f"      H coords: {h_coords.min()}-{h_coords.max()}")
            print(f"      W coords: {w_coords.min()}-{w_coords.max()}")
            print(f"      D coords: {d_coords.min()}-{d_coords.max()}")
            print(f"      Box (xyzxyz): [{x1:.1f}, {y1:.1f}, {z1:.1f}, {x2:.1f}, {y2:.1f}, {z2:.1f}]")
            print(f"      Box size (WxHxD): {width:.1f} x {height:.1f} x {depth:.1f}")
            print(f"      Voxel count: {voxel_count} (min_size: {min_size})")
        
        boxes.append([x1, y1, z1, x2, y2, z2])
        labels.append(0)  # Foreground class
    
    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


# ============================================================================
# Custom Transform: Generate box mask for cropping
# ============================================================================

class GenerateBoxMaskd(MapTransform):
    """Generate box mask from segmentation for positive/negative cropping"""
    
    def __init__(
        self,
        keys: KeysCollection,
        image_key: str,
        label_key: str,
        box_key: str = "box",
        label_class_key: str = "label",
        mask_key: str = "mask_image",
        min_size: int = 10,
        merge_distance_mm: float = 20.0,
        debug: bool = False,
    ):
        super().__init__(keys)
        self.image_key = image_key
        self.label_key = label_key
        self.box_key = box_key
        self.label_class_key = label_class_key
        self.mask_key = mask_key
        self.min_size = min_size
        self.merge_distance_mm = merge_distance_mm
        self.debug = debug
        self.call_count = 0
    
    def __call__(self, data):
        d = dict(data)
        
        # Extract boxes from segmentation mask
        seg_mask = d[self.label_key][0] if d[self.label_key].ndim == 4 else d[self.label_key]
        
        # Get voxel spacing from metadata if available
        spacing = None
        meta_key = f"{self.label_key}_meta_dict"
        if meta_key in d:
            meta = d[meta_key]
            # Try to get spacing from metadata
            if "pixdim" in meta:
                # NIfTI pixdim: [qfac, sx, sy, sz, ...]
                pixdim = meta["pixdim"]
                if torch.is_tensor(pixdim):
                    pixdim = pixdim.cpu().numpy()
                spacing = pixdim[1:4]  # [sx, sy, sz]
            elif "spacing" in meta:
                spacing = meta["spacing"]
                if torch.is_tensor(spacing):
                    spacing = spacing.cpu().numpy()
        
        # Debug only first sample
        debug_this = self.debug and self.call_count == 0
        self.call_count += 1
        
        boxes, labels = extract_bboxes_from_mask(
            seg_mask, 
            min_size=self.min_size,
            merge_distance_mm=self.merge_distance_mm,
            spacing=spacing,
            debug=debug_this
        )
        
        # Store boxes and labels
        d[self.box_key] = torch.tensor(boxes, dtype=torch.float32)
        d[self.label_class_key] = torch.tensor(labels, dtype=torch.long)
        
        # Generate mask for cropping (just use the original segmentation)
        d[self.mask_key] = (d[self.label_key] > 0).float()
        
        return d


# ============================================================================
# Dataset
# ============================================================================

class MedicalDetectionDataset(Dataset):
    """Dataset for medical image detection from image/label folders"""
    
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_files = sorted(glob(os.path.join(image_dir, "*.nii.gz")))
        self.label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
        
        assert len(self.image_files) == len(self.label_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.label_files)} labels"
        
        self.data_dicts = [
            {"image": img, "label": lbl}
            for img, lbl in zip(self.image_files, self.label_files)
        ]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data_dicts)
    
    def __getitem__(self, idx):
        data = self.data_dicts[idx]
        if self.transform:
            data = self.transform(data)
        return data


# ============================================================================
# Transforms
# ============================================================================

def generate_train_transform(patch_size, batch_size, amp=True, debug=False, min_size=10, merge_distance_mm=20.0, intensity_min=0, intensity_max=120):
    """
    Generate training transform for detection
    
    Uses RandCropByPosNegLabeld to ensure balanced sampling:
    - 50% patches contain lesions (positive)
    - 50% patches are background (negative)
    
    This is much more efficient than random cropping for sparse lesion detection!
    """
    
    compute_dtype = torch.float16 if amp else torch.float32
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Generate box mask for positive/negative sampling
        # This creates "mask_image" which indicates where lesions are
        GenerateBoxMaskd(
            keys=["label"],
            image_key="image",
            label_key="label",
            box_key="box_temp",  # Temporary, will regenerate after augmentation
            label_class_key="label_class_temp",
            mask_key="mask_image",  # This is used by RandCropByPosNegLabeld
            min_size=min_size,
            merge_distance_mm=merge_distance_mm,  # Use config value (consistent with validation)
            debug=debug,
        ),
        # Positive/Negative crop for detection (like original detection code)
        # This ensures 50% patches contain lesions, 50% are background
        # Much more efficient for training than random cropping!
        RandCropByPosNegLabeld(
            keys=["image", "label"],  # Crop both image and label (segmentation mask)
            label_key="mask_image",   # Use the mask generated by GenerateBoxMaskd to decide pos/neg
            spatial_size=patch_size,
            num_samples=batch_size,
            pos=1,  # 1 positive sample (with lesion)
            neg=1,  # 1 negative sample (background)
        ),
        # Spatial augmentation (apply to both image and label)
        # Label must be transformed together with image since we extract boxes from label later
        RandZoomd(
            keys=["image", "label"],
            prob=0.2,
            min_zoom=0.9,
            max_zoom=1.1,
            padding_mode="constant",
            keep_size=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        # Intensity augmentation (image only)
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0, std=0.1),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.1,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys=["image"], prob=0.15, factors=0.25),
        RandShiftIntensityd(keys=["image"], prob=0.15, offsets=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        # Delete temporary boxes from before augmentation
        DeleteItemsd(keys=["box_temp", "label_class_temp"]),
        # After augmentation, extract final boxes from augmented label
        GenerateBoxMaskd(
            keys=["label"],
            image_key="image",
            label_key="label",
            box_key="box",
            label_class_key="label_class",
            mask_key="mask_image_final",  # Different name to avoid confusion
            min_size=min_size,
            merge_distance_mm=merge_distance_mm,  # Use config value (consistent with validation)
            debug=debug,
        ),
        # Note: Boxes extracted from mask are already in image coordinates (pixel indices)
        # No need for AffineBoxToImageCoordinated transform
        StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
        # Final type conversion
        EnsureTyped(keys=["image"], dtype=compute_dtype),
        EnsureTyped(keys=["label_class"], dtype=torch.long),
        # Clean up
        DeleteItemsd(keys=["label", "mask_image", "mask_image_final"]),
    ])
    
    return train_transforms


def generate_val_transform(amp=True, debug=False, min_size=10, merge_distance_mm=20.0, intensity_min=0, intensity_max=120):
    """Generate validation transform"""
    
    compute_dtype = torch.float16 if amp else torch.float32
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Copy label before processing for debugging
        CopyItemsd(keys=["label"], times=1, names=["seg_label"]),
        # Extract boxes from segmentation mask
        GenerateBoxMaskd(
            keys=["label"],
            image_key="image",
            label_key="label",
            box_key="box",
            label_class_key="label_class",
            mask_key="mask_image",
            min_size=min_size,
            merge_distance_mm=merge_distance_mm,
            debug=debug,  # Enable debug only when --debug flag is given
        ),
        # Note: Boxes extracted from mask are already in image coordinates (pixel indices)
        # No need for AffineBoxToImageCoordinated transform
        StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
        EnsureTyped(keys=["image"], dtype=compute_dtype),
        EnsureTyped(keys=["label_class"], dtype=torch.long),
        # Keep seg_label for debugging, delete others
        DeleteItemsd(keys=["label", "mask_image"]),
    ])
    
    return val_transforms


def generate_test_transform(amp=True, intensity_min=0, intensity_max=120):
    """Generate test/inference transform (image only, no labels required)"""
    
    compute_dtype = torch.float16 if amp else torch.float32
    
    test_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        # Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"], dtype=compute_dtype),
    ])
    
    return test_transforms


# ============================================================================
# Training
# ============================================================================

def train_epoch(detector, loader, optimizer, device, epoch, scaler=None, amp=True, verbose=True):
    """Train one epoch"""
    detector.train()
    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_box_loss = 0
    
    # Only show progress bar on main process to avoid DDP conflicts
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", disable=not verbose)
    for batch_idx, batch_data in enumerate(pbar):
        # Prepare inputs and targets
        # RandCropByPosNegLabeld with num_samples > 1 returns nested list
        # Flatten: [[{dict1}, {dict2}]] -> [{dict1}, {dict2}]
        if isinstance(batch_data[0], list):
            batch_data = [item for sublist in batch_data for item in sublist]
        
        inputs = [data_i["image"].to(device) for data_i in batch_data]
        targets = [
            dict(
                label_class=data_i["label_class"].to(device),  # Must match set_target_keys()
                box=data_i["box"].to(device),
            )
            for data_i in batch_data
        ]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward + backward
        if amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = detector(inputs, targets)
                # Get loss keys (handle DataParallel wrapping)
                det_module = detector.module if hasattr(detector, 'module') else detector
                loss = outputs[det_module.cls_key] + outputs[det_module.box_reg_key]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = detector(inputs, targets)
            det_module = detector.module if hasattr(detector, 'module') else detector
            loss = outputs[det_module.cls_key] + outputs[det_module.box_reg_key]
            loss.backward()
            optimizer.step()
        
        # Stats
        epoch_loss += loss.detach().item()
        epoch_cls_loss += outputs[det_module.cls_key].detach().item()
        epoch_box_loss += outputs[det_module.box_reg_key].detach().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{outputs[det_module.cls_key].item():.4f}',
            'box': f'{outputs[det_module.box_reg_key].item():.4f}'
        })
    
    epoch_loss /= (batch_idx + 1)
    epoch_cls_loss /= (batch_idx + 1)
    epoch_box_loss /= (batch_idx + 1)
    
    return epoch_loss, epoch_cls_loss, epoch_box_loss


def boxes_to_mask(boxes, image_shape):
    """
    Convert boxes to binary mask for visualization
    
    Args:
        boxes: (N, 6) numpy array in xyzxyz format (x1, y1, z1, x2, y2, z2)
        image_shape: tuple of (dim0, dim1, dim2)
    
    Returns:
        mask: binary mask with same shape as image_shape
    """
    if len(boxes) == 0:
        spatial_shape = image_shape[-3:] if len(image_shape) == 4 else image_shape
        return np.zeros(spatial_shape, dtype=np.uint8)
    
    # Get spatial shape
    spatial_shape = image_shape[-3:] if len(image_shape) == 4 else image_shape
    mask = np.zeros(spatial_shape, dtype=np.uint8)
    
    # Fill each box
    for box in boxes:
        x1, y1, z1, x2, y2, z2 = box
        
        # Box format: [x1, y1, z1, x2, y2, z2] where x=W, y=H, z=D
        # spatial_shape format: (H, W, D) - NumPy array order
        # Convert to int and clip to valid range
        x1 = int(max(0, min(x1, spatial_shape[1])))  # x (Width) -> shape[1]
        x2 = int(max(0, min(x2, spatial_shape[1])))  # x (Width) -> shape[1]
        y1 = int(max(0, min(y1, spatial_shape[0])))  # y (Height) -> shape[0]
        y2 = int(max(0, min(y2, spatial_shape[0])))  # y (Height) -> shape[0]
        z1 = int(max(0, min(z1, spatial_shape[2])))  # z (Depth) -> shape[2]
        z2 = int(max(0, min(z2, spatial_shape[2])))  # z (Depth) -> shape[2]
        
        # Fill box region - NumPy array order is [H, W, D]
        if x2 > x1 and y2 > y1 and z2 > z1:
            mask[y1:y2, x1:x2, z1:z2] = 1  # Correct: mask[H, W, D]
    
    return mask


def validate(detector, loader, device, coco_metric, amp=True, verbose=True, save_debug=False, output_dir=None, epoch=0):
    """
    Validate with detection metrics (F1, Sensitivity, Precision)
    
    Note: For detection tasks, we use F1 score as the primary metric instead of validation loss.
    F1 score balances precision (avoiding false positives) and sensitivity (finding all lesions).
    """
    detector.eval()
    
    # Handle DataParallel wrapping
    det_module = detector.module if hasattr(detector, 'module') else detector
    
    val_outputs_all = []
    val_targets_all = []
    val_images_all = []  # Store first image for debugging
    val_seg_labels_all = []  # Store original segmentation labels for debugging
    val_meta_all = []  # Store metadata for coordinate conversion
    val_raw_data = []  # Store raw batch_data for original label access
    
    with torch.no_grad():
        iterator = tqdm(loader, desc="[Val]") if verbose else loader
        for batch_idx, batch_data in enumerate(iterator):
            inputs = [batch_data_i["image"].to(device) for batch_data_i in batch_data]
            targets = [
                dict(
                    label_class=batch_data_i["label_class"].to(device),  # Must match set_target_keys()
                    box=batch_data_i["box"].to(device),
                )
                for batch_data_i in batch_data
            ]
            
            # Direct inference with AMP (full image, no sliding window)
            # Optimized for median image size 512×512×32
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = detector(inputs, use_inferer=False)
            
            val_outputs_all += outputs
            val_targets_all += targets
    
            # Save first image, seg_label and metadata for debugging
            if batch_idx == 0 and save_debug:
                val_images_all = [inp.cpu().numpy() for inp in inputs]
                # Store segmentation labels if available
                val_seg_labels_all = [batch_data_i["seg_label"].cpu().numpy() if "seg_label" in batch_data_i else None for batch_data_i in batch_data]
                # Store metadata (affine, original_shape, etc.)
                val_meta_all = [batch_data_i.get("image_meta_dict", {}) for batch_data_i in batch_data]
                # Store raw data to access file paths
                val_raw_data = batch_data
    
    # (Debug output removed for cleaner logs)
    
    # Compute metrics
    # Extract predictions and targets with correct keys
    pred_boxes_list = []
    pred_classes_list = []
    pred_scores_list = []
    
    for val_data_i in val_outputs_all:
        # Handle box predictions
        if "box" in val_data_i and val_data_i["box"].numel() > 0:
            pred_boxes_list.append(val_data_i["box"].cpu().detach().numpy())
            
            # Try different key names for classes
            if "label_class" in val_data_i:
                pred_classes_list.append(val_data_i["label_class"].cpu().detach().numpy())
            elif "label" in val_data_i:
                pred_classes_list.append(val_data_i["label"].cpu().detach().numpy())
            else:
                pred_classes_list.append(np.zeros((len(val_data_i["box"]),), dtype=np.int64))
            
            # Try different key names for scores
            if "label_class_scores" in val_data_i:
                pred_scores_list.append(val_data_i["label_class_scores"].cpu().detach().numpy())
            elif "label_scores" in val_data_i:
                pred_scores_list.append(val_data_i["label_scores"].cpu().detach().numpy())
            elif "scores" in val_data_i:
                pred_scores_list.append(val_data_i["scores"].cpu().detach().numpy())
            else:
                pred_scores_list.append(np.ones((len(val_data_i["box"]),), dtype=np.float32))
        else:
            # No detections for this image
            pred_boxes_list.append(np.zeros((0, 6)))
            pred_classes_list.append(np.zeros((0,), dtype=np.int64))
            pred_scores_list.append(np.zeros((0,), dtype=np.float32))
    
    gt_boxes_list = [val_data_i["box"].cpu().detach().numpy() for val_data_i in val_targets_all]
    gt_classes_list = [val_data_i["label_class"].cpu().detach().numpy() for val_data_i in val_targets_all]
    
    # Compute matching results
    results_metric = matching_batch(
        iou_fn=box_utils.box_iou,
        iou_thresholds=coco_metric.iou_thresholds,
        pred_boxes=pred_boxes_list,
        pred_classes=pred_classes_list,
        pred_scores=pred_scores_list,
        gt_boxes=gt_boxes_list,
        gt_classes=gt_classes_list,
    )
    
    # Calculate detection statistics at IoU=0.1 (primary threshold for medical imaging)
    # Extract TP, FP, FN from matching results
    tp_count = 0
    fp_count = 0
    fn_count = 0
    total_pred = sum(len(p) for p in pred_boxes_list)
    total_gt = sum(len(g) for g in gt_boxes_list)
    
    # Calculate score range
    all_scores = np.concatenate([s for s in pred_scores_list if len(s) > 0]) if any(len(s) > 0 for s in pred_scores_list) else np.array([])
    score_min = float(all_scores.min()) if len(all_scores) > 0 else 0.0
    score_max = float(all_scores.max()) if len(all_scores) > 0 else 0.0
    score_mean = float(all_scores.mean()) if len(all_scores) > 0 else 0.0
    
    # Count matches at IoU threshold 0.1 (first threshold in coco_metric)
    iou_idx = 0  # Index for IoU=0.1
    for img_idx in range(len(results_metric)):
        result = results_metric[img_idx]
        
        # Handle both dict and array formats
        if isinstance(result, dict):
            # Dict format: {'dtMatches': [...], 'gtMatches': [...], ...}
            if 'dtMatches' in result and len(result['dtMatches']) > 0:
                # dtMatches[iou_idx] contains matched detection indices
                matched_indices = result['dtMatches'][iou_idx]
                tp_count += int((matched_indices > 0).sum())  # Matched detections
                fp_count += int((matched_indices == 0).sum())  # Unmatched detections
        elif hasattr(result, '__len__') and len(result) > 0:
            # Array format: [num_iou_thresholds, num_predictions]
            matched = result[iou_idx]
            if hasattr(matched, 'sum'):
                tp_count += int(matched.sum())
                fp_count += int((~matched).sum())
        
    # Calculate FN from total counts
    fn_count = total_gt - tp_count
    
    # Calculate metrics
    sensitivity = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0  # Recall
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    specificity = 1.0  # Not applicable for detection (no true negatives)
    
    # F1 Score (harmonic mean of precision and recall)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    # COCOMetric computation (following original detection/luna16_training.py)
    # Validate matching_batch results before passing to COCOMetric
    if verbose and save_debug:
        print(f"\n🔍 Validating matching_batch results:")
        print(f"   - Type: {type(results_metric)}")
        print(f"   - Length: {len(results_metric)}")
        if len(results_metric) > 0:
            print(f"   - First element type: {type(results_metric[0])}")
            if isinstance(results_metric[0], dict):
                print(f"   - First element keys: {results_metric[0].keys()}")
            elif hasattr(results_metric[0], 'shape'):
                print(f"   - First element shape: {results_metric[0].shape}")
    
    try:
        # Pass matching_batch results to coco_metric
        val_epoch_metric_dict = coco_metric(results_metric)[0]
        metric_results = val_epoch_metric_dict
        
        if verbose:
            print(f"\n✅ COCOMetric computed successfully!")
        
    except Exception as e:
        if verbose:
            import traceback
            print(f"\n{'='*70}")
            print(f"⚠️  COCOMetric Failed - Debugging Information")
            print(f"{'='*70}")
            print(f"\n1️⃣ Error Type: {type(e).__name__}")
            print(f"   Error Message: {str(e)}")
            
            print(f"\n2️⃣ results_metric Structure:")
            print(f"   - Type: {type(results_metric)}")
            print(f"   - Length: {len(results_metric) if hasattr(results_metric, '__len__') else 'N/A'}")
            
            if len(results_metric) > 0:
                print(f"\n3️⃣ First Result Sample:")
                first_result = results_metric[0]
                print(f"   - Type: {type(first_result)}")
                
                if isinstance(first_result, dict):
                    print(f"   - Keys: {list(first_result.keys())}")
                    for key in first_result.keys():
                        val = first_result[key]
                        if hasattr(val, 'shape'):
                            print(f"   - {key}: shape={val.shape}, dtype={val.dtype}")
                        elif hasattr(val, '__len__'):
                            print(f"   - {key}: length={len(val)}, type={type(val)}")
                        else:
                            print(f"   - {key}: {val}")
                elif hasattr(first_result, 'shape'):
                    print(f"   - Shape: {first_result.shape}")
                    print(f"   - Dtype: {first_result.dtype}")
                elif hasattr(first_result, '__len__'):
                    print(f"   - Length: {len(first_result)}")
            
            print(f"\n4️⃣ Prediction/GT Counts:")
            print(f"   - Total predictions: {sum(len(p) for p in pred_boxes_list)}")
            print(f"   - Total GT boxes: {sum(len(g) for g in gt_boxes_list)}")
            print(f"   - Num images: {len(pred_boxes_list)}")
            
            print(f"\n5️⃣ COCOMetric State:")
            print(f"   - Type: {type(coco_metric)}")
            if hasattr(coco_metric, 'iou_thresholds'):
                print(f"   - IoU thresholds: {coco_metric.iou_thresholds}")
            
            print(f"\n6️⃣ Full Traceback:")
            traceback.print_exc()
            
            print(f"\n{'='*70}")
            print(f"⚠️  Using basic metrics only (TP/FP/FN/Sensitivity/Precision/F1)")
            print(f"{'='*70}\n")
            
            # Save debug information to file for offline analysis
            if save_debug and output_dir:
                import pickle
                debug_dir = os.path.join(output_dir, "debug_samples")
                os.makedirs(debug_dir, exist_ok=True)
                
                debug_data = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'results_metric': results_metric,
                    'pred_boxes_list': pred_boxes_list,
                    'pred_classes_list': pred_classes_list,
                    'pred_scores_list': pred_scores_list,
                    'gt_boxes_list': gt_boxes_list,
                    'gt_classes_list': gt_classes_list,
                    'epoch': epoch,
                }
                
                try:
                    debug_file = os.path.join(debug_dir, "coco_metric_debug.pkl")
                    with open(debug_file, 'wb') as f:
                        pickle.dump(debug_data, f)
                    print(f"   💾 Debug data saved to: {debug_file}")
                    print(f"      You can analyze this offline to understand the error\n")
                except Exception as save_err:
                    print(f"   ⚠️  Could not save debug data: {save_err}\n")
        
        metric_results = {}
    
    # Save debug visualization
    if save_debug and output_dir and len(val_images_all) > 0 and len(val_meta_all) > 0:
        debug_dir = os.path.join(output_dir, "debug_samples")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Get first sample
        image = val_images_all[0][0]  # (C, H, W, D) -> (H, W, D) in MONAI format
        gt_boxes = val_targets_all[0]["box"].cpu().numpy()
        pred_boxes = pred_boxes_list[0]
        meta_dict = val_meta_all[0]
        
        # Get original affine and shape from metadata
        if "affine" in meta_dict:
            # Convert affine tensor to numpy
            affine = meta_dict["affine"].cpu().numpy() if torch.is_tensor(meta_dict["affine"]) else meta_dict["affine"]
        else:
            affine = np.eye(4)
            if verbose:
                print(f"⚠️  Warning: No affine in metadata, using identity")
        
        # Get original spatial shape (before any transforms)
        if "spatial_shape" in meta_dict:
            original_shape = meta_dict["spatial_shape"]
            if torch.is_tensor(original_shape):
                original_shape = original_shape.cpu().numpy()
        else:
            original_shape = image.shape
            if verbose:
                print(f"⚠️  Warning: No original spatial_shape in metadata, using current shape")
        
        # Current shape after transforms
        current_shape = image.shape  # Spatial dimensions after channel removal
        
        # Create masks from boxes (in current space)
        gt_mask = boxes_to_mask(gt_boxes, current_shape)
        pred_mask = boxes_to_mask(pred_boxes, current_shape)
        
        # Get original segmentation label
        seg_label = None
        if len(val_seg_labels_all) > 0 and val_seg_labels_all[0] is not None:
            seg_label = val_seg_labels_all[0][0]  # (C, H, W, D) -> (H, W, D)
        
        # Save with original affine
        # Note: Boxes are in current (transformed) image space, but we save with proper affine
        
        # Save image (overwrite each time)
        img_nii = nib.Nifti1Image(image.astype(np.float32), affine)
        nib.save(img_nii, os.path.join(debug_dir, "image.nii.gz"))
        
        # Save original segmentation label
        if seg_label is not None:
            seg_nii = nib.Nifti1Image(seg_label.astype(np.uint8), affine)
            nib.save(seg_nii, os.path.join(debug_dir, "gt_segmentation.nii.gz"))
        
        # Save GT boxes as mask
        gt_nii = nib.Nifti1Image(gt_mask.astype(np.uint8), affine)
        nib.save(gt_nii, os.path.join(debug_dir, "gt_boxes.nii.gz"))
        
        # Save prediction mask
        pred_nii = nib.Nifti1Image(pred_mask.astype(np.uint8), affine)
        nib.save(pred_nii, os.path.join(debug_dir, "pred_boxes.nii.gz"))
        
        # Try to load and save original image and label for comparison
        try:
            if len(val_raw_data) > 0:
                raw_sample = val_raw_data[0]
                
                # Get original file paths if available
                image_path = raw_sample.get("image", None)
                label_path = raw_sample.get("label", None)
                
                # Convert tensor to string if needed
                if torch.is_tensor(image_path):
                    image_path = str(image_path.item()) if image_path.numel() == 1 else None
                if torch.is_tensor(label_path):
                    label_path = str(label_path.item()) if label_path.numel() == 1 else None
                
                # Load original files
                if image_path and isinstance(image_path, str):
                    orig_img = nib.load(image_path)
                    nib.save(orig_img, os.path.join(debug_dir, "original_image.nii.gz"))
                    
                if label_path and isinstance(label_path, str):
                    orig_label = nib.load(label_path)
                    nib.save(orig_label, os.path.join(debug_dir, "original_label.nii.gz"))
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not save original files: {e}")
        
        # Save box coordinates and scores to text file
        with open(os.path.join(debug_dir, "info.txt"), 'w') as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Current image shape (after transform): {current_shape}\n")
            f.write(f"Original image shape (before transform): {tuple(original_shape)}\n")
            f.write(f"\nAffine matrix:\n{affine}\n\n")
            
            # Add file paths if available
            if len(val_raw_data) > 0:
                raw_sample = val_raw_data[0]
                if "image" in raw_sample:
                    f.write(f"Image file: {raw_sample['image']}\n")
                if "label" in raw_sample:
                    f.write(f"Label file: {raw_sample['label']}\n")
                f.write(f"\n")
            
            # Segmentation info
            if seg_label is not None:
                seg_unique = np.unique(seg_label)
                seg_nonzero = np.sum(seg_label > 0)
                f.write(f"Segmentation label:\n")
                f.write(f"  - Shape: {seg_label.shape}\n")
                f.write(f"  - Unique values: {seg_unique}\n")
                f.write(f"  - Non-zero voxels: {seg_nonzero}\n\n")
            
            f.write(f"Ground Truth Boxes ({len(gt_boxes)}) - extracted from segmentation:\n")
            for i, box in enumerate(gt_boxes):
                f.write(f"  GT {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f}]\n")
            
            f.write(f"\nPredicted Boxes ({len(pred_boxes)}) - from model inference:\n")
            pred_scores = pred_scores_list[0]
            pred_classes = pred_classes_list[0]
            for i, (box, score, cls) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
                f.write(f"  Pred {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f}] score={score:.3f} class={cls}\n")
    
    # Add detection statistics to metric_results for TensorBoard logging
    metric_results['detection/TP'] = tp_count
    metric_results['detection/FP'] = fp_count
    metric_results['detection/FN'] = fn_count
    metric_results['detection/sensitivity'] = sensitivity
    metric_results['detection/precision'] = precision
    metric_results['detection/f1_score'] = f1_score
    metric_results['detection/score_min'] = score_min
    metric_results['detection/score_max'] = score_max
    metric_results['detection/score_mean'] = score_mean
    
    # Print concise detection statistics
    if verbose:
        print(f"\n📊 Detection Statistics (IoU ≥ 0.1):")
        print(f"   ├─ True Positives:  {tp_count:4d} (correctly detected lesions)")
        print(f"   ├─ False Positives: {fp_count:4d} (incorrect detections)")
        print(f"   ├─ False Negatives: {fn_count:4d} (missed lesions)")
        print(f"   ├─ Total GT:        {total_gt:4d}")
        print(f"   ├─ Total Pred:      {total_pred:4d}")
        if total_pred > 0:
            print(f"   ├─ Score Range:     {score_min:.3f} - {score_max:.3f} (mean: {score_mean:.3f})")
        print(f"   ├─ Sensitivity:     {sensitivity:.1%} (TP / [TP + FN])")
        print(f"   ├─ Precision:       {precision:.1%} (TP / [TP + FP])")
        print(f"   └─ F1 Score:        {f1_score:.1%} (harmonic mean of precision & recall)")
    
    return results_metric, metric_results, f1_score, sensitivity, precision


# ============================================================================
# Testing / Inference
# ============================================================================

def test(detector, test_loader, device, test_data_dicts, output_dir, amp=True, save_predictions=False, save_nifti=False):
    """Run inference on test data
    
    Args:
        detector: RetinaNet detector
        test_loader: DataLoader for test data
        device: torch device
        test_data_dicts: List of test data dictionaries
        output_dir: Output directory
        amp: Use automatic mixed precision
        save_predictions: Save predictions to JSON
        save_nifti: Save predicted boxes as NIfTI masks
    """
    detector.eval()
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Running inference on {len(test_loader)} images...")
    print(f"{'='*60}\n")
    
    # Create output directories
    if save_nifti:
        nifti_dir = os.path.join(output_dir, "predictions_nifti")
        os.makedirs(nifti_dir, exist_ok=True)
        print(f"💾 NIfTI output directory: {nifti_dir}\n")
    
    # Debug flag
    debug_printed = False
    
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader, desc="[Test]")):
            inputs = [batch_data_i["image"].to(device) for batch_data_i in batch_data]
            
            # Get original image path and metadata
            image_path = test_data_dicts[idx]["image"]
            image_name = os.path.basename(image_path)
            
            # Get image metadata for NIfTI saving
            image_data = batch_data[0]["image"]  # (C, H, W, D)
            image_meta = batch_data[0].get("image_meta_dict", None)
            
            # Inference with AMP (direct inference, no sliding window)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = detector(inputs, use_inferer=False)
            
            # Debug: Print keys on first iteration
            if not debug_printed and len(outputs) > 0:
                print(f"\n🔍 Debug - Output keys: {outputs[0].keys()}")
                for key in outputs[0].keys():
                    val = outputs[0][key]
                    if isinstance(val, torch.Tensor):
                        print(f"   - {key}: shape {val.shape}, dtype {val.dtype}")
            
            # Extract predictions (flexible key handling)
            output = outputs[0]
            
            # Try to get boxes (try multiple key names)
            pred_boxes = None
            used_box_key = None
            for box_key in ["box", "boxes", "pred_box", "pred_boxes"]:
                if box_key in output and output[box_key].numel() > 0:
                    pred_boxes = output[box_key].cpu().detach().numpy()
                    used_box_key = box_key
                    break
            
            if pred_boxes is None:
                pred_boxes = np.zeros((0, 6))
            
            # Print which keys are being used (first time only)
            if not debug_printed:
                print(f"   ✅ Using keys: box='{used_box_key}'")
                debug_printed = True
            
            # Try to get classes and scores
            if len(pred_boxes) > 0:
                # Try to get classes
                pred_classes = None
                for class_key in ["label_class", "labels", "label", "pred_label", "pred_labels"]:
                    if class_key in output:
                        pred_classes = output[class_key].cpu().detach().numpy()
                        break
                if pred_classes is None:
                    pred_classes = np.zeros((len(pred_boxes),), dtype=np.int64)
                
                # Try to get scores
                pred_scores = None
                for score_key in ["label_class_scores", "labels_scores", "label_scores", "scores", "pred_scores"]:
                    if score_key in output:
                        pred_scores = output[score_key].cpu().detach().numpy()
                        break
                if pred_scores is None:
                    pred_scores = np.ones((len(pred_boxes),), dtype=np.float32)
            else:
                pred_classes = np.zeros((0,), dtype=np.int64)
                pred_scores = np.zeros((0,), dtype=np.float32)
            
            # Store results
            result = {
                "image": image_name,
                "num_detections": len(pred_boxes),
                "boxes": pred_boxes.tolist(),
                "scores": pred_scores.tolist(),
                "classes": pred_classes.tolist(),
            }
            all_results.append(result)
            
            # Save predicted boxes as NIfTI mask
            if save_nifti:
                # Get image shape (remove channel dimension)
                if image_data.dim() == 4:  # (C, H, W, D)
                    image_shape = image_data.shape[1:]  # (H, W, D)
                else:
                    image_shape = image_data.shape
                
                # Convert boxes to mask
                pred_mask = boxes_to_mask(pred_boxes, image_shape)
                
                # Get affine from metadata
                if image_meta is not None and "affine" in image_meta:
                    affine = image_meta["affine"]
                    if torch.is_tensor(affine):
                        affine = affine.cpu().numpy()
                else:
                    # Try to load affine from original file
                    try:
                        orig_nii = nib.load(image_path)
                        affine = orig_nii.affine
                    except:
                        affine = np.eye(4)
                
                # Save as NIfTI
                output_name = image_name.replace('.nii.gz', '_pred.nii.gz')
                output_path = os.path.join(nifti_dir, output_name)
                pred_nii = nib.Nifti1Image(pred_mask.astype(np.uint8), affine)
                nib.save(pred_nii, output_path)
            
            # Print summary for each image
            if len(pred_boxes) > 0:
                nifti_msg = f" (NIfTI: {output_name})" if save_nifti else ""
                print(f"  [{idx+1}/{len(test_loader)}] {image_name}: {len(pred_boxes)} detections (scores: {pred_scores.min():.3f}-{pred_scores.max():.3f}){nifti_msg}")
            else:
                print(f"  [{idx+1}/{len(test_loader)}] {image_name}: No detections")
    
    # Save predictions to JSON
    if save_predictions:
        output_file = os.path.join(output_dir, "predictions.json")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Predictions saved to: {output_file}")
    
    # NIfTI summary
    if save_nifti:
        print(f"✅ NIfTI masks saved to: {nifti_dir}/")
        print(f"   Format: <image_name>_pred.nii.gz")
    
    # Print summary statistics
    total_detections = sum(r["num_detections"] for r in all_results)
    images_with_detections = sum(1 for r in all_results if r["num_detections"] > 0)
    
    print(f"\n{'='*60}")
    print(f"Inference Summary:")
    print(f"  - Total images: {len(all_results)}")
    print(f"  - Images with detections: {images_with_detections}/{len(all_results)}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Avg detections per image: {total_detections/len(all_results):.2f}")
    if save_nifti:
        print(f"  - NIfTI masks: {nifti_dir}/")
    print(f"{'='*60}\n")
    
    return all_results


# ============================================================================
# DDP Setup
# ============================================================================

def setup_ddp(rank, world_size):
    """Initialize DDP (environment variables should be set by torchrun)"""
    # torchrun automatically sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    # We just need to initialize the process group
    # Increase timeout for slow validation (default: 10min -> 30min)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"⚠️  Warning: Config file not found: {config_path}")
        print(f"   Using command-line arguments only")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# Main
# ============================================================================

def main(rank=0, world_size=1):
    parser = argparse.ArgumentParser(
        description='3D Detection with MONAI RetinaNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train (default mode)
  python3 nndet_simple.py
  
  # Train with custom config
  python3 nndet_simple.py --config my_config.yaml
  
  # Test/Inference
  python3 nndet_simple.py --mode test
  
  # Multi-GPU training with torchrun
  torchrun --nproc_per_node=4 nndet_simple.py

All parameters (data paths, model settings, training hyperparameters) are read from config.yaml.
To modify settings, edit config.yaml directly.
        """
    )
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test (default: train)')
    args = parser.parse_args()
    
    # Load all parameters from config
    config = load_config(args.config)
    if config is None:
        raise FileNotFoundError(f"Config file not found: {args.config}. Please create config.yaml first.")
    
    # Extract all parameters from config
    # Data
    args.image_dir = config['data']['image_dir']
    args.label_dir = config['data']['label_dir']
    args.output_dir = config['data']['output_dir']
    args.test_image_dir = config['data'].get('test_image_dir', None)
    args.test_label_dir = config['data'].get('test_label_dir', None)
    
    # Model
    args.backbone = config['model']['backbone']
    args.num_classes = config['model']['num_classes']
    
    # Training
    args.epochs = config['training']['epochs']
    args.batch_size = config['training']['batch_size']
    args.lr = config['training']['lr']
    args.patch_size = config['training']['patch_size']
    args.val_interval = config['training']['val_interval']
    
    # Detection (test mode)
    args.score_thresh = config['detection']['score_thresh_test']
    args.detections_per_img = config['detection']['detections_per_img_test']
    args.checkpoint = config['detection'].get('checkpoint', None)
    
    # Anchor
    args.min_size = config['anchor']['min_size']
    args.merge_distance_mm = config['anchor']['merge_distance_mm']
    
    # Augmentation
    args.intensity_min = config['augmentation']['intensity_min']
    args.intensity_max = config['augmentation']['intensity_max']
    
    # Hardware
    args.no_amp = not config['hardware']['use_amp']
    args.multi_gpu = config['hardware'].get('multi_gpu', False)
    
    # Output
    args.save_predictions = config['output'].get('save_predictions', False)
    args.save_nifti = config['output'].get('save_nifti', False)
    args.debug = config['output'].get('debug', False)
    
    # Gradient accumulation (not commonly used, default to 1)
    args.gradient_accumulation_steps = 1
    
    # Setup DDP if multi-GPU
    is_ddp = world_size > 1
    if is_ddp:
        setup_ddp(rank, world_size)
    
    # Only rank 0 prints and saves
    is_main_process = (rank == 0)
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # TEST MODE
    # ========================================================================
    if args.mode == 'test':
        if is_main_process:
            print(f"\n🔍 TEST MODE")
            print(f"{'='*60}\n")
        
        # Check checkpoint
        if args.checkpoint is None:
            checkpoint_path = os.path.join(args.output_dir, f"best_model_{args.backbone}.pth")
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"No checkpoint found at {checkpoint_path}. Please specify --checkpoint")
            args.checkpoint = checkpoint_path
        
        if is_main_process:
            print(f"Loading model from: {args.checkpoint}")
        
        # Test data directory
        test_image_dir = args.test_image_dir if args.test_image_dir else args.image_dir
        test_label_dir = args.test_label_dir  # Can be None
        
        # Test transform (image only, no labels required)
        test_transforms = generate_test_transform(
            amp=not args.no_amp,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max
        )
        
        # Test dataset
        test_image_files = sorted(glob(os.path.join(test_image_dir, "*.nii.gz")))
        if len(test_image_files) == 0:
            raise ValueError(f"No images found in {test_image_dir}")
        
        test_data_dicts = [{"image": img} for img in test_image_files]
        if test_label_dir:
            test_label_files = sorted(glob(os.path.join(test_label_dir, "*.nii.gz")))
            for i, lbl in enumerate(test_label_files):
                if i < len(test_data_dicts):
                    test_data_dicts[i]["label"] = lbl
        
        test_ds = Dataset(data=test_data_dicts, transform=test_transforms)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=no_collation,
        )
        
        if is_main_process:
            print(f"Test images: {len(test_ds)}")
        
        # Load checkpoint first to get anchor_shapes
        if is_main_process:
            print(f"\n🔄 Loading checkpoint: {args.checkpoint}")
        
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            
            # Extract anchor_shapes from checkpoint
            if 'anchor_shapes' in checkpoint:
                base_anchor_shapes = checkpoint['anchor_shapes']
                if is_main_process:
                    anchor_str = ", ".join([f"[{s[0]}×{s[1]}×{s[2]}]" for s in base_anchor_shapes])
                    print(f"   ✅ Loaded anchor shapes from checkpoint: {anchor_str}")
            else:
                # Fallback to default if not in checkpoint (5 anchors for old checkpoints)
                base_anchor_shapes = [
                    [10, 10, 1],   # Very Small
                    [20, 20, 2],   # Small
                    [30, 30, 3],   # Small variant
                    [40, 40, 4],   # Median
                    [50, 50, 5],   # Large
                ]
                if is_main_process:
                    print(f"   ⚠️  WARNING: No anchor_shapes in checkpoint!")
                    print(f"   ⚠️  Using default fallback (5 anchors):")
                    anchor_str = ", ".join([f"[{s[0]}×{s[1]}×{s[2]}]" for s in base_anchor_shapes])
                    print(f"      {anchor_str}")
                    print(f"\n   🔴 IMPORTANT: This checkpoint was trained with DIFFERENT anchors!")
                    print(f"   💡 For accurate results, please re-train with current code:")
                    print(f"      1. bash run_optimize_anchors.sh")
                    print(f"      2. bash train_ddp.sh")
                    print(f"   ⚠️  Current test results may be suboptimal.\n")
            
            # Verify configuration matches
            if is_main_process:
                print(f"\n📋 Checkpoint Information:")
                if 'epoch' in checkpoint:
                    print(f"   - Trained epoch: {checkpoint['epoch']+1}")
                # Try different checkpoint formats
                if 'best_f1' in checkpoint:
                    print(f"   - Best F1: {checkpoint['best_f1']:.1%}")
                elif 'best_loss' in checkpoint:
                    print(f"   - Best loss: {checkpoint['best_loss']:.4f} (old format)")
                elif 'best_metric' in checkpoint:
                    print(f"   - Best metric: {checkpoint['best_metric']:.4f} (old format)")
                if 'backbone' in checkpoint:
                    print(f"   - Backbone: {checkpoint['backbone']}")
                if 'num_classes' in checkpoint:
                    print(f"   - Num classes: {checkpoint['num_classes']}")
                if 'patch_size' in checkpoint:
                    print(f"   - Patch size (training): {checkpoint['patch_size']}")
                if 'monai_version' in checkpoint:
                    print(f"   - MONAI version: {checkpoint['monai_version']}")
                
                # Check for mismatches
                print(f"\n🔍 Configuration Check:")
                mismatches = []
                
                if 'backbone' in checkpoint and checkpoint['backbone'] != args.backbone:
                    mismatches.append(f"   ⚠️  Backbone: che0ckpoint={checkpoint['backbone']}, current={args.backbone}")
                else:
                    print(f"   ✅ Backbone: {args.backbone}")
                
                if 'num_classes' in checkpoint and checkpoint['num_classes'] != args.num_classes:
                    mismatches.append(f"   ⚠️  Num classes: checkpoint={checkpoint['num_classes']}, current={args.num_classes}")
                else:
                    print(f"   ✅ Num classes: {args.num_classes}")
                
                if 'anchor_shapes' in checkpoint:
                    print(f"   ✅ Anchor shapes: Loaded from checkpoint ({len(base_anchor_shapes)} anchors)")
                else:
                    print(f"   ⚠️  Anchor shapes: Using fallback (may not match training!)")
                
                if mismatches:
                    print(f"\n{'='*60}")
                    print(f"🔴 CONFIGURATION MISMATCH DETECTED:")
                    for msg in mismatches:
                        print(msg)
                    print(f"\n⚠️  Test results may be inaccurate due to configuration mismatch!")
                    print(f"💡 Please re-train or use correct checkpoint.")
                    print(f"{'='*60}\n")
                else:
                    print(f"   ✅ All configurations match!\n")
                    
        except Exception as e:
            if is_main_process:
                print(f"   ❌ Failed to load checkpoint: {str(e)[:200]}")
                print(f"   ❌ Please check checkpoint path and format\n")
            raise
        
        # Build model with anchor_shapes from checkpoint
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**l for l in range(4)],
            base_anchor_shapes=base_anchor_shapes,
        )
        
        # Build backbone based on args (same as training)
        conv1_t_stride = [2, 2, 1]
        conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]
        
        if args.backbone == 'resnet101':
            backbone = resnet.ResNet(
                block=resnet.ResNetBottleneck,
                layers=[3, 4, 23, 3],  # ResNet101
                block_inplanes=resnet.get_inplanes(),
                n_input_channels=1,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=conv1_t_size,
            )
        else:  # resnet50 (default)
            backbone = resnet.ResNet(
                block=resnet.ResNetBottleneck,
                layers=[3, 4, 6, 3],  # ResNet50
                block_inplanes=resnet.get_inplanes(),
                n_input_channels=1,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=conv1_t_size,
            )
        
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=3,
            pretrained_backbone=True,
            trainable_backbone_layers=None,
            returned_layers=[1, 2, 3],
        )
        
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        size_divisible = [s * 2 * 2 ** max([1, 2, 3]) for s in feature_extractor.body.conv1.stride]
        
        net = RetinaNet(
            spatial_dims=3,
            num_classes=args.num_classes,
            num_anchors=num_anchors,
            feature_extractor=feature_extractor,
            size_divisible=size_divisible,
        )
        
        detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)
        
        # Set inference parameters
        if is_main_process:
            print(f"\n📊 Inference Parameters:")
            print(f"  - Score threshold: {args.score_thresh}")
            print(f"  - NMS threshold: 0.22")
            print(f"  - Max detections per image: {args.detections_per_img}")
            print(f"\n💡 Tip: If scores are too low or too many detections:")
            print(f"     Increase --score_thresh (e.g., 0.3) or decrease --detections_per_img (e.g., 10)\n")
        
        detector.set_box_selector_parameters(
            score_thresh=args.score_thresh,
            topk_candidates_per_level=100,
            nms_thresh=0.22,
            detections_per_img=args.detections_per_img,
        )
        # No sliding window - use direct full-image inference
        
        # Load model weights from checkpoint (checkpoint already loaded above)
        if is_main_process:
            print(f"🔄 Loading model weights...")
        
        try:
            detector.network.load_state_dict(checkpoint['model_state_dict'])
            
            if is_main_process:
                print(f"   ✅ Model weights loaded successfully!\n")
        except (RuntimeError, KeyError) as e:
            if is_main_process:
                print(f"   ❌ Failed to load model weights: {str(e)[:100]}")
                print(f"   ❌ This usually means:")
                print(f"      1) Architecture mismatch (backbone changed)")
                print(f"      2) Anchor shapes mismatch (check anchor_shapes in checkpoint)")
                print(f"      3) Corrupted checkpoint file")
                print(f"\n   💡 Solution: Re-train with current settings or use correct checkpoint\n")
            raise  # Re-raise to stop execution (can't do inference without weights)
        
        # Run inference
        results = test(
            detector,
            test_loader,
            device,
            test_data_dicts,
            args.output_dir,
            amp=not args.no_amp,
            save_predictions=args.save_predictions,
            save_nifti=args.save_nifti
        )
        
        if is_ddp:
            cleanup_ddp()
        
        return
    
    # ========================================================================
    # TRAIN MODE
    # ========================================================================
    
    # Transforms
    train_transforms = generate_train_transform(
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        debug=args.debug,
        min_size=args.min_size,
        merge_distance_mm=args.merge_distance_mm,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max
    )
    val_transforms = generate_val_transform(
        amp=not args.no_amp,
        debug=args.debug,
        min_size=args.min_size,
        merge_distance_mm=args.merge_distance_mm,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max
    )
    
    # Dataset
    full_dataset = MedicalDetectionDataset(args.image_dir, args.label_dir, None)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    
    train_data = [full_dataset.data_dicts[i] for i in train_indices]
    val_data = [full_dataset.data_dicts[i] for i in val_indices]
    
    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    
    if is_main_process:
        print(f"\nLoaded {len(full_dataset)} samples")
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
        
        print(f"\n💡 Batch configuration:")
        print(f"   - GPUs: {world_size}")
        print(f"   - Per-GPU batch size: {args.batch_size}")
        print(f"   - Total effective batch size: {args.batch_size * world_size}")
    
    # DataLoader with DistributedSampler for DDP
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle_train = False
    else:
        train_sampler = None
        shuffle_train = True
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # Per-GPU batch size
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        num_workers=4,
        pin_memory=True,
        collate_fn=no_collation,
        persistent_workers=True,  # Keep workers alive to prevent hang
    )
    
    # Validation loader: Only created on rank 0 (only rank 0 validates)
    if is_main_process:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=no_collation,
            persistent_workers=True,  # Keep workers alive
        )
    else:
        val_loader = None  # Non-main processes don't need val_loader
    
    # Build model
    # 1) Load optimized settings from optimize_anchors.py output
    # Try to load: Anchor shapes + Patch sizes
    optimized_file = os.path.join(args.output_dir, '..', 'eda', 'optimized_anchors.json')
    base_anchor_shapes = None
    optimized_patch_size = None
    
    # Priority 1: K-Means optimized anchors + recommended patch sizes
    if os.path.exists(optimized_file):
        try:
            with open(optimized_file, 'r') as f:
                opt_data = json.load(f)
            
            # Load K-Means optimized anchor shapes (already in feature map scale)
            base_anchor_shapes = opt_data['anchor_shapes_feature']
            avg_iou = opt_data.get('avg_iou', 0)
            
            # Load recommended patch size (NEW!)
            if 'recommended_patch_sizes' in opt_data:
                optimized_patch_size = opt_data['recommended_patch_sizes']['train_patch_size']
                # Override config.yaml patch_size with optimized value
                args.patch_size = optimized_patch_size
            
            if is_main_process:
                print(f"\n🎯 Loaded optimization results:")
                print(f"   File: {optimized_file}")
                print(f"   Number of anchors: {len(base_anchor_shapes)}")
                print(f"   Average IoU: {avg_iou:.4f}")
                print(f"   Anchor shapes (feature map scale):")
                for i, (w, h, d) in enumerate(base_anchor_shapes):
                    print(f"      Anchor {i+1}: [{w:3d}×{h:3d}×{d:3d}]")
                
                if optimized_patch_size:
                    print(f"   Recommended patch size: {optimized_patch_size} (W, H, D)")
                    print(f"      ✅ Using optimized patch_size for training")
                else:
                    print(f"   Patch size: {args.patch_size} (from config.yaml)")
        except Exception as e:
            if is_main_process:
                print(f"\n⚠️  Warning: Failed to load optimization results: {e}")
            base_anchor_shapes = None
            optimized_patch_size = None
    
    # Priority 2: Default anchors + config.yaml patch_size
    if base_anchor_shapes is None:
        if is_main_process:
            print(f"\n📌 Using default settings (not optimized)")
            print(f"   💡 Tip: Run 'bash run_optimize_anchors.sh' for optimal anchors and patch sizes!")
        base_anchor_shapes = [
            [10, 10, 1],   # Very Small variant
            [20, 20, 2],   # Small variant
            [30, 30, 3],   # Small variant
            [40, 40, 4],   # Median lesion
            [50, 50, 5],   # Large variant
        ]
    
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(4)],  # [1, 2, 4, 8] for 3 returned layers
        base_anchor_shapes=base_anchor_shapes,
    )
    
    # Print model configuration (after anchor shapes are loaded)
    if is_main_process:
        backbone_name = args.backbone.upper().replace('-', '')
        print(f"\n{'='*60}")
        print(f"3D Detection with MONAI RetinaNet")
        print(f"  - Backbone: {backbone_name} + FPN (3 layers)")
        print(f"  - Anchor-based detection with ATSS matcher")
        
        # Format anchor shapes for display
        anchor_str = ", ".join([f"[{s[0]}×{s[1]}×{s[2]}]" for s in base_anchor_shapes])
        print(f"  - Anchor shapes (feature map): {anchor_str}")
        
        print(f"  - Foreground classes: {args.num_classes}")
        print(f"  - Patch size (train): {args.patch_size}")
        print(f"  - Patch size (val): Full image (direct inference)")
        print(f"  - AMP: {not args.no_amp}")
        if is_ddp:
            print(f"  - Normalization: SyncBatchNorm (synced across {world_size} GPUs)")
        else:
            print(f"  - Normalization: BatchNorm")
        print(f"{'='*60}\n")
    
    # 2) Network: Backbone + FPN
    conv1_t_stride = [2, 2, 1]
    conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]
    
    if args.backbone == 'resnet101':
        if is_main_process:
            print(f"\n🔧 Using ResNet101 backbone (deeper, more capacity)")
        
        backbone = resnet.ResNet(
            block=resnet.ResNetBottleneck,
            layers=[3, 4, 23, 3],  # ResNet101 (more layers in layer3)
            block_inplanes=resnet.get_inplanes(),
            n_input_channels=1,
            conv1_t_stride=conv1_t_stride,
            conv1_t_size=conv1_t_size,
        )
    else:  # resnet50 (default)
        if is_main_process:
            print(f"\n🔧 Using ResNet50 backbone (baseline)")
    
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        layers=[3, 4, 6, 3],  # ResNet50
        block_inplanes=resnet.get_inplanes(),
        n_input_channels=1,
        conv1_t_stride=conv1_t_stride,
        conv1_t_size=conv1_t_size,
    )
    
    # Feature extractor with FPN
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=3,
        pretrained_backbone=False,
        trainable_backbone_layers=None,
        returned_layers=[1, 2, 3],
    )
    
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max([1, 2, 3]) for s in feature_extractor.body.conv1.stride]
    
    # Don't use torch.jit.script - incompatible with AMP + DDP
    net = RetinaNet(
        spatial_dims=3,
        num_classes=args.num_classes,
        num_anchors=num_anchors,
        feature_extractor=feature_extractor,
        size_divisible=size_divisible,
    )
    
    # 3) Detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)
    
    # Convert to SyncBatchNorm for multi-GPU training
    # This allows batch statistics to be synchronized across all GPUs
    # Very important for small batch sizes (effective batch size = per_gpu_batch × num_gpus)
    if is_ddp:
        detector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(detector)
        if is_main_process:
            print(f"✅ Converted to SyncBatchNorm (batch statistics synced across {world_size} GPUs)")
    
    # Set training components
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=0.3,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label_class")
    
    # Set validation components
    detector.set_box_selector_parameters(
        score_thresh=0.2,  # Increased from 0.02 to reduce false positives
        topk_candidates_per_level=1000,
        nms_thresh=0.2,    # Slightly decreased for stricter NMS
        detections_per_img=10,  # Reduced from 100 to max 20 per image
    )
    # No sliding window inferer - use direct full-image inference
    # This is optimal for datasets with consistent small image sizes (median: 512×512×32)
    
    # Wrap with DDP if multi-GPU
    if is_ddp:
        # find_unused_parameters=True: RetinaNet detector uses different forward paths for train/inference
        detector = DDP(detector, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        if is_main_process:
            print(f"✅ DDP initialized with {world_size} GPUs")
    
    # Optimizer
    # Get the actual network (handle DataParallel wrapping)
    det_network = detector.module.network if hasattr(detector, 'module') else detector.network
    optimizer = torch.optim.SGD(
        det_network.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True,
    )
    
    # Scheduler with warmup (using LambdaLR to avoid deprecation warnings)
    warmup_epochs = 5
    step_size = 50
    gamma = 0.1
    
    def lr_lambda(epoch):
        """Custom learning rate schedule with warmup"""
        if epoch < warmup_epochs:
            # Warmup: linearly increase from 0.1 to 1.0
            return 0.1 + (0.9 * epoch / warmup_epochs)
        else:
            # Step decay after warmup
            return gamma ** ((epoch - warmup_epochs) // step_size)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if not args.no_amp else None
    
    # Metric
    coco_metric = COCOMetric(classes=["lesion"], iou_list=[0.1, 0.3, 0.5], max_detection=[20])
    
    # TensorBoard (only main process)
    if is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
    else:
        writer = None
    
    # Load pretrained weights if best_model_{backbone}.pth exists
    best_model_path = os.path.join(args.output_dir, f"best_model_{args.backbone}.pth")
    if os.path.exists(best_model_path):
        if is_main_process:
            print(f"\n🔄 Attempting to load pretrained weights from: {best_model_path}")
        
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            
            # Load only model weights (not optimizer)
            det_network = detector.module.network if hasattr(detector, 'module') else detector.network
            det_network.load_state_dict(checkpoint['model_state_dict'])
            
            if is_main_process:
                print(f"   ✅ Successfully loaded model weights!")
                print(f"   ✅ Previous epoch: {checkpoint.get('epoch', 'unknown')}")
                # Try different checkpoint formats
                if 'best_f1' in checkpoint:
                    print(f"   ✅ Previous best F1: {checkpoint.get('best_f1'):.1%}")
                elif 'best_loss' in checkpoint:
                    print(f"   ✅ Previous best loss: {checkpoint.get('best_loss'):.4f} (old format)")
                elif 'best_metric' in checkpoint:
                    print(f"   ✅ Previous best metric: {checkpoint.get('best_metric'):.4f} (old format)")
                print(f"   ⚠️  Optimizer reset (starting fresh)\n")
        except (RuntimeError, KeyError) as e:
            # Weight loading failed - likely due to architecture mismatch
            if is_main_process:
                print(f"   ⚠️  Failed to load weights: {str(e)[:100]}")
                print(f"   ⚠️  This is expected if you changed the backbone architecture")
                print(f"   ℹ️  Starting training from scratch (random initialization)\n")
    
    if is_main_process:
        print(f"\nStarting training for {args.epochs} epochs\n")
    
    best_f1 = 0.0  # Higher is better (F1 score for detection)
    best_epoch = -1
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if is_ddp:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 60)
        
        # Train
        train_loss, train_cls_loss, train_box_loss = train_epoch(
            detector, train_loader, optimizer, device, epoch, scaler, not args.no_amp, 
            verbose=is_main_process  # Only show progress bar on rank 0
        )
        
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, box: {train_box_loss:.4f}) | LR: {current_lr:.2e}")
            
            if writer is not None:
                writer.add_scalar("train/total_loss", train_loss, epoch)
                writer.add_scalar("train/cls_loss", train_cls_loss, epoch)
                writer.add_scalar("train/box_loss", train_box_loss, epoch)
                writer.add_scalar("train/lr", current_lr, epoch)
        
        # Validate (only rank 0, no DistributedSampler = no deadlock)
        if (epoch + 1) % args.val_interval == 0:
            # Synchronize before validation
            if is_ddp:
                dist.barrier()
            
            if is_main_process:
                print("\nRunning validation...")
                
                # Use unwrapped model for validation (rank 0 only)
                val_detector = detector.module if is_ddp else detector
                
                results, metric_results, f1_score, sensitivity, precision = validate(
                    val_detector,
                    val_loader,
                    device,
                    coco_metric,
                    amp=not args.no_amp,
                    verbose=True,  # Always verbose since only rank 0 validates
                    save_debug=True,  # Save debug samples every validation
                    output_dir=args.output_dir,
                    epoch=epoch
                )
                
                # Print validation results
                print(f"\n📈 Validation Results (Epoch {epoch+1}):")
                print(f"  ├─ F1 Score:      {f1_score:.1%}")
                print(f"  ├─ Sensitivity:   {sensitivity:.1%}")
                print(f"  ├─ Precision:     {precision:.1%}")
                print(f"  ├─ mAP@0.10-0.50: {metric_results.get('mAP_IoU_0.10_0.50_0.05_MaxDet_100', 0.0):.4f}")
                print(f"  ├─ AP@0.10:       {metric_results.get('AP_IoU_0.10_MaxDet_100', 0.0):.4f}")
                print(f"  ├─ AP@0.30:       {metric_results.get('AP_IoU_0.30_MaxDet_100', 0.0):.4f}")
                print(f"  ├─ AP@0.50:       {metric_results.get('AP_IoU_0.50_MaxDet_100', 0.0):.4f}")
                print(f"  ├─ mAR@0.10-0.50: {metric_results.get('mAR_IoU_0.10_0.50_0.05_MaxDet_100', 0.0):.4f}")
                print(f"  └─ 🎯 Best F1:     {f1_score:.1%} {'← New Best! 🏆' if f1_score > best_f1 else f'(Best: {best_f1:.1%})'}")
                
                # Save all metrics to tensorboard
                if writer is not None:
                    writer.add_scalar("val/f1_score", f1_score, epoch)
                    writer.add_scalar("val/sensitivity", sensitivity, epoch)
                    writer.add_scalar("val/precision", precision, epoch)
                    for k, v in metric_results.items():
                        writer.add_scalar(f"val/{k}", v, epoch)
                
                # Save best model (based on highest F1 score)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_epoch = epoch
                    save_path = os.path.join(args.output_dir, f"best_model_{args.backbone}.pth")
                    # Get the actual network (handle DDP wrapping)
                    save_network = detector.module.network if hasattr(detector, 'module') else detector.network
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': save_network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_f1': best_f1,
                        'sensitivity': sensitivity,
                        'precision': precision,
                        # Model architecture
                        'backbone': args.backbone,
                        'num_classes': args.num_classes,
                        'anchor_shapes': base_anchor_shapes,
                        # Training settings
                        'patch_size': args.patch_size,
                        'batch_size': args.batch_size,
                        'lr': args.lr,
                        # Detection parameters
                        'score_thresh': args.score_thresh,
                        # Software version (for debugging)
                        'monai_version': monai.__version__,
                    }, save_path)
                    print(f"   💾 Saved: {os.path.basename(save_path)}")
            
            # Synchronize after validation
            if is_ddp:
                dist.barrier()
        
        # Sync all processes before next epoch
        if is_ddp:
            dist.barrier()
        
        # Step scheduler
        scheduler.step()
        
        # Note: Epoch checkpoints are disabled. Only best_model_{backbone}.pth is saved.
    
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best F1 score: {best_f1:.1%} at epoch {best_epoch+1}")
        print(f"{'='*60}\n")
        if writer is not None:
            writer.close()
    
    # Cleanup DDP
    if is_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    import sys
    import os
    
    # Check if launched with torchrun (DDP)
    # torchrun sets RANK and WORLD_SIZE environment variables
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun - use DDP
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        main(rank, world_size)
    elif '--multi_gpu' in sys.argv:
        # User requested multi-GPU but not using torchrun
        print("\n⚠️  Multi-GPU requested but not launched with torchrun!")
        print("Please use: torchrun --nproc_per_node=4 nndet_simple.py --mode train --batch_size 1 --epochs 100")
        print("Falling back to single GPU...\n")
        main(rank=0, world_size=1)
    else:
        # Single GPU mode
        main(rank=0, world_size=1)
