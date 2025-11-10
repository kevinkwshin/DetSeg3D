#!/usr/bin/env python3
"""
3D Detection with MONAI RetinaNet
Based on MONAI detection module examples
"""

import os
import argparse
import json
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
from monai.networks.nets import resnet, EfficientNetBN
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    RandSpatialCropd, ScaleIntensityRanged, RandFlipd, RandRotate90d, RandZoomd,
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
    
    Args:
        mask: (H, W, D) binary mask tensor or array
        min_size: minimum voxel count for valid box
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
    
    # Apply morphological closing to merge nearby lesions
    if merge_distance_mm > 0 and spacing is not None:
        # Calculate kernel size in voxels for each dimension
        # kernel_size = merge_distance_mm / spacing (convert mm to voxels)
        spacing_array = np.array(spacing) if spacing is not None else np.array([1.0, 1.0, 1.0])
        kernel_voxels = np.ceil(merge_distance_mm / spacing_array).astype(int)
        
        # Ensure odd kernel size (required for symmetry)
        kernel_voxels = kernel_voxels + (1 - kernel_voxels % 2)
        
        if debug:
            print(f"   Voxel spacing: {spacing_array}")
            print(f"   Kernel size (voxels): {kernel_voxels}")
        
        # Create structuring element (ellipsoid for isotropic merging)
        from scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
        
        # Use binary closing (dilation then erosion)
        # Dilation: expand lesions
        struct_elem = generate_binary_structure(3, 1)  # 3D connectivity
        mask_dilated = mask_np > 0
        
        # Perform multiple iterations based on kernel size
        iterations_x = max(1, kernel_voxels[0] // 2)
        iterations_y = max(1, kernel_voxels[1] // 2)
        iterations_z = max(1, kernel_voxels[2] // 2)
        max_iterations = max(iterations_x, iterations_y, iterations_z)
        
        mask_dilated = binary_dilation(mask_dilated, structure=struct_elem, iterations=max_iterations)
        
        # Erosion: shrink back
        mask_closed = binary_erosion(mask_dilated, structure=struct_elem, iterations=max_iterations)
        
        if debug:
            print(f"   Morphological closing iterations: {max_iterations}")
            print(f"   Components before closing: {ndimage.label(mask_np > 0)[1]}")
            print(f"   Components after closing: {ndimage.label(mask_closed)[1]}")
        
        mask_np = mask_closed.astype(mask_np.dtype)
    
    # Find connected components
    labeled_mask, num_components = ndimage.label(mask_np > 0)
    
    if debug:
        print(f"   Final connected components: {num_components}")
    
    boxes = []
    labels = []
    
    # Extract box for each component
    for component_id in range(1, num_components + 1):
        component_mask = (labeled_mask == component_id)
        voxel_count = component_mask.sum()
        
        if voxel_count < min_size:
            continue
            
        # Get bounding box coordinates
        coords = np.argwhere(component_mask)  # Returns (N, 3) with indices [dim0, dim1, dim2]
        if len(coords) == 0:
            continue
            
        # Min and max for each dimension
        mins = coords.min(axis=0)  # [min_dim0, min_dim1, min_dim2]
        maxs = coords.max(axis=0)  # [max_dim0, max_dim1, max_dim2]
        
        # Convert to xyzxyz format
        # MONAI expects boxes as (x1, y1, z1, x2, y2, z2)
        x1, y1, z1 = float(mins[0]), float(mins[1]), float(mins[2])
        x2, y2, z2 = float(maxs[0] + 1), float(maxs[1] + 1), float(maxs[2] + 1)  # +1 for exclusive end
        
        if debug and component_id == 1:
            print(f"\n   Component {component_id} (voxels: {voxel_count}):")
            print(f"      Coordinate mins: {mins}")
            print(f"      Coordinate maxs: {maxs}")
            print(f"      Box (xyzxyz): [{x1}, {y1}, {z1}, {x2}, {y2}, {z2}]")
            print(f"      Box size: {x2-x1} x {y2-y1} x {z2-z1}")
        
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

def generate_train_transform(patch_size, batch_size, amp=True, debug=False):
    """Generate training transform"""
    
    compute_dtype = torch.float16 if amp else torch.float32
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=120,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Random crop to patch_size (no resize!)
        # This crops a random patch from the original image
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=patch_size,
            random_center=True,
            random_size=False,
        ),
        # Spatial augmentation (on image and label together)
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
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
        # After augmentation, extract boxes from augmented label
        GenerateBoxMaskd(
            keys=["label"],
            image_key="image",
            label_key="label",
            box_key="box",
            label_class_key="label_class",
            mask_key="mask_image",
            min_size=10,  # Back to original value
            merge_distance_mm=0,  # Disable merge (causes issues with spacing)
            debug=debug,
        ),
        # Note: Boxes extracted from mask are already in image coordinates (pixel indices)
        # No need for AffineBoxToImageCoordinated transform
        StandardizeEmptyBoxd(box_keys=["box"], box_ref_image_keys="image"),
        # Final type conversion
        EnsureTyped(keys=["image"], dtype=compute_dtype),
        EnsureTyped(keys=["label_class"], dtype=torch.long),
        # Clean up
        DeleteItemsd(keys=["label", "mask_image"]),
    ])
    
    return train_transforms


def generate_val_transform(amp=True, debug=False):
    """Generate validation transform"""
    
    compute_dtype = torch.float16 if amp else torch.float32
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=120,
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
            min_size=10,
            merge_distance_mm=0,  # Disable merge (causes issues with spacing)
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


def generate_test_transform(amp=True):
    """Generate test/inference transform (image only, no labels required)"""
    
    compute_dtype = torch.float16 if amp else torch.float32
    
    test_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False, meta_key_postfix="meta_dict"),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=120,
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
        # batch_data is a list from no_collation: [{dict}]
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
        
        # Convert to int and clip to valid range
        x1 = int(max(0, min(x1, spatial_shape[0])))
        x2 = int(max(0, min(x2, spatial_shape[0])))
        y1 = int(max(0, min(y1, spatial_shape[1])))
        y2 = int(max(0, min(y2, spatial_shape[1])))
        z1 = int(max(0, min(z1, spatial_shape[2])))
        z2 = int(max(0, min(z2, spatial_shape[2])))
        
        # Fill box region - assuming array order is [x, y, z]
        if x2 > x1 and y2 > y1 and z2 > z1:
            mask[x1:x2, y1:y2, z1:z2] = 1
    
    return mask


def validate(detector, loader, device, coco_metric, amp=True, verbose=True, save_debug=False, output_dir=None, epoch=0):
    """Validate with AMP support"""
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
            
            # Inference with AMP
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = detector(inputs, use_inferer=True)
            
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
    
    # Debug: Print keys to understand the output structure
    if verbose and len(val_outputs_all) > 0:
        print(f"\n🔍 Debug - Output keys: {val_outputs_all[0].keys()}")
        print(f"🔍 Debug - Target keys: {val_targets_all[0].keys()}")
        # Check if predictions are empty
        for key in val_outputs_all[0].keys():
            val = val_outputs_all[0][key]
            if isinstance(val, torch.Tensor):
                print(f"   - {key}: shape {val.shape}")
    
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
    
    # Print detection stats
    total_pred = sum(len(p) for p in pred_boxes_list)
    total_gt = sum(len(g) for g in gt_boxes_list)
    if verbose:
        print(f"\n📊 Detection Summary:")
        print(f"   - Total predictions: {total_pred}")
        print(f"   - Total ground truth: {total_gt}")
        print(f"   - Images with detections: {sum(1 for p in pred_boxes_list if len(p) > 0)}/{len(pred_boxes_list)}")
    
    results_metric = matching_batch(
        iou_fn=box_utils.box_iou,
        iou_thresholds=coco_metric.iou_thresholds,
        pred_boxes=pred_boxes_list,
        pred_classes=pred_classes_list,
        pred_scores=pred_scores_list,
        gt_boxes=gt_boxes_list,
        gt_classes=gt_classes_list,
    )
    
    # COCOMetric computation (following original detection/luna16_training.py)
    try:
        # Pass matching_batch results to coco_metric
        # coco_metric expects the output from matching_batch
        val_epoch_metric_dict = coco_metric(results_metric)[0]
        metric_results = val_epoch_metric_dict
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Warning: COCOMetric failed: {e}")
            print(f"   Using basic metrics only")
        
        # Fallback: compute basic precision/recall
        metric_results = {}
        for iou_thresh in coco_metric.iou_thresholds:
            # Extract TP, FP, FN from results_metric at this IoU threshold
            tp_count = 0
            fp_count = 0
            fn_count = 0
            
            for result_per_image in results_metric:
                if isinstance(result_per_image, dict):
                    tp_count += result_per_image.get(f'tp_{iou_thresh}', 0)
                    fp_count += result_per_image.get(f'fp_{iou_thresh}', 0)
                    fn_count += result_per_image.get(f'fn_{iou_thresh}', 0)
            
            # Calculate precision and recall
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            
            metric_results[f"precision_IoU_{iou_thresh}"] = precision
            metric_results[f"recall_IoU_{iou_thresh}"] = recall
        
        # Add average metric
        if metric_results:
            avg_metric = sum(metric_results.values()) / len(metric_results)
            metric_results["avg_metric"] = avg_metric
    
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
        
        if verbose:
            print(f"\n📐 Shape info:")
            print(f"   - Current shape: {current_shape}")
            print(f"   - Original shape: {original_shape}")
            print(f"   - Affine available: {affine is not None}")
            print(f"   - NOTE: Interpretation - shape axis order matches box (x,y,z) directly")
        
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
        
        if verbose:
            print(f"\n💾 Debug samples saved to: {debug_dir}")
            print(f"   - Current shape: {current_shape}, Original shape: {tuple(original_shape)}")
            print(f"   - Segmentation saved: {'Yes' if seg_label is not None else 'No'}")
            print(f"   - GT boxes: {len(gt_boxes)}")
            print(f"   - Pred boxes: {len(pred_boxes)}")
            if len(pred_boxes) > 0:
                print(f"   - Score range: {pred_scores.min():.3f} - {pred_scores.max():.3f}")
    
    return results_metric, metric_results


# ============================================================================
# Testing / Inference
# ============================================================================

def test(detector, test_loader, device, test_data_dicts, output_dir, amp=True, save_predictions=False):
    """Run inference on test data"""
    detector.eval()
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Running inference on {len(test_loader)} images...")
    print(f"{'='*60}\n")
    
    # Debug flag
    debug_printed = False
    
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader, desc="[Test]")):
            inputs = [batch_data_i["image"].to(device) for batch_data_i in batch_data]
            
            # Get original image path
            image_path = test_data_dicts[idx]["image"]
            image_name = os.path.basename(image_path)
            
            # Inference with AMP
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = detector(inputs, use_inferer=True)
            
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
            
            # Print summary for each image
            if len(pred_boxes) > 0:
                print(f"  [{idx+1}/{len(test_loader)}] {image_name}: {len(pred_boxes)} detections (scores: {pred_scores.min():.3f}-{pred_scores.max():.3f})")
            else:
                print(f"  [{idx+1}/{len(test_loader)}] {image_name}: No detections")
    
    # Save predictions to JSON
    if save_predictions:
        output_file = os.path.join(output_dir, "predictions.json")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Predictions saved to: {output_file}")
    
    # Print summary statistics
    total_detections = sum(r["num_detections"] for r in all_results)
    images_with_detections = sum(1 for r in all_results if r["num_detections"] > 0)
    
    print(f"\n{'='*60}")
    print(f"Inference Summary:")
    print(f"  - Total images: {len(all_results)}")
    print(f"  - Images with detections: {images_with_detections}/{len(all_results)}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Avg detections per image: {total_detections/len(all_results):.2f}")
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
# Main
# ============================================================================

def main(rank=0, world_size=1):
    parser = argparse.ArgumentParser(description='3D Detection with MONAI RetinaNet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--image_dir', type=str, default='/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/image')
    parser.add_argument('--label_dir', type=str, default='/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/mask')
    parser.add_argument('--output_dir', type=str, default='./outputs_detection')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[512, 512, 16], help='Training patch size (W,H,D)')
    parser.add_argument('--val_patch_size', type=int, nargs=3, default=[512, 512, 16], help='Validation patch size (W,H,D)')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate (default: 5e-3 for batch_size=8)')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of foreground classes (excluding background)')
    parser.add_argument('--multi_gpu', action='store_true', help='Use DDP for multi-GPU')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision (default: enabled)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    # Test mode arguments
    parser.add_argument('--test_image_dir', type=str, default=None, help='Test image directory (if different from image_dir)')
    parser.add_argument('--test_label_dir', type=str, default=None, help='Test label directory (optional, for evaluation)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction results to JSON')
    parser.add_argument('--score_thresh', type=float, default=0.02, help='Score threshold for detection (lower = more detections)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for bbox extraction')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'efficientnet-b3'],
                        help='Backbone architecture (resnet50 or efficientnet-b3). Note: efficientnet-b3 needs custom FPN')
    args = parser.parse_args()
    
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
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"No checkpoint found at {checkpoint_path}. Please specify --checkpoint")
            args.checkpoint = checkpoint_path
        
        if is_main_process:
            print(f"Loading model from: {args.checkpoint}")
        
        # Test data directory
        test_image_dir = args.test_image_dir if args.test_image_dir else args.image_dir
        test_label_dir = args.test_label_dir  # Can be None
        
        # Test transform (image only, no labels required)
        test_transforms = generate_test_transform(amp=not args.no_amp)
        
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
        
        # Build model (same as training)
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**l for l in range(4)],
            base_anchor_shapes=[[30, 30, 3], [40, 40, 4], [50, 50, 5]],
        )
        
        # Build backbone based on args
        if args.backbone == 'efficientnet-b3':
            backbone = EfficientNetBN(
                model_name='efficientnet-b3',
                spatial_dims=3,
                in_channels=1,
                num_classes=1,
                pretrained=False,
            )
        else:
            conv1_t_stride = [2, 2, 1]
            conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]
            
            backbone = resnet.ResNet(
                block=resnet.ResNetBottleneck,
                layers=[3, 4, 6, 3],
                block_inplanes=resnet.get_inplanes(),
                n_input_channels=1,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=conv1_t_size,
            )
        
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=3,
            pretrained_backbone=False,
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
            print(f"\nInference parameters:")
            print(f"  - Score threshold: {args.score_thresh}")
            print(f"  - NMS threshold: 0.22")
            print(f"  - Max detections per image: 100\n")
        
        detector.set_box_selector_parameters(
            score_thresh=args.score_thresh,
            topk_candidates_per_level=1000,
            nms_thresh=0.22,
            detections_per_img=20,
        )
        detector.set_sliding_window_inferer(
            roi_size=args.val_patch_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        detector.network.load_state_dict(checkpoint['model_state_dict'])
        
        if is_main_process:
            print(f"✅ Model loaded successfully")
            if 'epoch' in checkpoint:
                print(f"   - Trained epoch: {checkpoint['epoch']+1}")
            if 'best_metric' in checkpoint:
                print(f"   - Best metric: {checkpoint['best_metric']:.4f}")
        
        # Run inference
        results = test(
            detector,
            test_loader,
            device,
            test_data_dicts,
            args.output_dir,
            amp=not args.no_amp,
            save_predictions=args.save_predictions
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
        debug=args.debug
    )
    val_transforms = generate_val_transform(amp=not args.no_amp, debug=args.debug)
    
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
    # 1) Anchor generator
    # Try to load anchor shapes from EDA results
    eda_file = os.path.join(args.output_dir, '..', 'eda', 'dataset.json')
    base_anchor_shapes = None
    
    if os.path.exists(eda_file):
        try:
            with open(eda_file, 'r') as f:
                eda_data = json.load(f)
            
            # Load recommended anchor shapes from EDA
            anchor_shapes_raw = eda_data['recommended_anchors']['anchor_shapes']
            
            # Divide by feature map stride (4) to get feature map scale
            # Assuming feature map stride = 4
            feature_stride = 4
            base_anchor_shapes = [
                [max(1, int(w // feature_stride)), 
                 max(1, int(h // feature_stride)), 
                 max(1, int(d // feature_stride))]
                for w, h, d in anchor_shapes_raw
            ]
            
            if is_main_process:
                print(f"\n🎯 Loaded anchor shapes from EDA:")
                print(f"   EDA file: {eda_file}")
                print(f"   Original sizes (pixels):")
                for i, shape in enumerate(anchor_shapes_raw):
                    print(f"      {i}: {shape}")
                print(f"   Feature map anchors (stride={feature_stride}):")
                for i, shape in enumerate(base_anchor_shapes):
                    print(f"      {i}: {shape}")
        except Exception as e:
            if is_main_process:
                print(f"⚠️  Could not load EDA results: {e}")
                print(f"   Using default anchor shapes")
    
    # Use default anchor shapes if EDA not available
    if base_anchor_shapes is None:
        if is_main_process:
            print(f"\n📌 Using default anchor shapes (run eda_dataset.py to optimize)")
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
        print(f"  - Patch size (val): {args.val_patch_size}")
        print(f"  - AMP: {not args.no_amp}")
        print(f"{'='*60}\n")
    
    # 2) Network: Backbone + FPN
    if args.backbone == 'efficientnet-b3':
        if is_main_process:
            print(f"\n🔧 Using EfficientNetBN-B3 backbone")
        
        backbone = EfficientNetBN(
            model_name='efficientnet-b3',
            spatial_dims=3,
            in_channels=1,
            num_classes=1,  # Dummy, will be replaced by FPN
            pretrained=False,
        )
        
        # EfficientNetBN용 FPN
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=3,
            pretrained_backbone=False,
            trainable_backbone_layers=None,
            returned_layers=[1, 2, 3],
        )
    else:
        # ResNet50 (default)
        if is_main_process:
            print(f"\n🔧 Using ResNet50 backbone")
        
        conv1_t_stride = [2, 2, 1]
        conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]
        
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
            pretrained_backbone=False,
            trainable_backbone_layers=None,
            returned_layers=[1, 2, 3],  # Use layers 1,2,3 for large lesions
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
    detector.set_sliding_window_inferer(
        roi_size=args.val_patch_size,
        overlap=0.125,  # Reduced overlap for faster inference
        sw_batch_size=4,  # Increased batch size for faster inference
        mode="constant",
        device="cpu",
    )
    
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
    
    # Load pretrained weights if best_model.pth exists
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        if is_main_process:
            print(f"\n🔄 Loading pretrained weights from: {best_model_path}")
        
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        
        # Load only model weights (not optimizer)
        det_network = detector.module.network if hasattr(detector, 'module') else detector.network
        det_network.load_state_dict(checkpoint['model_state_dict'])
        
        if is_main_process:
            print(f"   ✅ Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"   ✅ Previous best metric: {checkpoint.get('best_metric', 0.0):.4f}")
            print(f"   ⚠️  Optimizer reset (starting fresh)\n")
    
    if is_main_process:
        print(f"\nStarting training for {args.epochs} epochs\n")
    
    best_metric = 0.0
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
            print(f"Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, box: {train_box_loss:.4f})")
            
            writer.add_scalar("train/total_loss", train_loss, epoch)
            writer.add_scalar("train/cls_loss", train_cls_loss, epoch)
            writer.add_scalar("train/box_loss", train_box_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        
        # Validate (only rank 0, no DistributedSampler = no deadlock)
        if (epoch + 1) % args.val_interval == 0:
            # Synchronize before validation
            if is_ddp:
                dist.barrier()
            
            if is_main_process:
                print("\nRunning validation...")
                
                # Use unwrapped model for validation (rank 0 only)
                val_detector = detector.module if is_ddp else detector
                
                results, metric_results = validate(
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
                
                # Print metrics (only key metrics, not all)
                print(f"\n📈 Validation Results:")
                
                # Main metrics to display
                key_metrics = [
                    "mAP_IoU_0.10_0.50_0.05_MaxDet_100",
                    "AP_IoU_0.10_MaxDet_100",
                    "AP_IoU_0.30_MaxDet_100", 
                    "AP_IoU_0.50_MaxDet_100",
                    "mAR_IoU_0.10_0.50_0.05_MaxDet_100",
                ]
                
                # Print only key metrics
                for k in key_metrics:
                    if k in metric_results:
                        print(f"  {k}: {metric_results[k]:.4f}")
                
                # Save all metrics to tensorboard
                for k, v in metric_results.items():
                    if writer is not None:
                        writer.add_scalar(f"val/{k}", v, epoch)
                
                # Calculate average metric (following original detection/luna16_training.py)
                if len(metric_results) > 0:
                    avg_metric = sum(metric_results.values()) / len(metric_results)
                else:
                    avg_metric = 0.0
                
                print(f"  avg_metric: {avg_metric:.4f}")
                
                if writer is not None:
                    writer.add_scalar("val/avg_metric", avg_metric, epoch)
                
                # Save best model
                if avg_metric > best_metric:
                    best_metric = avg_metric
                    best_epoch = epoch
                    save_path = os.path.join(args.output_dir, "best_model.pth")
                    # Get the actual network (handle DDP wrapping)
                    save_network = detector.module.network if hasattr(detector, 'module') else detector.network
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': save_network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_metric': best_metric,
                    }, save_path)
                    print(f"✅ Saved best model (avg metric: {best_metric:.4f})")
            
            # Synchronize after validation
            if is_ddp:
                dist.barrier()
        
        # Sync all processes before next epoch
        if is_ddp:
            dist.barrier()
        
        # Step scheduler
        scheduler.step()
        
        # Note: Epoch checkpoints are disabled. Only best_model.pth is saved.
    
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best avg metric: {best_metric:.4f} at epoch {best_epoch+1}")
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
