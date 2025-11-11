#!/usr/bin/env python3
"""
Anchor Shape Optimization using K-Means Clustering
Based on YOLO anchor optimization approach
"""

import os
import json
import numpy as np
from glob import glob
import nibabel as nib
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import yaml


def extract_bboxes_from_mask(mask_np, min_size=100, spacing=None, merge_distance_mm=20.0, return_mask=False):
    """
    Extract bounding boxes from segmentation mask
    
    Pipeline:
        1. Binary mask creation
        2. Initial cluster analysis (connected components)
        3. Merge nearby clusters (morphological closing)
        4. Filter by minimum size (remove small merged clusters)
        5. Final cluster analysis
        6. Box extraction
    
    Args:
        mask_np: Segmentation mask (H, W, D)
        min_size: Minimum volume in voxels AFTER merging (default: 100 voxels)
        spacing: Voxel spacing [x, y, z] in mm (for merging nearby lesions)
        merge_distance_mm: Merge lesions within this distance (mm)
        return_mask: If True, also return the labeled mask after merging
    
    Returns:
        boxes: List of [width, height, depth] in voxels
        labeled_mask: (optional) Labeled mask after merging and filtering
    """
    from scipy import ndimage
    
    # Step 1: Binary mask creation
    binary_mask = (mask_np > 0).astype(np.uint8)
    struct_elem = ndimage.generate_binary_structure(3, 1)  # 3D connectivity
    
    # Step 2: Initial cluster analysis (connected components)
    # This identifies individual lesions before merging
    initial_labeled, initial_num = ndimage.label(binary_mask, structure=struct_elem)
    
    # Step 3: Merge nearby clusters (if requested)
    if spacing is not None and merge_distance_mm > 0:
        # Calculate dilation distance in voxels for each dimension
        # Use half of merge_distance_mm for dilation (the other half comes from the other lesion)
        dilation_voxels = np.round(merge_distance_mm / (2.0 * np.array(spacing))).astype(int)
        dilation_voxels = np.maximum(dilation_voxels, 1)  # At least 1 voxel
        
        # CRITICAL FIX: Use TRUE anisotropic dilation (different iterations per axis)
        # Depth spacing is usually much larger (e.g., 4.8mm) than X/Y spacing (e.g., 0.4mm)
        # So we need: X=13 iterations, Y=13 iterations, Z=1 iteration
        
        # Anisotropic morphological closing: dilate each axis separately
        dilated = binary_mask.copy()
        
        # Dilate along each axis with its specific iteration count
        for axis in range(3):
            iterations = int(dilation_voxels[axis])
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
                
                # Apply dilation along this axis
                for _ in range(iterations):
                    dilated = ndimage.binary_dilation(dilated, structure=struct_1d)
        
        # Erosion: use ANISOTROPIC erosion (same as dilation but reversed)
        # To preserve small lesions, we erode each axis separately with 50% of dilation iterations
        eroded = dilated.copy()
        
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
                    eroded = ndimage.binary_erosion(eroded, structure=struct_1d)
        
        # Use merged mask for further processing
        binary_mask = eroded.astype(np.uint8)
    
    # Step 4 & 5: Final cluster analysis and size filtering
    labeled_mask, num_labels = ndimage.label(binary_mask, structure=struct_elem)
    
    # Step 6: Box extraction with size filtering
    boxes = []
    final_labeled = np.zeros_like(labeled_mask)
    new_label_id = 1
    
    for label_id in range(1, num_labels + 1):
        component_mask = (labeled_mask == label_id)
        voxel_count = component_mask.sum()
        
        # Step 4: Filter by minimum voxel count AFTER merging
        # Small clusters that merged together can now pass the threshold
        if voxel_count < min_size:
            continue
        
        coords = np.argwhere(component_mask)
        
        if len(coords) == 0:
            continue
        
        # coords is (N, 3) with order matching np.argwhere output
        # For 3D array: (axis0, axis1, axis2) = (H, W, D) in our case
        h_coords = coords[:, 0]  # Height (Y)
        w_coords = coords[:, 1]  # Width (X)
        d_coords = coords[:, 2]  # Depth (Z)
        
        x_min, x_max = w_coords.min(), w_coords.max()
        y_min, y_max = h_coords.min(), h_coords.max()
        z_min, z_max = d_coords.min(), d_coords.max()
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        depth = z_max - z_min + 1
        
        boxes.append([width, height, depth])
        final_labeled[component_mask] = new_label_id
        new_label_id += 1
    
    if return_mask:
        return boxes, final_labeled
    return boxes


def iou_width_height_depth(boxes, clusters):
    """
    Calculate IoU between boxes and cluster centers
    boxes: (N, 3) array of [w, h, d]
    clusters: (K, 3) array of [w, h, d]
    Returns: (N, K) array of IoU values
    """
    # Expand dimensions for broadcasting
    boxes = np.expand_dims(boxes, axis=1)  # (N, 1, 3)
    clusters = np.expand_dims(clusters, axis=0)  # (1, K, 3)
    
    # Calculate intersection (minimum of each dimension)
    intersection = np.minimum(boxes, clusters).prod(axis=2)  # (N, K)
    
    # Calculate union
    box_area = boxes.prod(axis=2)  # (N, K)
    cluster_area = clusters.prod(axis=2)  # (N, K)
    union = box_area + cluster_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)
    return iou


def kmeans_anchors(boxes, n_clusters=5, max_iter=300, verbose=True):
    """
    Run K-Means clustering on bounding boxes to find optimal anchors
    
    Args:
        boxes: List or array of [w, h, d] for each lesion
        n_clusters: Number of anchor shapes to generate
        max_iter: Maximum iterations for K-Means
        verbose: Print progress
    
    Returns:
        anchors: (n_clusters, 3) array of optimal anchor shapes
        avg_iou: Average IoU between boxes and their nearest anchors
    """
    boxes = np.array(boxes, dtype=np.float32)
    
    if verbose:
        print(f"\n🔍 Running K-Means clustering with {n_clusters} clusters...")
        print(f"   Total lesions: {len(boxes)}")
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init=10)
    kmeans.fit(boxes)
    
    # Get cluster centers (anchors)
    anchors = kmeans.cluster_centers_
    
    # Sort anchors by size (volume)
    volumes = anchors.prod(axis=1)
    sorted_idx = np.argsort(volumes)
    anchors = anchors[sorted_idx]
    
    # Calculate average IoU
    iou = iou_width_height_depth(boxes, anchors)
    best_iou = iou.max(axis=1)
    avg_iou = best_iou.mean()
    
    if verbose:
        print(f"   ✅ Converged in {kmeans.n_iter_} iterations")
        print(f"   ✅ Average IoU: {avg_iou:.4f}")
    
    return anchors, avg_iou


def analyze_dataset(image_dir, label_dir, min_size=100, merge_distance_mm=20.0):
    """Analyze dataset and extract all lesion sizes
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        min_size: Minimum lesion volume in voxels (default: 100 for 3D cluster)
        merge_distance_mm: Merge lesions within this distance (mm)
    """
    
    print(f"\n{'='*60}")
    print(f"Anchor Shape Optimization using K-Means Clustering")
    print(f"{'='*60}\n")
    
    # Find all label files
    label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
    
    if len(label_files) == 0:
        raise ValueError(f"No .nii.gz files found in {label_dir}")
    
    print(f"📁 Dataset:")
    print(f"   - Label directory: {label_dir}")
    print(f"   - Total images: {len(label_files)}")
    print(f"   - Minimum lesion volume: {min_size} voxels (3D cluster)")
    print(f"   - Merge distance: {merge_distance_mm} mm\n")
    
    # Extract all lesion sizes and image shapes
    all_boxes = []
    all_spacings = []
    all_image_shapes = []
    
    print("📊 Extracting and merging lesions...")
    for label_path in tqdm(label_files, desc="Processing"):
        label_nii = nib.load(label_path)
        label_np = label_nii.get_fdata()
        spacing = label_nii.header.get_zooms()[:3]  # (x, y, z) spacing
        image_shape = label_np.shape  # (H, W, D)
        
        # Store image shape
        all_image_shapes.append(image_shape)
        
        # Extract boxes with merging
        boxes = extract_bboxes_from_mask(
            label_np, 
            min_size=min_size,
            spacing=spacing,
            merge_distance_mm=merge_distance_mm
        )
        
        if len(boxes) > 0:
            all_boxes.extend(boxes)
            all_spacings.extend([spacing] * len(boxes))
    
    all_boxes = np.array(all_boxes, dtype=np.float32)
    all_spacings = np.array(all_spacings, dtype=np.float32)
    all_image_shapes = np.array(all_image_shapes, dtype=np.int32)
    
    print(f"\n✅ Extraction complete:")
    print(f"   - Total images: {len(all_image_shapes)}")
    print(f"   - Total lesions found: {len(all_boxes)}")
    if merge_distance_mm > 0:
        print(f"   - Note: Nearby lesions within {merge_distance_mm}mm were merged")
    
    if len(all_boxes) == 0:
        raise ValueError("No lesions found! Check min_size or data.")
    
    return all_boxes, all_spacings, all_image_shapes


def recommend_patch_size(image_shapes, lesion_sizes, spacings):
    """
    Recommend optimal patch_size for training and validation
    
    Args:
        image_shapes: (N, 3) array of image shapes [H, W, D]
        lesion_sizes: (M, 3) array of lesion sizes [W, H, D]
        spacings: (M, 3) array of voxel spacings
    
    Returns:
        train_patch_size: [W, H, D] for training
        val_patch_size: [W, H, D] for validation (or None for full image)
    """
    print(f"\n{'='*70}")
    print(f"📐 Patch Size Recommendation")
    print(f"{'='*70}\n")
    
    # 1. Image shape statistics
    median_img_shape = np.median(image_shapes, axis=0).astype(int)
    percentile_75_img = np.percentile(image_shapes, 75, axis=0).astype(int)
    percentile_90_img = np.percentile(image_shapes, 90, axis=0).astype(int)
    
    print(f"📊 Image Shape Analysis:")
    print(f"   - Median:       {tuple(median_img_shape)} (H, W, D)")
    print(f"   - 75th %ile:    {tuple(percentile_75_img)}")
    print(f"   - 90th %ile:    {tuple(percentile_90_img)}")
    
    # 2. Lesion size statistics (in voxels)
    median_lesion = np.median(lesion_sizes, axis=0).astype(int)
    percentile_75_lesion = np.percentile(lesion_sizes, 75, axis=0).astype(int)
    percentile_90_lesion = np.percentile(lesion_sizes, 90, axis=0).astype(int)
    
    print(f"\n📊 Lesion Size Analysis:")
    print(f"   - Median:       {tuple(median_lesion)} (W, H, D)")
    print(f"   - 75th %ile:    {tuple(percentile_75_lesion)}")
    print(f"   - 90th %ile:    {tuple(percentile_90_lesion)}")
    
    # Convert to physical size (mm)
    median_spacing = np.median(spacings, axis=0)
    median_lesion_mm = median_lesion * median_spacing
    
    print(f"\n   Physical size (median):")
    print(f"   - Lesion: {median_lesion_mm[0]:.1f} × {median_lesion_mm[1]:.1f} × {median_lesion_mm[2]:.1f} mm")
    
    # 3. Training patch size recommendation
    # Rule: 1.5-2x median lesion size to include context
    train_patch_raw = (percentile_75_lesion * 1.5).astype(int)
    
    # Round to nearest multiple of 16 for GPU efficiency
    def round_to_multiple(x, base=16):
        return int(np.round(x / base) * base)
    
    train_patch_size = np.array([
        round_to_multiple(train_patch_raw[0]),  # Width
        round_to_multiple(train_patch_raw[1]),  # Height
        max(round_to_multiple(train_patch_raw[2], base=4), 16)  # Depth (at least 16)
    ])
    
    # Ensure minimum size
    train_patch_size = np.maximum(train_patch_size, [128, 128, 16])
    
    # 4. Validation patch size recommendation
    # Rule: Use full image if median < 640, else use slightly larger than training
    if np.all(median_img_shape <= 640):
        val_patch_size = None  # Use full image
        val_patch_desc = f"{tuple(median_img_shape)} (full image)"
    else:
        val_patch_size = np.minimum(train_patch_size * 1.5, median_img_shape).astype(int)
        val_patch_size = np.array([
            round_to_multiple(val_patch_size[0]),
            round_to_multiple(val_patch_size[1]),
            round_to_multiple(val_patch_size[2], base=4)
        ])
        val_patch_desc = f"{tuple(val_patch_size)}"
    
    print(f"\n🎯 Recommended Patch Sizes:")
    print(f"   - Training:   {tuple(train_patch_size)} (W, H, D)")
    print(f"     └─ Rationale: 1.5× 75th %ile lesion size, rounded to multiples of 16")
    print(f"   - Validation: {val_patch_desc}")
    print(f"     └─ Rationale: {'Full image (median < 640)' if val_patch_size is None else 'Larger than training'}")
    
    # Convert to [W, H, D] order for config.yaml
    train_patch_list = train_patch_size.tolist()
    val_patch_list = val_patch_size.tolist() if val_patch_size is not None else median_img_shape.tolist()
    
    print(f"\n📋 For config.yaml:")
    print(f"   patch_size: {train_patch_list}  # Training")
    print(f"   # Validation uses full image (direct inference)")
    
    return train_patch_list, val_patch_list, {
        'median_image_shape': median_img_shape.tolist(),
        'median_lesion_size_voxels': median_lesion.tolist(),
        'median_lesion_size_mm': median_lesion_mm.tolist(),
        'p75_lesion_size': percentile_75_lesion.tolist(),
        'p90_lesion_size': percentile_90_lesion.tolist(),
    }


def find_optimal_k(boxes, k_range=(3, 10), max_iter=300):
    """Find optimal number of clusters using elbow method"""
    print(f"\n🔍 Finding optimal number of anchors (K={k_range[0]}-{k_range[1]})...")
    
    k_values = range(k_range[0], k_range[1] + 1)
    avg_ious = []
    
    for k in k_values:
        _, avg_iou = kmeans_anchors(boxes, n_clusters=k, max_iter=max_iter, verbose=False)
        avg_ious.append(avg_iou)
        print(f"   K={k}: Avg IoU = {avg_iou:.4f}")
    
    # Find elbow point (diminishing returns)
    # Simple heuristic: largest improvement drop
    improvements = np.diff(avg_ious)
    elbow_idx = np.argmin(improvements) + 1
    optimal_k = k_values[elbow_idx]
    
    print(f"\n✅ Optimal K (elbow method): {optimal_k}")
    print(f"   IoU improvement drops after K={optimal_k}")
    
    return optimal_k, k_values, avg_ious


def visualize_anchors(boxes, anchors, spacings, output_dir):
    """Visualize anchor distribution"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to physical size (mm)
    boxes_mm = boxes * spacings
    anchors_mm = anchors * np.median(spacings, axis=0)
    
    # Create 2x2 layout for better visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: 3D scatter (width vs height vs depth)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(boxes[:, 0], boxes[:, 1], boxes[:, 2], alpha=0.3, s=10, c='blue', label='Lesions (voxels)')
    ax1.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], s=200, c='red', marker='*', 
                edgecolors='black', linewidths=2, label='Anchors')
    ax1.set_xlabel('Width (voxels)')
    ax1.set_ylabel('Height (voxels)')
    ax1.set_zlabel('Depth (voxels)')
    ax1.set_title('Lesion Distribution & Anchors (Voxel Space)')
    ax1.legend()
    
    # Plot 2: Width vs Height (Physical Space)
    ax2 = fig.add_subplot(222)
    ax2.scatter(boxes_mm[:, 0], boxes_mm[:, 1], alpha=0.3, s=10, c='blue', label='Lesions')
    ax2.scatter(anchors_mm[:, 0], anchors_mm[:, 1], s=200, c='red', marker='*', 
                edgecolors='black', linewidths=2, label='Anchors')
    ax2.set_xlabel('Width (mm)')
    ax2.set_ylabel('Height (mm)')
    ax2.set_title('Width vs Height (Physical Space)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Depth distribution (NEW!)
    ax3 = fig.add_subplot(223)
    depths_mm = boxes_mm[:, 2]  # Depth in mm
    anchor_depths_mm = anchors_mm[:, 2]
    ax3.hist(depths_mm, bins=50, alpha=0.7, color='blue', label='Lesions', edgecolor='black')
    ax3.axvline(np.median(depths_mm), color='blue', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(depths_mm):.1f}mm')
    for i, depth in enumerate(anchor_depths_mm):
        ax3.axvline(depth, color='red', linestyle='-', linewidth=2)
    ax3.set_xlabel('Depth (mm)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Depth Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volume distribution
    ax4 = fig.add_subplot(224)
    volumes_mm = boxes_mm.prod(axis=1)
    anchor_volumes_mm = anchors_mm.prod(axis=1)
    ax4.hist(volumes_mm, bins=50, alpha=0.7, color='blue', label='Lesions', edgecolor='black')
    median_vol = np.median(volumes_mm)
    ax4.axvline(median_vol, color='blue', linestyle='--', linewidth=2,
                label=f'Median: {median_vol:.0f}mm³ ({median_vol/1000:.2f}cc)')
    for i, vol in enumerate(anchor_volumes_mm):
        ax4.axvline(vol, color='red', linestyle='-', linewidth=2)
    ax4.set_xlabel('Volume (mm³ / cc)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Volume Distribution')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anchor_optimization.png'), dpi=150)
    print(f"\n📊 Visualization saved: {output_dir}/anchor_optimization.png")
    plt.close()


def create_box_mask(boxes, labeled_mask, shape):
    """
    Create mask showing bounding boxes (not the original mask)
    
    Args:
        boxes: List of [width, height, depth] (not used, we extract boxes from labeled_mask)
        labeled_mask: Labeled mask after merging
        shape: Output shape
    
    Returns:
        box_mask: Mask with rectangular bounding boxes filled
    """
    box_mask = np.zeros(shape, dtype=np.uint8)
    
    # For each label, find its bounding box and fill that rectangular region
    for label_id in range(1, len(boxes) + 1):
        binary_mask = (labeled_mask == label_id).astype(np.uint8)
        coords = np.argwhere(binary_mask > 0)
        
        if len(coords) == 0:
            continue
        
        # Get bounding box coordinates
        # coords is (N, 3) with order (H, W, D)
        h_coords = coords[:, 0]  # Height (Y)
        w_coords = coords[:, 1]  # Width (X)
        d_coords = coords[:, 2]  # Depth (Z)
        
        h_min, h_max = h_coords.min(), h_coords.max()
        w_min, w_max = w_coords.min(), w_coords.max()
        d_min, d_max = d_coords.min(), d_coords.max()
        
        # Fill the bounding box region (rectangular region, not the original mask)
        box_mask[h_min:h_max+1, w_min:w_max+1, d_min:d_max+1] = label_id
    
    return box_mask


def save_sample_cases(image_dir, label_dir, output_dir, min_size, merge_distance_mm, num_samples=5):
    """
    Save sample cases showing box merging before/after as NIfTI files
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        output_dir: Output directory for samples
        min_size: Minimum lesion volume (voxels)
        merge_distance_mm: Merge distance (mm)
        num_samples: Number of samples to save (default: 5)
    """
    print(f"\n{'='*70}")
    print("Sample Case Visualization: Box Merging Before/After")
    print(f"{'='*70}\n")
    
    # Create output directory
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Find label files
    label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))[:num_samples]
    
    if len(label_files) == 0:
        print(f"❌ No label files found in {label_dir}")
        return
    
    print(f"📁 Settings:")
    print(f"   - Image directory: {image_dir}")
    print(f"   - Label directory: {label_dir}")
    print(f"   - Output directory: {samples_dir}")
    print(f"   - Number of samples: {len(label_files)}")
    print(f"   - Min size: {min_size} voxels")
    print(f"   - Merge distance: {merge_distance_mm} mm\n")
    
    print(f"💾 Processing {len(label_files)} samples...\n")
    
    for idx, label_path in enumerate(label_files):
        sample_name = os.path.basename(label_path).replace('.nii.gz', '')
        print(f"  [{idx+1}/{len(label_files)}] {sample_name}")
        
        # Load label
        label_nii = nib.load(label_path)
        label_np = label_nii.get_fdata()
        spacing = label_nii.header.get_zooms()[:3]
        affine = label_nii.affine
        
        # Try to load corresponding image
        image_path = os.path.join(image_dir, os.path.basename(label_path))
        if os.path.exists(image_path):
            image_nii = nib.load(image_path)
            image_np = image_nii.get_fdata()
        else:
            image_np = None
            print(f"      ⚠️  Image not found: {image_path}")
        
        # Extract boxes WITHOUT merging
        boxes_before, mask_before = extract_bboxes_from_mask(
            label_np,
            min_size=min_size,
            spacing=spacing,
            merge_distance_mm=0,  # No merging
            return_mask=True
        )
        
        # Extract boxes WITH merging
        boxes_after, mask_after = extract_bboxes_from_mask(
            label_np,
            min_size=min_size,
            spacing=spacing,
            merge_distance_mm=merge_distance_mm,
            return_mask=True
        )
        
        # Create box masks
        box_mask_before = create_box_mask(boxes_before, mask_before, label_np.shape)
        box_mask_after = create_box_mask(boxes_after, mask_after, label_np.shape)
        
        # Create sample directory
        sample_dir = os.path.join(samples_dir, f"sample_{idx+1:02d}_{sample_name}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save files
        if image_np is not None:
            img_nii = nib.Nifti1Image(image_np.astype(np.float32), affine)
            nib.save(img_nii, os.path.join(sample_dir, "image.nii.gz"))
        
        # Original mask
        mask_nii = nib.Nifti1Image(label_np.astype(np.uint8), affine)
        nib.save(mask_nii, os.path.join(sample_dir, "mask_original.nii.gz"))
        
        # Box masks (before/after merging)
        before_nii = nib.Nifti1Image(box_mask_before.astype(np.uint8), affine)
        nib.save(before_nii, os.path.join(sample_dir, "boxes_before_merge.nii.gz"))
        
        after_nii = nib.Nifti1Image(box_mask_after.astype(np.uint8), affine)
        nib.save(after_nii, os.path.join(sample_dir, "boxes_after_merge.nii.gz"))
        
        # Save info
        with open(os.path.join(sample_dir, "info.txt"), 'w') as f:
            f.write(f"Sample: {sample_name}\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Image Information:\n")
            f.write(f"  - Shape: {label_np.shape}\n")
            f.write(f"  - Spacing: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm\n\n")
            
            f.write(f"Processing Parameters:\n")
            f.write(f"  - Min size: {min_size} voxels\n")
            f.write(f"  - Merge distance: {merge_distance_mm} mm\n\n")
            
            f.write(f"Results:\n")
            f.write(f"  - Boxes BEFORE merging: {len(boxes_before)}\n")
            for i, box in enumerate(boxes_before):
                vol_mm3 = box[0] * spacing[0] * box[1] * spacing[1] * box[2] * spacing[2]
                vol_cc = vol_mm3 / 1000.0  # 1 cc = 1000 mm³
                f.write(f"      Box {i+1}: W={box[0]:.0f}, H={box[1]:.0f}, D={box[2]:.0f} voxels")
                f.write(f" ({box[0]*spacing[0]:.1f} × {box[1]*spacing[1]:.1f} × {box[2]*spacing[2]:.1f} mm, {vol_mm3:.0f} mm³ = {vol_cc:.2f} cc)\n")
            
            f.write(f"\n  - Boxes AFTER merging: {len(boxes_after)}\n")
            for i, box in enumerate(boxes_after):
                vol_mm3 = box[0] * spacing[0] * box[1] * spacing[1] * box[2] * spacing[2]
                vol_cc = vol_mm3 / 1000.0  # 1 cc = 1000 mm³
                f.write(f"      Box {i+1}: W={box[0]:.0f}, H={box[1]:.0f}, D={box[2]:.0f} voxels")
                f.write(f" ({box[0]*spacing[0]:.1f} × {box[1]*spacing[1]:.1f} × {box[2]*spacing[2]:.1f} mm, {vol_mm3:.0f} mm³ = {vol_cc:.2f} cc)\n")
            
            f.write(f"\n  - Reduction: {len(boxes_before)} → {len(boxes_after)} ")
            f.write(f"({len(boxes_before) - len(boxes_after)} boxes merged)\n")
        
        print(f"      Boxes: {len(boxes_before)} → {len(boxes_after)} (saved)")
    
    print(f"\n{'='*70}")
    print(f"✅ All samples saved to: {samples_dir}")
    print(f"\nSaved files for each sample:")
    print(f"  - image.nii.gz              : Original image")
    print(f"  - mask_original.nii.gz      : Original segmentation mask")
    print(f"  - boxes_before_merge.nii.gz : Bounding boxes BEFORE merging")
    print(f"  - boxes_after_merge.nii.gz  : Bounding boxes AFTER merging")
    print(f"  - info.txt                  : Detailed information")
    print(f"{'='*70}\n")


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Optimize anchor shapes and recommend patch sizes using K-Means clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 optimize_anchors.py                    # Use default config.yaml
  python3 optimize_anchors.py --config my.yaml   # Use custom config file

All parameters are read from config.yaml. To change settings, edit config.yaml directly.
        """
    )
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    # Load all parameters from config
    print(f"📄 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Extract parameters from config
    image_dir = config['data']['image_dir']
    label_dir = config['data']['label_dir']
    output_dir = os.path.join(config['data']['output_dir'], '..', 'eda')
    min_size = config['anchor']['min_size']
    merge_distance_mm = config['anchor']['merge_distance_mm']
    feature_stride = config['anchor']['feature_stride']
    should_find_optimal_k = config['anchor']['find_optimal_k']
    num_anchors = config['anchor']['num_anchors']
    save_samples = config['anchor']['save_samples']
    num_samples = config['anchor']['num_samples']
    
    print(f"✅ Configuration loaded successfully\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze dataset
    boxes, spacings, image_shapes = analyze_dataset(
        image_dir, 
        label_dir, 
        min_size=min_size,
        merge_distance_mm=merge_distance_mm
    )
    
    # Recommend patch sizes
    train_patch_size, val_patch_size, patch_stats = recommend_patch_size(
        image_shapes, 
        boxes, 
        spacings
    )
    
    # Find optimal K if requested
    if should_find_optimal_k or num_anchors is None:
        optimal_k, k_values, avg_ious = find_optimal_k(boxes, k_range=(3, 10))
        
        # Save K-curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, avg_ious, 'bo-', linewidth=2, markersize=8)
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
        plt.xlabel('Number of Anchors (K)')
        plt.ylabel('Average IoU')
        plt.title('Anchor Optimization: IoU vs Number of Anchors')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'optimal_k_curve.png'), dpi=150)
        print(f"📊 K-curve saved: {output_dir}/optimal_k_curve.png")
        plt.close()
        
        num_anchors = optimal_k if num_anchors is None else num_anchors
    else:
        num_anchors = num_anchors
    
    # Run K-Means with optimal K
    anchors_voxel, avg_iou = kmeans_anchors(boxes, n_clusters=num_anchors, max_iter=300, verbose=True)
    
    # Convert to feature map scale (divide by stride)
    anchors_feature = anchors_voxel / feature_stride
    anchors_feature = np.round(anchors_feature).astype(int)
    anchors_feature = np.maximum(anchors_feature, 1)  # Ensure at least 1
    
    # Convert to physical size (mm) for reporting
    median_spacing = np.median(spacings, axis=0)
    anchors_mm = anchors_voxel * median_spacing
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Optimization Results")
    print(f"{'='*60}\n")
    
    print(f"Number of anchors: {num_anchors}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Feature stride: {feature_stride}")
    print(f"Median spacing: [{median_spacing[0]:.2f}, {median_spacing[1]:.2f}, {median_spacing[2]:.2f}] mm/voxel\n")
    
    print("Optimized Anchor Shapes:\n")
    print("Voxel Space (for feature map):")
    for i, anchor in enumerate(anchors_feature):
        volume_voxel = anchors_voxel[i].prod()
        print(f"  Anchor {i+1}: [{anchor[0]:3d}, {anchor[1]:3d}, {anchor[2]:3d}]  (volume: {volume_voxel:8.0f} voxels)")
    
    print("\nPhysical Space (mm):")
    for i, anchor in enumerate(anchors_mm):
        volume_mm = anchor.prod()
        volume_cc = volume_mm / 1000.0  # 1 cc = 1000 mm³
        print(f"  Anchor {i+1}: [{anchor[0]:6.1f}, {anchor[1]:6.1f}, {anchor[2]:6.1f}] mm  (volume: {volume_mm:8.0f} mm³ = {volume_cc:6.2f} cc)")
    
    # Calculate coverage statistics
    iou_matrix = iou_width_height_depth(boxes, anchors_voxel)
    best_ious = iou_matrix.max(axis=1)
    
    print(f"\n📈 Coverage Statistics:")
    print(f"   - Average IoU: {best_ious.mean():.4f}")
    print(f"   - Median IoU:  {np.median(best_ious):.4f}")
    print(f"   - Min IoU:     {best_ious.min():.4f}")
    print(f"   - Lesions with IoU > 0.5: {(best_ious > 0.5).sum()} / {len(boxes)} ({100*(best_ious > 0.5).mean():.1f}%)")
    print(f"   - Lesions with IoU > 0.3: {(best_ious > 0.3).sum()} / {len(boxes)} ({100*(best_ious > 0.3).mean():.1f}%)")
    
    # Visualize
    visualize_anchors(boxes, anchors_voxel, spacings, output_dir)
    
    # Save results to JSON
    results = {
        'num_anchors': num_anchors,
        'avg_iou': float(avg_iou),
        'feature_stride': feature_stride,
        'median_spacing': median_spacing.tolist(),
        'total_lesions': len(boxes),
        'total_images': len(image_shapes),
        'anchor_shapes_voxel': anchors_voxel.tolist(),
        'anchor_shapes_feature': anchors_feature.tolist(),
        'anchor_shapes_mm': anchors_mm.tolist(),
        'coverage_stats': {
            'avg_iou': float(best_ious.mean()),
            'median_iou': float(np.median(best_ious)),
            'min_iou': float(best_ious.min()),
            'iou_gt_0.5': float((best_ious > 0.5).mean()),
            'iou_gt_0.3': float((best_ious > 0.3).mean()),
        },
        'recommended_patch_sizes': {
            'train_patch_size': train_patch_size,
            'val_patch_size': val_patch_size,
            'statistics': patch_stats,
        }
    }
    
    output_file = os.path.join(output_dir, 'optimized_anchors.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Save sample cases if requested
    if save_samples:
        save_sample_cases(
            image_dir=image_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            min_size=min_size,
            merge_distance_mm=merge_distance_mm,
            num_samples=num_samples
        )
    
    # Print usage instructions
    print(f"\n{'='*60}")
    print(f"🚀 Usage Instructions")
    print(f"{'='*60}\n")
    print(f"Use these anchor shapes in your model:\n")
    print(f"Python code:")
    print(f"base_anchor_shapes = {anchors_feature.tolist()}\n")
    
    if save_samples:
        print(f"📊 Sample cases saved to: {output_dir}/samples/")
        print(f"   Use these to visualize the effect of box merging\n")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

