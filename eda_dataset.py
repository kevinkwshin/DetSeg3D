#!/usr/bin/env python3
"""
EDA for 3D Detection Dataset
Analyzes mask files to extract bounding box statistics and recommend anchor shapes
"""

import os
import json
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import scipy.ndimage as ndimage
import nibabel as nib


def extract_bboxes_from_mask(mask, spacing=None, min_size=10, merge_distance_mm=20.0):
    """
    Extract bounding boxes from binary segmentation mask
    
    Args:
        mask: 3D binary mask
        spacing: (3,) array of voxel spacing in mm [sx, sy, sz]
        min_size: minimum voxel count
        merge_distance_mm: merge lesions within this distance (mm)
    
    Returns:
        boxes: List of (width, height, depth) tuples
    """
    # Apply morphological closing to merge nearby lesions
    if merge_distance_mm > 0 and spacing is not None:
        from scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
        
        spacing_array = np.array(spacing)
        kernel_voxels = np.ceil(merge_distance_mm / spacing_array).astype(int)
        kernel_voxels = kernel_voxels + (1 - kernel_voxels % 2)  # Make odd
        
        struct_elem = generate_binary_structure(3, 1)
        max_iterations = max(1, max(kernel_voxels) // 2)
        
        mask_dilated = binary_dilation(mask > 0, structure=struct_elem, iterations=max_iterations)
        mask = binary_erosion(mask_dilated, structure=struct_elem, iterations=max_iterations)
    
    # Find connected components
    labeled_mask, num_components = ndimage.label(mask > 0)
    
    boxes = []
    for component_id in range(1, num_components + 1):
        component_mask = (labeled_mask == component_id)
        voxel_count = component_mask.sum()
        
        if voxel_count < min_size:
            continue
        
        # Get bounding box coordinates
        coords = np.argwhere(component_mask)
        if len(coords) == 0:
            continue
        
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        
        # Calculate box size (width, height, depth)
        width = maxs[0] - mins[0] + 1
        height = maxs[1] - mins[1] + 1
        depth = maxs[2] - mins[2] + 1
        
        boxes.append((int(width), int(height), int(depth)))
    
    return boxes


def analyze_dataset(image_dir, label_dir, output_dir='./eda', min_size=10, merge_distance_mm=20.0):
    """
    Analyze all masks in dataset and generate statistics
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing masks
        output_dir: Output directory for EDA results
        min_size: Minimum voxel count for valid lesion
        merge_distance_mm: Merge lesions within this physical distance (mm)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all label files
    label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
    
    if len(label_files) == 0:
        print(f"❌ No .nii.gz files found in {label_dir}")
        return
    
    print(f"📊 Analyzing {len(label_files)} mask files...")
    print(f"   Label directory: {label_dir}")
    print(f"   Minimum lesion size: {min_size} voxels")
    print(f"   Merge distance: {merge_distance_mm} mm")
    
    # Collect all box sizes
    all_widths = []
    all_heights = []
    all_depths = []
    all_volumes = []
    all_aspect_ratios = []
    
    file_stats = []
    total_lesions = 0
    
    for label_file in tqdm(label_files, desc="Processing masks"):
        try:
            # Load mask
            label_nii = nib.load(label_file)
            label_data = label_nii.get_fdata()
            
            # Get voxel spacing from NIfTI header
            spacing = label_nii.header.get_zooms()[:3]  # [sx, sy, sz]
            
            # Extract boxes
            boxes = extract_bboxes_from_mask(
                label_data, 
                spacing=spacing,
                min_size=min_size,
                merge_distance_mm=merge_distance_mm
            )
            
            if len(boxes) > 0:
                for w, h, d in boxes:
                    all_widths.append(w)
                    all_heights.append(h)
                    all_depths.append(d)
                    all_volumes.append(w * h * d)
                    
                    # Aspect ratios
                    aspect_wh = w / h if h > 0 else 1.0
                    aspect_wd = w / d if d > 0 else 1.0
                    aspect_hd = h / d if d > 0 else 1.0
                    all_aspect_ratios.append({
                        'w/h': aspect_wh,
                        'w/d': aspect_wd,
                        'h/d': aspect_hd
                    })
                
                file_stats.append({
                    'file': os.path.basename(label_file),
                    'num_lesions': len(boxes),
                    'boxes': [(int(w), int(h), int(d)) for w, h, d in boxes]
                })
                total_lesions += len(boxes)
        
        except Exception as e:
            print(f"⚠️  Error processing {label_file}: {e}")
            continue
    
    if total_lesions == 0:
        print("❌ No lesions found in dataset!")
        return
    
    # Convert to numpy arrays
    widths = np.array(all_widths)
    heights = np.array(all_heights)
    depths = np.array(all_depths)
    volumes = np.array(all_volumes)
    
    # Calculate statistics
    percentiles = [10, 25, 50, 75, 90, 95]
    
    stats = {
        'dataset_info': {
            'num_files': len(label_files),
            'num_files_with_lesions': len(file_stats),
            'total_lesions': total_lesions,
            'min_lesion_size': min_size,
            'merge_distance_mm': merge_distance_mm
        },
        'box_sizes': {
            'width': {
                'min': int(widths.min()),
                'max': int(widths.max()),
                'mean': float(widths.mean()),
                'median': float(np.median(widths)),
                'std': float(widths.std()),
                'percentiles': {f'p{p}': float(np.percentile(widths, p)) for p in percentiles}
            },
            'height': {
                'min': int(heights.min()),
                'max': int(heights.max()),
                'mean': float(heights.mean()),
                'median': float(np.median(heights)),
                'std': float(heights.std()),
                'percentiles': {f'p{p}': float(np.percentile(heights, p)) for p in percentiles}
            },
            'depth': {
                'min': int(depths.min()),
                'max': int(depths.max()),
                'mean': float(depths.mean()),
                'median': float(np.median(depths)),
                'std': float(depths.std()),
                'percentiles': {f'p{p}': float(np.percentile(depths, p)) for p in percentiles}
            },
            'volume': {
                'min': int(volumes.min()),
                'max': int(volumes.max()),
                'mean': float(volumes.mean()),
                'median': float(np.median(volumes)),
                'std': float(volumes.std()),
                'percentiles': {f'p{p}': float(np.percentile(volumes, p)) for p in percentiles}
            }
        },
        'aspect_ratios': {
            'w_h_mean': float(np.mean([ar['w/h'] for ar in all_aspect_ratios])),
            'w_d_mean': float(np.mean([ar['w/d'] for ar in all_aspect_ratios])),
            'h_d_mean': float(np.mean([ar['h/d'] for ar in all_aspect_ratios]))
        }
    }
    
    # Generate recommended anchor shapes
    # Use percentiles to cover small, medium, and large lesions
    anchor_shapes = []
    
    # Small lesions (25th percentile)
    small_w = int(np.percentile(widths, 25))
    small_h = int(np.percentile(heights, 25))
    small_d = int(np.percentile(depths, 25))
    anchor_shapes.append([small_w, small_h, small_d])
    
    # Medium lesions (50th percentile - median)
    med_w = int(np.median(widths))
    med_h = int(np.median(heights))
    med_d = int(np.median(depths))
    anchor_shapes.append([med_w, med_h, med_d])
    
    # Large lesions (75th percentile)
    large_w = int(np.percentile(widths, 75))
    large_h = int(np.percentile(heights, 75))
    large_d = int(np.percentile(depths, 75))
    anchor_shapes.append([large_w, large_h, large_d])
    
    # Very large lesions (90th percentile) - optional
    vlarge_w = int(np.percentile(widths, 90))
    vlarge_h = int(np.percentile(heights, 90))
    vlarge_d = int(np.percentile(depths, 90))
    anchor_shapes.append([vlarge_w, vlarge_h, vlarge_d])
    
    stats['recommended_anchors'] = {
        'anchor_shapes': anchor_shapes,
        'description': [
            'Small (p25)',
            'Medium (p50)',
            'Large (p75)',
            'Very Large (p90)'
        ],
        'usage': 'Use these shapes in AnchorGeneratorWithAnchorShape'
    }
    
    # Add per-file statistics
    stats['per_file_stats'] = file_stats[:20]  # Save first 20 for inspection
    
    # Save to JSON
    output_file = os.path.join(output_dir, 'dataset.json')
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Output saved to: {output_file}")
    print(f"\n📊 Summary:")
    print(f"   Total files: {stats['dataset_info']['num_files']}")
    print(f"   Files with lesions: {stats['dataset_info']['num_files_with_lesions']}")
    print(f"   Total lesions: {total_lesions}")
    print(f"   Merge distance: {merge_distance_mm} mm")
    print(f"\n📏 Box Size Statistics:")
    print(f"   Width:  {stats['box_sizes']['width']['min']}-{stats['box_sizes']['width']['max']} "
          f"(median: {stats['box_sizes']['width']['median']:.1f})")
    print(f"   Height: {stats['box_sizes']['height']['min']}-{stats['box_sizes']['height']['max']} "
          f"(median: {stats['box_sizes']['height']['median']:.1f})")
    print(f"   Depth:  {stats['box_sizes']['depth']['min']}-{stats['box_sizes']['depth']['max']} "
          f"(median: {stats['box_sizes']['depth']['median']:.1f})")
    print(f"\n🎯 Recommended Anchor Shapes:")
    for desc, shape in zip(stats['recommended_anchors']['description'], anchor_shapes):
        print(f"   {desc:15s}: {shape}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EDA for 3D Detection Dataset')
    parser.add_argument('--image_dir', type=str, 
                        default='/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/image',
                        help='Image directory')
    parser.add_argument('--label_dir', type=str,
                        default='/mnt/nas206/ANO_DET/GAN_brain/NeuroCAD_preprocessing/1.Asan_data/Asan/AMC_train/hemo/mask',
                        help='Label/mask directory')
    parser.add_argument('--output_dir', type=str, default='./eda',
                        help='Output directory for EDA results')
    parser.add_argument('--min_size', type=int, default=10,
                        help='Minimum voxel count for valid lesion')
    parser.add_argument('--merge_distance_mm', type=float, default=20.0,
                        help='Merge lesions within this physical distance (mm). Set to 0 to disable.')
    
    args = parser.parse_args()
    
    analyze_dataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        min_size=args.min_size,
        merge_distance_mm=args.merge_distance_mm
    )

