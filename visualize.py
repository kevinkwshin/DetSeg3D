#!/usr/bin/env python3
"""
Visualization tool for DetSeg3D results
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_3d_segmentation(image, label, prediction, slice_idx=None, save_path=None):
    """
    3D 볼륨의 중간 slice를 시각화
    
    Args:
        image: (D, H, W) numpy array
        label: (D, H, W) numpy array - ground truth
        prediction: (D, H, W) numpy array - prediction
        slice_idx: visualization할 slice index (None이면 중간)
        save_path: 저장 경로
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image[slice_idx], cmap='gray')
    axes[1].imshow(label[slice_idx], cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(image[slice_idx], cmap='gray')
    axes[2].imshow(prediction[slice_idx], cmap='Blues', alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(image[slice_idx], cmap='gray')
    axes[3].imshow(label[slice_idx], cmap='Reds', alpha=0.3)
    axes[3].imshow(prediction[slice_idx], cmap='Blues', alpha=0.3)
    axes[3].set_title('Overlay (Red=GT, Blue=Pred)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize DetSeg3D results')
    parser.add_argument('--image', type=str, required=True, help='Image file (.npy)')
    parser.add_argument('--label', type=str, required=True, help='Label file (.npy)')
    parser.add_argument('--pred', type=str, required=True, help='Prediction file (.npy)')
    parser.add_argument('--slice', type=int, default=None, help='Slice index (default: middle)')
    parser.add_argument('--output', type=str, default='visualization.png', help='Output path')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    image = np.load(args.image)
    label = np.load(args.label)
    pred = np.load(args.pred)
    
    # Normalize image
    if image.max() > 1:
        image = (image - image.min()) / (image.max() - image.min())
    
    # Visualize
    print("Generating visualization...")
    visualize_3d_segmentation(image, label, pred, args.slice, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()

