#!/usr/bin/env python3
"""
Simple inference example for DetSeg3D
단일 이미지에 대한 추론 예제
"""

import torch
import numpy as np
from main import DetSegModel


def inference_single_image(model, image_path, device='cuda'):
    """
    단일 이미지에 대한 추론
    
    Args:
        model: DetSegModel
        image_path: 이미지 파일 경로 (.npy or .nii)
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: {
            'segmentation': numpy array (D, H, W),
            'roi_info': list of dict with RoI information
        }
    """
    model.eval()
    
    # Load image
    if image_path.endswith('.npy'):
        image = np.load(image_path)
    elif image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
        import nibabel as nib
        image = nib.load(image_path).get_fdata()
    else:
        raise ValueError("Unsupported file format")
    
    # Preprocess
    # Add batch and channel dimensions
    image_tensor = torch.from_numpy(image).float()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    # Normalize
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)
    image_tensor = image_tensor.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Extract results
    full_seg = outputs['full_segmentation'][0, 0].cpu().numpy()  # (D, H, W)
    full_seg_binary = (full_seg > 0.5).astype(np.uint8)
    roi_info = outputs['roi_info']
    
    # Print RoI information
    print(f"\n{'='*50}")
    print(f"Detected {len(roi_info)} RoIs:")
    print(f"{'='*50}")
    for i, info in enumerate(roi_info):
        print(f"RoI {i+1}:")
        print(f"  - Center: {info['center']}")
        print(f"  - Size: {info['size']}")
        print(f"  - Confidence: {info['confidence']:.3f}")
        print(f"  - BBox: {info['bbox']}")
    
    return {
        'segmentation': full_seg_binary,
        'segmentation_prob': full_seg,
        'roi_info': roi_info
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DetSeg3D Inference Example')
    parser.add_argument('--model', type=str, default='./outputs/best_model.pth', help='모델 파일')
    parser.add_argument('--image', type=str, required=True, help='입력 이미지')
    parser.add_argument('--output', type=str, default='prediction.npy', help='출력 파일')
    parser.add_argument('--roi_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = DetSegModel(roi_size=args.roi_size).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    
    # Inference
    print(f"Running inference on {args.image}...")
    results = inference_single_image(model, args.image, args.device)
    
    # Save
    np.save(args.output, results['segmentation'])
    print(f"\nPrediction saved to {args.output}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Inference Summary:")
    print(f"{'='*50}")
    print(f"Input shape: {results['segmentation'].shape}")
    print(f"Number of RoIs detected: {len(results['roi_info'])}")
    print(f"Foreground voxels: {results['segmentation'].sum()}")
    print(f"Output saved: {args.output}")


if __name__ == '__main__':
    main()

