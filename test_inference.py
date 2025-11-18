#!/usr/bin/env python3
"""Test inference on a trained BraTS model."""

import sys
sys.path.insert(0, '/Users/kampo77/Desktop/rmtv3')

from brats_pipeline import predict_brats_volume, get_tumor_presence
import torch

def test_inference():
    """Test inference functions on a validation subject."""
    
    print("=== Testing BraTS Inference Functions ===\n")
    
    # Test 1: Full volume segmentation
    print("1️⃣ Testing predict_brats_volume()...")
    try:
        model_path = "/Users/kampo77/Desktop/rmtv3/backend/model/brats2020_unet3d_pilot.pth"
        subject_dir = "/Users/kampo77/Desktop/d/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_001"
        
        prediction = predict_brats_volume(
            subject_dir=subject_dir,
            model_path=model_path,
            device=None,  # Auto-detect MPS
        )
        
        print(f"   ✅ Segmentation mask shape: {prediction.shape}")
        print(f"   ✅ Unique values: {torch.unique(prediction).tolist()}")
        print(f"   ✅ Tumor fraction: {(prediction > 0).sum().item() / prediction.numel():.4f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Binary classification
    print("\n2️⃣ Testing get_tumor_presence()...")
    try:
        result = get_tumor_presence(
            subject_dir=subject_dir,
            model_path=model_path,
            device=None,
        )
        
        print(f"   ✅ Result: {result}")
        print(f"   ✅ Has tumor: {result['has_tumor']}")
        print(f"   ✅ Tumor fraction: {result['tumor_fraction']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Inference tests completed!")

if __name__ == "__main__":
    test_inference()
