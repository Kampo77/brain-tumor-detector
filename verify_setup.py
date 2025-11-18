#!/usr/bin/env python3
"""Quick setup verification for BraTS implementation."""

import sys
from pathlib import Path
import subprocess

def check_file_exists(path: str, name: str) -> bool:
    """Check if file exists and print status."""
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {name}: {path}")
    return exists

def check_module_import(module: str) -> bool:
    """Check if module can be imported."""
    try:
        __import__(module)
        print(f"✅ Module '{module}' available")
        return True
    except ImportError as e:
        print(f"❌ Module '{module}' not available: {e}")
        return False

def main():
    print("=" * 70)
    print("🧠 BraTS 2020 IMPLEMENTATION - SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    all_good = True
    
    # 1. Check Python files
    print("1️⃣ Python Files:")
    files_to_check = [
        ("/Users/kampo77/Desktop/rmtv3/brats_pipeline.py", "BraTS Pipeline"),
        ("/Users/kampo77/Desktop/rmtv3/backend/api/brats_views.py", "BraTS Views"),
        ("/Users/kampo77/Desktop/rmtv3/backend/api/brats_utils.py", "BraTS Utils"),
        ("/Users/kampo77/Desktop/rmtv3/backend/api/urls.py", "API URLs"),
    ]
    for path, name in files_to_check:
        if not check_file_exists(path, name):
            all_good = False
    print()
    
    # 2. Check model file
    print("2️⃣ Trained Model:")
    model_path = "/Users/kampo77/Desktop/rmtv3/backend/model/brats2020_unet3d.pth"
    if Path(model_path).exists():
        size_mb = Path(model_path).stat().st_size / 1024 / 1024
        print(f"✅ Model file: {size_mb:.2f} MB")
    else:
        print(f"❌ Model file not found: {model_path}")
        all_good = False
    print()
    
    # 3. Check dependencies
    print("3️⃣ Python Dependencies:")
    modules = [
        'torch',
        'torchvision',
        'nibabel',
        'scipy',
        'skimage',
        'django',
        'rest_framework',
    ]
    for module in modules:
        if not check_module_import(module):
            all_good = False
    print()
    
    # 4. Check Dataset
    print("4️⃣ Dataset Availability:")
    dataset_paths = [
        ("/Users/kampo77/Desktop/d/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", "Training Data (369 subjects)"),
        ("/Users/kampo77/Desktop/d/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData", "Validation Data (125 subjects)"),
    ]
    for path, name in dataset_paths:
        if check_file_exists(path, name):
            num_subjects = len(list(Path(path).glob("BraTS20_*")))
            print(f"   └─ Found {num_subjects} subjects")
        else:
            all_good = False
    print()
    
    # 5. Summary
    print("=" * 70)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready for use!")
        print()
        print("Next Steps:")
        print("1. Start Django development server: python manage.py runserver")
        print("2. Test health endpoint: curl http://localhost:8000/api/brats/health/")
        print("3. Create test ZIP and test prediction endpoint")
        print("4. Integrate frontend with /api/brats/predict/")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
    print("=" * 70)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
