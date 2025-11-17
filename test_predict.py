#!/usr/bin/env python3
"""Test API predict endpoint"""

import requests
import sys
from pathlib import Path

# Find a test image
datasets_path = Path("/Users/kampo77/Desktop/rmtv3/datasets")
test_images = list(datasets_path.glob("**/Train/Tumor/*.jpg"))

if not test_images:
    print("❌ No test images found")
    sys.exit(1)

test_image = test_images[0]
print(f"📸 Testing with image: {test_image.name}")

# Upload image
with open(test_image, 'rb') as f:
    files = {'image': f}
    response = requests.post(
        'http://127.0.0.1:8000/api/predict/',
        files=files
    )

print(f"Status: {response.status_code}")
print(f"Response:")
print(response.json())
