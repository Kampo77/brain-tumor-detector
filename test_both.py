#!/usr/bin/env python3
"""Test API predict endpoint with both Normal and Tumor images"""

import requests
from pathlib import Path

datasets_path = Path("/Users/kampo77/Desktop/rmtv3/datasets")

# Test with Tumor image
tumor_images = list(datasets_path.glob("**/Train/Tumor/*.jpg"))
if tumor_images:
    test_image = tumor_images[0]
    print(f"\n📸 Testing TUMOR image: {test_image.name}")
    with open(test_image, 'rb') as f:
        response = requests.post(
            'http://127.0.0.1:8000/api/predict/',
            files={'image': f}
        )
    result = response.json()
    print(f"Status: {response.status_code}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")

# Test with Normal image
normal_images = list(datasets_path.glob("**/Train/Normal/*.jpg"))
if normal_images:
    test_image = normal_images[0]
    print(f"\n📸 Testing NORMAL image: {test_image.name}")
    with open(test_image, 'rb') as f:
        response = requests.post(
            'http://127.0.0.1:8000/api/predict/',
            files={'image': f}
        )
    result = response.json()
    print(f"Status: {response.status_code}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
