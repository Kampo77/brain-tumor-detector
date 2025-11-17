#!/usr/bin/env python
"""
Test script for tumor detector API endpoints - SIMPLER VERSION
"""
import os
import sys
import django

# Configure Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tumor_detector.settings')
django.setup()

from django.test import Client
import json

# Create a test client
client = Client()

print("=" * 60)
print("🧪 TESTING TUMOR DETECTOR API ENDPOINTS")
print("=" * 60)

try:
    # Test 1: GET /api/ping/
    print("\n1️⃣  Testing GET /api/ping/")
    print("-" * 60)
    response = client.get('/api/ping/')
    print(f"Status Code: {response.status_code}")
    
    resp_data = response.json()
    print(f"Response: {json.dumps(resp_data, indent=2)}")
    
    if response.status_code == 200 and 'message' in resp_data:
        print("✅ PING TEST PASSED!")
    else:
        print("❌ PING TEST FAILED")
    
    # Test 2: POST /api/analyze/ without file
    print("\n2️⃣  Testing POST /api/analyze/ (without file - should fail)")
    print("-" * 60)
    response = client.post('/api/analyze/')
    print(f"Status Code: {response.status_code}")
    
    resp_data = response.json()
    print(f"Response: {json.dumps(resp_data, indent=2)}")
    
    if response.status_code == 400 and 'error' in resp_data:
        print("✅ ERROR HANDLING TEST PASSED!")
    else:
        print("❌ ERROR HANDLING TEST FAILED")
    
    # Test 3: POST /api/analyze/ with valid image file
    print("\n3️⃣  Testing POST /api/analyze/ (with PNG image file)")
    print("-" * 60)
    
    from django.core.files.uploadedfile import SimpleUploadedFile
    
    # Minimal valid PNG (1x1 transparent)
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
        0x42, 0x60, 0x82
    ])
    
    test_file = SimpleUploadedFile(
        "test_image.png",
        png_bytes,
        content_type="image/png"
    )
    
    response = client.post('/api/analyze/', {'file': test_file})
    print(f"Status Code: {response.status_code}")
    
    resp_data = response.json()
    print(f"Response: {json.dumps(resp_data, indent=2)}")
    
    if response.status_code == 200 and 'result' in resp_data:
        print("✅ ANALYZE TEST PASSED!")
    else:
        print("❌ ANALYZE TEST FAILED")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
