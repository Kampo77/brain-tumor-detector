#!/bin/bash
# Example curl commands for testing the Tumor Detector API
# 
# Before running these, make sure:
# 1. Virtual environment is activated: source venv/bin/activate
# 2. Server is running: python manage.py runserver
# 3. Server URL: http://127.0.0.1:8000

set -e  # Exit on error

BASE_URL="http://127.0.0.1:8000/api"

echo "======================================================================"
echo "🧬 Tumor Detector API - Testing Examples"
echo "======================================================================"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================================================
# TEST 1: Health Check (Ping)
# ========================================================================
echo ""
echo -e "${BLUE}TEST 1: GET /api/ping/ (Health Check)${NC}"
echo "======================================================================"
echo "Command:"
echo "  curl -X GET $BASE_URL/ping/"
echo ""
echo "Response:"
curl -s -X GET "$BASE_URL/ping/" | python3 -m json.tool
echo ""

# ========================================================================
# TEST 2: Analyze without file (Error handling)
# ========================================================================
echo ""
echo -e "${BLUE}TEST 2: POST /api/analyze/ without file (Error Handling)${NC}"
echo "======================================================================"
echo "Command:"
echo "  curl -X POST $BASE_URL/analyze/"
echo ""
echo "Response (should show error):"
curl -s -X POST "$BASE_URL/analyze/" | python3 -m json.tool
echo ""

# ========================================================================
# TEST 3: Analyze with a real image file
# ========================================================================
echo ""
echo -e "${BLUE}TEST 3: POST /api/analyze/ with image file${NC}"
echo "======================================================================"

# Create a simple test image if it doesn't exist
TEST_IMAGE="/tmp/test_medical_image.jpg"

if [ ! -f "$TEST_IMAGE" ]; then
    echo "Creating a test JPEG image..."
    # Create a minimal JPEG using Python
    python3 << 'EOF'
from PIL import Image
import os

# Create a simple image (requires PIL)
img = Image.new('RGB', (100, 100), color='red')
img.save('/tmp/test_medical_image.jpg')
print("✓ Test image created at /tmp/test_medical_image.jpg")
EOF
else
    echo "✓ Test image already exists at $TEST_IMAGE"
fi

echo ""
echo "Command:"
echo "  curl -X POST -F \"file=@$TEST_IMAGE\" $BASE_URL/analyze/"
echo ""
echo "Response:"
curl -s -X POST -F "file=@$TEST_IMAGE" "$BASE_URL/analyze/" | python3 -m json.tool
echo ""

# ========================================================================
# TEST 4: Verbose output with headers
# ========================================================================
echo ""
echo -e "${BLUE}TEST 4: GET /api/ping/ with HTTP Headers (verbose)${NC}"
echo "======================================================================"
echo "Command:"
echo "  curl -v -X GET $BASE_URL/ping/"
echo ""
echo "Response (with headers):"
curl -v -X GET "$BASE_URL/ping/" 2>&1 | grep -E "< HTTP|< Content|{.*}"
echo ""

# ========================================================================
# Alternative: Using httpie (more readable)
# ========================================================================
echo ""
echo -e "${BLUE}BONUS: Alternative using httpie (if installed)${NC}"
echo "======================================================================"
echo "Install httpie: brew install httpie"
echo ""
echo "Commands:"
echo "  http GET $BASE_URL/ping/"
echo "  http POST $BASE_URL/analyze/"
echo "  http -f POST $BASE_URL/analyze/ file@$TEST_IMAGE"
echo ""

# Try httpie if available
if command -v http &> /dev/null; then
    echo "httpie is installed! Running example:"
    echo ""
    http GET "$BASE_URL/ping/"
else
    echo "(httpie not installed yet - install with: brew install httpie)"
fi

echo ""
echo "======================================================================"
echo -e "${GREEN}✅ All example tests completed!${NC}"
echo "======================================================================"
