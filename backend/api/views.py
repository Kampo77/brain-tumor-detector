from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from pathlib import Path
import sys
import os
import ssl

# Fix SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# Global model instance
_model = None
_device = None
_model_loaded = False


def _build_and_load_model(model_path):
    """Load the trained model"""
    global _model, _device, _model_loaded
    
    if _model_loaded:
        return True
    
    try:
        # Import torch here (lazy loading)
        import torch
        import torch.nn as nn
        from torchvision import models
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {_device}")
        
        # Build model
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # Load weights - always load to cpu first
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        _model = model.to(_device)
        _model_loaded = True
        print(f"✓ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def _predict_image(image_file):
    """Run inference on image"""
    global _model, _device
    
    if not _model_loaded or _model is None:
        return {"error": "Model not loaded", "prediction": None, "confidence": 0.0}
    
    try:
        # Import here (lazy loading)
        import torch
        from torchvision import transforms
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_file).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(_device)
        
        # Inference
        with torch.no_grad():
            outputs = _model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)
        
        class_names = ["Normal", "Tumor"]
        class_name = class_names[class_idx.item()]
        confidence_score = confidence.item()
        
        return {
            "prediction": class_name,
            "confidence": round(confidence_score, 4),
            "class_index": class_idx.item(),
            "error": None
        }
    except Exception as e:
        return {
            "error": str(e),
            "prediction": None,
            "confidence": 0.0
        }


class PingView(APIView):
    """
    Health check endpoint.
    GET /ping/ → returns {"message": "API is working"}
    """
    def get(self, request):
        import os
        import torch
        
        model_path = Path(__file__).parent.parent / "model" / "model_weights.pth"
        brats_path = Path(__file__).parent.parent / "model" / "brats2020_unet3d.pth"
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        return Response(
            {
                "message": "API is working",
                "backend": "http://127.0.0.1:8000",
                "models": {
                    "2d_model": model_path.exists(),
                    "3d_model": brats_path.exists(),
                },
                "device": device
            },
            status=status.HTTP_200_OK
        )


class AnalyzeView(APIView):
    """
    Image analysis endpoint.
    POST /analyze/ accepts an uploaded image file and returns a placeholder prediction.
    
    Request: multipart/form-data with "file" field
    Response: {"result": "clean" | "tumor", "confidence": 0.0-1.0}
    """
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        # Check if file was uploaded
        if 'file' not in request.FILES:
            return Response(
                {"error": "No file provided. Please upload an image with field name 'file'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['file']
        
        # Validate file is an image (basic check)
        if not uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.dcm')):
            return Response(
                {"error": "Invalid file type. Please upload an image (jpg, png, gif, bmp) or DICOM file (dcm)."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # TODO: Here we will later call the real ML model from model/ folder
        # For now, return a placeholder prediction
        placeholder_result = {
            "result": "clean",
            "confidence": 0.99,
            "message": "Placeholder prediction. Real ML model will be integrated later."
        }
        
        return Response(placeholder_result, status=status.HTTP_200_OK)


class PredictView(APIView):
    """
    Brain tumor detection prediction endpoint.
    POST /predict/ accepts an uploaded MRI image and returns tumor detection result.
    
    Request: multipart/form-data with "image" field
    Response: {"prediction": "Normal" | "Tumor", "confidence": 0.0-1.0}
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        # Check if file was uploaded
        if 'image' not in request.FILES:
            return Response(
                {
                    "error": "No file provided. Please upload an image with field name 'image'.",
                    "prediction": None,
                    "confidence": 0.0
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        uploaded_file = request.FILES['image']

        # Validate file is an image
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        if not uploaded_file.name.lower().endswith(valid_extensions):
            return Response(
                {
                    "error": "Invalid file type. Please upload an image (jpg, png, gif, bmp).",
                    "prediction": None,
                    "confidence": 0.0
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Load model on first request
        global _model_loaded
        if not _model_loaded:
            model_path = Path(__file__).parent.parent / "model" / "model_weights.pth"
            
            if not model_path.exists():
                return Response(
                    {
                        "error": "Model not available. Please train the model first using train.py.",
                        "prediction": None,
                        "confidence": 0.0
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            if not _build_and_load_model(str(model_path)):
                return Response(
                    {
                        "error": "Failed to load model.",
                        "prediction": None,
                        "confidence": 0.0
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        # Run inference
        result = _predict_image(uploaded_file)

        if result.get("error"):
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        return Response(result, status=status.HTTP_200_OK)
