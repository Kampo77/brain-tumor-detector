"""
Brain Tumor Detection Model - Inference Module
Handles model loading and prediction for uploaded MRI images.
"""

import os
import ssl
from pathlib import Path
from typing import Tuple, Dict
import io

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context


class BrainTumorDetector:
    """Singleton for brain tumor detection model inference."""

    _instance = None
    _model = None
    _device = None
    _transform = None
    _class_names = ["Normal", "Tumor"]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BrainTumorDetector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model on first use (lazy loading)."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _build_model(self) -> nn.Module:
        """Build ResNet18 model architecture."""
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        return model

    def load_model(self, model_path: str = None) -> bool:
        """
        Load trained model weights.

        Args:
            model_path: Path to model weights file. Defaults to backend/model/model_weights.pth

        Returns:
            True if model loaded successfully, False otherwise
        """
        if model_path is None:
            # Relative to project root
            model_path = Path(__file__).parent / "model_weights.pth"

        model_path = Path(model_path)

        if not model_path.exists():
            print(f"⚠️  Model not found at {model_path}")
            return False

        try:
            self._model = self._build_model().to(self._device)
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))
            self._model.eval()
            print(f"✓ Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def predict(self, image_file) -> Dict[str, any]:
        """
        Run inference on uploaded image.

        Args:
            image_file: Django UploadedFile or file-like object

        Returns:
            Dict with prediction, confidence, and class
        """
        if not self.is_model_loaded():
            return {
                "error": "Model not loaded",
                "prediction": None,
                "confidence": 0.0
            }

        try:
            # Load and prepare image
            image = Image.open(image_file).convert("RGB")
            image_tensor = self._transform(image).unsqueeze(0).to(self._device)

            # Run inference
            with torch.no_grad():
                outputs = self._model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, class_idx = torch.max(probabilities, dim=1)

            class_name = self._class_names[class_idx.item()]
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


# Global instance for use in views
detector = BrainTumorDetector()
