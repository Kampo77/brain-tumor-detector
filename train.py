"""
Brain Tumor Detection - PyTorch Training Script
Supports both dataset structures:
1. Brain MRI images/ with Train/Validation split
2. Brain_tumor_images/ without split (80/20 auto-split)
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import ssl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
import shutil
import tempfile

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context


class BrainTumorModelTrainer:
    def __init__(self, dataset_root: str, model_save_path: str = "backend/model/model_weights.pth",
                 batch_size: int = 32, epochs: int = 5, learning_rate: float = 0.001):
        """
        Initialize trainer for brain tumor detection.

        Args:
            dataset_root: Path to datasets folder containing Brain MRI images/ or Brain_tumor_images/
            model_save_path: Path to save trained model weights
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        self.dataset_root = Path(dataset_root)
        self.model_save_path = Path(model_save_path)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model save directory if it doesn't exist
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"🧠 Brain Tumor Detection Training")
        print(f"Device: {self.device}")
        print(f"Dataset root: {self.dataset_root}")

    def _detect_dataset_structure(self) -> Tuple[Path, Path]:
        """
        Detect dataset structure and return train/val paths.
        Returns (train_path, val_path)
        """
        # Check for structured dataset (Brain MRI images with Train/Validation)
        # First check direct path
        structured_path = self.dataset_root / "Brain MRI images"
        if not structured_path.exists():
            # Look for variations (spaces, case differences)
            for item in self.dataset_root.iterdir():
                if item.is_dir() and "brain" in item.name.lower() and "mri" in item.name.lower():
                    structured_path = item
                    break
        
        if structured_path.exists():
            # Check if it has Train/Validation directly
            train_path = structured_path / "Train"
            val_path = structured_path / "Validation"
            
            if not train_path.exists() or not val_path.exists():
                # Look for Train/Validation in subdirectories
                for subdir in structured_path.iterdir():
                    if subdir.is_dir():
                        sub_train = subdir / "Train"
                        sub_val = subdir / "Validation"
                        if sub_train.exists() and sub_val.exists():
                            train_path = sub_train
                            val_path = sub_val
                            break
            
            if train_path.exists() and val_path.exists():
                print(f"✓ Found structured dataset: {train_path.parent}")
                return train_path, val_path

        # Check for single folder dataset (Brain_tumor_images)
        single_path = self.dataset_root / "Brain_tumor_images"
        if not single_path.exists():
            # Look for variations
            for item in self.dataset_root.iterdir():
                if item.is_dir() and "brain" in item.name.lower() and "tumor" in item.name.lower():
                    single_path = item
                    break
        
        if single_path.exists():
            print(f"✓ Found single-folder dataset: {single_path}")
            print(f"  Creating 80/20 train/validation split...")
            return self._create_split_from_folder(single_path)

        raise FileNotFoundError(
            f"No recognized dataset structure found in {self.dataset_root}. "
            "Expected 'Brain MRI images/' or 'Brain_tumor_images/'"
        )

    def _create_split_from_folder(self, source_folder: Path) -> Tuple[Path, Path]:
        """
        Create train/validation split from single folder.
        Uses temporary directories for compatibility.
        Returns (train_path, val_path)
        """
        classes = ["Normal", "Tumor"]
        temp_dir = Path(tempfile.mkdtemp(prefix="brain_tumor_split_"))

        train_dir = temp_dir / "Train"
        val_dir = temp_dir / "Validation"

        for split_dir in [train_dir, val_dir]:
            for cls in classes:
                (split_dir / cls).mkdir(parents=True, exist_ok=True)

        # Split images for each class
        for cls in classes:
            class_dir = source_folder / cls
            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            images = list(class_dir.glob("*.*"))
            if not images:
                raise ValueError(f"No images found in {class_dir}")

            # 80/20 split
            train_imgs, val_imgs = train_test_split(
                images, test_size=0.2, random_state=42
            )

            # Copy files
            for img in train_imgs:
                shutil.copy2(img, train_dir / cls / img.name)
            for img in val_imgs:
                shutil.copy2(img, val_dir / cls / img.name)

            print(f"  {cls}: {len(train_imgs)} train, {len(val_imgs)} val")

        return train_dir, val_dir

    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get data transforms for training and validation."""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return train_transform, val_transform

    def _build_model(self) -> nn.Module:
        """Build ResNet18 model with binary classification head."""
        try:
            # Try loading with ImageNet weights
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("✓ Loaded ResNet18 with ImageNet pre-trained weights")
        except Exception as e:
            # Fallback: load without weights (will train from scratch)
            print(f"⚠️  Could not load pre-trained weights: {e}")
            print("   Training from scratch instead...")
            model = models.resnet18(weights=None)

        # Freeze initial layers
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False

        # Replace final layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2 classes: Normal, Tumor
        )

        return model.to(self.device)

    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Train for one epoch. Returns average loss."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(train_loader), 100 * correct / total

    def _validate(self, model: nn.Module, val_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float]:
        """Validate model. Returns (loss, accuracy)."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(val_loader), 100 * correct / total

    def train(self):
        """Main training pipeline."""
        # Detect dataset structure
        train_path, val_path = self._detect_dataset_structure()

        # Load datasets
        train_transform, val_transform = self._get_transforms()
        train_dataset = datasets.ImageFolder(str(train_path), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(val_path), transform=val_transform)

        print(f"\n📊 Dataset loaded:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Classes: {train_dataset.classes}")

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Build model
        model = self._build_model()
        print(f"\n🏗️  Model: ResNet18 (transfer learning)")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_val_acc = 0
        print(f"\n🚀 Training for {self.epochs} epochs...")
        print("-" * 70)

        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            scheduler.step()

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.model_save_path)
                print(f"  ✓ Best model saved (Acc: {val_acc:.2f}%)")

        print("-" * 70)
        print(f"\n✅ Training complete!")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
        print(f"  Model saved to: {self.model_save_path}")

        return model


def main():
    """Main entry point."""
    dataset_root = "datasets"
    model_save_path = "backend/model/model_weights.pth"

    # Verify dataset directory exists
    if not Path(dataset_root).exists():
        print(f"❌ Dataset directory not found: {dataset_root}")
        sys.exit(1)

    trainer = BrainTumorModelTrainer(
        dataset_root=dataset_root,
        model_save_path=model_save_path,
        batch_size=32,
        epochs=5,
        learning_rate=0.001
    )

    trainer.train()


if __name__ == "__main__":
    main()
