"""
BraTS 2020 Dataset Pipeline with 3D U-Net Segmentation Model

This module provides:
- Brats2020Dataset: PyTorch Dataset for loading 3D NIfTI volumes
- UNet3D: 3D U-Net model for brain tumor segmentation
- Training routine with Dice loss
- Inference function for predictions
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import nibabel as nib
from scipy import ndimage


class Brats2020Dataset(Dataset):
    """PyTorch Dataset for BraTS 2020 3D MRI volumes."""

    def __init__(
        self,
        root_dir: str,
        subject_ids: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        target_shape: Optional[Tuple[int, int, int]] = (80, 96, 96),
        normalize: bool = True,
    ):
        """
        Initialize BraTS 2020 Dataset.

        Args:
            root_dir: Path to MICCAI_BraTS2020_TrainingData directory
            subject_ids: List of subject folder names (e.g., ['BraTS20_Training_001', ...])
                        If None, auto-discover from root_dir
            modalities: List of modality suffixes (e.g., ['flair', 't1', 't1ce', 't2'])
            target_shape: Target volume shape after resizing (D, H, W)
            normalize: Whether to apply z-score normalization per volume
        """
        self.root_dir = Path(root_dir)
        self.target_shape = target_shape
        self.normalize = normalize
        self.modalities = modalities or ['flair', 't1', 't1ce', 't2']

        if subject_ids is None:
            # Auto-discover subject directories
            subject_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
            self.subject_ids = [d.name for d in subject_dirs]
        else:
            self.subject_ids = subject_ids

        if not self.subject_ids:
            raise ValueError(f"No subject directories found in {root_dir}")

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject_id = self.subject_ids[idx]
        subject_dir = self.root_dir / subject_id

        # Load modalities (supports both .nii and .nii.gz)
        modality_volumes = []
        for mod in self.modalities:
            mod_path_gz = subject_dir / f"{subject_id}_{mod}.nii.gz"
            mod_path = subject_dir / f"{subject_id}_{mod}.nii"
            
            if mod_path_gz.exists():
                mod_path = mod_path_gz
            elif not mod_path.exists():
                raise FileNotFoundError(f"Missing {mod_path} or {mod_path_gz}")
            
            img = nib.load(str(mod_path))
            vol = img.get_fdata().astype(np.float32)
            modality_volumes.append(vol)

        # Stack modalities: (C, D, H, W)
        image = np.stack(modality_volumes, axis=0)

        # Load segmentation mask (supports both .nii and .nii.gz, optional)
        seg_path_gz = subject_dir / f"{subject_id}_seg.nii.gz"
        seg_path = subject_dir / f"{subject_id}_seg.nii"
        
        mask = None
        if seg_path_gz.exists():
            seg_img = nib.load(str(seg_path_gz))
            mask = seg_img.get_fdata().astype(np.float32)
        elif seg_path.exists():
            seg_img = nib.load(str(seg_path))
            mask = seg_img.get_fdata().astype(np.float32)
        else:
            # For validation data without segmentation, create dummy mask
            mask = np.zeros_like(image[0])

        # Resize volumes if target_shape is specified
        if self.target_shape is not None:
            image = self._resize_volume(image, self.target_shape)
            mask = self._resize_volume(mask[np.newaxis, ...], self.target_shape)[0]

        # Normalize each channel
        if self.normalize:
            image = self._normalize(image)

        # Binary segmentation: any non-zero region is tumor
        mask = (mask > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'subject_id': subject_id,
        }

    @staticmethod
    def _resize_volume(
        volume: np.ndarray,
        target_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Resize volume to target shape using interpolation."""
        if volume.ndim == 4:  # (C, D, H, W)
            C = volume.shape[0]
            resized = []
            for c in range(C):
                vol_3d = volume[c]
                resized_3d = ndimage.zoom(
                    vol_3d,
                    zoom=[target_shape[i] / vol_3d.shape[i] for i in range(3)],
                    order=1,
                )
                resized.append(resized_3d)
            return np.stack(resized, axis=0)
        else:  # (D, H, W)
            return ndimage.zoom(
                volume,
                zoom=[target_shape[i] / volume.shape[i] for i in range(3)],
                order=1,
            )

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        """Apply z-score normalization per channel."""
        normalized = np.zeros_like(volume)
        for c in range(volume.shape[0]):
            vol = volume[c]
            mean = vol.mean()
            std = vol.std()
            if std > 0:
                normalized[c] = (vol - mean) / (std + 1e-8)
            else:
                normalized[c] = vol - mean
        return normalized


class UNet3D(nn.Module):
    """3D U-Net for medical image segmentation."""

    def __init__(self, in_channels: int = 4, out_channels: int = 1, features: int = 32):
        """
        Initialize 3D U-Net.

        Args:
            in_channels: Number of input channels (e.g., 4 for BraTS modalities)
            out_channels: Number of output channels (e.g., 1 for binary segmentation)
            features: Base number of features in first conv layer
        """
        super(UNet3D, self).__init__()
        self.features = features

        # Encoder
        self.enc1 = self._conv_block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self._conv_block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self._conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(features * 4, features * 8)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(features * 2, features)

        # Output
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Double convolution block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        return x


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            predictions: Model output (logits or probabilities)
            targets: Ground truth binary masks
        """
        predictions = torch.sigmoid(predictions)
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


def train_brats_model(
    dataset_root: str,
    model_save_path: str,
    subject_ids: Optional[List[str]] = None,
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    val_split: float = 0.1,
) -> None:
    """
    Train 3D U-Net model on BraTS 2020 data.

    Args:
        dataset_root: Path to MICCAI_BraTS2020_TrainingData directory
        model_save_path: Path to save best model weights
        subject_ids: List of subject IDs to use (if None, all subjects)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        num_workers: Number of workers for DataLoader
        device: Torch device (defaults to cuda if available)
        val_split: Fraction of data to use for validation
    """
    if device is None:
        # Prefer MPS (Mac GPU) > CUDA (Nvidia GPU) > CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load dataset
    dataset = Brats2020Dataset(
        root_dir=dataset_root,
        subject_ids=subject_ids,
        target_shape=(80, 96, 96),
        normalize=True,
    )

    # Split into train/val
    num_val = max(1, int(len(dataset) * val_split))
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [num_train, num_val],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Initialize model
    model = UNet3D(in_channels=4, out_channels=1, features=32).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = DiceLoss()

    best_val_dice = 0.0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f"Training BraTS 3D U-Net for {num_epochs} epochs...")
    print(f"Train samples: {num_train}, Val samples: {num_val}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_scores = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                logits = model(images)
                loss = criterion(logits, masks.unsqueeze(1))
                val_loss += loss.item()

                # Compute Dice score
                preds = torch.sigmoid(logits) > 0.5
                dice = _compute_dice(preds.float(), masks.unsqueeze(1))
                val_dice_scores.append(dice.item())

        val_loss /= len(val_loader)
        val_dice = np.mean(val_dice_scores) if val_dice_scores else 0.0

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ Best model saved to {model_save_path}")

    print(f"Training complete. Best Val Dice: {best_val_dice:.4f}")


def _compute_dice(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute Dice coefficient."""
    smooth = 1.0
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )
    return dice


def predict_brats_volume(
    nifti_folder_path: str,
    model_weights_path: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Predict segmentation for a single BraTS subject.

    Args:
        nifti_folder_path: Path to subject folder containing .nii or .nii.gz files
        model_weights_path: Path to trained model weights
        device: Torch device (defaults to cuda if available)

    Returns:
        Predicted segmentation mask as 3D tensor (D, H, W)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load modalities (supports both .nii and .nii.gz)
    modalities = ['flair', 't1', 't1ce', 't2']
    modality_volumes = []

    for mod in modalities:
        mod_path_gz = os.path.join(nifti_folder_path, f"*_{mod}.nii.gz")
        mod_path_nii = os.path.join(nifti_folder_path, f"*_{mod}.nii")
        
        mod_files = glob.glob(mod_path_gz)
        if not mod_files:
            mod_files = glob.glob(mod_path_nii)
        
        if not mod_files:
            raise FileNotFoundError(f"No files matching {mod_path_gz} or {mod_path_nii}")

        img = nib.load(mod_files[0])
        vol = img.get_fdata().astype(np.float32)
        modality_volumes.append(vol)

    # Stack and prepare
    image = np.stack(modality_volumes, axis=0)

    # Resize to model input size
    target_shape = (80, 96, 96)
    image = Brats2020Dataset._resize_volume(image, target_shape)

    # Normalize
    image = Brats2020Dataset._normalize(image)

    # Add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    # Load model
    model = UNet3D(in_channels=4, out_channels=1, features=32).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        predictions = torch.sigmoid(logits) > 0.5

    # Remove batch dimension and convert to numpy
    mask_pred = predictions.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    return torch.from_numpy(mask_pred)


def predict_brats_volume_with_image(
    nifti_folder_path: str,
    model_weights_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Run inference and return both prediction mask and original volume.

    Args:
        nifti_folder_path: Path to subject folder with 4 modalities
        model_weights_path: Path to trained model weights
        device: Torch device

    Returns:
        Tuple of (prediction_mask, original_volume)
        - prediction_mask: (D, H, W) binary tensor
        - original_volume: (4, D, H, W) numpy array of input modalities
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # Load 4 modalities
    subject_dir = Path(nifti_folder_path)
    subject_id = subject_dir.name

    modalities = ['flair', 't1', 't1ce', 't2']
    modality_volumes = []

    for mod in modalities:
        # Try multiple patterns to find the file
        nii_path_gz = subject_dir / f"{subject_id}_{mod}.nii.gz"
        nii_path = subject_dir / f"{subject_id}_{mod}.nii"
        
        found_file = None
        if nii_path_gz.exists():
            found_file = nii_path_gz
        elif nii_path.exists():
            found_file = nii_path
        else:
            # Try glob patterns for flexible naming
            patterns = [
                f"*_{mod}.nii.gz",
                f"*_{mod}.nii",
                f"*{mod}.nii.gz",
                f"*{mod}.nii",
            ]
            for pattern in patterns:
                matches = list(subject_dir.glob(pattern))
                if matches:
                    found_file = matches[0]
                    break
        
        if found_file is None:
            raise FileNotFoundError(f"Missing modality {mod} for {subject_id}")
        
        img = nib.load(str(found_file))
        volume = img.get_fdata().astype(np.float32)
        modality_volumes.append(volume)

    # Stack modalities: (4, D, H, W)
    image = np.stack(modality_volumes, axis=0)
    
    # Resize to (4, 64, 64, 64)
    target_shape = (64, 64, 64)
    resized_channels = []
    for c in range(image.shape[0]):
        resized = ndimage.zoom(
            image[c],
            zoom=[target_shape[i] / image.shape[i+1] for i in range(3)],
            order=1
        )
        resized_channels.append(resized)
    image_resized = np.stack(resized_channels, axis=0)

    # Normalize per channel
    for c in range(image_resized.shape[0]):
        channel = image_resized[c]
        mean = channel.mean()
        std = channel.std()
        if std > 0:
            image_resized[c] = (channel - mean) / std

    # Keep original volume for visualization
    original_volume = image_resized.copy()

    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).unsqueeze(0).to(device)

    # Load model
    model = UNet3D(in_channels=4, out_channels=1, features=32).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        predictions = torch.sigmoid(logits) > 0.5

    # Remove batch dimension: (1, 1, D, H, W) -> (D, H, W)
    mask_pred = predictions.squeeze(0).squeeze(0).cpu()

    return mask_pred, original_volume


def get_tumor_presence(
    nifti_folder_path: str,
    model_weights_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Simple binary classification: is there a tumor or not?

    Args:
        nifti_folder_path: Path to subject folder
        model_weights_path: Path to trained model weights
        device: Torch device

    Returns:
        Dictionary with 'has_tumor' (bool) and 'confidence' (float 0-1)
    """
    mask = predict_brats_volume(nifti_folder_path, model_weights_path, device)
    tumor_pixels = (mask > 0).sum().item()
    total_pixels = mask.numel()
    confidence = tumor_pixels / total_pixels if total_pixels > 0 else 0.0

    return {
        'has_tumor': bool(tumor_pixels > 0),
        'tumor_fraction': confidence,
    }


if __name__ == "__main__":
    # Example usage with both training and validation datasets
    
    # Choose dataset path - supports both .nii and .nii.gz formats
    dataset_root = "/Users/kampo77/Desktop/d/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    # Alternative validation dataset:
    # dataset_root = "/Users/kampo77/Desktop/d/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
    
    model_save_path = "backend/model/brats2020_unet3d.pth"

    # Check if dataset exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset not found at {dataset_root}")
        print("Please check the path or download BraTS 2020 data.")
    else:
        # Train model
        train_brats_model(
            dataset_root=dataset_root,
            model_save_path=model_save_path,
            num_epochs=10,
            batch_size=2,
            learning_rate=1e-4,
            val_split=0.1,
        )

        print(f"\nModel saved to {model_save_path}")
