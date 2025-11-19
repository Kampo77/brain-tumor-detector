"""
BraTS 3D U-Net Training Script
Trains 3D segmentation model on BraTS dataset with ground truth masks
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import sys

# Import model from brats_pipeline
sys.path.insert(0, str(Path(__file__).parent))
from brats_pipeline import UNet3D


class BraTSDataset(Dataset):
    """Dataset for BraTS 3D volumes with segmentation masks"""
    
    def __init__(self, data_root, split='train', train_ratio=0.8, target_shape=(64, 64, 64)):
        """
        Args:
            data_root: Path to BraTS data (e.g., datasets/BraTS2020_TrainingData)
            split: 'train' or 'val'
            train_ratio: Ratio of training data (0.8 = 80% train, 20% val)
            target_shape: Target volume shape after resizing
        """
        self.data_root = Path(data_root)
        self.target_shape = target_shape
        
        # Find all subject directories with segmentation masks
        all_subjects = []
        
        # Check if data has HGG/LGG subdirectories (BraTS 2018 format)
        hgg_dir = self.data_root / 'HGG'
        lgg_dir = self.data_root / 'LGG'
        
        search_dirs = []
        if hgg_dir.exists() or lgg_dir.exists():
            # BraTS 2018 format with HGG/LGG
            if hgg_dir.exists():
                search_dirs.append(hgg_dir)
            if lgg_dir.exists():
                search_dirs.append(lgg_dir)
        else:
            # Direct format (BraTS 2020+)
            search_dirs.append(self.data_root)
        
        # Search for subjects in all directories
        for search_dir in search_dirs:
            for subject_dir in search_dir.glob('*'):
                if subject_dir.is_dir():
                    seg_files = list(subject_dir.glob('*_seg.nii*'))
                    if seg_files:
                        all_subjects.append(subject_dir)
        
        all_subjects = sorted(all_subjects)
        
        # Split train/val
        split_idx = int(len(all_subjects) * train_ratio)
        if split == 'train':
            self.subjects = all_subjects[:split_idx]
        else:
            self.subjects = all_subjects[split_idx:]
        
        print(f"[{split.upper()}] Found {len(self.subjects)} subjects")
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_dir = self.subjects[idx]
        subject_id = subject_dir.name
        
        # Load 4 modalities
        modalities = ['flair', 't1', 't1ce', 't2']
        volumes = []
        
        for mod in modalities:
            # Find file with flexible naming
            nii_gz = list(subject_dir.glob(f"*_{mod}.nii.gz"))
            nii = list(subject_dir.glob(f"*_{mod}.nii"))
            
            if nii_gz:
                img = nib.load(str(nii_gz[0]))
            elif nii:
                img = nib.load(str(nii[0]))
            else:
                raise FileNotFoundError(f"Missing {mod} for {subject_id}")
            
            volume = img.get_fdata().astype(np.float32)
            volumes.append(volume)
        
        # Load segmentation mask
        seg_gz = list(subject_dir.glob("*_seg.nii.gz"))
        seg_nii = list(subject_dir.glob("*_seg.nii"))
        
        if seg_gz:
            seg_img = nib.load(str(seg_gz[0]))
        elif seg_nii:
            seg_img = nib.load(str(seg_nii[0]))
        else:
            raise FileNotFoundError(f"Missing segmentation for {subject_id}")
        
        mask = seg_img.get_fdata().astype(np.float32)
        
        # Convert multi-class mask to binary (tumor vs non-tumor)
        mask = (mask > 0).astype(np.float32)
        
        # Stack modalities: (4, D, H, W)
        image = np.stack(volumes, axis=0)
        
        # Resize to target shape
        image_resized = self._resize_volume(image, self.target_shape)
        mask_resized = self._resize_mask(mask, self.target_shape)
        
        # Normalize each modality
        for c in range(image_resized.shape[0]):
            channel = image_resized[c]
            mean = channel.mean()
            std = channel.std()
            if std > 0:
                image_resized[c] = (channel - mean) / std
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_resized).float()
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()  # (1, D, H, W)
        
        return image_tensor, mask_tensor
    
    def _resize_volume(self, volume, target_shape):
        """Resize 4D volume (C, D, H, W)"""
        resized_channels = []
        for c in range(volume.shape[0]):
            zoom_factors = [target_shape[i] / volume.shape[i+1] for i in range(3)]
            resized = ndimage.zoom(volume[c], zoom_factors, order=1)
            resized_channels.append(resized)
        return np.stack(resized_channels, axis=0)
    
    def _resize_mask(self, mask, target_shape):
        """Resize 3D mask (D, H, W)"""
        zoom_factors = [target_shape[i] / mask.shape[i] for i in range(3)]
        resized = ndimage.zoom(mask, zoom_factors, order=0)  # nearest neighbor for masks
        return resized


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Dice Loss + Binary Cross Entropy"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate Dice score
            pred = torch.sigmoid(outputs) > 0.5
            intersection = (pred * masks).sum()
            dice = (2. * intersection) / (pred.sum() + masks.sum() + 1e-8)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice


def train_brats_model(
    data_root='datasets/BraTS2020_TrainingData',
    output_path='backend/model/brats2020_unet3d.pth',
    batch_size=2,
    epochs=50,
    learning_rate=1e-4,
    target_shape=(64, 64, 64)
):
    """
    Train BraTS 3D U-Net model
    
    Args:
        data_root: Path to BraTS training data
        output_path: Where to save trained model
        batch_size: Batch size (2-4 for 3D volumes)
        epochs: Number of training epochs
        learning_rate: Learning rate
        target_shape: Target volume shape
    """
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎮 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU (slow!)")
    
    # Create datasets
    print("\n📁 Loading datasets...")
    train_dataset = BraTSDataset(data_root, split='train', target_shape=target_shape)
    val_dataset = BraTSDataset(data_root, split='val', target_shape=target_shape)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("\n🧠 Creating 3D U-Net model...")
    model = UNet3D(in_channels=4, out_channels=1, features=32)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\n🚀 Starting training for {epochs} epochs...")
    best_dice = 0.0
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate schedule
        scheduler.step(val_loss)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice:   {val_dice:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  ✅ Saved best model (Dice: {val_dice:.4f})")
    
    print(f"\n{'='*60}")
    print(f"✨ Training complete!")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BraTS 3D U-Net')
    parser.add_argument('--data', type=str, default='datasets/BraTS2020_TrainingData',
                        help='Path to BraTS training data')
    parser.add_argument('--output', type=str, default='backend/model/brats2020_unet3d.pth',
                        help='Output model path')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    train_brats_model(
        data_root=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
