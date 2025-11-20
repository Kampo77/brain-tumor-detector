"""
Evaluate BraTS model on validation dataset
Calculate Dice Score, IoU, Precision, Recall
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train_brats_3d import BraTSDataset, UNet3D


def calculate_metrics(pred, target):
    """Calculate segmentation metrics"""
    pred = pred > 0.5
    target = target > 0.5
    
    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # True positives, false positives, false negatives
    tp = (pred_flat & target_flat).sum()
    fp = (pred_flat & ~target_flat).sum()
    fn = (~pred_flat & target_flat).sum()
    tn = (~pred_flat & ~target_flat).sum()
    
    # Dice Score
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # IoU (Jaccard)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Precision & Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item()
    }


def evaluate_model(
    model_path='backend/model/brats2020_unet3d.pth',
    data_root='datasets/MICCAI_BraTS_2018_Data_Training',
    num_samples=50
):
    """Evaluate model on validation set"""
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"🔍 Evaluating model on {device}")
    
    # Load model
    model = UNet3D(in_channels=4, out_channels=1, features=32)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load validation dataset
    val_dataset = BraTSDataset(data_root, split='val')
    
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Evaluate
    all_metrics = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(val_dataset))), desc='Evaluating'):
            image, mask = val_dataset[i]
            image = image.unsqueeze(0).to(device)
            mask = mask.to(device)
            
            # Predict
            output = model(image)
            pred = torch.sigmoid(output).squeeze(0)
            
            # Calculate metrics
            metrics = calculate_metrics(pred, mask)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\n" + "="*60)
    print("📊 Model Evaluation Results")
    print("="*60)
    print(f"Dice Score:  {avg_metrics['dice']:.4f}")
    print(f"IoU:         {avg_metrics['iou']:.4f}")
    print(f"Precision:   {avg_metrics['precision']:.4f}")
    print(f"Recall:      {avg_metrics['recall']:.4f}")
    print(f"Accuracy:    {avg_metrics['accuracy']:.4f}")
    print("="*60)
    
    return avg_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='backend/model/brats2020_unet3d.pth')
    parser.add_argument('--data', default='datasets/MICCAI_BraTS_2018_Data_Training')
    parser.add_argument('--samples', type=int, default=50, help='Number of validation samples')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data, args.samples)
