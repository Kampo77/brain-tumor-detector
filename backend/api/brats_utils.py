"""BraTS 3D brain tumor detection utilities for Django API."""

import torch
import sys
from pathlib import Path
import tempfile
import zipfile
import shutil as sh
import base64
from io import BytesIO
from typing import Tuple

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brats_pipeline import UNet3D, get_tumor_presence, predict_brats_volume_with_image
import nibabel as nib
import numpy as np


class BraTSModelManager:
    """Singleton manager for BraTS 3D U-Net model."""
    
    _instance = None
    _model = None
    _device = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BraTSModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_model(cls, model_path: str):
        """Load BraTS 3D U-Net model."""
        manager = cls()
        
        if manager._model is not None and manager._model_path == model_path:
            return manager._model, manager._device
        
        # Determine device: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        print(f"Loading BraTS model on device: {device}")
        
        # Load model
        model = UNet3D(in_channels=4, out_channels=1)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)
        
        manager._model = model
        manager._device = device
        manager._model_path = model_path
        
        return model, device
    
    @classmethod
    def predict(cls, subject_dir: str, model_path: str) -> dict:
        """Run inference on BraTS subject."""
        model, device = cls.load_model(model_path)
        
        # Use the brats_pipeline function
        result = get_tumor_presence(
            nifti_folder_path=subject_dir,
            model_weights_path=model_path,
            device=device,
        )
        
        return result
    
    @classmethod
    def predict_with_visualization(cls, subject_dir: str, model_path: str) -> dict:
        """
        Run inference and return result with visualization overlay.
        
        Returns:
            dict with keys:
            - has_tumor: bool
            - tumor_fraction: float
            - overlay_png: base64 encoded PNG image (optional)
        """
        model, device = cls.load_model(model_path)
        
        try:
            print(f"[DEBUG] Starting predict_with_visualization for {subject_dir}")
            
            # Get prediction mask and original volume
            print(f"[DEBUG] Calling predict_brats_volume_with_image...")
            mask_pred, original_volume = predict_brats_volume_with_image(
                nifti_folder_path=subject_dir,
                model_weights_path=model_path,
                device=device,
            )
            print(f"[DEBUG] Got mask shape: {mask_pred.shape}, volume shape: {original_volume.shape}")
            
            # Calculate tumor presence
            tumor_pixels = (mask_pred > 0).sum().item()
            total_pixels = mask_pred.numel()
            tumor_fraction = tumor_pixels / total_pixels if total_pixels > 0 else 0.0
            print(f"[DEBUG] Tumor pixels: {tumor_pixels}/{total_pixels} = {tumor_fraction:.4f}")
            
            # Generate overlay
            print(f"[DEBUG] Generating overlay PNG...")
            overlay_base64 = cls._create_overlay_png(original_volume, mask_pred)
            print(f"[DEBUG] Overlay generated: {'YES' if overlay_base64 else 'NO'}")
            
            # Determine brain region
            brain_region = cls._get_brain_region(mask_pred)
            print(f"[DEBUG] Tumor location: {brain_region}")
            
            return {
                'has_tumor': bool(tumor_pixels > 0),
                'tumor_fraction': tumor_fraction,
                'overlay_png': overlay_base64,
                'brain_region': brain_region
            }
        
        except Exception as e:
            print(f"[ERROR] Error in predict_with_visualization: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to prediction without visualization
            print(f"[DEBUG] Falling back to prediction without visualization")
            result = get_tumor_presence(
                nifti_folder_path=subject_dir,
                model_weights_path=model_path,
                device=device,
            )
            result['overlay_png'] = None
            result['brain_region'] = None
            return result
    
    @staticmethod
    def _get_brain_region(mask: torch.Tensor) -> str:
        """
        Determine approximate brain region where tumor is located.
        
        Args:
            mask: (D, H, W) torch tensor - binary segmentation mask
        
        Returns:
            String describing brain region
        """
        try:
            mask_np = mask.cpu().numpy()
            D, H, W = mask_np.shape
            
            # Find tumor center of mass
            tumor_coords = np.argwhere(mask_np > 0)
            if len(tumor_coords) == 0:
                return "No tumor detected"
            
            center_z, center_y, center_x = tumor_coords.mean(axis=0)
            
            # Determine hemisphere (L/R mirrored in medical imaging)
            hemisphere = "Left" if center_x > W/2 else "Right"
            
            # Determine anterior/posterior (front/back)
            if center_y < H/3:
                anterior_posterior = "Frontal"
            elif center_y < 2*H/3:
                anterior_posterior = "Parietal/Central"
            else:
                anterior_posterior = "Occipital"
            
            # Determine superior/inferior (top/bottom)
            if center_z < D/3:
                superior_inferior = "Superior"
            elif center_z < 2*D/3:
                superior_inferior = "Middle"
            else:
                superior_inferior = "Inferior"
            
            return f"{hemisphere} {anterior_posterior} ({superior_inferior})"
        
        except Exception as e:
            print(f"Error determining brain region: {e}")
            return "Unknown region"
    
    @staticmethod
    def _create_overlay_png(volume: np.ndarray, mask: torch.Tensor) -> str:
        """
        Create PNG overlay of middle axial slice with tumor mask.
        
        Args:
            volume: (4, D, H, W) numpy array - normalized input modalities
            mask: (D, H, W) torch tensor - binary segmentation mask
        
        Returns:
            Base64 encoded PNG string with data URI prefix
        """
        try:
            from PIL import Image
            from scipy import ndimage
            
            # Get slice with maximum tumor area (instead of middle)
            D = volume.shape[1]
            mask_np = mask.cpu().numpy()
            
            # Find slice with most tumor pixels
            tumor_per_slice = [mask_np[i].sum() for i in range(D)]
            best_slice_idx = int(np.argmax(tumor_per_slice))
            
            # Fallback to middle if no tumor found
            if tumor_per_slice[best_slice_idx] == 0:
                best_slice_idx = D // 2
            
            print(f"[DEBUG] Using slice {best_slice_idx}/{D} (max tumor: {tumor_per_slice[best_slice_idx]} pixels)")
            
            # Extract slice from T2 modality (index 3)
            slice_img = volume[3, best_slice_idx, :, :]  # (H, W)
            
            # Upscale image for better visualization (64x64 -> 1024x1024)
            zoom_factor = 1024 / slice_img.shape[0]
            slice_upscaled = ndimage.zoom(slice_img, zoom_factor, order=3)  # bicubic interpolation
            
            # Normalize to 0-255 for visualization with contrast enhancement
            slice_min = np.percentile(slice_upscaled, 2)  # Remove outliers
            slice_max = np.percentile(slice_upscaled, 98)
            if slice_max > slice_min:
                slice_normalized = np.clip((slice_upscaled - slice_min) / (slice_max - slice_min) * 255, 0, 255).astype(np.uint8)
            else:
                slice_normalized = np.zeros_like(slice_upscaled, dtype=np.uint8)
            
            # Extract mask slice (same as image)
            mask_slice = mask_np[best_slice_idx, :, :]  # (H, W)
            mask_upscaled = ndimage.zoom(mask_slice.astype(float), zoom_factor, order=0)  # nearest neighbor for mask
            mask_upscaled = (mask_upscaled > 0.5).astype(np.uint8)  # binarize after upscaling
            
            # Debug: print mask statistics
            tumor_pixels = mask_upscaled.sum()
            total_pixels = mask_upscaled.size
            print(f"[DEBUG] Mask stats: {tumor_pixels}/{total_pixels} pixels ({tumor_pixels/total_pixels*100:.2f}%)")
            
            # Create RGB image from grayscale
            height, width = slice_normalized.shape
            rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_img[:, :, 0] = slice_normalized  # R
            rgb_img[:, :, 1] = slice_normalized  # G
            rgb_img[:, :, 2] = slice_normalized  # B
            
            # Directly overlay red color on tumor regions (no transparency mixing)
            if tumor_pixels > 0:
                # Create bright red overlay (255, 0, 0) on tumor regions
                # Mix: 30% original + 70% red for high visibility
                red_overlay = rgb_img.copy()
                red_overlay[mask_upscaled > 0, 0] = np.clip(rgb_img[mask_upscaled > 0, 0] * 0.3 + 255 * 0.7, 0, 255).astype(np.uint8)  # R channel
                red_overlay[mask_upscaled > 0, 1] = (rgb_img[mask_upscaled > 0, 1] * 0.3).astype(np.uint8)  # G channel dimmed
                red_overlay[mask_upscaled > 0, 2] = (rgb_img[mask_upscaled > 0, 2] * 0.3).astype(np.uint8)  # B channel dimmed
                
                # Add bright yellow/white contour for better visibility
                try:
                    from scipy import ndimage as ndi
                    # Dilate mask to get border
                    dilated = ndi.binary_dilation(mask_upscaled, iterations=2)
                    contour = dilated & ~mask_upscaled.astype(bool)
                    # Make contour bright yellow (255, 255, 0)
                    red_overlay[contour, 0] = 255  # R
                    red_overlay[contour, 1] = 255  # G
                    red_overlay[contour, 2] = 0    # B
                except:
                    pass
                
                rgb_img = red_overlay
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_img, mode='RGB')
            
            # Save to bytes buffer
            buf = BytesIO()
            pil_img.save(buf, format='PNG', optimize=False)
            
            # Encode to base64
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            print(f"[DEBUG] Generated overlay PNG: {len(img_base64)} bytes")
            
            return f"data:image/png;base64,{img_base64}"
        
        except Exception as e:
            print(f"Error creating overlay PNG: {e}")
            import traceback
            traceback.print_exc()
            return None


def extract_brats_zip(zip_file_path: str, extract_dir: str) -> str:
    """
    Extract BraTS zip file to a directory.
    
    Expected zip structure:
    - BraTS20_Training_001/
      - BraTS20_Training_001_flair.nii.gz
      - BraTS20_Training_001_t1.nii.gz
      - BraTS20_Training_001_t1ce.nii.gz
      - BraTS20_Training_001_t2.nii.gz
      - BraTS20_Training_001_seg.nii.gz (optional)
    
    OR files directly in ZIP without folder
    
    Returns:
        Path to extracted subject directory
    """
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Find subject directory - try multiple patterns
    dirs = (
        list(extract_path.glob('BraTS*')) +  # BraTS20_, BraTS28_, etc.
        list(extract_path.glob('*Training*')) +
        list(extract_path.glob('*Validation*'))
    )
    
    # Filter to get only directories
    dirs = [d for d in dirs if d.is_dir()]
    
    if dirs:
        return str(dirs[0])
    
    # If no subdirectory found, check if files are directly in extract_path
    nii_files = list(extract_path.glob('*.nii*'))
    if nii_files:
        # Files extracted directly, return extract_path
        return str(extract_path)
    
    raise ValueError(f"No BraTS subject directory or NIfTI files found in {extract_path}")


def process_nifti_file(nifti_path: str) -> dict:
    """
    Process single NIfTI file for quick analysis.
    
    Returns:
        Dictionary with file stats
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    return {
        'shape': data.shape,
        'dtype': str(data.dtype),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
    }


def validate_brats_subject(subject_dir: str) -> dict:
    """
    Validate that a BraTS subject has required modalities.
    
    Returns:
        Dictionary with validation results
    """
    subject_path = Path(subject_dir)
    required_modalities = ['flair', 't1', 't1ce', 't2']
    optional_modalities = ['seg']
    
    found = {}
    missing = []
    
    # Get all .nii and .nii.gz files
    all_nii_files = list(subject_path.glob('*.nii*'))
    print(f"[DEBUG] Found {len(all_nii_files)} NIfTI files in {subject_path}")
    for f in all_nii_files:
        print(f"  - {f.name}")
    
    for modality in required_modalities:
        # Try multiple patterns for flexibility
        patterns = [
            f"*_{modality}.nii.gz",
            f"*_{modality}.nii",
            f"*{modality}.nii.gz",
            f"*{modality}.nii",
            f"*_{modality.upper()}.nii.gz",
            f"*_{modality.upper()}.nii",
            f"*{modality.capitalize()}.nii.gz",
            f"*{modality.capitalize()}.nii",
        ]
        
        found_file = None
        for pattern in patterns:
            matches = list(subject_path.glob(pattern))
            if matches:
                found_file = str(matches[0])
                break
        
        if found_file:
            found[modality] = found_file
            print(f"[DEBUG] Found {modality}: {found_file}")
        else:
            missing.append(modality)
            print(f"[DEBUG] Missing {modality}")
    
    # Check optional segmentation
    seg_found = False
    seg_patterns = ["*_seg.nii.gz", "*_seg.nii", "*seg.nii.gz", "*seg.nii"]
    for pattern in seg_patterns:
        if list(subject_path.glob(pattern)):
            seg_found = True
            break
    
    return {
        'is_valid': len(missing) == 0,
        'found_modalities': found,
        'missing_modalities': missing,
        'has_segmentation': seg_found,
        'num_files': len(all_nii_files),
    }


def create_subject_from_nifti(nifti_file_path: str, temp_dir: str) -> str:
    """
    Create a BraTS subject directory structure from a single NIFTI file.
    
    Args:
        nifti_file_path: Path to NIFTI file (.nii or .nii.gz)
        temp_dir: Temporary directory to create subject structure
    
    Returns:
        Path to created subject directory
    """
    import gzip
    import shutil as sh
    
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    # Extract filename without extension
    nifti_name = Path(nifti_file_path).stem.replace('.nii', '')
    subject_name = nifti_name.rsplit('_', 1)[0] if '_' in nifti_name else nifti_name
    
    # Create subject directory
    subject_dir = temp_path / f"{subject_name}"
    subject_dir.mkdir(exist_ok=True)
    
    # Load NIFTI file
        # Load NIFTI file
    if nifti_file_path.endswith(".nii.gz"):
        # просто копируем исходный gzip как есть
        nifti_path = subject_dir / f"{subject_name}_t2.nii.gz"
        sh.copy(nifti_file_path, nifti_path)
    else:
        # не сжатый .nii копируем как .nii
        nifti_path = subject_dir / f"{subject_name}_t2.nii"
        sh.copy(nifti_file_path, nifti_path)
        


    
    # Load and duplicate the volume to simulate 4 modalities
    img = nib.load(str(nifti_file_path))
    data = img.get_fdata()
    
    # Create synthetic modalities (T1, T1ce, FLAIR) by slightly modifying T2
    for modality in ['t1', 't1ce', 'flair']:
        # Apply slight intensity variations to simulate different modalities
        if modality == 't1':
            mod_data = data * 0.9  # 90% intensity
        elif modality == 't1ce':
            mod_data = data * 1.1  # 110% intensity (contrast enhanced)
        else:  # flair
            mod_data = data * 1.05  # 105% intensity
        
        mod_img = nib.Nifti1Image(mod_data, img.affine, img.header)
        mod_path = subject_dir / f"{subject_name}_{modality}.nii.gz"
        nib.save(mod_img, str(mod_path))
    
    return str(subject_dir)
