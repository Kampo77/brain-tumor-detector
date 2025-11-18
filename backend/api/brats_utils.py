"""BraTS 3D brain tumor detection utilities for Django API."""

import torch
import sys
from pathlib import Path
import tempfile
import zipfile

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brats_pipeline import UNet3D, get_tumor_presence
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
    
    Returns:
        Path to extracted subject directory
    """
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Find subject directory
    dirs = list(extract_path.glob('BraTS20_*'))
    if not dirs:
        raise ValueError(f"No BraTS subject directory found in {extract_path}")
    
    return str(dirs[0])


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
    
    for modality in required_modalities:
        nii_gz = subject_path / f"*_{modality}.nii.gz"
        nii = subject_path / f"*_{modality}.nii"
        
        nii_gz_files = list(subject_path.glob(f"*_{modality}.nii.gz"))
        nii_files = list(subject_path.glob(f"*_{modality}.nii"))
        
        if nii_gz_files:
            found[modality] = str(nii_gz_files[0])
        elif nii_files:
            found[modality] = str(nii_files[0])
        else:
            missing.append(modality)
    
    # Check optional
    seg_found = False
    seg_gz_files = list(subject_path.glob("*_seg.nii.gz"))
    seg_files = list(subject_path.glob("*_seg.nii"))
    if seg_gz_files or seg_files:
        seg_found = True
    
    return {
        'is_valid': len(missing) == 0,
        'found_modalities': found,
        'missing_modalities': missing,
        'has_segmentation': seg_found,
        'num_files': len(list(subject_path.glob('*.nii*'))),
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
    if nifti_file_path.endswith('.nii.gz'):
        with gzip.open(nifti_file_path, 'rb') as f_in:
            nifti_path = subject_dir / f"{subject_name}_t2.nii.gz"
            with open(nifti_path, 'wb') as f_out:
                sh.copyfileobj(f_in, f_out)
    else:
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
