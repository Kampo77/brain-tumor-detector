"""BraTS 3D brain tumor detection views for Django REST API."""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from pathlib import Path
import tempfile
import shutil
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.brats_utils import BraTSModelManager, extract_brats_zip, validate_brats_subject


class BraTSPredictView(APIView):
    """
    API endpoint for BraTS 3D brain tumor detection.
    
    Accepts:
    - Single NIfTI file (will use as one modality)
    - Zipped BraTS subject directory
    
    Returns:
    - Tumor presence (bool)
    - Tumor fraction (0-1)
    - Confidence scores
    - Overlay PNG visualization (base64)
    """
    
    parser_classes = (MultiPartParser, FormParser)
    
    def get_model_path(self):
        """Get path to trained BraTS model."""
        model_path = Path(__file__).parent.parent / "model" / "brats2020_unet3d.pth"
        return str(model_path)
    
    def post(self, request):
        """
        POST endpoint for BraTS predictions.
        
        Form parameters:
        - file: ZIP file containing BraTS subject directory OR NIFTI file
        
        Accepts:
        - .zip files with BraTS structure
        - .nii or .nii.gz NIFTI files (single 3D volume)
        
        Returns:
        {
            'success': bool,
            'has_tumor': bool,
            'tumor_fraction': float (0-1),
            'confidence': float (0-1),
            'message': str,
            'subject_id': str (optional),
            'overlay_png': str (base64 data URI)
        }
        """
        try:
            # Check if file is provided
            if 'file' not in request.FILES:
                return Response(
                    {
                        'success': False,
                        'error': 'No file provided. Please upload a ZIP file or NIFTI file (.nii/.nii.gz).',
                        'expected_format': 'ZIP with BraTS20_Training_XXX/ or NIFTI volume'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            uploaded_file = request.FILES['file']
            file_name = uploaded_file.name.lower()
            
            # Validate file format
            is_zip = file_name.endswith('.zip')
            is_nifti = file_name.endswith('.nii') or file_name.endswith('.nii.gz')
            
            if not (is_zip or is_nifti):
                return Response(
                    {
                        'success': False,
                        'error': f'Invalid file format. Expected .zip or .nii/.nii.gz, got {uploaded_file.name}',
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create temporary directory for extraction
            temp_dir = tempfile.mkdtemp(prefix='brats_')
            
            try:
                # Save uploaded file temporarily
                temp_file = Path(temp_dir) / uploaded_file.name
                with open(temp_file, 'wb') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)
                
                # Extract or process file
                if is_zip:
                    subject_dir = extract_brats_zip(str(temp_file), temp_dir)
                else:  # NIFTI file
                    from api.brats_utils import create_subject_from_nifti
                    subject_dir = create_subject_from_nifti(str(temp_file), temp_dir)
                
                # Validate
                validation = validate_brats_subject(subject_dir)
                if not validation['is_valid']:
                    return Response(
                        {
                            'success': False,
                            'error': f"Invalid BraTS subject directory. Missing modalities: {validation['missing_modalities']}",
                            'validation': validation,
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Get model path
                model_path = self.get_model_path()
                if not Path(model_path).exists():
                    return Response(
                        {
                            'success': False,
                            'error': f'Model not found at {model_path}. Please train the model first.',
                        },
                        status=status.HTTP_503_SERVICE_UNAVAILABLE
                    )
                
                # Run prediction WITH VISUALIZATION
                print(f"Running BraTS prediction with visualization on {subject_dir}")
                result = BraTSModelManager.predict_with_visualization(subject_dir, model_path)
                
                # Extract subject ID from directory name
                subject_name = Path(subject_dir).name
                
                return Response(
                    {
                        'success': True,
                        'has_tumor': result['has_tumor'],
                        'tumor_fraction': result['tumor_fraction'],
                        'confidence': 1.0 - result['tumor_fraction'] if result['has_tumor'] else result['tumor_fraction'],
                        'subject_id': subject_name,
                        'message': f"{'Tumor detected' if result['has_tumor'] else 'No tumor detected'} ({result['tumor_fraction']:.1%} of volume)",
                        'overlay_png': result.get('overlay_png'),  # Base64 PNG or None
                        'brain_region': result.get('brain_region', 'Unknown')  # Brain region
                    },
                    status=status.HTTP_200_OK
                )
            
            finally:
                # Cleanup temp directory
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
        
        except Exception as e:
            print(f"Error in BraTS prediction: {e}")
            import traceback
            traceback.print_exc()
            
            return Response(
                {
                    'success': False,
                    'error': f'Internal server error: {str(e)}',
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BraTSHealthView(APIView):
    """Health check endpoint for BraTS model."""
    
    def get(self, request):
        """Check if BraTS model is available."""
        model_path = Path(__file__).parent.parent / "model" / "brats2020_unet3d.pth"
        
        return Response(
            {
                'model_available': model_path.exists(),
                'model_path': str(model_path),
                'message': 'BraTS model service is ready' if model_path.exists() else 'Model not trained yet'
            },
            status=status.HTTP_200_OK
        )
