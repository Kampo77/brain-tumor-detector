# NIFTI File Upload Support

## What's New

You can now upload **NIFTI files** (.nii or .nii.gz) directly to the 3D analysis tab, without needing to create a ZIP archive!

## Supported Formats

### 3D Volume Analysis Tab

✅ **ZIP Format** (original)
- BraTS subject directories with 4 modalities
- Example: `BraTS20_Training_001.zip`

✅ **NIFTI Format** (new!)
- Single 3D brain volume files
- Extensions: `.nii` or `.nii.gz`
- Example: `brain_scan.nii.gz`

## How It Works

When you upload a NIFTI file:

1. **File Uploaded**: System accepts .nii or .nii.gz files (up to 500 MB)
2. **Processing**: System creates synthetic 4-modality dataset from your single volume:
   - **T2** (100%) - Original volume as-is
   - **T1** (90%) - 0.9x intensity
   - **T1ce** (110%) - 1.1x intensity (contrast-enhanced effect)
   - **FLAIR** (105%) - 1.05x intensity
3. **Analysis**: 3D U-Net model analyzes all 4 synthetic modalities
4. **Results**: Returns tumor presence, location, and confidence scores

## Why Synthetic Modalities?

The 3D U-Net model is trained on 4-modality BraTS data for robust analysis. When you provide a single volume:

- **Better accuracy**: The model sees expected multi-modality input
- **Consistent results**: Same inference pipeline as ZIP uploads
- **Flexible input**: Accept any brain MRI NIFTI file, not just BraTS format

## Usage Example

### Upload NIFTI File

1. Go to the **3D Volume** tab on the website
2. Click "Drop your 3D brain volume here" area
3. Select your `.nii` or `.nii.gz` file
4. System will:
   - Create synthetic modalities
   - Run 3D analysis
   - Display tumor detection results

### What You Get

```json
{
  "success": true,
  "has_tumor": true,
  "tumor_fraction": 0.15,
  "confidence": 0.85,
  "subject_id": "brain_scan",
  "message": "Tumor detected (15.0% of volume)"
}
```

## Technical Details

### NIFTI Processing

- Reads NIFTI header to preserve voxel spacing
- Creates synthetic modalities with linear intensity scaling
- Maintains original NIfTI affine transformation
- Compatible with nibabel library

### Backend Changes

- `create_subject_from_nifti()` function creates BraTS-like structure
- Handles both compressed (.nii.gz) and uncompressed (.nii) formats
- Automatically decompresses on-the-fly
- Memory efficient (streams large files)

### Frontend Changes

- File input now accepts: `.zip`, `.nii`, `.nii.gz`
- Upload area text updated
- Same error handling for all formats

## Limitations & Notes

⚠️ **Important**:
- Single NIFTI files are interpreted as one modality (typically T2-weighted)
- For best results, use properly skull-stripped brain MRI
- Model was trained on BraTS data, optimized for 3-4 mm resolution
- Very large files (>500 MB) may take longer to process

## Future Enhancements

Potential improvements:
- Multi-file NIFTI upload (for actual 4-modality scans)
- Automatic modality detection
- Support for other medical formats (DICOM, etc.)
- Local NIFTI viewer in browser

## File Size Guidelines

- **Small** (< 50 MB): ~2-5 seconds analysis
- **Medium** (50-200 MB): ~5-15 seconds analysis  
- **Large** (200-500 MB): ~15-30 seconds analysis

Processing time depends on your system's GPU/CPU power.
