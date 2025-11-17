'use client';

import { useState, useRef } from 'react';

interface AnalysisResult {
  result: 'clean' | 'tumor' | 'no_tumor';
  confidence: number;
  message?: string;
}

interface ImageUploadProps {
  onAnalysisComplete?: (result: AnalysisResult) => void;
  apiBaseUrl?: string;
}

export default function ImageUpload({
  onAnalysisComplete,
  apiBaseUrl = 'http://127.0.0.1:8000',
}: ImageUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Valid image and medical file extensions
  const VALID_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'dcm'];
  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

  /**
   * Validate file before upload
   */
  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > MAX_FILE_SIZE) {
      return `File size must be less than ${MAX_FILE_SIZE / 1024 / 1024}MB. Your file is ${(file.size / 1024 / 1024).toFixed(2)}MB.`;
    }

    // Check file extension
    const fileName = file.name.toLowerCase();
    const hasValidExtension = VALID_EXTENSIONS.some((ext) =>
      fileName.endsWith(`.${ext}`)
    );

    if (!hasValidExtension) {
      return `Invalid file type. Supported formats: ${VALID_EXTENSIONS.join(', ').toUpperCase()}`;
    }

    return null;
  };

  /**
   * Handle file selection
   */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    // Validate file
    const validationError = validateFile(selectedFile);
    if (validationError) {
      setError(validationError);
      setFile(null);
      setPreview(null);
      return;
    }

    setError(null);
    setFile(selectedFile);
    setResult(null); // Clear previous results

    // Create preview for image files (not for DICOM)
    if (!selectedFile.name.toLowerCase().endsWith('.dcm')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null); // DICOM files don't have easy preview
    }
  };

  /**
   * Handle drag and drop
   */
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    const droppedFile = e.dataTransfer.files?.[0];
    if (!droppedFile) return;

    // Validate file
    const validationError = validateFile(droppedFile);
    if (validationError) {
      setError(validationError);
      setFile(null);
      setPreview(null);
      return;
    }

    setError(null);
    setFile(droppedFile);
    setResult(null);

    // Create preview for image files
    if (!droppedFile.name.toLowerCase().endsWith('.dcm')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(droppedFile);
    } else {
      setPreview(null);
    }
  };

  /**
   * Upload file and get analysis
   */
  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${apiBaseUrl}/analyze/`, {
        method: 'POST',
        body: formData,
        // Note: Don't set Content-Type header; browser will set it automatically
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error ||
            `Server error: ${response.status} ${response.statusText}`
        );
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
      onAnalysisComplete?.(data);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'Unknown error occurred';
      setError(
        `Failed to analyze image: ${errorMessage}. Make sure the backend is running at ${apiBaseUrl}`
      );
    } finally {
      setLoading(false);
    }
  };

  /**
   * Reset form
   */
  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center bg-blue-50 hover:bg-blue-100 transition-colors cursor-pointer"
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.gif,.bmp,.dcm"
          onChange={handleFileChange}
          disabled={loading}
          className="hidden"
        />

        <svg
          className="mx-auto h-12 w-12 text-blue-400 mb-4"
          stroke="currentColor"
          fill="none"
          viewBox="0 0 48 48"
          aria-hidden="true"
        >
          <path
            d="M28 8H12a4 4 0 00-4 4v20a4 4 0 004 4h24a4 4 0 004-4V20m-4-12l-8-8m8 8v12m-16 4l-4-4m4 4l4-4"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>

        <p className="text-lg font-medium text-gray-900">
          Drag & drop your image here
        </p>
        <p className="text-sm text-gray-600 mt-2">
          or click to select a file
        </p>
        <p className="text-xs text-gray-500 mt-2">
          Supported: JPG, PNG, GIF, BMP, DICOM (max 50MB)
        </p>
      </div>

      {/* File Info */}
      {file && (
        <div className="mt-6 p-4 bg-gray-100 rounded-lg">
          <p className="text-sm font-medium text-gray-900">Selected file:</p>
          <p className="text-sm text-gray-600 mt-1">{file.name}</p>
          <p className="text-xs text-gray-500 mt-1">
            {(file.size / 1024).toFixed(2)} KB
          </p>
        </div>
      )}

      {/* Preview */}
      {preview && (
        <div className="mt-6">
          <p className="text-sm font-medium text-gray-900 mb-3">Preview:</p>
          <img
            src={preview}
            alt="Selected image preview"
            className="w-full max-h-96 object-contain rounded-lg border border-gray-300"
          />
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm font-medium text-red-800">Error:</p>
          <p className="text-sm text-red-700 mt-1">{error}</p>
        </div>
      )}

      {/* Analysis Result */}
      {result && (
        <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-lg font-semibold text-green-900 mb-3">
            Analysis Result
          </p>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-700">Status:</span>
              <span
                className={`font-semibold px-3 py-1 rounded ${
                  result.result === 'clean' || result.result === 'no_tumor'
                    ? 'bg-green-200 text-green-900'
                    : 'bg-red-200 text-red-900'
                }`}
              >
                {result.result.toUpperCase()}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-700">Confidence:</span>
              <div className="flex items-center gap-3">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      result.confidence > 0.8
                        ? 'bg-green-500'
                        : 'bg-yellow-500'
                    }`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
                <span className="font-semibold text-gray-900 min-w-fit">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {result.message && (
              <div className="pt-3 border-t border-green-200">
                <p className="text-sm text-gray-600">{result.message}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="mt-6 flex gap-3">
        {!result ? (
          <>
            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="flex-1 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <svg
                    className="animate-spin h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Analyzing...
                </>
              ) : (
                'Analyze Image'
              )}
            </button>

            <button
              onClick={handleReset}
              disabled={!file || loading}
              className="px-6 py-3 bg-gray-300 text-gray-900 font-medium rounded-lg hover:bg-gray-400 disabled:bg-gray-200 disabled:cursor-not-allowed transition-colors"
            >
              Clear
            </button>
          </>
        ) : (
          <button
            onClick={handleReset}
            className="w-full px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
          >
            Analyze Another Image
          </button>
        )}
      </div>
    </div>
  );
}
