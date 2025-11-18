'use client';

import { useState, useRef } from 'react';

interface BraTSResult {
  success: boolean;
  has_tumor: boolean;
  tumor_fraction: number;
  confidence: number;
  subject_id: string;
  message: string;
  error?: string;
  modalities?: Record<string, number>;
  stats?: {
    affected_regions: string;
  };
}

interface BraTS3DUploadProps {
  onResultComplete?: (result: BraTSResult) => void;
  apiBaseUrl?: string;
}

export default function BraTS3DUpload({
  onResultComplete,
  apiBaseUrl = 'http://127.0.0.1:8000',
}: BraTS3DUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BraTSResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    const fileName = file.name.toLowerCase();
    const isZip = fileName.endsWith('.zip');
    const isNii = fileName.endsWith('.nii') || fileName.endsWith('.nii.gz');
    
    if (!isZip && !isNii) {
      return 'File must be ZIP or NIFTI (.nii/.nii.gz) format';
    }
    
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      return `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB > 500MB)`;
    }
    return null;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    const err = validateFile(selectedFile);
    if (err) {
      setError(err);
      setFile(null);
      return;
    }

    setError(null);
    setFile(selectedFile);
    setResult(null);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files?.[0];
    if (!droppedFile) return;

    const err = validateFile(droppedFile);
    if (err) {
      setError(err);
      return;
    }

    setError(null);
    setFile(droppedFile);
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a ZIP file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${apiBaseUrl}/api/brats/predict/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Error: ${response.status}`);
      }

      const data: BraTSResult = await response.json();
      setResult(data);
      onResultComplete?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
          isDragging
            ? 'border-blue-400 bg-blue-500/10 shadow-xl shadow-blue-400/20'
            : 'border-blue-400/30 bg-blue-900/20 hover:border-blue-400/60 hover:bg-blue-500/5'
        }`}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".zip,.nii,.nii.gz"
          onChange={handleFileChange}
          disabled={loading}
          className="hidden"
        />

        <div className="relative z-10">
          <div className="text-5xl mb-4 font-light text-blue-300">Upload</div>
          <h3 className="text-xl font-semibold text-white mb-2">
            Drop your 3D brain volume here
          </h3>
          <p className="text-blue-300 text-sm mb-4">
            or click to browse from your computer
          </p>
          <p className="text-xs text-blue-400/60">
            ZIP, NIFTI (.nii), or Compressed NIFTI (.nii.gz) • Maximum 500 MB
          </p>
        </div>
      </div>

      {/* File Info */}
      {file && !result && (
        <div className="mt-6 p-4 bg-slate-900/50 backdrop-blur-md border border-blue-900/50 rounded-xl">
          <p className="text-sm font-bold text-blue-300 mb-2">Selected File</p>
          <p className="text-sm text-blue-200/90 font-mono break-all">{file.name}</p>
          <p className="text-xs text-blue-400/70 mt-1">
            Size: {(file.size / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-6 p-4 bg-red-500/20 backdrop-blur-md border border-red-400/50 rounded-xl">
          <p className="text-sm font-bold text-red-300">Error</p>
          <p className="text-sm text-red-200 mt-1">{error}</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="mt-6 p-6 bg-slate-900/50 backdrop-blur-md border border-blue-900/50 rounded-xl text-center">
          <div className="flex justify-center mb-4">
            <div className="w-8 h-8 border-4 border-blue-400/30 border-t-blue-400 rounded-full animate-spin"></div>
          </div>
          <p className="text-sm font-bold text-blue-300">Analyzing 3D MRI volume...</p>
          <p className="text-xs text-blue-400/70 mt-2">This may take 1-3 seconds</p>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="mt-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {result.success ? (
            <div className="p-6 bg-slate-900/80 backdrop-blur-md border border-green-900/50 rounded-xl">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-sm font-bold text-blue-400 mb-1">Subject</p>
                  <p className="text-lg font-mono text-blue-200">{result.subject_id}</p>
                </div>
                <div
                  className={`px-4 py-2 rounded-lg font-bold text-sm ${
                    result.has_tumor
                      ? 'bg-red-500/30 text-red-200 border border-red-400/50'
                      : 'bg-green-500/30 text-green-200 border border-green-400/50'
                  }`}
                >
                  {result.has_tumor ? 'Tumor' : 'Normal'}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-900/20 backdrop-blur-sm p-3 rounded-lg border border-blue-900/30">
                  <p className="text-xs text-blue-400/70 font-semibold uppercase">Tumor Fraction</p>
                  <p className="text-2xl font-bold text-blue-200 mt-1">
                    {(result.tumor_fraction * 100).toFixed(2)}%
                  </p>
                </div>

                <div className="bg-blue-900/20 backdrop-blur-sm p-3 rounded-lg border border-blue-900/30">
                  <p className="text-xs text-blue-400/70 font-semibold uppercase">Confidence</p>
                  <p className="text-2xl font-bold text-blue-200 mt-1">
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mt-4">
                <div className="flex justify-between mb-2">
                  <p className="text-xs text-blue-400/70 font-semibold">Tumor Detection Score</p>
                  <p className="text-xs text-blue-400/70 font-semibold">
                    {(result.confidence * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden border border-blue-900/30">
                  <div
                    className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-500"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              <p className="text-sm text-blue-300 mt-4 text-center italic">
                {result.message}
              </p>
            </div>
          ) : (
            <div className="p-6 bg-red-500/20 backdrop-blur-md border border-red-400/50 rounded-xl">
              <p className="text-sm font-bold text-red-300 mb-2">Analysis Failed</p>
              <p className="text-sm text-red-200">{result.error || 'Unknown error'}</p>
            </div>
          )}

          <button
            onClick={handleReset}
            className="w-full mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg transition-all duration-300 transform hover:scale-105 active:scale-95"
          >
            Analyze Another File
          </button>
        </div>
      )}

      {/* Analyze Button */}
      {file && !result && !loading && (
        <button
          onClick={handleAnalyze}
          className="w-full mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-lg shadow-blue-600/50 hover:shadow-blue-500/70"
        >
          Analyze 3D Volume
        </button>
      )}
    </div>
  );
}
