'use client';

import { useState, useRef } from 'react';

interface AnalysisResult {
  prediction: 'Normal' | 'Tumor';
  confidence: number;
  class_index?: number;
  error?: string | null;
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
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const VALID_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'dcm'];
  const MAX_FILE_SIZE = 50 * 1024 * 1024;

  const validateFile = (file: File): string | null => {
    if (file.size > MAX_FILE_SIZE) return 'File too large';
    const fileName = file.name.toLowerCase();
    const hasValid = VALID_EXTENSIONS.some((ext) => fileName.endsWith(`.${ext}`));
    return hasValid ? null : 'Invalid file type';
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    const err = validateFile(selectedFile);
    if (err) {
      setError(err);
      setFile(null);
      setPreview(null);
      return;
    }
    setError(null);
    setFile(selectedFile);
    setResult(null);
    if (!selectedFile.name.toLowerCase().endsWith('.dcm')) {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }
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
    if (!droppedFile.name.toLowerCase().endsWith('.dcm')) {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(droppedFile);
    } else {
      setPreview(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('image', file);
      const response = await fetch(`${apiBaseUrl}/api/predict/`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || `Error: ${response.status}`);
      }
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResult(data);
      onAnalysisComplete?.(data);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to analyze image. Make sure backend is running.';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 group overflow-hidden ${
          isDragging
            ? 'border-blue-400 bg-blue-500/10 shadow-xl shadow-blue-400/20'
            : 'border-blue-400/30 bg-blue-900/20 hover:border-blue-400/60 hover:bg-blue-500/5'
        }`}
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
        <div className="relative z-10">
          <div className="text-5xl mb-4 font-light text-blue-300">Upload</div>
          <h3 className="text-xl font-semibold text-white mb-2">
            Drop your MRI image here
          </h3>
          <p className="text-blue-300 text-sm mb-4">
            or click to browse from your computer
          </p>
          <p className="text-xs text-blue-400/60">
            Supported: JPG, PNG, GIF, BMP, DICOM • Max 50 MB
          </p>
        </div>
      </div>

      {file && (
        <div className="mt-6 p-4 bg-slate-900/50 backdrop-blur-md border border-blue-900/50 rounded-xl">
          <p className="text-sm font-bold text-blue-300 mb-2">Selected: {file.name}</p>
          <p className="text-xs text-blue-400/70">{(file.size / 1024).toFixed(2)} KB</p>
        </div>
      )}

      {preview && (
        <div className="mt-8">
          <p className="text-sm font-bold text-blue-300 mb-4">
            Preview
          </p>
          <img
            src={preview}
            alt="Preview"
            className="w-full max-h-96 object-contain rounded-xl border border-blue-900/50 shadow-2xl shadow-blue-500/10 hover:shadow-blue-500/20 transition-shadow"
          />
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-500/20 border border-red-400/50 rounded-xl backdrop-blur-md">
          <p className="text-red-300 text-sm font-bold">Error: {error}</p>
        </div>
      )}

      {result && (
        <div className="mt-8 space-y-6 animate-in fade-in">
          <div
            className={`relative p-8 rounded-2xl border-2 backdrop-blur-xl overflow-hidden group ${
              result.prediction === 'Normal'
                ? 'border-green-900/50 bg-slate-900/80'
                : 'border-red-900/50 bg-slate-900/80'
            }`}
          >
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <p
                  className={`text-xs font-bold uppercase ${
                    result.prediction === 'Normal' ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  Result
                </p>
                <p
                  className={`text-5xl font-black mt-3 ${
                    result.prediction === 'Normal'
                      ? 'text-green-300'
                      : 'text-red-300'
                  }`}
                >
                  {result.prediction}
                </p>
              </div>
              <div
                className={`text-7xl ${
                  result.prediction === 'Normal' ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {result.prediction === 'Normal' ? '✓' : '!'}
              </div>
            </div>
          </div>

          <div className="bg-slate-900/80 backdrop-blur-xl p-6 rounded-xl border border-blue-900/50">
            <p className="text-xs font-bold uppercase text-blue-400 mb-4">Confidence</p>
            <div className="w-full bg-slate-800/80 rounded-full h-4 overflow-hidden border border-blue-900/30">
              <div
                className={`h-full transition-all duration-700 ${
                  result.prediction === 'Normal'
                    ? 'bg-gradient-to-r from-green-600 to-green-400'
                    : 'bg-gradient-to-r from-red-600 to-red-400'
                }`}
                style={{ width: `${result.confidence * 100}%` }}
              ></div>
            </div>
            <p className="text-right text-blue-300 font-bold mt-2">
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      )}

      <div className="mt-6 flex gap-4">
        {!result ? (
          <>
            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 disabled:opacity-50 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-blue-500/50 disabled:shadow-none flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                  Analyzing...
                </>
              ) : (
                <>Analyze Image</>
              )}
            </button>
            <button
              onClick={handleReset}
              disabled={!file || loading}
              className="px-6 py-4 border border-blue-600/50 rounded-xl text-blue-300 font-semibold hover:bg-blue-600/10 transition-all disabled:opacity-50"
            >
              Clear
            </button>
          </>
        ) : (
          <button
            onClick={handleReset}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg flex items-center justify-center gap-2"
          >
            Analyze Another
          </button>
        )}
      </div>
    </div>
  );
}
