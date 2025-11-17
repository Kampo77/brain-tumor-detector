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
      if (!response.ok) throw new Error(`Error: ${response.status}`);
      const data: AnalysisResult = await response.json();
      setResult(data);
      onAnalysisComplete?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed');
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
            ? 'border-cyan-400 bg-cyan-400/10 shadow-xl shadow-cyan-400/20'
            : 'border-blue-300/50 bg-white/5 hover:border-blue-400/70 hover:bg-blue-400/5'
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
        <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/0 via-cyan-500/0 to-blue-500/0 group-hover:from-blue-500/5 group-hover:via-cyan-500/5 group-hover:to-blue-500/5 transition-all duration-300"></div>
        </div>
        <div className="relative z-10">
          <div className="text-6xl mb-4 animate-bounce">🖼️</div>
          <h3 className="text-2xl font-bold bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text text-transparent mb-2">
            Drop your MRI image
          </h3>
          <p className="text-blue-200 text-sm mb-4">
            or click to select a file from your computer
          </p>
          <p className="text-xs text-blue-300/60">
            Supported: JPG, PNG, GIF, BMP, DICOM • Max 50MB
          </p>
        </div>
      </div>

      {file && (
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 backdrop-blur-md border border-blue-400/30 rounded-xl">
          <p className="text-sm font-bold text-blue-300 mb-2">📄 Selected: {file.name}</p>
          <p className="text-xs text-blue-300/70">{(file.size / 1024).toFixed(2)} KB</p>
        </div>
      )}

      {preview && (
        <div className="mt-8">
          <p className="text-sm font-bold bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text text-transparent mb-4">
            📸 Preview
          </p>
          <img
            src={preview}
            alt="Preview"
            className="w-full max-h-96 object-contain rounded-xl border border-blue-400/30 shadow-2xl shadow-blue-500/20 hover:shadow-blue-500/40 transition-shadow"
          />
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-gradient-to-r from-red-500/20 to-orange-500/20 border border-red-400/50 rounded-xl backdrop-blur-md">
          <p className="text-red-300 text-sm font-bold">⚠️ {error}</p>
        </div>
      )}

      {result && (
        <div className="mt-8 space-y-6 animate-in fade-in">
          <div
            className={`relative p-8 rounded-2xl border-2 backdrop-blur-xl overflow-hidden group ${
              result.prediction === 'Normal'
                ? 'border-green-400/50 bg-gradient-to-br from-green-500/20 to-emerald-500/10'
                : 'border-red-400/50 bg-gradient-to-br from-red-500/20 to-orange-500/10'
            }`}
          >
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <p
                  className={`text-xs font-bold uppercase ${
                    result.prediction === 'Normal' ? 'text-green-300' : 'text-red-300'
                  }`}
                >
                  Result
                </p>
                <p
                  className={`text-5xl font-black mt-3 bg-clip-text text-transparent ${
                    result.prediction === 'Normal'
                      ? 'bg-gradient-to-r from-green-300 to-emerald-300'
                      : 'bg-gradient-to-r from-red-300 to-orange-300'
                  }`}
                >
                  {result.prediction}
                </p>
              </div>
              <div className="text-7xl animate-pulse">
                {result.prediction === 'Normal' ? '✅' : '⚠️'}
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 backdrop-blur-xl p-6 rounded-xl border border-blue-400/30">
            <p className="text-xs font-bold uppercase text-blue-300 mb-4">Confidence</p>
            <div className="w-full bg-blue-900/40 rounded-full h-4 overflow-hidden border border-blue-400/30">
              <div
                className={`h-full transition-all duration-700 bg-gradient-to-r ${
                  result.prediction === 'Normal'
                    ? 'from-green-400 to-emerald-400'
                    : 'from-red-400 to-orange-400'
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
              className="flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 disabled:from-gray-500 disabled:to-gray-600 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-blue-500/50 disabled:shadow-none flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                  Analyzing...
                </>
              ) : (
                <>⚡ Analyze Image</>
              )}
            </button>
            <button
              onClick={handleReset}
              disabled={!file || loading}
              className="px-6 py-4 border border-blue-400/50 rounded-xl text-blue-300 font-semibold hover:bg-white/5 transition-all"
            >
              Clear
            </button>
          </>
        ) : (
          <button
            onClick={handleReset}
            className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg flex items-center justify-center gap-2"
          >
            🔄 Analyze Another
          </button>
        )}
      </div>
    </div>
  );
}
