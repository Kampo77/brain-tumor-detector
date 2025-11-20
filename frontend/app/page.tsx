'use client';

import { useState } from 'react';
import ImageUpload from '@/components/ImageUpload';
import BraTS3DUpload from '@/components/BraTS3DUpload';
import HealthCheck from '@/components/HealthCheck';

interface AnalysisResult {
  prediction: 'Normal' | 'Tumor';
  confidence: number;
  class_index?: number;
  error?: string | null;
}

export default function Home() {
  const [analysisTab, setAnalysisTab] = useState<'2d' | '3d'>('3d');
  const [analysisHistory, setAnalysisHistory] = useState<
    Array<{ file: string; result: AnalysisResult; timestamp: string }>
  >([]);
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

  const handleAnalysisComplete = (result: AnalysisResult) => {
    const timestamp = new Date().toLocaleString();
    setAnalysisHistory((prev) => [
      { file: `Analysis #${prev.length + 1}`, result, timestamp },
      ...prev,
    ]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-blue-950 to-slate-950">
      {/* Header */}
      <header className="border-b border-blue-900/50 bg-slate-900/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white tracking-tight">
                Medical Analysis
              </h1>
              <p className="text-blue-300 text-sm mt-1">Diagnostic System</p>
            </div>
            <div className="flex items-center gap-3 bg-blue-900/30 border border-blue-700/50 px-4 py-2 rounded-lg backdrop-blur">
              <div className="w-2.5 h-2.5 bg-emerald-400 rounded-full animate-pulse"></div>
              <span className="text-blue-200 text-sm font-medium">System Ready</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="bg-slate-900/80 border border-blue-900/50 rounded-xl shadow-2xl p-8 backdrop-blur">
              {/* Tab Selector */}
              <div className="flex gap-2 mb-8">
                <button
                  onClick={() => setAnalysisTab('2d')}
                  className={`px-6 py-3 font-semibold rounded-lg transition-all duration-300 ${
                    analysisTab === '2d'
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                      : 'bg-slate-800 text-blue-300 hover:bg-slate-700 border border-blue-900/30'
                  }`}
                >
                  2D Analysis
                </button>
                <button
                  onClick={() => setAnalysisTab('3d')}
                  className={`px-6 py-3 font-semibold rounded-lg transition-all duration-300 ${
                    analysisTab === '3d'
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                      : 'bg-slate-800 text-blue-300 hover:bg-slate-700 border border-blue-900/30'
                  }`}
                >
                  3D Volume
                </button>
              </div>

              {/* 2D Upload */}
              {analysisTab === '2d' && (
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">2D MRI Scan</h2>
                  <p className="text-blue-300 mb-6">Upload single MRI slice image</p>
                  <ImageUpload
                    onAnalysisComplete={handleAnalysisComplete}
                    apiBaseUrl={apiBaseUrl}
                  />
                </div>
              )}

              {/* 3D Upload */}
              {analysisTab === '3d' && (
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">3D MRI Volume</h2>
                  <p className="text-blue-300 mb-6">Upload complete BraTS 3D volume dataset</p>
                  <BraTS3DUpload
                    onResultComplete={() => {
                      console.log('3D analysis complete');
                    }}
                    apiBaseUrl={apiBaseUrl}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Info & History */}
          <div className="space-y-6">
            {/* Info Card - Clean and minimal */}
            <div className="bg-slate-900/80 border border-blue-900/50 rounded-xl shadow-2xl p-6 backdrop-blur">
              <h3 className="text-lg font-bold text-white mb-4">Model Information</h3>

              {analysisTab === '2d' ? (
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-blue-300">Architecture</span>
                    <span className="text-white font-medium">ResNet18 2D</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-300">Input Format</span>
                    <span className="text-white font-medium">Single Slice</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-300">Max File Size</span>
                    <span className="text-white font-medium">50 MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-300">Processing</span>
                    <span className="text-white font-medium">Real-time</span>
                  </div>
                </div>
              ) : (
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-blue-300">Architecture</span>
                    <span className="text-white font-medium">3D U-Net</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-300">Input Modalities</span>
                    <span className="text-white font-medium">4 (FLAIR, T1, T1ce, T2)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-300">Max File Size</span>
                    <span className="text-white font-medium">500 MB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-300">Validation Score</span>
                    <span className="text-white font-medium">0.7751 Dice</span>
                  </div>
                </div>
              )}

              <div className="mt-6 pt-6 border-t border-blue-900/30">
                <h4 className="font-semibold text-white mb-3 text-sm">Process</h4>
                <ol className="space-y-2 text-xs text-blue-300">
                  <li className="flex gap-2">
                    <span className="font-bold text-blue-400">1.</span>
                    <span>{analysisTab === '2d' ? 'Upload MRI image' : 'Upload BraTS dataset ZIP'}</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-bold text-blue-400">2.</span>
                    <span>AI processes {analysisTab === '2d' ? 'slice' : '3D volume'}</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-bold text-blue-400">3.</span>
                    <span>Get detailed results instantly</span>
                  </li>
                </ol>
              </div>
            </div>

            {/* Analysis History */}
            {analysisHistory.length > 0 && (
              <div className="bg-slate-900/80 border border-blue-900/50 rounded-xl shadow-2xl p-6 backdrop-blur">
                <h3 className="text-lg font-bold text-white mb-4">Analysis History</h3>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  {analysisHistory.map((item, index) => (
                    <div
                      key={index}
                      className="p-4 bg-slate-800/50 rounded-lg border border-blue-900/30 hover:border-blue-700/50 transition-all"
                    >
                      <p className="text-sm font-semibold text-white">
                        {item.file}
                      </p>
                      <div className="flex justify-between items-center mt-2">
                        <span
                          className={`text-xs font-bold px-3 py-1 rounded-lg ${
                            item.result.prediction === 'Normal'
                              ? 'bg-emerald-500/30 text-emerald-200 border border-emerald-500/50'
                              : 'bg-red-500/30 text-red-200 border border-red-500/50'
                          }`}
                        >
                          {item.result.prediction}
                        </span>
                        <span className="text-xs text-blue-300 font-semibold">
                          {(item.result.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-xs text-blue-300/50 mt-2">
                        {item.timestamp}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-blue-900/30 mt-12 py-6 bg-slate-900/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-blue-300 text-sm">
            Medical Analysis System • Powered by PyTorch & Next.js
          </p>
          <p className="text-blue-400/60 text-xs mt-2 font-mono">
            API: http://127.0.0.1:8000
          </p>
        </div>
      </footer>

      {/* Health Check Indicator */}
      <HealthCheck />
    </div>
  );
}
