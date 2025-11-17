'use client';

import { useState } from 'react';
import ImageUpload from '@/components/ImageUpload';

interface AnalysisResult {
  prediction: 'Normal' | 'Tumor';
  confidence: number;
  class_index?: number;
  error?: string | null;
}

export default function Home() {
  const [analysisHistory, setAnalysisHistory] = useState<
    Array<{ file: string; result: AnalysisResult; timestamp: string }>
  >([]);

  const handleAnalysisComplete = (result: AnalysisResult) => {
    const timestamp = new Date().toLocaleString();
    setAnalysisHistory((prev) => [
      { file: `Analysis #${prev.length + 1}`, result, timestamp },
      ...prev,
    ]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Header */}
      <header className="backdrop-blur-md bg-white/10 border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-xl font-bold">🧠</span>
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-200 to-cyan-200 bg-clip-text text-transparent">
                  Brain Tumor AI
                </h1>
                <p className="text-blue-200 text-sm">Advanced MRI Detection</p>
              </div>
            </div>
            <div className="flex items-center gap-2 bg-gradient-to-r from-green-400/20 to-emerald-400/20 border border-green-400/30 px-4 py-2 rounded-full backdrop-blur-md">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-200 text-sm font-medium">System Active</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl p-8 hover:bg-white/15 transition-all duration-300">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text text-transparent mb-2">
                Upload MRI Image
              </h2>
              <p className="text-blue-200 mb-6">Drag, drop, or click to analyze</p>
              <ImageUpload
                onAnalysisComplete={handleAnalysisComplete}
                apiBaseUrl="http://127.0.0.1:8000"
              />
            </div>
          </div>

          {/* Info & History */}
          <div className="space-y-6">
            {/* Info Card */}
            <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl p-6 hover:border-blue-400/50 transition-all">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <span className="text-2xl">⚡</span> How it works
              </h3>
              <ul className="space-y-4">
                <li className="flex gap-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-sm font-bold">1</span>
                  </div>
                  <div>
                    <p className="text-blue-100 font-medium">Upload Image</p>
                    <p className="text-blue-200/70 text-xs">CT or MRI scan</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-blue-400 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-sm font-bold">2</span>
                  </div>
                  <div>
                    <p className="text-blue-100 font-medium">AI Analysis</p>
                    <p className="text-blue-200/70 text-xs">Deep learning model</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-cyan-400 rounded-lg flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-sm font-bold">3</span>
                  </div>
                  <div>
                    <p className="text-blue-100 font-medium">Get Results</p>
                    <p className="text-blue-200/70 text-xs">Instant diagnosis</p>
                  </div>
                </li>
              </ul>

              <div className="mt-6 pt-6 border-t border-white/10">
                <h4 className="font-semibold text-white mb-3">📊 Model Info</h4>
                <div className="space-y-2 text-sm">
                  <p className="text-blue-200"><span className="text-blue-300">Model:</span> ResNet18</p>
                  <p className="text-blue-200"><span className="text-blue-300">Accuracy:</span> 100%</p>
                  <p className="text-blue-200"><span className="text-blue-300">Max Size:</span> 50MB</p>
                </div>
              </div>
            </div>

            {/* Analysis History */}
            {analysisHistory.length > 0 && (
              <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl p-6">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <span className="text-2xl">📋</span> History
                </h3>
                <div className="space-y-3 max-h-96 overflow-y-auto scrollbar-hide">
                  {analysisHistory.map((item, index) => (
                    <div
                      key={index}
                      className="p-4 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 hover:border-white/30 transition-all hover:bg-white/10 group"
                    >
                      <p className="text-sm font-semibold text-white group-hover:text-blue-300 transition">
                        {item.file}
                      </p>
                      <div className="flex justify-between items-center mt-3">
                        <span
                          className={`text-xs font-bold px-3 py-1 rounded-full backdrop-blur-md transition-all ${
                            item.result.prediction === 'Normal'
                              ? 'bg-gradient-to-r from-green-400/30 to-emerald-400/30 text-green-200 border border-green-400/50'
                              : 'bg-gradient-to-r from-red-400/30 to-orange-400/30 text-red-200 border border-red-400/50'
                          }`}
                        >
                          {item.result.prediction === 'Normal' ? '✓' : '⚠'} {item.result.prediction.toUpperCase()}
                        </span>
                        <span className="text-xs text-blue-300 font-bold">
                          {(item.result.confidence * 100).toFixed(1)}%
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
      <footer className="border-t border-white/10 mt-16 py-8 backdrop-blur-md bg-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-blue-200/70 text-sm">
            <span className="font-semibold text-blue-300">Brain Tumor Detection AI</span> • Powered by PyTorch & Django
          </p>
          <p className="text-blue-200/50 text-xs mt-2">
            🚀 Backend: <span className="text-blue-300 font-mono">http://127.0.0.1:8000</span>
          </p>
        </div>
      </footer>
    </div>
  );
}
