'use client';

import { useState } from 'react';
import ImageUpload from '@/components/ImageUpload';

interface AnalysisResult {
  result: 'clean' | 'tumor' | 'no_tumor';
  confidence: number;
  message?: string;
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
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900">
                Medical Image Analyzer
              </h1>
              <p className="text-gray-600 mt-2">
                AI-powered CT/MRI tumor detection system
              </p>
            </div>
            <div className="text-right">
              <div className="inline-block bg-blue-100 text-blue-800 px-4 py-2 rounded-lg font-medium">
                ✓ AI Ready
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">
                Upload Medical Image
              </h2>
              <ImageUpload
                onAnalysisComplete={handleAnalysisComplete}
                apiBaseUrl="http://127.0.0.1:8000"
              />
            </div>
          </div>

          {/* Info & History */}
          <div className="space-y-6">
            {/* Info Card */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                How it works
              </h3>
              <ul className="space-y-3 text-sm text-gray-700">
                <li className="flex gap-3">
                  <span className="text-blue-600 font-bold flex-shrink-0">1</span>
                  <span>Upload a CT or MRI image</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-blue-600 font-bold flex-shrink-0">2</span>
                  <span>Our AI analyzes the image</span>
                </li>
                <li className="flex gap-3">
                  <span className="text-blue-600 font-bold flex-shrink-0">3</span>
                  <span>Get instant results with confidence score</span>
                </li>
              </ul>

              <div className="mt-6 pt-6 border-t border-gray-200">
                <h4 className="font-semibold text-gray-900 mb-3">
                  Supported formats
                </h4>
                <p className="text-xs text-gray-600">
                  JPG, PNG, GIF, BMP, and DICOM files up to 50MB
                </p>
              </div>
            </div>

            {/* Analysis History */}
            {analysisHistory.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Recent Analyses
                </h3>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {analysisHistory.map((item, index) => (
                    <div
                      key={index}
                      className="p-3 bg-gray-50 rounded border border-gray-200"
                    >
                      <p className="text-sm font-medium text-gray-900">
                        {item.file}
                      </p>
                      <div className="flex justify-between items-center mt-2">
                        <span
                          className={`text-xs font-semibold px-2 py-1 rounded ${
                            item.result.result === 'clean' ||
                            item.result.result === 'no_tumor'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {item.result.result.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-500">
                          {(item.result.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
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
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-center text-sm text-gray-600">
          <p>
            Built with Next.js 16 & Django REST Framework |{' '}
            <span className="text-gray-500">Backend: http://127.0.0.1:8000</span>
          </p>
        </div>
      </footer>
    </div>
  );
}
