'use client';

import { useEffect, useState } from 'react';

interface HealthStatus {
  message: string;
  backend: string;
  models: {
    '2d_model': boolean;
    '3d_model': boolean;
  };
  device: string;
}

export default function HealthCheck() {
  const [status, setStatus] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        setChecking(true);
        const response = await fetch('http://127.0.0.1:8000/api/ping/', {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
        });

        if (!response.ok) {
          throw new Error(`Backend error: ${response.status}`);
        }

        const data = await response.json();
        setStatus(data);
        setError(null);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : 'Cannot connect to backend at http://127.0.0.1:8000'
        );
        setStatus(null);
      } finally {
        setChecking(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, []);

  if (checking && !status && !error) {
    return null;
  }

  if (error) {
    return (
      <div className="fixed bottom-4 right-4 max-w-sm">
        <div className="bg-red-900/80 border border-red-500/50 rounded-lg p-4 backdrop-blur">
          <p className="text-red-200 text-sm font-bold">Backend Connection Issue</p>
          <p className="text-red-300 text-xs mt-1">{error}</p>
          <div className="text-red-400 text-xs mt-3 space-y-1">
            <p className="font-mono bg-red-950/50 px-2 py-1 rounded">
              cd backend && python3 manage.py runserver
            </p>
            <p className="text-red-300">Make sure Django is running on port 8000</p>
          </div>
        </div>
      </div>
    );
  }

  if (status) {
    return (
      <div className="fixed bottom-4 right-4 max-w-sm">
        <div className="bg-green-900/80 border border-green-500/50 rounded-lg p-4 backdrop-blur">
          <p className="text-green-200 text-sm font-bold">Backend Connected</p>
          <div className="text-xs text-green-300 mt-2 space-y-1">
            <p>2D Model: {status.models['2d_model'] ? '✓ Ready' : '✗ Missing'}</p>
            <p>3D Model: {status.models['3d_model'] ? '✓ Ready' : '✗ Missing'}</p>
            <p>Device: {status.device}</p>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
