import React, { useState } from 'react';
import { UploadCard } from '../components/Upload/UploadCard';
import { SummaryCards } from '../components/SummaryCards/SummaryCards';
import { PredictionTable } from '../components/PredictionTable/PredictionTable';
import { LoadingState } from '../components/Loading/LoadingState';
import { ErrorAlert } from '../components/ErrorAlert/ErrorAlert';
import { predictBatchCSV, exportPredictionsToCSV } from '../services/api';
import { PredictionResponse, ApiError } from '../types/api';
import { UploadCloud, Download, RotateCcw, ShieldCheck, CheckCircle2 } from 'lucide-react';
import axios from 'axios';

type PredictionStatus = 'idle' | 'uploading' | 'predicting' | 'success' | 'error';

export const Upload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<PredictionStatus>('idle');
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [predictionResponse, setPredictionResponse] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
    if (status === 'error') {
      setStatus('idle');
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPredictionResponse(null);
    setError(null);
    setStatus('idle');
    setUploadProgress(0);
  };

  const handleSubmit = async (file: File) => {
    setStatus('uploading');
    setUploadProgress(0);
    setError(null);

    try {
      const data = await predictBatchCSV(file, (progressEvent) => {
        if (progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
          if (percent >= 100) {
            setStatus('predicting');
          }
        }
      });

      setPredictionResponse(data);
      setStatus('success');
    } catch (err: unknown) {
      setStatus('error');
      setPredictionResponse(null);

      if (axios.isAxiosError(err)) {
        if (err.response) {
          const detail = err.response.data?.detail;
          if (typeof detail === 'object' && detail !== null) {
            setError({
              message: detail.message || 'Missing required dataset feature columns.',
              missing_features: detail.missing_features,
              status: err.response.status,
            });
          } else if (typeof detail === 'string') {
            setError({
              message: detail,
              status: err.response.status,
            });
          } else {
            setError({
              message: `Server returned error (${err.response.status}). Please verify your CSV format.`,
              status: err.response.status,
            });
          }
        } else if (err.request) {
          setError({
            message: 'Network Failure: Unable to reach FastAPI backend server at http://localhost:8000. Ensure server is running.',
          });
        } else {
          setError({ message: err.message || 'An unexpected error occurred during prediction.' });
        }
      } else {
        setError({ message: 'An unexpected error occurred during prediction.' });
      }
    }
  };

  const handleExport = () => {
    if (predictionResponse && predictionResponse.predictions) {
      const exportFilename = selectedFile
        ? `predictions_${selectedFile.name.replace('.csv', '')}.csv`
        : 'prediction_results.csv';
      exportPredictionsToCSV(predictionResponse.predictions, exportFilename);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center space-x-3">
            <UploadCloud className="w-7 h-7 text-cyan-400" />
            <span>Multiclass Batch Prediction Workflow</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Upload CSV network flow data to evaluate real-time threat classification using the trained Top-14 XGBoost model.
          </p>
        </div>

        {/* Action Header Pill */}
        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs font-mono text-cyan-400">
            Model: XGBoost Top-14
          </span>
          <span className="px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-xs font-mono text-emerald-400 flex items-center space-x-1">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>v1.2 Release</span>
          </span>
        </div>
      </div>

      {/* Error Alert Display */}
      {error && (
        <ErrorAlert
          error={error}
          onRetry={() => selectedFile && handleSubmit(selectedFile)}
          onClose={() => setError(null)}
        />
      )}

      {/* Loading State Display */}
      {(status === 'uploading' || status === 'predicting') && (
        <LoadingState uploadProgress={uploadProgress} />
      )}

      {/* Upload Dropzone (when idle or error) */}
      {(status === 'idle' || status === 'error') && (
        <UploadCard
          onFileSelect={handleFileSelect}
          onSubmit={handleSubmit}
          onClear={handleClear}
          selectedFile={selectedFile}
          isLoading={false}
          uploadProgress={0}
        />
      )}

      {/* Prediction Results View (when success) */}
      {status === 'success' && predictionResponse && (
        <div className="space-y-8 animate-fadeIn">
          {/* Success Banner & Actions Bar */}
          <div className="glass-card p-6 rounded-2xl border border-emerald-500/30 bg-emerald-500/5 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center space-x-3">
              <div className="p-2.5 rounded-xl bg-emerald-500/20 text-emerald-400">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">Batch Classification Complete</h3>
                <p className="text-xs text-slate-300">
                  Successfully evaluated <span className="font-mono font-bold text-cyan-400">{predictionResponse.summary.total_samples.toLocaleString()}</span> network flow records from <span className="font-mono text-slate-200">{selectedFile?.name}</span>.
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={handleExport}
                className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all transform hover:-translate-y-0.5"
              >
                <Download className="w-4 h-4" />
                <span>Download Predictions CSV</span>
              </button>

              <button
                onClick={handleClear}
                className="inline-flex items-center space-x-1.5 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Analyze New File</span>
              </button>
            </div>
          </div>

          {/* Summary Metric Cards */}
          <SummaryCards
            summary={predictionResponse.summary}
            predictions={predictionResponse.predictions}
          />

          {/* Interactive Searchable/Paginated Table */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-slate-100">Batch Prediction Telemetry Table</h2>
              <span className="text-xs font-mono text-slate-400">
                Showing predictions for {predictionResponse.predictions.length.toLocaleString()} rows
              </span>
            </div>

            <PredictionTable predictions={predictionResponse.predictions} />
          </div>
        </div>
      )}
    </div>
  );
};
