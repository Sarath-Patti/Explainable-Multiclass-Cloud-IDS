import React, { useState } from 'react';
import { UploadCloud, FileSpreadsheet, CheckCircle2, AlertCircle, ArrowRight, X, RotateCcw } from 'lucide-react';

interface UploadCardProps {
  onFileSelect: (file: File) => void;
  onSubmit: (file: File) => void;
  onClear: () => void;
  selectedFile: File | null;
  isLoading: boolean;
  uploadProgress: number;
}

export const UploadCard: React.FC<UploadCardProps> = ({
  onFileSelect,
  onSubmit,
  onClear,
  selectedFile,
  isLoading,
  uploadProgress,
}) => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validateFile = (file: File): boolean => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setValidationError('Invalid file format. Only CSV files (.csv) are supported.');
      return false;
    }
    if (file.size === 0) {
      setValidationError('Uploaded CSV file is empty (0 bytes).');
      return false;
    }
    if (file.size > 50 * 1024 * 1024) {
      setValidationError('File size exceeds the 50 MB limit.');
      return false;
    }
    setValidationError(null);
    return true;
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        onFileSelect(file);
      }
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        onFileSelect(file);
      }
    }
  };

  return (
    <div className="space-y-4">
      {/* Drag & Drop Area */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`glass-card p-8 sm:p-10 rounded-3xl border-2 border-dashed transition-all text-center space-y-4 relative ${
          dragActive
            ? 'border-cyan-400 bg-cyan-500/10 shadow-lg shadow-cyan-500/10'
            : selectedFile
            ? 'border-emerald-500/50 bg-emerald-500/5'
            : 'border-slate-800 hover:border-slate-700'
        }`}
      >
        <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-tr from-cyan-500/20 to-indigo-500/20 flex items-center justify-center text-cyan-400">
          {selectedFile ? (
            <FileSpreadsheet className="w-8 h-8 text-emerald-400" />
          ) : (
            <UploadCloud className="w-8 h-8" />
          )}
        </div>

        {selectedFile ? (
          <div className="space-y-3 max-w-md mx-auto">
            <div className="flex items-center justify-center space-x-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
              <h3 className="text-base font-semibold text-emerald-400 truncate">
                {selectedFile.name}
              </h3>
            </div>

            <p className="text-xs text-slate-400 font-mono">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Network Flow Dataset CSV
            </p>

            {/* Upload Progress Bar when submitting */}
            {isLoading && uploadProgress > 0 && uploadProgress < 100 && (
              <div className="space-y-1.5 pt-2">
                <div className="flex justify-between text-[11px] font-mono text-slate-400">
                  <span>Uploading CSV...</span>
                  <span className="text-cyan-400">{uploadProgress}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-cyan-500 to-indigo-500 h-2 transition-all duration-300 rounded-full"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
              </div>
            )}

            <button
              onClick={() => {
                setValidationError(null);
                onClear();
              }}
              disabled={isLoading}
              className="inline-flex items-center space-x-1 text-xs text-slate-400 hover:text-rose-400 transition-colors pt-2 disabled:opacity-50"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Select a different file</span>
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <h3 className="text-base font-semibold text-slate-200">
              Drag and drop your network flow CSV file here
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Upload preprocessed flow records (.csv up to 50 MB) containing the Top-14 network features.
            </p>

            <div className="pt-2">
              <label className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs border border-slate-700 cursor-pointer transition-colors inline-block shadow-sm">
                Browse Files
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleChange}
                  className="hidden"
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Validation Error Message */}
      {validationError && (
        <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{validationError}</span>
          </div>
          <button onClick={() => setValidationError(null)} className="text-rose-400 hover:text-rose-300">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Submit Action Button */}
      {selectedFile && !isLoading && (
        <div className="flex justify-end pt-2">
          <button
            onClick={() => selectedFile && onSubmit(selectedFile)}
            className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-bold text-sm shadow-lg shadow-cyan-500/25 transition-all transform hover:-translate-y-0.5"
          >
            <span>Run Multiclass Inference</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};
