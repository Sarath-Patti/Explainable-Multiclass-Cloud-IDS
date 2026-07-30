import React, { useState } from 'react';
import { UploadCloud, FileSpreadsheet, CheckCircle2, AlertCircle, HelpCircle, ArrowRight } from 'lucide-react';

export const Upload: React.FC = () => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-6">
      {/* Page Title */}
      <div className="border-b border-slate-800 pb-6 space-y-2">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center space-x-3">
          <UploadCloud className="w-7 h-7 text-cyan-400" />
          <span>Upload Network Flow CSV (Placeholder)</span>
        </h1>
        <p className="text-xs text-slate-400">
          Upload preprocessed network traffic CSV files for real-time multiclass threat classification and SHAP feature attribution explanations.
        </p>
      </div>

      {/* Upload Drag & Drop Area */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`glass-card p-10 rounded-3xl border-2 border-dashed transition-all text-center space-y-4 ${
          dragActive
            ? 'border-cyan-400 bg-cyan-500/10'
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
          <div className="space-y-2">
            <h3 className="text-base font-semibold text-emerald-400 flex items-center justify-center space-x-2">
              <CheckCircle2 className="w-5 h-5" />
              <span>{selectedFile.name}</span>
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Selected CSV File
            </p>
            <button
              onClick={() => setSelectedFile(null)}
              className="text-xs text-rose-400 underline hover:text-rose-300 pt-2 block mx-auto"
            >
              Remove file
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            <h3 className="text-base font-semibold text-slate-200">
              Drag and drop your network flow CSV file here
            </h3>
            <p className="text-xs text-slate-400">
              Supports CICIDS2017 flow feature schema (.csv format up to 50MB)
            </p>

            <div className="pt-4">
              <label className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs border border-slate-700 cursor-pointer transition-colors inline-block">
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

      {/* Action Button Placeholder */}
      <div className="flex justify-end">
        <button
          disabled={!selectedFile}
          className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-cyan-500/20"
        >
          <span>Run Multiclass Inference (v1.1)</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Schema Guidance Card */}
      <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
        <h4 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
          <HelpCircle className="w-4 h-4 text-cyan-400" />
          <span>Expected CSV Schema Requirements</span>
        </h4>
        <p className="text-xs text-slate-400 leading-relaxed">
          The inference pipeline expects standard 70 flow parameters or the optimal Top-14 SHAP feature columns including <code className="text-cyan-400">Destination Port</code>, <code className="text-cyan-400">Init_Win_bytes_forward</code>, <code className="text-cyan-400">min_seg_size_forward</code>, <code className="text-cyan-400">Bwd Packet Length Std</code>, and <code className="text-cyan-400">Flow IAT Min</code>.
        </p>
      </div>
    </div>
  );
};
