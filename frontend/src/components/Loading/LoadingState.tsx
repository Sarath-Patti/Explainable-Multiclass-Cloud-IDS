import React from 'react';
import { Cpu, RefreshCw } from 'lucide-react';

interface LoadingStateProps {
  uploadProgress: number;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ uploadProgress }) => {
  return (
    <div className="glass-card p-12 rounded-3xl border border-slate-800 text-center space-y-6 max-w-lg mx-auto my-8">
      <div className="relative w-20 h-20 mx-auto">
        <div className="absolute inset-0 rounded-full border-4 border-cyan-500/20 animate-ping"></div>
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-cyan-500/20 to-indigo-500/20 flex items-center justify-center text-cyan-400 border border-cyan-500/30">
          <Cpu className="w-10 h-10 animate-pulse" />
        </div>
      </div>

      <div className="space-y-2">
        <h3 className="text-lg font-bold text-white">Evaluating Multiclass Intrusion Detection</h3>
        <p className="text-xs text-slate-400 max-w-xs mx-auto">
          Executing Top-14 XGBoost model inference on uploaded network flow records...
        </p>
      </div>

      {uploadProgress > 0 && uploadProgress < 100 ? (
        <div className="space-y-2 max-w-xs mx-auto">
          <div className="flex justify-between text-xs font-mono text-slate-400">
            <span>Uploading Dataset</span>
            <span className="text-cyan-400">{uploadProgress}%</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-cyan-500 to-indigo-500 h-2 transition-all duration-300 rounded-full"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        </div>
      ) : (
        <div className="inline-flex items-center space-x-2 text-xs font-mono text-cyan-400 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20">
          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
          <span>Computing predict_proba() confidence scores...</span>
        </div>
      )}
    </div>
  );
};
