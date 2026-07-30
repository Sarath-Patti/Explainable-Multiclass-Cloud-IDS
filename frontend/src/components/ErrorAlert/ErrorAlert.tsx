import React from 'react';
import { AlertTriangle, X, RefreshCcw, FileWarning } from 'lucide-react';
import { ApiError } from '../../types/api';

interface ErrorAlertProps {
  error: ApiError;
  onRetry?: () => void;
  onClose?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({ error, onRetry, onClose }) => {
  return (
    <div className="glass-card p-6 rounded-2xl border border-rose-500/30 bg-rose-500/5 space-y-4 my-4 max-w-3xl mx-auto shadow-lg shadow-rose-500/5">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <div className="p-2 rounded-xl bg-rose-500/20 text-rose-400 mt-0.5">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-rose-300">Prediction Processing Failed</h3>
            <p className="text-xs text-rose-200/90 leading-relaxed">{error.message}</p>
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Render list of missing features if present */}
      {error.missing_features && error.missing_features.length > 0 && (
        <div className="p-4 rounded-xl bg-slate-950/80 border border-rose-500/20 space-y-2">
          <div className="flex items-center space-x-2 text-xs font-semibold text-rose-400">
            <FileWarning className="w-4 h-4" />
            <span>Missing Required Features ({error.missing_features.length})</span>
          </div>
          <p className="text-[11px] text-slate-400">
            The uploaded CSV must contain all Top-14 features used during model training. Please ensure the following columns exist:
          </p>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {error.missing_features.map((feat) => (
              <code
                key={feat}
                className="px-2 py-0.5 rounded bg-rose-500/10 border border-rose-500/20 text-rose-300 text-[11px] font-mono"
              >
                {feat}
              </code>
            ))}
          </div>
        </div>
      )}

      {onRetry && (
        <div className="flex justify-end pt-1">
          <button
            onClick={onRetry}
            className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition-colors"
          >
            <RefreshCcw className="w-3.5 h-3.5" />
            <span>Try Again</span>
          </button>
        </div>
      )}
    </div>
  );
};
