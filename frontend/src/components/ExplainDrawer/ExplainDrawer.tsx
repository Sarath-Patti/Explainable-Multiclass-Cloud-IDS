import React, { useEffect, useState } from 'react';
import { ExplainResponse, PredictionItem } from '../../types/api';
import { explainPrediction } from '../../services/api';
import { SHAPBarChart } from '../SHAPBarChart/SHAPBarChart';
import { FeatureContributionTable } from '../FeatureContributionTable/FeatureContributionTable';
import { X, Sparkles, RefreshCw, AlertCircle, ShieldCheck, ShieldAlert, Cpu, Layers } from 'lucide-react';

interface ExplainDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  selectedRow: PredictionItem | null;
  featureData?: Record<string, number | string>;
}

export const ExplainDrawer: React.FC<ExplainDrawerProps> = ({
  isOpen,
  onClose,
  selectedRow,
  featureData,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);

  useEffect(() => {
    if (isOpen && selectedRow) {
      fetchExplanation();
    } else {
      setExplanation(null);
      setError(null);
    }
  }, [isOpen, selectedRow]);

  const fetchExplanation = async () => {
    if (!selectedRow) return;
    setLoading(true);
    setError(null);

    try {
      // Use provided featureData or fallback features dict
      const featuresPayload = featureData || {};
      const response = await explainPrediction({
        row: selectedRow.row,
        features: featuresPayload,
      });
      setExplanation(response);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to compute SHAP explanation from backend server.');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen || !selectedRow) return null;

  const isBenign = selectedRow.prediction.toUpperCase() === 'BENIGN';

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Dark Overlay Backdrop */}
      <div
        onClick={onClose}
        className="absolute inset-0 bg-black/70 backdrop-blur-sm transition-opacity animate-fadeIn"
      ></div>

      {/* Right-Side Drawer Panel */}
      <div className="fixed inset-y-0 right-0 max-w-2xl w-full bg-[#0d1322] border-l border-slate-800 shadow-2xl flex flex-col z-10">
        {/* Drawer Header */}
        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-cyan-500/20 to-indigo-500/20 text-cyan-400 border border-cyan-500/30">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-lg font-bold text-white">Row #{selectedRow.row} SHAP Explanation</h2>
                <span className="px-2.5 py-0.5 rounded-full bg-slate-800 text-xs font-mono text-cyan-400 border border-slate-700">
                  XGBoost Top-14
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Local feature attributions calculated using TreeExplainer
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Drawer Body Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Prediction Status Card */}
          <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {isBenign ? (
                  <ShieldCheck className="w-5 h-5 text-emerald-400" />
                ) : (
                  <ShieldAlert className="w-5 h-5 text-rose-400" />
                )}
                <span className="text-xs text-slate-400 uppercase tracking-wider font-mono">
                  Model Output Prediction
                </span>
              </div>
              <span className="text-xs font-mono text-slate-400">
                Confidence: <span className="text-emerald-400 font-bold">{(selectedRow.confidence * 100).toFixed(2)}%</span>
              </span>
            </div>

            <div className="flex items-center justify-between pt-1">
              <span
                className={`text-xl font-bold font-mono px-3 py-1 rounded-xl border ${
                  isBenign
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                    : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                }`}
              >
                {selectedRow.prediction}
              </span>

              {explanation && (
                <div className="text-right font-mono text-xs text-slate-400">
                  <span className="block text-[10px] uppercase text-slate-500">Base Expected Value</span>
                  <span className="text-cyan-400 font-bold">{explanation.base_value.toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>

          {/* Loading Indicator */}
          {loading && (
            <div className="py-16 text-center space-y-4">
              <div className="w-12 h-12 mx-auto rounded-xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center border border-cyan-500/20">
                <RefreshCw className="w-6 h-6 animate-spin" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-semibold text-slate-200">Computing SHAP Feature Attributions...</p>
                <p className="text-xs text-slate-400 font-mono">Executing TreeExplainer on Top-14 network features</p>
              </div>
            </div>
          )}

          {/* Error Indicator */}
          {error && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
              <div className="space-y-2">
                <span className="font-semibold block text-sm">Explanation Failed</span>
                <p className="text-rose-200/90">{error}</p>
                <button
                  onClick={fetchExplanation}
                  className="px-3 py-1 rounded-lg bg-rose-500/20 text-rose-300 border border-rose-500/30 text-[11px] font-semibold"
                >
                  Retry Calculation
                </button>
              </div>
            </div>
          )}

          {/* Explanation Data Views */}
          {!loading && !error && explanation && (
            <div className="space-y-6">
              {/* SHAP Contribution Bar Chart */}
              <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center space-x-2">
                    <Cpu className="w-4 h-4 text-cyan-400" />
                    <span>Top Attributed Features (|SHAP|)</span>
                  </h3>
                  <span className="text-[11px] font-mono text-slate-400">
                    Base: {explanation.base_value} &rarr; Output: {explanation.confidence}
                  </span>
                </div>

                <SHAPBarChart features={explanation.top_features} maxDisplay={10} />
              </div>

              {/* Feature Attribution Detail Tables */}
              <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-4">
                <div className="border-b border-slate-800/80 pb-3">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center space-x-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    <span>Feature Contribution Breakdown</span>
                  </h3>
                </div>

                <FeatureContributionTable features={explanation.top_features} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
