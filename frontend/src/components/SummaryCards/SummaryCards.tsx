import React from 'react';
import { PredictionSummary, PredictionItem } from '../../types/api';
import { ShieldCheck, ShieldAlert, Activity, AlertTriangle } from 'lucide-react';

interface SummaryCardsProps {
  summary: PredictionSummary;
  predictions: PredictionItem[];
}

export const SummaryCards: React.FC<SummaryCardsProps> = ({ summary, predictions }) => {
  const benignPct = summary.total_samples > 0
    ? ((summary.predicted_benign / summary.total_samples) * 100).toFixed(1)
    : '0.0';

  const attackPct = summary.total_samples > 0
    ? ((summary.predicted_attacks / summary.total_samples) * 100).toFixed(1)
    : '0.0';

  // Count unique attack types detected (excluding BENIGN)
  const attackTypes = new Set(
    predictions
      .map((item) => item.prediction)
      .filter((label): label is string => typeof label === 'string' && label.toUpperCase() !== 'BENIGN')
  );

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total Flows Evaluated */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-xs font-medium">Total Evaluated Flows</span>
          <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
            <Activity className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-bold text-white font-mono">
            {summary.total_samples.toLocaleString()}
          </span>
          <span className="text-xs font-mono text-cyan-400">100.0%</span>
        </div>
        <p className="text-[11px] text-slate-500">Processed batch records</p>
      </div>

      {/* Benign Traffic */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-xs font-medium">Benign Traffic</span>
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
            <ShieldCheck className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-bold text-emerald-400 font-mono">
            {summary.predicted_benign.toLocaleString()}
          </span>
          <span className="text-xs font-mono text-emerald-400">{benignPct}%</span>
        </div>
        <p className="text-[11px] text-slate-500">Normal non-malicious flows</p>
      </div>

      {/* Malicious Attack Traffic */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-xs font-medium">Malicious Attacks</span>
          <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400">
            <ShieldAlert className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-bold text-rose-400 font-mono">
            {summary.predicted_attacks.toLocaleString()}
          </span>
          <span className="text-xs font-mono text-rose-400">{attackPct}%</span>
        </div>
        <p className="text-[11px] text-slate-500">Alert triggers & threats</p>
      </div>

      {/* Attack Categories Detected */}
      <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-xs font-medium">Attack Categories</span>
          <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
            <AlertTriangle className="w-4 h-4" />
          </div>
        </div>
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-bold text-amber-400 font-mono">
            {attackTypes.size}
          </span>
          <span className="text-xs font-mono text-amber-400">Classes</span>
        </div>
        <p className="text-[11px] text-slate-500">Distinct attack vectors</p>
      </div>
    </div>
  );
};
