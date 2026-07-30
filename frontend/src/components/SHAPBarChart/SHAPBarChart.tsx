import React from 'react';
import { FeatureContribution } from '../../types/api';

interface SHAPBarChartProps {
  features: FeatureContribution[];
  maxDisplay?: number;
}

export const SHAPBarChart: React.FC<SHAPBarChartProps> = ({ features, maxDisplay = 10 }) => {
  const displayFeatures = features.slice(0, maxDisplay);

  // Determine maximum absolute SHAP magnitude for bar scaling
  const maxAbsShap = Math.max(
    0.001,
    ...displayFeatures.map((f) => Math.abs(f.shap_value))
  );

  return (
    <div className="space-y-3 font-mono text-xs">
      <div className="flex items-center justify-between text-[11px] text-slate-400 font-semibold border-b border-slate-800 pb-2">
        <span>Feature & Observed Value</span>
        <span>SHAP Attribution (|f(x)|)</span>
      </div>

      <div className="space-y-2.5">
        {displayFeatures.map((item) => {
          const isPositive = item.shap_value >= 0;
          const absVal = Math.abs(item.shap_value);
          const barWidthPercent = Math.min(100, Math.max(4, (absVal / maxAbsShap) * 100));

          return (
            <div key={item.feature} className="space-y-1 group">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-200 font-semibold truncate max-w-[200px]" title={item.feature}>
                  {item.feature}
                </span>
                <div className="flex items-center space-x-2">
                  <span className="text-[11px] text-slate-400">
                    x = <span className="text-slate-300 font-mono">{item.value.toLocaleString()}</span>
                  </span>
                  <span
                    className={`font-bold font-mono px-1.5 py-0.5 rounded text-[11px] ${
                      isPositive
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                    }`}
                  >
                    {isPositive ? `+${item.shap_value.toFixed(4)}` : item.shap_value.toFixed(4)}
                  </span>
                </div>
              </div>

              {/* Horizontal Contribution Bar */}
              <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden flex items-center">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    isPositive
                      ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                      : 'bg-gradient-to-r from-rose-500 to-amber-500'
                  }`}
                  style={{ width: `${barWidthPercent}%` }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
