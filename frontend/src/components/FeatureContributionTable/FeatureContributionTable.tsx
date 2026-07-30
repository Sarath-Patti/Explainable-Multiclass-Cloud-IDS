import React from 'react';
import { FeatureContribution } from '../../types/api';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface FeatureContributionTableProps {
  features: FeatureContribution[];
}

export const FeatureContributionTable: React.FC<FeatureContributionTableProps> = ({ features }) => {
  const positiveFeatures = features.filter((f) => f.shap_value >= 0);
  const negativeFeatures = features.filter((f) => f.shap_value < 0);

  return (
    <div className="space-y-6">
      {/* Top Positive Contributions */}
      <div className="space-y-2">
        <div className="flex items-center space-x-2 text-xs font-bold text-emerald-400">
          <TrendingUp className="w-4 h-4" />
          <span>Top Positive Attributions (Increases Class Confidence)</span>
        </div>

        <div className="overflow-x-auto border border-slate-800 rounded-xl">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                <th className="py-2.5 px-3">Feature Name</th>
                <th className="py-2.5 px-3">Observed Value</th>
                <th className="py-2.5 px-3">SHAP Value</th>
                <th className="py-2.5 px-3">Impact Direction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {positiveFeatures.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-4 text-center text-slate-500 text-xs">
                    No positive feature attributions for this prediction.
                  </td>
                </tr>
              ) : (
                positiveFeatures.slice(0, 5).map((item) => (
                  <tr key={item.feature} className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 font-semibold text-slate-100">{item.feature}</td>
                    <td className="py-2.5 px-3 text-slate-300">{item.value.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">+{item.shap_value.toFixed(4)}</td>
                    <td className="py-2.5 px-3">
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] font-semibold uppercase">
                        + Positive Push
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Top Negative Contributions */}
      <div className="space-y-2">
        <div className="flex items-center space-x-2 text-xs font-bold text-rose-400">
          <TrendingDown className="w-4 h-4" />
          <span>Top Negative Attributions (Decreases Class Confidence)</span>
        </div>

        <div className="overflow-x-auto border border-slate-800 rounded-xl">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                <th className="py-2.5 px-3">Feature Name</th>
                <th className="py-2.5 px-3">Observed Value</th>
                <th className="py-2.5 px-3">SHAP Value</th>
                <th className="py-2.5 px-3">Impact Direction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {negativeFeatures.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-4 text-center text-slate-500 text-xs">
                    No negative feature attributions for this prediction.
                  </td>
                </tr>
              ) : (
                negativeFeatures.slice(0, 5).map((item) => (
                  <tr key={item.feature} className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 font-semibold text-slate-100">{item.feature}</td>
                    <td className="py-2.5 px-3 text-slate-300">{item.value.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-rose-400 font-bold">{item.shap_value.toFixed(4)}</td>
                    <td className="py-2.5 px-3">
                      <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[10px] font-semibold uppercase">
                        - Negative Push
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
