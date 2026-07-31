import React from 'react';
import { BarChart2, Layers, Activity } from 'lucide-react';
import { HealthStatus } from '../components/HealthStatus';

export const Dashboard: React.FC = () => {
  return (
    <div className="space-y-8 py-6">
      {/* Header Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center space-x-3">
            <BarChart2 className="w-7 h-7 text-cyan-400" />
            <span>Security Operations Dashboard</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Model Evaluation Metrics • Optimal SHAP Features • Baseline Performance Overview
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs font-mono text-cyan-400">
            Dataset: CICIDS2017 Clean
          </span>
          <span className="px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-xs font-mono text-emerald-400">
            Model: XGBoost Selected (Top-14)
          </span>
        </div>
      </div>

      {/* Metric Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-slate-400 text-xs font-medium block">Baseline Macro F1</span>
          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-bold text-white font-mono">0.9984</span>
            <span className="text-xs font-mono text-emerald-400">Random Forest & XGBoost</span>
          </div>
          <p className="text-[11px] text-slate-500">Evaluated on 378k test network flows</p>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-slate-400 text-xs font-medium block">Optimal Feature Space</span>
          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-bold text-cyan-400 font-mono">14 Features</span>
            <span className="text-xs font-mono text-cyan-400">-80.0% Reduction</span>
          </div>
          <p className="text-[11px] text-slate-500">Selected via SHAP Pareto threshold</p>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-slate-400 text-xs font-medium block">F1 Score Retention</span>
          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-bold text-emerald-400 font-mono">99.8%</span>
            <span className="text-xs font-mono text-emerald-400">F1 Loss &lt; 0.2%</span>
          </div>
          <p className="text-[11px] text-slate-500">Preserved baseline detection rates</p>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-slate-400 text-xs font-medium block">Test Throughput</span>
          <div className="flex items-baseline justify-between">
            <span className="text-2xl font-bold text-indigo-400 font-mono">120k+</span>
            <span className="text-xs font-mono text-indigo-400">flows/sec</span>
          </div>
          <p className="text-[11px] text-slate-500">Multi-run benchmarked inference</p>
        </div>
      </div>

      {/* Main Grid: SHAP Feature Teaser & System Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left 2 Cols: SHAP Feature Importance Table Teaser */}
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-card rounded-2xl p-6 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-slate-100 flex items-center space-x-2">
                <Layers className="w-5 h-5 text-cyan-400" />
                <span>Top SHAP-Guided Selected Features (Top-14)</span>
              </h3>
              <span className="text-xs font-mono text-slate-400">Ranked by Mean |SHAP|</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2.5 px-3">Rank</th>
                    <th className="py-2.5 px-3">Feature Name</th>
                    <th className="py-2.5 px-3">Category</th>
                    <th className="py-2.5 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  <tr className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">#1</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-100">Destination Port</td>
                    <td className="py-2.5 px-3 text-slate-400">Header Parameter</td>
                    <td className="py-2.5 px-3"><span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">Selected</span></td>
                  </tr>
                  <tr className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">#2</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-100">Init_Win_bytes_forward</td>
                    <td className="py-2.5 px-3 text-slate-400">TCP Window Size</td>
                    <td className="py-2.5 px-3"><span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">Selected</span></td>
                  </tr>
                  <tr className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">#3</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-100">min_seg_size_forward</td>
                    <td className="py-2.5 px-3 text-slate-400">Header Size</td>
                    <td className="py-2.5 px-3"><span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">Selected</span></td>
                  </tr>
                  <tr className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">#4</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-100">Bwd Packet Length Std</td>
                    <td className="py-2.5 px-3 text-slate-400">Packet Distribution</td>
                    <td className="py-2.5 px-3"><span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">Selected</span></td>
                  </tr>
                  <tr className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">#5</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-100">Flow IAT Min</td>
                    <td className="py-2.5 px-3 text-slate-400">Inter-Arrival Time</td>
                    <td className="py-2.5 px-3"><span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">Selected</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="pt-2 text-right">
              <span className="text-[11px] text-slate-400 italic">
                + 9 additional optimal features saved in recommended_features.csv
              </span>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Telemetry */}
        <div className="space-y-6">
          <h3 className="text-lg font-bold text-slate-100 flex items-center space-x-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            <span>API & Runtime Health</span>
          </h3>

          <HealthStatus />
        </div>
      </div>
    </div>
  );
};
