import React from 'react';
import { ShieldCheck, GitBranch, Cpu } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-[#070a11] text-slate-400 text-xs py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Left info */}
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            <span className="font-semibold text-slate-200">Explainable Multiclass Cloud IDS</span>
            <span className="text-slate-600">|</span>
            <span>SHAP & XGBoost/Random Forest Security Telemetry</span>
          </div>

          {/* Center Metadata */}
          <div className="flex items-center space-x-6 text-slate-500 font-mono">
            <div className="flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>CICIDS2017 Preprocessed</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <GitBranch className="w-3.5 h-3.5 text-cyan-400" />
              <span>v1.0 Release</span>
            </div>
          </div>

          {/* Right copyright */}
          <div>
            &copy; {new Date().getFullYear()} Explainable Cloud IDS Framework. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
};
