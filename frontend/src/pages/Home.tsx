import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck, Cpu, Eye, ArrowRight, Zap, Database, BarChart3, Lock } from 'lucide-react';
import { HealthStatus } from '../components/HealthStatus';

export const Home: React.FC = () => {
  return (
    <div className="space-y-12 py-6">
      {/* Hero Section */}
      <section className="relative overflow-hidden glass-card rounded-3xl p-8 sm:p-12 border border-slate-800">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute bottom-0 left-0 -ml-16 -mb-16 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-mono">
            <Zap className="w-3.5 h-3.5" />
            <span>Milestone v1.0 • React + FastAPI Foundation</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Explainable Multiclass <br />
            <span className="bg-gradient-to-r from-cyan-400 via-indigo-400 to-emerald-400 bg-clip-text text-transparent">
              Cloud Intrusion Detection
            </span>
          </h1>

          <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
            A high-performance machine learning framework for detecting, classifying, and explaining malicious cloud network flows using SHAP feature attribution and optimal tree-based models.
          </p>

          <div className="flex flex-wrap gap-4 pt-2">
            <Link
              to="/dashboard"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-semibold text-sm shadow-lg shadow-cyan-500/25 transition-all transform hover:-translate-y-0.5"
            >
              <span>Explore Dashboard</span>
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              to="/upload"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm border border-slate-700 transition-all"
            >
              <Database className="w-4 h-4 text-cyan-400" />
              <span>Upload CSV Flow</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Main Grid: Features & System Health */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column (2 cols): System Architecture Features */}
        <div className="lg:col-span-2 space-y-6">
          <h2 className="text-xl font-bold text-slate-100 flex items-center space-x-2">
            <ShieldCheck className="w-5 h-5 text-cyan-400" />
            <span>Framework Core Capabilities</span>
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
              <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-400 w-fit">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-slate-200">Multiclass Classifier</h3>
              <p className="text-slate-400 text-xs leading-relaxed">
                Discriminates benign traffic from DDoS, Botnet, Web Attacks, and PortScan across 1.76M preprocessed CICIDS2017 flow samples.
              </p>
            </div>

            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
              <div className="p-3 rounded-xl bg-indigo-500/10 text-indigo-400 w-fit">
                <Eye className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-slate-200">SHAP Explainability (XAI)</h3>
              <p className="text-slate-400 text-xs leading-relaxed">
                Provides global importance rankings and local waterfall/decision plots explaining exact flow parameter feature attributions.
              </p>
            </div>

            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
              <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400 w-fit">
                <BarChart3 className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-slate-200">Feature Optimization</h3>
              <p className="text-slate-400 text-xs leading-relaxed">
                Prunes feature dimensionality from 70 to 14 optimal features while preserving &ge;99% of baseline Macro F1 detection score.
              </p>
            </div>

            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-3">
              <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400 w-fit">
                <Lock className="w-6 h-6" />
              </div>
              <h3 className="text-base font-semibold text-slate-200">Production REST API</h3>
              <p className="text-slate-400 text-xs leading-relaxed">
                Decoupled FastAPI backend exposing versioned `/api/v1` health check, prediction, and SHAP report telemetry.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column (1 col): System Health & API Monitor */}
        <div className="space-y-6">
          <h2 className="text-xl font-bold text-slate-100 flex items-center space-x-2">
            <Zap className="w-5 h-5 text-emerald-400" />
            <span>Live System Telemetry</span>
          </h2>

          <HealthStatus />
        </div>
      </div>
    </div>
  );
};
