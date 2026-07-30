import React from 'react';
import { useHealth } from '../hooks/useHealth';
import { CheckCircle2, AlertCircle, RefreshCw, Server, Clock, ShieldAlert } from 'lucide-react';

export const HealthStatus: React.FC = () => {
  const { health, loading, error, latencyMs, refetch } = useHealth();

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800 relative overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-slate-800 text-cyan-400">
            <Server className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-100">FastAPI Backend Status</h3>
            <p className="text-xs text-slate-400 font-mono">GET /api/v1/health</p>
          </div>
        </div>

        <button
          onClick={() => refetch()}
          disabled={loading}
          className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition-colors disabled:opacity-50"
          title="Refresh Health Status"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
        </button>
      </div>

      {loading && !health ? (
        <div className="py-6 text-center text-slate-400 text-sm animate-pulse">
          Connecting to backend server at http://localhost:8000/api/v1...
        </div>
      ) : error ? (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Backend Connection Error</span>
            <span className="text-xs text-rose-400/90">{error}</span>
            <p className="text-[11px] mt-2 text-slate-400 font-mono">
              Ensure FastAPI backend is running: <code className="text-cyan-400">uvicorn app.main:app --reload</code>
            </p>
          </div>
        </div>
      ) : health ? (
        <div className="space-y-3 text-xs">
          <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
            <span className="text-slate-400 flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Operational State</span>
            </span>
            <span className="font-mono font-medium text-emerald-400 uppercase tracking-wider bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
              {health.status}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800/80">
              <span className="text-slate-500 block text-[10px] uppercase font-mono">Service Name</span>
              <span className="text-slate-200 font-medium text-xs truncate block">{health.service}</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800/80">
              <span className="text-slate-500 block text-[10px] uppercase font-mono">API Version</span>
              <span className="text-cyan-400 font-mono font-medium text-xs">v{health.version}</span>
            </div>
          </div>

          <div className="flex items-center justify-between pt-2 text-[11px] text-slate-400 border-t border-slate-800/60 font-mono">
            <span className="flex items-center space-x-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-500" />
              <span>Latency:</span>
              <span className="text-emerald-400">{latencyMs ?? 0} ms</span>
            </span>
            <span className="text-slate-500">
              {new Date(health.timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
      ) : null}
    </div>
  );
};
