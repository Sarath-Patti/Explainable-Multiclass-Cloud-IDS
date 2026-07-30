import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ShieldCheck, LayoutDashboard, UploadCloud, Home, Activity } from 'lucide-react';
import { useHealth } from '../../hooks/useHealth';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const { health, loading, error } = useHealth();

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 shadow-lg shadow-cyan-500/20 group-hover:scale-105 transition-transform duration-200">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                XAI Cloud-IDS
              </span>
              <span className="block text-[10px] uppercase font-mono text-cyan-400 tracking-wider">
                Multiclass Security
              </span>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            <Link
              to="/"
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive('/')
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Home className="w-4 h-4" />
              <span>Home</span>
            </Link>

            <Link
              to="/dashboard"
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive('/dashboard')
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Dashboard</span>
            </Link>

            <Link
              to="/upload"
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive('/upload')
                  ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <UploadCloud className="w-4 h-4" />
              <span>Predict</span>
            </Link>
          </nav>

          {/* API Health Status Pill */}
          <div className="hidden md:flex items-center">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-xs font-mono">
              <Activity className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">API:</span>
              {loading ? (
                <span className="text-yellow-400 animate-pulse">Connecting...</span>
              ) : health && !error ? (
                <span className="flex items-center space-x-1.5 text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping inline-block"></span>
                  <span>Online (v{health.version})</span>
                </span>
              ) : (
                <span className="text-rose-400">Offline</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
