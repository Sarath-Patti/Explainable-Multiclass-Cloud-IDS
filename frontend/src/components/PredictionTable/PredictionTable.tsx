import React, { useState, useMemo } from 'react';
import { PredictionItem } from '../../types/api';
import { Search, ChevronLeft, ChevronRight, ArrowUpDown, ShieldCheck, ShieldAlert, Filter } from 'lucide-react';

interface PredictionTableProps {
  predictions: PredictionItem[];
}

type SortField = 'row' | 'prediction' | 'confidence';
type SortDirection = 'asc' | 'desc';
type FilterCategory = 'all' | 'attacks' | 'benign';

export const PredictionTable: React.FC<PredictionTableProps> = ({ predictions }) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<FilterCategory>('all');
  const [sortField, setSortField] = useState<SortField>('row');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(10);

  // Handle column header sorting toggle
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Filtered & Sorted predictions dataset
  const processedData = useMemo(() => {
    return predictions
      .filter((item) => {
        // Category Filter
        const isBenign = item.prediction.toUpperCase() === 'BENIGN';
        if (categoryFilter === 'attacks' && isBenign) return false;
        if (categoryFilter === 'benign' && !isBenign) return false;

        // Search Term Filter
        if (!searchTerm.trim()) return true;
        const term = searchTerm.toLowerCase();
        return (
          item.row.toString().includes(term) ||
          item.prediction.toLowerCase().includes(term)
        );
      })
      .sort((a, b) => {
        let valA: string | number = a[sortField];
        let valB: string | number = b[sortField];

        if (typeof valA === 'string') {
          valA = valA.toLowerCase();
          valB = (valB as string).toLowerCase();
        }

        if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
        if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
  }, [predictions, searchTerm, categoryFilter, sortField, sortDirection]);

  // Pagination calculations
  const totalPages = Math.max(1, Math.ceil(processedData.length / pageSize));
  const validCurrentPage = Math.min(currentPage, totalPages);
  const startIndex = (validCurrentPage - 1) * pageSize;
  const paginatedData = processedData.slice(startIndex, startIndex + pageSize);

  return (
    <div className="glass-card rounded-2xl border border-slate-800 space-y-4 p-6">
      {/* Controls Bar: Search, Category Filter, Page Size */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Search Input */}
        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by class or row #..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full bg-slate-900/80 border border-slate-800 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>

        {/* Category Filter Pills & Page Size Selector */}
        <div className="flex items-center space-x-3 w-full sm:w-auto justify-between sm:justify-end">
          <div className="inline-flex p-1 rounded-xl bg-slate-900/80 border border-slate-800 text-xs">
            <button
              onClick={() => {
                setCategoryFilter('all');
                setCurrentPage(1);
              }}
              className={`px-3 py-1 rounded-lg transition-colors font-medium ${
                categoryFilter === 'all'
                  ? 'bg-slate-800 text-cyan-400 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All ({predictions.length})
            </button>
            <button
              onClick={() => {
                setCategoryFilter('attacks');
                setCurrentPage(1);
              }}
              className={`px-3 py-1 rounded-lg transition-colors font-medium ${
                categoryFilter === 'attacks'
                  ? 'bg-rose-500/20 text-rose-400 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Attacks Only
            </button>
            <button
              onClick={() => {
                setCategoryFilter('benign');
                setCurrentPage(1);
              }}
              className={`px-3 py-1 rounded-lg transition-colors font-medium ${
                categoryFilter === 'benign'
                  ? 'bg-emerald-500/20 text-emerald-400 font-semibold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Benign Only
            </button>
          </div>

          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500"
          >
            <option value={10}>10 / page</option>
            <option value={25}>25 / page</option>
            <option value={50}>50 / page</option>
            <option value={100}>100 / page</option>
          </select>
        </div>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto border border-slate-800/80 rounded-xl">
        <table className="w-full text-left text-xs font-mono border-collapse">
          <thead>
            <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
              <th
                onClick={() => handleSort('row')}
                className="py-3 px-4 cursor-pointer hover:text-slate-200 transition-colors select-none"
              >
                <div className="flex items-center space-x-1.5">
                  <span>Row #</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-500" />
                </div>
              </th>

              <th
                onClick={() => handleSort('prediction')}
                className="py-3 px-4 cursor-pointer hover:text-slate-200 transition-colors select-none"
              >
                <div className="flex items-center space-x-1.5">
                  <span>Predicted Class</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-500" />
                </div>
              </th>

              <th
                onClick={() => handleSort('confidence')}
                className="py-3 px-4 cursor-pointer hover:text-slate-200 transition-colors select-none"
              >
                <div className="flex items-center space-x-1.5">
                  <span>Confidence Score</span>
                  <ArrowUpDown className="w-3 h-3 text-slate-500" />
                </div>
              </th>

              <th className="py-3 px-4 select-none">Risk Level</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-slate-300">
            {paginatedData.length === 0 ? (
              <tr>
                <td colSpan={4} className="py-8 text-center text-slate-500 text-sm">
                  No prediction records match your search or filter criteria.
                </td>
              </tr>
            ) : (
              paginatedData.map((item) => {
                const isBenign = item.prediction.toUpperCase() === 'BENIGN';
                const confidencePct = (item.confidence * 100).toFixed(2);

                return (
                  <tr key={item.row} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 text-cyan-400 font-bold">#{item.row}</td>

                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        {isBenign ? (
                          <ShieldCheck className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                        ) : (
                          <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0" />
                        )}
                        <span className={`font-semibold ${isBenign ? 'text-slate-100' : 'text-rose-300'}`}>
                          {item.prediction}
                        </span>
                      </div>
                    </td>

                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-3 max-w-[180px]">
                        <span className="w-14 font-mono font-medium text-slate-200">{confidencePct}%</span>
                        <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-1.5 rounded-full ${
                              isBenign ? 'bg-emerald-400' : 'bg-gradient-to-r from-amber-500 to-rose-500'
                            }`}
                            style={{ width: `${confidencePct}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>

                    <td className="py-3 px-4">
                      {isBenign ? (
                        <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-semibold tracking-wide uppercase">
                          Safe
                        </span>
                      ) : (
                        <span className="px-2.5 py-1 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-[10px] font-semibold tracking-wide uppercase">
                          Critical Alert
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2 text-xs text-slate-400 font-mono">
        <div>
          Showing <span className="text-slate-200">{startIndex + 1}</span> to{' '}
          <span className="text-slate-200">{Math.min(startIndex + pageSize, processedData.length)}</span> of{' '}
          <span className="text-cyan-400">{processedData.length}</span> matching entries
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
            disabled={validCurrentPage === 1}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <span className="px-3 py-1 rounded-xl bg-slate-900 border border-slate-800 text-slate-300">
            Page {validCurrentPage} of {totalPages}
          </span>

          <button
            onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))}
            disabled={validCurrentPage >= totalPages}
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
