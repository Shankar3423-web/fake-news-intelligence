import React, { useState, useEffect } from 'react';
import { predictionService } from '../services/predictionService';
import { useToast } from '../hooks/useToast';
import { 
  IoSearchOutline, 
  IoTimeOutline, 
  IoClose, 
  IoGlobeOutline, 
  IoShieldCheckmarkOutline, 
  IoOpenOutline,
  IoPulseOutline
} from 'react-icons/io5';

export default function History() {
  const { addToast } = useToast();
  const [loading, setLoading] = useState(true);
  const [history, setHistory] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Modal details window state
  const [selectedItem, setSelectedItem] = useState(null);

  useEffect(() => {
    predictionService.getHistory()
      .then((data) => {
        setHistory(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load history:', err);
        addToast('Failed to fetch verification history.', 'error');
        setLoading(false);
      });
  }, []);

  const filteredHistory = history.filter((item) => {
    const titleMatch = item.title?.toLowerCase().includes(searchTerm.toLowerCase());
    const contentMatch = item.text_content?.toLowerCase().includes(searchTerm.toLowerCase());
    const labelMatch = item.predicted_label?.toLowerCase() === searchTerm.toLowerCase();
    return titleMatch || contentMatch || labelMatch;
  });

  const getVerdictStyle = (label) => {
    return label === 'REAL'
      ? 'bg-success-green/10 text-success-green border border-success-green/20'
      : 'bg-danger-red/10 text-danger-red border border-danger-red/20';
  };

  return (
    <div className="space-y-10 max-w-layout mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight text-text-main dark:text-dark-text">Verification History</h1>
        <p className="text-sm text-text-secondary dark:text-dark-text-secondary">
          Audit past consensus decisions and corresponding verification evidence logs
        </p>
      </div>

      {/* Actions & Filters */}
      <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="relative w-full md:max-w-md">
          <IoSearchOutline className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary dark:text-dark-text-secondary" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by article title, content, or label (REAL/FAKE)..."
            className="w-full pl-12 pr-4 py-3 bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-xl focus:outline-none focus:ring-1 focus:ring-primary-blue text-sm text-text-main dark:text-dark-text shadow-premium"
          />
        </div>

        <div className="flex items-center gap-1.5 text-xs text-text-secondary dark:text-dark-text-secondary bg-slate-100 dark:bg-slate-800 py-2 px-3 rounded-lg font-semibold">
          <IoTimeOutline className="w-4 h-4" />
          <span>{filteredHistory.length} Total Logs Cached</span>
        </div>
      </div>

      {/* History Table Card */}
      <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl shadow-premium overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-100 dark:border-dark-border/40 text-xs font-bold text-text-secondary dark:text-dark-text-secondary uppercase">
                <th className="py-4 px-6">Headline</th>
                <th className="py-4 px-6">Verdict</th>
                <th className="py-4 px-6">Confidence</th>
                <th className="py-4 px-6">Model</th>
                <th className="py-4 px-6">Verified Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-dark-border/30 text-sm">
              {loading ? (
                [...Array(5)].map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-3/4" /></td>
                    <td className="py-4 px-6"><div className="h-6 bg-slate-200 dark:bg-slate-850 rounded-full w-16" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-10" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-16" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-28" /></td>
                  </tr>
                ))
              ) : filteredHistory.length === 0 ? (
                <tr>
                  <td colSpan="5" className="py-12 text-center text-text-secondary dark:text-dark-text-secondary">
                    No matching records discovered. Run a new verification to create database logs.
                  </td>
                </tr>
              ) : (
                filteredHistory.map((item) => (
                  <tr 
                    key={item.id} 
                    onClick={() => setSelectedItem(item)}
                    className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30 transition-colors cursor-pointer"
                  >
                    <td className="py-4 px-6 font-semibold text-text-main dark:text-dark-text max-w-sm md:max-w-md truncate">
                      {item.title || item.text_content.slice(0, 80) + '...'}
                    </td>
                    <td className="py-4 px-6">
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase ${getVerdictStyle(item.predicted_label)}`}>
                        {item.predicted_label}
                      </span>
                    </td>
                    <td className="py-4 px-6 font-mono font-bold text-text-main dark:text-dark-text">
                      {(item.confidence_score * 100).toFixed(0)}%
                    </td>
                    <td className="py-4 px-6 font-semibold text-xs text-text-secondary dark:text-dark-text-secondary">
                      {item.model_version}
                    </td>
                    <td className="py-4 px-6 text-xs text-text-secondary dark:text-dark-text-secondary">
                      {item.created_at ? new Date(item.created_at).toLocaleString() : 'N/A'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal Detail View */}
      {selectedItem && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-fadeIn">
          <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 shadow-premium-lg flex flex-col justify-between">
            <div>
              {/* Modal Title Banner */}
              <div className="flex items-start justify-between border-b border-slate-100 dark:border-dark-border/40 pb-4 mb-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase ${getVerdictStyle(selectedItem.predicted_label)}`}>
                      {selectedItem.predicted_label}
                    </span>
                    <span className="text-[10px] font-semibold text-text-secondary dark:text-dark-text-secondary font-mono">
                      Confidence: {(selectedItem.confidence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <h3 className="text-base font-bold text-text-main dark:text-dark-text pr-4">
                    {selectedItem.title || 'Untitled Verification Record'}
                  </h3>
                </div>
                <button
                  onClick={() => setSelectedItem(null)}
                  className="p-1 text-text-secondary hover:text-text-main dark:text-dark-text-secondary dark:hover:text-dark-text rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                  <IoClose className="w-5 h-5" />
                </button>
              </div>

              {/* Snippet contents */}
              <div className="space-y-4 text-xs">
                <div>
                  <h4 className="font-bold text-text-main dark:text-dark-text uppercase tracking-wider mb-1.5">
                    Analyzed Text Content
                  </h4>
                  <div className="p-3 bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-dark-border/40 rounded-xl max-h-36 overflow-y-auto text-text-secondary dark:text-dark-text-secondary font-sans leading-relaxed">
                    {selectedItem.text_content}
                  </div>
                </div>

                {/* Explanation */}
                {selectedItem.explanation && (
                  <div>
                    <h4 className="font-bold text-text-main dark:text-dark-text uppercase tracking-wider mb-1">
                      Consensus Explanation
                    </h4>
                    <p className="text-text-secondary dark:text-dark-text-secondary italic">
                      {selectedItem.explanation}
                    </p>
                  </div>
                )}

                {/* Live Verifications list */}
                <div>
                  <h4 className="font-bold text-text-main dark:text-dark-text uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <IoGlobeOutline className="w-4 h-4 text-primary-blue" />
                    Web Verification References ({selectedItem.live_verifications?.length || 0})
                  </h4>
                  {selectedItem.live_verifications?.length === 0 ? (
                    <p className="text-text-secondary dark:text-dark-text-secondary italic">
                      No web reference details saved for this record.
                    </p>
                  ) : (
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {selectedItem.live_verifications.map((v, i) => (
                        <div key={i} className="flex justify-between items-center p-2.5 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-100 dark:border-dark-border/40">
                          <div>
                            <div className="font-semibold text-text-main dark:text-dark-text">{v.fact_checking_source}</div>
                            <span className="text-[10px] text-text-secondary dark:text-dark-text-secondary uppercase">
                              Verdict: {v.verdict.replace('VERIFIED_', '')}
                            </span>
                          </div>
                          <a 
                            href={v.source_url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="p-1 hover:bg-slate-200 dark:hover:bg-slate-800 rounded text-text-secondary hover:text-text-main dark:hover:text-dark-text transition-colors"
                          >
                            <IoOpenOutline className="w-4 h-4" />
                          </a>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex justify-end border-t border-slate-100 dark:border-dark-border/40 pt-4 mt-6">
              <button
                onClick={() => setSelectedItem(null)}
                className="px-4 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-xs font-semibold rounded-xl text-text-main dark:text-dark-text transition-colors"
              >
                Close Details
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
