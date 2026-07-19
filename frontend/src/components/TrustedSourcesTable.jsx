import React from 'react';
import { IoOpenOutline, IoGlobeOutline, IoCheckmarkCircle } from 'react-icons/io5';

export default function TrustedSourcesTable({ articles = [], status = 'idle' }) {
  const getInitials = (source) => {
    if (!source) return 'N';
    return source.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-6 shadow-premium w-full overflow-hidden">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-sm font-bold text-text-main dark:text-dark-text">Live Matched Articles</h3>
          <p className="text-xs text-text-secondary dark:text-dark-text-secondary">
            Cross-referenced articles discovered from live internet search APIs
          </p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 bg-success-green/10 dark:bg-success-green/20 rounded-full border border-success-green/20">
          <IoCheckmarkCircle className="w-4 h-4 text-success-green" />
          <span className="text-xs font-semibold text-success-green">
            {articles.length} Matches Found
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-100 dark:border-dark-border/40 text-xs font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider">
              <th className="py-3 px-4">Source</th>
              <th className="py-3 px-4">Article Title</th>
              <th className="py-3 px-4">Similarity Score</th>
              <th className="py-3 px-4">Published Date</th>
              <th className="py-3 px-4 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-dark-border/30 text-sm">
            {articles.length === 0 ? (
              <tr>
                <td colSpan="5" className="py-8 text-center text-text-secondary dark:text-dark-text-secondary">
                  {status === 'completed' 
                    ? 'Live API search yielded 0 matches for this article. The system relied on the ML Prediction consensus.' 
                    : 'No articles matched. Run a verification check to search live sources.'}
                </td>
              </tr>
            ) : (
              articles.map((art, idx) => {
                const similarityPct = Math.round(art.similarity_score * 100);
                
                return (
                  <tr 
                    key={idx} 
                    className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30 transition-colors cursor-pointer"
                    onClick={() => window.open(art.url, '_blank')}
                  >
                    {/* Source Logo/Text */}
                    <td className="py-4 px-4 flex items-center gap-3">
                      <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary-blue/10 dark:bg-primary-blue/20 text-primary-blue font-bold text-xs">
                        {getInitials(art.source)}
                      </div>
                      <div>
                        <div className="font-semibold text-text-main dark:text-dark-text">{art.source || 'Unknown'}</div>
                        {art.is_trusted && (
                          <span className="text-[10px] font-semibold text-success-green uppercase bg-success-green/10 px-1.5 py-0.5 rounded">
                            Trusted
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Title */}
                    <td className="py-4 px-4 max-w-xs md:max-w-md truncate">
                      <a 
                        href={art.url} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="font-medium text-text-main dark:text-dark-text hover:text-primary-blue dark:hover:text-primary-blue hover:underline"
                        title={art.title}
                      >
                        {art.title}
                      </a>
                    </td>

                    {/* Similarity Progress Bar */}
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-3">
                        <span className="font-semibold font-mono text-xs text-text-main dark:text-dark-text min-w-[32px]">
                          {similarityPct}%
                        </span>
                        <div className="w-24 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all duration-500 ${
                              similarityPct > 70 
                                ? 'bg-success-green' 
                                : similarityPct > 40 
                                ? 'bg-primary-blue' 
                                : 'bg-warning-orange'
                            }`}
                            style={{ width: `${similarityPct}%` }}
                          />
                        </div>
                      </div>
                    </td>

                    {/* Date */}
                    <td className="py-4 px-4 text-xs font-semibold text-text-secondary dark:text-dark-text-secondary">
                      {formatDate(art.published_date)}
                    </td>

                    {/* Link action */}
                    <td className="py-4 px-4 text-center">
                      <a 
                        href={art.url} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="inline-flex p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 text-text-secondary hover:text-text-main dark:hover:text-dark-text rounded-lg transition-colors"
                      >
                        <IoOpenOutline className="w-4 h-4" />
                      </a>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
