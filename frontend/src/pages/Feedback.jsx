import React, { useState, useEffect } from 'react';
import { feedbackService } from '../services/feedbackService';
import { useToast } from '../hooks/useToast';
import { 
  IoChatboxEllipsesOutline,
  IoThumbsUpOutline, 
  IoThumbsDownOutline, 
  IoCheckmarkCircle, 
  IoCloseCircle 
} from 'react-icons/io5';

export default function Feedback() {
  const { addToast } = useToast();
  const [feedbacks, setFeedbacks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    feedbackService.getFeedbacks()
      .then((data) => {
        setFeedbacks(data.feedbacks || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load feedback logs:', err);
        addToast('Failed to fetch user feedbacks from database.', 'error');
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-10 max-w-layout mx-auto">
      {/* Title */}
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight text-text-main dark:text-dark-text">User Feedbacks</h1>
        <p className="text-sm text-text-secondary dark:text-dark-text-secondary">
          Monitor correctness logs and comments submitted by platform verify agents
        </p>
      </div>

      {/* Feedback logs card */}
      <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl shadow-premium overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-100 dark:border-dark-border/40 text-xs font-bold text-text-secondary dark:text-dark-text-secondary uppercase">
                <th className="py-4 px-6">News Headline</th>
                <th className="py-4 px-6">Model Verdict</th>
                <th className="py-4 px-6">User Verdict</th>
                <th className="py-4 px-6">User Comments</th>
                <th className="py-4 px-6">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-dark-border/30 text-sm">
              {loading ? (
                [...Array(3)].map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-64" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-12" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-16" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-48" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-850 rounded w-24" /></td>
                  </tr>
                ))
              ) : feedbacks.length === 0 ? (
                <tr>
                  <td colSpan="5" className="py-12 text-center text-text-secondary dark:text-dark-text-secondary">
                    No user feedbacks registered in PostgreSQL database.
                  </td>
                </tr>
              ) : (
                feedbacks.map((f, idx) => (
                  <tr key={f.feedback_id || idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30 transition-colors">
                    <td className="py-4 px-6 font-semibold text-text-main dark:text-dark-text max-w-xs truncate">
                      {f.prediction_title}
                    </td>
                    <td className="py-4 px-6">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                        f.prediction_label === 'REAL'
                          ? 'bg-success-green/10 text-success-green'
                          : 'bg-danger-red/10 text-danger-red'
                      }`}>
                        {f.prediction_label}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-1.5 text-xs">
                        {f.is_correct ? (
                          <>
                            <IoCheckmarkCircle className="w-4 h-4 text-success-green" />
                            <span className="font-semibold text-success-green">Correct</span>
                          </>
                        ) : (
                          <>
                            <IoCloseCircle className="w-4 h-4 text-danger-red" />
                            <span className="font-semibold text-danger-red">Incorrect</span>
                          </>
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-6 text-xs text-text-secondary dark:text-dark-text-secondary max-w-sm truncate italic">
                      {f.user_comment || 'No comment provided'}
                    </td>
                    <td className="py-4 px-6 text-xs text-text-secondary dark:text-dark-text-secondary">
                      {f.created_at ? new Date(f.created_at).toLocaleDateString() : 'N/A'}
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
}
