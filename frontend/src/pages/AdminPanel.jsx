import React, { useState, useEffect } from 'react';
import { adminService } from '../services/adminService';
import { useToast } from '../hooks/useToast';
import { 
  IoShieldCheckmarkOutline, 
  IoCheckmarkCircle, 
  IoCloseCircle, 
  IoTimeOutline, 
  IoChatboxEllipsesOutline,
  IoThumbsUpOutline,
  IoThumbsDownOutline,
  IoAlertCircleOutline,
  IoRefreshOutline,
  IoCheckmark,
  IoClose
} from 'react-icons/io5';

export default function AdminPanel() {
  const { addToast } = useToast();
  const [pendingFeedbacks, setPendingFeedbacks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionLoadingId, setActionLoadingId] = useState(null);
  const [reviewerNotes, setReviewerNotes] = useState({});

  const loadPendingReviews = () => {
    setLoading(true);
    adminService.getPendingReviews()
      .then((data) => {
        setPendingFeedbacks(data.pending_feedbacks || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load pending reviews:', err);
        addToast(err.message || 'Failed to fetch pending admin reviews.', 'error');
        setLoading(false);
      });
  };

  useEffect(() => {
    loadPendingReviews();
  }, []);

  const handleReview = (feedbackId, status) => {
    setActionLoadingId(feedbackId);
    const notes = reviewerNotes[feedbackId] || '';
    
    adminService.reviewFeedback(feedbackId, status, 'Platform Administrator', notes)
      .then(() => {
        const actionVerb = status === 'APPROVED' ? 'accepted & approved for model retraining' : 'rejected';
        addToast(`Feedback #${feedbackId} successfully ${actionVerb}.`, 'success');
        // Remove reviewed item from list
        setPendingFeedbacks(prev => prev.filter(item => item["Feedback ID"] !== feedbackId));
        setActionLoadingId(null);
      })
      .catch((err) => {
        console.error('Failed to review feedback:', err);
        addToast(err.message || `Failed to ${status.toLowerCase()} feedback.`, 'error');
        setActionLoadingId(null);
      });
  };

  const handleNotesChange = (id, val) => {
    setReviewerNotes(prev => ({ ...prev, [id]: val }));
  };

  return (
    <div className="space-y-8 max-w-layout mx-auto">
      {/* Header section */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-primary-blue font-semibold text-xs uppercase tracking-wider mb-1">
            <IoShieldCheckmarkOutline className="w-4 h-4" />
            <span>Administrator Control Center</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-text-main dark:text-dark-text">
            Fake News Admin Panel
          </h1>
          <p className="text-sm text-text-secondary dark:text-dark-text-secondary mt-1">
            Review user feedbacks to approve dataset corrections for upcoming AI model retraining rounds.
          </p>
        </div>

        <button
          onClick={loadPendingReviews}
          disabled={loading}
          className="self-start sm:self-auto flex items-center gap-2 px-4 py-2.5 text-xs font-semibold bg-white dark:bg-dark-card text-text-main dark:text-dark-text border border-border dark:border-dark-border rounded-xl hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors shadow-sm disabled:opacity-50"
        >
          <IoRefreshOutline className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh Queue
        </button>
      </div>

      {/* Overview Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-5 shadow-premium">
          <div className="flex justify-between items-center text-text-secondary dark:text-dark-text-secondary mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Pending Reviews</span>
            <IoTimeOutline className="w-5 h-5 text-amber-500" />
          </div>
          <p className="text-3xl font-extrabold text-text-main dark:text-dark-text">
            {loading ? '...' : pendingFeedbacks.length}
          </p>
          <span className="text-xs text-text-secondary dark:text-dark-text-secondary mt-1 inline-block">
            Awaiting administrator decision
          </span>
        </div>

        <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-5 shadow-premium">
          <div className="flex justify-between items-center text-text-secondary dark:text-dark-text-secondary mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Retraining Status</span>
            <IoShieldCheckmarkOutline className="w-5 h-5 text-success-green" />
          </div>
          <p className="text-xl font-bold text-success-green">
            Ready to Accept
          </p>
          <span className="text-xs text-text-secondary dark:text-dark-text-secondary mt-1 inline-block">
            PostgreSQL Database Synced
          </span>
        </div>

        <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-5 shadow-premium">
          <div className="flex justify-between items-center text-text-secondary dark:text-dark-text-secondary mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">System Pipeline</span>
            <IoAlertCircleOutline className="w-5 h-5 text-primary-blue" />
          </div>
          <p className="text-xl font-bold text-text-main dark:text-dark-text">
            Active Mode
          </p>
          <span className="text-xs text-text-secondary dark:text-dark-text-secondary mt-1 inline-block">
            Automated verification pipeline
          </span>
        </div>
      </div>

      {/* Pending Reviews Table */}
      <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl shadow-premium overflow-hidden">
        <div className="p-6 border-b border-slate-100 dark:border-dark-border/40 flex justify-between items-center">
          <h2 className="text-base font-bold text-text-main dark:text-dark-text">
            Feedback Queue ({pendingFeedbacks.length})
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-100 dark:border-dark-border/40 text-xs font-bold text-text-secondary dark:text-dark-text-secondary uppercase bg-slate-50/50 dark:bg-slate-900/30">
                <th className="py-4 px-6">ID & Timestamp</th>
                <th className="py-4 px-6">Article Headline</th>
                <th className="py-4 px-6">Model Verdict</th>
                <th className="py-4 px-6">User Feedback</th>
                <th className="py-4 px-6">User Comment</th>
                <th className="py-4 px-6 text-center">Admin Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-dark-border/30 text-sm">
              {loading ? (
                [...Array(3)].map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-16" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-48" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-16" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-20" /></td>
                    <td className="py-4 px-6"><div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-32" /></td>
                    <td className="py-4 px-6"><div className="h-8 bg-slate-200 dark:bg-slate-800 rounded w-32 mx-auto" /></td>
                  </tr>
                ))
              ) : pendingFeedbacks.length === 0 ? (
                <tr>
                  <td colSpan="6" className="py-16 text-center text-text-secondary dark:text-dark-text-secondary">
                    <div className="flex flex-col items-center gap-2">
                      <IoCheckmarkCircle className="w-10 h-10 text-success-green/60" />
                      <p className="font-semibold text-base">No Pending Feedbacks</p>
                      <p className="text-xs">All user feedback submissions have been processed and reviewed!</p>
                    </div>
                  </td>
                </tr>
              ) : (
                pendingFeedbacks.map((f) => {
                  const fid = f["Feedback ID"];
                  const isProcessing = actionLoadingId === fid;
                  const userAgreed = f["User Feedback"] === 'AGREE';

                  return (
                    <tr key={fid} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/30 transition-colors">
                      {/* ID & Date */}
                      <td className="py-4 px-6 font-mono text-xs text-text-secondary dark:text-dark-text-secondary">
                        <span className="font-bold text-text-main dark:text-dark-text block">#{fid}</span>
                        {f.Timestamp ? new Date(f.Timestamp).toLocaleDateString() : 'N/A'}
                      </td>

                      {/* Headline */}
                      <td className="py-4 px-6 font-semibold text-text-main dark:text-dark-text max-w-xs truncate">
                        {f.Headline || f.Prediction || 'News Article'}
                      </td>

                      {/* Model Verdict */}
                      <td className="py-4 px-6">
                        <span className={`px-2.5 py-1 rounded-lg text-xs font-bold uppercase ${
                          f.Prediction === 'REAL'
                            ? 'bg-success-green/10 text-success-green border border-success-green/20'
                            : 'bg-danger-red/10 text-danger-red border border-danger-red/20'
                        }`}>
                          {f.Prediction}
                        </span>
                      </td>

                      {/* User Feedback */}
                      <td className="py-4 px-6">
                        <div className="flex items-center gap-1.5 text-xs font-semibold">
                          {userAgreed ? (
                            <>
                              <IoThumbsUpOutline className="w-4 h-4 text-success-green" />
                              <span className="text-success-green">Correct (Agrees)</span>
                            </>
                          ) : (
                            <>
                              <IoThumbsDownOutline className="w-4 h-4 text-danger-red" />
                              <span className="text-danger-red">Incorrect (Disagrees)</span>
                            </>
                          )}
                        </div>
                      </td>

                      {/* User Comment & Admin Notes */}
                      <td className="py-4 px-6 max-w-xs">
                        <p className="text-xs text-text-main dark:text-dark-text italic mb-2">
                          "{f.Comment || 'No comment provided'}"
                        </p>
                        <input
                          type="text"
                          placeholder="Optional reviewer notes..."
                          value={reviewerNotes[fid] || ''}
                          onChange={(e) => handleNotesChange(fid, e.target.value)}
                          className="w-full text-xs px-2.5 py-1 bg-slate-50 dark:bg-slate-900 border border-border dark:border-dark-border rounded-lg focus:outline-none focus:ring-1 focus:ring-primary-blue text-text-main dark:text-dark-text"
                        />
                      </td>

                      {/* Actions */}
                      <td className="py-4 px-6">
                        <div className="flex items-center justify-center gap-2">
                          <button
                            onClick={() => handleReview(fid, 'APPROVED')}
                            disabled={isProcessing}
                            className="flex items-center gap-1 px-3 py-1.5 text-xs font-semibold bg-success-green hover:bg-emerald-600 disabled:opacity-50 text-white rounded-xl shadow-sm transition-colors"
                            title="Accept and approve for retraining"
                          >
                            <IoCheckmark className="w-4 h-4" />
                            Accept
                          </button>
                          
                          <button
                            onClick={() => handleReview(fid, 'REJECTED')}
                            disabled={isProcessing}
                            className="flex items-center gap-1 px-3 py-1.5 text-xs font-semibold bg-danger-red hover:bg-rose-600 disabled:opacity-50 text-white rounded-xl shadow-sm transition-colors"
                            title="Reject feedback"
                          >
                            <IoClose className="w-4 h-4" />
                            Reject
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
