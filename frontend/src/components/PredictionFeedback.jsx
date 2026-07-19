import React, { useState } from 'react';
import { feedbackService } from '../services/feedbackService';
import { useToast } from '../hooks/useToast';
import { IoThumbsUp, IoThumbsDown, IoChatbubbleOutline, IoSend } from 'react-icons/io5';

export default function PredictionFeedback({ 
  prediction, 
  confidence, 
  verificationStatus, 
  evidenceScore, 
  similarityScore, 
  finalDecision,
  variant = 'card'
}) {
  const { addToast } = useToast();
  const [feedbackType, setFeedbackType] = useState(null); // 'AGREE' (Yes) or 'DISAGREE' (No)
  const [comment, setComment] = useState('');
  const [showCommentForm, setShowCommentForm] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleFeedbackClick = (type) => {
    setFeedbackType(type);
    setShowCommentForm(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!feedbackType) return;

    setSubmitting(true);
    try {
      await feedbackService.submitFeedback({
        prediction: prediction || 'UNKNOWN',
        prediction_confidence: confidence || 0.0,
        verification_status: verificationStatus || 'UNKNOWN',
        evidence_score: evidenceScore || 0.0,
        similarity_score: similarityScore || 0.0,
        final_decision: finalDecision || 'UNKNOWN',
        user_feedback: feedbackType, // 'AGREE' or 'DISAGREE'
        comment: comment || undefined
      });
      
      setSubmitted(true);
      addToast('Thank you! Your feedback has been submitted.', 'success');
      setShowCommentForm(false);
    } catch (err) {
      console.error(err);
      addToast(err.message || 'Failed to submit feedback.', 'error');
    } finally {
      setSubmitting(false);
    }
  };

  const containerClasses = variant === 'inline' 
    ? "w-full pt-4 border-t border-dashed border-slate-200 dark:border-dark-border mt-6" 
    : "bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-6 shadow-premium w-full";

  if (submitted) {
    return (
      <div className={`${containerClasses} text-center`}>
        <p className="text-sm font-semibold text-success-green">
          🎉 Feedback recorded successfully. Thank you for helping train our models!
        </p>
      </div>
    );
  }

  return (
    <div className={containerClasses}>
      <div className={`flex flex-col gap-4 ${variant === 'inline' ? '' : 'md:flex-row md:items-center md:justify-between'}`}>
        <div>
          <h3 className="text-sm font-bold text-text-main dark:text-dark-text">Was this prediction correct?</h3>
          <p className="text-xs text-text-secondary dark:text-dark-text-secondary mt-1">
            Providing feedback helps the model become more powerful.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => handleFeedbackClick('AGREE')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-xl border transition-colors ${
              feedbackType === 'AGREE'
                ? 'bg-success-green/10 text-success-green border-success-green'
                : 'bg-white dark:bg-dark-secondary-bg hover:bg-slate-50 dark:hover:bg-slate-800 text-text-main dark:text-dark-text border-border dark:border-dark-border'
            }`}
          >
            <IoThumbsUp className="w-4 h-4" />
            Yes, Correct
          </button>
          
          <button
            onClick={() => handleFeedbackClick('DISAGREE')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-xl border transition-colors ${
              feedbackType === 'DISAGREE'
                ? 'bg-danger-red/10 text-danger-red border-danger-red'
                : 'bg-white dark:bg-dark-secondary-bg hover:bg-slate-50 dark:hover:bg-slate-800 text-text-main dark:text-dark-text border-border dark:border-dark-border'
            }`}
          >
            <IoThumbsDown className="w-4 h-4" />
            No, Incorrect
          </button>

          {!showCommentForm && (
            <button
              onClick={() => setShowCommentForm(true)}
              className="flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-xl border border-slate-200 dark:border-dark-border hover:bg-slate-50 dark:hover:bg-slate-850 text-text-secondary dark:text-dark-text-secondary transition-colors"
            >
              <IoChatbubbleOutline className="w-4 h-4" />
              Provide Notes
            </button>
          )}
        </div>
      </div>

      {showCommentForm && (
        <form onSubmit={handleSubmit} className="mt-6 border-t border-slate-100 dark:border-dark-border/40 pt-4 flex flex-col gap-3">
          <label htmlFor="feedback-comment" className="text-xs font-semibold text-text-main dark:text-dark-text">
            Add Optional Notes
          </label>
          <textarea
            id="feedback-comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Explain why this prediction is correct or incorrect..."
            className="w-full min-h-[80px] p-3 text-xs bg-slate-50 dark:bg-slate-900 border border-border dark:border-dark-border rounded-xl focus:outline-none focus:ring-1 focus:ring-primary-blue text-text-main dark:text-dark-text"
          />
          <div className="flex justify-end gap-3 mt-2">
            <button
              type="button"
              onClick={() => setShowCommentForm(false)}
              className="px-4 py-2 text-xs font-semibold text-text-secondary dark:text-dark-text-secondary hover:bg-slate-50 dark:hover:bg-slate-800 rounded-xl"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold bg-primary-blue hover:bg-primary-hover disabled:opacity-50 text-white rounded-xl shadow-sm transition-colors"
            >
              <IoSend className="w-3.5 h-3.5" />
              {submitting ? 'Submitting...' : 'Submit Feedback'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
