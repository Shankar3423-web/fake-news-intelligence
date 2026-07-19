import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { predictionService } from '../services/predictionService';
import { useToast } from '../hooks/useToast';

import PredictionPipeline from '../components/PredictionPipeline';
import TrustedSourcesTable from '../components/TrustedSourcesTable';
import AnalysisSummary from '../components/AnalysisSummary';
import PredictionFeedback from '../components/PredictionFeedback';
import AuthModal from '../components/AuthModal';
import { useAuth } from '../context/AuthContext';

export default function VerifyNews() {
  const { addToast } = useToast();
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  
  // Pipeline & results state
  const [status, setStatus] = useState('idle'); // 'idle' | 'running' | 'completed'
  const [activeStep, setActiveStep] = useState(0);
  const [resultData, setResultData] = useState(null);

  // Auth State
  const { isAuthenticated } = useAuth();
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  // Character counter
  const charCount = text.length;

  const handleClear = () => {
    setTitle('');
    setText('');
    setStatus('idle');
    setActiveStep(0);
    setResultData(null);
  };

  const handleVerify = async (e) => {
    e?.preventDefault();
    if (!text.trim()) {
      addToast('Please enter the article content to verify.', 'warning');
      return;
    }
    
    if (!isAuthenticated) {
      setIsAuthModalOpen(true);
      return;
    }

    // Set state to running
    setStatus('running');
    setActiveStep(0);
    setResultData(null);

    // Start pipeline step animation loop
    const stepInterval = setInterval(() => {
      setActiveStep((prev) => {
        if (prev < 7) return prev + 1;
        clearInterval(stepInterval);
        return 7;
      });
    }, 850); // Speed matching standard analysis time

    try {
      // Execute the Ultimate consensus pipeline
      const response = await predictionService.analyzeFull(title, text);
      
      // Keep track of response data, but wait for pipeline steps to complete (or speed up if finished early)
      setTimeout(() => {
        clearInterval(stepInterval);
        setActiveStep(7);
        setResultData(response);
        setStatus('completed');
        addToast('News verification complete.', 'success');
      }, 6000); // 6s duration of pipeline visual stages
    } catch (err) {
      clearInterval(stepInterval);
      setStatus('idle');
      setActiveStep(0);
      addToast(err.message || 'Analysis failed. Please check network connection.', 'error');
    }
  };

  return (
    <div className="space-y-10 max-w-layout mx-auto">
      <AuthModal 
        isOpen={isAuthModalOpen} 
        onClose={() => setIsAuthModalOpen(false)} 
        onSuccess={() => handleVerify()} 
      />

      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight text-text-main dark:text-dark-text">
          Verify News Article
        </h1>
        <p className="text-sm text-text-secondary dark:text-dark-text-secondary">
          Enter news article details and get AI-powered verification
        </p>
      </div>

      {/* Main Layout: Left Side (Input + Result) and Right Side (Pipeline) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        
        {/* Left Side: Input and Result (Stack) */}
        <div className="lg:col-span-2 space-y-8 flex flex-col">
          
          {/* Top: Input Card */}
          <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-6 shadow-premium">
          <div className="mb-4">
            <h2 className="text-sm font-bold text-text-main dark:text-dark-text uppercase tracking-wider">
              News Article Input
            </h2>
            <p className="text-xs text-text-secondary dark:text-dark-text-secondary">
              Provide article title and body content for consensus analysis
            </p>
          </div>

          <form onSubmit={handleVerify} className="space-y-4">
            <div>
              <label htmlFor="article-title" className="block text-xs font-semibold text-text-main dark:text-dark-text mb-1.5">
                News Title
              </label>
              <input
                id="article-title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter article headline or title..."
                className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-border dark:border-dark-border rounded-xl focus:outline-none focus:ring-1 focus:ring-primary-blue text-sm text-text-main dark:text-dark-text"
              />
            </div>

            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label htmlFor="article-text" className="block text-xs font-semibold text-text-main dark:text-dark-text">
                  News Content
                </label>
                <span className="text-[10px] text-text-secondary dark:text-dark-text-secondary font-mono">
                  {charCount} characters
                </span>
              </div>
              <textarea
                id="article-text"
                rows="10"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the full article content here to scan for misinformation..."
                className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-border dark:border-dark-border rounded-xl focus:outline-none focus:ring-1 focus:ring-primary-blue text-sm text-text-main dark:text-dark-text"
              />
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button
                type="submit"
                disabled={status === 'running'}
                className="flex-1 px-5 py-3 bg-primary-blue hover:bg-primary-hover disabled:bg-primary-blue/50 text-white font-semibold rounded-xl text-sm transition-colors shadow-sm"
              >
                {status === 'running' ? 'Scanning claims...' : 'Verify News'}
              </button>
              
              <button
                type="button"
                onClick={handleClear}
                disabled={status === 'running'}
                className="px-5 py-3 bg-white dark:bg-dark-card border border-border dark:border-dark-border hover:bg-slate-50 dark:hover:bg-slate-800 text-text-main dark:text-dark-text font-semibold rounded-xl text-sm transition-colors shadow-sm"
              >
                Clear
              </button>
            </div>
          </form>
        </div>

          {/* Bottom: Result Card */}
          <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-6 shadow-premium min-h-[445px] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-sm font-bold text-text-main dark:text-dark-text uppercase tracking-wider">
                Verification Result
              </h2>
              
              {/* Status Badge */}
              <div className="flex items-center gap-4">
                {resultData && (
                  <span className="text-[10px] text-text-secondary dark:text-dark-text-secondary font-semibold font-mono">
                    Time: {resultData.processing_time.toFixed(2)}s
                  </span>
                )}
                <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                  status === 'completed'
                    ? 'bg-success-green/10 text-success-green border border-success-green/20'
                    : status === 'running'
                    ? 'bg-primary-blue/10 text-primary-blue border border-primary-blue/20 animate-pulse'
                    : 'bg-slate-100 dark:bg-slate-800 text-text-secondary dark:text-dark-text-secondary border border-border dark:border-dark-border'
                }`}>
                  {status === 'completed' ? 'Completed' : status === 'running' ? 'Processing' : 'Idle'}
                </span>
              </div>
            </div>

            {/* Verdict Display */}
            {status === 'idle' && (
              <div className="flex flex-col items-center justify-center py-20 text-center text-text-secondary dark:text-dark-text-secondary">
                <p className="text-sm font-medium">No active analysis results.</p>
                <p className="text-xs mt-1">Input text on the left to activate the Consensus Verification Engine.</p>
              </div>
            )}

            {status === 'running' && (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <div className="w-10 h-10 border-4 border-primary-blue border-t-transparent rounded-full animate-spin mb-4" />
                <p className="text-sm font-semibold text-text-main dark:text-dark-text">Analyzing claims and evidence...</p>
                <p className="text-xs text-text-secondary dark:text-dark-text-secondary mt-1">Executing 12-phase pipeline (Node {activeStep + 1}/8)</p>
              </div>
            )}

            {status === 'completed' && resultData && (
              <div className="space-y-8 animate-fadeIn">
                <div className="text-center py-4 bg-slate-50/50 dark:bg-slate-900/30 rounded-2xl border border-dashed border-border dark:border-dark-border/40">
                  <h3 className="text-xs font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider mb-1">
                    Final Verdict
                  </h3>
                  
                  {/* HUGE TYPOGRAPHY */}
                  <div className={`text-6xl font-extrabold tracking-tighter uppercase mb-2 ${
                    resultData.final_verdict === 'REAL' ? 'text-success-green' : 'text-danger-red'
                  }`}>
                    {resultData.final_verdict}
                  </div>
                  
                  <p className="text-sm font-medium text-text-main dark:text-dark-text">
                    This news is verified as {resultData.final_verdict.toLowerCase()}
                  </p>
                </div>

                {/* Two Equal Cards */}
                <div className="grid grid-cols-2 gap-4">
                  {/* ML Prediction & Real-Time Screening */}
                  <div className="bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-dark-border/50 rounded-xl p-3.5 text-center">
                    <span className="block text-[10px] font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider mb-1">
                      ML Prediction & Real-Time Screening
                    </span>
                    <span className={`text-xs font-extrabold uppercase truncate block ${
                      resultData.final_verdict === 'REAL' ? 'text-success-green' : 'text-danger-red'
                    }`}>
                      {resultData.final_verdict}
                    </span>
                  </div>

                  {/* Confidence */}
                  <div className="bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-dark-border/50 rounded-xl p-3.5 text-center">
                    <span className="block text-[10px] font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider mb-1">
                      Confidence Score
                    </span>
                    <span className="text-xs font-extrabold text-success-green font-mono">
                      {(resultData.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Inline Feedback Form */}
                <PredictionFeedback
                  prediction={resultData.final_verdict}
                  confidence={resultData.confidence}
                  verificationStatus={resultData.verification_status}
                  evidenceScore={resultData.verification_details?.evidence_score}
                  similarityScore={resultData.verification_details?.similarity_score}
                  finalDecision={resultData.final_verdict}
                  variant="inline"
                />
              </div>
            )}
          </div>
          </div>
          
          {/* Trusted Sources Table */}
          {status === 'completed' && resultData && (
            <div className="mt-8 animate-fadeIn">
              <TrustedSourcesTable 
                articles={resultData.verification_details?.matched_articles} 
                status={status}
              />
            </div>
          )}
        
        </div> {/* End Left Side Stack */}

        {/* Right Side: Execution Pipeline & Summary */}
        <div className="lg:col-span-1">
          <div className="sticky top-6 space-y-8">
            <PredictionPipeline status={status} activeStep={activeStep} layout="vertical" />
            
            {status === 'completed' && resultData && (
              <div className="animate-fadeIn">
                <AnalysisSummary
                  evidenceScore={resultData.verification_details?.evidence_score}
                  similarityScore={resultData.verification_details?.similarity_score}
                  trustedSourcesCount={resultData.verification_details?.trusted_source_count}
                  confidence={resultData.confidence}
                  modelName={resultData.prediction_details?.model_name || 'ensemble'}
                  evaluationVersion={resultData.prediction_details?.evaluation_version || '1.0.1'}
                  processingTime={resultData.processing_time}
                />
              </div>
            )}
          </div>
        </div>
      </div> {/* End Main Grid Layout */}
    </div>
  );
}
