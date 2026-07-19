import React, { useState, useEffect } from 'react';
import { modelService } from '../services/modelService';
import { predictionService } from '../services/predictionService';
import { useToast } from '../hooks/useToast';
import { 
  IoHardwareChipOutline, 
  IoCheckmarkCircleOutline,
  IoGitMergeOutline,
  IoStatsChartOutline
} from 'react-icons/io5';

export default function Statistics() {
  const { addToast } = useToast();
  const [modelInfo, setModelInfo] = useState({
    version: '1.0.1',
    accuracy: 0.885,
    f1_score: 0.879,
    algorithm: 'Linear SVM',
    training_date: '2026-07-19',
    status: 'online'
  });

  const [loading, setLoading] = useState(true);

  // Mocking all models for display as requested
  const allModels = [
    { name: 'Linear SVM (Primary)', acc: 88.5, f1: 87.9 },
    { name: 'XGBoost', acc: 87.1, f1: 86.4 },
    { name: 'Random Forest', acc: 86.2, f1: 85.8 },
    { name: 'Logistic Regression', acc: 84.5, f1: 83.9 },
  ];

  useEffect(() => {
    // Parallel load
    Promise.all([
      modelService.getModelInfo(),
      predictionService.getStats()
    ]).then(([modelData, statsData]) => {
      setModelInfo({
        version: modelData.version || '1.0.1',
        accuracy: modelData.accuracy || 0.885,
        f1_score: modelData.f1_score || 0.879,
        algorithm: modelData.algorithm_name || 'Linear SVM',
        training_date: modelData.training_date ? modelData.training_date.split('T')[0] : '2026-07-19',
        status: 'online'
      });
      setLoading(false);
    }).catch(err => {
      console.error(err);
      addToast('Failed to retrieve model parameters.', 'error');
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="h-[calc(100vh-100px)] flex flex-col space-y-4 animate-pulse p-4">
        <div className="h-8 bg-slate-200 dark:bg-slate-850 rounded-xl w-48" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 flex-1">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-slate-200 dark:bg-slate-850 rounded-2xl" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-120px)] flex flex-col space-y-4 max-w-layout mx-auto overflow-hidden p-2">
      {/* Title (Compact) */}
      <div className="flex-shrink-0">
        <h1 className="text-2xl font-extrabold tracking-tight text-text-main dark:text-dark-text">Model Analytics</h1>
        <p className="text-xs text-text-secondary dark:text-dark-text-secondary">
          Monitor prediction metrics, accuracy reports, and automated retraining workflows
        </p>
      </div>

      {/* Models Grid - Takes available space */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 flex-1 min-h-0">
        {allModels.map((model, idx) => (
          <div key={idx} className="bg-white/40 dark:bg-slate-800/40 backdrop-blur-md border border-white/30 dark:border-white/10 rounded-2xl p-4 shadow-xl flex flex-col justify-between transition-all hover:bg-white/50 dark:hover:bg-slate-800/60">
            <div>
              <div className="flex items-center gap-2 mb-3">
                {idx === 0 ? <IoHardwareChipOutline className="w-5 h-5 text-purple-500" /> : <IoStatsChartOutline className="w-5 h-5 text-primary-blue" />}
                <h3 className="text-xs font-bold text-text-main dark:text-dark-text uppercase tracking-wider truncate">
                  {model.name}
                </h3>
              </div>
              
              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1.5 border-b border-black/5 dark:border-white/5">
                  <span className="text-text-secondary dark:text-dark-text-secondary">Validation Accuracy</span>
                  <span className="font-bold text-success-green font-mono">{model.acc}%</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-black/5 dark:border-white/5">
                  <span className="text-text-secondary dark:text-dark-text-secondary">F1 Evaluation Score</span>
                  <span className="font-bold text-indigo-500 font-mono">{model.f1}%</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-black/5 dark:border-white/5">
                  <span className="text-text-secondary dark:text-dark-text-secondary">Release Tag</span>
                  <span className="font-mono font-semibold text-text-main dark:text-dark-text">v{modelInfo.version}</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-text-secondary dark:text-dark-text-secondary">Last Retrained</span>
                  <span className="font-mono font-semibold text-text-main dark:text-dark-text">{modelInfo.training_date}</span>
                </div>
              </div>
            </div>
            <div className="mt-3 text-[10px] text-center font-medium bg-primary-blue/10 text-primary-blue py-1 rounded-full">
              Production Ready Candidate
            </div>
          </div>
        ))}
      </div>

      {/* Retraining Flow Explanation (Compact Glassmorphism) */}
      <div className="flex-shrink-0 bg-white/40 dark:bg-slate-800/40 backdrop-blur-md border border-white/30 dark:border-white/10 rounded-2xl p-4 shadow-xl">
        <div className="mb-3 flex items-center gap-2">
          <IoGitMergeOutline className="w-5 h-5 text-indigo-500" />
          <div>
            <h3 className="text-xs font-bold text-text-main dark:text-dark-text uppercase">Feedback Loop & Retraining Protocol</h3>
            <p className="text-[10px] text-text-secondary dark:text-dark-text-secondary">
              How ML models incorporate user comments and maintain accuracy
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
          <div className="p-3 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm rounded-xl border border-white/20 dark:border-white/5">
            <div className="font-bold text-text-main dark:text-dark-text mb-1 flex items-center gap-1">
              <span className="bg-primary-blue text-white w-4 h-4 rounded-full flex items-center justify-center text-[9px]">1</span>
              Accumulation
            </div>
            <p className="text-text-secondary dark:text-dark-text-secondary leading-tight text-[10px]">
              User feedbacks on predictions are continuously collected and aggregated for evaluation.
            </p>
          </div>
          <div className="p-3 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm rounded-xl border border-white/20 dark:border-white/5">
            <div className="font-bold text-text-main dark:text-dark-text mb-1 flex items-center gap-1">
              <span className="bg-primary-blue text-white w-4 h-4 rounded-full flex items-center justify-center text-[9px]">2</span>
              Admin Audit
            </div>
            <p className="text-text-secondary dark:text-dark-text-secondary leading-tight text-[10px]">
              Administrators review the collected feedback to create a high-confidence verified dataset.
            </p>
          </div>
          <div className="p-3 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm rounded-xl border border-white/20 dark:border-white/5">
            <div className="font-bold text-text-main dark:text-dark-text mb-1 flex items-center gap-1">
              <span className="bg-primary-blue text-white w-4 h-4 rounded-full flex items-center justify-center text-[9px]">3</span>
              Split Testing
            </div>
            <p className="text-text-secondary dark:text-dark-text-secondary leading-tight text-[10px]">
              The system splits the verified dataset to retrain and evaluate all models against historical benchmarks.
            </p>
          </div>
          <div className="p-3 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm rounded-xl border border-white/20 dark:border-white/5">
            <div className="font-bold text-text-main dark:text-dark-text mb-1 flex items-center gap-1">
              <span className="bg-primary-blue text-white w-4 h-4 rounded-full flex items-center justify-center text-[9px]">4</span>
              Hot Deployment
            </div>
            <p className="text-text-secondary dark:text-dark-text-secondary leading-tight text-[10px]">
              If a newly trained model outperforms the current production version, it seamlessly deploys as the active model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
