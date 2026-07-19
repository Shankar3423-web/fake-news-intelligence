import React from 'react';
import { 
  IoShieldCheckmarkOutline, 
  IoGitCompareOutline, 
  IoGlobeOutline, 
  IoTrendingUpOutline, 
  IoHardwareChipOutline, 
  IoGitBranchOutline, 
  IoTimeOutline 
} from 'react-icons/io5';

export default function AnalysisSummary({ 
  evidenceScore = 0.0,
  similarityScore = 0.0,
  trustedSourcesCount = 0,
  confidence = 0.0,
  modelName = 'svm',
  evaluationVersion = '1.0.1',
  processingTime = 0.0
}) {
  const summaryRows = [
    {
      icon: <IoShieldCheckmarkOutline className="w-5 h-5 text-primary-blue" />,
      label: 'Evidence Score',
      value: (evidenceScore * 100).toFixed(1) + '%'
    },
    {
      icon: <IoGitCompareOutline className="w-5 h-5 text-indigo-500" />,
      label: 'Similarity Score',
      value: (similarityScore * 100).toFixed(1) + '%'
    },
    {
      icon: <IoGlobeOutline className="w-5 h-5 text-teal-500" />,
      label: 'Trusted Sources',
      value: `${trustedSourcesCount} Sources`
    },
    {
      icon: <IoTrendingUpOutline className="w-5 h-5 text-success-green" />,
      label: 'Verification Confidence',
      value: (confidence * 100).toFixed(1) + '%'
    },
    {
      icon: <IoHardwareChipOutline className="w-5 h-5 text-purple-500" />,
      label: 'Resources Used',
      value: `${modelName.toUpperCase() === 'SVM' ? 'Linear SVM' : modelName.toUpperCase()}, TF-IDF, NewsAPI`
    },
    {
      icon: <IoGitBranchOutline className="w-5 h-5 text-orange-500" />,
      label: 'Evaluation Version',
      value: `v${evaluationVersion}`
    },
    {
      icon: <IoTimeOutline className="w-5 h-5 text-pink-500" />,
      label: 'Request Time',
      value: processingTime ? `${processingTime.toFixed(3)}s` : '0.000s'
    }
  ];

  return (
    <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-6 shadow-premium w-full">
      <div className="mb-6">
        <h3 className="text-sm font-bold text-text-main dark:text-dark-text">Analysis Summary</h3>
        <p className="text-xs text-text-secondary dark:text-dark-text-secondary">
          Consensus metrics and deployment pipeline meta indicators
        </p>
      </div>

      <div className="divide-y divide-slate-100 dark:divide-dark-border/30">
        {summaryRows.map((row, idx) => (
          <div key={idx} className="flex items-center justify-between py-3.5 first:pt-0 last:pb-0">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-slate-50 dark:bg-slate-900 rounded-xl">
                {row.icon}
              </div>
              <span className="text-xs font-semibold text-text-secondary dark:text-dark-text-secondary">
                {row.label}
              </span>
            </div>
            <span className="text-sm font-bold text-text-main dark:text-dark-text font-mono">
              {row.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
