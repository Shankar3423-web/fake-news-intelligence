import React from 'react';
import { 
  IoShieldCheckmarkOutline, 
  IoGitNetworkOutline, 
  IoHardwareChipOutline,
} from 'react-icons/io5';

export default function About() {
  const models = [
    { name: 'XGBoost', desc: 'Extreme Gradient Boosting framework optimizing complex non-linear patterns.' },
    { name: 'Support Vector Machine', desc: 'High-dimensional hyperplane separations for robust text categorization.' },
    { name: 'Random Forest', desc: 'Ensemble of decision trees mitigating overfitting via bagging.' },
    { name: 'Gradient Boosting', desc: 'Sequentially built trees minimizing prediction residuals iteratively.' },
    { name: 'Decision Tree', desc: 'Rule-based classification using feature thresholds.' },
    { name: 'Logistic Regression', desc: 'Fast, interpretable baseline linear classifier outputting class probabilities.' }
  ];

  const phases = [
    { num: '01', title: 'Data Ingestion', desc: 'Loads raw text datasets from DB layers.' },
    { num: '02', title: 'Deduplication', desc: 'Removes identical or redundant records.' },
    { num: '03', title: 'Text Preprocessing', desc: 'Cleans HTML, URLs, and special characters.' },
    { num: '04', title: 'Tokenization', desc: 'Splits cleaned text into individual words.' },
    { num: '05', title: 'Stem & Stop-words', desc: 'Reduces words to roots, removes filler words.' },
    { num: '06', title: 'TF-IDF Vectorization', desc: 'Converts text to term-frequency matrices.' },
    { num: '07', title: 'Chi-Square Selection', desc: 'Isolates top predictive vocabulary features.' },
    { num: '08', title: 'Model Inference', desc: 'Generates probabilities via ML ensemble.' },
    { num: '09', title: 'Live Verification', desc: 'Cross-references active web search APIs.' },
    { num: '10', title: 'Feedback Loop', desc: 'Logs user/agent notes on correctness.' },
    { num: '11', title: 'Admin Oversight', desc: 'Curates and approves feedback subsets.' },
    { num: '12', title: 'Retraining Pipelines', desc: 'Hot-swaps production models upon gains.' }
  ];

  return (
    <div className="max-w-[1400px] mx-auto px-4 py-2 h-[calc(100vh-80px)] flex flex-col overflow-hidden relative">
      
      {/* Optional subtle background blob for glassmorphism effect */}
      <div className="absolute top-10 left-10 w-72 h-72 bg-primary-blue/10 dark:bg-primary-blue/20 rounded-full blur-3xl -z-10 animate-pulse pointer-events-none"></div>
      <div className="absolute bottom-10 right-10 w-96 h-96 bg-purple-500/10 dark:bg-purple-500/20 rounded-full blur-3xl -z-10 animate-pulse pointer-events-none animation-delay-2000"></div>

      {/* Header */}
      <div className="text-center shrink-0 mb-4 mt-2">
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">Intelligence Architecture</h1>
        <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mt-1.5">
          Algorithmic models, 12-phase technical pipeline, and consensus verification engine
        </p>
      </div>

      {/* Main Content Grid */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0 pb-4">
        
        {/* Left Column: Models */}
        <div className="lg:col-span-3 flex flex-col gap-2 bg-white/60 dark:bg-slate-900/50 backdrop-blur-xl border border-white/50 dark:border-slate-700/30 rounded-2xl p-3.5 shadow-premium overflow-hidden transition-all duration-500 hover:shadow-2xl">
          <h3 className="text-sm font-bold text-text-main dark:text-dark-text flex items-center gap-2 shrink-0 mb-0.5">
            <IoHardwareChipOutline className="text-primary-blue w-5 h-5" />
            Machine Learning Models
          </h3>
          <div className="flex-1 flex flex-col justify-between gap-1.5">
            {models.map((model, idx) => (
              <div key={idx} className="bg-white/50 dark:bg-slate-800/40 backdrop-blur-md rounded-xl px-3 py-2 border border-white/60 dark:border-slate-700/40 flex-1 flex flex-col justify-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:bg-white/90 dark:hover:bg-slate-700/70 hover:border-primary-blue/40 group">
                <h4 className="font-bold text-xs text-slate-900 dark:text-white group-hover:text-primary-blue transition-colors">{model.name}</h4>
                <p className="text-[10px] text-slate-700 dark:text-slate-300 font-medium leading-tight mt-0.5">{model.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Middle/Right Column */}
        <div className="lg:col-span-9 flex flex-col gap-4 min-h-0">
          
          {/* Top: 12 Phases */}
          <div className="flex-[2] bg-white/60 dark:bg-slate-900/50 backdrop-blur-xl border border-white/50 dark:border-slate-700/30 rounded-2xl p-4 shadow-premium overflow-hidden flex flex-col min-h-0 transition-all duration-500 hover:shadow-2xl">
            <h3 className="text-sm font-bold text-text-main dark:text-dark-text flex items-center gap-2 shrink-0 mb-3">
              <IoGitNetworkOutline className="text-purple-500 w-5 h-5" />
              12-Phase Processing Pipeline
            </h3>
            <div className="flex-1 grid grid-cols-2 md:grid-cols-4 grid-rows-3 gap-2.5 min-h-0">
              {phases.map((phase) => (
                <div key={phase.num} className="bg-white/50 dark:bg-slate-800/40 backdrop-blur-md rounded-xl p-2.5 border border-white/60 dark:border-slate-700/40 flex items-start gap-2.5 transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:bg-white/90 dark:hover:bg-slate-700/70 hover:border-purple-500/40 group">
                  <div className="px-1.5 py-0.5 bg-purple-500/10 text-purple-600 dark:text-purple-400 rounded text-[10px] font-bold shrink-0 transition-colors group-hover:bg-purple-500 group-hover:text-white">
                    {phase.num}
                  </div>
                  <div className="flex flex-col justify-center h-full min-w-0">
                    <h4 className="font-bold text-[11px] text-slate-900 dark:text-white truncate group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">{phase.title}</h4>
                    <p className="text-[10px] text-slate-700 dark:text-slate-300 font-medium leading-tight mt-0.5 line-clamp-2">{phase.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom: Consensus Engine */}
          <div className="flex-1 bg-white/60 dark:bg-slate-900/50 backdrop-blur-xl border border-white/50 dark:border-slate-700/30 rounded-2xl p-4 shadow-premium overflow-hidden flex flex-col min-h-0 transition-all duration-500 hover:shadow-2xl">
            <h3 className="text-sm font-bold text-text-main dark:text-dark-text flex items-center gap-2 shrink-0 mb-3">
              <IoShieldCheckmarkOutline className="text-teal-500 w-5 h-5" />
              Consensus Decision Engine
            </h3>
            <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-3 min-h-0">
              
              {/* Card 1 */}
              <div className="bg-success-green/10 dark:bg-success-green/5 backdrop-blur-md border border-success-green/20 rounded-xl p-3.5 flex flex-col justify-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:bg-success-green/20 dark:hover:bg-success-green/15 hover:border-success-green/50">
                <span className="text-[10px] font-extrabold text-success-green mb-1.5 tracking-wider">FULL AGREEMENT</span>
                <p className="text-xs text-slate-800 dark:text-slate-200 font-medium leading-snug">
                  Machine learning models and live web articles confirm the same label. Confidence scores are reinforced.
                </p>
              </div>

              {/* Card 2 */}
              <div className="bg-primary-blue/10 dark:bg-primary-blue/5 backdrop-blur-md border border-primary-blue/20 rounded-xl p-3.5 flex flex-col justify-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:bg-primary-blue/20 dark:hover:bg-primary-blue/15 hover:border-primary-blue/50">
                <span className="text-[10px] font-extrabold text-primary-blue mb-1.5 tracking-wider">API OVERRIDE</span>
                <p className="text-xs text-slate-800 dark:text-slate-200 font-medium leading-snug">
                  ML prediction contradicted by highly-similar factual article. Trusted web reference overrides ML model.
                </p>
              </div>

              {/* Card 3 */}
              <div className="bg-warning-orange/10 dark:bg-warning-orange/5 backdrop-blur-md border border-warning-orange/20 rounded-xl p-3.5 flex flex-col justify-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:bg-warning-orange/20 dark:hover:bg-warning-orange/15 hover:border-warning-orange/50">
                <span className="text-[10px] font-extrabold text-warning-orange mb-1.5 tracking-wider">ML FALLBACK</span>
                <p className="text-xs text-slate-800 dark:text-slate-200 font-medium leading-snug">
                  Web reference API calls return inconclusive search matches. System relies strictly on machine learning prediction.
                </p>
              </div>

            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

