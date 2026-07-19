import React from 'react';
import { motion } from 'framer-motion';
import { 
  IoDocumentTextOutline, 
  IoCodeWorkingOutline, 
  IoBuildOutline, 
  IoHardwareChipOutline, 
  IoSearchOutline, 
  IoGitCompareOutline, 
  IoRibbonOutline, 
  IoCheckmarkCircle,
  IoCheckmark
} from 'react-icons/io5';

const PIPELINE_STEPS = [
  { name: 'Input Received', icon: <IoDocumentTextOutline className="w-5 h-5" /> },
  { name: 'NLP Processing', icon: <IoCodeWorkingOutline className="w-5 h-5" /> },
  { name: 'Feature Engineering', icon: <IoBuildOutline className="w-5 h-5" /> },
  { name: 'SVM Prediction', icon: <IoHardwareChipOutline className="w-5 h-5" /> },
  { name: 'NewsAPI Search', icon: <IoSearchOutline className="w-5 h-5" /> },
  { name: 'Similarity Matching', icon: <IoGitCompareOutline className="w-5 h-5" /> },
  { name: 'Trusted Sources', icon: <IoRibbonOutline className="w-5 h-5" /> },
  { name: 'Final Verdict', icon: <IoCheckmarkCircle className="w-5 h-5" /> },
];

export default function PredictionPipeline({ status, activeStep, layout = 'horizontal' }) {
  const isRunning = status === 'running';
  const isCompleted = status === 'completed';
  const isVertical = layout === 'vertical';

  return (
    <div className={`w-full bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-6 shadow-premium ${!isVertical ? 'overflow-x-auto' : ''}`}>
      <div className="mb-4">
        <h3 className="text-sm font-bold text-text-main dark:text-dark-text">Process Execution Pipeline</h3>
        <p className="text-xs text-text-secondary dark:text-dark-text-secondary">
          Track real-time AI consensus analysis step-by-step
        </p>
      </div>

      <div className={`flex ${isVertical ? 'flex-col min-h-full py-2' : 'items-center min-w-[900px] justify-between py-6 px-4'}`}>
        {PIPELINE_STEPS.map((step, idx) => {
          const isStepCompleted = isCompleted || (isRunning && idx < activeStep);
          const isStepActive = isRunning && idx === activeStep;
          
          return (
            <React.Fragment key={step.name}>
              {/* Node Item */}
              <div className={`flex relative z-10 ${isVertical ? 'flex-row items-center w-full gap-4' : 'flex-col items-center w-24'}`}>
                <motion.div
                  animate={isStepActive ? { scale: [1, 1.1, 1] } : {}}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                  className={`flex-shrink-0 flex items-center justify-center w-12 h-12 rounded-full border-2 transition-colors duration-300 relative ${
                    isStepCompleted
                      ? 'bg-success-green border-success-green text-white'
                      : isStepActive
                      ? 'bg-white dark:bg-dark-secondary-bg border-primary-blue text-primary-blue shadow-md'
                      : 'bg-white dark:bg-dark-secondary-bg border-slate-200 dark:border-dark-border text-slate-400 dark:text-slate-600'
                  }`}
                >
                  {isStepCompleted ? (
                    <IoCheckmark className="w-6 h-6 stroke-[3px]" />
                  ) : (
                    step.icon
                  )}

                  {/* Ripple Effect for active step */}
                  {isStepActive && (
                    <span className="absolute inset-0 rounded-full border-2 border-primary-blue animate-ping opacity-75" />
                  )}
                </motion.div>

                {/* Step Name */}
                <span
                  className={`text-[10px] font-semibold uppercase tracking-wider transition-colors duration-300 ${
                    isVertical ? 'text-left' : 'mt-3 text-center'
                  } ${
                    isStepCompleted
                      ? 'text-success-green'
                      : isStepActive
                      ? 'text-primary-blue font-bold'
                      : 'text-text-secondary dark:text-dark-text-secondary'
                  }`}
                >
                  {step.name}
                </span>
              </div>

              {/* Progress Line Connector */}
              {idx < PIPELINE_STEPS.length - 1 && (
                <div 
                  className={`${
                    isVertical 
                      ? 'w-[2px] min-h-[30px] my-2 ml-[23px] flex-shrink-0' 
                      : 'flex-1 h-[2px] mx-2'
                  } bg-slate-100 dark:bg-dark-border relative`}
                >
                  <div
                    className={`absolute ${isVertical ? 'top-0 left-0 w-full' : 'top-0 left-0 h-full'} transition-all duration-500 ${
                      isStepCompleted 
                        ? (isVertical ? 'bg-success-green h-full' : 'bg-success-green w-full')
                        : isStepActive 
                        ? (isVertical ? 'bg-primary-blue/30 h-1/2' : 'bg-primary-blue/30 w-1/2')
                        : (isVertical ? 'bg-transparent h-0' : 'bg-transparent w-0')
                    }`}
                  />
                  {isStepActive && (
                    <div className={`absolute top-0 left-0 ${isVertical ? 'w-full h-full' : 'w-full h-full'} bg-gradient-to-${isVertical ? 'b' : 'r'} from-success-green to-primary-blue animate-pulse`} />
                  )}
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
