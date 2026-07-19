import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { IoShieldCheckmark, IoGitNetworkOutline, IoSearchOutline, IoArrowForward } from 'react-icons/io5';

export default function Home() {
  return (
    <div className="flex flex-col justify-center items-center h-[calc(100vh-8rem)] max-w-5xl mx-auto text-center px-4 md:px-0">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full flex flex-col items-center"
      >
        {/* Banner badge */}
        <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-primary-blue/10 rounded-full border border-primary-blue/20 text-[11px] uppercase tracking-wider font-bold text-primary-blue mb-6 shadow-sm">
          <IoShieldCheckmark className="w-3 h-3" />
          <span>Consensus Engine v1.0.1 Active</span>
        </div>

        {/* Heading */}
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-text-main dark:text-dark-text mb-5 leading-tight">
          Uncover the Truth with <br className="hidden md:block" />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-blue to-teal-500">
            Consensus AI
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-base md:text-lg text-text-secondary dark:text-dark-text-secondary max-w-2xl mx-auto mb-10 leading-relaxed font-medium">
          Navigate the information age with confidence. Our system instantly cross-references thousands of active sources to separate fact from fiction.
        </p>

        {/* Action Button */}
        <div className="flex justify-center gap-4 mb-12">
          <Link
            to="/verify"
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary-blue to-blue-600 hover:from-blue-600 hover:to-primary-blue text-white text-sm font-semibold rounded-full shadow-lg hover:shadow-xl hover:shadow-primary-blue/30 transition-all hover:-translate-y-0.5"
          >
            Verify News Article
            <IoArrowForward className="w-4 h-4" />
          </Link>
          <Link
            to="/about"
            className="px-6 py-3 bg-white/50 dark:bg-dark-card/50 backdrop-blur-xl border border-white/60 dark:border-white/10 text-text-main dark:text-dark-text hover:bg-white/80 dark:hover:bg-dark-card/80 text-sm font-semibold rounded-full shadow-sm hover:shadow-md transition-all hover:-translate-y-0.5"
          >
            How It Works
          </Link>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-left w-full mt-2">
          <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-xl border border-white/60 dark:border-white/10 p-5 rounded-2xl shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] hover:shadow-[0_8px_32px_0_rgba(31,38,135,0.12)] transition-all hover:-translate-y-1 group relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-primary-blue/5 rounded-bl-full -mr-4 -mt-4 transition-transform group-hover:scale-110"></div>
            <div className="w-10 h-10 bg-primary-blue/10 text-primary-blue rounded-2xl flex items-center justify-center mb-4 relative z-10">
              <IoGitNetworkOutline className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-text-main dark:text-dark-text mb-1.5 relative z-10">ML Ensemble Models</h3>
            <p className="text-[13px] text-text-secondary dark:text-dark-text-secondary leading-relaxed relative z-10">
              Powered by SVM, Logistic Regression, Random Forest, and XGBoost models running in parallel to analyze linguistic patterns and predict truthfulness.
            </p>
          </div>

          <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-xl border border-white/60 dark:border-white/10 p-5 rounded-2xl shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] hover:shadow-[0_8px_32px_0_rgba(31,38,135,0.12)] transition-all hover:-translate-y-1 group relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-teal-500/5 rounded-bl-full -mr-4 -mt-4 transition-transform group-hover:scale-110"></div>
            <div className="w-10 h-10 bg-teal-500/10 text-teal-500 rounded-2xl flex items-center justify-center mb-4 relative z-10">
              <IoSearchOutline className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-text-main dark:text-dark-text mb-1.5 relative z-10">Live API Verification</h3>
            <p className="text-[13px] text-text-secondary dark:text-dark-text-secondary leading-relaxed relative z-10">
              Cross-references claims in real-time against Google Fact Check, News APIs, and active search indexes to validate information against trusted sources.
            </p>
          </div>

          <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-xl border border-white/60 dark:border-white/10 p-5 rounded-2xl shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] hover:shadow-[0_8px_32px_0_rgba(31,38,135,0.12)] transition-all hover:-translate-y-1 group relative overflow-hidden">
            <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-bl-full -mr-4 -mt-4 transition-transform group-hover:scale-110"></div>
            <div className="w-10 h-10 bg-emerald-500/10 text-emerald-500 rounded-2xl flex items-center justify-center mb-4 relative z-10">
              <IoShieldCheckmark className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-text-main dark:text-dark-text mb-1.5 relative z-10">Adaptive Feedback Loop</h3>
            <p className="text-[13px] text-text-secondary dark:text-dark-text-secondary leading-relaxed relative z-10">
              A continuous retraining pipeline that ingests user feedback and expert verdicts, allowing the system to rapidly adapt to emerging narratives and new data.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
