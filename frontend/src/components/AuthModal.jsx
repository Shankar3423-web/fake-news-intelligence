import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../hooks/useToast';

export default function AuthModal({ isOpen, onClose, onSuccess }) {
  const { login, signup } = useAuth();
  const { addToast } = useToast();
  
  const [isLoginView, setIsLoginView] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  
  // Form state
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      if (isLoginView) {
        await login({ email, password });
        addToast('Successfully logged in!', 'success');
      } else {
        await signup({ username: name, email, password });
        addToast('Account created successfully!', 'success');
      }
      onSuccess && onSuccess();
      onClose();
    } catch (err) {
      addToast(err.message || 'Authentication failed', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleView = () => {
    setIsLoginView(!isLoginView);
    // Reset fields on toggle
    setPassword('');
    if (!isLoginView) setName('');
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] flex items-center justify-center px-4 bg-slate-900/40 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="w-full max-w-md bg-white dark:bg-dark-card rounded-2xl shadow-2xl overflow-hidden border border-border dark:border-dark-border"
        >
          {/* Header */}
          <div className="relative p-6 pb-4 border-b border-border dark:border-dark-border bg-slate-50/50 dark:bg-slate-900/50">
            <button 
              onClick={onClose}
              className="absolute top-6 right-6 text-text-secondary hover:text-text-main dark:text-dark-text-secondary dark:hover:text-dark-text transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <h2 className="text-xl font-extrabold text-text-main dark:text-dark-text tracking-tight">
              {isLoginView ? 'Welcome Back' : 'Create an Account'}
            </h2>
            <p className="text-sm text-text-secondary dark:text-dark-text-secondary mt-1">
              {isLoginView 
                ? 'Sign in to verify news and access your history.' 
                : 'Sign up to start verifying news with AI consensus.'}
            </p>
          </div>

          {/* Form */}
          <div className="p-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              
              {!isLoginView && (
                <div>
                  <label className="block text-xs font-semibold text-text-main dark:text-dark-text mb-1.5">
                    Full Name
                  </label>
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-border dark:border-dark-border rounded-xl focus:outline-none focus:ring-1 focus:ring-primary-blue text-sm text-text-main dark:text-dark-text"
                    placeholder="John Doe"
                  />
                </div>
              )}

              <div>
                <label className="block text-xs font-semibold text-text-main dark:text-dark-text mb-1.5">
                  Email Address
                </label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-border dark:border-dark-border rounded-xl focus:outline-none focus:ring-1 focus:ring-primary-blue text-sm text-text-main dark:text-dark-text"
                  placeholder="you@example.com"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-text-main dark:text-dark-text mb-1.5">
                  Password
                </label>
                <input
                  type="password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-50 dark:bg-slate-900 border border-border dark:border-dark-border rounded-xl focus:outline-none focus:ring-1 focus:ring-primary-blue text-sm text-text-main dark:text-dark-text"
                  placeholder="••••••••"
                />
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full mt-2 px-5 py-3 bg-primary-blue hover:bg-primary-hover disabled:bg-primary-blue/50 text-white font-bold rounded-xl text-sm transition-all shadow-sm flex justify-center items-center h-[46px]"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  isLoginView ? 'Sign In' : 'Sign Up'
                )}
              </button>
            </form>

            <div className="mt-6 text-center text-sm text-text-secondary dark:text-dark-text-secondary">
              {isLoginView ? "Don't have an account? " : "Already have an account? "}
              <button 
                onClick={toggleView}
                className="font-bold text-primary-blue hover:text-primary-hover transition-colors"
              >
                {isLoginView ? 'Sign Up' : 'Log In'}
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
