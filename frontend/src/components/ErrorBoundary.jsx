import React from 'react';
import { IoWarning } from 'react-icons/io5';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center p-8 bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl shadow-premium max-w-xl mx-auto my-12 text-center">
          <div className="p-3 bg-red-50 dark:bg-red-950/30 rounded-2xl mb-4">
            <IoWarning className="w-8 h-8 text-danger-red" />
          </div>
          <h2 className="text-xl font-bold text-text-main dark:text-dark-text mb-2">Something went wrong</h2>
          <p className="text-sm text-text-secondary dark:text-dark-text-secondary mb-6">
            An error occurred while rendering this component. Please try refreshing the page or contact support if the issue persists.
          </p>
          <div className="w-full text-left bg-slate-50 dark:bg-slate-900/50 p-4 rounded-xl border border-slate-100 dark:border-slate-800 font-mono text-xs overflow-x-auto text-danger-red mb-6 max-h-40">
            {this.state.error?.toString() || 'Unknown Error'}
          </div>
          <button
            onClick={() => window.location.reload()}
            className="px-5 py-2.5 bg-primary-blue hover:bg-primary-hover text-white font-semibold rounded-xl transition-colors shadow-sm"
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
