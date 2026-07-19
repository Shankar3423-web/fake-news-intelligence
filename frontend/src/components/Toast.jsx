import React from 'react';
import { motion } from 'framer-motion';
import { IoCheckmarkCircle, IoCloseCircle, IoAlertCircle, IoInformationCircle, IoClose } from 'react-icons/io5';

export default function Toast({ message, type, onClose }) {
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <IoCheckmarkCircle className="w-5 h-5 text-success-green" />;
      case 'error':
        return <IoCloseCircle className="w-5 h-5 text-danger-red" />;
      case 'warning':
        return <IoAlertCircle className="w-5 h-5 text-warning-orange" />;
      default:
        return <IoInformationCircle className="w-5 h-5 text-primary-blue" />;
    }
  };

  const getBorderColor = () => {
    switch (type) {
      case 'success':
        return 'border-success-green/20';
      case 'error':
        return 'border-danger-red/20';
      case 'warning':
        return 'border-warning-orange/20';
      default:
        return 'border-primary-blue/20';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className={`flex items-center justify-between w-full p-4 bg-white dark:bg-dark-card border ${getBorderColor()} rounded-xl shadow-premium`}
    >
      <div className="flex items-center gap-3">
        {getIcon()}
        <span className="text-sm font-medium text-text-main dark:text-dark-text">{message}</span>
      </div>
      <button
        onClick={onClose}
        className="p-1 text-text-secondary hover:text-text-main dark:text-dark-text-secondary dark:hover:text-dark-text rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
      >
        <IoClose className="w-4 h-4" />
      </button>
    </motion.div>
  );
}
