import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { modelService } from '../services/modelService';
import { 
  IoHomeOutline, 
  IoShieldCheckmarkOutline, 
  IoBarChartOutline, 
  IoTimeOutline, 
  IoAnalyticsOutline, 
  IoChatbubbleEllipsesOutline,
  IoInformationCircleOutline,
  IoClose,
  IoPulseOutline,
  IoLockClosedOutline
} from 'react-icons/io5';

export default function Sidebar({ isOpen, setIsOpen }) {
  const location = useLocation();
  const { user, logout } = useAuth();
  const [modelInfo, setModelInfo] = useState({
    version: '1.0.1',
    accuracy: 0.885,
    trainingDate: '2026-07-19',
    status: 'loading'
  });

  useEffect(() => {
    // Fetch current model info from the database/api
    modelService.getModelInfo()
      .then((data) => {
        setModelInfo({
          version: data.version || '1.0.1',
          accuracy: data.accuracy || 0.885,
          trainingDate: data.training_date ? data.training_date.split('T')[0] : '2026-07-19',
          status: 'online'
        });
      })
      .catch((err) => {
        console.error('Failed to load sidebar model info:', err);
        setModelInfo(prev => ({ ...prev, status: 'offline' }));
      });
  }, [location.pathname]); // Reload stats occasionally when page routing updates

  const sidebarLinks = [
    { name: 'Home', path: '/', icon: <IoHomeOutline className="w-5 h-5" /> },
    { name: 'Verify News', path: '/verify', icon: <IoShieldCheckmarkOutline className="w-5 h-5" /> },
    { name: 'Dashboard', path: '/dashboard', icon: <IoBarChartOutline className="w-5 h-5" /> },
    { name: 'History', path: '/history', icon: <IoTimeOutline className="w-5 h-5" /> },
    { name: 'Statistics', path: '/statistics', icon: <IoAnalyticsOutline className="w-5 h-5" /> },
    { name: 'Feedback', path: '/feedback', icon: <IoChatbubbleEllipsesOutline className="w-5 h-5" /> },
    { name: 'About', path: '/about', icon: <IoInformationCircleOutline className="w-5 h-5" /> },
  ];

  return (
    <>
      {/* Mobile Sidebar Backdrop overlay */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-40 bg-slate-900/20 backdrop-blur-sm md:hidden"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-border dark:border-dark-border bg-white dark:bg-dark-card transition-transform duration-300 md:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Mobile close button */}
        <div className="flex h-16 items-center justify-between px-6 border-b border-border dark:border-dark-border md:hidden">
          <span className="font-bold text-text-main dark:text-dark-text">Navigation</span>
          <button
            onClick={() => setIsOpen(false)}
            className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg text-text-secondary dark:text-dark-text-secondary"
          >
            <IoClose className="w-6 h-6" />
          </button>
        </div>

        {/* Sidebar Nav Links */}
        <nav className="flex-1 space-y-1.5 px-4 py-6 overflow-y-auto">
          {sidebarLinks.map((link) => (
            <NavLink
              key={link.path}
              to={link.path}
              onClick={() => setIsOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-xl transition-all ${
                  isActive
                    ? 'bg-slate-100 dark:bg-dark-secondary-bg text-primary-blue font-semibold shadow-sm'
                    : 'text-text-secondary dark:text-dark-text-secondary hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-text-main dark:hover:text-dark-text'
                }`
              }
            >
              {link.icon}
              {link.name}
            </NavLink>
          ))}
        </nav>

        {/* Bottom System Status Card */}
        <div className="p-4 border-t border-border dark:border-dark-border bg-slate-50/50 dark:bg-slate-900/30 space-y-4">
          
          {/* User Account Info */}
          {user && (
            <div className="p-3 bg-white dark:bg-dark-secondary-bg border border-border dark:border-dark-border rounded-xl shadow-sm flex flex-col gap-2">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-primary-blue text-white flex items-center justify-center font-bold">
                  {user.username.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 overflow-hidden">
                  <p className="text-sm font-bold text-text-main dark:text-dark-text truncate">{user.username}</p>
                  <p className="text-xs text-text-secondary dark:text-dark-text-secondary truncate">{user.email}</p>
                </div>
              </div>
              <button 
                onClick={logout}
                className="w-full mt-1 py-1.5 px-3 text-xs font-semibold text-danger-red hover:bg-danger-red/10 rounded-lg transition-colors border border-danger-red/20"
              >
                Log Out
              </button>
            </div>
          )}

          <div className="p-4 bg-white dark:bg-dark-secondary-bg border border-border dark:border-dark-border rounded-2xl shadow-premium">
            <div className="flex items-center gap-2 mb-3">
              <IoPulseOutline className="w-5 h-5 text-primary-blue animate-pulse" />
              <span className="text-xs font-semibold uppercase tracking-wider text-text-main dark:text-dark-text">
                System Status
              </span>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-text-secondary dark:text-dark-text-secondary">Model Version</span>
                <span className="font-semibold text-text-main dark:text-dark-text">{modelInfo.version}</span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-text-secondary dark:text-dark-text-secondary">Accuracy</span>
                <span className="font-semibold text-success-green">{(modelInfo.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center text-xs border-t border-slate-100 dark:border-dark-border/40 pt-2 mt-1">
                <span className="text-text-secondary dark:text-dark-text-secondary">Last Retrained</span>
                <span className="font-mono text-[10px] text-text-main dark:text-dark-text">{modelInfo.trainingDate}</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
