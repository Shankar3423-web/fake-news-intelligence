import React, { useState, useRef, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme.jsx';
import { useAuth } from '../context/AuthContext';
import { IoMoonOutline, IoSunnyOutline, IoShieldCheckmark, IoPersonCircleOutline, IoMenu, IoLogOutOutline } from 'react-icons/io5';
import AuthModal from './AuthModal';

export default function Navbar({ toggleSidebar }) {
  const location = useLocation();
  const { isDark, toggleTheme } = useTheme();
  const { user, logout } = useAuth();
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowProfileMenu(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Verify News', path: '/verify' },
    { name: 'Dashboard', path: '/dashboard' },
    { name: 'Statistics', path: '/statistics' },
    { name: 'About', path: '/about' },
  ];

  const isActive = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <>
    <header className="sticky top-4 z-40 mx-4 md:mx-8 mt-4 rounded-full bg-white/60 dark:bg-dark-card/60 backdrop-blur-xl border border-white/60 dark:border-dark-border shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] hover:shadow-[0_8px_32px_0_rgba(31,38,135,0.12)] transition-all duration-300">
      <div className="flex h-14 items-center justify-between px-4 md:px-6 max-w-layout mx-auto">
        
        {/* Left section: Logo and Mobile Menu toggle */}
        <div className="flex items-center gap-4">
          <button
            onClick={toggleSidebar}
            className="p-2 md:hidden hover:bg-slate-100 dark:hover:bg-slate-800 text-text-main dark:text-dark-text rounded-lg"
          >
            <IoMenu className="w-6 h-6" />
          </button>
          
          <Link to="/" className="flex items-center gap-2">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary-blue text-white shadow-sm">
              <IoShieldCheckmark className="w-5 h-5" />
            </div>
            <span className="font-bold text-base md:text-lg tracking-tight text-text-main dark:text-dark-text">
              Fake News Intelligence
            </span>
          </Link>
        </div>

        {/* Center: Navigation Links (hidden on mobile) */}
        <nav className="hidden md:flex items-center gap-2">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`text-sm font-medium transition-all px-4 py-2 rounded-full ${
                isActive(item.path)
                  ? 'bg-primary-blue text-white shadow-md'
                  : 'text-text-secondary dark:text-dark-text-secondary hover:text-primary-blue hover:bg-primary-blue/10'
              }`}
            >
              {item.name}
            </Link>
          ))}
        </nav>

        {/* Right: Actions */}
        <div className="flex items-center gap-4">
          {/* Dark Mode Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 text-text-secondary hover:text-text-main dark:text-dark-text-secondary dark:hover:text-dark-text rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            title="Toggle theme"
          >
            {isDark ? <IoSunnyOutline className="w-5 h-5" /> : <IoMoonOutline className="w-5 h-5" />}
          </button>

          {/* Profile link/dropdown */}
          <div className="relative" ref={menuRef}>
            <button
              onClick={() => setShowProfileMenu(!showProfileMenu)}
              className="p-1 text-text-secondary hover:text-text-main dark:text-dark-text-secondary dark:hover:text-dark-text rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors focus:outline-none"
              title="Profile"
            >
              {user ? (
                <div className="w-7 h-7 rounded-full bg-primary-blue flex items-center justify-center text-white font-bold text-xs">
                  {user.username.charAt(0).toUpperCase()}
                </div>
              ) : (
                <IoPersonCircleOutline className="w-7 h-7" />
              )}
            </button>
            
            {showProfileMenu && user && (
              <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-xl shadow-premium overflow-hidden z-50 animate-fadeIn origin-top-right">
                <div className="px-4 py-3 border-b border-border dark:border-dark-border bg-slate-50/50 dark:bg-slate-900/30">
                  <p className="text-sm font-bold text-text-main dark:text-dark-text truncate">{user.username}</p>
                  <p className="text-xs text-text-secondary dark:text-dark-text-secondary truncate">{user.email}</p>
                </div>
                <div className="p-2">
                  <button
                    onClick={() => {
                      logout();
                      setShowProfileMenu(false);
                    }}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm font-semibold text-danger-red hover:bg-danger-red/10 rounded-lg transition-colors"
                  >
                    <IoLogOutOutline className="w-4 h-4" />
                    Log Out
                  </button>
                </div>
              </div>
            )}
            
            {showProfileMenu && !user && (
              <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-xl shadow-premium overflow-hidden z-50 animate-fadeIn origin-top-right">
                <div className="p-3 text-center">
                  <p className="text-sm text-text-secondary dark:text-dark-text-secondary mb-3">You are not logged in.</p>
                  <button
                    onClick={() => {
                      setShowProfileMenu(false);
                      setIsAuthModalOpen(true);
                    }}
                    className="block w-full py-2 bg-primary-blue text-white text-sm font-semibold rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Sign In
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
    <AuthModal 
      isOpen={isAuthModalOpen} 
      onClose={() => setIsAuthModalOpen(false)} 
    />
    </>
  );
}
