import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { predictionService } from '../services/predictionService';
import { useToast } from '../hooks/useToast';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart, 
  Pie, 
  Cell, 
  Legend 
} from 'recharts';
import { 
  IoShieldCheckmarkOutline, 
  IoAlertCircleOutline, 
  IoShieldHalfOutline, 
  IoSpeedometerOutline,
  IoArrowForwardOutline,
  IoPulseOutline,
  IoTimeOutline
} from 'react-icons/io5';

export default function Dashboard() {
  const { addToast } = useToast();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total_predictions: 0,
    real_count: 0,
    fake_count: 0,
    average_confidence: 0,
    system_accuracy: 0.88,
    label_distribution: [],
    recent_activity: [],
    timeline: []
  });

  useEffect(() => {
    predictionService.getStats()
      .then((data) => {
        setStats(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load dashboard stats:', err);
        addToast('Could not load statistics from database.', 'error');
        setLoading(false);
      });
  }, []);

  const COLORS = ['#22C55E', '#EF4444'];

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse max-w-layout mx-auto">
        <div className="h-10 bg-slate-200 dark:bg-slate-800 rounded-xl w-48" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-24 bg-slate-200 dark:bg-slate-800 rounded-2xl" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 h-64 bg-slate-200 dark:bg-slate-800 rounded-2xl" />
          <div className="h-64 bg-slate-200 dark:bg-slate-800 rounded-2xl" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-layout mx-auto pb-4">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-gradient-to-r from-primary-blue/10 to-transparent p-5 rounded-2xl border border-primary-blue/20">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-text-main dark:text-dark-text flex items-center gap-2">
            <IoPulseOutline className="w-6 h-6 text-primary-blue" />
            System Dashboard
          </h1>
          <p className="text-xs text-text-secondary dark:text-dark-text-secondary mt-1">
            Real-time insights and performance metrics for the intelligence pipeline
          </p>
        </div>
        <Link
          to="/verify"
          className="inline-flex items-center gap-1.5 px-4 py-2 bg-primary-blue hover:bg-primary-hover text-white text-xs font-semibold rounded-xl shadow-md hover:shadow-lg transition-all duration-300 transform hover:-translate-y-0.5"
        >
          Verify New Article
          <IoArrowForwardOutline className="w-4 h-4" />
        </Link>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* Total Verifications */}
        <div className="relative overflow-hidden bg-white dark:bg-dark-card border border-border dark:border-dark-border p-5 rounded-2xl shadow-sm hover:shadow-md transition-all duration-300 group">
          <div className="absolute top-0 right-0 -mr-4 -mt-4 w-20 h-20 rounded-full bg-primary-blue/10 blur-xl group-hover:bg-primary-blue/20 transition-colors duration-500" />
          <div className="flex items-center justify-between mb-3 relative z-10">
            <span className="text-[11px] font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider">
              Total Checked
            </span>
            <div className="p-1.5 bg-primary-blue/10 rounded-lg">
              <IoShieldHalfOutline className="w-4 h-4 text-primary-blue" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-text-main dark:text-dark-text font-mono relative z-10">
            {stats.total_predictions.toLocaleString()}
          </div>
          <p className="text-[10px] text-text-secondary dark:text-dark-text-secondary mt-1.5 relative z-10">
            Verified claims in database
          </p>
        </div>

        {/* Real Claims */}
        <div className="relative overflow-hidden bg-white dark:bg-dark-card border border-border dark:border-dark-border p-5 rounded-2xl shadow-sm hover:shadow-md transition-all duration-300 group">
          <div className="absolute top-0 right-0 -mr-4 -mt-4 w-20 h-20 rounded-full bg-success-green/10 blur-xl group-hover:bg-success-green/20 transition-colors duration-500" />
          <div className="flex items-center justify-between mb-3 relative z-10">
            <span className="text-[11px] font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider">
              Verified Real
            </span>
            <div className="p-1.5 bg-success-green/10 rounded-lg">
              <IoShieldCheckmarkOutline className="w-4 h-4 text-success-green" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-success-green font-mono relative z-10">
            {stats.real_count.toLocaleString()}
          </div>
          <p className="text-[10px] text-text-secondary dark:text-dark-text-secondary mt-1.5 relative z-10">
            Confirmed factual narratives
          </p>
        </div>

        {/* Fake Claims */}
        <div className="relative overflow-hidden bg-white dark:bg-dark-card border border-border dark:border-dark-border p-5 rounded-2xl shadow-sm hover:shadow-md transition-all duration-300 group">
          <div className="absolute top-0 right-0 -mr-4 -mt-4 w-20 h-20 rounded-full bg-danger-red/10 blur-xl group-hover:bg-danger-red/20 transition-colors duration-500" />
          <div className="flex items-center justify-between mb-3 relative z-10">
            <span className="text-[11px] font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider">
              Verified Fake
            </span>
            <div className="p-1.5 bg-danger-red/10 rounded-lg">
              <IoAlertCircleOutline className="w-4 h-4 text-danger-red" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-danger-red font-mono relative z-10">
            {stats.fake_count.toLocaleString()}
          </div>
          <p className="text-[10px] text-text-secondary dark:text-dark-text-secondary mt-1.5 relative z-10">
            Misinformation debunked
          </p>
        </div>

        {/* System Accuracy */}
        <div className="relative overflow-hidden bg-white dark:bg-dark-card border border-border dark:border-dark-border p-5 rounded-2xl shadow-sm hover:shadow-md transition-all duration-300 group">
          <div className="absolute top-0 right-0 -mr-4 -mt-4 w-20 h-20 rounded-full bg-warning-orange/10 blur-xl group-hover:bg-warning-orange/20 transition-colors duration-500" />
          <div className="flex items-center justify-between mb-3 relative z-10">
            <span className="text-[11px] font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider">
              System Accuracy
            </span>
            <div className="p-1.5 bg-warning-orange/10 rounded-lg">
              <IoSpeedometerOutline className="w-4 h-4 text-warning-orange" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-text-main dark:text-dark-text font-mono relative z-10">
            {(stats.system_accuracy * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-text-secondary dark:text-dark-text-secondary mt-1.5 relative z-10">
            Based on approved feedback
          </p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Trend Area Chart (Left) */}
        <div className="lg:col-span-2 bg-white dark:bg-dark-card border border-border dark:border-dark-border p-5 rounded-2xl shadow-sm flex flex-col">
          <div className="mb-4">
            <h3 className="text-sm font-bold text-text-main dark:text-dark-text">Prediction Trends</h3>
            <p className="text-[11px] text-text-secondary dark:text-dark-text-secondary">
              Confidence tracking of recently analyzed articles
            </p>
          </div>
          
          <div className="h-48 flex-grow">
            {stats.timeline.length === 0 ? (
              <div className="h-full flex items-center justify-center text-xs text-text-secondary dark:text-dark-text-secondary bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-dashed border-slate-200 dark:border-slate-700">
                No historical records available to map charts.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={stats.timeline} margin={{ top: 5, right: 0, left: -25, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563EB" stopOpacity={0.25}/>
                      <stop offset="95%" stopColor="#2563EB" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" className="dark:stroke-slate-800" />
                  <XAxis dataKey="date" stroke="#94A3B8" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 100]} stroke="#94A3B8" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                      borderRadius: '12px',
                      borderColor: '#E2E8F0',
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
                      padding: '8px 12px'
                    }} 
                    itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
                    labelStyle={{ fontSize: '11px', color: '#64748B', marginBottom: '4px' }}
                  />
                  <Area type="monotone" dataKey="confidence" name="Confidence Score" stroke="#2563EB" strokeWidth={2.5} fillOpacity={1} fill="url(#colorConfidence)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Label Distribution Pie Chart (Right) */}
        <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border p-5 rounded-2xl shadow-sm flex flex-col">
          <div>
            <h3 className="text-sm font-bold text-text-main dark:text-dark-text">Label Distribution</h3>
            <p className="text-[11px] text-text-secondary dark:text-dark-text-secondary mb-2">
              Ratio of Real to Fake articles
            </p>
          </div>
          
          <div className="flex-grow relative flex items-center justify-center min-h-[160px]">
            {stats.total_predictions === 0 ? (
              <div className="text-[11px] text-text-secondary dark:text-dark-text-secondary bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-dashed border-slate-200 dark:border-slate-700 w-full h-full flex items-center justify-center">
                No distribution data.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={stats.label_distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={65}
                    paddingAngle={4}
                    dataKey="value"
                    stroke="none"
                  >
                    {stats.label_distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} className="hover:opacity-80 transition-opacity outline-none" />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', fontSize: '12px' }}
                    itemStyle={{ color: '#1E293B', fontWeight: 'bold' }}
                  />
                  <Legend verticalAlign="bottom" height={24} iconType="circle" iconSize={6} formatter={(value) => <span className="text-[11px] font-semibold text-text-main dark:text-dark-text">{value}</span>} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Recent Activity Section */}
      <div className="bg-white dark:bg-dark-card border border-border dark:border-dark-border rounded-2xl p-5 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <IoTimeOutline className="w-5 h-5 text-text-secondary dark:text-dark-text-secondary" />
            <div>
              <h3 className="text-sm font-bold text-text-main dark:text-dark-text">Recent Activity</h3>
              <p className="text-[11px] text-text-secondary dark:text-dark-text-secondary">
                Latest verifications processed by the system
              </p>
            </div>
          </div>
          <Link 
            to="/history" 
            className="text-[11px] font-semibold text-primary-blue hover:text-primary-hover flex items-center gap-1 bg-primary-blue/5 hover:bg-primary-blue/10 px-3 py-1.5 rounded-lg transition-colors"
          >
            View All
            <IoArrowForwardOutline className="w-3 h-3" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse whitespace-nowrap">
            <thead>
              <tr className="border-b border-slate-100 dark:border-dark-border/60 text-[10px] font-bold text-text-secondary dark:text-dark-text-secondary uppercase tracking-wider bg-slate-50/50 dark:bg-slate-800/30">
                <th className="py-2.5 px-4 rounded-tl-lg">Title / Snippet</th>
                <th className="py-2.5 px-4">Verdict</th>
                <th className="py-2.5 px-4">Confidence</th>
                <th className="py-2.5 px-4 rounded-tr-lg text-right">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-dark-border/40 text-xs">
              {stats.recent_activity.length === 0 ? (
                <tr>
                  <td colSpan="4" className="py-6 text-center text-[11px] text-text-secondary dark:text-dark-text-secondary">
                    No recent predictions found.
                  </td>
                </tr>
              ) : (
                stats.recent_activity.slice(0, 4).map((act) => (
                  <tr key={act.id} className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors group">
                    <td className="py-2.5 px-4 font-medium text-text-main dark:text-dark-text max-w-[200px] md:max-w-md truncate">
                      {act.title}
                    </td>
                    <td className="py-2.5 px-4">
                      <span className={`inline-flex items-center justify-center px-2 py-0.5 rounded-md text-[9px] font-bold uppercase tracking-wide ${
                        act.label === 'REAL'
                          ? 'bg-success-green/10 text-success-green border border-success-green/20'
                          : 'bg-danger-red/10 text-danger-red border border-danger-red/20'
                      }`}>
                        {act.label}
                      </span>
                    </td>
                    <td className="py-2.5 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-12 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${act.label === 'REAL' ? 'bg-success-green' : 'bg-danger-red'}`} 
                            style={{ width: `${act.confidence * 100}%` }}
                          />
                        </div>
                        <span className="font-mono font-semibold text-text-main dark:text-dark-text text-[11px]">
                          {(act.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-4 text-[10px] text-text-secondary dark:text-dark-text-secondary text-right">
                      {act.time ? new Date(act.time).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'N/A'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

