/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#FFFFFF',
        'secondary-bg': '#F8FAFC',
        card: '#FFFFFF',
        border: '#E5E7EB',
        primary: {
          blue: '#2563EB',
          hover: '#1D4ED8',
        },
        success: {
          green: '#22C55E',
        },
        danger: {
          red: '#EF4444',
        },
        warning: {
          orange: '#F59E0B',
        },
        text: {
          main: '#111827',
          secondary: '#6B7280',
        },
        // Dark Mode custom palette
        dark: {
          bg: '#0F172A',
          'secondary-bg': '#1E293B',
          card: '#1E293B',
          border: '#334155',
          text: '#F8FAFC',
          'text-secondary': '#94A3B8',
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'premium': '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03)',
        'premium-lg': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
      },
      maxWidth: {
        'layout': '1440px',
      }
    },
  },
  plugins: [],
}
