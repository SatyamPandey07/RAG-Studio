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
        background: 'rgba(5, 8, 16, 1)',
        card: 'rgba(10, 15, 30, 0.45)',
        border: 'rgba(255, 255, 255, 0.08)',
        teal: {
          500: '#06b6d4',
          600: '#0891b2',
        },
        violet: {
          500: '#8b5cf6',
          600: '#7c3aed',
        }
      },
      boxShadow: {
        elegant: '0 20px 40px rgba(0, 0, 0, 0.4)',
        glow: '0 0 20px rgba(6, 182, 212, 0.3)',
      }
    },
  },
  plugins: [],
}
