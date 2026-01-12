/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      boxShadow: {
        glass: '0 18px 55px rgba(0,0,0,0.35)',
      },
    },
  },
  plugins: [],
}

