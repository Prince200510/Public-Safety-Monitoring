import { useEffect, useState } from 'react'
import { getTheme, setTheme } from '../auth/theme.js'

export default function ThemeToggle() {
  const [theme, setThemeState] = useState(getTheme())

  useEffect(() => {
    setTheme(theme)
  }, [theme])

  return (
    <button type="button" className="rounded-xl border border-slate-200 bg-white/70 px-3 py-2 text-sm font-semibold text-slate-800 shadow-sm backdrop-blur hover:bg-white dark:border-white/10 dark:bg-white/10 dark:text-slate-100 dark:hover:bg-white/15" onClick={() => setThemeState((t) => (t === 'dark' ? 'light' : 'dark'))} title="Toggle theme">{theme === 'dark' ? 'Light mode' : 'Dark mode'}</button>
  )
}
