const KEY = 'crowd_detection_theme_v1'

export function getTheme() {
  const raw = localStorage.getItem(KEY)
  return raw === 'light' || raw === 'dark' ? raw : 'dark'
}

export function setTheme(theme) {
  const t = theme === 'light' ? 'light' : 'dark'
  localStorage.setItem(KEY, t)
  document.documentElement.classList.toggle('dark', t === 'dark')
}

export function initTheme() {
  setTheme(getTheme())
}
