const KEY = 'crowd_detection_session_v1'

export function setSession(session) {
  localStorage.setItem(KEY, JSON.stringify(session))
}

export function getSession() {
  const raw = localStorage.getItem(KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

export function clearSession() {
  localStorage.removeItem(KEY)
}
