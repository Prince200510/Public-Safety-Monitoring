export const DEFAULTS = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  defaultLocation: import.meta.env.VITE_DEFAULT_LOCATION || 'Kandavli',
  forceLocation: String(import.meta.env.VITE_FORCE_LOCATION || 'true') === 'true',
  police: {
    email: import.meta.env.VITE_POLICE_EMAIL || '',
    policeId: import.meta.env.VITE_POLICE_ID || '',
    password: import.meta.env.VITE_POLICE_PASSWORD || '',
  },
  user: {
    email: import.meta.env.VITE_USER_EMAIL || '',
    password: import.meta.env.VITE_USER_PASSWORD || '',
  },
}
