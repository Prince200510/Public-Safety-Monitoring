import { Navigate, Route, Routes } from 'react-router-dom'
import Login from './pages/Login.jsx'
import UserDashboard from './pages/UserDashboard.jsx'
import PoliceDashboard from './pages/PoliceDashboard.jsx'
import { getSession } from './auth/session.js'

function RequireRole({ role, children }) {
  const session = getSession()
  if (!session) return <Navigate to="/" replace />
  if (role && session.role !== role) return <Navigate to="/" replace />
  return children
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/user" element={ <RequireRole role="user"><UserDashboard /></RequireRole>} />
      <Route path="/police" element={ <RequireRole role="police"><PoliceDashboard /></RequireRole> } />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
