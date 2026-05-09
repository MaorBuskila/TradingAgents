import { Routes, Route, NavLink } from 'react-router-dom'
import { PlayCircle, FileText, PieChart, ListOrdered, Video, FlaskConical, Activity } from 'lucide-react'
import RunAnalysis from './pages/RunAnalysis'
import Reports from './pages/Reports'
import Portfolio from './portfolio'
import Watchlist from './pages/Watchlist'
import YouTubeSummary from './pages/YouTubeSummary'
import RSILab from './pages/RSILab'
import MACDLab from './pages/MACDLab'

const nav = [
  { to: '/', label: 'Run analysis', icon: PlayCircle, end: true },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/portfolio', label: 'Portfolio', icon: PieChart },
  { to: '/watchlist', label: 'Watchlist', icon: ListOrdered },
  { to: '/youtube', label: 'YouTube', icon: Video },
  { to: '/rsi-lab', label: 'RSI Lab', icon: FlaskConical },
  { to: '/macd-lab', label: 'MACD Lab', icon: Activity },
]

function App() {
  return (
    <div className="app">
      <nav className="sidebar" aria-label="Main">
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <span className="sidebar-logo-mark" aria-hidden>
              TA
            </span>
            TradingAgents
          </div>
          <p className="sidebar-tagline">Research &amp; portfolio workspace</p>
        </div>
        <ul className="nav-links">
          {nav.map(({ to, label, icon: Icon, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
                title={label}
              >
                <Icon size={20} strokeWidth={2} aria-hidden />
                <span className="nav-label">{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      <main className="content">
        <Routes>
          <Route path="/" element={<RunAnalysis />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/portfolio" element={<Portfolio />} />
          <Route path="/watchlist" element={<Watchlist />} />
          <Route path="/youtube" element={<YouTubeSummary />} />
          <Route path="/rsi-lab" element={<RSILab />} />
          <Route path="/macd-lab" element={<MACDLab />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
