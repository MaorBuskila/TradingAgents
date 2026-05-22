import { NavLink } from 'react-router-dom'
import {
  PlayCircle, FileText, PieChart, ListOrdered,
  Video, FlaskConical, Activity, Crosshair, Target, BarChart2,
  Sun, Moon, Monitor,
  type LucideIcon,
} from 'lucide-react'
import { type Theme } from '../hooks/useTheme'

interface NavItem {
  to: string
  label: string
  sub: string
  icon: LucideIcon
  end?: boolean
}

interface NavGroup {
  label?: string
  items: NavItem[]
}

const NAV_GROUPS: NavGroup[] = [
  {
    items: [
      { to: '/',          label: 'Run analysis', sub: 'Full LLM agent pipeline',  icon: PlayCircle,  end: true },
      { to: '/reports',   label: 'Reports',      sub: 'Past analysis results',    icon: FileText },
      { to: '/portfolio', label: 'Portfolio',     sub: 'Positions & P&L tracker', icon: PieChart },
      { to: '/watchlist', label: 'Watchlist',     sub: 'Tickers to monitor',      icon: ListOrdered },
      { to: '/youtube',   label: 'YouTube',       sub: 'Summarise transcripts',   icon: Video },
    ],
  },
  {
    label: 'Fundamentals',
    items: [
      { to: '/fundamentals-lab', label: 'Fundamentals Lab', sub: 'Earnings · SEC filings · LLM analysis', icon: BarChart2 },
    ],
  },
  {
    label: 'Quant Labs',
    items: [
      { to: '/rsi-lab',    label: 'RSI Lab',    sub: 'Walk-forward RSI optimizer', icon: FlaskConical },
      { to: '/macd-lab',   label: 'MACD Lab',   sub: 'Walk-forward MACD optimizer', icon: Activity },
      { to: '/sniper-lab', label: 'Sniper Lab',  sub: 'EMA + DT precision entries',   icon: Crosshair },
      { to: '/tp-tracker', label: 'TP Tracker', sub: 'Track sniper trade targets',   icon: Target },
    ],
  },
]

interface SidebarProps {
  theme: Theme
  setTheme: (t: Theme) => void
}

const THEME_OPTS: { value: Theme; icon: LucideIcon; label: string }[] = [
  { value: 'light',  icon: Sun,     label: 'Light' },
  { value: 'system', icon: Monitor, label: 'Auto'  },
  { value: 'dark',   icon: Moon,    label: 'Dark'  },
]

export default function Sidebar({ theme, setTheme }: SidebarProps) {
  return (
    <nav className="sidebar" aria-label="Main navigation">
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <span className="sidebar-logo-mark" aria-hidden="true">TA</span>
          TradingAgents
        </div>
        <p className="sidebar-tagline">Research &amp; portfolio workspace</p>
      </div>

      <ul className="nav-groups" role="list">
        {NAV_GROUPS.map((group, gi) => (
          <li key={gi}>
            <ul className="nav-group" role="list">
              {group.label && (
                <li className="nav-group-label" aria-hidden="true">
                  {group.label}
                </li>
              )}
              {group.items.map(({ to, label, sub, icon: Icon, end }) => (
                <li key={to}>
                  <NavLink
                    to={to}
                    end={end}
                    className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
                  >
                    <Icon size={17} strokeWidth={2} aria-hidden={true} />
                    <span className="nav-label">
                      <span className="nav-label-main">{label}</span>
                      <span className="nav-sub">{sub}</span>
                    </span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>

      {/* Theme toggle — sits at the bottom of the sidebar */}
      <div className="theme-toggle" role="group" aria-label="Color theme">
        {THEME_OPTS.map(({ value, icon: Icon, label }) => (
          <button
            key={value}
            type="button"
            className={`theme-toggle-btn${theme === value ? ' active' : ''}`}
            onClick={() => setTheme(value)}
            aria-pressed={theme === value}
            title={label}
          >
            <Icon size={13} strokeWidth={2} aria-hidden="true" />
            <span>{label}</span>
          </button>
        ))}
      </div>
    </nav>
  )
}
