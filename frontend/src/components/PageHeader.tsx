import type { ReactNode } from 'react'

interface PageHeaderProps {
  title: string
  description?: ReactNode
  actions?: ReactNode
}

/** Shared page title + description strip used across all pages. */
export default function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <header className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
      <div style={{ minWidth: 0 }}>
        <h1 className="page-title">{title}</h1>
        {description && (
          typeof description === 'string'
            ? <p className="page-desc">{description}</p>
            : <div className="page-desc">{description}</div>
        )}
      </div>
      {actions && (
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexShrink: 0 }}>
          {actions}
        </div>
      )}
    </header>
  )
}
