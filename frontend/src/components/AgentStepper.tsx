/** Visual pipeline stepper — replaces the raw text status table in RunAnalysis. */

interface AgentStepperProps {
  agentStatus: Record<string, string>
  selectedAnalysts: string[]
  progressTeams: [string, string[]][]
  analystKeyToAgent: Record<string, string>
}

function ringClass(status: string): string {
  if (status === 'completed')  return 'done'
  if (status === 'in_progress') return 'running'
  if (status === 'error')      return 'error'
  return 'pending'
}

function ringIcon(status: string): string {
  if (status === 'completed')   return '✓'
  if (status === 'in_progress') return '…'
  if (status === 'error')       return '✕'
  return ''
}

export default function AgentStepper({
  agentStatus,
  selectedAnalysts,
  progressTeams,
  analystKeyToAgent,
}: AgentStepperProps) {
  return (
    <div className="agent-stepper">
      {progressTeams.map(([teamName, agents]) => {
        const active =
          teamName === 'Analyst Team'
            ? agents.filter((a) => selectedAnalysts.some((k) => analystKeyToAgent[k] === a))
            : agents
        if (active.length === 0) return null

        return (
          <div key={teamName} className="agent-team-group">
            <div className="agent-team-label">{teamName}</div>
            {active.map((agent) => {
              const st = agentStatus[agent] ?? 'pending'
              const rc = ringClass(st)
              return (
                <div key={agent} className={`agent-row ${rc}`}>
                  <div className={`agent-status-ring ${rc}`} aria-label={st}>
                    {rc === 'running' ? (
                      <span className="cli-spinner" aria-hidden="true" />
                    ) : (
                      <span style={{ fontSize: '0.65rem', fontWeight: 700 }} aria-hidden="true">
                        {ringIcon(st)}
                      </span>
                    )}
                  </div>
                  <span className="agent-row-name">{agent}</span>
                  {st === 'in_progress' && (
                    <span style={{ fontSize: 'var(--text-xs)', color: 'var(--accent)', fontWeight: 600 }}>
                      running
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
