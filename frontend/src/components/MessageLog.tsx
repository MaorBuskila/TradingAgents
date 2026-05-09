/** Real-time messages & tool-calls feed for RunAnalysis. */

export interface MsgRow {
  id: number
  time: string
  kind: string
  text: string
}

interface MessageLogProps {
  rows: MsgRow[]
  maxHeight?: number
}

export default function MessageLog({ rows, maxHeight = 360 }: MessageLogProps) {
  return (
    <div className="run-analysis-panel">
      <h3 className="run-analysis-panel-title">Messages &amp; Tools</h3>
      <div className="cli-table-wrap" style={{ maxHeight }}>
        <table className="cli-table cli-table-messages">
          <thead>
            <tr>
              <th>Time</th>
              <th>Type</th>
              <th>Content</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={3} className="cli-table-empty">
                  Waiting for stream…
                </td>
              </tr>
            ) : (
              rows.map((r) => (
                <tr key={r.id}>
                  <td className="cli-mono">{r.time}</td>
                  <td>{r.kind}</td>
                  <td className="cli-msg-content">{r.text}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
