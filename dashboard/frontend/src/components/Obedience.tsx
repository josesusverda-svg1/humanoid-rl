/**
 * Does it do what it was told, split by what it was told.
 *
 * The Numbers and Gait tabs answer "is it learning" and "is it walking like a human".
 * Neither answers the four questions a person actually asks while watching a video: how
 * often does it fall, how fast does it go forward, does it turn when told, can it walk
 * backwards. Those cannot be recovered from the aggregate metrics, because a single
 * `lin_vel_error` of 0.33 m/s is equally consistent with tracking forward perfectly while
 * ignoring every turn, and with the exact reverse.
 *
 * The bar for each direction is the tracking ratio (delivered / asked for). 1.0 is perfect.
 * The bar is drawn to the LEFT of zero when the ratio is negative, which means the humanoid
 * moved the opposite way to its command, a failure that reads as merely "a bit slow" in any
 * averaged number.
 */
import { useEffect, useState } from 'react'

type DirectionRow = {
  key: string
  label: string
  unit: string
  note: string
  share: number
  commanded: number | null
  achieved: number | null
  tracking: number | null
  verdict: string
}

type Latest = {
  iteration: number
  fall_rate: number
  seconds_upright: number | null
  episode_limit_s: number | null
  directions: DirectionRow[]
}

type Payload = {
  latest: Latest | null
  history: Record<string, number | null>[]
  supported: boolean
}

const fmt = (v: number | null, digits = 2) => (v === null ? '--' : v.toFixed(digits))

/** Green when it obeys, amber when it half obeys, red when it ignores or reverses. */
function toneFor(tracking: number | null): string {
  if (tracking === null) return 'muted'
  if (tracking < 0) return 'bad'
  if (tracking < 0.4) return 'bad'
  if (tracking < 0.75) return 'warn'
  return 'good'
}

function TrackingBar({ tracking }: { tracking: number | null }) {
  if (tracking === null) return <div className="obey-bar" />
  // Zero sits 20% in, so a negative ratio has somewhere to go and is visibly different
  // from simply not moving.
  const zero = 20
  const scale = 60 // percent of width per 1.0 of ratio
  const clamped = Math.max(-0.33, Math.min(1.33, tracking))
  const width = Math.abs(clamped) * scale
  const left = clamped >= 0 ? zero : zero - width
  return (
    <div className="obey-bar">
      <div className="obey-zero" style={{ left: `${zero}%` }} />
      <div className="obey-one" style={{ left: `${zero + scale}%` }} />
      <div
        className={`obey-fill ${toneFor(tracking)}`}
        style={{ left: `${left}%`, width: `${width}%` }}
      />
    </div>
  )
}

export function Obedience({ runId }: { runId: string | null }) {
  const [data, setData] = useState<Payload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    const load = async () => {
      try {
        const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/commands`)
        if (!res.ok) throw new Error(`commands: ${res.status}`)
        const payload = (await res.json()) as Payload
        if (!cancelled) {
          setData(payload)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    }
    load()
    const timer = setInterval(load, 5000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [runId])

  if (error) return <div className="panel">obedience unavailable: {error}</div>
  if (!data) return <div className="panel">loading...</div>
  if (!data.latest)
    return <div className="panel">no evaluation yet, this appears after the first one</div>
  if (!data.supported)
    return (
      <div className="panel">
        <h2>Does it do what it is told?</h2>
        <p className="card-sub">
          This run predates the per-direction metrics, so only the fall rate below is
          available for it. Start a new run to populate the rest.
        </p>
        <div className="obey-headline">
          <div className="obey-stat">
            <span className="obey-value bad">{(data.latest.fall_rate * 100).toFixed(0)}%</span>
            <span className="obey-label">of attempts end in a fall</span>
          </div>
        </div>
      </div>
    )

  const { latest } = data
  const fallTone = latest.fall_rate > 0.5 ? 'bad' : latest.fall_rate > 0.2 ? 'warn' : 'good'

  return (
    <div className="panel">
      <h2>Does it do what it is told?</h2>
      <p className="card-sub">
        Split by what was actually commanded. An averaged tracking error cannot tell you
        whether it walks forward well and ignores every turn, or the reverse.
      </p>

      <div className="obey-headline">
        <div className="obey-stat">
          <span className={`obey-value ${fallTone}`}>
            {(latest.fall_rate * 100).toFixed(0)}%
          </span>
          <span className="obey-label">of attempts end in a fall</span>
        </div>
        <div className="obey-stat">
          <span className="obey-value">{fmt(latest.seconds_upright, 1)}s</span>
          <span className="obey-label">
            upright per attempt
            {latest.episode_limit_s !== null && ` (limit ${latest.episode_limit_s.toFixed(0)}s)`}
          </span>
        </div>
      </div>

      <table className="obey-table">
        <thead>
          <tr>
            <th>Direction</th>
            <th>Asked for</th>
            <th>Delivered</th>
            <th>Follows the command</th>
            <th>Practised</th>
          </tr>
        </thead>
        <tbody>
          {latest.directions.map((d) => (
            <tr key={d.key}>
              <td>
                <strong>{d.label}</strong>
                <div className="obey-note">{d.note}</div>
              </td>
              <td className="mono">
                {fmt(d.commanded)} {d.unit}
              </td>
              <td className="mono">
                <span className={toneFor(d.tracking)}>{fmt(d.achieved)}</span> {d.unit}
              </td>
              <td>
                <TrackingBar tracking={d.tracking} />
                <div className="obey-note">
                  {d.tracking === null ? 'not commanded' : `${(d.tracking * 100).toFixed(0)}% - ${d.verdict}`}
                </div>
              </td>
              <td className="mono">{(d.share * 100).toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="card-sub" style={{ marginTop: 12 }}>
        "Delivered" is negative when it moves the opposite way to its command. "Practised" is
        the share of evaluation time that direction was commanded at all, so a direction it
        rarely sees is not being hidden behind a good-looking average.
      </p>
    </div>
  )
}
