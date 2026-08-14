/**
 * How human the gait is, shown as gauges against measured human ranges.
 *
 * The chart panel answers "is training progressing". This answers a different question that
 * a reward curve provably cannot: "is it actually walking". Every gait failure in this
 * project scored well on episode return and was caught by a person watching a video, so each
 * gauge below is one of those failures turned into a number with the human range drawn on it.
 *
 * The design goal is that a defect is visible without reading any numbers: the human range is
 * a bright band, the current value is a marker, and a marker outside its band is the defect.
 */
import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

type BandView = {
  key: string
  label: string
  value: number | null
  low: number
  high: number
  unit: string
  catches: string
  score: number | null
}

type GroupView = {
  name: string
  blurb: string
  score: number | null
  bands: BandView[]
}

export type GaitCard = {
  overall: number | null
  iteration?: number
  groups: GroupView[]
  worst: BandView[]
  history: HistoryPoint[]
  group_names?: string[]
}

type HistoryPoint = {
  iteration: number
  env_steps: number
  overall: number
  [group: string]: number | null
}

/** One colour per group, held stable so a line means the same thing across refreshes. */
const GROUP_COLORS: Record<string, string> = {
  Posture: 'var(--series-2)',
  Rhythm: 'var(--series-3)',
  Smoothness: 'var(--series-4)',
  Balance: 'var(--series-5)',
  Reliability: 'var(--series-6)',
}

/** Red below a third, amber to two thirds, green above. */
function tone(score: number | null): string {
  if (score === null) return 'var(--muted)'
  if (score >= 0.75) return '#3ecf8e'
  if (score >= 0.45) return '#e8c14a'
  return '#e8615a'
}

function pct(score: number | null): string {
  return score === null ? '--' : `${Math.round(score * 100)}%`
}

/**
 * One measurement as a horizontal gauge.
 *
 * The axis spans one band width either side of the human range, so "just outside" and "wildly
 * outside" are visually different: a marker pinned to the end means at least a full band away.
 */
function Gauge({ band }: { band: BandView }) {
  const width = Math.max(band.high - band.low, 1e-9)
  const axisLow = band.low - width
  const axisHigh = band.high + width
  const place = (v: number) =>
    Math.max(0, Math.min(100, ((v - axisLow) / (axisHigh - axisLow)) * 100))

  const humanLeft = place(band.low)
  const humanWidth = place(band.high) - humanLeft
  const marker = band.value === null ? null : place(band.value)
  const inRange = band.score !== null && band.score >= 0.999

  return (
    <div className="gauge" title={band.catches}>
      <div className="gauge-head">
        <span className="gauge-label">{band.label}</span>
        <span className="gauge-value" style={{ color: tone(band.score) }}>
          {band.value === null ? '--' : band.value.toFixed(2)}
          <span className="gauge-unit">{band.unit}</span>
        </span>
      </div>
      <div className="gauge-track">
        <div className="gauge-human" style={{ left: `${humanLeft}%`, width: `${humanWidth}%` }} />
        {marker !== null && (
          <div
            className={`gauge-marker${inRange ? ' in-range' : ''}`}
            style={{ left: `${marker}%`, background: tone(band.score) }}
          />
        )}
      </div>
      <div className="gauge-foot">
        human {band.low.toFixed(2)} to {band.high.toFixed(2)}
        {band.unit}
      </div>
    </div>
  )
}

/** Overall score as a ring, because a single number deserves to be findable instantly. */
function Ring({ score }: { score: number | null }) {
  const radius = 52
  const circumference = 2 * Math.PI * radius
  const filled = score === null ? 0 : score * circumference
  return (
    <svg className="ring" viewBox="0 0 128 128" role="img" aria-label="human-likeness score">
      <circle cx="64" cy="64" r={radius} className="ring-track" />
      <circle
        cx="64"
        cy="64"
        r={radius}
        className="ring-fill"
        stroke={tone(score)}
        strokeDasharray={`${filled} ${circumference}`}
        transform="rotate(-90 64 64)"
      />
      <text x="64" y="60" className="ring-value" style={{ fill: tone(score) }}>
        {pct(score)}
      </text>
      <text x="64" y="80" className="ring-caption">
        human-like
      </text>
    </svg>
  )
}

/**
 * The score over the whole run, overall plus each group.
 *
 * The overall line alone tells you it dropped. The group lines tell you WHY it dropped:
 * whether the gait stopped alternating, started falling over, or began to skate. That is
 * the difference between a chart you watch and a chart you can act on.
 *
 * Expect it to be jagged. Reinforcement learning genuinely does oscillate, and a run that
 * looks flat here is usually one that has stopped exploring rather than one that is stable.
 */
function History({ history, groups }: { history: HistoryPoint[]; groups: string[] }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  // Everything is scored in 0..1 but the axis is a percentage, so scale here rather than
  // per-series: a group line left in 0..1 renders as a flat line pinned to the bottom.
  const data = useMemo(
    () =>
      history.map((p) => {
        const point: Record<string, number> = {
          iteration: p.iteration,
          overallPct: p.overall * 100,
        }
        for (const name of groups) {
          const v = p[name]
          if (typeof v === 'number') point[name] = v * 100
        }
        return point
      }),
    [history, groups],
  )

  if (history.length < 2) {
    return (
      <div className="history-empty">
        The score appears here after two evaluations, then updates live.
      </div>
    )
  }

  const toggle = (name: string) =>
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })

  const best = Math.max(...history.map((p) => p.overall))
  const latest = history[history.length - 1].overall
  const previous = history.length > 1 ? history[history.length - 2].overall : latest
  const delta = latest - previous

  return (
    <div className="history">
      <div className="history-head">
        <h3>Score over training</h3>
        <div className="history-stats">
          <span>
            now <strong style={{ color: tone(latest) }}>{pct(latest)}</strong>
          </span>
          <span className={delta >= 0 ? 'delta up' : 'delta down'}>
            {delta >= 0 ? '\u25b2' : '\u25bc'} {Math.abs(delta * 100).toFixed(0)} pts
          </span>
          <span>
            best <strong>{pct(best)}</strong>
          </span>
          <span className="history-count">{history.length} evaluations</span>
        </div>
      </div>

      <div className="history-chart">
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
            <defs>
              <linearGradient id="overallFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.32} />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="iteration"
              stroke="var(--text-faint)"
              tick={{ fontSize: 11 }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              stroke="var(--text-faint)"
              tick={{ fontSize: 11 }}
              tickLine={false}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              contentStyle={{
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                fontSize: 12,
              }}
              labelFormatter={(v) => `iteration ${v}`}
              formatter={(value: number, name: string) => [
                `${Math.round(value)}%`,
                name === 'overallPct' ? 'Overall' : name,
              ]}
            />
            <Legend
              onClick={(e: any) => toggle(String(e.dataKey))}
              wrapperStyle={{ fontSize: 11.5, cursor: 'pointer', paddingTop: 6 }}
              formatter={(value: string) => (value === 'overallPct' ? 'Overall' : value)}
            />
            <Area
              type="monotone"
              dataKey="overallPct"
              stroke="var(--accent)"
              strokeWidth={2.5}
              fill="url(#overallFill)"
              dot={false}
              isAnimationActive={false}
              hide={hidden.has('overallPct')}
            />
            {groups.map((name) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={GROUP_COLORS[name] ?? 'var(--series-7)'}
                strokeWidth={1.4}
                dot={false}
                isAnimationActive={false}
                connectNulls
                hide={hidden.has(name)}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="history-note">
        Click a name in the legend to hide that line. Jaggedness is expected: RL oscillates,
        and a perfectly flat curve here usually means exploration has collapsed.
      </p>
    </div>
  )
}

export function GaitScore({ runId }: { runId: string | null }) {
  const [card, setCard] = useState<GaitCard | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    const load = async () => {
      try {
        const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/gait`)
        if (!res.ok) throw new Error(`gait score: ${res.status}`)
        const data = (await res.json()) as GaitCard
        if (!cancelled) {
          setCard(data)
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

  if (error) return <div className="panel gait-panel">gait score unavailable: {error}</div>
  if (!card) return <div className="panel gait-panel">scoring gait...</div>
  if (card.overall === null)
    return <div className="panel gait-panel">no evaluation yet, the score appears after the first one</div>

  return (
    <div className="panel gait-panel">
      <div className="gait-header">
        <Ring score={card.overall} />
        <div className="gait-intro">
          <h2>Is it actually walking?</h2>
          <p>
            Scored against measured human walking, not against reward. Every gait failure in
            this project scored well on return and was caught by eye: folding at the waist, a
            sliding brace, a one-sided gait, a rocking split stance, pogoing. Each gauge is one
            of those, with the human range drawn on it.
          </p>
        </div>
      </div>

      <History history={card.history} groups={card.group_names ?? []} />

      {card.worst.length > 0 && (
        <div className="gait-worst">
          <span className="worst-title">Furthest from human</span>
          {card.worst.map((b) => (
            <span key={b.key} className="worst-chip" title={b.catches}>
              <em>{b.label}</em> {b.value === null ? '--' : b.value.toFixed(2)}
              {b.unit} <span className="worst-target">vs {b.low.toFixed(2)}-{b.high.toFixed(2)}</span>
            </span>
          ))}
        </div>
      )}

      <div className="gait-groups">
        {card.groups.map((group) => (
          <section key={group.name} className="gait-group">
            <header>
              <span className="group-name">{group.name}</span>
              <span className="group-score" style={{ color: tone(group.score) }}>
                {pct(group.score)}
              </span>
            </header>
            <p className="group-blurb">{group.blurb}</p>
            {group.bands.map((band) => (
              <Gauge key={band.key} band={band} />
            ))}
          </section>
        ))}
      </div>
    </div>
  )
}
