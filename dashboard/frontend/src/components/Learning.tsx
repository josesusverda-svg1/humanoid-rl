/**
 * What actually happens when the model learns, made visible.
 *
 * Training is usually a black box with one number coming out of it. It is not a black box:
 * every iteration is the same four steps, and each step logs something you can look at. This
 * panel lays those out in order, so "the policy improved" becomes "the gait reward went up,
 * which pulled the actor's first layer 3% and its output layer 60%, and the step size was
 * held at the KL target".
 *
 * Nothing here is a new measurement. It is the numbers the trainer already writes, arranged
 * as the story of one update instead of as a wall of charts.
 */
import { useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { MetricRow } from '../types'

const PALETTE = [
  '#4a9eff', '#3ecf8e', '#e8c14a', '#e8615a', '#b57ce8', '#4ecdc4',
  '#f0883e', '#6ea8fe', '#8bd450', '#e86ba0', '#7de2d1', '#c9a227',
  '#5eead4', '#fca5a5', '#a5b4fc', '#fcd34d',
]

function keysWithPrefix(rows: MetricRow[], prefix: string): string[] {
  const found = new Set<string>()
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (key.startsWith(prefix)) found.add(key.slice(prefix.length))
    }
  }
  return [...found].sort()
}

/** Step 1 and 2: what the humanoid was rewarded for, as a share of total reward. */
function RewardComposition({ rows }: { rows: MetricRow[] }) {
  const terms = keysWithPrefix(rows, 'reward/')
  const data = useMemo(() => {
    return rows
      .filter((r) => terms.some((t) => typeof r[`reward/${t}`] === 'number'))
      .map((r) => {
        // Shares of the total MAGNITUDE, so penalties (which are negative) are visible as
        // the share of the signal they occupy rather than cancelling rewards out to zero.
        const magnitudes = terms.map((t) => Math.abs((r[`reward/${t}`] as number) ?? 0))
        const total = magnitudes.reduce((a, b) => a + b, 0) || 1
        const point: Record<string, number> = { iteration: r.iteration as number }
        terms.forEach((t, i) => {
          point[t] = (magnitudes[i] / total) * 100
        })
        return point
      })
  }, [rows, terms])

  if (data.length < 2) return null
  const latest = data[data.length - 1]
  const ranked = terms
    .map((t) => ({ term: t, share: latest[t] ?? 0 }))
    .sort((a, b) => b.share - a.share)

  return (
    <section className="learn-card">
      <header>
        <span className="step-badge">1</span>
        <div>
          <h3>What is it being rewarded for?</h3>
          <p>
            Every reward term as a share of the total signal, over training. This is the
            "based on what": whichever band is widest is what the humanoid is currently being
            paid to do. When a band grows, the policy will follow it.
          </p>
        </div>
      </header>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="iteration" stroke="var(--text-faint)" tick={{ fontSize: 11 }} tickLine={false} />
          <YAxis
            stroke="var(--text-faint)"
            tick={{ fontSize: 11 }}
            tickLine={false}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
            labelFormatter={(v) => `iteration ${v}`}
            formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
          />
          {terms.map((t, i) => (
            <Area
              key={t}
              type="monotone"
              dataKey={t}
              stackId="reward"
              stroke={PALETTE[i % PALETTE.length]}
              fill={PALETTE[i % PALETTE.length]}
              fillOpacity={0.55}
              strokeWidth={0.6}
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
      <div className="chips">
        {ranked.slice(0, 6).map((r) => (
          <span className="chip" key={r.term}>
            <i style={{ background: PALETTE[terms.indexOf(r.term) % PALETTE.length] }} />
            {r.term} <strong>{r.share.toFixed(0)}%</strong>
          </span>
        ))}
      </div>
    </section>
  )
}

/** Step 3: which parts of the network the update actually rewrote. */
function WeightChange({ rows }: { rows: MetricRow[] }) {
  const layers = keysWithPrefix(rows, 'weight_change/')
  const data = useMemo(
    () =>
      rows
        .filter((r) => layers.some((l) => typeof r[`weight_change/${l}`] === 'number'))
        .map((r) => {
          const point: Record<string, number> = { iteration: r.iteration as number }
          for (const l of layers) {
            const v = r[`weight_change/${l}`]
            if (typeof v === 'number') point[l] = v * 100
          }
          return point
        }),
    [rows, layers],
  )

  if (data.length < 2) {
    return (
      <section className="learn-card">
        <header>
          <span className="step-badge">3</span>
          <div>
            <h3>Which weights changed?</h3>
            <p>Available from the next run onward: this is newly recorded.</p>
          </div>
        </header>
      </section>
    )
  }

  const latest = data[data.length - 1]
  const bars = layers
    .map((l) => ({ layer: l.replace('.weight', ''), change: latest[l] ?? 0 }))
    .sort((a, b) => b.change - a.change)

  return (
    <section className="learn-card">
      <header>
        <span className="step-badge">3</span>
        <div>
          <h3>Which weights changed?</h3>
          <p>
            How far each layer moved in one update, as a percentage of its own size. Relative
            rather than absolute, because the layers differ hugely in size and the raw numbers
            would not be comparable. The output layer normally moves most: it starts
            deliberately tiny, so the same nudge is a larger fraction of it.
          </p>
        </div>
      </header>

      <div className="weight-split">
        <div className="weight-bars">
          <span className="mini-title">this iteration</span>
          <ResponsiveContainer width="100%" height={170}>
            <BarChart data={bars} layout="vertical" margin={{ top: 2, right: 26, bottom: 2, left: 4 }}>
              <XAxis type="number" hide />
              <YAxis
                type="category"
                dataKey="layer"
                width={76}
                stroke="var(--text-faint)"
                tick={{ fontSize: 10.5 }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                cursor={{ fill: 'var(--bg-elevated)' }}
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => [`${v.toFixed(2)}%`, 'moved']}
              />
              <Bar dataKey="change" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {bars.map((b) => (
                  <Cell key={b.layer} fill={b.layer.startsWith('actor') ? '#4a9eff' : '#3ecf8e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="weight-trend">
          <span className="mini-title">over training</span>
          <ResponsiveContainer width="100%" height={170}>
            <ComposedChart data={data} margin={{ top: 2, right: 8, bottom: 0, left: -22 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" stroke="var(--text-faint)" tick={{ fontSize: 11 }} tickLine={false} />
              <YAxis stroke="var(--text-faint)" tick={{ fontSize: 11 }} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                labelFormatter={(v) => `iteration ${v}`}
                formatter={(v: number, n: string) => [`${v.toFixed(2)}%`, n.replace('.weight', '')]}
              />
              {layers.map((l, i) => (
                <Line
                  key={l}
                  type="monotone"
                  dataKey={l}
                  stroke={PALETTE[i % PALETTE.length]}
                  strokeWidth={1.3}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  )
}

/** Step 4: the guardrail that decides how large a step is allowed. */
function StepSize({ rows }: { rows: MetricRow[] }) {
  const data = useMemo(
    () =>
      rows
        .filter((r) => typeof r.approx_kl === 'number')
        .map((r) => ({
          iteration: r.iteration as number,
          kl: (r.approx_kl as number) * 1000,
          lr: ((r.learning_rate as number) ?? 0) * 1e5,
          clip: ((r.clip_fraction as number) ?? 0) * 100,
        })),
    [rows],
  )
  if (data.length < 2) return null

  return (
    <section className="learn-card">
      <header>
        <span className="step-badge">4</span>
        <div>
          <h3>How big was the step, and who decided?</h3>
          <p>
            KL divergence measures how much the policy actually changed. Rather than following
            a fixed schedule, the learning rate is adjusted every iteration to hold KL near its
            target: when the policy moves too far the rate drops, when it barely moves the rate
            rises. So the two lines below should mirror each other. That is the trainer steering
            itself, and it is why nothing here needed a hand-tuned decay.
          </p>
        </div>
      </header>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="iteration" stroke="var(--text-faint)" tick={{ fontSize: 11 }} tickLine={false} />
          <YAxis stroke="var(--text-faint)" tick={{ fontSize: 11 }} tickLine={false} />
          <Tooltip
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
            labelFormatter={(v) => `iteration ${v}`}
            formatter={(v: number, n: string) =>
              n === 'kl'
                ? [`${(v / 1000).toFixed(4)}`, 'policy change (KL)']
                : n === 'lr'
                  ? [`${(v / 1e5).toExponential(2)}`, 'learning rate']
                  : [`${v.toFixed(1)}%`, 'clipped samples']
            }
          />
          <Area type="monotone" dataKey="kl" stroke="#e8c14a" fill="#e8c14a" fillOpacity={0.16} strokeWidth={1.8} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="lr" stroke="#4a9eff" strokeWidth={1.8} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="clip" stroke="#b57ce8" strokeWidth={1.1} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="chips">
        <span className="chip"><i style={{ background: '#e8c14a' }} />policy change (KL x1000)</span>
        <span className="chip"><i style={{ background: '#4a9eff' }} />learning rate (x100k)</span>
        <span className="chip"><i style={{ background: '#b57ce8' }} />clipped samples %</span>
      </div>
    </section>
  )
}

const STEPS = [
  ['1', 'Collect', 'Thousands of humanoids act in parallel for a few dozen steps each, and every step is scored by the reward terms.'],
  ['2', 'Judge', 'The critic predicted how good each moment would be. The difference between what happened and that prediction is the "advantage": positive means better than expected.'],
  ['3', 'Nudge', 'Weights move so actions with positive advantage become more likely and negative ones less. This is the only step that changes the network.'],
  ['4', 'Check', 'Measure how far the policy moved. If it moved too far the learning rate drops for next time; too little and it rises.'],
]

export function Learning({ rows }: { rows: MetricRow[] }) {
  const [showPrimer, setShowPrimer] = useState(true)
  if (rows.length < 2) return <div className="panel">Waiting for training data...</div>

  return (
    <div className="panel learn-panel">
      <div className="learn-intro">
        <h2>What changed, and why</h2>
        <button className="link-button" onClick={() => setShowPrimer((v) => !v)}>
          {showPrimer ? 'hide' : 'show'} the four steps
        </button>
      </div>

      {showPrimer && (
        <div className="primer">
          {STEPS.map(([n, title, body]) => (
            <div className="primer-step" key={n}>
              <span className="step-badge">{n}</span>
              <div>
                <strong>{title}</strong>
                <p>{body}</p>
              </div>
            </div>
          ))}
          <p className="primer-foot">
            That loop repeats every iteration, thousands of times. Everything below is the
            trainer's own record of it.
          </p>
        </div>
      )}

      <RewardComposition rows={rows} />
      <WeightChange rows={rows} />
      <StepSize rows={rows} />
    </div>
  )
}
