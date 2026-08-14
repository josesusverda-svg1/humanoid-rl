import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { MetricRow } from '../types'
import { fmt } from '../api'

const SERIES_COLORS = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
  'var(--series-6)',
  'var(--series-7)',
]

interface SeriesSpec {
  key: string
  label: string
}

interface ChartCardProps {
  title: string
  subtitle?: string
  rows: MetricRow[]
  series: SeriesSpec[]
  /** X axis field. Environment steps is the honest default: it is the unit RL sample
   *  budgets are quoted in, and unlike iteration count it stays comparable across runs
   *  that used different batch sizes. */
  xKey?: 'env_steps' | 'iteration' | 'total_wall_hours'
  yScale?: 'linear' | 'log'
  height?: number
}

function TooltipContent({ active, payload, label, xKey }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip mono">
      <div style={{ color: 'var(--text-dim)', marginBottom: 4 }}>
        {xKey === 'env_steps' ? `${fmt(label)} steps` : `${fmt(label)}`}
      </div>
      {payload.map((p: any) => (
        <div className="row" key={p.dataKey}>
          <span style={{ color: p.color }}>{p.name}</span>
          <span>{fmt(p.value, 4)}</span>
        </div>
      ))}
    </div>
  )
}

export function ChartCard({
  title,
  subtitle,
  rows,
  series,
  xKey = 'env_steps',
  yScale = 'linear',
  height = 190,
}: ChartCardProps) {
  // Only plot series that actually carry data. Reward terms in particular appear and
  // disappear as tasks change, and an empty line in the legend is just noise.
  const present = series.filter((s) => rows.some((r) => Number.isFinite(r[s.key])))
  if (present.length === 0) return null

  return (
    <div className="card">
      <h3 className="card-title">{title}</h3>
      {subtitle && <p className="card-sub">{subtitle}</p>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={rows} margin={{ top: 4, right: 8, left: -14, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey={xKey}
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(v) => fmt(v, 1)}
            stroke="var(--border)"
            tick={{ fontSize: 10 }}
          />
          <YAxis
            scale={yScale}
            domain={yScale === 'log' ? ['auto', 'auto'] : ['auto', 'auto']}
            tickFormatter={(v) => fmt(v, 2)}
            stroke="var(--border)"
            tick={{ fontSize: 10 }}
            width={54}
          />
          <Tooltip content={<TooltipContent xKey={xKey} />} />
          {present.map((s, i) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {present.length > 1 && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', padding: '2px 0 8px' }}>
          {present.map((s, i) => (
            <span key={s.key} style={{ fontSize: 11, color: 'var(--text-dim)' }}>
              <span
                style={{
                  display: 'inline-block',
                  width: 8,
                  height: 8,
                  borderRadius: 2,
                  marginRight: 5,
                  background: SERIES_COLORS[i % SERIES_COLORS.length],
                }}
              />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

/** The Numbers view. Ordered so the question "is it improving, and how fast" is answered
 *  by the first row, before any diagnostic detail. */
export function NumbersCharts({ rows }: { rows: MetricRow[] }) {
  // Reward term keys are dynamic: each task declares its own, and they are logged with a
  // "reward/" prefix so they can be discovered rather than hardcoded here.
  const rewardKeys = Array.from(
    new Set(rows.flatMap((r) => Object.keys(r).filter((k) => k.startsWith('reward/')))),
  ).sort()

  return (
    <div className="charts">
      <ChartCard
        title="Episode return"
        subtitle="Mean over the last 200 completed episodes. The headline learning curve."
        rows={rows}
        series={[{ key: 'episode_return', label: 'return' }]}
      />
      <ChartCard
        title="Course success rate"
        subtitle="Fraction of episodes completing the whole course. The definition of done."
        rows={rows}
        series={[{ key: 'success_rate', label: 'success' }]}
      />
      <ChartCard
        title="Episode length"
        subtitle="Steps before falling or hitting the time limit. Rises before return does."
        rows={rows}
        series={[{ key: 'episode_length', label: 'steps' }]}
      />
      <ChartCard
        title="Throughput"
        subtitle="Environment steps per second. A sustained drop usually means thermal throttling."
        rows={rows}
        series={[{ key: 'env_steps_per_sec', label: 'env steps/s' }]}
      />
      <ChartCard
        title="Reward terms"
        subtitle="Each component separately. Debugging locomotion is usually finding the one term that quietly dominates."
        rows={rows}
        series={rewardKeys.map((k) => ({ key: k, label: k.replace('reward/', '') }))}
        height={210}
      />
      <ChartCard
        title="Policy update health"
        subtitle="KL should track the 0.01 target. Clip fraction far above 0.2 means the steps are too large."
        rows={rows}
        series={[
          { key: 'approx_kl', label: 'approx KL' },
          { key: 'clip_fraction', label: 'clip frac' },
        ]}
      />
      <ChartCard
        title="Learning rate and exploration"
        subtitle="Learning rate is driven by measured KL. Action std falling to near zero means exploration has collapsed."
        rows={rows}
        series={[
          { key: 'learning_rate', label: 'lr' },
          { key: 'action_std', label: 'action std' },
        ]}
      />
      <ChartCard
        title="Losses"
        subtitle="Value loss should fall and stabilise. Entropy falling fast is an early warning of premature convergence."
        rows={rows}
        series={[
          { key: 'value_loss', label: 'value' },
          { key: 'entropy', label: 'entropy' },
          { key: 'grad_norm', label: 'grad norm' },
        ]}
      />
      <ChartCard
        title="Time breakdown per iteration"
        subtitle="Seconds in physics, policy inference and the PPO update. Shows where the wall clock actually goes."
        rows={rows}
        series={[
          { key: 'time_env', label: 'physics (CPU)' },
          { key: 'time_update', label: 'PPO update (GPU)' },
          { key: 'time_policy', label: 'inference' },
        ]}
      />
    </div>
  )
}
