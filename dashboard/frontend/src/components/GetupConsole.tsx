/** The get-up mission control: one screen that answers "как у него дела" at lab grade.
 *
 * Layout philosophy: the console leads with the SEQUENCE metrics that decide this project
 * (film-riding, hold completions, the exam ladder), keeps the honest strict predicate
 * beside every relaxed one, and ends with the last visual inspection, because every large
 * defect in this repo was found by looking and none by a metric.
 */
import { useEffect, useState } from 'react'
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api'
import type { GetupSeries, Inspection, RunSummary } from '../types'

const POLL_MS = 4000

const LEVELS = [
  { knee: 1.3, hold: 0.5 },
  { knee: 1.0, hold: 1.0 },
  { knee: 0.8, hold: 1.5 },
  { knee: 0.6, hold: 2.0 },
]

function pct(v: number | undefined | null, digits = 1): string {
  if (v === undefined || v === null || !Number.isFinite(v)) return '–'
  return `${(v * 100).toFixed(digits)}%`
}

function StatTile(props: {
  label: string
  value: string
  hint?: string
  tone?: 'ok' | 'warn' | 'bad' | 'idle'
}) {
  return (
    <div className={`tile tone-${props.tone ?? 'idle'}`}>
      <div className="tile-label">{props.label}</div>
      <div className="tile-value">{props.value}</div>
      {props.hint && <div className="tile-hint">{props.hint}</div>}
    </div>
  )
}

function Panel(props: { title: string; sub?: string; children: React.ReactNode; span?: number }) {
  return (
    <section className="panel" style={props.span ? { gridColumn: `span ${props.span}` } : undefined}>
      <header className="panel-head">
        <h3>{props.title}</h3>
        {props.sub && <span className="panel-sub">{props.sub}</span>}
      </header>
      {props.children}
    </section>
  )
}

const chartCommon = {
  margin: { top: 6, right: 8, bottom: 0, left: -12 },
}

function axis() {
  return {
    stroke: 'var(--line)',
    tick: { fill: 'var(--text-dim)', fontSize: 10.5 },
    tickLine: false as const,
  }
}

const tooltipStyle = {
  contentStyle: {
    background: 'var(--panel-2)', border: '1px solid var(--line)', borderRadius: 8,
    fontSize: 12, color: 'var(--text)',
  },
  labelStyle: { color: 'var(--text-dim)' },
}

export function GetupConsole(props: { run: RunSummary }) {
  const { run } = props
  const [series, setSeries] = useState<GetupSeries | null>(null)
  const [inspection, setInspection] = useState<Inspection | null>(null)

  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const [s, insp] = await Promise.all([api.getup(run.id), api.inspection()])
        if (!stop) {
          setSeries(s)
          setInspection(insp)
        }
      } catch {
        /* транзиентные сбои поллинга не должны ронять консоль */
      }
    }
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => { stop = true; clearInterval(id) }
  }, [run.id])

  if (!series) return <div className="loading">Загружаю телеметрию…</div>

  const t = series.train
  const e = series.eval
  const last = t[t.length - 1] ?? {}
  const lastEval = e[e.length - 1] ?? {}
  const level = Math.round((lastEval['eval/exam_level'] as number) ?? 0)
  const latchRate = (last.latch_rate_100 ?? 0)
  const std = last.action_std ?? 0
  const track10 = t.slice(-10).reduce((a, r) => a + (r['reward/track'] ?? 0), 0) / Math.max(t.slice(-10).length, 1)

  return (
    <div className="console">
      <div className="tiles">
        <StatTile
          label="Экзамен · уровень"
          value={`${level} / 3`}
          hint={`колено ≤ ${LEVELS[level].knee} · холд ${LEVELS[level].hold}s`}
          tone={level >= 3 ? 'ok' : level >= 1 ? 'warn' : 'idle'}
        />
        <StatTile
          label="Холды · доля итераций (окно 100)"
          value={pct(latchRate, 0)}
          hint="завершённые удержания уровня"
          tone={latchRate > 0.3 ? 'ok' : latchRate > 0.05 ? 'warn' : 'idle'}
        />
        <StatTile
          label="Езда по фильму · track₁₀"
          value={track10.toFixed(3)}
          hint="платит до 2.0 за идеал"
          tone={track10 > 0.5 ? 'ok' : track10 > 0.1 ? 'warn' : 'idle'}
        />
        <StatTile
          label="Строгий экзамен · strict"
          value={pct(lastEval['eval/standing_frac_strict'], 2)}
          hint="полные 13 условий, отчётность не смягчается"
          tone={(lastEval['eval/standing_frac_strict'] ?? 0) > 0 ? 'ok' : 'idle'}
        />
        <StatTile
          label="Разброс действий · std"
          value={std.toFixed(3)}
          hint="потолок 1.00 (E30)"
          tone={std > 1.01 ? 'bad' : std > 0.95 ? 'warn' : 'ok'}
        />
        <StatTile
          label="Успех с пола"
          value={pct(lastEval['eval/held_ever_frac'], 2)}
          hint="упал → встал → удержал"
          tone={(lastEval['eval/held_ever_frac'] ?? 0) > 0 ? 'ok' : 'idle'}
        />
      </div>

      <div className="grid">
        <Panel title="Езда по фильму" sub="reward/track · главная метрика E42" span={2}>
          <ResponsiveContainer width="100%" height={190}>
            <AreaChart data={t} {...chartCommon}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" {...axis()} minTickGap={48} />
              <YAxis {...axis()} width={46} />
              <Tooltip {...tooltipStyle} />
              <Area type="monotone" dataKey="reward/track" stroke="var(--accent)"
                fill="var(--accent-soft)" strokeWidth={1.8} dot={false} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Завершённые удержания" sub="доля итераций с защёлкой, окно 100" span={2}>
          <ResponsiveContainer width="100%" height={190}>
            <AreaChart data={t} {...chartCommon}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" {...axis()} minTickGap={48} />
              <YAxis {...axis()} width={46} domain={[0, 1]} />
              <Tooltip {...tooltipStyle} />
              <Area type="monotone" dataKey="latch_rate_100" stroke="var(--good)"
                fill="var(--good-soft)" strokeWidth={1.8} dot={false} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Зарплата стойки" sub="reward/stand и reward/hold (стаж)" span={2}>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={t} {...chartCommon}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" {...axis()} minTickGap={48} />
              <YAxis {...axis()} width={46} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="reward/stand" stroke="var(--accent)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="reward/hold" stroke="var(--warn)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Оценка: стоит ли он" sub="уровневый · строгий · успех, детерминированные прогоны" span={2}>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={e} {...chartCommon}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" {...axis()} minTickGap={48} />
              <YAxis {...axis()} width={46} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="eval/standing_frac" name="уровень" stroke="var(--accent)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="eval/standing_frac_strict" name="строгий" stroke="var(--good)" strokeWidth={1.8} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="eval/held_ever_frac" name="успех" stroke="var(--warn)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Высоты тела" sub="таз и голова в оценке (стоя: 0.877 / 1.0)" span={2}>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={e} {...chartCommon}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" {...axis()} minTickGap={48} />
              <YAxis {...axis()} width={46} domain={[0, 1]} />
              <Tooltip {...tooltipStyle} />
              <ReferenceLine y={0.877} stroke="var(--good)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="eval/root_height" name="таз" stroke="var(--accent)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="eval/head_height_ratio" name="голова" stroke="var(--pink)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Дисциплина движения" sub="коридор силы открыт · превышение 1 м/с вверх" span={2}>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={e} {...chartCommon}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" {...axis()} minTickGap={48} />
              <YAxis {...axis()} width={46} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="eval/gate_frac" name="коридор" stroke="var(--accent)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="eval/launch_overspeed_frac" name="прыжковость" stroke="var(--bad)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Здоровье оптимизации" sub="std (потолок 1.0) и KL (цель 0.01)" span={2}>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={t} {...chartCommon}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="iteration" {...axis()} minTickGap={48} />
              <YAxis {...axis()} width={46} />
              <Tooltip {...tooltipStyle} />
              <ReferenceLine y={1.0} stroke="var(--bad)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="action_std" name="std" stroke="var(--warn)" strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="approx_kl" name="KL" stroke="var(--text-dim)" strokeWidth={1.2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {inspection && (
        <div className="inspection">
          <Panel
            title="Последний визуальный осмотр"
            sub={inspection.age_seconds !== null
              ? `${Math.round(inspection.age_seconds / 60)} мин назад · каждый крупный дефект проекта найден глазами`
              : 'ещё не проводился'}
            span={4}
          >
            <div className="inspection-body">
              {inspection.has_strip && (
                <figure className="strip">
                  <img src={`/api/inspection/strip?t=${Math.floor(Date.now() / 60000)}`} alt="Кадры текущей политики" />
                  <figcaption>Текущая политика · 0 / 1.2 / 2.4 / 4 / 8 / 15 с</figcaption>
                </figure>
              )}
              {inspection.has_reference && (
                <figure className="strip">
                  <img src="/api/inspection/reference" alt="Эталон вставания" />
                  <figcaption>Эталон E42: лёг → отжался → подобрал ноги → присед → встал</figcaption>
                </figure>
              )}
              {inspection.conjuncts.length > 0 && (
                <div className="conjuncts">
                  <h4>13 условий стойки · доля времени выполнено</h4>
                  {inspection.conjuncts.map((c) => (
                    <div key={c.index} className="conjunct-row">
                      <span className="conjunct-name">{c.index}. {c.name}</span>
                      <div className="conjunct-bar">
                        <div className="conjunct-fill" style={{ width: `${Math.min(c.frac * 100, 100)}%` }} />
                      </div>
                      <span className="conjunct-val">{pct(c.frac, 1)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Panel>
        </div>
      )}
    </div>
  )
}
