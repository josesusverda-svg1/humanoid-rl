import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, fmt, fmtDuration } from './api'
import type { MetricRow, RunEvent, RunSummary, VideoEntry } from './types'
import { NumbersCharts } from './components/Charts'
import { GaitScore } from './components/GaitScore'
import { Learning } from './components/Learning'
import { Skeletons } from './components/Skeleton'
import { Obedience } from './components/Obedience'
import { GetupConsole } from './components/GetupConsole'
import { Logbook } from './components/Logbook'
import { Compare } from './components/Compare'

const POLL_MS = 2000

type Tab =
  | 'console' | 'logbook' | 'compare'
  | 'numbers' | 'obedience' | 'gait' | 'learning' | 'skeleton' | 'videos'

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [rows, setRows] = useState<MetricRow[]>([])
  const [events, setEvents] = useState<RunEvent[]>([])
  const [videos, setVideos] = useState<VideoEntry[]>([])
  const [totalIters, setTotalIters] = useState<number | null>(null)
  const [tab, setTab] = useState<Tab>('console')
  const [error, setError] = useState<string | null>(null)
  // Sidebar collapse persists: a console left open for hours should reopen the way it was
  // closed, and on a laptop the run list is worth 240px only while you are switching runs.
  const [railOpen, setRailOpen] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const saved = window.localStorage.getItem('railOpen')
    return saved !== null ? saved === '1' : window.innerWidth >= 1100
  })
  useEffect(() => {
    window.localStorage.setItem('railOpen', railOpen ? '1' : '0')
  }, [railOpen])

  const loadedRows = useRef(0)

  const refreshRuns = useCallback(async () => {
    try {
      const list = await api.runs()
      setRuns(list)
      setError(null)
      setSelected((current) => current ?? list.find((r) => r.active)?.id ?? list[0]?.id ?? null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    loadedRows.current = 0
    setRows([])
    setEvents([])
    setVideos([])
    setTotalIters(null)
    if (selected) {
      // Budget for the progress bar: total_env_steps / (num_envs * horizon).
      api.config(selected).then((cfg: any) => {
        const total = cfg?.run?.total_env_steps
        const envs = cfg?.env?.num_envs
        const horizon = cfg?.ppo?.horizon
        if (total && envs && horizon) setTotalIters(Math.ceil(total / (envs * horizon)))
      }).catch(() => undefined)
      setTab(selected.startsWith('getup') ? 'console' : 'numbers')
    }
  }, [selected])

  const refreshRun = useCallback(async (runId: string) => {
    try {
      const [metrics, evs, vids] = await Promise.all([
        api.metrics(runId, loadedRows.current),
        api.events(runId),
        api.videos(runId),
      ])
      setRows((prev) => {
        if (loadedRows.current === 0) return metrics.rows
        return metrics.rows.length > 0 ? [...prev, ...metrics.rows] : prev
      })
      loadedRows.current = metrics.total_rows
      setEvents(evs)
      setVideos(vids)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault()
        setRailOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    void refreshRuns()
    const t = setInterval(() => void refreshRuns(), POLL_MS * 3)
    return () => clearInterval(t)
  }, [refreshRuns])

  useEffect(() => {
    if (!selected) return
    void refreshRun(selected)
    const t = setInterval(() => void refreshRun(selected), POLL_MS)
    return () => clearInterval(t)
  }, [selected, refreshRun])

  const run = useMemo(() => runs.find((r) => r.id === selected) ?? null, [runs, selected])
  const latest = rows.length > 0 ? rows[rows.length - 1] : null
  const progress = run && totalIters ? Math.min(run.iterations / totalIters, 1) : null
  const etaHours = run && totalIters && latest?.env_steps_per_sec && run.active
    ? ((totalIters - run.iterations) * 98304) / latest.env_steps_per_sec / 3600
    : null

  const TABS: { id: Tab; label: string }[] = [
    { id: 'console', label: 'Консоль' },
    { id: 'logbook', label: 'Журнал' },
    { id: 'compare', label: 'Сравнение' },
    { id: 'numbers', label: 'Кривые' },
    { id: 'learning', label: 'Обучение' },
    { id: 'videos', label: videos.length ? `Видео (${videos.length})` : 'Видео' },
    { id: 'gait', label: 'Походка' },
    { id: 'obedience', label: 'Команды' },
    { id: 'skeleton', label: 'Флипбук' },
  ]

  return (
    <div className={`app ${railOpen ? '' : 'rail-closed'}`}>
      <aside className="sidebar">
        <div className="rail-head">
          <h1 className="brand">
            {railOpen ? <>HUMANOID<span className="brand-accent">LAB</span></> : <span className="brand-accent">HL</span>}
          </h1>
          <button
            className="rail-toggle"
            onClick={() => setRailOpen((v) => !v)}
            aria-label={railOpen ? 'Свернуть список прогонов' : 'Развернуть список прогонов'}
            title={railOpen ? 'Свернуть (⌘B)' : 'Развернуть (⌘B)'}
          >
            {railOpen ? '‹' : '›'}
          </button>
        </div>
        {railOpen && <p className="brand-sub mono">M3 Max · MuJoCo · PPO/MPS</p>}
        <div className="run-list">
          {runs.map((r) => (
            <button
              key={r.id}
              className={`run-item ${r.id === selected ? 'selected' : ''}`}
              onClick={() => setSelected(r.id)}
              title={r.id}
            >
              <div className="run-item-top">
                <span className="run-item-name">
                  {railOpen
                    ? r.id.replace(/^getup-/, 'встаёт · ').replace(/^envelope-/, 'ходьба · ')
                    /* Collapsed: day and hour, e.g. "16·12" for getup-20260816-120145.
                       Parsed, not sliced: slicing from the end depends on the prefix length
                       and silently produced "08·12" for a run started on the 16th. */
                    : (() => {
                        const m = r.id.match(/(\d{4})(\d{2})(\d{2})-(\d{2})/)
                        return m ? `${m[3]}·${m[4]}` : r.id.slice(-4)
                      })()}
                </span>
                <span className={`dot ${r.active ? 'active' : ''}`} />
              </div>
              {railOpen && (
                <div className="run-item-meta mono">
                  {fmt(r.iterations, 0)} ит · {fmt(r.env_steps)} шагов
                </div>
              )}
            </button>
          ))}
        </div>
      </aside>

      <main className="main">
        {error && (
          <div className="error-banner">
            Бэкенд недоступен: {error}
            <br />
            Запуск: <code>.venv/bin/python scripts/dashboard.py</code>
          </div>
        )}

        {!run ? (
          <div className="empty"><p>Прогонов пока нет.</p></div>
        ) : (
          <>
            <header className="header">
              <div className="header-id">
                <h1 className="mono">{run.id}</h1>
                <span className={`badge ${run.active ? 'live' : ''}`}>
                  {run.active ? '● ИДЁТ ОБУЧЕНИЕ' : 'завершён'}
                </span>
              </div>
              <div className="header-progress">
                {progress !== null && (
                  <>
                    <div className="progress-rail">
                      <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
                    </div>
                    <span className="mono progress-text">
                      {fmt(run.iterations, 0)} / {fmt(totalIters, 0)} итераций
                      {latest?.env_steps_per_sec ? ` · ${fmt(latest.env_steps_per_sec, 0)} шаг/с` : ''}
                      {etaHours !== null ? ` · осталось ${fmtDuration(etaHours)}` : ''}
                    </span>
                  </>
                )}
              </div>
              <nav className="tabs" role="tablist">
                {TABS.map((t) => (
                  <button
                    key={t.id}
                    className={`tab ${tab === t.id ? 'selected' : ''}`}
                    onClick={() => setTab(t.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </nav>
            </header>

            {tab === 'console' ? (
              <GetupConsole run={run} />
            ) : tab === 'logbook' ? (
              <Logbook />
            ) : tab === 'compare' ? (
              <Compare />
            ) : tab === 'obedience' ? (
              <Obedience runId={selected} />
            ) : tab === 'gait' ? (
              <GaitScore runId={selected} />
            ) : tab === 'learning' ? (
              <Learning rows={rows} />
            ) : tab === 'skeleton' ? (
              <Skeletons runId={selected} />
            ) : tab === 'numbers' ? (
              rows.length === 0 ? (
                <div className="empty">Жду первую итерацию…</div>
              ) : (
                <>
                  <NumbersCharts rows={rows} />
                  <div className="card" style={{ marginTop: 12 }}>
                    <h3 className="card-title">События</h3>
                    <div className="events mono">
                      {events.slice().reverse().map((e, i) => (
                        <div className="event-row" key={i}>
                          <span className="event-kind">{e.kind}</span>
                          <span style={{ color: 'var(--text-dim)' }}>
                            {Object.entries(e)
                              .filter(([k]) => !['kind', 'timestamp', 'wall_time'].includes(k))
                              .map(([k, v]) => `${k}=${String(v)}`)
                              .join('  ')}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )
            ) : (
              <VideoGallery videos={videos} />
            )}
          </>
        )}
      </main>
    </div>
  )
}

function VideoGallery({ videos }: { videos: VideoEntry[] }) {
  if (videos.length === 0) {
    return <div className="empty"><p>Видео с оценок пока нет.</p></div>
  }
  return (
    <div className="gallery">
      {videos.map((v) => (
        <div className="card" key={v.name}>
          <video src={v.url} controls preload="metadata" />
          <div className="video-meta mono">
            {v.iteration !== undefined && <span>ит {fmt(v.iteration, 0)}</span>}
            {v.tag === 'best' && <span style={{ color: 'var(--good)' }}>рекорд</span>}
            {v.fell !== undefined && (
              <span style={{ color: v.fell ? 'var(--bad)' : 'var(--good)' }}>
                {v.fell ? `упал${v.fell_at ? ` @${fmt(v.fell_at, 1)}с` : ''}` : 'устоял'}
              </span>
            )}
            <span style={{ color: 'var(--text-faint)' }}>{(v.size_bytes / 1e6).toFixed(1)} MB</span>
          </div>
        </div>
      ))}
    </div>
  )
}
