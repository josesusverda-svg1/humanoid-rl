import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, fmt, fmtDuration } from './api'
import type { MetricRow, RunEvent, RunSummary, VideoEntry } from './types'
import { NumbersCharts } from './components/Charts'
import { GaitScore } from './components/GaitScore'
import { Learning } from './components/Learning'
import { Skeletons } from './components/Skeleton'
import { Obedience } from './components/Obedience'

const POLL_MS = 2000

type Tab = 'numbers' | 'obedience' | 'gait' | 'learning' | 'skeleton' | 'videos'

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [rows, setRows] = useState<MetricRow[]>([])
  const [events, setEvents] = useState<RunEvent[]>([])
  const [videos, setVideos] = useState<VideoEntry[]>([])
  const [tab, setTab] = useState<Tab>('numbers')
  const [error, setError] = useState<string | null>(null)

  // Tracks how many raw rows the server has, so live polling can request only the new
  // ones instead of refetching an entire multi-day history every two seconds.
  const loadedRows = useRef(0)

  const refreshRuns = useCallback(async () => {
    try {
      const list = await api.runs()
      setRuns(list)
      setError(null)
      // Default to the run that is actually training. Falling back to list[0] alone meant a
      // refresh during training could land on an idle run, and the whole point of opening
      // the dashboard mid-run is to watch the run. An explicit choice still sticks, because
      // this only fills in when nothing is selected yet.
      setSelected((current) => current ?? list.find((r) => r.active)?.id ?? list[0]?.id ?? null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  // Reset accumulated state whenever the selected run changes, otherwise rows from the
  // previous run would be appended to the new one.
  useEffect(() => {
    loadedRows.current = 0
    setRows([])
    setEvents([])
    setVideos([])
  }, [selected])

  const refreshRun = useCallback(async (runId: string) => {
    try {
      const [metrics, evs, vids] = await Promise.all([
        api.metrics(runId, loadedRows.current),
        api.events(runId),
        api.videos(runId),
      ])
      setRows((prev) => {
        // The server downsamples, so `rows.length` is not the raw count. Track the raw
        // total it reports instead, and append only what is new.
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

  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="brand">Humanoid RL</h1>
        <p className="brand-sub mono">M3 Max · MuJoCo CPU · PyTorch MPS</p>
        {runs.length === 0 && <p className="brand-sub">No runs yet.</p>}
        {runs.map((r) => (
          <button
            key={r.id}
            className={`run-item ${r.id === selected ? 'selected' : ''}`}
            onClick={() => setSelected(r.id)}
          >
            <div className="run-item-top">
              <span className="run-item-name">{r.id}</span>
              <span className={`dot ${r.active ? 'active' : ''}`} />
            </div>
            <div className="run-item-meta mono">
              {fmt(r.env_steps)} steps · ret {fmt(r.episode_return)}
            </div>
          </button>
        ))}
      </aside>

      <main className="main">
        {error && (
          <div className="error-banner">
            Backend unreachable: {error}
            <br />
            Start it with <code>.venv/bin/python scripts/dashboard.py</code>
          </div>
        )}

        {!run ? (
          <div className="empty">
            <p>No training runs found.</p>
            <p className="mono">
              <code>.venv/bin/python -m humanoid_rl.train</code>
            </p>
          </div>
        ) : (
          <>
            <header className="header">
              <h1 className="mono">{run.id}</h1>
              <span className={`badge ${run.active ? 'live' : ''}`}>
                {run.active ? 'training' : 'idle'}
              </span>
              <div className="tabs">
                <button
                  className={`tab ${tab === 'numbers' ? 'selected' : ''}`}
                  onClick={() => setTab('numbers')}
                >
                  Numbers
                </button>
                <button
                  className={`tab ${tab === 'obedience' ? 'selected' : ''}`}
                  onClick={() => setTab('obedience')}
                >
                  Obedience
                </button>
                <button
                  className={`tab ${tab === 'gait' ? 'selected' : ''}`}
                  onClick={() => setTab('gait')}
                >
                  Gait quality
                </button>
                <button
                  className={`tab ${tab === 'learning' ? 'selected' : ''}`}
                  onClick={() => setTab('learning')}
                >
                  Learning
                </button>
                <button
                  className={`tab ${tab === 'skeleton' ? 'selected' : ''}`}
                  onClick={() => setTab('skeleton')}
                >
                  Flip-book
                </button>
                <button
                  className={`tab ${tab === 'videos' ? 'selected' : ''}`}
                  onClick={() => setTab('videos')}
                >
                  Videos {videos.length > 0 && `(${videos.length})`}
                </button>
              </div>
            </header>

            <StatTiles run={run} latest={latest} />

            {tab === 'obedience' ? (
              <Obedience runId={selected} />
            ) : tab === 'gait' ? (
              <GaitScore runId={selected} />
            ) : tab === 'learning' ? (
              <Learning rows={rows} />
            ) : tab === 'skeleton' ? (
              <Skeletons runId={selected} />
            ) : tab === 'numbers' ? (
              rows.length === 0 ? (
                <div className="empty">Waiting for the first logged iteration...</div>
              ) : (
                <>
                  <NumbersCharts rows={rows} />
                  <div className="card" style={{ marginTop: 12 }}>
                    <h3 className="card-title">Events</h3>
                    <p className="card-sub">Checkpoints, videos and thermal samples.</p>
                    <div className="events mono">
                      {events
                        .slice()
                        .reverse()
                        .map((e, i) => (
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

function StatTiles({ run, latest }: { run: RunSummary; latest: MetricRow | null }) {
  const tiles: { label: string; value: string; sub?: string }[] = [
    {
      label: 'Episode return',
      value: fmt(latest?.episode_return ?? run.episode_return),
      sub: latest?.episode_return_std ? `± ${fmt(latest.episode_return_std)}` : undefined,
    },
    {
      label: 'Success rate',
      value: `${fmt((latest?.success_rate ?? 0) * 100, 1)}%`,
      sub: 'full course completed',
    },
    {
      label: 'Env steps',
      value: fmt(run.env_steps),
      sub: `${fmt(run.iterations)} iterations`,
    },
    {
      label: 'Throughput',
      value: fmt(latest?.env_steps_per_sec ?? run.env_steps_per_sec, 0),
      sub: 'env steps / sec',
    },
    {
      label: 'Wall time',
      value: fmtDuration(latest?.total_wall_hours ?? run.wall_hours),
    },
    {
      label: 'Episode length',
      value: fmt(latest?.episode_length, 1),
      sub: 'steps before ending',
    },
    {
      label: 'Checkpoints',
      value: String(run.checkpoints),
      sub: `${run.videos} videos`,
    },
  ]

  return (
    <div className="stats">
      {tiles.map((t) => (
        <div className="stat" key={t.label}>
          <div className="stat-label">{t.label}</div>
          <div className="stat-value mono">{t.value}</div>
          {t.sub && <div className="stat-sub">{t.sub}</div>}
        </div>
      ))}
    </div>
  )
}

function VideoGallery({ videos }: { videos: VideoEntry[] }) {
  if (videos.length === 0) {
    return (
      <div className="empty">
        <p>No evaluation videos yet.</p>
        <p style={{ fontSize: 12 }}>
          The trainer renders one evaluation episode offscreen every N updates, and always on
          a new best score. Arrives in Phase 3.
        </p>
      </div>
    )
  }
  return (
    <div className="gallery">
      {videos.map((v) => (
        <div className="card" key={v.name}>
          <video src={v.url} controls preload="metadata" />
          <div className="video-meta mono">
            {v.iteration !== undefined && <span>iter {fmt(v.iteration, 0)}</span>}
            {v.tag === 'best' && <span style={{ color: 'var(--good)' }}>new best</span>}
            {v.mean_speed !== undefined && <span>{fmt(v.mean_speed)} m/s</span>}
            {v.mean_slip !== undefined && <span>slip {fmt(v.mean_slip)}</span>}
            {/* Whether it fell, and when, is the fastest way to decide if a clip is worth
                watching. Shown in red so a wall of failures is obvious at a glance. */}
            {v.fell !== undefined && (
              <span style={{ color: v.fell ? 'var(--bad)' : 'var(--good)' }}>
                {v.fell ? `fell ${v.fell_at ? `@${fmt(v.fell_at, 1)}s` : ''}` : 'stayed up'}
              </span>
            )}
            {v.terrain_level !== undefined && <span>terrain {v.terrain_level}</span>}
            {v.waypoints_completed !== undefined && <span>waypoints {v.waypoints_completed}</span>}
            <span style={{ color: 'var(--text-faint)' }}>{(v.size_bytes / 1e6).toFixed(1)} MB</span>
          </div>
        </div>
      ))}
    </div>
  )
}
