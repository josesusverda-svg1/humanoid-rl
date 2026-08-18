import type { CompareRow, GetupSeries, Inspection, LogbookEntry, MetricsResponse, RunEvent, RunSummary, VideoEntry } from './types'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`)
  return (await res.json()) as T
}

export const api = {
  runs: () => get<RunSummary[]>('/api/runs'),

  /**
   * Metric rows for a run.
   *
   * `since` skips rows the client already holds. While a run is live the frontend polls
   * with since = rowsAlreadyLoaded, so each poll transfers only the new iterations rather
   * than the entire history, which keeps polling cheap on multi-day runs.
   */
  metrics: (runId: string, since = 0, maxPoints = 1500) =>
    get<MetricsResponse>(
      `/api/runs/${encodeURIComponent(runId)}/metrics?since=${since}&max_points=${maxPoints}`,
    ),

  events: (runId: string) => get<RunEvent[]>(`/api/runs/${encodeURIComponent(runId)}/events`),

  videos: (runId: string) => get<VideoEntry[]>(`/api/runs/${encodeURIComponent(runId)}/videos`),

  config: (runId: string) => get<Record<string, unknown>>(`/api/runs/${encodeURIComponent(runId)}/config`),

  getup: (runId: string, maxPoints = 700) =>
    get<GetupSeries>(`/api/runs/${encodeURIComponent(runId)}/getup?max_points=${maxPoints}`),

  inspection: () => get<Inspection>('/api/inspection'),

  logbook: () => get<LogbookEntry[]>('/api/logbook'),

  compare: () => get<CompareRow[]>('/api/getup/compare'),
}

/** Compact number formatting for stat tiles: 1.2M, 45.3k, 0.87. */
export function fmt(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  const abs = Math.abs(value)
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}k`
  if (abs > 0 && abs < 0.01) return value.toExponential(1)
  return value.toFixed(digits)
}

export function fmtDuration(hours: number | null | undefined): string {
  if (hours === null || hours === undefined || !Number.isFinite(hours)) return '-'
  const totalMinutes = Math.round(hours * 60)
  const h = Math.floor(totalMinutes / 60)
  const m = totalMinutes % 60
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}
