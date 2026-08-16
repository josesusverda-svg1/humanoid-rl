/** Shapes returned by the FastAPI backend. Kept in one file so the API contract is
 *  visible in a single place and a backend change surfaces as a type error here. */

export interface RunSummary {
  id: string
  name: string
  started: number
  modified: number
  iterations: number
  env_steps: number
  episode_return: number | null
  success_rate: number | null
  env_steps_per_sec: number | null
  wall_hours: number | null
  checkpoints: number
  videos: number
  /** True when the metrics file changed in the last 60 s, i.e. training is still running. */
  active: boolean
}

/** One training iteration. Keys beyond these are dynamic (reward/<term> and so on),
 *  which is why the index signature is present. */
export interface MetricRow {
  iteration: number
  env_steps: number
  wall_time: number
  timestamp: number
  episode_return?: number
  episode_return_std?: number
  episode_length?: number
  success_rate?: number
  episodes_total?: number
  env_steps_per_sec?: number
  approx_kl?: number
  learning_rate?: number
  action_std?: number
  entropy?: number
  value_loss?: number
  policy_loss?: number
  clip_fraction?: number
  grad_norm?: number
  total_wall_hours?: number
  time_env?: number
  time_policy?: number
  time_update?: number
  [key: string]: number | undefined
}

export interface MetricsResponse {
  run_id: string
  total_rows: number
  returned: number
  keys: string[]
  rows: MetricRow[]
}

export interface RunEvent {
  kind: string
  timestamp: number
  wall_time: number
  [key: string]: unknown
}

export interface VideoEntry {
  name: string
  url: string
  size_bytes: number
  modified: number
  /** Derived from the filename by the backend, so always present for trainer-rendered clips. */
  iteration?: number
  /** "best" when rendered on a new best score, "periodic" on the fixed cadence. */
  tag?: string
  seconds?: number
  frames?: number
  /** Whether the humanoid fell during the clip, and when. The single most useful thing
   *  to know before deciding whether a video is worth watching. */
  fell?: boolean
  fell_at?: number | null
  mean_speed?: number
  mean_slip?: number
  /** Populated from Phase 4 onward. */
  terrain_level?: number
  waypoints_completed?: number
}

// ---------------------------------------------------------------- getup mission control

export interface GetupTrainRow {
  iteration: number
  'reward/track'?: number
  'reward/stand'?: number
  'reward/hold'?: number
  'reward/latch'?: number
  'reward/lift'?: number
  'reward/launch'?: number
  latch_rate_100?: number
  action_std?: number
  approx_kl?: number
  lr?: number
}

export interface GetupEvalRow {
  iteration: number
  'eval/standing_frac'?: number
  'eval/standing_frac_strict'?: number
  'eval/held_ever_frac'?: number
  'eval/exam_level'?: number
  'eval/root_height'?: number
  'eval/head_height_ratio'?: number
  'eval/gate_frac'?: number
  'eval/launch_overspeed_frac'?: number
  'eval/episode_return'?: number
  'eval/foot_load_bw'?: number
  'eval/knee_max'?: number
}

export interface GetupSeries {
  train: GetupTrainRow[]
  eval: GetupEvalRow[]
  total_train_rows: number
}

export interface ConjunctRow {
  index: number
  name: string
  frac: number
}

export interface InspectionFrame {
  t: number
  pelvis: number
  head: number
  feet_bw: number
  knee: number
  hands: number
}

export interface Inspection {
  conjuncts: ConjunctRow[]
  frames: InspectionFrame[]
  age_seconds: number | null
  has_strip: boolean
  has_reference: boolean
}

export interface LogbookEntry {
  title: string
  verdict: string | null
  excerpt: string
}

export interface CompareRow {
  id: string
  iterations: number
  latch_iters: number
  latch_share: number
  max_standing_frac: number
  max_strict_frac: number
  final_exam_level: number
  modified: number
}
