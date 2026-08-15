/**
 * Multi-view stick figures of the gait, animated in the browser.
 *
 * A cheap alternative to video, and cheap in both directions: the trainer produces a capture
 * in 0.59 s and 80 KB where a video costs 18 s and 11 MB, and the browser draws it as SVG
 * lines rather than decoding frames. That means captures can be frequent enough to scrub
 * through a whole run and watch the gait develop, which a handful of expensive clips cannot
 * show.
 *
 * Six angles because a gait defect is often invisible from one: the one-sided gait looked
 * normal head-on and was obvious from the side, and the split-stance rocking was the reverse.
 */
import { useEffect, useMemo, useRef, useState } from 'react'

type Capture = {
  bodies: string[]
  bones: [number, number][]
  view_names: string[]
  fps: number
  trails: number[]
  views: number[][][][] // [view][frame][body][xy]
  meta: { iteration?: number; env_steps?: number; command?: number[]; seconds?: number }
}

type CaptureInfo = { name: string; iteration: number; env_steps: number; size_kb: number }

/** Whether you are looking at its face or its back, which decides which way its sides read. */
const FACING: Record<string, string> = {
  front: 'facing you',
  'front-left': 'facing you',
  'front-right': 'facing you',
  back: 'facing away',
  left: 'side on',
  right: 'side on',
}

const LIMB_COLOR = (bodies: string[], a: number, b: number) => {
  const name = `${bodies[a]} ${bodies[b]}`
  if (name.includes('left')) return '#4a9eff'
  if (name.includes('right')) return '#e8615a'
  return '#8b949e'
}

/** One camera angle. Pure SVG: no canvas, no raster, sharp at any size. */
function View({
  capture,
  view,
  frame,
  showTrails,
}: {
  capture: Capture
  view: number
  frame: number
  showTrails: boolean
}) {
  const frames = capture.views[view]
  const pose = frames[Math.min(frame, frames.length - 1)]

  // A fixed world window in metres, shared by every view and every capture, so a figure
  // cannot appear to grow or shrink between captures. Auto-fitting each frame would make
  // a change in posture look like a change in camera.
  const halfWidth = 1.1
  const top = 2.0
  const toX = (x: number) => ((x + halfWidth) / (2 * halfWidth)) * 100
  const toY = (y: number) => (1 - y / top) * 100

  return (
    <div className="skel-view">
      <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
        {/* Ground. Vertical position is absolute in the capture, so this line is honest:
            a figure floating above it is genuinely off the floor. */}
        <line x1="0" y1={toY(0)} x2="100" y2={toY(0)} className="skel-ground" />

        {showTrails &&
          capture.trails.map((b) => (
            <polyline
              key={b}
              className="skel-trail"
              points={frames
                .slice(Math.max(0, frame - 22), frame + 1)
                .map((f) => `${toX(f[b][0])},${toY(f[b][1])}`)
                .join(' ')}
              stroke={capture.bodies[b].includes('left') ? '#4a9eff' : '#e8615a'}
            />
          ))}

        {capture.bones.map(([a, b], i) => (
          <line
            key={i}
            x1={toX(pose[a][0])}
            y1={toY(pose[a][1])}
            x2={toX(pose[b][0])}
            y2={toY(pose[b][1])}
            stroke={LIMB_COLOR(capture.bodies, a, b)}
            strokeWidth={1.6}
            strokeLinecap="round"
          />
        ))}
        {pose.map((p, i) => (
          <circle key={i} cx={toX(p[0])} cy={toY(p[1])} r={i === 0 ? 1.5 : 0.9} className="skel-joint" />
        ))}
      </svg>
      <span className="skel-view-name">
        {capture.view_names[view]}
        {/* Which way the humanoid faces in this view, so the sides are never ambiguous. */}
        <em>{FACING[capture.view_names[view]] ?? ''}</em>
      </span>
    </div>
  )
}

export function Skeletons({ runId }: { runId: string | null }) {
  const [list, setList] = useState<CaptureInfo[]>([])
  const [index, setIndex] = useState(0)
  // null means "follow the newest capture". A number means the viewer deliberately scrubbed
  // back to that one and should be left there.
  //
  // The previous rule was `setIndex(i => i >= data.length - 1 ? data.length - 1 : i)`, which
  // reads as "stick to the newest" and does the opposite on first load: index starts at 0,
  // so `0 >= length - 1` is false for any list longer than one and it pins to the OLDEST
  // capture. It only ever followed the newest if you had already dragged to the end
  // yourself, which is exactly the dragging this was supposed to remove.
  const [pinned, setPinned] = useState<number | null>(null)
  const [capture, setCapture] = useState<Capture | null>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [showTrails, setShowTrails] = useState(true)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    const load = async () => {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/skeletons`)
      if (!res.ok) return
      const data = (await res.json()) as CaptureInfo[]
      if (cancelled) return
      setList(data)
      // Follow the newest unless the viewer has pinned an older one.
      setIndex(pinned === null
        ? Math.max(0, data.length - 1)
        : Math.min(pinned, Math.max(0, data.length - 1)))
    }
    load()
    const poll = setInterval(load, 15000)
    return () => {
      cancelled = true
      clearInterval(poll)
    }
  }, [runId, pinned])

  const current = list[index]
  useEffect(() => {
    if (!runId || !current) return
    let cancelled = false
    fetch(`/api/runs/${encodeURIComponent(runId)}/skeletons/${current.name}`)
      .then((r) => r.json())
      .then((d: Capture) => {
        if (!cancelled) {
          setCapture(d)
          setFrame(0)
        }
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [runId, current?.name])

  const frameCount = capture?.views[0]?.length ?? 0
  useEffect(() => {
    if (!playing || frameCount === 0) return
    timer.current = window.setInterval(
      () => setFrame((f) => (f + 1) % frameCount),
      1000 / (capture?.fps ?? 20),
    )
    return () => {
      if (timer.current) window.clearInterval(timer.current)
    }
  }, [playing, frameCount, capture?.fps])

  const steps = useMemo(
    () => (current ? `${(current.env_steps / 1e6).toFixed(1)}M steps` : ''),
    [current],
  )

  if (!runId) return <div className="panel">Select a run.</div>
  if (list.length === 0)
    return (
      <div className="panel">
        <h2>Gait flip-book</h2>
        <p className="muted">
          No captures yet. One is taken every few million environment steps; they cost about
          0.6 s and 80 KB each, against 18 s and 11 MB for a video.
        </p>
      </div>
    )

  return (
    <div className="panel skel-panel">
      <div className="skel-head">
        <div>
          <h2>Gait flip-book</h2>
          <p className="muted">
            Body positions projected orthographically from six angles, relative to the way
            the humanoid is <em>facing</em>, not to world axes. No rendering is involved,
            which is why these can be frequent enough to scrub through the whole run.
          </p>
          <p className="muted skel-legend">
            <span className="key left" /> the humanoid's own <strong>left</strong>
            <span className="key right" /> its own <strong>right</strong>
            <span className="skel-caveat">
              In <em>front</em> you are facing it, so its left appears on your right, exactly
              as when you face a person. In <em>back</em> you are behind it and the sides
              line up with yours.
            </span>
          </p>
        </div>
        <div className="skel-controls">
          <button className="chip-button" onClick={() => setPlaying((p) => !p)}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <button className="chip-button" onClick={() => setShowTrails((t) => !t)}>
            {showTrails ? 'Hide trails' : 'Show trails'}
          </button>
        </div>
      </div>

      {/* With a single capture the slider has nowhere to go, which reads as broken rather
          than as empty. Disable it and say why, so the state is legible. */}
      <div className={`skel-scrub${list.length < 2 ? ' inert' : ''}`}>
        <input
          type="range"
          min={0}
          max={Math.max(0, list.length - 1)}
          value={index}
          disabled={list.length < 2}
          onChange={(e) => {
            const next = Number(e.target.value)
            setIndex(next)
            // Dragging to the far right means "keep up with the run" rather than "pin the
            // capture that happens to be last right now".
            setPinned(next >= list.length - 1 ? null : next)
          }}
        />
        <span className="skel-when">
          iteration <strong>{current?.iteration.toLocaleString()}</strong> · {steps} ·{' '}
          {index + 1} of {list.length}
          {pinned === null
            ? <em className="skel-live"> · live</em>
            : (
              <button className="skel-jump" onClick={() => setPinned(null)}>
                jump to newest
              </button>
            )}
        </span>
      </div>
      {list.length < 2 && (
        <p className="skel-hint">
          Only one capture so far, so there is nothing to scrub between yet. To build the
          flip-book from checkpoints already on disk:{' '}
          <code>python scripts/capture_skeletons.py --run runs/{runId}</code>
        </p>
      )}

      {capture && (
        <div className="skel-grid">
          {capture.view_names.map((_, v) => (
            <View key={v} capture={capture} view={v} frame={frame} showTrails={showTrails} />
          ))}
        </div>
      )}
    </div>
  )
}
