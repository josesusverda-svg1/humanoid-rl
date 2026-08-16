/** The lab journal, rendered as experiment cards. The logbook is this project's
 *  institutional memory (every change, every verdict, every retraction); a console that
 *  shows the live run without its history invites repeating settled mistakes. */
import { useEffect, useState } from 'react'
import { api } from '../api'
import type { LogbookEntry } from '../types'

const VERDICT_TONE: Record<string, string> = {
  WORKED: 'ok',
  MIXED: 'warn',
  'NO EFFECT': 'idle',
  WORSE: 'bad',
  INVALID: 'bad',
  RETRACTED: 'bad',
  'PRE-REGISTERED': 'accent',
  TRIGGERED: 'warn',
}

export function Logbook() {
  const [entries, setEntries] = useState<LogbookEntry[]>([])
  const [open, setOpen] = useState<number | null>(0)

  useEffect(() => {
    api.logbook().then(setEntries).catch(() => setEntries([]))
  }, [])

  if (!entries.length) return <div className="loading">Журнал пуст или не найден.</div>

  return (
    <div className="logbook">
      {entries.map((e, i) => (
        <article
          key={i}
          className={`log-card ${open === i ? 'open' : ''}`}
          onClick={() => setOpen(open === i ? null : i)}
        >
          <header>
            <span className={`verdict tone-${VERDICT_TONE[e.verdict ?? ''] ?? 'idle'}`}>
              {e.verdict ?? '—'}
            </span>
            <h3>{e.title}</h3>
          </header>
          {open === i && <pre className="log-body">{e.excerpt}</pre>}
        </article>
      ))}
    </div>
  )
}
