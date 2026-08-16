/** Cross-run table for the get-up campaign. Returns are NOT comparable across reward
 *  versions (the logbook's own rule), so the table leads with the quantities that are:
 *  hold completions, the strict predicate, and the exam ladder. */
import { useEffect, useState } from 'react'
import { api } from '../api'
import type { CompareRow } from '../types'

export function Compare() {
  const [rows, setRows] = useState<CompareRow[]>([])

  useEffect(() => {
    api.compare().then(setRows).catch(() => setRows([]))
  }, [])

  if (!rows.length) return <div className="loading">Считаю сводку по прогонам…</div>

  const maxLatch = Math.max(...rows.map((r) => r.latch_share), 0.0001)

  return (
    <div className="compare">
      <table>
        <thead>
          <tr>
            <th>Прогон</th>
            <th>Итераций</th>
            <th>Холды · доля итераций</th>
            <th>Стойка · уровень предиката</th>
            <th>Строгий предикат · пик</th>
            <th>Экзамен · финал</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td className="mono">{r.id.replace('getup-', '')}</td>
              <td className="mono">{r.iterations.toLocaleString()}</td>
              <td>
                <div className="bar-cell">
                  <div className="bar" style={{ width: `${(r.latch_share / maxLatch) * 100}%` }} />
                  <span className="mono">{(r.latch_share * 100).toFixed(1)}%</span>
                </div>
              </td>
              <td className="mono">{(r.max_standing_frac * 100).toFixed(2)}%</td>
              <td className="mono">{(r.max_strict_frac * 100).toFixed(2)}%</td>
              <td className="mono">{r.final_exam_level}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="compare-note">
        Возвраты между прогонами не сравниваются: система наград менялась (E31→E42), и одна
        и та же цифра в разных прогонах означает разное. Сравниваются только физические
        достижения.
      </p>
    </div>
  )
}
