"""SEARCH for a physically feasible get-up instead of scripting one. CEM in the simulator.

E47. Two scripted attempts to author a reference both failed, and the second failure is the
informative one:

* reversing a gentle descent produced poses no controller can execute (measured: commanding
  the reversed film's own joint angles as servo targets, from its own first pose, moved the
  pelvis 0.094 -> 0.111 against a "reference" going 0.095 -> 0.879, because descending is
  gravity-assisted and rising fights gravity with the same actuators);
* hand-designed rising waypoints stood the body up in 0 of 1200 attempts (best pelvis 0.181
  against 0.82 required).

So stop authoring. The simulator is the only authority on what this body can do, and here it
is massively parallel, so search it: cross-entropy method over a waypoint trajectory of
servo targets, scored by how high the pelvis gets and where it ends after the commands stop.
Either a feasible get-up exists and CEM finds it, or a serious search fails and THAT is a
finding about the body, established by measurement rather than assumed after each new reward.

The search runs in the env's own ACTION units, through the env's own filter and decimation,
so anything it finds is directly reproducible by a policy. No privileged control path.

    python scripts/search_getup_trajectory.py
    python scripts/search_getup_trajectory.py --pop 512 --iters 40 --waypoints 6
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.getup import GetUpTask  # noqa: E402

#: Seconds per waypoint segment, and the hold at the end. The hold commands the nominal
#: stand (action zero) and is what separates "stood up" from "threw itself upward": the body
#: has to still be up when the choreography stops.
SEG_SECONDS = 0.75
HOLD_SECONDS = 1.5
#: Acceptance is NOT a height. It is the task's own 13-conjunct standing predicate, held for
#: --min-stand-frac of the hold. Height thresholds have been gamed here by a headstand and by
#: knees locked backwards; the predicate is the instrument that rejected both.


class SearchTask(GetUpTask):
    """The get-up body, with everything that would corrupt a deterministic search removed.

    Same physics, same actuators, same action scaling. What is off: the shove and the thrown
    ball (they would score candidates on luck), termination (the search resets explicitly),
    and the reset draw (every env starts from the SAME pose so candidates are comparable).
    """

    start_q: np.ndarray | None = None

    def reset_pose(self, state, idx, rng):  # noqa: ANN001, ARG002
        state.task_state["from_standing"][idx] = False
        state.task_state["from_midrise"][idx] = False
        return (np.tile(self.start_q, (idx.size, 1)),
                np.zeros((idx.size, state.qvel.shape[1])))

    def terminated_batch(self, state):  # noqa: ANN001
        return ~np.isfinite(state.qpos).all(axis=1)

    def push_request(self, state):  # noqa: ANN001, ARG002
        return None


def rollout(env: ThreadedVecEnv, targets: np.ndarray, seg_steps: int, hold_steps: int,
            record: bool = False):
    """Run one batch of candidates. `targets` is (N, waypoints, nu) in action units.

    Servo targets ramp linearly from the previous waypoint to the next, then the hold
    commands action zero (the nominal stand) and nothing else intervenes.
    """
    n, n_way = targets.shape[0], targets.shape[1]
    env.reset()
    frames_q, frames_v = [], []
    current = np.zeros((n, env.nu), dtype=np.float32)
    peak = np.zeros(n)
    hold_sum = np.zeros(n)
    hold_min = np.full(n, np.inf)
    stand_sum = np.zeros(n)
    flight = np.zeros(n)
    foot_h = env.task.cfg.u_foot_height

    def advance(a: np.ndarray, in_hold: bool) -> None:
        env.step(np.clip(a, -1.0, 1.0))
        s = env.state
        np.maximum(peak, s.root_height, out=peak)
        # Both feet clear of the floor is a FLIGHT frame. A person getting up never leaves
        # the ground, and every height statistic in this score can otherwise be earned by
        # jumping: measured live, a candidate peaking at 1.017 (above the 0.877 standing
        # height, so airborne) outscored one that reached 0.874 and stayed there.
        flight[:] += (s.key_body_pos[:, 0:2, 2].min(axis=1) > foot_h)
        if in_hold:
            hold_sum[:] += s.root_height
            np.minimum(hold_min, s.root_height, out=hold_min)
            # The task's OWN standing predicate, all 13 conjuncts, at the hardest exam
            # level. Judging by raw pelvis height would accept a headstand and a
            # knees-locked-backwards pose, both of which this project has already produced.
            stand_sum[:] += s.task_state["standing"]
        if record:
            frames_q.append(s.qpos.copy())
            frames_v.append(s.qvel.copy())

    for w in range(n_way):
        goal = targets[:, w, :].astype(np.float32)
        for t in range(seg_steps):
            advance(current + (goal - current) * ((t + 1) / seg_steps), False)
        current = goal
    zero = np.zeros((n, env.nu), dtype=np.float32)
    for _ in range(hold_steps):
        advance(zero, True)

    total_steps = n_way * seg_steps + hold_steps
    final = env.state.root_height.copy()
    stand_frac = stand_sum / hold_steps
    # The predicate is the thing itself and dominates once anything reaches it. Everything
    # else is the ramp that gets a search there from a flat landscape of bodies lying at 0.1,
    # and every part of that ramp is written to be unearnable by a jump:
    #   * the MINIMUM height over the hold carries the weight, because a body that leaps and
    #     collapses has a low minimum however high its mean;
    #   * the peak is capped at standing height, so going higher than a stand buys nothing;
    #   * flight time is subtracted outright.
    capped_peak = np.minimum(peak, env.prepared.standing_height)
    score = (4.0 * stand_frac + 2.0 * hold_min + 0.5 * (hold_sum / hold_steps)
             + 0.3 * capped_peak - 2.0 * (flight / total_steps))
    if record:
        return (score, final, peak, stand_frac,
                np.stack(frames_q, 1), np.stack(frames_v, 1))
    return score, final, peak, stand_frac, None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pop", type=int, default=384)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--waypoints", type=int, default=5)
    ap.add_argument("--elite", type=float, default=0.12)
    ap.add_argument("--explore", type=float, default=0.35,
                    help="injected sampling noise, decayed to zero over the run")
    ap.add_argument("--sigma-floor", type=float, default=0.08)
    ap.add_argument("--patience", type=int, default=8,
                    help="iterations without improvement before reseating on the champion")
    ap.add_argument("--warm-start", action="store_true",
                    help="resume each family from its saved parameters")
    ap.add_argument("--min-stand-frac", type=float, default=0.9,
                    help="fraction of the hold the standing predicate must be true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", type=str, default="supine,prone,side_left,side_right")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/fallen/getup_refs_v2.npz")
    args = ap.parse_args()

    cfg = Config.load(REPO_ROOT / "configs" / "getup.yaml")
    bank = np.load(REPO_ROOT / cfg.getup.bank_path, allow_pickle=False)
    lab = bank["label"]
    rng = np.random.default_rng(args.seed)

    task = SearchTask(replace(cfg.getup, ref_path="", ball_enabled=False,
                              standing_reset_frac=0.0, midrise_reset_frac=0.0,
                              rising_reset_frac=0.0, track_reset_frac=0.0))
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, task, num_envs=args.pop, num_workers=8,
        decimation=cfg.env.decimation, max_episode_steps=1_000_000,
        action_filter_hz=cfg.env.action_filter_hz, seed=args.seed,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode)
    # The hardest exam level, not the one the curriculum happens to sit on. A reference is
    # worth having only if it satisfies the predicate the run is ultimately graded by.
    task._exam_level = len(cfg.getup.exam_levels) - 1  # noqa: SLF001
    seg_steps = max(1, int(round(SEG_SECONDS / env.dt)))
    hold_steps = max(1, int(round(HOLD_SECONDS / env.dt)))
    print(f"control {1/env.dt:.0f} Hz, {args.waypoints} waypoints x {seg_steps} steps "
          f"+ {hold_steps} hold, {args.waypoints * env.nu} search dimensions, "
          f"pop {args.pop}, exam level {task._exam_level}", flush=True)  # noqa: SLF001

    clips_q, clips_v, bounds = [], [], []
    for family in args.families.split(","):
        pool = np.flatnonzero(lab == family)
        if pool.size == 0:
            print(f"  {family}: no poses in the bank, skipped")
            continue
        task.start_q = bank["qpos"][pool[rng.integers(0, pool.size)]]
        param_path = REPO_ROOT / f"data/fallen/getup_params_{family}.npy"
        # Warm start from a previous search when its shape matches. CEM converges its own
        # sigma to nothing, so continuing a search means reopening it, not restarting it.
        mu = np.zeros((args.waypoints, env.nu))
        if args.warm_start and param_path.exists():
            prev = np.load(param_path)
            if prev.shape == mu.shape:
                mu = prev
                print(f"  {family}: warm start from {param_path.name}")
        sigma = np.full((args.waypoints, env.nu), 0.7)
        best_z, best_stand, best_score, best_params = 0.0, 0.0, -1e9, mu.copy()
        k = max(4, int(args.pop * args.elite))
        # Rank weights, CMA-ES style: the top elite gets an order of magnitude more say than
        # the last. A plain elite mean cannot be moved by one outstanding sample, and that is
        # exactly what happened here: a candidate reaching pelvis 0.874 sat in an elite of 76
        # for twenty-five iterations while the mean of that elite stayed at 0.127, so the
        # sampler kept drawing around a posture the champion had already beaten.
        w = np.log(k + 0.5) - np.log(np.arange(1, k + 1))
        w = (w / w.sum())[:, None, None]
        stale = 0
        for it in range(args.iters):
            # Injected exploration on top of the elite spread, decayed over the run. Without
            # it the first search collapsed from sigma 0.70 to 0.17 by iteration 20 and then
            # spent 20 more iterations resampling the same basin: the plateau at pelvis 0.720
            # was the sampler running out of variance, not the body running out of strength.
            extra = args.explore * max(0.0, 1.0 - it / max(1, args.iters - 1))
            spread = np.sqrt(sigma ** 2 + extra ** 2)
            cand = np.clip(
                mu[None] + spread[None] * rng.normal(size=(args.pop, args.waypoints, env.nu)),
                -1.0, 1.0)
            cand[0] = mu                      # always evaluate the mean itself
            cand[1] = best_params             # and never lose the incumbent
            score, final, peak, stand, _, _ = rollout(env, cand, seg_steps, hold_steps)
            order = np.argsort(-score)[:k]
            elite = cand[order]
            mu = (w * elite).sum(0)
            sigma = np.maximum(elite.std(0), args.sigma_floor)
            # Best by SCORE, not by final height: a candidate whose final frame happens to be
            # high while it topples is not the one to keep, and the score is what separates
            # the two.
            top = int(np.argmax(score))
            if score[top] > best_score + 1e-4:
                best_score = float(score[top])
                best_z, best_stand = float(final[top]), float(stand[top])
                best_params = cand[top].copy()
                stale = 0
            else:
                stale += 1
            # Nothing better in `patience` iterations: reseat the search on the champion and
            # reopen the spread. Global exploration has said what it has to say; the remaining
            # value is in refining the best posture actually found.
            if args.patience and stale >= args.patience:
                mu = best_params.copy()
                sigma = np.full_like(sigma, 0.20)
                stale = 0
                print(f"  {family:<10} it {it:>2}  RESEAT on the champion "
                      f"(pelvis {best_z:.3f})", flush=True)
            print(f"  {family:<10} it {it:>2}  best pelvis {best_z:.3f} stand {best_stand:.2f}"
                  f"  | iter pelvis {float(final.max()):.3f} stand {float(stand.max()):.2f}"
                  f"  elite {float(final[order].mean()):.3f}"
                  f"  peak {float(peak.max()):.3f}  spread {spread.mean():.2f}", flush=True)

        # Keep the parameters whatever the verdict. A rejected 0.72 is the starting point of
        # the next search, and the first run threw one away.
        np.save(param_path, best_params)
        # Replay the best candidate to record it. Deterministic: fixed start, no domain
        # randomisation, no pushes, so this reproduces the scoring rollout exactly.
        replay = np.tile(best_params[None], (args.pop, 1, 1))
        _s, final, peak, stand, fq, fv = rollout(
            env, replay, seg_steps, hold_steps, record=True)
        z, sf = float(final[0]), float(stand[0])
        keep = sf >= args.min_stand_frac
        print(f"  {family:<10} SOLUTION pelvis {z:.3f}, peak {float(peak[0]):.3f}, "
              f"standing {sf * 100:.0f}% of the hold  {'KEPT' if keep else 'REJECTED'}")
        if keep:
            clips_q.append(fq[0])
            clips_v.append(fv[0])
    env.close()

    if not clips_q:
        print("\nNO FEASIBLE GET-UP FOUND. Read that literally: under these servos and this "
              "action range, a direct search over open-loop servo trajectories did not stand "
              "this body up from lying. A closed-loop policy may still succeed where an "
              "open-loop schedule cannot, but no reference clip can be handed to it, and any "
              "reward that assumes one is assuming something unproven.")
        return 1

    off = 0
    for c in clips_q:
        bounds.append(off)
        off += c.shape[0]
    bounds.append(off)
    np.savez_compressed(
        args.out, qpos=np.concatenate(clips_q), qvel=np.concatenate(clips_v),
        bounds=np.array(bounds, dtype=np.int64),
        nq=env.model.nq, nv=env.model.nv, dt=env.dt,
        model_sha1=bank["model_sha1"], generator_version=3)
    print(f"\n{len(clips_q)} feasible clips -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
