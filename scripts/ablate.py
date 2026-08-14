"""Find which change broke a run, by turning them off one at a time.

Written because six changes landed together and the resulting run fell over on 100% of
evaluations from the very first one. With everything changed at once there is no way to
attribute that, which is the same mistake as bundling a fix and hoping.

Method: start from the CURRENT (broken) config and disable one change per run. Whichever
disabling restores the fall rate is the culprit. Each arm is short, because the defect shows
up within the first hundred iterations, so this costs minutes rather than a full run.

    python scripts/ablate.py --iterations 200
    python scripts/ablate.py --only exploration,commands

Every arm writes to `runs/ablate-<name>-<stamp>` and the summary is printed at the end,
sorted by fall rate. Arms are run SEQUENTIALLY on purpose: they each want all ten physics
workers, and interleaving them would measure contention rather than the change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Each arm reverts ONE change back to what the last acceptable run used. `None` marks the
#: control, which changes nothing and is what every other arm is compared against.
ARMS: dict[str, dict] = {
    "control": {},
    # Exploration noise: the previous run had log_std pinned at std 1.0 with an entropy
    # bonus. It was frozen, but it was also the configuration that reached 35% falls.
    "exploration": {
        "ppo": {"log_std_max": 0.0, "entropy_coef": 0.005},
        "network": {"init_noise_std": 1.0},
    },
    # Command envelope: backward 0.8 and lateral 0.6 are genuinely harder than the old
    # 0.5 and 0.4, and the policy now practises them 12% of the time instead of 4%.
    "commands": {"task": {"lin_vel_x_range": [-0.5, 1.5], "lin_vel_y_range": [-0.4, 0.4]}},
    # Mid-episode resampling: the policy had never seen a command change while walking.
    # A very long hold makes this effectively once-per-episode again.
    "resample": {"task": {"command_hold_range": [999.0, 1000.0]}},
    # Clock authority: 10% of segments train with the gait schedule switched off.
    "free_gait": {"task": {"free_gait_prob": 0.0}},
    # The heading term itself, which measurement showed is unlearnable as formulated.
    "heading": {"task": {"w_heading": 0.0}},
    # Swing target: was 0.25 s, now derived from the schedule with 0.44 s as the fallback.
    "air_time": {"task": {"air_time_target": 0.25}},
    # Both culprits together. Single-factor arms cannot tell whether the two interact, and
    # 28% and 35% separately does not imply anything in particular jointly.
    "both": {"task": {"w_heading": 0.0, "free_gait_prob": 0.0}},
}


def deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


#: Which config section holds the task settings, per run.task. This is a trap: an AMP run
#: reads `amp_task` and ignores `task` entirely, and `amp.yaml` has no `task` section at all.
#: An earlier version of this script patched `task` for every arm, so six of seven arms
#: changed nothing and returned byte-identical numbers. The giveaway was that they were
#: identical rather than merely similar.
TASK_SECTION = {"locomotion": "task", "amp": "amp_task", "tracking": "tracking"}


def run_arm(name: str, patch: dict, base_config: Path, iterations: int,
            init_from: Path | None) -> dict:
    raw = yaml.safe_load(base_config.read_text())

    # Retarget any `task` patch at the section this run actually reads.
    section = TASK_SECTION.get(raw.get("run", {}).get("task", "locomotion"), "task")
    if "task" in patch and section != "task":
        patch = {**{k: v for k, v in patch.items() if k != "task"}, section: patch["task"]}
    if any(key not in raw and key in TASK_SECTION.values() for key in patch):
        raise SystemExit(f"arm {name} patches a section absent from {base_config.name}")
    merged = deep_merge(raw, patch)
    merged.setdefault("run", {})["name"] = f"ablate-{name}"

    tmp = REPO_ROOT / f".ablate-{name}.yaml"
    tmp.write_text(yaml.safe_dump(merged, sort_keys=False))

    cmd = [str(REPO_ROOT / ".venv/bin/python"), "-u", "-m", "humanoid_rl.train",
           "--config", str(tmp), "--iterations", str(iterations)]
    if init_from:
        cmd += ["--init-from", str(init_from)]

    started = time.perf_counter()
    print(f"\n--- {name} ---", flush=True)
    print(f"    {json.dumps(patch) if patch else 'no change (control)'}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"    FAILED: {proc.stderr.strip().splitlines()[-1] if proc.stderr else '?'}")
        return {"arm": name, "failed": True}

    runs = sorted((REPO_ROOT / "runs").glob(f"ablate-{name}-*"))
    if not runs:
        return {"arm": name, "failed": True}
    metrics = runs[-1] / "metrics.jsonl"
    evals = [json.loads(l) for l in metrics.read_text().splitlines()
             if "eval/episode_return" in l]
    if not evals:
        return {"arm": name, "failed": True}

    falls = [e["eval/fall_rate"] for e in evals]
    result = {
        "arm": name,
        "run": runs[-1].name,
        "evals": len(evals),
        "fall_rate_best": min(falls),
        "fall_rate_last": falls[-1],
        "return_best": max(e["eval/episode_return"] for e in evals),
        "heading_best": min(
            (e.get("eval/heading_error_deg", 999.0) for e in evals), default=999.0),
        "minutes": (time.perf_counter() - started) / 60.0,
    }
    print(f"    best falls {result['fall_rate_best']:.0%}  best return "
          f"{result['return_best']:.0f}  best heading {result['heading_best']:.0f}d  "
          f"({result['minutes']:.1f} min)", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "amp.yaml")
    ap.add_argument("--iterations", type=int, default=200,
                    help="per arm; the defect appeared by iteration 50 in the broken run")
    ap.add_argument("--init-from", type=Path,
                    default=REPO_ROOT / "runs/phase2-walking-121M/checkpoints/best.pt")
    ap.add_argument("--only", type=str, default="", help="comma-separated arm names")
    args = ap.parse_args()

    wanted = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARMS)
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        print(f"unknown arms {unknown}; available: {list(ARMS)}")
        return 2

    print(f"ablation: {len(wanted)} arms x {args.iterations} iterations, run sequentially")
    results = [run_arm(name, ARMS[name], args.config, args.iterations, args.init_from)
               for name in wanted]

    ok = [r for r in results if not r.get("failed")]
    ok.sort(key=lambda r: r["fall_rate_best"])
    print("\n\n=== ablation summary, best fall rate first ===\n")
    print(f"{'arm':<14}{'best falls':>12}{'best return':>13}{'best heading':>14}")
    for r in ok:
        print(f"{r['arm']:<14}{r['fall_rate_best']:>11.0%}{r['return_best']:>13.0f}"
              f"{r['heading_best']:>13.0f}d")
    if ok:
        control = next((r for r in ok if r["arm"] == "control"), None)
        if control:
            better = [r for r in ok if r["fall_rate_best"] < control["fall_rate_best"] - 0.10]
            print()
            if better:
                print("Disabling these improved on the control, worst offender first:")
                for r in sorted(better, key=lambda r: r["fall_rate_best"]):
                    print(f"  {r['arm']}: {r['fall_rate_best']:.0%} against the control's "
                          f"{control['fall_rate_best']:.0%}")
            else:
                print("No single arm improved on the control. The regression is either a "
                      "combination, or something not covered by these arms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
