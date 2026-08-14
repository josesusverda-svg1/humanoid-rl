# Human-like humanoid locomotion with RL, on Apple Silicon

A humanoid that learns to walk, turn and navigate, trained entirely locally on a MacBook
Pro. No cloud, no CUDA, no paid APIs.

Every architectural decision here was made by measuring on the target machine rather than
by reputation. The reasoning and the numbers live in **[DESIGN.md](DESIGN.md)**; the raw
research it came from is archived in **[docs/research/](docs/research/)**.

---

## Quick start

```bash
# 1. verify the machine (Metal GPU is real, all performance cores usable, video works)
.venv/bin/python scripts/hwcheck.py

# 2. train
.venv/bin/python -m humanoid_rl.train --config configs/default.yaml

# 3. watch it, in a second terminal
.venv/bin/python scripts/dashboard.py
```

The dashboard opens at `http://127.0.0.1:8000`. Training writes JSONL; the dashboard reads
it. Neither depends on the other, so either can be restarted without disturbing the other.

---

## Setup

Requires macOS on Apple Silicon, [Homebrew](https://brew.sh), and Node 18+ for the
dashboard frontend.

```bash
brew install uv ffmpeg
cd /path/to/HumonoidAI
uv sync                    # creates .venv with Python 3.12 and pinned dependencies
uv sync --extra dashboard  # adds FastAPI and uvicorn
```

Python 3.12 is pinned deliberately: it has the best wheel coverage across MuJoCo, PyTorch
and MLX on macOS arm64, and 3.13 still has gaps in this specific stack.

Verify everything before training. `hwcheck.py` exits non-zero on failure and proves each
claim by measurement rather than trusting an `is_available()` flag:

```bash
.venv/bin/python scripts/hwcheck.py
```

It checks Metal GPU throughput against CPU (a silent CPU fallback costs 4-6x and is
otherwise invisible), that parallel physics saturates the performance cores, and that
headless rendering plus mp4 encoding work.

---

## The stack, and why

| Layer | Choice | Reason |
| --- | --- | --- |
| Physics | **CPU MuJoCo**, threads, `mj_step(nstep=4)` | There is no working GPU physics path on Apple Silicon: `jax-metal` was abandoned in 2024, and Isaac Gym / MJX / MuJoCo Warp are CUDA-only |
| Networks | **PyTorch on MPS** | 6.2x the CPU on matmul; MLX ties it end to end but PyTorch has the ecosystem. MLX stays installed as a documented fallback |
| Parallelism | 10 worker threads, `USER_INITIATED` QoS | MuJoCo releases the GIL, so threads give real parallelism with no IPC. macOS has no CPU affinity API; QoS is how you stay on performance cores |
| Control | PD position servos | Learning from raw torques is far harder, and every strong humanoid result uses position control |
| Model | MimicKit humanoid (Apache-2.0), 28 DoF | **Box feet.** MuJoCo's default humanoid has capsule feet, which are line contacts and physically cannot produce a heel-to-toe roll |

Measured throughput: **~95,000 environment steps/second** in the environment,
**~56,000** end to end including the PPO update, at 4096 environments.

---

## Commands

### Training

```bash
.venv/bin/python -m humanoid_rl.train --config configs/default.yaml     # velocity-command walking
.venv/bin/python -m humanoid_rl.train --config configs/tracking.yaml    # mocap tracking
.venv/bin/python -m humanoid_rl.train --resume runs/humanoid-2026...    # resume
```

Useful overrides: `--iterations N`, `--num-envs N`, `--eval-interval N`.

For long runs, prevent macOS sleeping and keep the log readable:

```bash
PYTHONUNBUFFERED=1 nohup caffeinate -dimsu .venv/bin/python -u -m humanoid_rl.train \
  --config configs/default.yaml > /tmp/train.log 2>&1 &
```

Ctrl+C saves a checkpoint before exiting. A second Ctrl+C exits immediately.

### Watching

```bash
.venv/bin/python scripts/dashboard.py                # UI + API at :8000
.venv/bin/python scripts/watch_training.py           # terminal alerts, silent unless notable
```

The watchdog stays quiet while things are healthy and emits one line per real event:
`WALKING`, `TRACKING`, `BEST`, `MILESTONE`, plus the failure alarms `SATURATION`,
`SKATING`, `REGRESSION`, `SLOWDOWN`, `STALLED`, `CRASH`. The failure alarms matter more
than the good news: **silence has to mean healthy, never "died twenty minutes ago"**.

### Looking at the gait

```bash
.venv/bin/python scripts/render.py                                  # newest run, best checkpoint
.venv/bin/python scripts/render.py --run runs/... --width 1280 --height 960
```

Renders a scripted command sequence (stand, walk, turn both ways, fast, sidestep, backward,
stop) with a telemetry overlay showing commanded vs actual velocity, foot contacts and
height. Safe to run during training.

The trainer also renders automatically on every new best score and every 5 evaluations,
into the dashboard's Videos tab. **Do this often.** Three separate failures in this project
scored excellently on every metric and were only visible in video.

```bash
.venv/bin/python scripts/gait_report.py --run runs/... --checkpoint best.pt
```

Scores a checkpoint on the quantities that distinguish walking from shuffling, next to the
same numbers measured on the mocap and on real human walking:

```
                   policy    mocap   real human walking
double_support       0.90     0.22   0.20 - 0.25   (1.0 means never lifting a foot)
step_rate            1.81     1.27   1.6 - 2.0     foot strikes per second
stance_width         0.61     0.26   0.10 - 0.15 m
stride_length        0.06     0.49   0.6 - 0.8 m   per strike
```

That is a real reading, from a policy that scored 0.91 on gait symmetry. Note `step_rate`
alone looks healthy: it was twitching its feet at a normal cadence inside a static brace.
The combination is what identifies the failure.

### Is it actually walking? (dashboard, Gait quality tab)

Episode return cannot tell walking from several things that are not walking. Every gait
failure in this project scored well on return and was caught by a person watching a video.
The **Gait quality** tab scores the gait against measured human walking instead, on axes that
each correspond to one of those failures:

| group | measures | the failure it catches |
| --- | --- | --- |
| Posture | torso upright, head height, stance width | folding at the waist; standing splay-legged |
| Rhythm | lead changes/s, double support | rocking in a fixed split stance; refusing to lift a foot |
| Smoothness | vertical bounce, foot slip | pogoing; skating |
| Balance | left/right evenness | a one-sided gait |
| Reliability | fall rate, speed tracking | falling; ignoring the command |

Each gauge draws the human range as a bright band with the current value as a marker, so a
defect is visible without reading any numbers: a marker outside its band **is** the defect.
The axis spans one band width either side, so "just outside" and "wildly outside" look
different. The headline strip names the three measurements furthest from human.

The bands are measurements of human walking, not thresholds someone picked, and each one is
documented in `humanoid_rl/gait_score.py` with the specific failure that motivated it. The
same scoring is available from Python via `score_row` and `human_summary`, so it is not
locked inside the dashboard.

Scored against gaits we already know are bad, it discriminates correctly: the rocking policy
scores **Rhythm 0%**, and the one-legged Phase 2 walker scores **Posture 100%** but
**Smoothness 0%** for its foot slip.

### Watching it learn (dashboard, Learning tab)

Training is not a black box. Every iteration is the same four steps and each one logs
something you can look at, so the tab lays them out in order:

1. **Collect** thousands of humanoids act in parallel, every step scored by the reward terms
2. **Judge** the critic's prediction versus what happened gives the advantage
3. **Nudge** weights move so better-than-expected actions become more likely
4. **Check** measure how far the policy moved, and adjust the learning rate to hold it there

Three charts, one per question:

* **What is it being rewarded for?** Every reward term as a share of the total signal, over
  training. The widest band is what the humanoid is currently paid to do.
* **Which weights changed?** How far each layer moved in one update, as a percentage of its
  own norm (`weight_change/*`, logged by `PPO._layer_change`). Relative rather than absolute
  because the layers differ hugely in size. The actor's output layer typically moves ~60%
  while its hidden layers move 2-3%, since it is initialised deliberately small.
* **How big was the step?** KL divergence against the adaptive learning rate. They should
  mirror each other: that is the trainer steering its own step size.

### The gait flip-book (dashboard, Flip-book tab)

Multi-view stick figures, captured every 4M environment steps (`eval.skeleton_every_m_steps`).

**No rendering is involved**, which is the whole point. A body's screen position under an
orthographic camera is two dot products, so a pose from six angles is a small matrix
multiply. Measured against the video path on the same policy:

| | video | flip-book |
| --- | --- | --- |
| time | 18 s | **0.59 s** |
| size | 11 MB | **80 KB** |

That is cheap enough to run often, so a run accumulates a scrubbable record of the gait
developing rather than a handful of expensive clips. Six angles because a defect is often
invisible from one: the one-sided gait looked normal head-on and was obvious from the side,
and the split-stance rocking was the reverse. Blue is the left side, red the right, and the
ground line is absolute so bounce and float are honest.

### Driving it yourself with the keyboard

```bash
.venv/bin/mjpython scripts/play.py --skin
```

`mjpython`, not `python`: on macOS MuJoCo's interactive viewer must own the main thread, and
`launch_passive` raises otherwise. It ships with the mujoco wheel and is already in the venv.

No retraining is involved. The policy is velocity-command conditioned, so this writes your
keypresses into the same observation slot training filled with a random command each episode.

| key | effect |
|---|---|
| `W` / `S` | forward speed ∓0.25 m/s |
| `A` / `D` | turn left / right ∓0.25 rad/s |
| `Q` / `E` | strafe left / right ∓0.20 m/s |
| `1` `2` `3` | preset walk at 0.5 / 1.0 / 1.5 m/s |
| `X` stop · `Z` reset · `C` camera · `Esc` quit |

Presses adjust a **persistent** command rather than needing to be held, because MuJoCo's
viewer delivers key-down events only (`IsKeyDownEvent(act) { return act == GLFW_PRESS; }`),
with no key-up and no auto-repeat. The command then eases toward the target over ~0.3 s,
since training only ever resampled it at episode reset, so the policy has never seen a step
change mid-episode. Commands are clamped to the trained ranges; outside them it extrapolates.

```bash
.venv/bin/python scripts/play.py --selftest      # no window needed
```

Checks the controls in two separate parts, which matters more than it sounds. First the
plumbing, with no policy involved: does pressing `A` put a positive turn rate into the
observation? Then, separately, what the checkpoint actually does. Conflating them is a trap,
because a policy that cannot turn yet looks exactly like a reversed key. On the Phase 2
checkpoint the second part reported both turn commands producing rightward rotation, which
is the one-sided gait showing up through the controls.

### Making it look human

```bash
.venv/bin/python scripts/build_skin.py --verify --render /tmp/skin.png
```

Wraps the capsules in a human body mesh and writes `assets/humanoid_skinned.xml`. Pass
`--skin` to `render.py` or `play.py` to use it.

**It cannot affect the physics.** MuJoCo's `skin` is consumed only by the visualiser, and
`--verify` proves it here rather than citing the docs: masses, inertias, contact parameters
and a 1000-step trajectory all come out bitwise identical. That check exists because the
obvious cheaper approach is booby-trapped: hanging visual meshes on bodies with
`contype="0" conaffinity="0"` stops them colliding but does **not** remove their inertia, so
a purely cosmetic edit silently moved a test body's mass from 3.665 kg to 11.559 kg.

The mesh is [MakeHuman](https://github.com/makehumancommunity/makehuman)'s base body, used as
three CC0 data files (`base.obj`, `default.mhskel`, `default_weights.mhw`) with no GUI, no
account and no Blender. Each file carries `"license": "CC0"` internally. SMPL was rejected for
the same "No Distribution" clause that already ruled out LAFAN1 and LocoMuJoCo, and Mixamo for
forbidding redistribution of raw assets.

Three things the pipeline has to get right, all of them found by looking at a render:

* **`base.obj` is not just a body.** It carries hidden helper geometry for fitting clothing:
  `helper-skirt` put the humanoid in a floor-length dress, `helper-hair` gave it spikes, and
  the `joint-*` groups scattered cubes over it. Only the `body` group is kept.
* **Terminal segments cannot infer their own direction.** "Point away from the parent" sends
  the foot straight down, because the shin sits directly above it, which tipped the foot mesh
  under the floor. Head, hand and foot directions are stated explicitly.
* **The torso must end at the shoulder line, not the head.** Ending it at the head compressed
  the chest to 0.72 scale and put the mesh's shoulders 7 cm below the arm bodies they bind to,
  which is what produced peaked, creased shoulders.
* **Do not stretch the mesh onto the skeleton's proportions.** This humanoid has no neck: its
  head body sits at z=1.339 with the shoulders at 1.359, so the head is mounted *at* shoulder
  height. Fitting each mesh segment to its own bone length faithfully reproduced that, and the
  human came out neckless. `preserve_proportions=True` (the default) instead uses ONE global
  scale, set by hip-to-ground so the feet still meet the floor, and takes only the *direction*
  of each segment from the bones. The skeleton drives the motion; it no longer dictates how
  long a neck is.
* **The muscle morph is deliberately overdriven, and that has a cost.** MakeHuman's maximum
  muscle is athletic-normal, thickening the thigh by 10%, so the target is applied at weight
  2.0. Extrapolating a linear morph that far makes the surface self-intersect, which renders
  as torn slivers across the chest and deltoids; a neutral / 1.0 / 2.0 comparison shows the
  tearing scaling with the weight. Six passes of Taubin lambda-mu smoothing repair it while
  keeping the definition, because alternating the sign removes the high-frequency
  self-intersection without flattening the low-frequency shape. Plain Laplacian smoothing
  would erase exactly the musculature we went out of range to get.

### Mocap

```bash
.venv/bin/python scripts/retarget.py --inspect data/mocap/100STYLE/Neutral
.venv/bin/python scripts/retarget.py --glob 'data/mocap/100STYLE/Neutral/*.bvh' --render
```

`--inspect` prints the source skeleton, which is how the marker map gets written against
real files instead of guessed joint names.

### Benchmarks and tests

```bash
.venv/bin/python scripts/smoke_env.py       # correctness + a throughput floor
.venv/bin/python scripts/bench_arch.py      # the parallelism shoot-out
.venv/bin/python scripts/bench_networks.py  # torch-cpu vs torch-mps vs mlx
```

Run `smoke_env.py` after touching `vec_env.py` or any task. Its throughput floor is a
deliberate tripwire for per-environment Python creeping back into the worker loop, which
is the single mistake that costs 5x.

---

## Configuration

One YAML controls a run and is snapshotted into the run directory, so a checkpoint always
carries the config that produced it.

| Section | What it controls |
| --- | --- |
| `run` | name, seed, device, **`task`** (`locomotion` or `tracking`), total steps |
| `env` | model, `num_envs`, `num_workers` (null = one per performance core), decimation, episode limit |
| `network` | layer sizes, activation, initial exploration noise |
| `ppo` | horizon, discount, learning rate, epochs, clipping, **`bounds_loss_coef`**, `log_std_max` |
| `task` | velocity-command ranges and every reward weight |
| `tracking` | DeepMimic weights and kernel widths, termination thresholds |
| `domain_rand` | friction, mass, gains, sensor noise, push strength, model pool size |
| `eval` | evaluation interval and episodes, video cadence and size |
| `log` | checkpoint interval and retention |

Unknown keys raise rather than being ignored. A silently swallowed typo is one of the more
expensive ways to waste a multi-hour run.

Fields set to `null` are detected at runtime, which keeps the same file portable to a
machine with a different core count.

### Settings worth understanding before changing

- **`ppo.bounds_loss_coef: 10.0`** penalises the policy *mean* leaving [-1, 1]. Without it,
  the environment clips actions and nothing pushes back, so saturation becomes a free
  maximum-effort strategy. Measured with it disabled: mean |action| reached 2.3 (max 22.8)
  and deterministic evaluation collapsed from 30% to 100% falls **while the training curve
  kept rising**.
- **`task.w_feet_air_time`** is what produces stepping. Remove it and a velocity-tracking
  policy reliably converges on a shuffle that scores perfectly and looks nothing like
  walking.
- **`task.w_torso_upright` / `terminate_torso_upright`** measure the *torso*, not the
  pelvis. A humanoid can hold its pelvis level and at the right height while folding flat
  at the waist, satisfying every pelvis-based term.

---

## Layout

```
humanoid_rl/
  hardware.py        machine probe; all sizing derives from this
  qos.py             macOS QoS, since there is no CPU affinity API
  config.py          typed YAML config
  train.py           training entry point
  evaluate.py        deterministic evaluation
  render.py          offscreen mp4 rendering
  envs/
    vec_env.py       threaded vectorised environment (performance-critical)
    model_prep.py    torque motors -> PD servos, standing pose, sensors
    domain_rand.py   randomisation and the model pool
  tasks/
    base.py          Task interface and BatchState
    locomotion.py    velocity-command walking
    tracking.py      DeepMimic motion tracking
  motion/
    bvh.py           BVH parser and forward kinematics
    retarget.py      marker IK retargeting onto our model
    library.py       flat, index-addressable clip storage
  algos/
    networks.py      actor-critic, observation normalisation
    ppo.py           PPO with adaptive LR and the bounds loss
dashboard/
  backend/app.py     FastAPI, reads the JSONL directly
  frontend/          React + Vite + Recharts
```

---

## Adding a new skill (jump, climb, anything)

The architecture is built so a new skill is **a new Task, a terrain type, and reward
terms**, never a change to `vec_env.py`. Concretely:

**1. Write a Task** in `humanoid_rl/tasks/`, subclassing `Task` from `tasks/base.py`:

```python
class JumpTask(Task):
    reward_term_names = ("height_gain", "landing", "alive", "ctrl")

    @property
    def task_obs_dim(self) -> int:
        return 4                       # e.g. obstacle distance, height, and a phase

    def init_state(self, state, rng): ...       # allocate arrays in state.task_state
    def reset_batch(self, state, indices, rng): ...   # randomise per episode
    def observe_batch(self, state, out): ...    # write (N, task_obs_dim)
    def reward_batch(self, state, terms): ...   # write (N, n_terms), return (N,)
    def terminated_batch(self, state): ...      # return (N,) bool
```

**Every method takes the whole batch.** Write numpy over `(num_envs, ...)` arrays, never a
Python loop over environments. This is not style: with ten worker threads the GIL
serialises all Python bytecode, and per-environment callbacks measured roughly **five times
slower**. `tasks/base.py` documents the measurements.

`BatchState` already gives you body state, body-frame velocities, foot contact, foot air
time, foot slip velocity, torso uprightness, head height and key body positions, all
computed once and vectorised. Prefer these over recomputing anything.

**2. Register it** in `Trainer._build_task` in `train.py` and add a `run.task` value.

**3. Optional hooks** you are likely to want:
- `reset_pose` for an absolute initial state (used by tracking to start in a reference frame)
- `action_offset` to change what a zero action means (tracking sets it to the reference pose)
- `on_batch_end` for cross-environment bookkeeping such as a curriculum level
- `eval_metrics` to report skill-specific numbers, which flow to the dashboard automatically

**4. Write a config** in `configs/`, copying `default.yaml` and replacing the task section.

**5. Terrain**, when you need it, goes in `humanoid_rl/terrain/` as a heightfield
generator. Note from the research: `mujoco.Renderer` has no `update_hfield`, so regenerated
terrain will not reach the GPU and videos will silently show stale ground. That path needs
its own `GLContext` and `MjrContext`. See DESIGN.md section 10.

---

## Hard-won lessons

Each of these cost real debugging time and is documented where it applies:

1. **Per-environment Python is the enemy.** Four `mju_rotVecQuat` calls per step cost
   23.78 us under thread contention, more than the physics they supported. Workers do C
   only; all math is vectorised on the main thread.
2. **The model's "joint stiffness" was PD gains in disguise.** Isaac Gym reads those fields
   as drive gains; MuJoCo reads them as passive springs pulling to the T-pose. Left in
   place, the arms would not move while the actuator applied a constant 300 N*m.
3. **Metrics hide broken gaits.** A fold-forward lurch and an action-saturated policy both
   scored excellently. Render video early and often.
4. **Watch the data, not just the reward.** A "dead" velocity reward term looked like a
   badly tuned kernel; the kernel was correct and the *reference data* had 64 rad/s
   artifacts from independent per-frame IK.
5. **Verify claims on the machine.** Three well-sourced research claims (gradient penalty
   impossible on MPS, small MLPs favour CPU, the AMP humanoid will not load) all turned out
   false here. DESIGN.md section 9.
6. **A metric you add to catch a defect can be gamed by a worse defect.** `gait_symmetry`
   started as `min(left_stance, right_stance) / max(...)`, which is 1.0 for an even gait and
   *also* 1.0 for a humanoid that plants both feet and never lifts either. A policy under a
   symmetry loss found the second reading: 0.91 symmetry, 90% double support, a 61 cm wide
   stance, 6 cm of travel per foot strike. It is now defined over swing time, so standing
   scores 0, and `double_support` and `stance_width` are reported alongside it because
   neither can be satisfied by refusing to step. `scripts/gait_report.py` scores any
   checkpoint on these against the mocap.
7. **Prefer constraints that live in the data over penalties on the policy.** The symmetry
   *penalty*, `||pi(s) - M(pi(M(s)))||^2`, is exactly zero for any policy that ignores its
   input, so "hold a symmetric pose" is one of its global optima and PPO went straight
   there. Mirror *data augmentation* states the same belief by adding each transition's
   reflection to the batch, which is unbiased for a symmetric body and has no degenerate
   solution, because a constant policy still earns no return. See `ppo.symmetry_augment`.
8. **Check which distribution the defect actually came from.** The one-sided gait was
   blamed on asymmetric mocap. Measured, the reference was 1.08 on stance fraction while
   the policy was 1.90, so most of the asymmetry was RL's, not the data's. Mirroring the
   clips (`mirror_clips`) was still right and is kept, but it was not the fix.

---

## Licensing

Code in this repository is yours. Third-party components:

| Component | Licence | Note |
| --- | --- | --- |
| MimicKit humanoid MJCF | Apache-2.0 | body model |
| MuJoCo, PyTorch, NumPy, SciPy | permissive | |
| **100STYLE mocap** | **CC BY 4.0** | requires attribution; no registration |
| **MakeHuman base body** | **CC0 1.0** | `third_party/makehuman/`, visual skin only; each file states its own licence |
| ffmpeg (Homebrew) | GPL | invoked as a subprocess, so it does not affect this code |

**Avoided deliberately:** LAFAN1 and LocoMuJoCo's ready-made clips are CC BY-NC-**ND**. The
no-derivatives clause means retargeting them produces an infringing derivative, which would
quietly make the whole project non-redistributable. AMASS is non-commercial and needs
registration. For the visual mesh, **SMPL / SMPL-X** fail the same test: their licence has a
section headed "No Distribution". **Mixamo** permits derivatives but forbids redistributing
raw assets, so it cannot live in this repo. **Daz, MetaHuman and Ready Player Me** are
proprietary EULAs. Microsoft Rocketbox (MIT, 115 rigged avatars) is a clean fallback if
photoreal clothed characters are ever wanted.
