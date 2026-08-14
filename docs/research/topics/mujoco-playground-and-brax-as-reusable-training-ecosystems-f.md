# MuJoCo Playground and Brax as reusable training ecosystems for humanoid locomotion on Apple Silicon (M3 Max, no CUDA)

| field | value |
|---|---|
| apple_silicon_status | usable |
| current_version | mujoco_playground (PyPI: `playground`) 0.2.0; brax 0.14.2 |
| last_release_date | 2026-03-16 |
| maintenance_status | Both actively maintained, but with important scope changes. Playground: 0.2.0 released 2026-03-16, 0.1.0 on 2026-01-08, 0.0.5 on 2025-06-23 — steady cadence, repo activity through 2026-07. Brax: 0.14.2 on 2026-03-15, 0.14.1 on 2026-02-12, 0.14.0 on 2025-12-16 — actively developed, with 0.14.2 adding distributional PPO and 0.14.1 adding soft-sign policy clipping. HOWEVER, Brax's README states "Only `brax/training` is actively being maintained as of 0.13.0", and v0.12.4 (2025-06-13) "removed brax/v1" and redirected users to MJX and MuJoCo Playground. So Brax-as-physics-engine is wound down; Brax-as-PPO-library is alive. Supporting dependencies are also current: mujoco-mjx 3.11.0 (2026-07-28), warp-lang 1.16.0 (2026-08-03, daily dev builds), jax 0.11.0 (2026-07-16). By contrast jax-metal is effectively abandoned: last release 0.1.1 on 2024-10-08, ~22 months stale. |
| license | Apache-2.0 (both mujoco_playground and brax) |
| confidence | high |
| install | `uv venv --python 3.12 && source .venv/bin/activate && uv pip install playground brax mujoco mujoco-mjx  # do NOT install the `cuda` extra or `jax[cuda12]`; plain jax is CPU. warp-lang is pulled in as a hard dep and installs a macosx_11_0_arm64 wheel, but is CPU-only on this machine.` |

## Summary

Both projects are alive and actively released: MuJoCo Playground 0.2.0 (2026-03-16) and Brax 0.14.2 (2026-03-15). Brax has explicitly narrowed its scope — "Only `brax/training` is actively being maintained as of 0.13.0" — and now redirects users away from `brax/envs` and its own physics engines toward MJX and MuJoCo Playground. Playground has a real non-CUDA path: its trainer's `--impl` flag defaults to `"jax"` (MJX-JAX), which is a pure-Python, platform-independent wheel that runs on any XLA backend including CPU arm64; the MJX-Warp backend is NVIDIA-only and is disqualified here. The catch is that this "works" path is CPU-only in practice, because JAX has no usable Metal GPU backend: Apple's jax-metal has not shipped since 2024-10-08 and currently crashes on modern JAX with a StableHLO bytecode error reported on an M3 Max specifically. Combined with MuJoCo's own documented figure that MJX-JAX is "10x slower than MuJoCo" for a single scene, the whole GPU-parallel premise of Playground/Brax collapses on this machine, and the shipped 1024-env PPO configs are not realistic. The genuine value here is not the training loop but the content: reward functions, domain-randomization ranges, observation layouts, and MJCF models are all liftable into a CPU MuJoCo pipeline, though Brax's PPO itself is not, since it requires jittable/vmappable environments that CPU MuJoCo cannot provide.

## Key facts

- mujoco_playground is on PyPI as `playground`, latest 0.2.0 released 2026-03-16 (prior: 0.1.0 on 2026-01-08, 0.0.5 on 2025-06-23). Apache-2.0, requires Python >=3.11.
- brax latest is 0.14.2, released 2026-03-15 (0.14.1 on 2026-02-12, 0.14.0 on 2025-12-16). Apache-2.0, requires Python >=3.11. Playground pins brax>=0.14.2.
- Brax README states verbatim: "Only `brax/training` is actively being maintained as of 0.13.0", and directs users to MuJoCo Playground instead of `brax/envs`, and to MJX or MuJoCo Warp instead of Brax physics.
- Playground's trainer `learning/train_jax_ppo.py` defines `_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")` — the default is `jax`, NOT warp. This is the explicit non-CUDA path.
- MJX has two backends. MuJoCo docs: MJX-JAX "runs on: Nvidia and AMD GPUs, Apple Silicon, and Google Cloud TPUs"; MJX-Warp is NVIDIA-GPU-only and "does not support automatic differentiation".
- mujoco-mjx 3.11.0 (2026-07-28) ships as `py3-none-any` (pure Python, platform-independent) and lists warp-lang only as an OPTIONAL `warp` extra. So MJX-JAX itself imports and runs fine on macOS arm64.
- MuJoCo docs state MJX-JAX on a single scene "can be 10x slower than MuJoCo", and that MJX is designed for "simulating thousands or tens of thousands of scenes in parallel".
- Playground locomotion registry ships 9 robots: apollo, barkour, berkeley_humanoid, g1, go1, h1, op3, spot, t1. Humanoid entries: G1JoystickFlatTerrain, G1JoystickRoughTerrain, BerkeleyHumanoidJoystickFlatTerrain/RoughTerrain, T1JoystickFlatTerrain/RoughTerrain, ApolloJoystickFlatTerrain, Op3Joystick, H1InplaceGaitTracking, H1JoystickGaitTracking.
- Brax training agents are apg, ars, bc, es, ppo, sac. There is NO AMP (adversarial motion prior) agent in Brax, and no mocap motion-imitation environment in Playground.
- Playground's H1 gait-tracking envs are ANALYTIC phase-based, not mocap imitation: foot-height reference comes from `gait.get_rz(phase, swing_height=...)` with swing_height sampled in [0.08, 0.4], rewarded via `feet_phase` = jp.exp(-error / 0.01).
- Playground's "rough terrain" is a STATIC PNG heightfield, not procedural: `<hfield name="hfield" file="assets/hfield.png" size="10 10 .05 1.0"/>`. There is no procedural terrain generator or difficulty curriculum utility.
- Brax PPO is device-agnostic (no GPU-specific code; uses jax.pmap when multiple devices exist, else jax.vmap+jax.jit) BUT requires env.reset/env.step to be jittable and vmappable — which CPU MuJoCo C bindings are not.
- jax-metal's last release is 0.1.1 on 2024-10-08. JAX issue #34109 (opened 2025-12-26, still open, no maintainer response) reports jax-metal 0.1.1 failing on an Apple M3 Max under jax 0.8.2 with "error: unknown attribute code: 22 ... in bytecode version 6 produced by: StableHLO_v1.13.0" for even `jax.numpy.arange(10)`.
- warp-lang 1.16.0 (2026-08-03) does ship a `macosx_11_0_arm64` wheel and installs on Apple Silicon, but runs CPU-only there — macOS has had no CUDA support since 10.14 Mojave.
- mujoco_playground issue #212 "Make warp a soft dependency" (opened 2025-09-16 by bayerj, assigned btaba): warp became a hard dependency, and the reporter said "I had been using it quite a bit for dev on my Mac, and that's not possible anymore."
- PR #275 "Mac OS Changes for CPU Device Support" (opened ~2026-02-10) is still an unmerged DRAFT, blocked on an unsigned Google CLA, with no maintainer review. There is no officially supported macOS CPU path being actively developed.
- MuJoCo's `rollout` module is the realistic CPU batching primitive: it runs rollouts "in parallel with an internally managed thread pool", supports a `Rollout(nthread=N)` class and `persistent_pool`, and releases the GIL.
- MuJoCo issue #2813 "An MJX-style JAX FFI for CPU-based MuJoCo" (opened 2025-08-23) proposes exactly the CPU-MuJoCo-inside-JAX bridge that would let Brax PPO drive CPU MuJoCo — but it is open, unimplemented, with no maintainer response.

## Reusable for this project

- mujoco_playground/_src/locomotion/g1/joystick.py — the single highest-value file. A complete, tuned humanoid locomotion reward suite with verbatim scales: tracking_lin_vel=1.0, tracking_ang_vel=0.75, ang_vel_xy=-0.15, orientation=-2.0, feet_air_time=2.0, feet_slip=-0.25, feet_phase=1.0, stand_still=-1.0, termination=-100.0, collision=-0.1, contact_force=-0.01, joint_deviation_knee=-0.1, joint_deviation_hip=-0.25, dof_pos_limits=-1.0, pose=-0.1. Port the math to numpy/torch; the shaping and relative weighting is the real IP.
- The `feet_phase` gait-phase reward and the `gait.get_rz(phase, swing_height)` foot-height profile utility (used by g1/joystick.py and h1/joystick_gait_tracking.py). This is a strong analytic prior for periodic, human-like stepping and is a much cheaper first milestone than AMP — worth implementing BEFORE the mocap/AMP work.
- mujoco_playground/_src/locomotion/g1/randomize.py — the domain randomization RANGES (floor/foot friction U(0.4,1.0); dof frictionloss xU(0.5,2.0); armature xU(1.0,1.05); all body masses xU(0.9,1.1); torso mass +U(-1,1) kg; qpos0 offset +U(-0.05,0.05)). Copy the ranges, rewrite the mechanism against mjModel.
- The observation layout from g1/joystick.py: 100-dim actor state (local lin vel, noisy gyro, noisy gravity vector, 3-dim command, joint pos minus default pose, joint vel, previous action, and [cos,sin] gait phase per foot) plus a +46-dim privileged critic state (unnoised sensors, global velocities, root height, actuator forces, foot contacts, foot velocities, air time). This asymmetric actor-critic split is directly reusable and is a well-tested design.
- Episode/curriculum mechanics from g1/joystick.py: command resampling every 500 steps with 10% zero-command probability, command ranges vx in [-1,1], vy in [-0.5,0.5], yaw in [-1,1]; random push perturbations every 5-10s with magnitude 0.1-2.0 m/s in a uniformly random direction; termination on gravity-z < 0, foot-foot and foot-shin self-contact, and NaN guards. Also ctrl_dt=0.02 / sim_dt=0.002 (a 10:1 decimation) and action_scale=0.5 — sane defaults worth starting from.
- The MJCF scene scaffolding: g1/xmls/scene_mjx_feetonly_rough_terrain.xml shows the exact hfield idiom (`<hfield ... size="10 10 .05 1.0"/>` + `<geom type="hfield">`). Reuse the pattern but generate the heightfield array procedurally at runtime via mjModel.hfield_data rather than shipping a PNG — that is the clean route to your terrain-curriculum phase.
- mujoco_playground/_src/locomotion/t1/ and berkeley_humanoid/ — second and third independent reward-tuning references for bipeds. Cross-checking three tuned reward sets tells you which terms are load-bearing versus incidental.
- brax/training/agents/ppo/{train.py,losses.py,networks.py} — worth reading as a reference implementation even though you cannot run it against CPU MuJoCo. Specifically the recent additions: value bootstrap on timeout, clipped value loss, adaptive learning rate, episode metric normalization (0.14.0), soft-sign policy clipping (0.14.1), and distributional-critic PPO (0.14.2). These are concrete, tested improvements to fold into your own PPO.
- The `mujoco.rollout` module (nthread + persistent_pool, releases the GIL) is the CPU-side replacement for Brax's vectorization — pair it with your own PPO over the 10 performance cores instead of trying to force Brax PPO onto CPU MuJoCo.

## Blockers

- MJX-Warp / mujoco_warp is NVIDIA-only and therefore DISQUALIFIED on this machine. Its README says it is "designed for NVIDIA hardware" and "requires an NVIDIA GPU for fast simulation but supports CPU for development and debugging" — the CPU mode is a debugging aid, not a training path.
- There is no working Metal GPU backend for JAX. jax-metal has not been released since 2024-10-08 and currently crashes on modern JAX (StableHLO bytecode mismatch) on an M3 Max specifically. This means Playground/Brax on this machine run on the JAX CPU backend, and the 30-core GPU and Metal 4 sit unused.
- MJX-JAX on CPU is documented as ~10x slower than the regular MuJoCo C engine for a single scene. Playground's whole design premise (thousands of parallel envs) inverts on CPU, so its shipped defaults (num_envs=1024, batch_size=256) are not usable as-is.
- Brax PPO cannot be pointed at a CPU MuJoCo environment. It wraps env.reset/env.step in jax.pmap/jax.vmap and jit, so the env must be JAX-native. CPU MuJoCo via the C bindings is not traceable; bridging it would require jax.pure_callback, which serializes and destroys the batching that PPO depends on. The official fix (MuJoCo issue #2813) is unimplemented.
- No AMP anywhere in this ecosystem. Brax ships apg/ars/bc/es/ppo/sac only, and Playground has zero mocap-driven imitation environments. The AMP discriminator, motion dataset loading, and reference-state initialization must all be written from scratch.
- No procedural terrain. Rough-terrain variants load a fixed `assets/hfield.png`. There is no terrain generator, no difficulty curriculum, and no multi-waypoint navigation logic anywhere in Playground — all three of these project phases are unsupported.
- No human-proportioned humanoid. Every Playground humanoid is a real robot platform (Unitree G1/H1, Booster T1, Berkeley Humanoid, Apptronik Apollo, ROBOTIS OP3) with robot mass distribution, joint limits, and actuator models. None is a realistic human-proportioned figure, so the MJCF models do not directly serve the stated goal.
- Playground's domain_randomize functions are @jax.vmap-decorated and operate on mjx.Model pytrees. The randomization RANGES are portable to CPU MuJoCo, but the CODE is not — it must be rewritten against mjModel structs.
- `train_jax_ppo.py` hardcodes `MUJOCO_GL=egl`, which fails on macOS (see Playground issue #37, a MUJOCO_GL egl RuntimeError on a Mac M4 Pro). You must override to `glfw` or `osmesa` for any rendering/video.
- JIT compilation on Playground tasks is documented as slow (roughly 1-3 minutes per task), which is a painful iteration tax on a CPU backend where you cannot amortize it over huge batches.
- Version friction: jax is at 0.11.0 while the third-party jax-mps Metal backend requires jax>=0.10,<0.11, so even that fallback lags the current release. jax-mps is also single-device with no collectives and no float64.

## Performance evidence

- MuJoCo official docs: MJX-JAX simulating a single scene "can be 10x slower than MuJoCo" (which is heavily CPU-optimized). This is the single most decision-relevant number for a CPU-only machine.
- MuJoCo discussion #1101 PPO training benchmark (MJX + Brax, 60M steps): TPU v5e-8 = 249s, NVIDIA A100 = 718s, versus an Isaac Gym baseline of 600s on A100. All accelerator-class hardware; no Apple Silicon entry exists.
- I could NOT verify any credible Apple Silicon throughput figure for MJX or MuJoCo Playground. A web search surfaced a claim of ~650K steps/s for a single humanoid on an M3 Max, but fetching the cited MuJoCo discussion #1101 showed no Apple Silicon numbers at all — the search engine appears to have conflated sources. Treat that number as unsubstantiated.
- jax-mps (third-party MLX-backed JAX plugin, v0.10.10, 2026-07-18) reports ~3.7x speedup over CPU for ResNet18 training on an M4 MacBook Air. This is a dense-NN workload, not physics; it is NOT evidence that MJX would run on it, since MJX leans on scatter/gather and control flow that a partial StableHLO-to-MLX mapping is unlikely to cover.
- MuJoCo Playground's own published configs target 1024 parallel envs (with 128 eval envs, batch_size 256, 8 minibatches, unroll length 10), which is calibrated for datacenter GPUs, not a CPU backend.
- JIT compile time on Playground tasks is documented as roughly 1-3 minutes, a fixed per-run overhead independent of hardware speed.

## Sources

- [PyPI JSON API - playground (MuJoCo Playground) 0.2.0 metadata and release dates](https://pypi.org/pypi/playground/json) — 2026-03-16
- [PyPI JSON API - brax 0.14.2 metadata and release dates](https://pypi.org/pypi/brax/json) — 2026-03-15
- [GitHub Releases - google-deepmind/mujoco_playground (v0.2.0, v0.1.0 warp passthrough)](https://api.github.com/repos/google-deepmind/mujoco_playground/releases) — 2026-03-16
- [GitHub Releases - google/brax (v0.14.2 distributional PPO; v0.12.4 removed brax/v1)](https://api.github.com/repos/google/brax/releases) — 2026-03-15
- [Brax README - "Only brax/training is actively being maintained as of 0.13.0"](https://raw.githubusercontent.com/google/brax/main/README.md) — 2026-08-12
- [MuJoCo XLA (MJX) docs - MJX-JAX vs MJX-Warp backends, Apple Silicon support, 10x-slower-on-single-scene, feature parity limits](https://mujoco.readthedocs.io/en/stable/mjx.html) — 2026-08-12
- [mujoco_warp README - "designed for NVIDIA hardware", requires NVIDIA GPU](https://raw.githubusercontent.com/google-deepmind/mujoco_warp/main/README.md) — 2026-08-12
- [Playground train_jax_ppo.py - --impl flag defaults to "jax", MUJOCO_GL=egl hardcoded, num_envs=1024](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/learning/train_jax_ppo.py) — 2026-08-12
- [Playground locomotion registry - full env list incl. G1/T1/BerkeleyHumanoid rough-terrain variants](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/__init__.py) — 2026-08-12
- [Playground G1 joystick env - reward terms, reward scales, observation layout, termination, push logic](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/g1/joystick.py) — 2026-08-12
- [Playground G1 domain randomization ranges (jax.vmap over mjx.Model)](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/g1/randomize.py) — 2026-08-12
- [Playground G1 rough terrain scene - static assets/hfield.png heightfield, not procedural](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_rough_terrain.xml) — 2026-08-12
- [Playground H1 gait tracking - analytic gait.get_rz phase reference, NOT mocap imitation](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/h1/joystick_gait_tracking.py) — 2026-08-12
- [Brax PPO train.py - pmap/vmap device handling, requires jittable/vmappable env](https://raw.githubusercontent.com/google/brax/main/brax/training/agents/ppo/train.py) — 2026-08-12
- [Brax training agents listing - apg/ars/bc/es/ppo/sac, no AMP](https://api.github.com/repos/google/brax/contents/brax/training/agents) — 2026-08-12
- [JAX issue #34109 - jax-metal 0.1.1 fails on M3 Max, StableHLO bytecode error, open with no maintainer response](https://github.com/jax-ml/jax/issues/34109) — 2025-12-26
- [PyPI - jax-metal, last release 0.1.1 on 2024-10-08 (~22 months stale)](https://pypi.org/pypi/jax-metal/json) — 2024-10-08
- [PyPI - jax-mps 0.10.10, third-party MLX-based JAX backend for Apple Silicon, requires jax<0.11](https://pypi.org/pypi/jax-mps/json) — 2026-07-18
- [PyPI - mujoco-mjx 3.11.0, py3-none-any wheel, warp-lang only an optional extra](https://pypi.org/pypi/mujoco-mjx/json) — 2026-07-28
- [PyPI RSS - warp-lang 1.16.0 released 2026-08-03, daily 1.17.0 dev builds](https://pypi.org/rss/project/warp-lang/releases.xml) — 2026-08-03
- [NVIDIA Warp - runs on Apple Silicon (ARMv8) macOS but CPU-only; GPU requires CUDA-capable NVIDIA GPU](https://github.com/NVIDIA/warp) — 2026-08-12
- [Playground issue #212 - "Make warp a soft dependency"; warp hard dep broke Mac development](https://github.com/google-deepmind/mujoco_playground/issues/212) — 2025-09-16
- [Playground PR #275 - "Mac OS Changes for CPU Device Support", still a draft blocked on CLA, unreviewed](https://github.com/google-deepmind/mujoco_playground/pull/275) — 2026-02-10
- [MuJoCo issue #2813 - proposal for an MJX-style JAX FFI for CPU MuJoCo; open, unimplemented, no maintainer reply](https://github.com/google-deepmind/mujoco/issues/2813) — 2025-08-23
- [MuJoCo issue #2894 - warp-lang>=1.10.0.dev20251007 breaks --impl warp with the JAX PPO trainer](https://github.com/google-deepmind/mujoco/issues/2894) — 2025-10-11
- [MuJoCo Python docs - rollout module, internally managed thread pool, nthread, persistent_pool, GIL released](https://mujoco.readthedocs.io/en/stable/python.html) — 2026-08-12
- [MuJoCo 3 discussion - MJX+Brax PPO benchmarks (TPU v5e-8 249s, A100 718s); no Apple Silicon numbers](https://github.com/google-deepmind/mujoco/discussions/1101) — 2026-08-12
- [MuJoCo Playground paper (arXiv 2502.08844)](https://arxiv.org/pdf/2502.08844) — 2025-02-13
