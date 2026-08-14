# DESIGN.md

Human-like humanoid locomotion with reinforcement learning, on Apple Silicon.

> **Status.** Complete through Phase 3 Stage 2. Every number here was produced on the
> target machine by the scripts in `scripts/`, never quoted from a spec sheet or a paper.
> Where a published claim was tested and failed, section 9 records it.
>
> Sections 1-7 are the Phase 0 measurements. Section 8 is the stack decision, drawn from
> the research archived in [docs/research/](docs/research/). Sections 9-11 are what went
> wrong afterwards and what it cost to find out.

---

## 1. The machine

Detected by `humanoid_rl/hardware.py` on 2026-08-12.

| Property | Value |
| --- | --- |
| Chip | Apple M3 Max (`Mac15,11`, MacBook Pro) |
| CPU | 14 cores: **10 performance + 4 efficiency**, no SMT |
| GPU | **30 cores**, Metal 4 |
| Unified memory | **96 GB** |
| OS | macOS 26.5 (25F71), arm64 |
| Free disk | 873 GB |

The brief specified an M4 Max. This machine reports an M3 Max, so all tuning targets
10 performance cores and 30 GPU cores. Core counts are read at runtime rather than
hardcoded, so moving to a different Apple Silicon machine rescales the config
automatically (see `HardwareInfo.recommended_physics_workers`).

**Term:** *unified memory* means the CPU and GPU share one physical pool of RAM. There is
no PCIe bus between them, so moving data to the GPU is far cheaper than on an NVIDIA
system. This turns out to matter, and is measured in section 3.

---

## 2. Physics: how fast can this machine simulate a humanoid?

Model: the official MuJoCo `humanoid.xml` (nq=28, nv=27, **21 actuators**), 0.005 s
timestep. Control decimation 4, so the policy acts at **50 Hz** while physics runs at
200 Hz. This is the standard arrangement for locomotion control.

All throughput figures are quoted in **environment steps per second at the policy rate**,
because that is the unit RL sample budgets are quoted in. Multiply by 4 for raw physics
steps.

### 2.1 The measurement that changed the design

The first benchmark (`scripts/bench_physics.py`) showed thread scaling stalling badly:

| Workers | env steps/s | speedup |
| --- | --- | --- |
| 1 | 12,637 | 0.92x |
| 5 | 34,601 | 2.52x |
| 10 | 24,904 | **1.82x (worse than 5)** |

Meanwhile MuJoCo's own native C++ thread pool (`mujoco.rollout`) reached 83,499 env
steps/s at 10 workers, a 6.09x speedup. Same machine, same model, same core count. So
the physics engine was not the limit, the Python layer was.

The obvious suspect was the GIL.

**Term:** the *GIL* (Global Interpreter Lock) allows only one thread to run Python
bytecode at a time. A C extension can release it while doing pure C work, which is what
lets threads run truly in parallel.

A direct profile disproved the obvious explanation:

| Operation | Cost | GIL |
| --- | --- | --- |
| `mj_step` x4 (one 50 Hz step) | 75.31 us | **released** |
| `data.ctrl[:] = action` | 0.26 us | held |
| build 76-dim observation | 1.15 us | held |

Only **1.8%** of a step holds the GIL. By Amdahl's law that caps thread speedup near
50x, not 3x. So GIL *contention* was not the problem.

The real cause is GIL **handoff frequency**. Every `mj_step` call releases and then
reacquires the lock. Four calls per environment step, at roughly 13,000 env steps/s per
thread, is over 500,000 lock handoffs per second, and each handoff costs microseconds of
condition-variable signalling on macOS. The overhead is in the transitions, not the
holding.

### 2.2 The fix, and the architecture shoot-out

MuJoCo's Python binding exposes `mj_step(model, data, nstep=N)`, which performs N physics
steps inside a **single** call, so one GIL release covers the whole decimation window.

`scripts/bench_arch.py`, 10 workers, 150 control steps:

| Variant | GIL handoffs per env step | 256 envs | 1024 envs | 4096 envs |
| --- | --- | --- | --- | --- |
| A. `mj_step` x4 in a Python loop | 4 | 31,210 | 32,515 | 27,594 |
| **B. `mj_step(nstep=4)`** | **1** | **77,528** | 72,407 | **80,487** |
| C. batched native `rollout` | 1 per *batch* | 78,861 | 76,888 | 78,477 |

A one-line change (B over A) is worth **2.6x**.

B and C are within noise of each other, so the decision goes to flexibility:

- **C** (`mujoco.rollout`) executes a pre-computed open-loop control sequence inside one
  C++ call. It cannot run a Python policy in the loop, and per-environment resets,
  contact queries, terrain swaps and custom reward terms all become awkward. Every one
  of those is a hard requirement for this project.
- **B** keeps a normal `MjData` per environment in Python. Full access to contacts,
  sensors, per-environment reset, and heightfield mutation, at the same speed.

> **Decision: parallel environments are worker threads over per-environment `MjData`,
> advancing physics with a single `mj_step(model, data, nstep=decimation)` call.**
> Threads (not processes) because MuJoCo releases the GIL, so we get real parallelism
> with zero inter-process copying and zero serialisation of observations.

### 2.3 Keeping work off the efficiency cores

macOS deliberately has no `sched_setaffinity`. Thread placement is steered by **QoS
class** instead (`humanoid_rl/qos.py`).

This matters because a synchronised vectorised step finishes only when its *slowest*
worker finishes. An efficiency core is roughly a third the speed of a performance core,
so one worker parked on an E core drags the whole batch to its pace.

Physics workers request `USER_INITIATED`. `scripts/hwcheck.py` confirms the effect
empirically. macOS enumerates the 4 E cores first, and the per-core utilisation map
during parallel physics reads:

```
[+...##########]      # >70%   + >30%   . idle
 ^^^^ E cores          ^^^^^^^^^^ all 10 P cores at 98%
```

All ten performance cores saturated, the four efficiency cores left free for the OS, the
dashboard backend and ffmpeg.

### 2.4 The second bottleneck: per-environment Python is serialised

The first working environment implementation ran at **17,204 env steps/s**, 4.7x below the
benchmark that had justified the design. The benchmark was not wrong, it was simply
thinner than a real environment: it had no reward function, no termination check, no
observation assembly and no autoreset.

Isolating each addition (1024 environments, 10 workers) gave the marginal cost of each,
measured as extra microseconds per environment step:

| Addition to the worker loop | Throughput | Marginal cost |
| --- | --- | --- |
| control assign + `mj_step(nstep=4)` only | 100,908 /s | baseline |
| + copy `qpos`/`qvel` into batch arrays | 99,994 /s | **0.09 us (free)** |
| + numpy action clip and scale (3 ops) | 67,539 /s | 4.90 us |
| + numpy-scalar reward and `isfinite` | 42,956 /s | 13.37 us |
| + four `mju_rotVecQuat` pybind11 calls | 29,682 /s | **23.78 us (worst)** |

The governing rule is that **with ten threads the GIL serialises all Python bytecode**, so
what matters is total Python work per *batch*, not per thread. A `mju_rotVecQuat` call
costing well under a microsecond single-threaded costs about six microseconds under
ten-way contention. Multiplied by 1024 environments, per-environment Python became the
entire runtime and the physics it was meant to serve became a rounding error.

Copying raw state out being free is the key that unlocks the fix.

> **Decision: worker threads execute C only.** They apply control, call
> `mj_step(nstep=decimation)`, and copy `qpos`/`qvel` into batched arrays. Every piece of
> arithmetic (control scaling, body-frame rotations, observations, rewards, terminations)
> happens once on the main thread, vectorised over all environments.

Consequences, all of which turned out to be improvements rather than compromises:

- Tasks are written as numpy over `(num_envs, ...)` arrays instead of scalar loops, which
  is usually clearer as well as faster.
- MuJoCo's scalar math helpers are replaced with batched numpy equivalents in
  `humanoid_rl/tasks/base.py`. `quat_rotate_inverse` was verified against
  `mujoco.mju_rotVecQuat` to a maximum absolute error of **1.8e-15**, so no accuracy was
  traded for the speed.
- Autoreset runs as a second worker phase touching only the finished environments, which
  is typically a few percent of the batch.

Result: **78,736 env steps/s** end to end with the full reward, termination, observation
and autoreset machinery active. That is 4.6x the first implementation and 78% of the
bare-physics ceiling, with the remaining 22% being the vectorised main-thread work.

---

## 3. Networks: CPU, Metal GPU via PyTorch, or MLX?

Our networks are small multi-layer perceptrons (76 -> 512 -> 512 -> 21). The naive
assumption "GPU is faster" is not safe here: at small batch sizes the fixed cost of
launching a GPU kernel can exceed the arithmetic. `scripts/bench_networks.py` measures
the crossover instead of guessing.

Milliseconds per call, lower is better:

### Inference, tensors already on device

| batch | torch-cpu | torch-mps | mlx |
| --- | --- | --- | --- |
| 256 | 0.544 | **0.241** | 0.320 |
| 4096 | 3.007 | **0.557** | 0.755 |
| 16384 | 11.539 | **2.094** | 2.236 |

### Inference including the numpy handoff (the honest number)

The environments produce numpy arrays on the CPU and need actions back as numpy. This
row is what the training loop actually pays.

| batch | torch-cpu | torch-mps | mlx |
| --- | --- | --- | --- |
| 256 | 0.536 | 0.637 | **0.310** |
| 4096 | 3.035 | 1.086 | **0.788** |
| 16384 | 11.520 | 2.785 | **2.346** |

MLX is essentially unchanged by the transfer (0.320 -> 0.310 at batch 256) because
unified memory makes `mx.array(numpy_array)` near zero-copy. PyTorch pays a real copy on
`.to("mps")` and `.cpu()`, roughly 0.4 ms.

### Training step (PPO-shaped forward + backward + AdamW)

| batch | torch-cpu | torch-mps | mlx |
| --- | --- | --- | --- |
| 1024 | 6.631 | 1.637 | **1.355** |
| 4096 | 16.973 | **3.413** | 3.499 |
| 16384 | 53.758 | **12.495** | 14.338 |
| 65536 | 218.180 | 61.422 | **53.366** |

### What this means in context

PyTorch on the **CPU loses decisively** at every size (4x to 5x slower), and would also
compete with the physics threads for the same cores. It is out.

MLX wins the transfer-inclusive inference path, PyTorch MPS wins mid-size training, and
they trade places elsewhere. But the differences must be weighed against the physics
cost. At 4096 environments, one batched physics step takes 4096 / 80,487 = **50.9 ms**,
while an inference call costs about **1 ms**. Estimating a full PPO iteration
(24-step rollout, 5 epochs, 16384 minibatch):

| | PyTorch MPS | MLX |
| --- | --- | --- |
| rollout physics | 1222 ms | 1222 ms |
| rollout inference | 26 ms | 19 ms |
| PPO update (30 grad steps) | 375 ms | 429 ms |
| **total** | **1623 ms** | **1670 ms** |

The two are within 3% of each other end to end, which is inside run-to-run noise. So
raw speed does not decide this, and the tiebreaker is ecosystem: reference PPO and
adversarial-imitation implementations, distributions, checkpointing, and essentially
every open-source humanoid imitation codebase are PyTorch.

> **Decision: PyTorch with the MPS (Metal) backend for all networks.** MLX stays
> installed and benchmarked as a documented fallback, to be revisited only if MPS shows
> numerical trouble during adversarial training, where the discriminator is the most
> delicate component.

---

## 4. Do the CPU and GPU fight each other?

The design runs 10 CPU physics threads and Metal GPU network work in the same process.
Measured (6 s each, isolated then concurrent):

| | alone | concurrent | retained |
| --- | --- | --- | --- |
| physics | 81,789 env steps/s | 74,855 | **91.5%** |
| MPS training | 78.5 grad steps/s | 78.8 | **100.3%** |

They genuinely run in parallel, with the GPU essentially unaffected and the CPU giving
up 8.5% to Metal command dispatch.

This is a real architectural opportunity rather than just a null result: because the two
resources are independent, the PPO update can be **overlapped** with the next rollout
collection instead of alternating with it. On the numbers in section 3 that would hide
most of the 375 ms update behind the 1222 ms rollout, worth roughly 20% end-to-end. This
is deferred until after correctness is established, since asynchronous PPO changes the
on-policy data distribution slightly and must be introduced carefully.

---

## 5. Verified capability baseline

`scripts/hwcheck.py` proves each claim empirically instead of trusting an
`is_available()` flag, and exits non-zero if any required check fails.

| Check | Result |
| --- | --- |
| Metal GPU compute (PyTorch) | 8,562 GFLOP/s on mps vs 1,391 on cpu (**6.2x**, so genuinely on GPU) |
| MPS numerical agreement | max abs diff vs CPU = 0.00e+00 |
| Metal GPU compute (MLX) | 8,225 GFLOP/s, default device = gpu |
| Parallel physics cores busy | 10 cores above 70%, mean 98% |
| Offscreen headless rendering | (12, 240, 320, 3) uint8, non-blank |
| mp4 encoding | ffmpeg 9.0.1, `h264_videotoolbox` available |

The `h264_videotoolbox` encoder is significant for the Videos mode: video encoding runs
on the dedicated Apple media engine, so it costs neither GPU nor CPU time that training
needs.

---

## 6. Headline numbers for planning

| Quantity | Measured |
| --- | --- |
| Serial, one environment | 12,977 env steps/s |
| **Parallel, 10 P-cores, realistic loop** | **~80,000 env steps/s** |
| Parallel efficiency | 6.2x on 10 cores (62%) |
| Physics-only ceiling, no Python | 83,499 env steps/s |

Caveats that will lower this in later phases, and are budgeted for rather than ignored:

- A human-proportioned humanoid has far more degrees of freedom than the 27 of
  `humanoid.xml`, and physics cost grows with DoF and contact count.
- Heightfield terrain adds collision cost against a non-flat surface.
- The adversarial imitation discriminator adds a second network evaluation per step.

Planning therefore assumes an effective **20,000 to 40,000 env steps/s** in the fully
featured system, not 80,000. At 30,000 env steps/s a 500M-step run takes roughly
**4.6 hours**, and 1B steps roughly **9.3 hours**, which fits comfortably inside the
multi-day training budget.

---

## 7. Reproducing these numbers

```bash
.venv/bin/python scripts/hwcheck.py                              # verification, exits non-zero on failure
.venv/bin/python scripts/bench_physics.py                        # thread/process/rollout scaling
.venv/bin/python scripts/bench_arch.py                           # the architecture shoot-out
.venv/bin/python scripts/bench_networks.py                       # torch-cpu vs torch-mps vs mlx
```

Raw JSON output is written to `bench_results/`.

---

## 8. Stack selection from current documentation

A 12-agent documentation sweep ran on 2026-08-12/13. The **full raw output is archived in
the repository** at [docs/research/](docs/research/): 11 topic dossiers, a synthesis, an
adversarial critique, and a deep dive on motion imitation. This section records only the
decisions taken from it.

### 8.1 What is disqualified, and why that matters

| Candidate | Verdict | Evidence |
| --- | --- | --- |
| Isaac Gym / Isaac Lab | **Out.** CUDA and Linux only, and NVIDIA calls Isaac Gym "legacy software, no longer supported" | vendor docs |
| MJX via `jax-metal` | **Out.** `jax-metal` last released **2024-10-08**, abandoned | PyPI, jax discussion #34648 |
| MuJoCo Warp, mjlab, Newton | **Out.** NVIDIA-only, or macOS "evaluation only", or CPU-only | project READMEs |

This is the load-bearing negative result of Phase 0: **there is no working GPU physics path
on Apple Silicon today.** It turns the CPU-MuJoCo plus PyTorch-MPS design from a preference
into the only viable option, and retires MJX as a future upgrade path.

Worth stating plainly, because the table is easy to over-read: **none of the *algorithms*
need CUDA.** DeepMimic, AMP, ASE, PHC and ADD are PPO, MLPs and GANs in stock PyTorch. What
is CUDA-locked is Isaac Gym's GPU-resident physics API, and the tacit assumption that you
can afford 10^9 samples.

### 8.2 Decisions

| Question | Decision |
| --- | --- |
| Physics | CPU MuJoCo, worker threads, `mj_step(nstep=decimation)` (section 2) |
| Networks | PyTorch on MPS, MLX kept as a documented fallback (section 3) |
| Humanoid model | **MimicKit `humanoid.xml`** (Apache-2.0, 28 DoF, **box feet**) |
| Control | PD position servos derived from the model's own gains (section 9.4) |
| Imitation algorithm | **AMP**, with the paper's Table 4 hyperparameters, not Isaac's |
| Bring-up before AMP | **DeepMimic single-clip tracking first.** Non-negotiable |
| Mocap, day one | **100STYLE (CC BY 4.0, no registration)** and dm_control's CMU HDF5 |
| Mocap, upgrade | AMASS, once registration clears |
| Retargeting | GMR (MIT, CPU, accepts BVH with no SMPL registration) |
| Gait quality metric | LocoMuJoCo's DTW and discrete Frechet distance (MIT) |

**Why the MimicKit humanoid and not MuJoCo's own:** its feet are **boxes**, MuJoCo's default
humanoid uses **capsules**. A capsule foot is effectively a line contact and physically
cannot produce a heel-to-toe roll, so the default humanoid could never satisfy the
"heel-to-toe steps" requirement no matter how good the reward was.

**Why the AMP paper's hyperparameters:** the paper was trained on **16 CPU cores**, with
4096 samples per update and a 256 discriminator batch. Its Table 4 already *is* the
small-batch CPU configuration. The widely-copied `HumanoidAMPPPO.yaml` is a later
4096-GPU-environment retune that will not transfer to this machine. This is the single most
common mistake people make porting AMP.

### 8.3 Licensing, which is a real constraint here

The critique caught an error worth recording: **LocoMuJoCo's ready-made HuggingFace mocap
clips are CC BY-NC-ND**, the same no-derivatives clause that rules out LAFAN1, because they
are already retargeted derivatives of AMASS and LAFAN1. Retargeting them again produces a
derivative of a NoDerivatives work.

Genuinely clean day-one sources: **100STYLE (CC BY 4.0)** and dm_control's **CMU** HDF5.
MimicKit's own motion data sits behind a SharePoint link with unstated terms, so it stays
off the critical path. Apache-2.0 covers MimicKit's *code*, not its data.

---

## 9. Where measurement overruled the research

Every claim below was reported with confidence and then failed a direct test on this
machine. They are recorded because the corrected versions are load-bearing, and because
together they are a standing argument for verifying before building.

**9.1 The gradient penalty on MPS.** Reported as impossible (PyTorch #98498,
`aten::linear_backward` double-backward unimplemented, open since 2023), with the
recommended mitigation of keeping the AMP discriminator on the CPU. **Tested: it works.**
`torch.autograd.grad(..., create_graph=True)` through `nn.Linear` on torch 2.13.0 gives
0.0093 on MPS against 0.0094 on CPU. The discriminator stays on the GPU, and AMP's most
vital stabiliser is available to us.

**9.2 Small MLPs favour the CPU.** A widely repeated caution, and plausible in principle:
kernel launch overhead can exceed the arithmetic at small batch. **Measured on the actual
`[1024, 512]` discriminator, MPS wins 3x to 4x at every batch size tested**, including 256
(0.374 ms against 1.557 ms).

**9.3 The AMP humanoid may not load in MuJoCo.** Reported as authored for Isaac Gym's MJCF
subset. **MimicKit's loads unmodified** (nq=35, nv=34, nu=28, stable). ASE's genuinely does
fail, on a missing material `grid`, and is NVIDIA non-commercial anyway.

**9.4 The humanoid's "joint stiffness" is not passive material stiffness.** Not a research
error but a modelling trap that cost real debugging time. The MJCF specifies `stiffness` and
`damping` on every joint (abdomen 1000, hip 500, knee 500, shoulder 400, damping always
exactly kp/10). Those numbers are **DeepMimic-style PD gains**: Isaac Gym, which the model
targets, uses a DOF's stiffness and damping directly as its position-drive gains. MuJoCo
interprets the same fields as a **passive spring pulling the joint toward zero**, which here
is the T-pose.

Left in place they silently fight the position servo. The symptom was arms that would not
leave the T-pose while the actuator applied a constant 300 N*m: the servo sat in exact force
balance against the spring, holding the shoulder at 14 degrees instead of the commanded 83.
The fix is to move those numbers out of the springs and into the actuators, where they mean
what their author intended, and zero the springs. See `humanoid_rl/envs/model_prep.py`.

---

## 10. Risks carried into later phases

From the adversarial critique, which reviewed the synthesis for exactly this.

| Risk | Phase | Mitigation |
| --- | --- | --- |
| **Offscreen render buffer defaults to 640x480**, so videos would be permanently 480p | 3 | Already handled: `humanoid_scene.xml` sets `offwidth="1280" offheight="960"`. Every generated terrain scene must inherit it |
| **`mujoco.Renderer` has no `update_hfield`**, so regenerated terrain never reaches the GPU and videos silently show stale terrain | 4 | Drive the offscreen path manually (own `GLContext`, `MjrContext`, `mjr_uploadHField`) instead of using `Renderer` |
| **A CGL context is current on one thread at a time** | 4 | One dedicated render thread owns its context; physics workers post upload requests to it |
| **Root world-z leaks terrain into the AMP discriminator**, which can then separate policy from mocap with zero style content | 3, 4 | Replace world z with **height above the terrain under the root**. Do *not* simply drop it: root height is the most discriminative feature of a jump, so dropping it forecloses the jump and climb roadmap |
| **The discriminator observation width is baked into every checkpoint** | 3 | Freeze the discriminator observation spec before Phase 3 begins, not during Phase 4 |
| **`task_reward_w: 0.0` in the vendored AMP config** is the *no-task* setting | 5 | Phase 5 needs a non-zero task weight. Copying it forward yields a policy that ignores the goal, which reads as a waypoint bug |
| Thermal throttling over multi-day runs | all | `pmset -g therm` sampled every 100 iterations into `events.jsonl`; the throughput chart is the visible alarm |

---

## 11. Phase 3: what reference quality actually costs

Stage 1 (motion tracking) was run as the bring-up check the research insisted on. It never
converged, and that was its value: it proved the reference motion was not directly
trackable and identified exactly why. PD gains, contacts and reference state initialisation
were all validated in the process; none of them was the problem.

Four defects were found, each masking the next. They are recorded because all four are
invisible in a kinematic playback, which looks like a person walking in every case.

### 11.1 Ground alignment used a bounding sphere

`_ground_offset` computed each geom's lowest point as `geom_xpos.z - geom_rbound`.
`geom_rbound` is the **bounding sphere** radius. For the foot box that is
`sqrt(0.0885^2 + 0.045^2 + 0.0275^2) = 10.4 cm` against a true flat half-extent of
**2.75 cm**.

Every clip was therefore lifted about 7.6 cm, and the feet never touched the ground at any
point in any clip. Measured: minimum foot clearance 5.4 cm, and **0% of frames** within
3 cm of the floor. Physics simulation began with the humanoid airborne and dropping, which
accounted for roughly half the observed 9.7 cm sink.

The exact per-geom-type computation already existed (`_geom_lowest_z` in `model_prep.py`)
and simply was not used here.

### 11.2 The reference skated

Mocap retargeted by per-frame IK slides its planted feet. Measured on a 100STYLE walk:
planted feet moved at **0.134 m/s while the body travelled at 0.588 m/s**, so **23% of the
body's displacement came from the feet sliding along the ground**.

That is not physically reproducible: a planted foot grips. Simulating the reference joint
angles open-loop covered only **63%** of the reference distance. Raising the PD gains
eightfold changed that to 65% while octupling peak torque to 2198 N, which is how we
established the controller was never the constraint.

### 11.3 Stance detection, three attempts

| Approach | Result |
| --- | --- |
| Foot within 3 cm of its own lowest point over the clip | Mislabelled stance, 18% residual slip |
| Real `mj_forward` contacts | **Zero contacts found.** MuJoCo only generates a contact on actual penetration, and an exactly-aligned clip merely grazes the floor |
| Exact geom lowest point against an absolute 1.5 cm threshold | Works, once 11.1 made the threshold meaningful |

### 11.4 Pipeline ordering

De-skating detects stance from foot-ground proximity, so it must run **after** ground
alignment. It originally ran before, on a clip that was hovering 7.6 cm in the air, and
silently did nothing. Correct order: **align, de-skate, smooth, re-align**.

### 11.5 Result

| | before | after |
| --- | --- | --- |
| planted-foot slip | 0.134 m/s (23% of body travel) | **0.054 m/s (10%)** |
| open-loop distance ratio | 0.63 | **0.68 at 1x gain, 0.75 at 3x** |
| sink | -9.7 cm | **-4.3 cm at 1x, -2.6 cm at 3x** |
| frames with a foot within 2 cm of ground | 0% | 42% |
| marker residual (walk clips) | 0.9 cm | 0.9 cm, unaffected |

Note the last row: none of this cost pose accuracy. These were errors in *root trajectory
and ground registration*, which marker residual does not measure. A retarget can score
0.9 cm and still be physically fictional.

Also note that PD gain scale only began to matter **after** 11.1 was fixed. Before that the
humanoid was dropping from mid-air on every episode, and that dominated everything else.

### 11.6 Consequence for Stage 2

AMP is far more tolerant of the residual 10% slip than tracking was, because it never
requires the policy to be at a specific frame's exact position. It asks only whether the
motion looks human. The artifacts that made tracking impossible are a much smaller problem
for an adversarial prior, which is why Stage 2 proceeded on a reference that is good but
not perfect.
