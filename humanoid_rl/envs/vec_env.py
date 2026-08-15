"""Threaded vectorised MuJoCo environment.

This is the performance-critical core, and its design comes directly from measurement
(see DESIGN.md section 2). Three findings shaped it:

1. **MuJoCo releases the GIL during `mj_step`**, so worker *threads* give real
   parallelism and, unlike processes, never have to serialise observations.
2. **One `mj_step(nstep=decimation)` call beats a Python loop by 2.6x.** Calling `mj_step`
   per physics step forces a GIL release and reacquire every time, over 500,000 lock
   handoffs per second, and the transitions cost more than the physics.
3. **Per-environment Python is the real bottleneck.** With ten threads the GIL serialises
   all bytecode, so what matters is total Python work per batch, not per thread. Measured
   marginal cost per environment step: copying raw state out is 0.09 us (free), but four
   `mju_rotVecQuat` pybind11 calls cost 23.78 us and scalar numpy reward math 13.37 us.

Hence the split enforced here:

    worker threads   C only. Apply control, `mj_step(nstep=...)`, copy qpos/qvel out.
    main thread      Everything else, once, vectorised over all environments.

Physics alone measured 100,908 env steps/s at 1024 environments on an M3 Max. Keeping the
workers this thin is what preserves it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from humanoid_rl import hardware
from humanoid_rl.envs.domain_rand import (
    DomainRandConfig,
    ObservationNoise,
    build_model_pool,
)
from humanoid_rl.envs.model_prep import prepare
from humanoid_rl.qos import QoSClass, set_thread_qos
from humanoid_rl.tasks.base import BatchState, Task, quat_rotate_inverse, quat_to_heading

# Gravity direction in the world frame. Rotated into the body frame it gives the policy a
# heading-invariant sense of which way is down, the role an inner ear plays for a human.
_WORLD_DOWN = np.array([0.0, 0.0, -1.0])

_PHASE_STEP = 0
_PHASE_RESET = 1


@dataclass
class StepResult:
    """Output of one vectorised step. Every array is indexed by environment."""

    obs: np.ndarray  # (N, obs_dim) float32, first obs of the new episode where done
    reward: np.ndarray  # (N,) float32
    terminated: np.ndarray  # (N,) bool, ended through behaviour (fell over)
    truncated: np.ndarray  # (N,) bool, ended by the time limit
    reward_terms: np.ndarray  # (N, n_terms) float32, logging only
    final_obs: np.ndarray  # (N, obs_dim) float32, last obs of the ended episode
    episode_return: np.ndarray  # (N,) float32, valid where done
    episode_length: np.ndarray  # (N,) int32, valid where done
    success: np.ndarray  # (N,) bool, valid where done

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated


class ThreadedVecEnv:
    """N humanoid environments stepped in parallel by a pool of worker threads.

    Uses the standard autoreset convention. When an episode ends, the environment is reset
    immediately: `obs` holds the new episode's first observation while `final_obs` holds
    the last observation of the episode that ended. PPO needs both, and needs `terminated`
    and `truncated` distinguished so it bootstraps the value function only on time limits.
    """

    def __init__(
        self,
        model_path: str | Path,
        task: Task,
        num_envs: int,
        *,
        num_workers: int | None = None,
        decimation: int = 4,
        max_episode_steps: int = 1000,
        action_filter_hz: float = 8.0,
        seed: int = 0,
        domain_rand: DomainRandConfig | None = None,
    ) -> None:
        # Loads the MJCF, converts its torque motors into PD position servos, derives the
        # standing pose, and adds foot touch sensors. See envs/model_prep.py.
        self.prepared = prepare(model_path)
        self.model = self.prepared.model
        self.task = task
        self.num_envs = int(num_envs)
        self.decimation = int(decimation)
        self.max_episode_steps = int(max_episode_steps)
        self.rng = np.random.default_rng(seed)

        hw = hardware.detect()
        self.num_workers = max(1, min(int(num_workers or hw.recommended_physics_workers), self.num_envs))

        m = self.model
        self.nq, self.nv, self.nu = m.nq, m.nv, m.nu
        self.n_joint_pos = m.nq - 7
        self.n_joint_vel = m.nv - 6

        self.n_feet = int(self.prepared.foot_touch_adr.size)
        self.dt = self.decimation * float(m.opt.timestep)

        # Proprioception: joints, joint velocities, body-frame gravity, body-frame linear
        # and angular velocity, previous action, foot contact. Identical for every task.
        # Foot contact is included deliberately: it is physically realisable (real humanoids
        # have foot pressure sensors) and it makes learning a stepping gait much easier,
        # because the policy can tell stance from swing without inferring it.
        self.proprio_dim = (
            self.n_joint_pos + self.n_joint_vel + 3 + 3 + 3 + self.nu + self.n_feet
        )
        self.obs_dim = self.proprio_dim + task.task_obs_dim
        self.n_reward_terms = max(1, len(task.reward_term_names))

        # The actuators are PD position servos, so `ctrl` is a target joint angle in
        # radians. A policy action in [-1, 1] is an offset from the nominal standing pose,
        # scaled per joint, then clamped to that joint's anatomical limit.
        self._default_joint_pos = self.prepared.default_joint_pos.copy()
        # One-pole action filter state, see step(). beta = 1 - exp(-2*pi*fc*dt); a cutoff
        # of 0 disables. Reset rows to the nominal pose on episode reset, because carrying
        # the previous episode's filter state across a reset would let the first actions of
        # a new episode be shaped by the last actions of the old one.
        fc = float(action_filter_hz)
        self._filter_beta = 1.0 if fc <= 0.0 else 1.0 - float(np.exp(-2.0 * np.pi * fc * self.dt))
        self._ctrl_filtered = np.tile(self._default_joint_pos, (self.num_envs, 1))
        self._action_scale = self.prepared.action_scale.copy()
        self._ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
        self._touch_adr = self.prepared.foot_touch_adr
        # Column indices of every foot velocity component, flattened, so a single fancy
        # index pulls all of them out of sensordata at once.
        self._linvel_cols = np.concatenate(
            [np.arange(a, a + 3) for a in self.prepared.foot_linvel_adr]
        ).astype(np.int32)
        self._key_cols = (
            np.concatenate([np.arange(a, a + 3) for a in self.prepared.key_body_adr]).astype(
                np.int32
            )
            if self.prepared.key_body_adr.size
            else np.zeros(0, dtype=np.int32)
        )
        self.n_key_bodies = len(self.prepared.key_body_names)
        self._torso_adr = self.prepared.torso_zaxis_adr
        self._head_adr = self.prepared.head_pos_adr
        self._standing_head = max(1e-6, self.prepared.standing_head_height)
        #: Newtons under a foot before it counts as bearing load. About 2% of body weight,
        #: high enough to ignore grazing contacts and numerical noise.
        self.contact_force_threshold = 0.02 * float(m.body_mass.sum()) * 9.81

        # Domain randomisation. A pool of randomised models rather than one per
        # environment: an MjModel copy is 5.2 MB, so 4096 of them would cost 21 GB and
        # wreck the cache locality the physics loop depends on. MjData is compatible
        # across copies, so each environment is assigned a pool entry at reset.
        self.dr_cfg = domain_rand or DomainRandConfig()
        self.model_pool = build_model_pool(
            m, self.dr_cfg, self.rng, self.prepared.floor_geom_id
        )
        self._env_model = np.zeros(self.num_envs, dtype=np.int32)
        self._obs_noise = ObservationNoise(self.dr_cfg, self.n_joint_pos, self.n_joint_vel)

        self.datas = [mujoco.MjData(m) for _ in range(self.num_envs)]

        n = self.num_envs
        self.state = BatchState(
            qpos=np.zeros((n, self.nq)),
            qvel=np.zeros((n, self.nv)),
            ctrl=np.zeros((n, self.nu)),
            action=np.zeros((n, self.nu)),
            prev_action=np.zeros((n, self.nu)),
            episode_step=np.zeros(n, dtype=np.int32),
            root_pos=np.zeros((n, 3)),
            root_height=np.zeros(n),
            gravity_body=np.zeros((n, 3)),
            lin_vel_body=np.zeros((n, 3)),
            ang_vel_body=np.zeros((n, 3)),
            heading=np.zeros(n),
            foot_force=np.zeros((n, self.n_feet)),
            torque=np.zeros((n, self.nu)),
            prev_joint_vel=np.zeros((n, self.nu)),
            foot_contact=np.zeros((n, self.n_feet), dtype=bool),
            foot_air_time=np.zeros((n, self.n_feet)),
            foot_first_contact=np.zeros((n, self.n_feet), dtype=bool),
            foot_lin_vel=np.zeros((n, self.n_feet, 3)),
            torso_upright=np.ones(n),
            torso_zaxis=np.tile(np.array([0.0, 0.0, 1.0]), (n, 1)),
            head_height_ratio=np.ones(n),
            key_body_pos=np.zeros((n, max(1, len(self.prepared.key_body_names)), 3)),
            dt=self.dt,
        )
        # Let the task derive height-dependent thresholds from the actual model rather
        # than carrying numbers that silently go stale when the humanoid is swapped.
        if hasattr(task, "configure_for_model"):
            task.configure_for_model(self.prepared.standing_height)
        # Wider hook for tasks that need more than the standing height (torque ceilings, the
        # actuator->qpos map, key body layout). Kept separate so the existing narrow
        # signature, which several tasks implement, does not change.
        if hasattr(task, "configure_for_prepared"):
            task.configure_for_prepared(self.prepared)
        if hasattr(task, "set_joint_limits"):
            task.set_joint_limits(self._ctrl_lo.copy(), self._ctrl_hi.copy())
        # The limits above are per-ACTUATOR; the task reads joint angles out of qpos, and on
        # this model the two orders differ (hip_y/hip_z transposed on both legs). Hand over
        # the map so the pairing is correct rather than assumed.
        if hasattr(task, "set_joint_qpos_adr"):
            task.set_joint_qpos_adr(self.prepared.actuator_qpos_adr)
        # Published so a task can build an absolute reset pose that mixes reference frames
        # with the nominal stance, without needing to know how the engine derived it.
        self.state.task_state["_nominal_qpos"] = self.prepared.default_qpos.copy()
        task.init_state(self.state, self.rng)
        self._sensordata = np.zeros((n, max(1, m.nsensordata)))

        self._obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        self._final_obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        self._task_obs = np.zeros((n, max(1, task.task_obs_dim)))
        self._reward = np.zeros(n, dtype=np.float32)
        self._terms = np.zeros((n, self.n_reward_terms))
        self._terminated = np.zeros(n, dtype=bool)
        self._truncated = np.zeros(n, dtype=bool)
        self._success = np.zeros(n, dtype=bool)
        self._ep_return = np.zeros(n, dtype=np.float32)
        self._ep_return_out = np.zeros(n, dtype=np.float32)
        self._ep_length_out = np.zeros(n, dtype=np.int32)

        # Reset plumbing, filled by the main thread and consumed by workers in phase 2.
        self._reset_lists: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * self.num_workers
        self._reset_row = np.full(n, -1, dtype=np.int64)
        self._reset_qpos_noise = np.zeros((0, self.nq))
        self._reset_qvel_noise = np.zeros((0, self.nv))
        self._has_reset_noise = False
        # Absolute reset state, used instead of the nominal pose when a task supplies one.
        # Motion tracking must begin *in* a reference frame, which cannot sensibly be
        # expressed as a perturbation of the default stance.
        self._reset_qpos_abs = np.zeros((0, self.nq))
        self._reset_qvel_abs = np.zeros((0, self.nv))
        self._has_reset_pose = False

        # Random pushes. Each environment carries a countdown; when it expires the root
        # gets a horizontal velocity impulse. Applied inside the same worker phase as
        # resets, so shoving costs no extra barrier round.
        self._push_lists: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * self.num_workers
        self._push_vel = np.zeros((self.num_envs, 3))
        push_steps = max(1, int(self.dr_cfg.push_interval_s / self.dt))
        self._push_period = push_steps
        self._push_countdown = self.rng.integers(1, push_steps + 1, size=self.num_envs)

        bounds = np.linspace(0, n, self.num_workers + 1).astype(int)
        self._slices = [(int(bounds[i]), int(bounds[i + 1])) for i in range(self.num_workers)]
        self._start = threading.Barrier(self.num_workers + 1)
        self._end = threading.Barrier(self.num_workers + 1)
        self._phase = _PHASE_STEP
        self._shutdown = False
        self._errors: list[BaseException] = []
        self._closed = False

        self._threads = [
            threading.Thread(
                target=self._worker_loop, args=(i, lo, hi), daemon=True, name=f"phys-{i}"
            )
            for i, (lo, hi) in enumerate(self._slices)
        ]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------ workers

    def _worker_loop(self, widx: int, lo: int, hi: int) -> None:
        # Ask macOS to keep this thread on a performance core. There is no affinity API on
        # this platform; QoS is the only lever. A synchronised batch runs at the speed of
        # its slowest worker, so one worker on an efficiency core would halve throughput.
        set_thread_qos(QoSClass.USER_INITIATED)
        try:
            while True:
                self._start.wait()
                if self._shutdown:
                    return
                if self._phase == _PHASE_STEP:
                    self._step_slice(lo, hi)
                else:
                    self._reset_slice(widx)
                self._end.wait()
        except threading.BrokenBarrierError:
            return
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            self._errors.append(exc)
            self._start.abort()
            self._end.abort()

    def _step_slice(self, lo: int, hi: int) -> None:
        """The hot loop. Deliberately contains no numpy math and no pybind helper calls.

        Every statement here is paid `num_envs` times per step and serialised by the GIL,
        so anything added is expensive. Control scaling is done vectorised beforehand and
        the results are read straight out of `ctrl_batch`.
        """
        dec = self.decimation
        datas, ctrl, qpos, qvel = self.datas, self.state.ctrl, self.state.qpos, self.state.qvel
        sens, pool, env_model = self._sensordata, self.model_pool, self._env_model
        torq = self.state.torque
        for i in range(lo, hi):
            d = datas[i]
            d.ctrl[:] = ctrl[i]
            # One list index for the environment's randomised model. Measured as noise
            # against the 75 us of physics, unlike the numpy calls this loop avoids.
            mujoco.mj_step(pool[env_model[i]], d, nstep=dec)
            qpos[i] = d.qpos
            qvel[i] = d.qvel
            sens[i] = d.sensordata
            torq[i] = d.actuator_force

    def _reset_slice(self, widx: int) -> None:
        """Reset only the finished environments belonging to this worker."""
        model = self.model
        datas, qpos, qvel = self.datas, self.state.qpos, self.state.qvel
        sens = self._sensordata
        default_qpos = self.prepared.default_qpos
        has_noise = self._has_reset_noise

        # Shoves, applied to environments mid-episode. Done here rather than in the hot
        # step loop because only about one percent of environments are pushed on a given
        # step, so a scan of the whole batch would cost far more than the work itself.
        push_vel = self._push_vel
        for i in self._push_lists[widx]:
            d = datas[i]
            d.qvel[0:3] += push_vel[i]

        for i in self._reset_lists[widx]:
            d = datas[i]
            mujoco.mj_resetData(model, d)
            # The model's own qpos0 is a T-pose at z=0, i.e. inside the floor. Episodes
            # start from the derived standing pose instead.
            row = self._reset_row[i]
            if self._has_reset_pose:
                d.qpos[:] = self._reset_qpos_abs[row]
                d.qvel[:] = self._reset_qvel_abs[row]
            else:
                d.qpos[:] = default_qpos
                if has_noise:
                    d.qpos[:] += self._reset_qpos_noise[row]
                    d.qvel[:] += self._reset_qvel_noise[row]
            mujoco.mj_forward(model, d)
            qpos[i] = d.qpos
            qvel[i] = d.qvel
            sens[i] = d.sensordata

    def _run_phase(self, phase: int) -> None:
        if self._closed:
            raise RuntimeError("step() called on a closed ThreadedVecEnv")
        self._phase = phase
        try:
            self._start.wait()
            self._end.wait()
        except threading.BrokenBarrierError:
            if self._errors:
                raise self._errors[0] from None
            raise
        if self._errors:
            raise self._errors[0]

    # ------------------------------------------------- vectorised main-thread work

    def _compute_derived(self, idx: np.ndarray | slice = slice(None)) -> None:
        """Fill the derived fields of BatchState from raw qpos/qvel, vectorised."""
        s = self.state
        quat = s.qpos[idx, 3:7]
        s.root_pos[idx] = s.qpos[idx, 0:3]
        s.root_height[idx] = s.qpos[idx, 2]
        down = np.broadcast_to(_WORLD_DOWN, (quat.shape[0], 3))
        s.gravity_body[idx] = quat_rotate_inverse(quat, down)
        s.lin_vel_body[idx] = quat_rotate_inverse(quat, s.qvel[idx, 0:3])
        s.ang_vel_body[idx] = quat_rotate_inverse(quat, s.qvel[idx, 3:6])
        s.heading[idx] = quat_to_heading(quat)
        sens = self._sensordata[idx]
        s.foot_force[idx] = sens[:, self._touch_adr]
        s.foot_contact[idx] = s.foot_force[idx] > self.contact_force_threshold
        s.foot_lin_vel[idx] = sens[:, self._linvel_cols].reshape(-1, self.n_feet, 3)
        if self._torso_adr >= 0:
            s.torso_upright[idx] = sens[:, self._torso_adr + 2]
            s.torso_zaxis[idx] = sens[:, self._torso_adr:self._torso_adr + 3]
        if self._head_adr >= 0:
            s.head_height_ratio[idx] = sens[:, self._head_adr + 2] / self._standing_head
        if self._key_cols.size:
            s.key_body_pos[idx] = sens[:, self._key_cols].reshape(-1, self.n_key_bodies, 3)

    def _update_foot_air_time(self) -> None:
        """Track how long each foot has been airborne, and flag touchdowns.

        Runs on the whole batch once per step, after contact has been computed. The
        touchdown flag is what makes an air-time reward pay out exactly once per footfall
        rather than continuously, which is the difference between rewarding a real step and
        rewarding standing on one leg.
        """
        s = self.state
        np.logical_and(s.foot_contact, s.foot_air_time > 0.0, out=s.foot_first_contact)
        s.foot_air_time += self.dt
        # Reset the airborne clock for feet currently on the ground. Done after the
        # touchdown flag so the reward can still read the air time that was just completed.
        s.foot_air_time[s.foot_contact] = 0.0

    def _compute_obs(self, idx: np.ndarray | slice = slice(None)) -> None:
        """Assemble the observation matrix, vectorised."""
        s = self.state
        n1, n2 = self.n_joint_pos, self.n_joint_vel
        parts = [
            s.qpos[idx, 7:],
            s.qvel[idx, 6:],
            s.gravity_body[idx],
            s.lin_vel_body[idx],
            s.ang_vel_body[idx],
            s.prev_action[idx],
            s.foot_contact[idx].astype(np.float32),
        ]
        proprio = np.concatenate(parts, axis=1, dtype=np.float32)
        assert proprio.shape[1] == self.proprio_dim, (proprio.shape, self.proprio_dim)
        # Sensor noise is added to the observation only, never to the physics state, so
        # rewards and terminations still see ground truth. A real robot has noisy sensors,
        # but the world itself is not noisy.
        noise = self._obs_noise.sample(proprio.shape[0], self.rng)
        if noise is not None:
            proprio[:, : self._obs_noise.width] += noise
        self._obs[idx, : self.proprio_dim] = proprio
        if self.task.task_obs_dim:
            self.task.observe_batch(self.state, self._task_obs)
            self._obs[idx, self.proprio_dim :] = self._task_obs[idx].astype(np.float32)
        _ = n1, n2

    def _select_pushes(self) -> np.ndarray:
        """Tick every environment's push countdown and return those due for a shove."""
        cfg = self.dr_cfg
        if not cfg.enabled or cfg.push_interval_s <= 0:
            return np.empty(0, dtype=np.int64)

        self._push_countdown -= 1
        idx = np.flatnonzero(self._push_countdown <= 0)
        if idx.size:
            angle = self.rng.uniform(0.0, 2.0 * np.pi, size=idx.size)
            magnitude = self.rng.uniform(0.0, cfg.push_vel_xy, size=idx.size)
            self._push_vel[idx, 0] = magnitude * np.cos(angle)
            self._push_vel[idx, 1] = magnitude * np.sin(angle)
            self._push_vel[idx, 2] = 0.0
            # Randomise the next interval so pushes never fall into lockstep across
            # environments, which would put a periodic spike in the reward signal.
            self._push_countdown[idx] = self.rng.integers(
                self._push_period // 2, self._push_period * 2 + 1, size=idx.size
            )
        return idx

    def _do_resets(self, done_idx: np.ndarray, push_idx: np.ndarray | None = None) -> None:
        if done_idx.size:
            self._ctrl_filtered[done_idx] = self._default_joint_pos
        """Reset finished environments and apply pushes, in one worker phase.

        Task state is handled on the main thread, physics mutation in the workers.
        """
        push_idx = np.empty(0, dtype=np.int64) if push_idx is None else push_idx
        if done_idx.size == 0 and push_idx.size == 0:
            return
        s = self.state

        # Per-worker index lists, so a worker never scans environments it does not own.
        for w, (lo, hi) in enumerate(self._slices):
            left, right = np.searchsorted(done_idx, [lo, hi])
            self._reset_lists[w] = done_idx[left:right]
            pleft, pright = np.searchsorted(push_idx, [lo, hi])
            self._push_lists[w] = push_idx[pleft:pright]

        if done_idx.size == 0:
            # Pushes only: no task reset work, and the pushed states are picked up by the
            # next step's gather rather than needing a recompute here.
            self._run_phase(_PHASE_RESET)
            return

        # A resetting environment is reassigned to a different randomised model, so over a
        # run every environment experiences the whole pool rather than one fixed variant.
        if len(self.model_pool) > 1:
            self._env_model[done_idx] = self.rng.integers(
                0, len(self.model_pool), size=done_idx.size
            )

        # Task state is randomised FIRST, so that `reset_pose` and `reset_noise` can build a
        # physics state consistent with it. Doing this after them is a subtle and nasty bug:
        # a motion-tracking task would set its physics pose from the *previous* episode's
        # reference frame, so every episode would begin already out of sync with the clip it
        # is being scored against, and the failure looks like the tracking reward simply not
        # working.
        s.episode_step[done_idx] = 0
        s.prev_action[done_idx] = 0.0
        s.action[done_idx] = 0.0
        s.ctrl[done_idx] = self._default_joint_pos
        s.foot_air_time[done_idx] = 0.0
        s.foot_first_contact[done_idx] = False
        self.task.reset_batch(s, done_idx, self.rng)

        # An absolute pose takes precedence over additive noise.
        self._reset_row[done_idx] = np.arange(done_idx.size)
        pose = self.task.reset_pose(s, done_idx, self.rng)
        if pose is not None:
            self._reset_qpos_abs, self._reset_qvel_abs = pose
            self._has_reset_pose = True
            self._has_reset_noise = False
        else:
            self._has_reset_pose = False
            noise = self.task.reset_noise(s, done_idx, self.rng)
            if noise is None:
                self._has_reset_noise = False
            else:
                self._reset_qpos_noise, self._reset_qvel_noise = noise
                self._has_reset_noise = True

        # Seed the servo targets from the pose the humanoid is actually being reset INTO,
        # rather than from the standing pose.
        #
        # Both `s.ctrl` and `_ctrl_filtered` were set to `_default_joint_pos` above, which is
        # correct while every reset is a stand: the target equals the pose and the first step
        # demands nothing. It becomes badly wrong the moment a task resets into a pose on the
        # floor, because step 1 then commands a STANDING configuration to a body lying down.
        # Measured over 24 settled fallen poses, the first control step demands a mean
        # |torque| of 92.8 N.m peaking at 1002 N.m, with 6.5 of 28 joints pinned at their
        # ceiling. Seeding from the reset pose's own angles gives 1.47 N.m mean and 11.1 N.m
        # peak: a 63x reduction. Without this the opening 100 ms of every fallen episode is a
        # full-torque convulsion the policy never chose, and any analysis of how a get-up
        # begins would be studying the engine rather than the policy.
        #
        # Indexed through actuator_qpos_adr, never qpos[7:]: see E25, hip_y and hip_z are
        # transposed on both legs and a naive slice silently swaps four servo targets.
        if self._has_reset_pose and done_idx.size:
            rows = self._reset_row[done_idx]
            joints = self._reset_qpos_abs[rows][:, self.prepared.actuator_qpos_adr]
            s.ctrl[done_idx] = joints
            self._ctrl_filtered[done_idx] = joints

        self._run_phase(_PHASE_RESET)

        self._compute_derived(done_idx)
        self._compute_obs(done_idx)

    # ------------------------------------------------------------------ public API

    def reset(self) -> np.ndarray:
        """Reset every environment and return the initial observations."""
        all_idx = np.arange(self.num_envs)
        self._ep_return[:] = 0.0
        self.state.episode_step[:] = 0
        self.state.prev_action[:] = 0.0
        self.state.action[:] = 0.0
        self.state.ctrl[:] = 0.0
        self._ctrl_filtered[:] = self._default_joint_pos
        self._do_resets(all_idx)
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> StepResult:
        """Advance every environment by one control step.

        Args:
            actions: (num_envs, nu) in [-1, 1]. Values outside the range are clipped.
        """
        if actions.shape != (self.num_envs, self.nu):
            raise ValueError(
                f"actions must have shape {(self.num_envs, self.nu)}, got {actions.shape}"
            )
        s = self.state
        # Joint velocities from before this step, for the dof_acc penalty. qvel rows 6:
        # are the hinge joints, matching actuator order on this model.
        s.prev_joint_vel[:] = s.qvel[:, 6:6 + self.nu]
        s.prev_action[:] = s.action
        np.clip(actions, -1.0, 1.0, out=s.action)
        # Target joint angle = nominal standing pose + scaled action offset, clamped to the
        # joint limits. Vectorised once here rather than as numpy calls per environment
        # inside the workers, which measured 4.90 us per environment step.
        np.multiply(s.action, self._action_scale, out=s.ctrl)
        # A zero action means the task's baseline pose, which defaults to the nominal
        # stance but which a tracking task overrides to the current reference frame.
        baseline = self.task.action_offset(s)
        np.add(s.ctrl, self._default_joint_pos if baseline is None else baseline, out=s.ctrl)
        np.clip(s.ctrl, self._ctrl_lo, self._ctrl_hi, out=s.ctrl)

        # Low-pass the applied targets, one pole per joint, in the PLANT rather than the
        # policy. This exists to remove a capability: a policy was found to stabilise its
        # gait with its own exploration noise, and the dependence was specifically on
        # high-frequency tremor (falls 2% with 62 Hz noise, 32% at 4 Hz of the same power,
        # 100% deterministic). Gait content lives at 1-3 Hz and passes; tremor above the
        # cutoff loses its authority over the joints, so vibrational stabilisation stops
        # being possible for the noisy AND the deterministic policy alike, and evaluation
        # stops disagreeing with training. The filter is part of the environment, so every
        # consumer (training, evaluation, the interactive player, video rendering) sees the
        # same plant.
        if self._filter_beta < 1.0:
            self._ctrl_filtered += self._filter_beta * (s.ctrl - self._ctrl_filtered)
            s.ctrl[:] = self._ctrl_filtered

        self._run_phase(_PHASE_STEP)
        s.episode_step += 1

        self._compute_derived()
        self._update_foot_air_time()
        self._compute_obs()

        reward = self.task.reward_batch(s, self._terms)
        self._reward[:] = reward
        self._ep_return += self._reward

        # Per-step task bookkeeping that spans environments: curriculum levels, and the
        # motion-tracking playhead. Defined in the Task API from the start but not called
        # until now, which left the tracking reference frozen on its first frame.
        self.last_task_metrics = self.task.on_batch_end(s, {})

        np.copyto(self._terminated, self.task.terminated_batch(s))
        np.logical_and(
            ~self._terminated, s.episode_step >= self.max_episode_steps, out=self._truncated
        )
        done = self._terminated | self._truncated
        np.copyto(self._success, self.task.success_batch(s) & done)

        # Snapshot the ending episodes before autoreset overwrites their state.
        self._final_obs[done] = self._obs[done]
        self._ep_return_out[done] = self._ep_return[done]
        self._ep_length_out[done] = s.episode_step[done]

        done_idx = np.flatnonzero(done)
        self._do_resets(done_idx, self._select_pushes())
        self._ep_return[done_idx] = 0.0

        return StepResult(
            obs=self._obs.copy(),
            reward=self._reward.copy(),
            terminated=self._terminated.copy(),
            truncated=self._truncated.copy(),
            reward_terms=self._terms.astype(np.float32),
            final_obs=self._final_obs.copy(),
            episode_return=self._ep_return_out.copy(),
            episode_length=self._ep_length_out.copy(),
            success=self._success.copy(),
        )

    def close(self) -> None:
        """Shut the worker threads down cleanly. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._shutdown = True
        try:
            self._start.wait(timeout=5.0)
        except (threading.BrokenBarrierError, RuntimeError):
            pass
        for t in self._threads:
            t.join(timeout=5.0)

    def __enter__(self) -> "ThreadedVecEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise during interpreter shutdown
            pass
