"""Hardware probe for Apple Silicon.

Everything in the training stack that depends on machine size (number of parallel
physics workers, PPO batch size, evaluation worker count) reads from here rather
than hardcoding numbers. That keeps a single config file portable across an
M3 Max, an M4 Max or any other Apple Silicon machine.

Terminology used throughout this file:

* Performance core (P core): a fast, power-hungry CPU core. Apple Silicon chips have
  a small number of these and they do the real compute work.
* Efficiency core (E core): a slow, low-power CPU core. macOS parks background work
  here. Putting physics workers on E cores actively hurts, because a synchronised
  parallel step runs at the speed of its slowest worker.
* Unified memory: on Apple Silicon the CPU and GPU share one pool of physical RAM,
  so moving a tensor to the GPU does not cross a PCIe bus the way it does on NVIDIA.
"""

from __future__ import annotations

import functools
import platform
import re
import subprocess
from dataclasses import dataclass


def _sysctl(key: str) -> str | None:
    """Read one sysctl key, returning None if it does not exist on this machine."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return None
    value = out.stdout.strip()
    return value or None


def _sysctl_int(key: str, default: int = 0) -> int:
    raw = _sysctl(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _gpu_core_count() -> int:
    """Number of GPU cores.

    Uses ioreg (about 25 ms) rather than `system_profiler SPDisplaysDataType`
    (about 1.5 s), because this runs on every training start.
    """
    try:
        out = subprocess.run(
            ["ioreg", "-rc", "AGXAccelerator", "-d", "1"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return 0
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out.stdout)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class HardwareInfo:
    """Immutable snapshot of the machine's compute resources."""

    chip: str
    performance_cores: int
    efficiency_cores: int
    logical_cores: int
    gpu_cores: int
    memory_bytes: int
    os_name: str
    os_version: str
    arch: str

    @property
    def is_apple_silicon(self) -> bool:
        return self.arch == "arm64" and self.os_name == "Darwin"

    @property
    def memory_gb(self) -> float:
        return self.memory_bytes / 1024**3

    @property
    def recommended_physics_workers(self) -> int:
        """How many parallel physics worker threads to run.

        Set to the performance core count. The efficiency cores are deliberately left
        free for macOS itself, the dashboard backend and ffmpeg video encoding. A
        synchronised vectorised step finishes only when its slowest worker finishes,
        so scheduling even one worker onto an efficiency core drags the whole batch
        down to that core's speed.
        """
        if self.performance_cores > 0:
            return self.performance_cores
        # Non-Apple-Silicon fallback: leave two cores for the OS.
        return max(1, self.logical_cores - 2)

    def summary_lines(self) -> list[str]:
        return [
            f"chip                   {self.chip}",
            f"performance cores      {self.performance_cores}",
            f"efficiency cores       {self.efficiency_cores}",
            f"logical cores          {self.logical_cores}",
            f"gpu cores              {self.gpu_cores}",
            f"unified memory         {self.memory_gb:.0f} GB",
            f"os                     {self.os_name} {self.os_version} ({self.arch})",
            f"physics workers (rec)  {self.recommended_physics_workers}",
        ]


@functools.lru_cache(maxsize=1)
def detect() -> HardwareInfo:
    """Probe the machine once and cache the result for the process lifetime."""
    perf = _sysctl_int("hw.perflevel0.physicalcpu")
    eff = _sysctl_int("hw.perflevel1.physicalcpu")
    logical = _sysctl_int("hw.logicalcpu", default=1)

    # On Intel Macs the perflevel keys do not exist, so fall back to a flat core count.
    if perf == 0 and eff == 0:
        perf = _sysctl_int("hw.physicalcpu", default=logical)

    return HardwareInfo(
        chip=_sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        performance_cores=perf,
        efficiency_cores=eff,
        logical_cores=logical,
        gpu_cores=_gpu_core_count(),
        memory_bytes=_sysctl_int("hw.memsize"),
        os_name=platform.system(),
        os_version=platform.mac_ver()[0] or platform.release(),
        arch=platform.machine(),
    )


def thermal_state() -> dict[str, str]:
    """Current thermal and power pressure, for monitoring long training runs.

    A laptop under sustained multi-day load will throttle. Logging this alongside
    steps-per-second explains sudden throughput drops that would otherwise look
    like a bug in the training code.
    """
    state: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5, check=True
        )
        for line in out.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                state[key.strip().lstrip("- ").lower().replace(" ", "_")] = value.strip()
    except (subprocess.SubprocessError, OSError):
        pass

    # Whether the machine is on AC power. Training on battery throttles hard.
    try:
        out = subprocess.run(
            ["pmset", "-g", "ps"], capture_output=True, text=True, timeout=5, check=True
        )
        state["power_source"] = "ac" if "AC Power" in out.stdout else "battery"
    except (subprocess.SubprocessError, OSError):
        pass

    return state


if __name__ == "__main__":
    info = detect()
    for line in info.summary_lines():
        print(line)
    print()
    for key, value in thermal_state().items():
        print(f"{key:22s} {value}")
