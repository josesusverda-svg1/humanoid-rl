"""macOS Quality of Service (QoS) control for worker threads.

Why this module exists
----------------------
On Linux you pin a thread to a specific CPU core with `sched_setaffinity`. macOS
deliberately does not expose that. Instead the kernel decides placement from a
thread's *QoS class*, a hint describing how latency-sensitive the work is:

    USER_INTERACTIVE  highest, scheduled on performance cores
    USER_INITIATED    high, strongly prefers performance cores
    DEFAULT           unspecified, scheduler picks
    UTILITY           long-running work, may land on efficiency cores
    BACKGROUND        lowest, forced onto efficiency cores

This matters a lot for synchronised parallel physics. A vectorised environment step
finishes only when its *slowest* worker finishes. An efficiency core on M3 Max is
roughly a third the speed of a performance core, so a single worker scheduled onto
an E core can nearly triple the wall-clock time of the entire batch.

Setting USER_INITIATED on physics workers tells the scheduler to keep them on
performance cores, while leaving the efficiency cores free for macOS, the dashboard
backend and ffmpeg video encoding.

Reference: Apple, "Energy Efficiency Guide for Mac Apps: Prioritize Work at the
Task Level", and `man pthread_set_qos_class_self_np`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
from enum import IntEnum


class QoSClass(IntEnum):
    """Values of the `qos_class_t` enum from <sys/qos.h>."""

    USER_INTERACTIVE = 0x21
    USER_INITIATED = 0x19
    DEFAULT = 0x15
    UTILITY = 0x11
    BACKGROUND = 0x09
    UNSPECIFIED = 0x00


_libc: ctypes.CDLL | None = None


def _load_libc() -> ctypes.CDLL | None:
    """Load libSystem once, returning None on non-macOS platforms."""
    global _libc
    if _libc is not None:
        return _libc
    if platform.system() != "Darwin":
        return None
    path = ctypes.util.find_library("System")
    if path is None:
        return None
    try:
        _libc = ctypes.CDLL(path, use_errno=True)
    except OSError:
        return None
    return _libc


def set_thread_qos(qos: QoSClass = QoSClass.USER_INITIATED, relative_priority: int = 0) -> bool:
    """Set the QoS class of the *calling* thread.

    Must be called from inside the worker thread itself, not from the thread that
    created it, because the underlying pthread call only ever affects `self`.

    Args:
        qos: Desired QoS class. USER_INITIATED is the right choice for physics
            workers: high enough to stay on performance cores, without claiming the
            UI-latency priority that USER_INTERACTIVE implies.
        relative_priority: Offset within the class. Must be <= 0 per the API.

    Returns:
        True if the QoS class was applied, False on failure or on non-macOS systems.
        Callers should treat False as non-fatal: the code still runs correctly, just
        without placement hints.
    """
    libc = _load_libc()
    if libc is None:
        return False
    try:
        fn = libc.pthread_set_qos_class_self_np
    except AttributeError:
        return False
    fn.argtypes = [ctypes.c_uint, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn(ctypes.c_uint(int(qos)), ctypes.c_int(relative_priority)) == 0


def get_thread_qos() -> QoSClass | None:
    """Read back the calling thread's QoS class, or None if unavailable."""
    libc = _load_libc()
    if libc is None:
        return None
    try:
        fn = libc.pthread_get_qos_class_np
    except AttributeError:
        return None
    # int pthread_get_qos_class_np(pthread_t, qos_class_t *, int *)
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
    fn.restype = ctypes.c_int
    try:
        self_fn = libc.pthread_self
    except AttributeError:
        return None
    self_fn.restype = ctypes.c_void_p

    qos_out = ctypes.c_uint(0)
    prio_out = ctypes.c_int(0)
    if fn(self_fn(), ctypes.byref(qos_out), ctypes.byref(prio_out)) != 0:
        return None
    try:
        return QoSClass(qos_out.value)
    except ValueError:
        return None


def disable_app_nap() -> bool:
    """Best-effort attempt to stop macOS suspending a long-running background process.

    App Nap throttles timers and lowers priority for processes macOS considers idle,
    which can silently slow a multi-day training run. This is only reliably disabled
    via the Foundation framework's NSProcessInfo activity API, which requires pyobjc.
    Returns False if pyobjc is not installed, in which case the recommended fallback
    is to launch training under `caffeinate -dimsu`.
    """
    try:
        from Foundation import NSActivityUserInitiated, NSProcessInfo  # type: ignore
    except ImportError:
        return False
    # Hold the activity token on the module so it is never garbage collected, which
    # would immediately end the activity assertion.
    global _app_nap_token
    _app_nap_token = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        NSActivityUserInitiated, "humanoid-rl training"
    )
    return True


_app_nap_token = None
