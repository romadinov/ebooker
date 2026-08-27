"""Pick the compute device, so the same command works on any machine.

The point is that a run should not need to know what it is running on. CUDA is
preferred where present, then Apple's MPS, then CPU.

One caveat worth knowing rather than discovering: on MPS, Metal's MPSGraph
compilation cache is keyed on tensor shape, has no eviction path and lives for
the life of the process (pytorch#181213). Models whose input shapes vary per
call -- which includes every flow-matching TTS model here -- grow it without
bound until the process stops making progress. Only a restart clears it, so a
long MPS run should be driven a chapter at a time. `mps_needs_restarts()` says
when that applies.
"""

from __future__ import annotations

import os


def available() -> list[str]:
    """Devices this machine can actually use, best first."""
    out: list[str] = []
    try:
        import torch
    except ImportError:
        return ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out.append("mps")
    out.append("cpu")
    return out


def pick(preferred: str | None = None) -> str:
    """Resolve a device request. "auto" or None means detect.

    An explicit request is honoured even if unavailable, so a mistake surfaces
    as a clear failure from the backend rather than being silently downgraded.
    """
    if preferred and preferred != "auto":
        return preferred
    env = os.environ.get("EBOOKER_DEVICE", "").strip()
    if env and env != "auto":
        return env
    return available()[0]


def describe(device: str | None = None) -> str:
    """One line for logs: what we are on and what it is called."""
    dev = pick(device)
    if dev == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
            return f"cuda ({name}, {cap})"
        except Exception:
            return "cuda"
    if dev == "mps":
        return "mps (Apple Silicon)"
    return f"cpu ({os.cpu_count()} cores)"


def mps_needs_restarts(device: str | None = None) -> bool:
    """True when the caller should render a chapter per process."""
    return pick(device) == "mps"
