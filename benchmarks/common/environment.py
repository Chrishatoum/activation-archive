"""Record the machine/software state a benchmark ran under, and control page cache state.

Per EVALUATION_PROTOCOL.md Sec 5.3, every benchmark result must be self-describing
(hostname, GPU, package versions, timestamp), and cold/warm cache measurements must
be labeled separately rather than left uncontrolled.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = ("numpy", "pandas", "pyarrow", "zstandard", "torch", "zfpy", "pysz")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False}
    return {"available": True, "name": torch.cuda.get_device_name(0)}


def environment_snapshot() -> dict[str, Any]:
    """A JSON-serializable record of the machine/software state for one benchmark run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "gpu": _gpu_info(),
    }


def evict_from_page_cache(path: Path) -> None:
    """Advise the kernel to drop a file (or every file under a directory) from cache.

    Uses posix_fadvise(POSIX_FADV_DONTNEED), which needs no special privileges for
    files the process can already read. Best-effort: silently skipped on platforms
    or filesystems where it is unsupported, since cold-cache measurement is then
    simply unavailable rather than wrong.
    """
    paths = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    for item in paths:
        try:
            fd = os.open(item, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        finally:
            os.close(fd)
