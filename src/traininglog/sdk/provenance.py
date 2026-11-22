"""Git and system provenance collection (never imports torch eagerly)."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from typing import Any

MAX_DIFF_BYTES = 64 * 1024


def _run_git(args: list[str], cwd: str | None = None) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def collect_git_info(cwd: str | None = None) -> dict[str, Any]:
    """Capture branch, commit, dirty flag, and bounded diff if inside a git repo."""
    commit = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    if commit is None:
        return {}
    branch = (_run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd) or "").strip()
    status = (_run_git(["status", "--porcelain"], cwd=cwd) or "").strip()
    dirty = bool(status)
    diff = ""
    if dirty:
        raw_diff = _run_git(["diff", "--no-color"], cwd=cwd) or ""
        encoded = raw_diff.encode("utf-8", errors="replace")
        if len(encoded) > MAX_DIFF_BYTES:
            diff = encoded[:MAX_DIFF_BYTES].decode("utf-8", errors="ignore") + "\n... [truncated]"
        else:
            diff = raw_diff
    return {
        "commit": commit.strip(),
        "short_commit": commit.strip()[:8],
        "branch": branch,
        "dirty": dirty,
        "diff": diff,
    }


def collect_system_info() -> dict[str, Any]:
    """Collect static host/Python/CUDA metadata without importing PyTorch."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        info["torch_version"] = getattr(torch_mod, "__version__", None)
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is not None:
            try:
                is_avail = bool(cuda_mod.is_available())
                info["cuda_available"] = is_avail
                if is_avail:
                    info["cuda_device_count"] = int(cuda_mod.device_count())
                    info["cuda_device_name"] = str(cuda_mod.get_device_name(0))
            except Exception:
                pass
    return info
