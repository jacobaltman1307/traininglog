"""Background CPU, RAM, and GPU system vitals collector."""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any, Callable


def _read_proc_stat() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline()
        if not line.startswith("cpu "):
            return None
        parts = [int(p) for p in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        return total, idle
    except (OSError, ValueError, IndexError):
        return None


def _read_proc_meminfo() -> dict[str, float]:
    try:
        mem_total_kb = 0
        mem_avail_kb = 0
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
                if mem_total_kb and mem_avail_kb:
                    break
        if mem_total_kb <= 0:
            return {}
        used_kb = max(0, mem_total_kb - mem_avail_kb)
        total_gb = mem_total_kb / (1024.0 * 1024.0)
        used_gb = used_kb / (1024.0 * 1024.0)
        pct = (used_kb / mem_total_kb) * 100.0
        return {
            "sys/ram_used_gb": round(used_gb, 3),
            "sys/ram_total_gb": round(total_gb, 3),
            "sys/ram_pct": round(pct, 2),
        }
    except (OSError, ValueError, IndexError):
        return {}


def _query_nvidia_smi() -> tuple[dict[str, float], dict[str, Any]] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 5:
            return None

        def _num(s: str) -> float | None:
            try:
                v = float(s)
                return v
            except ValueError:
                return None

        name = parts[0]
        util = _num(parts[1])
        vram_used = _num(parts[2])
        vram_total = _num(parts[3])
        temp = _num(parts[4])
        power = _num(parts[5]) if len(parts) > 5 else None

        metrics: dict[str, float] = {}
        snap: dict[str, Any] = {"gpu_name": name}
        if util is not None:
            metrics["sys/gpu_util"] = util
            snap["gpu_util"] = util
        if vram_used is not None:
            metrics["sys/gpu_vram_used_mb"] = vram_used
            snap["gpu_vram_used_mb"] = vram_used
        if vram_total is not None and vram_total > 0:
            metrics["sys/gpu_vram_total_mb"] = vram_total
            snap["gpu_vram_total_mb"] = vram_total
            if vram_used is not None:
                vram_pct = round((vram_used / vram_total) * 100.0, 2)
                metrics["sys/gpu_vram_pct"] = vram_pct
                snap["gpu_vram_pct"] = vram_pct
        if temp is not None:
            metrics["sys/gpu_temp_c"] = temp
            snap["gpu_temp_c"] = temp
        if power is not None:
            metrics["sys/gpu_power_w"] = power
            snap["gpu_power_w"] = power
        return metrics, snap
    except (OSError, subprocess.SubprocessError):
        return None


class SystemMetricsCollector:
    """Periodic daemon sampler for CPU, RAM, and GPU vitals."""

    def __init__(
        self,
        callback: Callable[[dict[str, float], dict[str, Any]], None],
        interval_sec: float = 5.0,
    ):
        self._callback = callback
        self._interval = max(1.0, float(interval_sec))
        self._stop_event = threading.Event()
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        self._prev_cpu = _read_proc_stat()
        self._thread = threading.Thread(
            target=self._run_loop, name="traininglog-sysmetrics", daemon=True
        )
        self._thread.start()

    def _sample_once(self) -> None:
        metrics: dict[str, float] = {}
        gpu_snap: dict[str, Any] = {}

        cur_cpu = _read_proc_stat()
        if cur_cpu is not None and self._prev_cpu is not None:
            dt_total = cur_cpu[0] - self._prev_cpu[0]
            dt_idle = cur_cpu[1] - self._prev_cpu[1]
            if dt_total > 0:
                cpu_pct = max(0.0, min(100.0, (1.0 - dt_idle / dt_total) * 100.0))
                metrics["sys/cpu_pct"] = round(cpu_pct, 2)
        self._prev_cpu = cur_cpu

        metrics.update(_read_proc_meminfo())

        if self._has_nvidia:
            gpu_res = _query_nvidia_smi()
            if gpu_res is not None:
                g_metrics, g_snap = gpu_res
                metrics.update(g_metrics)
                gpu_snap.update(g_snap)

        if metrics or gpu_snap:
            self._callback(metrics, gpu_snap)

    def _run_loop(self) -> None:
        # Initial quick sample after 0.5s, then interval loop
        if self._stop_event.wait(0.5):
            return
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:
                pass
            if self._stop_event.wait(self._interval):
                break

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
