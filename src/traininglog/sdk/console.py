"""Non-recursive stdout/stderr tee capturing console output and crash tracebacks."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Callable, TextIO


class _TeeStream:
    """Proxy wrapping stdout or stderr while preserving original terminal writes."""

    def __init__(
        self,
        original: TextIO,
        stream_name: str,
        capture: ConsoleCapture,
    ):
        self._original = original
        self._stream_name = stream_name
        self._capture = capture

    def write(self, data: str) -> int:
        res = self._original.write(data)
        if data:
            self._capture.on_write(self._stream_name, data)
        return res

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._original.flush()
        self._capture.flush(self._stream_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class ConsoleCapture:
    """Tees sys.stdout and sys.stderr safely with a thread-local re-entrancy guard."""

    def __init__(
        self,
        callback: Callable[[int, str, int, str], None],
        flush_interval_sec: float = 0.5,
    ):
        self._callback = callback
        self._flush_interval = flush_interval_sec
        self._local = threading.local()
        self._lock = threading.Lock()
        self._seq = 0
        self._buffers: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self._last_flush: dict[str, float] = {
            "stdout": time.monotonic(),
            "stderr": time.monotonic(),
        }
        self._started = False
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr
            sys.stdout = _TeeStream(self._orig_stdout, "stdout", self)  # type: ignore[assignment]
            sys.stderr = _TeeStream(self._orig_stderr, "stderr", self)  # type: ignore[assignment]
            self._started = True

    def on_write(self, stream_name: str, data: str) -> None:
        if getattr(self._local, "active", False):
            return
        self._local.active = True
        try:
            to_emit: list[tuple[int, str, int, str]] = []
            now_mono = time.monotonic()
            ts_ms = int(time.time() * 1000)
            with self._lock:
                buf = self._buffers[stream_name]
                buf.append(data)
                joined = "".join(buf)
                should_flush = (
                    "\n" in data
                    or len(joined) >= 4096
                    or (now_mono - self._last_flush[stream_name]) >= self._flush_interval
                )
                if should_flush and joined:
                    lines = joined.splitlines(keepends=True)
                    # Keep any incomplete trailing chunk unless buffer is huge
                    if lines and not lines[-1].endswith(("\n", "\r")) and len(joined) < 4096:
                        trailing = lines.pop()
                        self._buffers[stream_name] = [trailing]
                    else:
                        self._buffers[stream_name] = []
                    if lines:
                        chunk = "".join(lines).rstrip("\r\n")
                        if chunk or any(l == "\n" for l in lines):
                            self._seq += 1
                            to_emit.append((self._seq, stream_name, ts_ms, chunk))
                        self._last_flush[stream_name] = now_mono

            for seq, sname, t_ms, text in to_emit:
                self._callback(seq, sname, t_ms, text)
        finally:
            self._local.active = False

    def emit_direct(self, stream_name: str, text: str) -> None:
        """Emit text directly (e.g., formatted crash traceback) without touching stdout/stderr."""
        if not text:
            return
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._callback(seq, stream_name, int(time.time() * 1000), text.rstrip("\r\n"))

    def flush(self, stream_name: str | None = None) -> None:
        if getattr(self._local, "active", False):
            return
        self._local.active = True
        try:
            targets = [stream_name] if stream_name else ["stdout", "stderr"]
            to_emit: list[tuple[int, str, int, str]] = []
            ts_ms = int(time.time() * 1000)
            with self._lock:
                for sname in targets:
                    buf = self._buffers[sname]
                    if buf:
                        joined = "".join(buf)
                        self._buffers[sname] = []
                        if joined:
                            self._seq += 1
                            to_emit.append((self._seq, sname, ts_ms, joined.rstrip("\r\n")))
                        self._last_flush[sname] = time.monotonic()
            for seq, sname, t_ms, text in to_emit:
                self._callback(seq, sname, t_ms, text)
        finally:
            self._local.active = False

    def stop(self) -> None:
        self.flush()
        with self._lock:
            if not self._started:
                return
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
            self._started = False
