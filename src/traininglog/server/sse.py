"""In-process Server-Sent Events (SSE) event bus."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class _Subscriber:
    queue: asyncio.Queue[dict[str, Any]]
    loop: asyncio.AbstractEventLoop
    run_id: str | None


def _put_drop_oldest(q: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


class EventBus:
    """Thread-safe pub/sub bus for streaming live server events to SSE clients."""

    def __init__(self, max_queue_size: int = 256):
        self._lock = threading.Lock()
        self._subscribers: list[_Subscriber] = []
        self._max_queue_size = max_queue_size

    def subscribe(self, run_id: str | None = None) -> _Subscriber:
        loop = asyncio.get_running_loop()
        sub = _Subscriber(
            queue=asyncio.Queue(maxsize=self._max_queue_size),
            loop=loop,
            run_id=run_id,
        )
        with self._lock:
            self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: _Subscriber) -> None:
        with self._lock:
            try:
                self._subscribers.remove(sub)
            except ValueError:
                pass

    def publish(
        self,
        event_type: str,
        data: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        payload = {"type": event_type, "run_id": run_id, "data": data}
        with self._lock:
            subs = list(self._subscribers)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        for sub in subs:
            if sub.run_id is not None and run_id is not None and sub.run_id != run_id:
                continue
            if sub.run_id is not None and run_id is None and event_type not in ("run_created", "anomaly"):
                continue
            try:
                if running_loop is sub.loop:
                    _put_drop_oldest(sub.queue, payload)
                else:
                    sub.loop.call_soon_threadsafe(_put_drop_oldest, sub.queue, payload)
            except RuntimeError:
                # Subscriber event loop closed
                self.unsubscribe(sub)
