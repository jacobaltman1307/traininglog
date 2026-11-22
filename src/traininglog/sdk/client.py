"""HTTP client for communicating with the TrainingLog server."""

from __future__ import annotations

from typing import Any

import httpx


class SyncHttpClient:
    """Thin synchronous HTTP client wrapping httpx.Client."""

    def __init__(
        self,
        server_url: str,
        api_token: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        self._client = httpx.Client(
            base_url=self.server_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post("/api/runs", json=payload)
        resp.raise_for_status()
        return resp.json()

    def post_events(
        self, run_id: str, seq: int, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        resp = self._client.post(
            f"/api/runs/{run_id}/events",
            json={"seq": seq, "events": events},
        )
        resp.raise_for_status()
        return resp.json()

    def finish_run(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(f"/api/runs/{run_id}/finish", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_pending_commands(self, run_id: str) -> list[dict[str, Any]]:
        resp = self._client.get(
            f"/api/runs/{run_id}/commands",
            params={"status": "pending"},
        )
        resp.raise_for_status()
        return list(resp.json().get("commands") or [])

    def ack_command(
        self,
        run_id: str,
        cmd_id: str,
        status: str = "acked",
        result_message: str | None = None,
    ) -> dict[str, Any]:
        resp = self._client.post(
            f"/api/runs/{run_id}/commands/{cmd_id}/ack",
            json={"status": status, "result_message": result_message},
        )
        resp.raise_for_status()
        return resp.json()
