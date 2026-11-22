"""Webhook & push notification dispatcher (ntfy, Discord, Pushover, Telegram, generic webhook)."""

from __future__ import annotations

import html
from typing import Any

import httpx

from .db import Database, now_ms


class AlertDispatcher:
    """Dispatches rule-matched events to external notification channels with cooldowns."""

    def __init__(
        self,
        db: Database,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.db = db
        self._transport = transport

    def _should_skip_cooldown(
        self, rule: dict[str, Any], run_id: str | None, event_type: str
    ) -> bool:
        cooldown_sec = int(rule.get("cooldown_sec", 300))
        if cooldown_sec <= 0:
            return False
        last_ts = self.db.last_alert_sent_at(rule["id"], run_id, event_type)
        if last_ts is None:
            return False
        return (now_ms() - last_ts) < (cooldown_sec * 1000)

    async def _send_to_channel(
        self,
        rule: dict[str, Any],
        *,
        event_type: str,
        severity: str,
        title: str,
        message: str,
        run: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, int | None, str | None, dict[str, Any]]:
        channel = rule["channel"]
        target = rule["target"]
        payload_log: dict[str, Any] = {}

        async with httpx.AsyncClient(
            transport=self._transport, timeout=10.0
        ) as client:
            try:
                if channel == "ntfy":
                    url = str(target["url"])
                    priority = (
                        "urgent"
                        if severity == "critical"
                        else "high"
                        if severity == "warning"
                        else "default"
                    )
                    tag = (
                        "rotating_light"
                        if severity == "critical"
                        else "warning"
                        if severity == "warning"
                        else "white_check_mark"
                    )
                    headers = {
                        "Title": title,
                        "Priority": priority,
                        "Tags": tag,
                    }
                    if target.get("token"):
                        headers["Authorization"] = f"Bearer {target['token']}"
                    elif target.get("auth_header"):
                        headers["Authorization"] = str(target["auth_header"])
                    payload_log = {"url": url, "headers": {k: v for k, v in headers.items() if k != "Authorization"}, "body": message}
                    resp = await client.post(url, content=message.encode("utf-8"), headers=headers)

                elif channel == "discord":
                    url = str(target["url"])
                    color = (
                        0xEF4444
                        if severity == "critical"
                        else 0xF59E0B
                        if severity == "warning"
                        else 0x10B981
                    )
                    fields = [
                        {"name": "Event", "value": event_type, "inline": True},
                    ]
                    if run:
                        fields.append(
                            {"name": "Run", "value": str(run.get("name") or run.get("id")), "inline": True}
                        )
                        fields.append(
                            {"name": "Step", "value": f"{int(run.get('current_step') or 0):,}", "inline": True}
                        )
                    body = {
                        "embeds": [
                            {
                                "title": title,
                                "description": message,
                                "color": color,
                                "fields": fields,
                            }
                        ]
                    }
                    payload_log = body
                    resp = await client.post(url, json=body)

                elif channel == "pushover":
                    url = str(target.get("url") or "https://api.pushover.net/1/messages.json")
                    form_data = {
                        "token": str(target["token"]),
                        "user": str(target["user"]),
                        "title": title,
                        "message": message,
                        "priority": "1" if severity == "critical" else "0",
                    }
                    payload_log = {
                        "url": url,
                        "title": title,
                        "message": message,
                        "priority": form_data["priority"],
                    }
                    resp = await client.post(url, data=form_data)

                elif channel == "telegram":
                    bot_token = str(target.get("bot_token") or "")
                    url = str(
                        target.get("url")
                        or f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    )
                    text = f"<b>{html.escape(title)}</b>\n{html.escape(message)}"
                    body = {
                        "chat_id": str(target["chat_id"]),
                        "text": text,
                        "parse_mode": "HTML",
                    }
                    payload_log = {"url": url, "chat_id": body["chat_id"], "text": text}
                    resp = await client.post(url, json=body)

                elif channel == "webhook":
                    url = str(target["url"])
                    headers = dict(target.get("headers") or {})
                    body = {
                        "event_type": event_type,
                        "severity": severity,
                        "title": title,
                        "message": message,
                        "run": {
                            "id": run.get("id"),
                            "name": run.get("name"),
                            "project_id": run.get("project_id"),
                            "project_name": run.get("project_name"),
                            "state": run.get("state"),
                            "current_step": run.get("current_step"),
                        }
                        if run
                        else None,
                        "metadata": metadata or {},
                        "sent_at": now_ms(),
                    }
                    payload_log = body
                    resp = await client.post(url, json=body, headers=headers)

                else:
                    return "failed", None, f"Unsupported channel: {channel}", payload_log

                if 200 <= resp.status_code < 300:
                    return "sent", resp.status_code, None, payload_log
                return (
                    "failed",
                    resp.status_code,
                    f"HTTP {resp.status_code}: {resp.text[:300]}",
                    payload_log,
                )
            except Exception as exc:
                return "failed", None, str(exc), payload_log

    async def dispatch_event(
        self,
        *,
        event_type: str,
        project_id: str | None,
        run: dict[str, Any] | None,
        title: str,
        message: str,
        severity: str = "warning",
        anomaly_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rules = self.db.list_alert_rules(enabled_only=True)
        run_id = run["id"] if run else None
        results: list[dict[str, Any]] = []

        for rule in rules:
            if rule["project_id"] is not None and rule["project_id"] != project_id:
                continue
            if event_type not in rule["events"]:
                continue
            if self._should_skip_cooldown(rule, run_id, event_type):
                continue

            status, code, err, payload_log = await self._send_to_channel(
                rule,
                event_type=event_type,
                severity=severity,
                title=title,
                message=message,
                run=run,
                metadata=metadata,
            )
            ev = self.db.record_alert_event(
                rule_id=rule["id"],
                run_id=run_id,
                anomaly_id=anomaly_id,
                event_type=event_type,
                status=status,
                response_code=code,
                error=err,
                payload=payload_log,
            )
            results.append(ev)

        return results

    async def send_test(self, rule_id: str) -> dict[str, Any] | None:
        rule = self.db.get_alert_rule(rule_id)
        if rule is None:
            return None
        fake_run = {
            "id": "run_test",
            "name": "test-cardgame-ppo",
            "project_id": rule["project_id"] or "proj_test",
            "project_name": "cardgame-ppo",
            "state": "running",
            "current_step": 125_000_000,
        }
        status, code, err, payload_log = await self._send_to_channel(
            rule,
            event_type="test",
            severity="info",
            title="[TrainingLog] Test Alert",
            message=f"Test notification for rule '{rule['name']}' via {rule['channel']}.",
            run=fake_run,
            metadata={"test": True},
        )
        return self.db.record_alert_event(
            rule_id=rule["id"],
            run_id=None,
            anomaly_id=None,
            event_type="test",
            status=status,
            response_code=code,
            error=err,
            payload=payload_log,
        )
