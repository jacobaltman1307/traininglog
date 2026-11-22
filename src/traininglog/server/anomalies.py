"""Automatic RL health & anomaly detection engine."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .alerts import AlertDispatcher
from .db import Database, now_ms
from .sse import EventBus

DEFAULT_KL_THRESHOLD = 0.1
DEFAULT_ENTROPY_FLOOR = 0.05
DEFAULT_SPS_DROP_RATIO = 0.40  # val < median * 0.40 => >60% drop
DEDUP_WINDOW_MS = 30_000


@dataclass
class _RunHealthState:
    point_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sps_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    last_alert_ms: dict[tuple[str, str], int] = field(default_factory=dict)


def _is_kl_metric(metric: str) -> bool:
    low = metric.lower()
    return low.endswith("approx_kl") or low.endswith("/kl") or low == "kl"


def _is_entropy_metric(metric: str) -> bool:
    low = metric.lower()
    return (
        low.endswith("/entropy")
        or low.endswith("/entropy_hand")
        or low.endswith("/entropy_board")
        or low in ("entropy", "entropy_hand", "entropy_board")
    )


def _is_sps_metric(metric: str) -> bool:
    low = metric.lower()
    return low.endswith("/sps") or low == "sps"


class AnomalyDetector:
    """Inspects incoming scalar streams and run heartbeats for RL pathologies."""

    def __init__(
        self,
        db: Database,
        event_bus: EventBus,
        alert_dispatcher: AlertDispatcher,
    ):
        self.db = db
        self.event_bus = event_bus
        self.alert_dispatcher = alert_dispatcher
        self._lock = threading.Lock()
        self._states: dict[str, _RunHealthState] = {}

    def _get_state(self, run_id: str) -> _RunHealthState:
        with self._lock:
            st = self._states.get(run_id)
            if st is None:
                st = _RunHealthState()
                self._states[run_id] = st
            return st

    def clear_run(self, run_id: str) -> None:
        with self._lock:
            self._states.pop(run_id, None)

    def _get_thresholds(self, project_id: str) -> tuple[float, float]:
        kl_thr = DEFAULT_KL_THRESHOLD
        ent_floor = DEFAULT_ENTROPY_FLOOR
        for rule in self.db.list_alert_rules(enabled_only=True):
            if rule["project_id"] is not None and rule["project_id"] != project_id:
                continue
            th = rule.get("thresholds") or {}
            if "kl_threshold" in th and th["kl_threshold"] is not None:
                kl_thr = float(th["kl_threshold"])
            if "entropy_floor" in th and th["entropy_floor"] is not None:
                ent_floor = float(th["entropy_floor"])
        return kl_thr, ent_floor

    def _can_emit(self, st: _RunHealthState, kind: str, metric: str, ts_ms: int) -> bool:
        key = (kind, metric)
        last = st.last_alert_ms.get(key)
        if last is not None and (ts_ms - last) < DEDUP_WINDOW_MS:
            return False
        st.last_alert_ms[key] = ts_ms
        return True

    async def inspect_scalars(
        self,
        run: dict[str, Any],
        points_by_metric: dict[str, list[tuple[int, int, float]]],
    ) -> list[dict[str, Any]]:
        if not points_by_metric:
            return []

        run_id = run["id"]
        project_id = run["project_id"]
        st = self._get_state(run_id)
        kl_thr, ent_floor = self._get_thresholds(project_id)
        detected: list[tuple[str, str, str, int, float | None, str]] = []
        now = int(time.time() * 1000)

        with self._lock:
            for metric, pts in points_by_metric.items():
                for step, _wall_ms, val in pts:
                    st.point_counts[metric] += 1
                    cnt = st.point_counts[metric]

                    if not math.isfinite(val):
                        if self._can_emit(st, "nan_inf", metric, now):
                            detected.append(
                                (
                                    "nan_inf",
                                    "critical",
                                    metric,
                                    int(step),
                                    None,
                                    f"Non-finite value ({val}) in '{metric}' at step {int(step):,}",
                                )
                            )
                        continue

                    if _is_kl_metric(metric) and cnt >= 5 and val > kl_thr:
                        if self._can_emit(st, "kl_spike", metric, now):
                            detected.append(
                                (
                                    "kl_spike",
                                    "warning",
                                    metric,
                                    int(step),
                                    float(val),
                                    f"KL divergence spike in '{metric}': {val:.4f} > {kl_thr:.4f} at step {int(step):,}",
                                )
                            )

                    if _is_entropy_metric(metric) and cnt >= 10 and val < ent_floor:
                        if self._can_emit(st, "entropy_collapse", metric, now):
                            low = metric.lower()
                            if "hand" in low:
                                label = "Hand policy entropy"
                            elif "board" in low:
                                label = "Board policy entropy"
                            else:
                                label = "Policy entropy"
                            detected.append(
                                (
                                    "entropy_collapse",
                                    "warning",
                                    metric,
                                    int(step),
                                    float(val),
                                    f"{label} collapsed in '{metric}': {val:.4f} < {ent_floor:.4f} at step {int(step):,}",
                                )
                            )

                    if _is_sps_metric(metric):
                        if len(st.sps_history) >= 10:
                            baseline = statistics.median(list(st.sps_history)[-10:])
                            if baseline > 100.0 and val < baseline * DEFAULT_SPS_DROP_RATIO:
                                if self._can_emit(st, "sps_drop", metric, now):
                                    drop_pct = (1.0 - (val / baseline)) * 100.0
                                    detected.append(
                                        (
                                            "sps_drop",
                                            "warning",
                                            metric,
                                            int(step),
                                            float(val),
                                            f"Throughput drop in '{metric}': {val:,.0f} SPS ({drop_pct:.0f}% below {baseline:,.0f} baseline)",
                                        )
                                    )
                        st.sps_history.append(float(val))

        emitted: list[dict[str, Any]] = []
        for kind, severity, metric, step, val, msg in detected:
            row = self.db.insert_anomaly(
                run_id=run_id,
                project_id=project_id,
                kind=kind,
                severity=severity,
                metric=metric,
                step=step,
                value=val,
                message=msg,
            )
            row["run_name"] = run["name"]
            row["project_name"] = run.get("project_name")
            emitted.append(row)
            self.event_bus.publish("anomaly", row, run_id=run_id)
            await self.alert_dispatcher.dispatch_event(
                event_type=kind,
                project_id=project_id,
                run=run,
                title=f"[{run['name']}] {kind.replace('_', ' ').title()}",
                message=msg,
                severity=severity,
                anomaly_id=row["id"],
                metadata={"metric": metric, "step": step, "value": val},
            )

        return emitted

    async def check_stalled_runs(self, stall_timeout_sec: int) -> list[dict[str, Any]]:
        cutoff_ms = now_ms() - int(stall_timeout_sec) * 1000
        stale_runs = self.db.find_stale_running_runs(cutoff_ms)
        stalled_anomalies: list[dict[str, Any]] = []
        for run in stale_runs:
            updated = self.db.mark_run_stalled(run["id"])
            if updated is None:
                continue
            msg = f"Run '{updated['name']}' stalled (no heartbeat for >{stall_timeout_sec}s at step {updated['current_step']:,})"
            anom = self.db.insert_anomaly(
                run_id=updated["id"],
                project_id=updated["project_id"],
                kind="stalled",
                severity="critical",
                step=updated["current_step"],
                message=msg,
            )
            anom["run_name"] = updated["name"]
            anom["project_name"] = updated.get("project_name")
            stalled_anomalies.append(anom)
            self.event_bus.publish("run_updated", updated, run_id=updated["id"])
            self.event_bus.publish("anomaly", anom, run_id=updated["id"])
            await self.alert_dispatcher.dispatch_event(
                event_type="stalled",
                project_id=updated["project_id"],
                run=updated,
                title=f"[{updated['name']}] Run Stalled",
                message=msg,
                severity="critical",
                anomaly_id=anom["id"],
            )
        return stalled_anomalies

    async def on_run_finished(
        self, run: dict[str, Any], error_message: str | None = None
    ) -> None:
        self.clear_run(run["id"])
        self.event_bus.publish("run_updated", run, run_id=run["id"])

        if run["state"] == "crashed":
            msg = error_message or f"Run '{run['name']}' crashed at step {run['current_step']:,}"
            anom = self.db.insert_anomaly(
                run_id=run["id"],
                project_id=run["project_id"],
                kind="crashed",
                severity="critical",
                step=run["current_step"],
                message=msg,
            )
            anom["run_name"] = run["name"]
            anom["project_name"] = run.get("project_name")
            self.event_bus.publish("anomaly", anom, run_id=run["id"])
            await self.alert_dispatcher.dispatch_event(
                event_type="crashed",
                project_id=run["project_id"],
                run=run,
                title=f"[{run['name']}] Run Crashed",
                message=msg,
                severity="critical",
                anomaly_id=anom["id"],
            )
        elif run["state"] == "finished":
            await self.alert_dispatcher.dispatch_event(
                event_type="finished",
                project_id=run["project_id"],
                run=run,
                title=f"[{run['name']}] Run Finished",
                message=f"Run '{run['name']}' completed at step {run['current_step']:,}.",
                severity="info",
            )

    async def inspect_plasticity(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """Check ReDo dormant neuron % and mean effective rank ratio against diagnostic thresholds."""
        project_id = report["project_id"]
        run_id = report.get("run_id")
        run = self.db.get_run(run_id) if run_id else None
        model_name = report.get("model_name") or "model"
        step = report.get("step")

        dormant_pct = float(report.get("dormant_pct") or 0.0)
        mean_rank_pct = float(report.get("mean_rank_ratio_pct") or 0.0)
        findings: list[tuple[str, str, str, float, str]] = []

        if dormant_pct > 35.0:
            findings.append(
                (
                    "high_dormancy",
                    "critical",
                    "plasticity/dormant_pct",
                    dormant_pct,
                    f"CRITICAL DORMANCY in '{model_name}': {dormant_pct:.1f}% dormant neurons (>35%)",
                )
            )
        elif dormant_pct > 20.0:
            findings.append(
                (
                    "high_dormancy",
                    "warning",
                    "plasticity/dormant_pct",
                    dormant_pct,
                    f"ELEVATED DORMANCY in '{model_name}': {dormant_pct:.1f}% dormant neurons (>20%)",
                )
            )

        if mean_rank_pct < 20.0:
            findings.append(
                (
                    "rank_collapse",
                    "warning",
                    "plasticity/mean_rank_ratio_pct",
                    mean_rank_pct,
                    f"FEATURE COLLAPSE in '{model_name}': mean effective rank {mean_rank_pct:.1f}% (<20%)",
                )
            )

        emitted: list[dict[str, Any]] = []
        for kind, severity, metric, val, msg in findings:
            anom_id = None
            if run_id and run is not None:
                row = self.db.insert_anomaly(
                    run_id=run_id,
                    project_id=project_id,
                    kind=kind,
                    severity=severity,
                    metric=metric,
                    step=step,
                    value=val,
                    message=msg,
                )
                row["run_name"] = run["name"]
                row["project_name"] = run.get("project_name")
                anom_id = row["id"]
                emitted.append(row)
                self.event_bus.publish("anomaly", row, run_id=run_id)
            else:
                synthetic = {
                    "id": None,
                    "run_id": None,
                    "project_id": project_id,
                    "kind": kind,
                    "severity": severity,
                    "metric": metric,
                    "step": step,
                    "value": val,
                    "message": msg,
                    "model_name": model_name,
                }
                emitted.append(synthetic)
                self.event_bus.publish("anomaly", synthetic, run_id=None)

            await self.alert_dispatcher.dispatch_event(
                event_type=kind,
                project_id=project_id,
                run=run,
                title=f"[{model_name}] {kind.replace('_', ' ').title()}",
                message=msg,
                severity=severity,
                anomaly_id=anom_id,
                metadata={"model_name": model_name, "metric": metric, "value": val},
            )

        return emitted

    async def on_new_best_rating(
        self,
        project_id: str,
        agent: dict[str, Any],
        rating_mu: float,
        rating_rd: float,
    ) -> None:
        run_id = agent.get("run_id")
        run = self.db.get_run(run_id) if run_id else None
        msg = (
            f"New league high rating for '{agent['name']}': "
            f"{rating_mu:.1f} ± {2.0 * rating_rd:.1f} Glicko-2 (95% CI)"
        )
        await self.alert_dispatcher.dispatch_event(
            event_type="new_best_elo",
            project_id=project_id,
            run=run,
            title=f"New Best Rating: {agent['name']}",
            message=msg,
            severity="info",
            metadata={"agent_id": agent["id"], "glicko_mu": rating_mu, "glicko_rd": rating_rd},
        )
