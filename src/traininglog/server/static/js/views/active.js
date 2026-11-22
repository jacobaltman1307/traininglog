import {
  apiFetch,
  computeETA,
  formatDuration,
  formatNumber,
  formatPct,
  formatStep,
  formatTimeAgo,
} from "../api.js";
import { Sparkline } from "../uplot_chart.js";
import { html, useEffect, useState } from "../../vendor/ui.mjs";

const SPARK_METRICS = [
  { key: "game/win_rate_overall", label: "Win %", color: "#34d399", pct: true },
  { key: "losses/policy_loss", label: "Policy Loss", color: "#e59838", pct: false },
  { key: "losses/entropy", label: "Entropy", color: "#fbbf24", pct: false },
];

export function ActiveView({ runs, onOpenRun, onOpenAlerts, onRefresh }) {
  const [sparks, setSparks] = useState({});
  const [cmdTarget, setCmdTarget] = useState(null);
  const [cmdSending, setCmdSending] = useState(false);

  const activeRuns = runs.filter(
    (r) => r.state === "running" || r.state === "stalled"
  );
  const recentRuns = runs
    .filter((r) => r.state !== "running" && r.state !== "stalled")
    .slice(0, 6);
  const displayed = activeRuns.length > 0 ? [...activeRuns, ...recentRuns] : runs.slice(0, 8);

  useEffect(() => {
    let cancelled = false;
    const ids = displayed.slice(0, 12).map((r) => r.id);
    if (ids.length === 0) return;

    Promise.all(
      ids.map(async (rid) => {
        try {
          const qs = SPARK_METRICS.map((m) => m.key).join(",");
          const res = await apiFetch(
            `/api/runs/${rid}/series?metrics=${encodeURIComponent(qs)}&max_points=60`
          );
          return [rid, res.series || {}];
        } catch {
          return [rid, {}];
        }
      })
    ).then((pairs) => {
      if (cancelled) return;
      const next = {};
      for (const [rid, s] of pairs) next[rid] = s;
      setSparks(next);
    });

    return () => {
      cancelled = true;
    };
  }, [runs.map((r) => `${r.id}:${r.current_step}:${r.state}`).join("|")]);

  async function sendCommand(runId, cmdType) {
    setCmdSending(true);
    try {
      await apiFetch(`/api/runs/${runId}/commands`, {
        method: "POST",
        body: JSON.stringify({ cmd_type: cmdType, payload: {} }),
      });
      setCmdTarget(null);
      if (onRefresh) onRefresh();
    } catch (err) {
      alert(`Command failed: ${err.message}`);
    } finally {
      setCmdSending(false);
    }
  }

  if (runs.length === 0) {
    return html`
      <div class="card empty-state">
        <h3 style="margin:0 0 6px;">No runs logged yet</h3>
        <p class="muted" style="margin:0 0 12px;">
          Start a training script or launch the built-in cardgame PPO demo:
        </p>
        <code class="mono" style="background:var(--bg);padding:6px 10px;border-radius:8px;display:inline-block;">
          uv run traininglog demo
        </code>
      </div>
    `;
  }

  return html`
    <div class="stack">
      <div class="space-between">
        <div>
          <h2 style="margin:0; font-size:17px;">
            ${activeRuns.length > 0
              ? `Active Training (${activeRuns.length})`
              : "Recent Runs"}
          </h2>
          <div class="muted" style="font-size:12px;">
            Live RL vitals, winrates & sparklines
          </div>
        </div>
        <button class="btn sm" onClick=${onRefresh}>Refresh</button>
      </div>

      <div class="grid-cards">
        ${displayed.map((run) => {
          const sum = run.summary || {};
          const gpu = run.last_gpu || {};
          const eta = computeETA(run);
          const pct =
            run.total_timesteps && run.total_timesteps > 0
              ? Math.min(100, (run.current_step / run.total_timesteps) * 100)
              : null;
          const winOverall = sum["game/win_rate_overall"];
          const winP1 = sum["game/win_rate_p1"];
          const winP2 = sum["game/win_rate_p2"];
          const runSparks = sparks[run.id] || {};

          return html`
            <div class="card" key=${run.id} style="cursor:pointer;" onClick=${() => onOpenRun(run.id)}>
              <div class="space-between" style="align-items:flex-start;">
                <div style="flex:1; min-width:0;">
                  <div style="font-weight:700; font-size:14px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    ${run.name}
                  </div>
                  <div class="row" style="gap:6px; margin-top:4px;">
                    <span class=${`pill ${run.state}`}>
                      <span class=${`dot ${run.state === "running" ? "pulse" : ""}`}></span>
                      ${run.state}
                    </span>
                    ${run.arch ? html`<span class="pill accent">${run.arch}</span>` : null}
                    <span class="pill">${run.project_name}</span>
                  </div>
                </div>
                <div class="row" style="gap:4px; flex-shrink:0;" onClick=${(e) => e.stopPropagation()}>
                  ${run.open_anomaly_count > 0
                    ? html`<button
                        class="pill critical"
                        style="cursor:pointer;"
                        onClick=${() => onOpenAlerts(run.id)}
                      >
                        ${run.open_anomaly_count} alert${run.open_anomaly_count > 1 ? "s" : ""}
                      </button>`
                    : null}
                  ${run.state === "running" || run.state === "stalled"
                    ? html`<button
                        class="btn sm"
                        title="Remote Controls"
                        onClick=${() => setCmdTarget(run)}
                      >
                        Control
                      </button>`
                    : null}
                </div>
              </div>

              <div style="margin-top:10px;">
                <div class="space-between" style="font-size:12px; margin-bottom:4px;">
                  <span class="mono">
                    Step <strong>${formatStep(run.current_step)}</strong>
                    ${run.total_timesteps
                      ? html`<span class="muted"> / ${formatStep(run.total_timesteps)}</span>`
                      : null}
                  </span>
                  <span class="muted mono">
                    ${eta !== null
                      ? `ETA ${formatDuration(eta)}`
                      : formatTimeAgo(run.last_heartbeat_at)}
                  </span>
                </div>
                ${pct !== null
                  ? html`<div class="progress-track">
                      <div class="progress-fill" style=${`width:${pct.toFixed(1)}%`}></div>
                    </div>`
                  : null}
              </div>

              <div class="stat-row" style="margin-top:10px;">
                <div class="stat-box">
                  <div class="stat-label">SPS</div>
                  <div class="stat-val mono">${formatNumber(run.current_sps || sum["charts/SPS"])}</div>
                </div>
                <div class="stat-box">
                  <div class="stat-label">Win Rate</div>
                  <div class="stat-val mono">
                    ${winOverall != null ? formatPct(winOverall) : "—"}
                  </div>
                  ${winP1 != null && winP2 != null
                    ? html`<div class="muted mono" style="font-size:10px;">
                        P1 ${formatPct(winP1)} · P2 ${formatPct(winP2)}
                      </div>`
                    : null}
                </div>
                <div class="stat-box">
                  <div class="stat-label">Hardware</div>
                  <div class="stat-val mono" style="font-size:12px;">
                    ${gpu.gpu_util != null
                      ? `${Math.round(gpu.gpu_util)}% GPU`
                      : sum["sys/cpu_pct"] != null
                        ? `${Math.round(sum["sys/cpu_pct"])}% CPU`
                        : "—"}
                  </div>
                  ${gpu.gpu_temp_c != null || gpu.gpu_vram_pct != null
                    ? html`<div class="muted mono" style="font-size:10px;">
                        ${gpu.gpu_vram_pct != null ? `${Math.round(gpu.gpu_vram_pct)}% VRAM` : ""}
                        ${gpu.gpu_temp_c != null ? ` · ${Math.round(gpu.gpu_temp_c)}°C` : ""}
                      </div>`
                    : sum["sys/ram_pct"] != null
                      ? html`<div class="muted mono" style="font-size:10px;">
                          ${Math.round(sum["sys/ram_pct"])}% RAM
                        </div>`
                      : null}
                </div>
              </div>

              <div class="spark-grid">
                ${SPARK_METRICS.map((sm) => {
                  const series = runSparks[sm.key];
                  const latest = sum[sm.key];
                  return html`
                    <div class="spark-box" key=${sm.key}>
                      <div class="space-between" style="font-size:10px; min-width:0; gap:6px;">
                        <span class="muted" style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${sm.label}</span>
                        <span class="mono" style=${`color:${sm.color};font-weight:700;flex-shrink:0;`}>
                          ${latest != null
                            ? sm.pct
                              ? formatPct(latest)
                              : formatNumber(latest)
                            : "—"}
                        </span>
                      </div>
                      <${Sparkline} ys=${series ? series.y : []} color=${sm.color} />
                    </div>
                  `;
                })}
              </div>
            </div>
          `;
        })}
      </div>

      ${cmdTarget
        ? html`
            <div class="modal-backdrop" onClick=${() => setCmdTarget(null)}>
              <div class="modal-card stack" onClick=${(e) => e.stopPropagation()}>
                <div class="space-between">
                  <strong>Remote Control: ${cmdTarget.name}</strong>
                  <button class="btn sm" onClick=${() => setCmdTarget(null)}>Close</button>
                </div>
                <p class="muted" style="margin:0; font-size:12px;">
                  Commands are queued cooperatively and picked up safely on the next PPO update via <code class="mono">run.poll_commands()</code>.
                </p>
                <div class="stack">
                  <button
                    class="btn"
                    disabled=${cmdSending}
                    onClick=${() => sendCommand(cmdTarget.id, "checkpoint")}
                  >
                    Save Checkpoint Now
                  </button>
                  <button
                    class="btn"
                    disabled=${cmdSending}
                    onClick=${() => sendCommand(cmdTarget.id, "eval")}
                  >
                    Trigger Eval Match Now
                  </button>
                  <button
                    class="btn warn"
                    disabled=${cmdSending}
                    onClick=${() => sendCommand(cmdTarget.id, "run_redo")}
                  >
                    Run ReDo Plasticity Check
                  </button>
                  <button
                    class="btn danger"
                    disabled=${cmdSending}
                    onClick=${() => {
                      if (window.confirm(`Send graceful STOP request to ${cmdTarget.name}?`)) {
                        sendCommand(cmdTarget.id, "stop");
                      }
                    }}
                  >
                    Graceful Stop Training
                  </button>
                </div>
              </div>
            </div>
          `
        : null}
    </div>
  `;
}
