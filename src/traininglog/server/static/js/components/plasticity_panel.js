import { formatNumber, formatStep } from "../api.js";
import { UPlotChart } from "../uplot_chart.js";
import { html, useEffect, useMemo, useState } from "../../vendor/ui.mjs";

export function PlasticityPanel({ reports, theme = "dark" }) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  useEffect(() => {
    setSelectedIdx(0);
  }, [reports?.length]);

  const ordered = useMemo(
    () => (reports || []).slice().sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
    [reports]
  );

  const trendData = useMemo(() => {
    const withStep = ordered
      .filter((r) => r.step !== null && r.step !== undefined)
      .slice()
      .sort((a, b) => a.step - b.step);
    if (withStep.length < 2) return null;
    const xs = withStep.map((r) => r.step);
    return {
      seriesMap: {
        dormant: { x: xs, y: withStep.map((r) => r.dormant_pct), non_finite_steps: [] },
        rank: { x: xs, y: withStep.map((r) => r.mean_rank_ratio_pct), non_finite_steps: [] },
      },
      runList: [
        { id: "dormant", name: "Dormant %" },
        { id: "rank", name: "Mean Rank %" },
      ],
    };
  }, [ordered]);

  if (!ordered || ordered.length === 0) {
    return html`<div class="card empty-state">
      No ReDo dormant neuron or effective rank (srank) reports logged yet.
    </div>`;
  }

  const active = ordered[Math.min(selectedIdx, ordered.length - 1)];
  const dormantPct = Number(active.dormant_pct || 0);
  const hardDeadPct = Number(active.hard_dead_pct || 0);
  const weakPct = Number(active.weak_pct || 0);
  const rankPct = Number(active.mean_rank_ratio_pct || 0);

  const dormStatus =
    dormantPct > 35
      ? { label: "CRITICAL DORMANCY", cls: "critical" }
      : dormantPct > 20
        ? { label: "ELEVATED DORMANCY", cls: "warning" }
        : { label: "HEALTHY CAPACITY", cls: "running" };

  const rankStatus =
    rankPct < 20
      ? { label: "FEATURE COLLAPSE", cls: "critical" }
      : rankPct < 40
        ? { label: "MODERATE RANK", cls: "warning" }
        : { label: "HIGH DIVERSITY", cls: "running" };

  return html`
    <div class="stack">
      <div class="card stack">
        <div class="space-between" style="flex-wrap:wrap;">
          <div>
            <strong>ReDo Dormancy & Effective Rank (srank)</strong>
            <div class="muted" style="font-size:12px;">
              ${active.model_name}
              ${active.arch ? ` · ${active.arch}` : ""}
              ${active.step != null ? ` · Step ${formatStep(active.step)}` : ""}
              ${active.total_obs ? ` · ${Number(active.total_obs).toLocaleString()} obs` : ""}
              ${` · Hook: ${active.hook_mode || "post_activation"}`}
            </div>
          </div>
          <div class="row" style="gap:6px;">
            <span class=${`pill ${dormStatus.cls}`}>${dormStatus.label}</span>
            <span class=${`pill ${rankStatus.cls}`}>${rankStatus.label}</span>
          </div>
        </div>

        ${ordered.length > 1
          ? html`
              <div class="row">
                <span class="muted" style="font-size:12px;">Snapshot:</span>
                <select
                  class="select"
                  style="max-width:300px;"
                  value=${selectedIdx}
                  onChange=${(e) => setSelectedIdx(parseInt(e.target.value, 10))}
                >
                  ${ordered.map(
                    (r, idx) => html`
                      <option value=${idx}>
                        ${r.model_name}${r.step != null ? ` @ ${formatStep(r.step)}` : ""} (Dormant ${Number(r.dormant_pct).toFixed(1)}%)
                      </option>
                    `
                  )}
                </select>
              </div>
            `
          : null}

        <div class="stat-row" style="grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));">
          <div class="stat-box">
            <div class="stat-label">Total Neurons</div>
            <div class="stat-val mono">${Number(active.total_neurons || 0).toLocaleString()}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Hard Dead (==0)</div>
            <div class="stat-val mono" style=${hardDeadPct > 20 ? "color:var(--danger);" : ""}>
              ${hardDeadPct.toFixed(1)}%
            </div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Dormant (&le;0.025)</div>
            <div class="stat-val mono" style=${dormantPct > 20 ? "color:var(--warn);" : "color:var(--ok);"}>
              ${dormantPct.toFixed(1)}%
            </div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Weak (&le;0.10)</div>
            <div class="stat-val mono">${weakPct.toFixed(1)}%</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Mean Rank Ratio</div>
            <div class="stat-val mono" style=${rankPct < 20 ? "color:var(--danger);" : "color:var(--accent);"}>
              ${rankPct.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      ${trendData
        ? html`
            <${UPlotChart}
              title="Plasticity Trend Across Steps (Dormant % vs Mean Effective Rank %)"
              seriesMap=${trendData.seriesMap}
              runList=${trendData.runList}
              xAxis="step"
              height=${190}
              theme=${theme}
            />
          `
        : null}

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Layer</th>
              <th>Hook</th>
              <th>Dim</th>
              <th>Hard Dead</th>
              <th>Dormant %</th>
              <th>Effective Rank (srank)</th>
            </tr>
          </thead>
          <tbody>
            ${(active.layers || []).map((lyr) => {
              const dPct = Number(lyr.dormant_pct || 0);
              const rPct = Number(lyr.rank_ratio_pct || 0);
              return html`
                <tr key=${lyr.name}>
                  <td class="mono" style="font-weight:600;">${lyr.name}</td>
                  <td><span class="pill">${lyr.hook_type}</span></td>
                  <td class="mono">${lyr.dim}</td>
                  <td class="mono">${Number(lyr.hard_dead_pct || 0).toFixed(1)}%</td>
                  <td>
                    <div class="bar-cell">
                      <div class="mini-bar-track">
                        <div
                          class="mini-bar-fill"
                          style=${`width:${Math.min(100, dPct).toFixed(1)}%; background:${dPct > 35 ? "var(--danger)" : dPct > 20 ? "var(--warn)" : "var(--ok)"};`}
                        ></div>
                      </div>
                      <span class="mono" style="min-width:46px; text-align:right;">
                        ${dPct.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td>
                    <div class="bar-cell">
                      <div class="mini-bar-track">
                        <div
                          class="mini-bar-fill"
                          style=${`width:${Math.min(100, rPct).toFixed(1)}%; background:${rPct < 20 ? "var(--danger)" : "var(--accent)"};`}
                        ></div>
                      </div>
                      <span class="mono" style="min-width:110px; text-align:right;">
                        ${formatNumber(lyr.effective_rank)}/${lyr.dim} (${rPct.toFixed(1)}%)
                      </span>
                    </div>
                  </td>
                </tr>
              `;
            })}
          </tbody>
        </table>
      </div>
    </div>
  `;
}
