import { apiFetch } from "../api.js";
import { CARDGAME_PRESETS, selectPresetMetrics } from "../presets.js";
import { SERIES_COLORS, UPlotChart } from "../uplot_chart.js";
import { html, useEffect, useMemo, useState } from "../../vendor/ui.mjs";

export function CompareView({
  runs,
  selectedIds,
  onSetSelectedIds,
  onOpenRun,
  theme,
}) {
  const [activeTab, setActiveTab] = useState("overview");
  const [xAxis, setXAxis] = useState("step");
  const [smoothing, setSmoothing] = useState(0.4);
  const [logScale, setLogScale] = useState(false);
  const [metricQuery, setMetricQuery] = useState("");
  const [diffOnly, setDiffOnly] = useState(true);
  const [compareData, setCompareData] = useState(null);
  const [configDiff, setConfigDiff] = useState(null);

  const activeIds = useMemo(() => {
    if (selectedIds.length > 0) return selectedIds;
    return runs.slice(0, 3).map((r) => r.id);
  }, [selectedIds, runs]);

  const selectedRuns = useMemo(() => {
    const map = new Map(runs.map((r) => [r.id, r]));
    return activeIds.map((id) => map.get(id)).filter(Boolean);
  }, [runs, activeIds]);

  useEffect(() => {
    if (activeIds.length === 0) {
      setCompareData(null);
      setConfigDiff(null);
      return;
    }
    let cancelled = false;
    Promise.all([
      apiFetch("/api/compare/series", {
        method: "POST",
        body: JSON.stringify({
          run_ids: activeIds,
          metrics: [],
          max_points: 800,
          x_axis: xAxis,
        }),
      }),
      apiFetch("/api/compare/config-diff", {
        method: "POST",
        body: JSON.stringify({ run_ids: activeIds }),
      }),
    ])
      .then(([seriesRes, diffRes]) => {
        if (cancelled) return;
        setCompareData(seriesRes);
        setConfigDiff(diffRes);
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [activeIds.join("|"), xAxis]);

  const allMetricNames = useMemo(
    () => (compareData && compareData.metrics ? compareData.metrics : []),
    [compareData]
  );

  const visibleMetrics = useMemo(() => {
    if (allMetricNames.length === 0) return [];
    if (activeTab === "all") {
      if (!metricQuery.trim()) return allMetricNames.slice(0, 24);
      try {
        const rx = new RegExp(metricQuery.trim(), "i");
        return allMetricNames.filter((m) => rx.test(m)).slice(0, 32);
      } catch {
        const q = metricQuery.trim().toLowerCase();
        return allMetricNames.filter((m) => m.toLowerCase().includes(q)).slice(0, 32);
      }
    }
    const picked = selectPresetMetrics(allMetricNames, activeTab);
    if (activeTab === "overview" && picked.length === 0) {
      return allMetricNames.slice(0, 12);
    }
    return picked;
  }, [allMetricNames, activeTab, metricQuery]);

  function toggleRunId(rid) {
    if (activeIds.includes(rid)) {
      onSetSelectedIds(activeIds.filter((id) => id !== rid));
    } else {
      onSetSelectedIds([...activeIds, rid]);
    }
  }

  const diffRows = useMemo(() => {
    if (!configDiff || !configDiff.rows) return [];
    return diffOnly ? configDiff.rows.filter((r) => r.differs) : configDiff.rows;
  }, [configDiff, diffOnly]);

  return html`
    <div class="stack">
      <div class="space-between" style="flex-wrap:wrap;">
        <div>
          <h2 style="margin:0; font-size:17px;">Compare Runs (${selectedRuns.length})</h2>
          <div class="muted" style="font-size:12px;">
            Synchronized multi-run curves & hyperparameter diff table
          </div>
        </div>
      </div>

      <div class="card stack" style="gap:8px;">
        <div style="font-size:12px; font-weight:600;" class="muted">
          Selected Runs (tap to toggle)
        </div>
        <div class="row" style="gap:6px;">
          ${runs.slice(0, 20).map((r) => {
            const idx = activeIds.indexOf(r.id);
            const active = idx !== -1;
            const color = active ? SERIES_COLORS[idx % SERIES_COLORS.length] : null;
            return html`
              <button
                key=${r.id}
                class="pill"
                style=${`cursor:pointer; ${
                  active
                    ? `border-color:${color}; color:var(--text); background:${color}22;`
                    : ""
                }`}
                onClick=${() => toggleRunId(r.id)}
              >
                ${active
                  ? html`<span class="dot" style=${`background:${color};`}></span>`
                  : null}
                <span>${r.name}</span>
                ${r.arch ? html`<span class="muted">(${r.arch})</span>` : null}
              </button>
            `;
          })}
        </div>
      </div>

      ${selectedRuns.length === 0
        ? html`<div class="card empty-state">Select at least one run to compare.</div>`
        : html`
            <div class="tabs">
              ${CARDGAME_PRESETS.map(
                (p) => html`
                  <button
                    class=${`tab-btn ${activeTab === p.id ? "active" : ""}`}
                    onClick=${() => setActiveTab(p.id)}
                  >
                    ${p.label}
                  </button>
                `
              )}
              <button
                class=${`tab-btn ${activeTab === "all" ? "active" : ""}`}
                onClick=${() => setActiveTab("all")}
              >
                All Metrics (${allMetricNames.length})
              </button>
            </div>

            <div class="card row" style="padding:10px 12px;">
              ${activeTab === "all"
                ? html`<div style="flex:2; min-width:180px;">
                    <input
                      class="input"
                      placeholder="Filter metrics (regex or substring)…"
                      value=${metricQuery}
                      onInput=${(e) => setMetricQuery(e.target.value)}
                    />
                  </div>`
                : null}
              <label class="row" style="font-size:12px;">
                <span class="muted">EMA</span>
                <input
                  type="range"
                  min="0"
                  max="0.95"
                  step="0.05"
                  value=${smoothing}
                  onInput=${(e) => setSmoothing(parseFloat(e.target.value))}
                />
                <span class="mono" style="width:32px;">${smoothing.toFixed(2)}</span>
              </label>
              <select
                class="select"
                style="width:auto;"
                value=${xAxis}
                onChange=${(e) => setXAxis(e.target.value)}
              >
                <option value="step">X: Step</option>
                <option value="relative_time">X: Elapsed</option>
                <option value="wall_time">X: Wall Time</option>
              </select>
              <button
                class=${`btn sm ${logScale ? "primary" : ""}`}
                onClick=${() => setLogScale(!logScale)}
              >
                Log Y
              </button>
            </div>

            ${visibleMetrics.length === 0
              ? html`<div class="card empty-state">No metrics found for this preset.</div>`
              : html`
                  <div class="chart-grid">
                    ${visibleMetrics.map(
                      (m) => html`
                        <${UPlotChart}
                          key=${m}
                          title=${m}
                          seriesMap=${(compareData && compareData.series && compareData.series[m]) || {}}
                          runList=${selectedRuns}
                          xAxis=${xAxis}
                          smoothing=${smoothing}
                          logScale=${logScale}
                          syncKey="compare-sync"
                          theme=${theme}
                        />
                      `
                    )}
                  </div>
                `}

            <div class="card stack">
              <div class="space-between">
                <strong>Hyperparameter Diff</strong>
                <label class="row" style="font-size:12px; cursor:pointer;">
                  <input
                    type="checkbox"
                    checked=${diffOnly}
                    onChange=${(e) => setDiffOnly(e.target.checked)}
                  />
                  Show differences only
                </label>
              </div>
              ${diffRows.length === 0
                ? html`<div class="muted" style="font-size:12px;">
                    ${diffOnly
                      ? "All compared hyperparameters are identical."
                      : "No config hyperparameters logged."}
                  </div>`
                : html`
                    <div class="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Parameter</th>
                            ${(configDiff.runs || []).map(
                              (r) => html`
                                <th key=${r.id} style="cursor:pointer;" onClick=${() => onOpenRun(r.id)}>
                                  ${r.name}
                                </th>
                              `
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          ${diffRows.map(
                            (row) => html`
                              <tr key=${row.key} class=${row.differs ? "diff-row" : ""}>
                                <td class="mono" style="font-weight:600;">${row.key}</td>
                                ${(configDiff.runs || []).map((r) => {
                                  const present = row.present[r.id];
                                  const val = row.values[r.id];
                                  return html`
                                    <td key=${r.id} class="mono">
                                      ${!present
                                        ? html`<span class="muted">—</span>`
                                        : typeof val === "object"
                                          ? JSON.stringify(val)
                                          : String(val)}
                                    </td>
                                  `;
                                })}
                              </tr>
                            `
                          )}
                        </tbody>
                      </table>
                    </div>
                  `}
            </div>
          `}
    </div>
  `;
}
