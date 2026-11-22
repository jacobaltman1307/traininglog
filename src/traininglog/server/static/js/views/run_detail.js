import {
  apiFetch,
  connectSSE,
  formatNumber,
  formatPct,
  formatStep,
  formatTimeAgo,
  withTokenQuery,
} from "../api.js";
import { CARDGAME_PRESETS, selectPresetMetrics } from "../presets.js";
import { PlasticityPanel } from "../components/plasticity_panel.js";
import { HistogramChart, UPlotChart } from "../uplot_chart.js";
import { html, useEffect, useMemo, useRef, useState } from "../../vendor/ui.mjs";

export function RunDetailView({
  runId,
  theme,
  onBack,
  onCompareRun,
  onRefreshGlobal,
  onOpenReplays,
  onOpenCards,
}) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");
  const [xAxis, setXAxis] = useState("step");
  const [smoothing, setSmoothing] = useState(0);
  const [logScale, setLogScale] = useState(false);
  const [metricQuery, setMetricQuery] = useState("");
  const [seriesByMetric, setSeriesByMetric] = useState({});

  const [consoleLines, setConsoleLines] = useState([]);
  const [streamFilter, setStreamFilter] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const consoleEndRef = useRef(null);
  const liveQueryRef = useRef({ visibleMetrics: [], xAxis: "step", activeTab: "overview" });

  const [selectedHist, setSelectedHist] = useState("");
  const [histSnapshots, setHistSnapshots] = useState([]);
  const [histIdx, setHistIdx] = useState(0);
  const [runCheckpoints, setRunCheckpoints] = useState([]);
  const [plasticityReports, setPlasticityReports] = useState([]);
  const [runReplays, setRunReplays] = useState([]);
  const [runCards, setRunCards] = useState([]);
  const [runDistNames, setRunDistNames] = useState([]);

  const [showControlModal, setShowControlModal] = useState(false);
  const [notesDraft, setNotesDraft] = useState("");
  const [tagsDraft, setTagsDraft] = useState("");
  const [savingMeta, setSavingMeta] = useState(false);

  async function loadDetail() {
    try {
      const res = await apiFetch(`/api/runs/${runId}`);
      setDetail(res);
      setNotesDraft(res.run.notes || "");
      setTagsDraft((res.run.tags || []).join(", "));
      if (!selectedHist && res.histograms && res.histograms.length > 0) {
        setSelectedHist(res.histograms[0].name);
      }
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    setDetail(null);
    setSeriesByMetric({});
    setConsoleLines([]);
    loadDetail();
  }, [runId]);

  useEffect(() => {
    const close = connectSSE(
      (ev) => {
        if (ev.type === "run_metrics" && ev.data && ev.data.run) {
          setDetail((prev) =>
            prev
              ? { ...prev, run: ev.data.run }
              : prev
          );
          const { visibleMetrics: vm, xAxis: xa, activeTab: tab } = liveQueryRef.current;
          if (
            tab !== "console" &&
            tab !== "config" &&
            tab !== "histograms" &&
            tab !== "plasticity" &&
            tab !== "replays_cards" &&
            vm.length > 0
          ) {
            fetchSeries(vm, xa);
          }
        } else if (ev.type === "run_updated" && ev.data) {
          setDetail((prev) => (prev ? { ...prev, run: ev.data } : prev));
        } else if (ev.type === "console_batch" && ev.data && ev.data.lines) {
          setConsoleLines((prev) => [...prev.slice(-1500), ...ev.data.lines]);
        } else if (ev.type === "command_updated" || ev.type === "anomaly") {
          loadDetail();
        }
      },
      { runId }
    );
    return close;
  }, [runId]);

  const allMetricNames = useMemo(
    () => (detail && detail.metrics ? detail.metrics.map((m) => m.name) : []),
    [detail]
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
    const presetPicked = selectPresetMetrics(allMetricNames, activeTab);
    if (activeTab === "overview" && presetPicked.length === 0) {
      return allMetricNames.slice(0, 12);
    }
    return presetPicked;
  }, [allMetricNames, activeTab, metricQuery]);

  async function fetchSeries(metrics, axisOverride = null) {
    if (!metrics || metrics.length === 0) return;
    const axis = axisOverride || xAxis;
    try {
      const qs = metrics.join(",");
      const res = await apiFetch(
        `/api/runs/${runId}/series?metrics=${encodeURIComponent(qs)}&max_points=800&x_axis=${axis}`
      );
      setSeriesByMetric((prev) => ({ ...prev, ...(res.series || {}) }));
    } catch {
      // ignore transient poll errors
    }
  }

  useEffect(() => {
    liveQueryRef.current = { visibleMetrics, xAxis, activeTab };
    if (
      activeTab !== "console" &&
      activeTab !== "config" &&
      activeTab !== "histograms" &&
      activeTab !== "plasticity" &&
      activeTab !== "replays_cards" &&
      visibleMetrics.length > 0
    ) {
      fetchSeries(visibleMetrics, xAxis);
    }
    if (activeTab === "plasticity") {
      Promise.all([
        apiFetch(`/api/checkpoints?run_id=${encodeURIComponent(runId)}`),
        apiFetch(`/api/plasticity?run_id=${encodeURIComponent(runId)}`),
      ])
        .then(([ckRes, plRes]) => {
          setRunCheckpoints(ckRes.checkpoints || []);
          setPlasticityReports(plRes.reports || []);
        })
        .catch(() => {});
    }
    if (activeTab === "replays_cards") {
      Promise.all([
        apiFetch(`/api/replays?run_id=${encodeURIComponent(runId)}`).catch(() => ({ replays: [] })),
        apiFetch(`/api/cards/meta?run_id=${encodeURIComponent(runId)}`).catch(() => ({ cards: [] })),
        apiFetch(`/api/distributions?run_id=${encodeURIComponent(runId)}`).catch(() => ({ names: [] })),
      ])
        .then(([rpRes, cdRes, dtRes]) => {
          setRunReplays(rpRes.replays || []);
          setRunCards(cdRes.cards || []);
          setRunDistNames(dtRes.names || []);
        })
        .catch(() => {});
    }
  }, [runId, activeTab, xAxis, visibleMetrics.join("|")]);

  useEffect(() => {
    if (activeTab === "console") {
      const qs = streamFilter ? `&stream=${encodeURIComponent(streamFilter)}` : "";
      apiFetch(`/api/runs/${runId}/console?limit=500${qs}`)
        .then((res) => setConsoleLines(res.lines || []))
        .catch(() => {});
    }
  }, [runId, activeTab, streamFilter]);

  useEffect(() => {
    if (activeTab === "console" && autoScroll && consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [consoleLines.length, activeTab, autoScroll]);

  useEffect(() => {
    if (activeTab === "histograms" && selectedHist) {
      apiFetch(
        `/api/runs/${runId}/histograms?name=${encodeURIComponent(selectedHist)}&limit=50`
      )
        .then((res) => {
          const snaps = res.snapshots || [];
          setHistSnapshots(snaps);
          setHistIdx(Math.max(0, snaps.length - 1));
        })
        .catch(() => {});
    }
  }, [runId, activeTab, selectedHist]);

  async function sendCommand(cmdType) {
    try {
      await apiFetch(`/api/runs/${runId}/commands`, {
        method: "POST",
        body: JSON.stringify({ cmd_type: cmdType, payload: {} }),
      });
      await loadDetail();
    } catch (err) {
      alert(`Command failed: ${err.message}`);
    }
  }

  async function saveNotesAndTags() {
    setSavingMeta(true);
    try {
      const tags = tagsDraft
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      await apiFetch(`/api/runs/${runId}`, {
        method: "PATCH",
        body: JSON.stringify({ notes: notesDraft, tags }),
      });
      await loadDetail();
      if (onRefreshGlobal) onRefreshGlobal();
    } catch (err) {
      alert(`Save failed: ${err.message}`);
    } finally {
      setSavingMeta(false);
    }
  }

  if (error) {
    return html`<div class="card">
      <button class="btn sm" onClick=${onBack}>← Back</button>
      <p style="color:var(--danger);">${error}</p>
    </div>`;
  }

  if (!detail) {
    return html`<div class="card empty-state">Loading run details…</div>`;
  }

  const run = detail.run;
  const sum = run.summary || {};
  const gpu = run.last_gpu || {};
  const histNames = detail.histograms || [];
  const availablePresets = CARDGAME_PRESETS.filter(
    (p) => p.id === "overview" || selectPresetMetrics(allMetricNames, p.id).length > 0
  );

  return html`
    <div class="stack">
      <div class="card stack">
        <div class="space-between" style="flex-wrap:wrap;">
          <div class="row">
            <button class="btn sm" onClick=${onBack}>← Runs</button>
            <div>
              <h2 style="margin:0; font-size:17px;">${run.name}</h2>
              <div class="row" style="gap:6px; margin-top:4px;">
                <span class=${`pill ${run.state}`}>${run.state}</span>
                ${run.arch ? html`<span class="pill accent">${run.arch}</span>` : null}
                <span class="pill">${run.project_name}</span>
                ${run.git && run.git.short_commit
                  ? html`<span class="pill mono">
                      ${run.git.branch}@${run.git.short_commit}${run.git.dirty ? "*" : ""}
                    </span>`
                  : null}
              </div>
            </div>
          </div>

          <div class="row">
            <button class="btn sm" onClick=${() => onCompareRun(run.id)}>
              Compare
            </button>
            <a
              class="btn sm"
              href=${withTokenQuery(`/api/runs/${run.id}/export?format=csv`)}
              download
            >
              CSV
            </a>
            <a
              class="btn sm"
              href=${withTokenQuery(`/api/runs/${run.id}/export?format=json`)}
              download
            >
              JSON
            </a>
            <button class="btn sm primary" onClick=${() => setShowControlModal(true)}>
              Controls
            </button>
          </div>
        </div>

        <div class="stat-row">
          <div class="stat-box">
            <div class="stat-label">Step</div>
            <div class="stat-val mono">${formatStep(run.current_step)}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">SPS / Win%</div>
            <div class="stat-val mono">
              ${formatNumber(run.current_sps)} · ${sum["game/win_rate_overall"] != null ? formatPct(sum["game/win_rate_overall"]) : "—"}
            </div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Hardware</div>
            <div class="stat-val mono" style="font-size:12px;">
              ${gpu.gpu_util != null
                ? `${Math.round(gpu.gpu_util)}% GPU · ${Math.round(gpu.gpu_temp_c || 0)}°C`
                : sum["sys/cpu_pct"] != null
                  ? `${Math.round(sum["sys/cpu_pct"])}% CPU · ${Math.round(sum["sys/ram_pct"] || 0)}% RAM`
                  : formatTimeAgo(run.last_heartbeat_at)}
            </div>
          </div>
        </div>

        ${detail.open_anomalies && detail.open_anomalies.length > 0
          ? html`
              <div class="stack" style="gap:6px;">
                ${detail.open_anomalies.map(
                  (a) => html`
                    <div
                      key=${a.id}
                      style="padding:8px 10px; border-radius:8px; background:var(--danger-dim); border:1px solid rgba(239,68,68,0.35); font-size:12px;"
                    >
                      <strong>${a.kind}:</strong> ${a.message}
                    </div>
                  `
                )}
              </div>
            `
          : null}
      </div>

      <div class="tabs">
        ${availablePresets.map(
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
        <button
          class=${`tab-btn ${activeTab === "histograms" ? "active" : ""}`}
          onClick=${() => setActiveTab("histograms")}
        >
          Histograms (${histNames.length})
        </button>
        <button
          class=${`tab-btn ${activeTab === "plasticity" ? "active" : ""}`}
          onClick=${() => setActiveTab("plasticity")}
        >
          Plasticity & Checkpoints
        </button>
        <button
          class=${`tab-btn ${activeTab === "replays_cards" ? "active" : ""}`}
          onClick=${() => setActiveTab("replays_cards")}
        >
          Replays & Cards
        </button>
        <button
          class=${`tab-btn ${activeTab === "console" ? "active" : ""}`}
          onClick=${() => setActiveTab("console")}
        >
          Console
        </button>
        <button
          class=${`tab-btn ${activeTab === "config" ? "active" : ""}`}
          onClick=${() => setActiveTab("config")}
        >
          Config & Git
        </button>
      </div>

      ${activeTab !== "console" && activeTab !== "config" && activeTab !== "histograms" && activeTab !== "plasticity" && activeTab !== "replays_cards"
        ? html`
            <div class="card row" style="padding:10px 12px;">
              ${activeTab === "all"
                ? html`<div style="flex:2; min-width:180px;">
                    <input
                      class="input"
                      placeholder="Filter metrics (regex or text)…"
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
              ? html`<div class="card empty-state">No metrics recorded yet for this view.</div>`
              : html`
                  <div class="chart-grid">
                    ${visibleMetrics.map(
                      (m) => html`
                        <${UPlotChart}
                          key=${m}
                          title=${m}
                          seriesMap=${{ [run.id]: seriesByMetric[m] }}
                          runList=${[run]}
                          xAxis=${xAxis}
                          smoothing=${smoothing}
                          logScale=${logScale}
                          syncKey=${`run-${run.id}`}
                          theme=${theme}
                        />
                      `
                    )}
                  </div>
                `}
          `
        : null}

      ${activeTab === "histograms"
        ? html`
            <div class="card stack">
              ${histNames.length === 0
                ? html`<div class="empty-state">No histograms logged for this run.</div>`
                : html`
                    <div class="row">
                      <select
                        class="select"
                        style="max-width:340px;"
                        value=${selectedHist}
                        onChange=${(e) => setSelectedHist(e.target.value)}
                      >
                        ${histNames.map(
                          (h) => html`<option value=${h.name}>${h.name} (${h.snapshots})</option>`
                        )}
                      </select>
                      ${histSnapshots.length > 0
                        ? html`<span class="mono muted" style="font-size:12px;">
                            Step ${formatStep(histSnapshots[histIdx]?.step)} · mean ${formatNumber(histSnapshots[histIdx]?.mean_val)} · std ${formatNumber(histSnapshots[histIdx]?.std_val)} · n=${histSnapshots[histIdx]?.total_count}
                          </span>`
                        : null}
                    </div>
                    ${histSnapshots.length > 1
                      ? html`<input
                          type="range"
                          min="0"
                          max=${histSnapshots.length - 1}
                          value=${histIdx}
                          onInput=${(e) => setHistIdx(parseInt(e.target.value, 10))}
                        />`
                      : null}
                    ${histSnapshots[histIdx]
                      ? html`<${HistogramChart} snapshot=${histSnapshots[histIdx]} />`
                      : null}
                  `}
            </div>
          `
        : null}

      ${activeTab === "console"
        ? html`
            <div class="stack">
              <div class="row">
                <select
                  class="select"
                  style="width:auto;"
                  value=${streamFilter}
                  onChange=${(e) => setStreamFilter(e.target.value)}
                >
                  <option value="">All Streams</option>
                  <option value="stdout">stdout</option>
                  <option value="stderr">stderr</option>
                </select>
                <label class="row" style="font-size:12px;">
                  <input
                    type="checkbox"
                    checked=${autoScroll}
                    onChange=${(e) => setAutoScroll(e.target.checked)}
                  />
                  Auto-scroll
                </label>
              </div>
              <div class="console-box">
                ${consoleLines.length === 0
                  ? html`<span class="muted">No console output captured yet.</span>`
                  : consoleLines.map(
                      (line, idx) => html`
                        <div key=${line.id || idx} class=${`console-line ${line.stream}`}>
                          ${line.text}
                        </div>
                      `
                    )}
                <div ref=${consoleEndRef}></div>
              </div>
            </div>
          `
        : null}

      ${activeTab === "config"
        ? html`
            <div class="stack">
              <div class="card stack">
                <strong>Notes & Tags</strong>
                <input
                  class="input"
                  placeholder="Tags (comma separated)"
                  value=${tagsDraft}
                  onInput=${(e) => setTagsDraft(e.target.value)}
                />
                <textarea
                  class="textarea"
                  rows="3"
                  placeholder="Experiment notes…"
                  value=${notesDraft}
                  onInput=${(e) => setNotesDraft(e.target.value)}
                ></textarea>
                <div>
                  <button class="btn primary sm" disabled=${savingMeta} onClick=${saveNotesAndTags}>
                    ${savingMeta ? "Saving…" : "Save Metadata"}
                  </button>
                </div>
              </div>

              <div class="card stack">
                <strong>Hyperparameters</strong>
                <div class="table-wrap">
                  <table>
                    <thead><tr><th>Key</th><th>Value</th></tr></thead>
                    <tbody>
                      ${Object.entries(run.config || {})
                        .sort(([a], [b]) => a.localeCompare(b))
                        .map(
                          ([k, v]) => html`
                            <tr key=${k}>
                              <td class="mono">${k}</td>
                              <td class="mono">${typeof v === "object" ? JSON.stringify(v) : String(v)}</td>
                            </tr>
                          `
                        )}
                    </tbody>
                  </table>
                </div>
              </div>

              <div class="card stack">
                <strong>Provenance & Git</strong>
                <div class="mono" style="font-size:12px;">
                  Host: ${run.host || "—"} · CLI: ${run.cli_args || "—"}
                </div>
                ${run.git && run.git.diff
                  ? html`<pre class="console-box" style="max-height:260px;">${run.git.diff}</pre>`
                  : html`<div class="muted" style="font-size:12px;">No uncommitted git diff recorded.</div>`}
              </div>
            </div>
          `
        : null}

      ${activeTab === "plasticity"
        ? html`
            <div class="stack">
              <div class="card stack">
                <strong>Run Checkpoints (${runCheckpoints.length})</strong>
                ${runCheckpoints.length === 0
                  ? html`<div class="muted" style="font-size:12px;">No checkpoints logged for this run yet.</div>`
                  : html`
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr><th>Name</th><th>Step</th><th>Glicko-2</th><th>Storage</th><th>Tags</th></tr>
                          </thead>
                          <tbody>
                            ${runCheckpoints.map(
                              (c) => html`
                                <tr key=${c.id}>
                                  <td style="font-weight:600;">${c.name}</td>
                                  <td class="mono">${formatStep(c.step)}</td>
                                  <td class="mono">${c.glicko_mu != null ? Number(c.glicko_mu).toFixed(1) : "—"}</td>
                                  <td class="mono">${c.storage_mode}</td>
                                  <td>${(c.tags || []).map((t) => html`<span class="pill" key=${t}>#${t}</span>`)}</td>
                                </tr>
                              `
                            )}
                          </tbody>
                        </table>
                      </div>
                    `}
              </div>
              <${PlasticityPanel} reports=${plasticityReports} theme=${theme} />
            </div>
          `
        : null}

      ${activeTab === "replays_cards"
        ? html`
            <div class="stack">
              <div class="card stack">
                <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                  <strong>Run Replays (${runReplays.length})</strong>
                  ${onOpenReplays
                    ? html`<button class="btn sm primary" onClick=${() => onOpenReplays(run.id)}>
                        Open in Replay Hub →
                      </button>`
                    : null}
                </div>
                ${runReplays.length === 0
                  ? html`<div class="muted" style="font-size:12px;">No replays recorded for this run yet.</div>`
                  : html`
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr><th>Title</th><th>Format</th><th>Step</th><th>Swing</th><th>Download</th></tr>
                          </thead>
                          <tbody>
                            ${runReplays.map(
                              (rp) => html`
                                <tr key=${rp.id}>
                                  <td style="font-weight:600;">${rp.title}</td>
                                  <td><span class="pill accent">${rp.format}</span></td>
                                  <td class="mono">${rp.step != null ? formatStep(rp.step) : "—"}</td>
                                  <td class="mono">
                                    ${rp.has_rl_trace
                                      ? `${Number(rp.max_value_swing || 0).toFixed(2)}${rp.blunder_turn != null ? ` @T${rp.blunder_turn}` : ""}`
                                      : "—"}
                                  </td>
                                  <td>
                                    <a
                                      class="btn sm"
                                      href=${withTokenQuery(`/api/replays/${rp.id}/file`)}
                                      download=${rp.filename || `${rp.id}.json`}
                                    >
                                      DL
                                    </a>
                                  </td>
                                </tr>
                              `
                            )}
                          </tbody>
                        </table>
                      </div>
                    `}
              </div>

              <div class="card stack">
                <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                  <strong>Run Card Meta (${runCards.length})</strong>
                  ${onOpenCards
                    ? html`<button class="btn sm" onClick=${() => onOpenCards(run.id)}>
                        Open Card Meta & Policy →
                      </button>`
                    : null}
                </div>
                ${runCards.length === 0
                  ? html`<div class="muted" style="font-size:12px;">No card stats logged for this run yet.</div>`
                  : html`
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr><th>Card</th><th>Played</th><th>Drawn</th><th>Draw→Play</th><th>Win Rate</th></tr>
                          </thead>
                          <tbody>
                            ${runCards.slice(0, 15).map(
                              (c) => html`
                                <tr key=${c.card_name}>
                                  <td style="font-weight:600;">${c.card_name}</td>
                                  <td class="mono">${c.played}</td>
                                  <td class="mono">${c.drawn}</td>
                                  <td class="mono">${formatPct(c.play_rate)}</td>
                                  <td class="mono">${formatPct(c.win_rate)}</td>
                                </tr>
                              `
                            )}
                          </tbody>
                        </table>
                      </div>
                    `}
              </div>

              ${runDistNames.length > 0
                ? html`
                    <div class="card stack">
                      <strong>Action / Policy Distributions (${runDistNames.length})</strong>
                      <div class="row">
                        ${runDistNames.map((dn) => html`<span class="pill accent" key=${dn}>${dn}</span>`)}
                      </div>
                    </div>
                  `
                : null}
            </div>
          `
        : null}

      ${showControlModal
        ? html`
            <div class="modal-backdrop" onClick=${() => setShowControlModal(false)}>
              <div class="modal-card stack" onClick=${(e) => e.stopPropagation()}>
                <div class="space-between">
                  <strong>Cooperative Run Controls</strong>
                  <button class="btn sm" onClick=${() => setShowControlModal(false)}>Close</button>
                </div>
                <div class="stack">
                  <button class="btn" onClick=${() => sendCommand("checkpoint")}>
                    Save Checkpoint Now
                  </button>
                  <button class="btn" onClick=${() => sendCommand("eval")}>
                    Run Eval Match Now
                  </button>
                  <button class="btn warn" onClick=${() => sendCommand("run_redo")}>
                    Run ReDo Plasticity Check
                  </button>
                  <button
                    class="btn danger"
                    onClick=${() => {
                      if (window.confirm(`Gracefully stop '${run.name}'?`)) {
                        sendCommand("stop");
                      }
                    }}
                  >
                    Graceful Stop
                  </button>
                </div>

                <div>
                  <strong style="font-size:12px;">Command History</strong>
                  <div class="stack" style="margin-top:6px; gap:6px;">
                    ${(detail.commands || []).length === 0
                      ? html`<div class="muted" style="font-size:12px;">No commands sent yet.</div>`
                      : detail.commands.map(
                          (c) => html`
                            <div class="space-between" key=${c.id} style="font-size:12px; padding:6px 8px; background:var(--bg); border-radius:8px;">
                              <span class="mono">${c.cmd_type}</span>
                              <span class=${`pill ${c.status === "completed" ? "running" : c.status === "pending" ? "warning" : "accent"}`}>
                                ${c.status}
                              </span>
                            </div>
                          `
                        )}
                  </div>
                </div>
              </div>
            </div>
          `
        : null}
    </div>
  `;
}
