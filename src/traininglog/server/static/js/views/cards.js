import {
  apiFetch,
  connectSSE,
  formatPct,
  formatStep,
  withTokenQuery,
} from "../api.js";
import { SERIES_COLORS, UPlotChart } from "../uplot_chart.js";
import { html, useEffect, useMemo, useState } from "../../vendor/ui.mjs";

function winRateColor(wr) {
  if (wr === null || wr === undefined) return "color: var(--muted);";
  if (wr >= 0.55) return "color: var(--ok); font-weight: 700;";
  if (wr <= 0.45) return "color: var(--danger); font-weight: 700;";
  return "";
}

export function CardsView({
  projects,
  runs,
  theme,
  initialRunId = "",
  onOpenReplays,
  onOpenRun,
}) {
  const [project, setProject] = useState("");
  const [runId, setRunId] = useState(initialRunId || "");
  const [minPlayed, setMinPlayed] = useState("0");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("played");
  const [sortDir, setSortDir] = useState("desc");

  const [cardsMeta, setCardsMeta] = useState([]);
  const [trendsMap, setTrendsMap] = useState({});
  const [selectedCards, setSelectedCards] = useState([]);
  const [trendMetric, setTrendMetric] = useState("win_rate");

  const [distRunId, setDistRunId] = useState(initialRunId || "");
  const [distNames, setDistNames] = useState([]);
  const [selectedDistName, setSelectedDistName] = useState("");
  const [distSeries, setDistSeries] = useState([]);

  useEffect(() => {
    if (initialRunId) {
      setRunId(initialRunId);
      setDistRunId(initialRunId);
    }
  }, [initialRunId]);

  async function loadCards() {
    try {
      const qs = new URLSearchParams();
      if (project) qs.set("project", project);
      if (runId) qs.set("run_id", runId);
      if (minPlayed && Number(minPlayed) > 0) qs.set("min_played", minPlayed);

      const metaRes = await apiFetch(`/api/cards/meta?${qs.toString()}`);
      const list = metaRes.cards || [];
      setCardsMeta(list);

      const tqs = new URLSearchParams();
      if (project) tqs.set("project", project);
      if (runId) tqs.set("run_id", runId);
      if (selectedCards.length > 0) {
        tqs.set("cards", selectedCards.join(","));
      }
      const trendsRes = await apiFetch(`/api/cards/trends?${tqs.toString()}`);
      setTrendsMap(trendsRes.trends || {});
    } catch (err) {
      console.error("Failed loading card meta:", err);
    }
  }

  async function loadDistributions(targetRun, targetName) {
    const rid = targetRun || runId || ((runs && runs[0]) ? runs[0].id : "");
    if (!rid) {
      setDistNames([]);
      setDistSeries([]);
      return;
    }
    if (rid !== distRunId) setDistRunId(rid);
    try {
      const qs = new URLSearchParams({ run_id: rid });
      if (targetName) qs.set("name", targetName);
      const res = await apiFetch(`/api/distributions?${qs.toString()}`);
      setDistNames(res.names || []);
      setSelectedDistName(res.name || "");
      setDistSeries(res.series || []);
    } catch {
      setDistNames([]);
      setDistSeries([]);
    }
  }

  useEffect(() => {
    loadCards();
  }, [project, runId, minPlayed, selectedCards.join(",")]);

  useEffect(() => {
    const target = distRunId || runId || ((runs && runs[0]) ? runs[0].id : "");
    if (target) {
      loadDistributions(target, selectedDistName);
    }
  }, [runId, distRunId, selectedDistName, (runs || []).length]);

  useEffect(() => {
    const close = connectSSE((ev) => {
      if (ev.type === "cards_updated") {
        loadCards();
        if (distRunId) loadDistributions(distRunId, selectedDistName);
      }
    });
    return close;
  }, [project, runId, distRunId, selectedDistName]);

  function toggleSort(col) {
    if (sortBy === col) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortBy(col);
      setSortDir("desc");
    }
  }

  function toggleCardSelection(cardName) {
    setSelectedCards((prev) =>
      prev.includes(cardName)
        ? prev.filter((c) => c !== cardName)
        : [...prev, cardName].slice(0, 8)
    );
  }

  const maxPlayedAll = useMemo(
    () => Math.max(1, ...cardsMeta.map((c) => c.played || 0)),
    [cardsMeta]
  );

  const sortedCards = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = q
      ? cardsMeta.filter((c) => c.card_name.toLowerCase().includes(q))
      : [...cardsMeta];
    filtered.sort((a, b) => {
      let av = a[sortBy];
      let bv = b[sortBy];
      if (sortBy === "card_name") {
        return sortDir === "asc"
          ? String(av).localeCompare(String(bv))
          : String(bv).localeCompare(String(av));
      }
      av = av == null ? -1e9 : Number(av);
      bv = bv == null ? -1e9 : Number(bv);
      return sortDir === "asc" ? av - bv : bv - av;
    });
    return filtered;
  }, [cardsMeta, search, sortBy, sortDir]);

  const cardTrendChart = useMemo(() => {
    const entries = Object.entries(trendsMap || {});
    const runList = [];
    const seriesMap = {};
    for (const [cardName, pts] of entries) {
      runList.push({ id: cardName, name: cardName });
      seriesMap[cardName] = {
        x: pts.map((p) => Number(p.step)),
        y: pts.map((p) => {
          const v = p[trendMetric];
          return v == null || !Number.isFinite(Number(v)) ? null : Number(v);
        }),
      };
    }
    return { runList, seriesMap };
  }, [trendsMap, trendMetric]);

  const distProbChart = useMemo(() => {
    if (!distSeries || distSeries.length === 0) return { runList: [], seriesMap: {} };
    const labels = distSeries[0].labels || [];
    const xs = distSeries.map((pt) => Number(pt.step));
    const runList = [];
    const seriesMap = {};
    labels.forEach((label, idx) => {
      const id = `cls_${idx}`;
      runList.push({ id, name: label });
      seriesMap[id] = {
        x: xs,
        y: distSeries.map((pt) => {
          const p = (pt.probs || [])[idx];
          return Number.isFinite(p) ? p : null;
        }),
      };
    });
    return { runList, seriesMap };
  }, [distSeries]);

  const distEntropyChart = useMemo(() => {
    if (!distSeries || distSeries.length === 0) return { runList: [], seriesMap: {} };
    const xs = [];
    const ys = [];
    for (const pt of distSeries) {
      if (Number.isFinite(pt.entropy)) {
        xs.push(Number(pt.step));
        ys.push(Number(pt.entropy));
      }
    }
    if (xs.length === 0) return { runList: [], seriesMap: {} };
    return {
      runList: [{ id: "entropy", name: "Entropy" }],
      seriesMap: { entropy: { x: xs, y: ys } },
    };
  }, [distSeries]);

  const latestDistPoint = distSeries.length > 0 ? distSeries[distSeries.length - 1] : null;

  const exportQuery = useMemo(() => {
    const qs = new URLSearchParams();
    if (project) qs.set("project", project);
    if (runId) qs.set("run_id", runId);
    return qs.toString();
  }, [project, runId]);

  return html`
    <div class="stack">
      <div class="space-between" style="flex-wrap:wrap; gap:8px;">
        <div>
          <h2 style="margin:0; font-size:17px;">Card Meta & Policy Distributions (${sortedCards.length})</h2>
          <div class="muted" style="font-size:12px;">
            Per-card play/draw/win rates across rollouts and action-type probability evolution
          </div>
        </div>
        <div class="row">
          <a
            class="btn sm"
            href=${withTokenQuery(`/api/cards/export?format=csv&${exportQuery}`)}
            download="card_stats.csv"
          >
            Export CSV
          </a>
          <a
            class="btn sm"
            href=${withTokenQuery(`/api/cards/export?format=json&${exportQuery}`)}
            download="card_stats.json"
          >
            Export JSON
          </a>
          ${onOpenReplays
            ? html`<button class="btn sm" onClick=${onOpenReplays}>Replays →</button>`
            : null}
        </div>
      </div>

      <div class="card">
        <div class="row">
          <select class="select" value=${project} onChange=${(e) => setProject(e.target.value)}>
            <option value="">All Projects</option>
            ${(projects || []).map((p) => html`<option value=${p.name}>${p.name}</option>`)}
          </select>
          <select
            class="select"
            value=${runId}
            onChange=${(e) => {
              setRunId(e.target.value);
              if (e.target.value) setDistRunId(e.target.value);
            }}
          >
            <option value="">All Runs (Aggregated)</option>
            ${(runs || []).map((r) => html`<option value=${r.id}>${r.name}</option>`)}
          </select>
          <select class="select" value=${minPlayed} onChange=${(e) => setMinPlayed(e.target.value)}>
            <option value="0">Min Played: All</option>
            <option value="10">Min Played: 10+</option>
            <option value="50">Min Played: 50+</option>
            <option value="200">Min Played: 200+</option>
          </select>
          <div style="flex:1; min-width:150px;">
            <input
              class="input"
              placeholder="Filter card name…"
              value=${search}
              onInput=${(e) => setSearch(e.target.value)}
            />
          </div>
          ${selectedCards.length > 0
            ? html`<button class="btn sm" onClick=${() => setSelectedCards([])}>
                Clear Chart Picks (${selectedCards.length})
              </button>`
            : null}
        </div>
      </div>

      <div class="card stack">
        <div class="space-between" style="flex-wrap:wrap; gap:8px;">
          <div>
            <strong>Card Play & Win Rate Meta Table</strong>
            <div class="muted" style="font-size:11.5px;">Click a card row to isolate it in the trend chart below</div>
          </div>
          ${runId && onOpenRun
            ? html`<button class="btn sm" onClick=${() => onOpenRun(runId)}>Open Run Detail</button>`
            : null}
        </div>
        ${sortedCards.length === 0
          ? html`<div class="muted" style="font-size:12.5px;">No card statistics logged yet. Call <code>run.log_card_stats(...)</code>.</div>`
          : html`
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th style="cursor:pointer;" onClick=${() => toggleSort("card_name")}>Card</th>
                      <th style="cursor:pointer;" onClick=${() => toggleSort("played")}>Played</th>
                      <th style="cursor:pointer;" onClick=${() => toggleSort("drawn")}>Drawn</th>
                      <th style="cursor:pointer;" onClick=${() => toggleSort("play_share")}>Play Share</th>
                      <th style="cursor:pointer;" onClick=${() => toggleSort("play_rate")}>Draw→Play</th>
                      <th style="cursor:pointer;" onClick=${() => toggleSort("win_rate")}>Win Rate</th>
                      <th style="cursor:pointer;" onClick=${() => toggleSort("delta_win_rate")}>Δ Recent</th>
                      <th>Latest Step</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${sortedCards.map((c) => {
                      const picked = selectedCards.includes(c.card_name);
                      const delta = Number(c.delta_win_rate || 0);
                      return html`
                        <tr
                          key=${c.card_name}
                          style=${`cursor:pointer; ${picked ? "background: var(--accent-dim);" : ""}`}
                          onClick=${() => toggleCardSelection(c.card_name)}
                        >
                          <td style="font-weight:600;">
                            ${picked ? "◆ " : ""}${c.card_name}
                          </td>
                          <td>
                            <div class="bar-cell">
                              <span class="mono" style="min-width:44px;">${c.played}</span>
                              <div class="mini-bar-track">
                                <div
                                  class="mini-bar-fill"
                                  style=${`width:${Math.max(3, (c.played / maxPlayedAll) * 100)}%; background: var(--accent);`}
                                ></div>
                              </div>
                            </div>
                          </td>
                          <td class="mono">${c.drawn}</td>
                          <td class="mono">${formatPct(c.play_share)}</td>
                          <td class="mono">${formatPct(c.play_rate)}</td>
                          <td class="mono" style=${winRateColor(c.win_rate)}>
                            ${formatPct(c.win_rate)}
                          </td>
                          <td>
                            ${delta !== 0
                              ? html`<span class=${`pill ${delta > 0 ? "ok" : "danger"}`}>
                                  ${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)}%
                                </span>`
                              : html`<span class="muted">—</span>`}
                          </td>
                          <td class="mono muted">${formatStep(c.latest_step)}</td>
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              </div>
            `}
      </div>

      ${cardTrendChart.runList.length > 0
        ? html`
            <div class="card stack">
              <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                <strong>
                  Card Metric Evolution (${selectedCards.length > 0 ? `${selectedCards.length} selected` : `Top ${cardTrendChart.runList.length}`})
                </strong>
                <div class="tabs" style="margin-bottom:0;">
                  ${[
                    ["win_rate", "Win Rate"],
                    ["play_rate", "Play Rate"],
                    ["played", "Played Count"],
                  ].map(
                    ([k, label]) => html`
                      <button
                        class=${`tab-btn ${trendMetric === k ? "active" : ""}`}
                        onClick=${() => setTrendMetric(k)}
                      >
                        ${label}
                      </button>
                    `
                  )}
                </div>
              </div>
              <${UPlotChart}
                title=${`Card ${trendMetric.replace("_", " ")} vs Step`}
                seriesMap=${cardTrendChart.seriesMap}
                runList=${cardTrendChart.runList}
                xAxis="step"
                smoothing=${0}
                theme=${theme}
                height=${220}
              />
            </div>
          `
        : null}

      <div class="card stack">
        <div class="space-between" style="flex-wrap:wrap; gap:8px;">
          <div>
            <strong>Action / Strategy Distribution Evolution</strong>
            <div class="muted" style="font-size:11.5px;">Discrete policy probabilities over training steps</div>
          </div>
          <div class="row">
            <select
              class="select"
              value=${distRunId}
              onChange=${(e) => {
                setDistRunId(e.target.value);
                setSelectedDistName("");
              }}
            >
              ${(runs || []).map((r) => html`<option value=${r.id}>${r.name}</option>`)}
            </select>
            ${distNames.length > 0
              ? html`
                  <select
                    class="select"
                    value=${selectedDistName}
                    onChange=${(e) => setSelectedDistName(e.target.value)}
                  >
                    ${distNames.map((n) => html`<option value=${n}>${n}</option>`)}
                  </select>
                `
              : null}
          </div>
        </div>

        ${distProbChart.runList.length === 0
          ? html`<div class="muted" style="font-size:12.5px;">No discrete distributions logged for this run yet. Call <code>run.log_distribution(...)</code>.</div>`
          : html`
              <div class="grid-2">
                <${UPlotChart}
                  title=${`${selectedDistName || "Distribution"} Class Probabilities`}
                  seriesMap=${distProbChart.seriesMap}
                  runList=${distProbChart.runList}
                  xAxis="step"
                  smoothing=${0}
                  theme=${theme}
                  height=${210}
                />
                ${distEntropyChart.runList.length > 0
                  ? html`
                      <${UPlotChart}
                        title=${`${selectedDistName || "Distribution"} Shannon Entropy`}
                        seriesMap=${distEntropyChart.seriesMap}
                        runList=${distEntropyChart.runList}
                        xAxis="step"
                        smoothing=${0}
                        theme=${theme}
                        height=${210}
                      />
                    `
                  : html`
                      <div class="card stack" style="background:var(--surface-2);">
                        <strong style="font-size:13px;">Latest Snapshot</strong>
                        <div class="muted" style="font-size:12px;">No entropy scalar present.</div>
                      </div>
                    `}
              </div>

              ${latestDistPoint
                ? html`
                    <div class="stack" style="gap:6px;">
                      <div class="space-between">
                        <strong style="font-size:12.5px;">Latest Distribution Breakdown</strong>
                        <span class="mono muted" style="font-size:12px;">
                          Step ${formatStep(latestDistPoint.step)}
                        </span>
                      </div>
                      <div class="prob-list">
                        ${(latestDistPoint.labels || []).map((lbl, idx) => {
                          const p = Number((latestDistPoint.probs || [])[idx] || 0);
                          const color = SERIES_COLORS[idx % SERIES_COLORS.length];
                          return html`
                            <div class="prob-row" key=${lbl}>
                              <span class="mono">${lbl}</span>
                              <div class="prob-bar-track">
                                <div
                                  class="prob-bar-fill"
                                  style=${`width:${Math.max(1.5, Math.min(100, p * 100))}%; background:${color};`}
                                ></div>
                              </div>
                              <span class="mono" style="text-align:right;">
                                ${(p * 100).toFixed(1)}%
                              </span>
                            </div>
                          `;
                        })}
                      </div>
                    </div>
                  `
                : null}
            `}
      </div>
    </div>
  `;
}
