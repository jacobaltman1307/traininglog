import {
  apiFetch,
  connectSSE,
  formatStep,
  formatTimeAgo,
  withTokenQuery,
} from "../api.js";
import { UPlotChart } from "../uplot_chart.js";
import { html, useEffect, useMemo, useState } from "../../vendor/ui.mjs";

function formatBytes(n) {
  if (!n && n !== 0) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function normalizeTraceTurns(traceObj) {
  if (Array.isArray(traceObj)) return traceObj;
  if (traceObj && Array.isArray(traceObj.turns)) return traceObj.turns;
  if (traceObj && Array.isArray(traceObj.decisions)) return traceObj.decisions;
  return [];
}

function turnSwing(t) {
  if (!t) return null;
  if (Number.isFinite(t.value_delta)) return t.value_delta;
  if (Number.isFinite(t.value_before) && Number.isFinite(t.value_after)) {
    return t.value_after - t.value_before;
  }
  return null;
}

export function ReplaysView({
  projects,
  runs,
  theme,
  initialRunId = "",
  onOpenCards,
  onOpenRun,
}) {
  const [project, setProject] = useState("");
  const [runId, setRunId] = useState(initialRunId || "");
  const [formatFilter, setFormatFilter] = useState("");
  const [deckFilter, setDeckFilter] = useState("");
  const [search, setSearch] = useState("");
  const [traceOnly, setTraceOnly] = useState(false);
  const [minSwing, setMinSwing] = useState("");
  const [sortBy, setSortBy] = useState("created_at");

  const [replays, setReplays] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedReplay, setSelectedReplay] = useState(null);
  const [traceData, setTraceData] = useState(null);
  const [traceLoading, setTraceLoading] = useState(false);
  const [turnIdx, setTurnIdx] = useState(0);
  const [copiedMsg, setCopiedMsg] = useState("");
  const [showGodotSnippet, setShowGodotSnippet] = useState(false);

  useEffect(() => {
    if (initialRunId) setRunId(initialRunId);
  }, [initialRunId]);

  async function loadReplays() {
    setLoading(true);
    try {
      const qs = new URLSearchParams();
      if (project) qs.set("project", project);
      if (runId) qs.set("run_id", runId);
      if (formatFilter) qs.set("format", formatFilter);
      if (deckFilter.trim()) qs.set("deck", deckFilter.trim());
      if (search.trim()) qs.set("q", search.trim());
      if (traceOnly) qs.set("has_trace", "true");
      if (minSwing !== "" && Number.isFinite(Number(minSwing))) {
        qs.set("min_swing", String(Number(minSwing)));
      }
      qs.set("sort_by", sortBy);
      qs.set("order", "desc");
      qs.set("limit", "100");
      const res = await apiFetch(`/api/replays?${qs.toString()}`);
      setReplays(res.replays || []);
    } catch (err) {
      console.error("Failed loading replays:", err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadReplays();
  }, [project, runId, formatFilter, deckFilter, search, traceOnly, minSwing, sortBy]);

  useEffect(() => {
    const close = connectSSE((ev) => {
      if (ev.type === "replay_recorded") {
        loadReplays();
      }
    });
    return close;
  }, [project, runId, formatFilter, sortBy]);

  async function inspectReplay(replay) {
    setSelectedReplay(replay);
    setTraceData(null);
    setTurnIdx(0);
    if (!replay.has_rl_trace) return;
    setTraceLoading(true);
    try {
      const res = await apiFetch(`/api/replays/${replay.id}/trace`);
      setTraceData(res);
      const turns = normalizeTraceTurns(res.trace);
      if (res.blunder_turn != null && turns.length > 0) {
        const idx = turns.findIndex(
          (t, i) => Number(t.turn ?? t.decision_idx ?? i) === Number(res.blunder_turn)
        );
        if (idx >= 0) setTurnIdx(idx);
      }
    } catch (err) {
      console.error("Failed loading trace:", err);
    } finally {
      setTraceLoading(false);
    }
  }

  async function togglePin(replay, e) {
    if (e) e.stopPropagation();
    try {
      const res = await apiFetch(`/api/replays/${replay.id}`, {
        method: "PATCH",
        body: JSON.stringify({ pinned: !replay.pinned }),
      });
      setReplays((prev) => prev.map((r) => (r.id === replay.id ? res.replay : r)));
      if (selectedReplay && selectedReplay.id === replay.id) {
        setSelectedReplay(res.replay);
      }
    } catch (err) {
      alert(`Pin failed: ${err.message}`);
    }
  }

  async function deleteReplay(replay, e) {
    if (e) e.stopPropagation();
    if (!window.confirm(`Delete replay '${replay.title}'?`)) return;
    try {
      await apiFetch(`/api/replays/${replay.id}`, { method: "DELETE" });
      if (selectedReplay && selectedReplay.id === replay.id) {
        setSelectedReplay(null);
        setTraceData(null);
      }
      loadReplays();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  }

  function copyText(text, label) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        setCopiedMsg(`Copied ${label}`);
        setTimeout(() => setCopiedMsg(""), 2000);
      });
    } else {
      window.prompt(`Copy ${label}:`, text);
    }
  }

  const turns = useMemo(
    () => (traceData ? normalizeTraceTurns(traceData.trace) : []),
    [traceData]
  );

  const clampedTurnIdx = turns.length > 0 ? Math.min(Math.max(0, turnIdx), turns.length - 1) : 0;
  const currentTurn = turns[clampedTurnIdx] || null;

  const valueChartData = useMemo(() => {
    if (turns.length === 0) return { runList: [], seriesMap: {} };
    const xs = turns.map((t, i) => Number(t.turn ?? t.decision_idx ?? i + 1));
    const beforeYs = turns.map((t) =>
      Number.isFinite(t.value_before) ? t.value_before : Number.isFinite(t.value) ? t.value : null
    );
    const afterYs = turns.map((t) =>
      Number.isFinite(t.value_after) ? t.value_after : null
    );
    const runList = [{ id: "before", name: "Value (before)" }];
    const seriesMap = { before: { x: xs, y: beforeYs } };
    if (afterYs.some((v) => v !== null)) {
      runList.push({ id: "after", name: "Value (after)" });
      seriesMap.after = { x: xs, y: afterYs };
    }
    return { runList, seriesMap };
  }, [turns]);

  const blunderIndex = useMemo(() => {
    if (!traceData || traceData.blunder_turn == null || turns.length === 0) return -1;
    return turns.findIndex(
      (t, i) => Number(t.turn ?? t.decision_idx ?? i) === Number(traceData.blunder_turn)
    );
  }, [traceData, turns]);

  const godotSnippet = useMemo(() => {
    const rid = selectedReplay ? selectedReplay.id : "<REPLAY_ID>";
    const base = window.location.origin;
    return `# Godot 4 GDScript snippet to fetch & load a ReplayRecord from TrainingLog
extends Node
func fetch_replay(replay_id: String = "${rid}") -> void:
\tvar http := HTTPRequest.new()
\tadd_child(http)
\thttp.request_completed.connect(func(_res, code, _hdrs, body):
\t\tif code == 200:
\t\t\tvar record = JSON.parse_string(body.get_string_from_utf8())
\t\t\tprint("Loaded ReplayRecord version=", record.get("version"), " actions=", record.get("actions", []).size())
\t\thttp.queue_free())
\thttp.request("${base}/api/replays/" + replay_id + "/file?inline=1")`;
  }, [selectedReplay]);

  return html`
    <div class="stack">
      <div class="space-between" style="flex-wrap:wrap; gap:8px;">
        <div>
          <h2 style="margin:0; font-size:17px;">Godot Replay Hub (${replays.length})</h2>
          <div class="muted" style="font-size:12px;">
            Raw Godot ReplayRecord JSONs, MP4/WebM recordings & RL critic blunder traces
          </div>
        </div>
        <div class="row">
          ${copiedMsg ? html`<span class="pill ok">${copiedMsg}</span>` : null}
          ${onOpenCards
            ? html`<button class="btn sm" onClick=${onOpenCards}>Card Meta & Policy →</button>`
            : null}
        </div>
      </div>

      <div class="card">
        <div class="row">
          <select class="select" value=${project} onChange=${(e) => setProject(e.target.value)}>
            <option value="">All Projects</option>
            ${(projects || []).map((p) => html`<option value=${p.name}>${p.name}</option>`)}
          </select>
          <select class="select" value=${runId} onChange=${(e) => setRunId(e.target.value)}>
            <option value="">All Runs</option>
            ${(runs || []).map((r) => html`<option value=${r.id}>${r.name}</option>`)}
          </select>
          <select class="select" value=${formatFilter} onChange=${(e) => setFormatFilter(e.target.value)}>
            <option value="">All Formats</option>
            <option value="cardgame_replay_json">Godot ReplayRecord</option>
            <option value="mp4">MP4 Video</option>
            <option value="webm">WebM Video</option>
            <option value="json">JSON</option>
            <option value="bin">Binary</option>
          </select>
          <select class="select" value=${sortBy} onChange=${(e) => setSortBy(e.target.value)}>
            <option value="created_at">Sort: Newest</option>
            <option value="max_value_swing">Sort: Largest Swing</option>
            <option value="step">Sort: Step</option>
            <option value="action_count">Sort: Action Count</option>
          </select>
        </div>
        <div class="row" style="margin-top:8px;">
          <div style="flex:1.5; min-width:150px;">
            <input
              class="input"
              placeholder="Search title, player, manifest…"
              value=${search}
              onInput=${(e) => setSearch(e.target.value)}
            />
          </div>
          <div style="flex:1; min-width:140px;">
            <input
              class="input"
              placeholder="Filter deck card (e.g. Firebolt)…"
              value=${deckFilter}
              onInput=${(e) => setDeckFilter(e.target.value)}
            />
          </div>
          <div style="width:110px;">
            <input
              class="input mono"
              type="number"
              step="0.05"
              min="0"
              placeholder="Min swing"
              value=${minSwing}
              onInput=${(e) => setMinSwing(e.target.value)}
            />
          </div>
          <label class="row muted" style="font-size:12px; gap:5px; cursor:pointer;">
            <input
              type="checkbox"
              checked=${traceOnly}
              onChange=${(e) => setTraceOnly(e.target.checked)}
            />
            RL Trace only
          </label>
        </div>
      </div>

      ${selectedReplay
        ? html`
            <div class="card stack">
              <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                <div class="row">
                  <strong style="font-size:15px;">${selectedReplay.title}</strong>
                  <span class="pill accent">${selectedReplay.format}</span>
                  ${selectedReplay.blunder_turn != null
                    ? html`<span class="pill danger">
                        Blunder T${selectedReplay.blunder_turn} (Δ ${Number(selectedReplay.max_value_swing || 0).toFixed(2)})
                      </span>`
                    : null}
                </div>
                <div class="row">
                  <a
                    class="btn sm primary"
                    href=${withTokenQuery(`/api/replays/${selectedReplay.id}/file`)}
                    download=${selectedReplay.filename || `${selectedReplay.id}.json`}
                  >
                    Download Raw
                  </a>
                  <button
                    class="btn sm"
                    onClick=${() =>
                      copyText(
                        `${window.location.origin}${withTokenQuery(`/api/replays/${selectedReplay.id}/file?inline=1`)}`,
                        "Godot URL"
                      )}
                  >
                    Copy Godot URL
                  </button>
                  <button
                    class="btn sm"
                    onClick=${() => setShowGodotSnippet((s) => !s)}
                  >
                    ${showGodotSnippet ? "Hide GDScript" : "Fetch in Godot"}
                  </button>
                  <button
                    class="btn sm"
                    onClick=${() => {
                      setSelectedReplay(null);
                      setTraceData(null);
                    }}
                  >
                    Close
                  </button>
                </div>
              </div>

              ${showGodotSnippet
                ? html`<pre class="code-block">${godotSnippet}</pre>`
                : null}

              ${selectedReplay.format === "mp4" || selectedReplay.format === "webm"
                ? html`
                    <video
                      class="replay-video"
                      controls
                      playsinline
                      preload="metadata"
                      src=${withTokenQuery(`/api/replays/${selectedReplay.id}/file?inline=1`)}
                    ></video>
                  `
                : null}

              <div class="metrics-grid">
                <div class="metric-tile">
                  <div class="metric-label">Matchup</div>
                  <div style="font-size:13px; font-weight:600;">
                    ${selectedReplay.p1_name || "P1"} vs ${selectedReplay.p2_name || "P2"}
                    ${selectedReplay.winner != null ? ` (Winner: P${selectedReplay.winner})` : ""}
                  </div>
                </div>
                <div class="metric-tile">
                  <div class="metric-label">Godot Record</div>
                  <div class="mono" style="font-size:12px;">
                    v${selectedReplay.version ?? "—"} · ${selectedReplay.action_count || 0} actions · ${formatBytes(selectedReplay.size_bytes)}
                  </div>
                </div>
                <div class="metric-tile">
                  <div class="metric-label">Seeds / Manifest</div>
                  <div class="mono" style="font-size:12px;">
                    ${selectedReplay.seed1 ?? "—"} / ${selectedReplay.seed2 ?? "—"} · ${selectedReplay.manifest_hash || "—"}
                  </div>
                </div>
              </div>

              ${(selectedReplay.deck1 || []).length > 0 || (selectedReplay.deck2 || []).length > 0
                ? html`
                    <div class="grid-2">
                      <div>
                        <div class="muted" style="font-size:11px; margin-bottom:4px;">Deck 1 (${(selectedReplay.deck1 || []).length})</div>
                        <div class="row">${(selectedReplay.deck1 || []).map((c) => html`<span class="pill">${c}</span>`)}</div>
                      </div>
                      <div>
                        <div class="muted" style="font-size:11px; margin-bottom:4px;">Deck 2 (${(selectedReplay.deck2 || []).length})</div>
                        <div class="row">${(selectedReplay.deck2 || []).map((c) => html`<span class="pill">${c}</span>`)}</div>
                      </div>
                    </div>
                  `
                : null}

              ${selectedReplay.has_rl_trace
                ? traceLoading
                  ? html`<div class="muted">Loading RL decision trace…</div>`
                  : turns.length > 0
                  ? html`
                      <div class="stack">
                        <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                          <strong>RL Turn-by-Turn Decision Trace (${turns.length} turns)</strong>
                          ${blunderIndex >= 0
                            ? html`
                                <button
                                  class="btn sm warn"
                                  onClick=${() => setTurnIdx(blunderIndex)}
                                >
                                  Jump to Blunder (Turn ${traceData.blunder_turn})
                                </button>
                              `
                            : null}
                        </div>

                        <${UPlotChart}
                          title="Critic Value Estimate by Turn"
                          seriesMap=${valueChartData.seriesMap}
                          runList=${valueChartData.runList}
                          xAxis="turn"
                          smoothing=${0}
                          theme=${theme}
                          height=${185}
                        />

                        <div class="turn-scrubber">
                          <button
                            class="btn sm"
                            disabled=${clampedTurnIdx <= 0}
                            onClick=${() => setTurnIdx((i) => Math.max(0, i - 1))}
                          >
                            Prev
                          </button>
                          <input
                            type="range"
                            min="0"
                            max=${Math.max(0, turns.length - 1)}
                            value=${clampedTurnIdx}
                            onInput=${(e) => setTurnIdx(Number(e.target.value))}
                          />
                          <button
                            class="btn sm"
                            disabled=${clampedTurnIdx >= turns.length - 1}
                            onClick=${() => setTurnIdx((i) => Math.min(turns.length - 1, i + 1))}
                          >
                            Next
                          </button>
                          <span class="mono" style="font-size:12px;">
                            Turn ${currentTurn?.turn ?? clampedTurnIdx} (${clampedTurnIdx + 1}/${turns.length})
                          </span>
                        </div>

                        ${currentTurn
                          ? html`
                              <div class="grid-2">
                                <div class="card stack" style="background:var(--surface-2);">
                                  <div class="space-between">
                                    <strong>
                                      Turn ${currentTurn.turn ?? clampedTurnIdx} · Player ${currentTurn.player ?? "?"}
                                    </strong>
                                    ${(() => {
                                      const sw = turnSwing(currentTurn);
                                      if (sw === null) return null;
                                      const isBlunder = sw <= -0.2;
                                      return html`<span class=${`pill ${isBlunder ? "danger" : sw >= 0.15 ? "ok" : ""}`}>
                                        ${sw >= 0 ? "+" : ""}${sw.toFixed(3)} value Δ
                                      </span>`;
                                    })()}
                                  </div>
                                  <div style="font-size:13px;">
                                    Chosen: <span class="mono" style="color:var(--ok); font-weight:700;">
                                      ${currentTurn.action_label || currentTurn.chosen_action || "—"}
                                    </span>
                                  </div>
                                  <div class="mono muted" style="font-size:12px;">
                                    V(s): ${currentTurn.value_before != null ? Number(currentTurn.value_before).toFixed(3) : "—"} →
                                    ${" "}V(s'): ${currentTurn.value_after != null ? Number(currentTurn.value_after).toFixed(3) : "—"}
                                  </div>
                                  ${(currentTurn.top_k_probs || []).length > 0
                                    ? html`
                                        <div class="stack" style="gap:6px; margin-top:4px;">
                                          <div class="muted" style="font-size:11px;">Policy Top-K Action Probabilities</div>
                                          <div class="prob-list">
                                            ${currentTurn.top_k_probs.map((opt) => {
                                              const label = opt.label || opt.action || String(opt.index ?? "");
                                              const prob = Number(opt.prob || 0);
                                              const chosen =
                                                opt.chosen ||
                                                label === (currentTurn.action_label || currentTurn.chosen_action);
                                              return html`
                                                <div class="prob-row">
                                                  <span class="mono" style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                                                    ${label}
                                                  </span>
                                                  <div class="prob-bar-track">
                                                    <div
                                                      class=${`prob-bar-fill ${chosen ? "chosen" : ""}`}
                                                      style=${`width: ${Math.max(2, Math.min(100, prob * 100))}%`}
                                                    ></div>
                                                  </div>
                                                  <span class="mono" style="text-align:right;">
                                                    ${(prob * 100).toFixed(1)}%
                                                  </span>
                                                </div>
                                              `;
                                            })}
                                          </div>
                                        </div>
                                      `
                                    : null}
                                </div>

                                <div class="turn-step-list">
                                  ${turns.map((t, i) => {
                                    const sw = turnSwing(t);
                                    const isBlunder = i === blunderIndex || (sw !== null && sw <= -0.2);
                                    return html`
                                      <div
                                        key=${i}
                                        class=${`turn-row ${i === clampedTurnIdx ? "active" : ""} ${isBlunder ? "blunder" : ""}`}
                                        onClick=${() => setTurnIdx(i)}
                                      >
                                        <span class="mono">T${t.turn ?? i} P${t.player ?? "?"}</span>
                                        <span class="mono" style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                                          ${t.action_label || t.chosen_action || "step"}
                                        </span>
                                        <span class=${`mono ${sw !== null && sw < -0.2 ? "pill danger" : "muted"}`}>
                                          ${sw !== null ? `${sw >= 0 ? "+" : ""}${sw.toFixed(2)}` : "—"}
                                        </span>
                                      </div>
                                    `;
                                  })}
                                </div>
                              </div>
                            `
                          : null}
                      </div>
                    `
                  : html`<div class="muted" style="font-size:12px;">RL trace contains no turns.</div>`
                : html`<div class="muted" style="font-size:12px;">No RL decision trace attached to this replay.</div>`}
            </div>
          `
        : null}

      <div class="card">
        ${loading && replays.length === 0
          ? html`<div class="muted">Loading replays…</div>`
          : replays.length === 0
          ? html`<div class="muted">No replays uploaded yet. Call <code>run.log_replay(...)</code> or upload via <code>/api/replays</code>.</div>`
          : html`
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Replay</th>
                      <th>Run / Step</th>
                      <th>Format</th>
                      <th>Actions</th>
                      <th>Value Swing</th>
                      <th>Updated</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    ${replays.map((r) => {
                      const swing = Number(r.max_value_swing || 0);
                      const swingClass = swing >= 0.35 ? "danger" : swing >= 0.15 ? "warn" : "";
                      return html`
                        <tr
                          key=${r.id}
                          style="cursor:pointer;"
                          onClick=${() => inspectReplay(r)}
                        >
                          <td>
                            <div style="font-weight:600;">
                              ${r.pinned ? "★ " : ""}${r.title}
                            </div>
                            <div class="muted" style="font-size:11.5px;">
                              ${r.p1_name || "P1"} vs ${r.p2_name || "P2"} · ${formatBytes(r.size_bytes)}
                              ${r.tags && r.tags.length > 0 ? ` · ${r.tags.map((t) => `#${t}`).join(" ")}` : ""}
                            </div>
                          </td>
                          <td>
                            ${r.run_id && onOpenRun
                              ? html`<a
                                  href=${`#/runs/${r.run_id}`}
                                  onClick=${(e) => {
                                    e.stopPropagation();
                                    e.preventDefault();
                                    onOpenRun(r.run_id);
                                  }}
                                >${r.run_name || r.run_id}</a>`
                              : html`<span class="muted">${r.run_name || "—"}</span>`}
                            <div class="mono muted" style="font-size:11px;">
                              ${r.step != null ? `step ${formatStep(r.step)}` : "no step"}
                            </div>
                          </td>
                          <td>
                            <span class="pill accent">${r.format}</span>
                            ${r.has_rl_trace ? html`<span class="pill ok" style="margin-left:4px;">Trace</span>` : null}
                          </td>
                          <td class="mono">${r.action_count || "—"}</td>
                          <td>
                            ${r.has_rl_trace
                              ? html`<span class=${`pill ${swingClass}`}>
                                  ${swing.toFixed(2)}${r.blunder_turn != null ? ` @T${r.blunder_turn}` : ""}
                                </span>`
                              : html`<span class="muted">—</span>`}
                          </td>
                          <td class="muted">${formatTimeAgo(r.created_at)}</td>
                          <td onClick=${(e) => e.stopPropagation()}>
                            <div class="row" style="gap:4px;">
                              <button class="btn sm" onClick=${(e) => togglePin(r, e)}>
                                ${r.pinned ? "Unpin" : "Pin"}
                              </button>
                              <a
                                class="btn sm"
                                href=${withTokenQuery(`/api/replays/${r.id}/file`)}
                                download=${r.filename || `${r.id}.bin`}
                              >
                                DL
                              </a>
                              <button class="btn sm danger" onClick=${(e) => deleteReplay(r, e)}>
                                Del
                              </button>
                            </div>
                          </td>
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              </div>
            `}
      </div>
    </div>
  `;
}
