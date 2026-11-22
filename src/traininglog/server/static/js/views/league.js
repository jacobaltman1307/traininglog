import {
  apiFetch,
  formatNumber,
  formatPct,
  formatStep,
  formatTimeAgo,
  withTokenQuery,
} from "../api.js";
import { PlasticityPanel } from "../components/plasticity_panel.js";
import { UPlotChart } from "../uplot_chart.js";
import { html, useEffect, useMemo, useState } from "../../vendor/ui.mjs";

function cellStyle(winRate) {
  if (winRate === null || winRate === undefined) return "";
  const pct = winRate * 100;
  if (pct >= 65) return "background: rgba(16, 185, 129, 0.30); color: #6ee7b7;";
  if (pct >= 53) return "background: rgba(16, 185, 129, 0.16); color: #a7f3d0;";
  if (pct <= 35) return "background: rgba(239, 68, 68, 0.30); color: #fca5a5;";
  if (pct <= 47) return "background: rgba(239, 68, 68, 0.16); color: #fecaca;";
  return "background: var(--surface-2); color: var(--text);";
}

function buildLineageRows(nodes) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const childrenMap = new Map();
  for (const n of nodes) {
    const pid = n.parent_id && byId.has(n.parent_id) ? n.parent_id : "__root__";
    if (!childrenMap.has(pid)) childrenMap.set(pid, []);
    childrenMap.get(pid).push(n);
  }
  const rows = [];
  const visited = new Set();
  function walk(parentId, depth) {
    const list = childrenMap.get(parentId) || [];
    for (const item of list) {
      if (visited.has(item.id)) continue;
      visited.add(item.id);
      rows.push({ ckpt: item, depth });
      walk(item.id, depth + 1);
    }
  }
  walk("__root__", 0);
  for (const n of nodes) {
    if (!visited.has(n.id)) rows.push({ ckpt: n, depth: 0 });
  }
  return rows;
}

export function LeagueView({ projects, theme, onOpenRun, onOpenCards }) {
  const [subTab, setSubTab] = useState("leaderboard");
  const [project, setProject] = useState("");
  const [ratingSystem, setRatingSystem] = useState("glicko2");
  const [savedOnly, setSavedOnly] = useState(false);
  const [includeBots, setIncludeBots] = useState(true);
  const [archFilter, setArchFilter] = useState("");
  const [search, setSearch] = useState("");

  const [leaderboard, setLeaderboard] = useState([]);
  const [matrixData, setMatrixData] = useState({ agents: [], cells: {} });
  const [seatData, setSeatData] = useState(null);
  const [ratingHist, setRatingHist] = useState({});
  const [lineageData, setLineageData] = useState({ nodes: [], edges: [] });
  const [allCheckpoints, setAllCheckpoints] = useState([]);

  const [matchupModal, setMatchupModal] = useState(null);
  const [plasticityModal, setPlasticityModal] = useState(null);
  const [tagDrafts, setTagDrafts] = useState({});
  const [recomputing, setRecomputing] = useState(false);

  const arches = useMemo(() => {
    const s = new Set();
    for (const r of allCheckpoints) {
      if (r.arch && r.arch !== "bot") s.add(r.arch);
    }
    return Array.from(s).sort();
  }, [allCheckpoints]);

  async function loadLeague() {
    try {
      const qs = new URLSearchParams();
      if (project) qs.set("project", project);
      if (archFilter) qs.set("arch", archFilter);
      if (savedOnly) qs.set("saved_only", "true");
      qs.set("include_bots", includeBots ? "true" : "false");
      if (search.trim()) qs.set("search", search.trim());
      const sortBy =
        ratingSystem === "elo"
          ? "elo"
          : ratingSystem === "openskill"
            ? "os_ordinal"
            : "glicko_mu";
      qs.set("sort_by", sortBy);

      const projQs = project ? `?project=${encodeURIComponent(project)}` : "";
      const [lbRes, matRes, seatRes, linRes, ckptRes] = await Promise.all([
        apiFetch(`/api/league/leaderboard?${qs.toString()}`),
        apiFetch(`/api/league/matrix?${qs.toString()}`),
        apiFetch(`/api/league/seats${projQs}`),
        apiFetch(`/api/checkpoints/lineage${projQs}`),
        apiFetch(`/api/checkpoints${projQs}`),
      ]);
      const entries = lbRes.entries || [];
      setLeaderboard(entries);
      setMatrixData(matRes);
      setSeatData(seatRes);
      setLineageData(linRes);
      setAllCheckpoints(ckptRes.checkpoints || []);

      const topIds = entries
        .filter((e) => !e.is_bot)
        .slice(0, 6)
        .map((e) => e.id);
      if (topIds.length > 0) {
        const hRes = await apiFetch(
          `/api/league/rating-history?agent_ids=${encodeURIComponent(topIds.join(","))}`
        );
        setRatingHist(hRes.history || {});
      } else {
        setRatingHist({});
      }
    } catch {
      // ignore transient errors
    }
  }

  useEffect(() => {
    loadLeague();
  }, [project, ratingSystem, savedOnly, includeBots, archFilter, search]);

  async function recomputeRatings() {
    const targetProject = project || (projects[0] && projects[0].name);
    if (!targetProject) return;
    setRecomputing(true);
    try {
      await apiFetch(
        `/api/league/recompute?project=${encodeURIComponent(targetProject)}`,
        { method: "POST" }
      );
      await loadLeague();
    } catch (err) {
      alert(`Recompute failed: ${err.message}`);
    } finally {
      setRecomputing(false);
    }
  }

  async function openCellDetail(agentA, agentB, cell) {
    try {
      const mRes = await apiFetch(
        `/api/matches?agent_id=${encodeURIComponent(agentA.id)}&limit=50`
      );
      const filtered = (mRes.matches || []).filter((m) =>
        (m.participants || []).some((p) => p.agent_id === agentB.id)
      );
      setMatchupModal({ agentA, agentB, cell, matches: filtered });
    } catch {
      setMatchupModal({ agentA, agentB, cell, matches: [] });
    }
  }

  async function openPlasticityForCheckpoint(ckpt) {
    try {
      const res = await apiFetch(
        `/api/plasticity?checkpoint_id=${encodeURIComponent(ckpt.id)}&limit=25`
      );
      setPlasticityModal({ ckpt, reports: res.reports || [] });
    } catch (err) {
      alert(`Failed to load plasticity: ${err.message}`);
    }
  }

  async function togglePin(ckpt) {
    await apiFetch(`/api/checkpoints/${ckpt.id}`, {
      method: "PATCH",
      body: JSON.stringify({ pinned: !ckpt.pinned }),
    });
    await loadLeague();
  }

  async function addTag(ckpt, tag) {
    const clean = tag.trim();
    if (!clean) return;
    const nextTags = Array.from(new Set([...(ckpt.tags || []), clean]));
    await apiFetch(`/api/checkpoints/${ckpt.id}`, {
      method: "PATCH",
      body: JSON.stringify({ tags: nextTags }),
    });
    setTagDrafts((prev) => ({ ...prev, [ckpt.id]: "" }));
    await loadLeague();
  }

  const ratingChartProps = useMemo(() => {
    const entries = Object.entries(ratingHist);
    if (entries.length === 0) return null;
    const runList = [];
    const seriesMap = {};
    for (const [aid, pts] of entries) {
      if (!pts || pts.length === 0) continue;
      runList.push({ id: aid, name: pts[0].agent_name || aid });
      seriesMap[aid] = {
        x: pts.map((_, idx) => idx + 1),
        y: pts.map((p) =>
          ratingSystem === "elo"
            ? p.elo
            : ratingSystem === "openskill"
              ? p.os_ordinal
              : p.glicko_mu
        ),
        non_finite_steps: [],
      };
    }
    return runList.length > 0 ? { runList, seriesMap } : null;
  }, [ratingHist, ratingSystem]);

  const lineageRows = useMemo(
    () => buildLineageRows(lineageData.nodes || []),
    [lineageData]
  );

  return html`
    <div class="stack">
      <div class="space-between" style="flex-wrap:wrap;">
        <div>
          <h2 style="margin:0; font-size:17px;">Self-Play League & Checkpoints</h2>
          <div class="muted" style="font-size:12px;">
            Glicko-2 · Elo · OpenSkill · P1/P2 seat bias · Checkpoint lineage & ReDo plasticity
          </div>
        </div>
        <div class="row">
          <select
            class="select"
            style="width:auto;"
            value=${project}
            onChange=${(e) => setProject(e.target.value)}
          >
            <option value="">All Projects</option>
            ${projects.map(
              (p) => html`<option value=${p.name}>${p.name}</option>`
            )}
          </select>
          <button
            class="btn sm"
            disabled=${recomputing || (!project && projects.length === 0)}
            onClick=${recomputeRatings}
          >
            ${recomputing ? "Recomputing…" : "Recompute"}
          </button>
          ${onOpenCards
            ? html`<button class="btn sm" onClick=${onOpenCards}>Card Meta →</button>`
            : null}
        </div>
      </div>

      <div class="tabs">
        <button
          class=${`tab-btn ${subTab === "leaderboard" ? "active" : ""}`}
          onClick=${() => setSubTab("leaderboard")}
        >
          Leaderboard & H2H
        </button>
        <button
          class=${`tab-btn ${subTab === "seats" ? "active" : ""}`}
          onClick=${() => setSubTab("seats")}
        >
          Seats & Decks
        </button>
        <button
          class=${`tab-btn ${subTab === "checkpoints" ? "active" : ""}`}
          onClick=${() => setSubTab("checkpoints")}
        >
          Checkpoints & Lineage (${lineageRows.length})
        </button>
      </div>

      ${subTab === "leaderboard"
        ? html`
            <div class="card row" style="padding:10px 12px;">
              <select
                class="select"
                style="width:auto;"
                value=${ratingSystem}
                onChange=${(e) => setRatingSystem(e.target.value)}
              >
                <option value="glicko2">Rating: Glicko-2</option>
                <option value="elo">Rating: Elo</option>
                <option value="openskill">Rating: OpenSkill</option>
              </select>
              <select
                class="select"
                style="width:auto;"
                value=${archFilter}
                onChange=${(e) => setArchFilter(e.target.value)}
              >
                <option value="">All Archs</option>
                ${arches.map((a) => html`<option value=${a}>${a}</option>`)}
              </select>
              <div style="flex:1; min-width:140px;">
                <input
                  class="input"
                  placeholder="Filter models…"
                  value=${search}
                  onInput=${(e) => setSearch(e.target.value)}
                />
              </div>
              <label class="row" style="font-size:12px; cursor:pointer;">
                <input
                  type="checkbox"
                  checked=${savedOnly}
                  onChange=${(e) => setSavedOnly(e.target.checked)}
                />
                Saved only
              </label>
              <label class="row" style="font-size:12px; cursor:pointer;">
                <input
                  type="checkbox"
                  checked=${includeBots}
                  onChange=${(e) => setIncludeBots(e.target.checked)}
                />
                Include bots
              </label>
            </div>

            ${leaderboard.length === 0
              ? html`<div class="card empty-state">
                  No league matches recorded yet. Run <code class="mono">traininglog import-league</code> or <code class="mono">traininglog demo</code>.
                </div>`
              : html`
                  <div class="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Model</th>
                          <th>Arch</th>
                          <th>
                            ${ratingSystem === "elo"
                              ? "Elo"
                              : ratingSystem === "openskill"
                                ? "OpenSkill"
                                : "Glicko-2 (95% CI)"}
                          </th>
                          <th>Win %</th>
                          <th>P1 %</th>
                          <th>P2 %</th>
                          <th>Avg Rnds</th>
                          <th>Games</th>
                        </tr>
                      </thead>
                      <tbody>
                        ${leaderboard.map(
                          (r) => html`
                            <tr key=${r.id}>
                              <td class="mono">${r.rank}</td>
                              <td>
                                <div style="font-weight:600;">${r.name}</div>
                                <div class="row" style="gap:4px; margin-top:2px;">
                                  ${r.is_bot ? html`<span class="pill">bot</span>` : null}
                                  ${r.pinned ? html`<span class="pill accent">pinned</span>` : null}
                                  ${r.run_id
                                    ? html`<a
                                        href=${`#/runs/${r.run_id}`}
                                        style="font-size:11px;"
                                        onClick=${(e) => {
                                          e.preventDefault();
                                          onOpenRun(r.run_id);
                                        }}
                                      >
                                        run
                                      </a>`
                                    : null}
                                </div>
                              </td>
                              <td>${r.arch ? html`<span class="pill">${r.arch}</span>` : "—"}</td>
                              <td class="mono" style="font-weight:700;">
                                ${ratingSystem === "elo"
                                  ? Math.round(r.elo)
                                  : ratingSystem === "openskill"
                                    ? `${r.os_ordinal.toFixed(1)} (μ=${r.os_mu.toFixed(1)})`
                                    : `${r.glicko_mu.toFixed(1)} ± ${r.ci_95.toFixed(1)}`}
                              </td>
                              <td class="mono">${formatPct(r.win_rate)}</td>
                              <td class="mono">${r.p1_win_rate != null ? formatPct(r.p1_win_rate) : "—"}</td>
                              <td class="mono">${r.p2_win_rate != null ? formatPct(r.p2_win_rate) : "—"}</td>
                              <td class="mono">${r.avg_rounds ? r.avg_rounds.toFixed(1) : "—"}</td>
                              <td class="mono">${Number(r.games_played).toLocaleString()}</td>
                            </tr>
                          `
                        )}
                      </tbody>
                    </table>
                  </div>
                `}

            ${matrixData.agents && matrixData.agents.length >= 2
              ? html`
                  <div class="card stack">
                    <div class="space-between">
                      <strong>Head-to-Head Win-Rate Matrix (Row vs Col)</strong>
                      <span class="muted" style="font-size:11px;">Tap any cell for P1/P2 splits & series</span>
                    </div>
                    <div class="table-wrap">
                      <table class="heatmap-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Model</th>
                            ${matrixData.agents.map(
                              (col, idx) => html`<th key=${col.id} title=${col.name}>${idx + 1}</th>`
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          ${matrixData.agents.map(
                            (rowA, i) => html`
                              <tr key=${rowA.id}>
                                <td class="mono muted">${i + 1}</td>
                                <td style="font-weight:600; white-space:nowrap;">${rowA.name}</td>
                                ${matrixData.agents.map((colB, j) => {
                                  if (i === j) {
                                    return html`<td key=${colB.id} class="mono muted" style="text-align:center;">—</td>`;
                                  }
                                  const cell =
                                    matrixData.cells[rowA.id] &&
                                    matrixData.cells[rowA.id][colB.id];
                                  if (!cell) {
                                    return html`<td key=${colB.id} class="mono muted" style="text-align:center;">·</td>`;
                                  }
                                  return html`
                                    <td
                                      key=${colB.id}
                                      class="mono heatmap-cell"
                                      style=${cellStyle(cell.win_rate)}
                                      onClick=${() => openCellDetail(rowA, colB, cell)}
                                    >
                                      <div>${(cell.win_rate * 100).toFixed(0)}%</div>
                                      <div style="font-size:9px; opacity:0.8;">${cell.total_games}g</div>
                                    </td>
                                  `;
                                })}
                              </tr>
                            `
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                `
              : null}

            ${ratingChartProps
              ? html`
                  <${UPlotChart}
                    title=${`Rating Progression (${ratingSystem.toUpperCase()} across matches)`}
                    seriesMap=${ratingChartProps.seriesMap}
                    runList=${ratingChartProps.runList}
                    xAxis="step"
                    height=${220}
                    theme=${theme}
                  />
                `
              : null}
          `
        : null}

      ${subTab === "seats" && seatData
        ? html`
            <div class="stack">
              <div class="card stack">
                <strong>First-Player vs Second-Player Side Bias (1v1)</strong>
                ${seatData.side_split.total_games === 0
                  ? html`<div class="muted" style="font-size:12px;">No seat data yet.</div>`
                  : html`
                      <div class="space-between mono" style="font-size:13px;">
                        <span style="color:var(--accent);">
                          P1: ${formatPct(seatData.side_split.p1_win_rate)} (${seatData.side_split.p1_wins.toLocaleString()} wins)
                        </span>
                        <span style="color:var(--warn);">
                          P2: ${formatPct(seatData.side_split.p2_win_rate)} (${seatData.side_split.p2_wins.toLocaleString()} wins)
                        </span>
                      </div>
                      <div class="seat-bar">
                        <div
                          class="seat-p1"
                          style=${`width:${(seatData.side_split.p1_win_rate * 100).toFixed(1)}%`}
                        ></div>
                      </div>
                    `}
              </div>

              ${seatData.seats && seatData.seats.length > 0
                ? html`
                    <div class="card stack">
                      <strong>Per-Seat Breakdown (1v1 & Multiplayer)</strong>
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr><th>Seat</th><th>Wins</th><th>Losses</th><th>Games</th><th>Win %</th></tr>
                          </thead>
                          <tbody>
                            ${seatData.seats.map(
                              (s) => html`
                                <tr key=${s.seat}>
                                  <td class="mono">Seat ${s.seat + 1}</td>
                                  <td class="mono">${s.wins.toLocaleString()}</td>
                                  <td class="mono">${s.losses.toLocaleString()}</td>
                                  <td class="mono">${s.games.toLocaleString()}</td>
                                  <td class="mono">${formatPct(s.win_rate)}</td>
                                </tr>
                              `
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  `
                : null}

              <div class="card stack">
                <strong>Deck / Faction Matchups</strong>
                ${(seatData.deck_matchups || []).length === 0
                  ? html`<div class="muted" style="font-size:12px;">No deck tags recorded in matches yet.</div>`
                  : html`
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr><th>Deck A</th><th>Deck B</th><th>A Win %</th><th>A / B / D</th><th>Games</th></tr>
                          </thead>
                          <tbody>
                            ${seatData.deck_matchups.map(
                              (d, idx) => html`
                                <tr key=${idx}>
                                  <td style="font-weight:600;">${d.deck_a}</td>
                                  <td style="font-weight:600;">${d.deck_b}</td>
                                  <td class="mono">${formatPct(d.a_win_rate)}</td>
                                  <td class="mono">${d.a_wins} / ${d.b_wins} / ${d.draws}</td>
                                  <td class="mono">${d.total_games}</td>
                                </tr>
                              `
                            )}
                          </tbody>
                        </table>
                      </div>
                    `}
              </div>
            </div>
          `
        : null}

      ${subTab === "checkpoints"
        ? html`
            <div class="stack">
              ${lineageRows.length === 0
                ? html`<div class="card empty-state">No model checkpoints registered yet.</div>`
                : html`
                    <div class="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Checkpoint (Lineage)</th>
                            <th>Arch</th>
                            <th>Step</th>
                            <th>Glicko-2</th>
                            <th>Storage</th>
                            <th>ReDo Plasticity</th>
                            <th>Tags & Pin</th>
                          </tr>
                        </thead>
                        <tbody>
                          ${lineageRows.map(({ ckpt, depth }) => {
                            const plast = ckpt.latest_plasticity;
                            return html`
                              <tr key=${ckpt.id}>
                                <td>
                                  <div style=${`padding-left:${depth * 18}px; display:flex; align-items:center; gap:6px;`}>
                                    ${depth > 0 ? html`<span class="muted mono">└─</span>` : null}
                                    <div>
                                      <div style="font-weight:600;">${ckpt.name}</div>
                                      <div class="muted mono" style="font-size:10px;">
                                        ${ckpt.manifest_hash ? `patch ${ckpt.manifest_hash} · ` : ""}
                                        ${formatTimeAgo(ckpt.created_at)}
                                      </div>
                                    </div>
                                  </div>
                                </td>
                                <td>${ckpt.arch ? html`<span class="pill">${ckpt.arch}</span>` : "—"}</td>
                                <td class="mono">${formatStep(ckpt.step)}</td>
                                <td class="mono">
                                  ${ckpt.glicko_mu != null ? `${Number(ckpt.glicko_mu).toFixed(1)}` : "—"}
                                </td>
                                <td>
                                  ${ckpt.storage_mode === "uploaded"
                                    ? html`<a
                                        class="btn sm"
                                        href=${withTokenQuery(`/api/checkpoints/${ckpt.id}/download`)}
                                        download
                                      >
                                        Download
                                      </a>`
                                    : html`<span class="muted mono" style="font-size:11px;" title=${ckpt.path_or_blob}>
                                        path_ref
                                      </span>`}
                                </td>
                                <td>
                                  ${plast
                                    ? html`<button
                                        class=${`pill ${plast.dormant_pct > 35 ? "critical" : plast.dormant_pct > 20 ? "warning" : "running"}`}
                                        style="cursor:pointer;"
                                        onClick=${() => openPlasticityForCheckpoint(ckpt)}
                                      >
                                        ${Number(plast.dormant_pct).toFixed(1)}% dorm · ${Number(plast.mean_rank_ratio_pct).toFixed(0)}% rank
                                      </button>`
                                    : html`<span class="muted" style="font-size:11px;">—</span>`}
                                </td>
                                <td>
                                  <div class="row" style="gap:4px;">
                                    <button
                                      class=${`btn sm ${ckpt.pinned ? "primary" : ""}`}
                                      onClick=${() => togglePin(ckpt)}
                                    >
                                      ${ckpt.pinned ? "Pinned" : "Pin"}
                                    </button>
                                    ${(ckpt.tags || []).map(
                                      (t) => html`<span class="pill" key=${t}>#${t}</span>`
                                    )}
                                    <input
                                      class="input mono"
                                      style="width:74px; padding:3px 6px; font-size:11px;"
                                      placeholder="+tag"
                                      value=${tagDrafts[ckpt.id] || ""}
                                      onInput=${(e) =>
                                        setTagDrafts((p) => ({ ...p, [ckpt.id]: e.target.value }))}
                                      onKeyDown=${(e) => {
                                        if (e.key === "Enter") addTag(ckpt, tagDrafts[ckpt.id] || "");
                                      }}
                                    />
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
          `
        : null}

      ${matchupModal
        ? html`
            <div class="modal-backdrop" onClick=${() => setMatchupModal(null)}>
              <div class="modal-card stack" style="max-width:580px;" onClick=${(e) => e.stopPropagation()}>
                <div class="space-between">
                  <strong>${matchupModal.agentA.name} vs ${matchupModal.agentB.name}</strong>
                  <button class="btn sm" onClick=${() => setMatchupModal(null)}>Close</button>
                </div>
                <div class="stat-row">
                  <div class="stat-box">
                    <div class="stat-label">Overall Win %</div>
                    <div class="stat-val mono">${formatPct(matchupModal.cell.win_rate)}</div>
                  </div>
                  <div class="stat-box">
                    <div class="stat-label">As P1</div>
                    <div class="stat-val mono">
                      ${matchupModal.cell.as_p1_wr != null ? formatPct(matchupModal.cell.as_p1_wr) : "—"}
                    </div>
                  </div>
                  <div class="stat-box">
                    <div class="stat-label">As P2</div>
                    <div class="stat-val mono">
                      ${matchupModal.cell.as_p2_wr != null ? formatPct(matchupModal.cell.as_p2_wr) : "—"}
                    </div>
                  </div>
                </div>
                <div class="muted mono" style="font-size:12px;">
                  Record: ${matchupModal.cell.wins}W - ${matchupModal.cell.losses}L - ${matchupModal.cell.draws}D (${matchupModal.cell.total_games} games)
                </div>
                ${matchupModal.matches.length > 0
                  ? html`
                      <div class="table-wrap" style="max-height:220px;">
                        <table>
                          <thead><tr><th>Games</th><th>Score</th><th>Avg Rnds</th><th>When</th></tr></thead>
                          <tbody>
                            ${matchupModal.matches.map((m) => {
                              const pa = m.participants.find((p) => p.agent_id === matchupModal.agentA.id);
                              const pb = m.participants.find((p) => p.agent_id === matchupModal.agentB.id);
                              return html`
                                <tr key=${m.id}>
                                  <td class="mono">${m.total_games}</td>
                                  <td class="mono">${pa?.wins ?? 0} - ${pb?.wins ?? 0}</td>
                                  <td class="mono">${Number(m.avg_rounds || 0).toFixed(1)}</td>
                                  <td class="muted">${formatTimeAgo(m.played_at)}</td>
                                </tr>
                              `;
                            })}
                          </tbody>
                        </table>
                      </div>
                    `
                  : null}
              </div>
            </div>
          `
        : null}

      ${plasticityModal
        ? html`
            <div class="modal-backdrop" onClick=${() => setPlasticityModal(null)}>
              <div class="modal-card stack" style="max-width:760px;" onClick=${(e) => e.stopPropagation()}>
                <div class="space-between">
                  <strong>Plasticity Report: ${plasticityModal.ckpt.name}</strong>
                  <button class="btn sm" onClick=${() => setPlasticityModal(null)}>Close</button>
                </div>
                <${PlasticityPanel} reports=${plasticityModal.reports} theme=${theme} />
              </div>
            </div>
          `
        : null}
    </div>
  `;
}
