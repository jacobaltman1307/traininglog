import {
  apiFetch,
  connectSSE,
  formatPct,
  formatSignedPct,
  withTokenQuery,
} from "../api.js";
import { QuadrantChart } from "../components/quadrant_chart.js";
import { UPlotChart } from "../uplot_chart.js";
import { html, useEffect, useMemo, useState } from "../../vendor/ui.mjs";

function signedStyle(val, thresh = 0.015) {
  if (val === null || val === undefined || !Number.isFinite(val)) {
    return "color: var(--muted);";
  }
  if (val >= thresh) return "color: var(--ok); font-weight: 700;";
  if (val <= -thresh) return "color: var(--danger); font-weight: 700;";
  return "";
}

function wrStyle(wr) {
  if (wr === null || wr === undefined || !Number.isFinite(wr)) {
    return "color: var(--muted);";
  }
  if (wr >= 0.54) return "color: var(--ok); font-weight: 700;";
  if (wr <= 0.46) return "color: var(--danger); font-weight: 700;";
  return "";
}

function LaneMiniBars({ lanePref }) {
  if (!lanePref || !lanePref.lanes || lanePref.lane_plays === 0) {
    if (lanePref && lanePref.global_plays > 0) {
      return html`<span class="pill">Global</span>`;
    }
    return html`<span class="muted">—</span>`;
  }
  const maxShare = Math.max(0.05, ...lanePref.lanes.map((l) => l.share || 0));
  return html`
    <div class="row" style="gap:6px; align-items:center; flex-wrap:nowrap;">
      <div class="lane-spark" title=${`Favored: L${lanePref.favored_lane ?? "?"} · Concentration: ${(lanePref.concentration * 100).toFixed(0)}%`}>
        ${lanePref.lanes.map((l) => {
          const h = Math.max(12, Math.round(((l.share || 0) / maxShare) * 100));
          const isFav = l.lane === lanePref.favored_lane;
          const isBest = l.lane === lanePref.best_lane && lanePref.has_misplay_gap;
          const cls = isBest ? "best" : isFav ? "fav" : "";
          return html`<div class=${`lane-spark-bar ${cls}`} style=${`height:${h}%`}></div>`;
        })}
      </div>
      <span class="mono" style="font-size:11px;">
        L${lanePref.favored_lane ?? "?"}
      </span>
      ${lanePref.has_misplay_gap
        ? html`<span
            class="pill warning"
            style="padding:1px 5px; font-size:10px;"
            title=${`Played mostly L${lanePref.favored_lane}, best WR in L${lanePref.best_lane} (+${(lanePref.misplay_gap * 100).toFixed(1)}%)`}
          >
            →L${lanePref.best_lane}
          </span>`
        : null}
    </div>
  `;
}

export function MetaView({
  projects,
  theme,
  queryStr = "",
  onOpenCard,
  onSyncQuery,
}) {
  const initParams = useMemo(() => new URLSearchParams(queryStr || ""), []);

  const [subTab, setSubTab] = useState(initParams.get("tab") || "overview");
  const [project, setProject] = useState(initParams.get("project") || "");
  const [source, setSource] = useState(initParams.get("source") || "");
  const [manifestHash, setManifestHash] = useState(initParams.get("manifest_hash") || "");
  const [mode, setMode] = useState(initParams.get("mode") || "1v1");
  const [excludeBots, setExcludeBots] = useState(initParams.get("exclude_bots") !== "false");
  const [excludeMirrors, setExcludeMirrors] = useState(initParams.get("exclude_mirrors") === "true");
  const [skillMode, setSkillMode] = useState(initParams.get("skill") || "all");
  const [skillVal, setSkillVal] = useState(initParams.get("skill_val") || "");
  const [minGames, setMinGames] = useState(initParams.get("min_games") || "10");
  const [faction, setFaction] = useState(initParams.get("faction") || "");
  const [cardType, setCardType] = useState(initParams.get("card_type") || "");
  const [search, setSearch] = useState(initParams.get("q") || "");

  const [quadrantMetric, setQuadrantMetric] = useState("drawn_wpa");
  const [sortBy, setSortBy] = useState("meta_war");
  const [sortDir, setSortDir] = useState("desc");

  const [filtersMeta, setFiltersMeta] = useState(null);
  const [overview, setOverview] = useState({ kpis: {}, cards: [] });
  const [loading, setLoading] = useState(true);

  const [synLevel, setSynLevel] = useState("played");
  const [synSearch, setSynSearch] = useState("");
  const [synData, setSynData] = useState({ positive: [], negative: [] });

  const [patchA, setPatchA] = useState("");
  const [patchB, setPatchB] = useState("");
  const [patchDiff, setPatchDiff] = useState(null);
  const [trendCards, setTrendCards] = useState([]);
  const [trendMetric, setTrendMetric] = useState("drawn_wpa");
  const [trendsMap, setTrendsMap] = useState({});

  function buildApiParams({ includeMinGames = true, includeCatalog = true } = {}) {
    const qs = new URLSearchParams();
    if (project) qs.set("project", project);
    if (source) qs.set("source", source);
    if (manifestHash) qs.set("manifest_hash", manifestHash);
    if (mode) qs.set("mode", mode);
    qs.set("exclude_bots", excludeBots ? "true" : "false");
    qs.set("exclude_mirrors", excludeMirrors ? "true" : "false");
    if (skillMode === "top_k" && Number(skillVal) > 0) {
      qs.set("top_k_agents", String(Math.round(Number(skillVal))));
    } else if (skillMode === "min_rating" && skillVal !== "" && Number.isFinite(Number(skillVal))) {
      qs.set("min_rating", String(Number(skillVal)));
    } else if (skillMode === "tier" && skillVal) {
      qs.set("tier", skillVal);
    }
    if (includeMinGames) qs.set("min_games", String(Number(minGames) || 0));
    if (includeCatalog) {
      if (faction) qs.set("faction", faction);
      if (cardType) qs.set("card_type", cardType);
    }
    return qs;
  }

  function buildShareableQuery() {
    const qs = buildApiParams({ includeMinGames: true, includeCatalog: true });
    if (subTab !== "overview") qs.set("tab", subTab);
    if (skillMode !== "all") {
      qs.set("skill", skillMode);
      if (skillVal) qs.set("skill_val", skillVal);
    }
    if (search.trim()) qs.set("q", search.trim());
    return qs.toString();
  }

  useEffect(() => {
    if (onSyncQuery) {
      onSyncQuery(buildShareableQuery());
    }
  }, [
    subTab,
    project,
    source,
    manifestHash,
    mode,
    excludeBots,
    excludeMirrors,
    skillMode,
    skillVal,
    minGames,
    faction,
    cardType,
    search,
  ]);

  async function loadFilters() {
    try {
      const qs = project ? `?project=${encodeURIComponent(project)}` : "";
      const res = await apiFetch(`/api/meta/filters${qs}`);
      setFiltersMeta(res);
      if (!source && res.defaults && res.defaults.source) {
        setSource(res.defaults.source);
      }
      if (!manifestHash && res.defaults && res.defaults.manifest_hash) {
        setManifestHash(res.defaults.manifest_hash);
      }
      const mans = res.manifests || [];
      if (mans.length >= 2 && !patchA && !patchB) {
        setPatchA(mans[1].manifest_hash);
        setPatchB(mans[0].manifest_hash);
      } else if (mans.length === 1 && !patchA && !patchB) {
        setPatchA(mans[0].manifest_hash);
        setPatchB(mans[0].manifest_hash);
      }
    } catch (err) {
      console.error("Failed loading meta filters:", err);
    }
  }

  async function loadOverview() {
    setLoading(true);
    try {
      const qs = buildApiParams();
      const res = await apiFetch(`/api/meta/overview?${qs.toString()}`);
      setOverview(res);
    } catch (err) {
      console.error("Failed loading meta overview:", err);
    } finally {
      setLoading(false);
    }
  }

  async function loadSynergies() {
    try {
      const qs = buildApiParams({ includeMinGames: false, includeCatalog: false });
      qs.set("level", synLevel);
      qs.set("limit", "60");
      const res = await apiFetch(`/api/meta/synergies?${qs.toString()}`);
      setSynData(res);
    } catch (err) {
      console.error("Failed loading synergies:", err);
    }
  }

  async function loadPatchDiffAndTrends() {
    try {
      if (patchA && patchB) {
        const dqs = new URLSearchParams();
        if (project) dqs.set("project", project);
        if (source) dqs.set("source", source);
        if (mode) dqs.set("mode", mode);
        dqs.set("exclude_bots", excludeBots ? "true" : "false");
        dqs.set("exclude_mirrors", excludeMirrors ? "true" : "false");
        dqs.set("manifest_a", patchA);
        dqs.set("manifest_b", patchB);
        const res = await apiFetch(`/api/meta/patch-diff?${dqs.toString()}`);
        setPatchDiff(res);
      }
      const tqs = buildApiParams({ includeMinGames: false, includeCatalog: false });
      tqs.set("manifest_hash", "all");
      if (trendCards.length > 0) tqs.set("cards", trendCards.join(","));
      const tres = await apiFetch(`/api/meta/trends?${tqs.toString()}`);
      setTrendsMap(tres.trends || {});
    } catch (err) {
      console.error("Failed loading patch diff/trends:", err);
    }
  }

  useEffect(() => {
    loadFilters();
  }, [project]);

  useEffect(() => {
    loadOverview();
  }, [
    project,
    source,
    manifestHash,
    mode,
    excludeBots,
    excludeMirrors,
    skillMode,
    skillVal,
    minGames,
    faction,
    cardType,
  ]);

  useEffect(() => {
    if (subTab === "synergies") loadSynergies();
    if (subTab === "patch_diff") loadPatchDiffAndTrends();
  }, [
    subTab,
    project,
    source,
    manifestHash,
    mode,
    excludeBots,
    excludeMirrors,
    skillMode,
    skillVal,
    synLevel,
    patchA,
    patchB,
    trendCards.join(","),
  ]);

  useEffect(() => {
    const close = connectSSE((ev) => {
      if (ev.type === "meta_updated") {
        loadFilters();
        loadOverview();
        if (subTab === "synergies") loadSynergies();
        if (subTab === "patch_diff") loadPatchDiffAndTrends();
      }
    });
    return close;
  }, [subTab, project, source, manifestHash]);

  function toggleSort(col) {
    if (sortBy === col) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortBy(col);
      setSortDir("desc");
    }
  }

  function openDossier(cardKey) {
    if (onOpenCard) onOpenCard(cardKey, buildShareableQuery());
  }

  const sortedCards = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = (overview.cards || []).filter((c) => {
      if (!q) return true;
      return (
        c.card_key.toLowerCase().includes(q) ||
        (c.display_name || "").toLowerCase().includes(q)
      );
    });
    list.sort((a, b) => {
      if (sortBy === "card_key") {
        return sortDir === "asc"
          ? a.card_key.localeCompare(b.card_key)
          : b.card_key.localeCompare(a.card_key);
      }
      const getVal = (c) => {
        if (sortBy === "top_synergy") return c.top_synergy ? c.top_synergy.lift : -1e9;
        if (sortBy === "lane_conc") return c.lane_preference ? c.lane_preference.concentration : -1e9;
        return c[sortBy] == null ? -1e9 : Number(c[sortBy]);
      };
      const av = getVal(a);
      const bv = getVal(b);
      return sortDir === "asc" ? av - bv : bv - av;
    });
    return list;
  }, [overview.cards, search, sortBy, sortDir]);

  const trendChart = useMemo(() => {
    const entries = Object.entries(trendsMap || {});
    const runList = [];
    const seriesMap = {};
    for (const [ckey, pts] of entries) {
      if (!pts || pts.length === 0) continue;
      runList.push({ id: ckey, name: ckey });
      seriesMap[ckey] = {
        x: pts.map((p) => Math.floor(Number(p.ts_ms) / 1000)),
        y: pts.map((p) => {
          const v = p[trendMetric];
          return v == null || !Number.isFinite(Number(v)) ? null : Number(v);
        }),
      };
    }
    return { runList, seriesMap };
  }, [trendsMap, trendMetric]);

  const filterQueryForExport = buildApiParams().toString();
  const kpis = overview.kpis || {};
  const hasCatalogData = (filtersMeta?.factions || []).length > 0 || (filtersMeta?.card_types || []).length > 0;
  const hasCriticData = (kpis.critic_coverage_pct || 0) > 0;

  return html`
    <div class="stack">
      <div class="space-between" style="flex-wrap:wrap; gap:8px;">
        <div>
          <h2 style="margin:0; font-size:17px;">Card Meta & Balance Explorer (${sortedCards.length})</h2>
          <div class="muted" style="font-size:12px;">
            Dual WPA, Regression Impact, Meta WAR, Multi-Level Synergy & 4-Lane Preference across League and Player matches
          </div>
        </div>
        <div class="row">
          <a
            class="btn sm"
            href=${withTokenQuery(`/api/meta/export?format=csv&${filterQueryForExport}`)}
            download="meta_cards.csv"
          >
            Export CSV
          </a>
          <a
            class="btn sm"
            href=${withTokenQuery(`/api/meta/export?format=json&${filterQueryForExport}`)}
            download="meta_cards.json"
          >
            Export JSON
          </a>
        </div>
      </div>

      <div class="card stack" style="gap:8px;">
        <div class="meta-filter-grid">
          <select
            class="select"
            value=${project}
            onChange=${(e) => {
              setProject(e.target.value);
              setSource("");
              setManifestHash("");
              setPatchA("");
              setPatchB("");
            }}
          >
            <option value="">Default Project</option>
            ${(projects || []).map((p) => html`<option value=${p.name}>${p.name}</option>`)}
          </select>

          <select class="select" value=${source} onChange=${(e) => setSource(e.target.value)}>
            <option value="all">Source: All</option>
            ${(filtersMeta?.sources || []).map(
              (s) => html`<option value=${s.source}>Source: ${s.source} (${s.games})</option>`
            )}
          </select>

          <select class="select" value=${manifestHash} onChange=${(e) => setManifestHash(e.target.value)}>
            <option value="all">Patch: All Patches</option>
            ${(filtersMeta?.manifests || []).map(
              (m) =>
                html`<option value=${m.manifest_hash}>
                  ${m.patch_label ? `${m.patch_label} (${m.manifest_hash})` : m.manifest_hash || "unversioned"} · ${m.games}g
                </option>`
            )}
          </select>

          <select class="select" value=${mode} onChange=${(e) => setMode(e.target.value)}>
            <option value="1v1">Mode: 1v1</option>
            <option value="multiplayer">Mode: Multiplayer FFA</option>
            <option value="all">Mode: All</option>
          </select>

          <select class="select" value=${minGames} onChange=${(e) => setMinGames(e.target.value)}>
            <option value="0">Min Games: All</option>
            <option value="10">Min Games: 10+</option>
            <option value="30">Min Games: 30+</option>
            <option value="100">Min Games: 100+</option>
            <option value="500">Min Games: 500+</option>
          </select>
        </div>

        <div class="row">
          <select
            class="select"
            style="width:auto; min-width:140px;"
            value=${skillMode}
            onChange=${(e) => {
              setSkillMode(e.target.value);
              setSkillVal(e.target.value === "top_k" ? "4" : "");
            }}
          >
            <option value="all">Skill: All Ratings</option>
            <option value="top_k">Top-K League Agents</option>
            <option value="min_rating">Min Rating ≥</option>
            ${(filtersMeta?.tiers || []).length > 0 ? html`<option value="tier">Player Tier</option>` : null}
          </select>

          ${skillMode === "top_k"
            ? html`<input
                class="input mono"
                style="width:90px;"
                type="number"
                min="2"
                placeholder="Top K"
                value=${skillVal}
                onInput=${(e) => setSkillVal(e.target.value)}
              />`
            : skillMode === "min_rating"
              ? html`<input
                  class="input mono"
                  style="width:110px;"
                  type="number"
                  placeholder="Min Rating"
                  value=${skillVal}
                  onInput=${(e) => setSkillVal(e.target.value)}
                />`
              : skillMode === "tier"
                ? html`
                    <select class="select" style="width:auto;" value=${skillVal} onChange=${(e) => setSkillVal(e.target.value)}>
                      <option value="">Select Tier…</option>
                      ${(filtersMeta?.tiers || []).map((t) => html`<option value=${t}>${t}</option>`)}
                    </select>
                  `
                : null}

          ${hasCatalogData
            ? html`
                <select class="select" style="width:auto;" value=${faction} onChange=${(e) => setFaction(e.target.value)}>
                  <option value="">All Factions</option>
                  ${(filtersMeta?.factions || []).map((f) => html`<option value=${f}>${f}</option>`)}
                </select>
                <select class="select" style="width:auto;" value=${cardType} onChange=${(e) => setCardType(e.target.value)}>
                  <option value="">All Types</option>
                  ${(filtersMeta?.card_types || []).map((t) => html`<option value=${t}>${t}</option>`)}
                </select>
              `
            : null}

          <div style="flex:1; min-width:160px;">
            <input
              class="input"
              placeholder="Search card name…"
              value=${search}
              onInput=${(e) => setSearch(e.target.value)}
            />
          </div>

          <label class="row muted" style="font-size:12px; gap:5px; cursor:pointer;">
            <input
              type="checkbox"
              checked=${excludeBots}
              onChange=${(e) => setExcludeBots(e.target.checked)}
            />
            Exclude Bots
          </label>

          <label
            class="row muted"
            style="font-size:12px; gap:5px; cursor:pointer;"
            title="Exclude 1v1 games where both players drew/played the same card from raw WR & WPA"
          >
            <input
              type="checkbox"
              checked=${excludeMirrors}
              onChange=${(e) => setExcludeMirrors(e.target.checked)}
            />
            Exclude Mirrors
          </label>
        </div>
      </div>

      <div class="tabs">
        ${[
          ["overview", "Overview"],
          ["synergies", "Synergies & Counters"],
          ["patch_diff", "Patch Diff & Trends"],
        ].map(
          ([k, label]) => html`
            <button
              class=${`tab-btn ${subTab === k ? "active" : ""}`}
              onClick=${() => setSubTab(k)}
            >
              ${label}
            </button>
          `
        )}
      </div>

      ${subTab === "overview"
        ? html`
            <div class="metrics-grid">
              <div class="stat-box card">
                <div class="stat-label">Filtered Games</div>
                <div class="stat-val mono">${(kpis.games || 0).toLocaleString()}</div>
              </div>
              <div class="stat-box card">
                <div class="stat-label">Tracked Cards</div>
                <div class="stat-val mono">${sortedCards.length}</div>
              </div>
              <div class="stat-box card">
                <div class="stat-label">Avg Rounds</div>
                <div class="stat-val mono">${kpis.avg_rounds ? Number(kpis.avg_rounds).toFixed(1) : "—"}</div>
              </div>
              <div class="stat-box card">
                <div class="stat-label">P1 Seat Win Rate</div>
                <div class="stat-val mono">${formatPct(kpis.p1_win_rate)}</div>
              </div>
              <div class="stat-box card">
                <div class="stat-label">Critic WPA Coverage</div>
                <div class="stat-val mono">
                  ${hasCriticData
                    ? `${Number(kpis.critic_coverage_pct).toFixed(0)}% (${kpis.critic_perspective || "auto"})`
                    : "None"}
                </div>
              </div>
            </div>

            ${!loading && sortedCards.length === 0
              ? html`
                  <div class="card empty-state">
                    <div style="font-weight:700; margin-bottom:4px;">No card meta telemetry matched these filters yet</div>
                    <div style="font-size:12.5px;">
                      Once league or player games are logged via <code>tl.log_meta_games(...)</code> or <code>POST /api/meta/games/batch</code>,
                      Dual WPA, Regression Impact, Synergy, and 4-Lane Preference will populate here.
                    </div>
                  </div>
                `
              : html`
                  <div class="card stack">
                    <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                      <div>
                        <strong>Meta Quadrant (Popularity vs Power)</strong>
                        <div class="muted" style="font-size:11.5px;">Click any card dot to open its full Meta Dossier</div>
                      </div>
                      <div class="tabs" style="margin-bottom:0;">
                        ${[
                          ["drawn_wpa", "Drawn WPA"],
                          ["isolated_impact", "Est. Impact (β)"],
                          ["meta_war", "Meta WAR"],
                          ["gih_wr", "GIH WR"],
                        ].map(
                          ([k, label]) => html`
                            <button
                              class=${`tab-btn ${quadrantMetric === k ? "active" : ""}`}
                              onClick=${() => setQuadrantMetric(k)}
                            >
                              ${label}
                            </button>
                          `
                        )}
                      </div>
                    </div>
                    <${QuadrantChart}
                      cards=${sortedCards}
                      yMetric=${quadrantMetric}
                      theme=${theme}
                      height=${280}
                      onSelectCard=${openDossier}
                    />
                  </div>

                  <div class="card stack">
                    <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                      <div>
                        <strong>All Cards Meta Table</strong>
                        <div class="muted" style="font-size:11.5px;">Click any card row to inspect funnel, lane WR/WPA, synergies & counters</div>
                      </div>
                      ${excludeMirrors
                        ? html`<span class="pill accent">Mirrors Excluded from Raw WR/WPA</span>`
                        : null}
                    </div>
                    <div class="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("card_key")}>Card</th>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("inclusion_rate")}>Incl %</th>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("gih_wr")}>GIH WR</th>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("drawn_wpa")}>Drawn WPA</th>
                            ${hasCriticData
                              ? html`<th style="cursor:pointer;" onClick=${() => toggleSort("critic_play_wpa")}>Critic WPA</th>`
                              : null}
                            <th style="cursor:pointer;" onClick=${() => toggleSort("isolated_impact")}>Est. Impact</th>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("meta_war")}>Meta WAR</th>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("top_synergy")}>Top Synergy</th>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("lane_conc")}>Lane Pref</th>
                            <th style="cursor:pointer;" onClick=${() => toggleSort("avg_turn_played")}>Avg Turn</th>
                          </tr>
                        </thead>
                        <tbody>
                          ${sortedCards.map((c) => {
                            const lowSample = c.games_drawn < 30;
                            const cat = c.catalog;
                            return html`
                              <tr
                                key=${c.card_key}
                                style=${`cursor:pointer; ${lowSample ? "opacity:0.74;" : ""}`}
                                onClick=${() => openDossier(c.card_key)}
                              >
                                <td>
                                  <div style="font-weight:700;">${c.display_name || c.card_key}</div>
                                  <div class="muted" style="font-size:11px;">
                                    ${cat
                                      ? `${cat.mana_cost != null ? `${cat.mana_cost}m · ` : ""}${cat.faction || ""}${cat.card_type ? ` · ${cat.card_type}` : ""}`
                                      : `${c.games_drawn} drawn`}
                                  </div>
                                </td>
                                <td class="mono">
                                  ${formatPct(c.inclusion_rate)}
                                  <div class="muted" style="font-size:10.5px;">${c.games_in_deck}g</div>
                                </td>
                                <td
                                  class="mono"
                                  style=${wrStyle(c.gih_wr)}
                                  title=${c.gih_ci_low != null ? `95% CI: ${formatPct(c.gih_ci_low)} – ${formatPct(c.gih_ci_high)} (${c.games_drawn} drawn)` : ""}
                                >
                                  ${formatPct(c.gih_wr)}
                                </td>
                                <td class="mono" style=${signedStyle(c.drawn_wpa)}>
                                  ${formatSignedPct(c.drawn_wpa)}
                                </td>
                                ${hasCriticData
                                  ? html`<td class="mono" style=${signedStyle(c.critic_play_wpa, 0.01)}>
                                      ${formatSignedPct(c.critic_play_wpa, 2)}
                                    </td>`
                                  : null}
                                <td class="mono" style=${signedStyle(c.isolated_impact, 0.015)}>
                                  ${formatSignedPct(c.isolated_impact)}
                                </td>
                                <td class="mono" style=${signedStyle(c.meta_war, 0.5)}>
                                  ${c.meta_war != null ? `${c.meta_war > 0 ? "+" : ""}${c.meta_war.toFixed(2)}` : "—"}
                                </td>
                                <td>
                                  ${c.top_synergy
                                    ? html`
                                        <span class="synergy-chip">
                                          <span>${c.top_synergy.partner}</span>
                                          <span class="mono" style="color:var(--ok);">${formatSignedPct(c.top_synergy.lift)}</span>
                                        </span>
                                      `
                                    : html`<span class="muted">—</span>`}
                                  ${c.synergy_reliance >= 0.025
                                    ? html`<div class="muted" style="font-size:10px;">Combo-reliant</div>`
                                    : null}
                                </td>
                                <td><${LaneMiniBars} lanePref=${c.lane_preference} /></td>
                                <td class="mono muted">
                                  ${c.avg_turn_played != null ? `T${c.avg_turn_played.toFixed(1)}` : "—"}
                                </td>
                              </tr>
                            `;
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                `}
          `
        : null}

      ${subTab === "synergies"
        ? html`
            <div class="card stack">
              <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                <div class="tabs" style="margin-bottom:0;">
                  ${[
                    ["played", "Both Played"],
                    ["same_lane", "Same Lane (4-Lane Combos)"],
                    ["deck", "Deck Co-Inclusion"],
                    ["counters", "Opponent Counters"],
                  ].map(
                    ([k, label]) => html`
                      <button
                        class=${`tab-btn ${synLevel === k ? "active" : ""}`}
                        onClick=${() => setSynLevel(k)}
                      >
                        ${label}
                      </button>
                    `
                  )}
                </div>
                <div style="min-width:180px;">
                  <input
                    class="input"
                    placeholder="Filter pair by card…"
                    value=${synSearch}
                    onInput=${(e) => setSynSearch(e.target.value)}
                  />
                </div>
              </div>

              <div class="grid-2">
                <div class="stack">
                  <strong>
                    ${synLevel === "counters" ? "Strongest Favorable Matchups" : "Top Positive Synergies"}
                  </strong>
                  ${(() => {
                    const q = synSearch.trim().toLowerCase();
                    const rows = (synData.positive || []).filter((r) => {
                      if (!q) return true;
                      const a = (r.card_a || r.card || "").toLowerCase();
                      const b = (r.card_b || r.opponent || "").toLowerCase();
                      return a.includes(q) || b.includes(q);
                    });
                    if (rows.length === 0) return html`<div class="muted">No qualifying pairs yet.</div>`;
                    return html`
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>${synLevel === "counters" ? "Card → Opponent" : "Card Pair"}</th>
                              <th>Games</th>
                              <th>WR</th>
                              <th>Exp WR</th>
                              <th>Lift</th>
                              <th>PMI</th>
                            </tr>
                          </thead>
                          <tbody>
                            ${rows.slice(0, 25).map((r) => {
                              const a = r.card_a || r.card;
                              const b = r.card_b || r.opponent;
                              return html`
                                <tr>
                                  <td>
                                    <a href="javascript:void(0)" onClick=${() => openDossier(a)}>${a}</a>
                                    <span class="muted"> ${synLevel === "counters" ? "vs" : "+"} </span>
                                    <a href="javascript:void(0)" onClick=${() => openDossier(b)}>${b}</a>
                                  </td>
                                  <td class="mono">${r.games}</td>
                                  <td class="mono">${formatPct(r.win_rate)}</td>
                                  <td class="mono muted">${formatPct(r.expected_wr)}</td>
                                  <td class="mono" style=${signedStyle(r.lift, 0.01)}>${formatSignedPct(r.lift)}</td>
                                  <td class="mono muted">${r.pmi != null ? r.pmi.toFixed(2) : "—"}</td>
                                </tr>
                              `;
                            })}
                          </tbody>
                        </table>
                      </div>
                    `;
                  })()}
                </div>

                <div class="stack">
                  <strong>
                    ${synLevel === "counters" ? "Hardest Opponent Counters" : "Anti-Synergies (Negative Lift)"}
                  </strong>
                  ${(() => {
                    const q = synSearch.trim().toLowerCase();
                    const rows = (synData.negative || []).filter((r) => {
                      if (!q) return true;
                      const a = (r.card_a || r.card || "").toLowerCase();
                      const b = (r.card_b || r.opponent || "").toLowerCase();
                      return a.includes(q) || b.includes(q);
                    });
                    if (rows.length === 0) return html`<div class="muted">No qualifying anti-synergies/counters yet.</div>`;
                    return html`
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>${synLevel === "counters" ? "Card vs Counter" : "Card Pair"}</th>
                              <th>Games</th>
                              <th>WR</th>
                              <th>Exp WR</th>
                              <th>Lift</th>
                            </tr>
                          </thead>
                          <tbody>
                            ${rows.slice(0, 25).map((r) => {
                              const a = r.card_a || r.card;
                              const b = r.card_b || r.opponent;
                              return html`
                                <tr>
                                  <td>
                                    <a href="javascript:void(0)" onClick=${() => openDossier(a)}>${a}</a>
                                    <span class="muted"> ${synLevel === "counters" ? "vs" : "+"} </span>
                                    <a href="javascript:void(0)" onClick=${() => openDossier(b)}>${b}</a>
                                  </td>
                                  <td class="mono">${r.games}</td>
                                  <td class="mono">${formatPct(r.win_rate)}</td>
                                  <td class="mono muted">${formatPct(r.expected_wr)}</td>
                                  <td class="mono" style=${signedStyle(r.lift, 0.01)}>${formatSignedPct(r.lift)}</td>
                                </tr>
                              `;
                            })}
                          </tbody>
                        </table>
                      </div>
                    `;
                  })()}
                </div>
              </div>
            </div>
          `
        : null}

      ${subTab === "patch_diff"
        ? html`
            <div class="stack">
              <div class="card stack">
                <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                  <div>
                    <strong>Patch-to-Patch Balance Diff</strong>
                    <div class="muted" style="font-size:11.5px;">Compare catalog stat changes alongside WPA, WR, Impact, and Meta WAR shifts</div>
                  </div>
                  <div class="row">
                    <select class="select" style="width:auto;" value=${patchA} onChange=${(e) => setPatchA(e.target.value)}>
                      ${(filtersMeta?.manifests || []).map(
                        (m) => html`<option value=${m.manifest_hash}>From: ${m.patch_label || m.manifest_hash}</option>`
                      )}
                    </select>
                    <span class="muted">→</span>
                    <select class="select" style="width:auto;" value=${patchB} onChange=${(e) => setPatchB(e.target.value)}>
                      ${(filtersMeta?.manifests || []).map(
                        (m) => html`<option value=${m.manifest_hash}>To: ${m.patch_label || m.manifest_hash}</option>`
                      )}
                    </select>
                  </div>
                </div>

                ${!patchDiff || (patchDiff.diffs || []).length === 0
                  ? html`<div class="muted">Log games across at least one manifest_hash to compare patches.</div>`
                  : html`
                      <div class="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Card</th>
                              <th>Catalog Changes</th>
                              <th>Δ Incl</th>
                              <th>GIH WR (A → B)</th>
                              <th>Δ Drawn WPA</th>
                              <th>Δ Impact (β)</th>
                              <th>Δ Meta WAR</th>
                            </tr>
                          </thead>
                          <tbody>
                            ${patchDiff.diffs.map((d) => html`
                              <tr style="cursor:pointer;" onClick=${() => openDossier(d.card_key)}>
                                <td style="font-weight:700;">${d.display_name}</td>
                                <td>
                                  ${(d.catalog_changes || []).length > 0
                                    ? html`<div class="row" style="gap:4px;">
                                        ${d.catalog_changes.map(
                                          (ch) => html`<span class="pill accent">
                                            ${ch.field}: ${String(ch.old ?? "—")}→${String(ch.new ?? "—")}
                                          </span>`
                                        )}
                                      </div>`
                                    : html`<span class="muted">—</span>`}
                                </td>
                                <td class="mono" style=${signedStyle(d.delta_inclusion, 0.02)}>
                                  ${formatSignedPct(d.delta_inclusion)}
                                </td>
                                <td class="mono">
                                  ${formatPct(d.gih_wr_a)} → ${formatPct(d.gih_wr_b)}
                                  <span style=${`margin-left:5px; ${signedStyle(d.delta_gih_wr, 0.015)}`}>
                                    (${formatSignedPct(d.delta_gih_wr)})
                                  </span>
                                </td>
                                <td class="mono" style=${signedStyle(d.delta_drawn_wpa, 0.015)}>
                                  ${formatSignedPct(d.delta_drawn_wpa)}
                                </td>
                                <td class="mono" style=${signedStyle(d.delta_impact, 0.015)}>
                                  ${formatSignedPct(d.delta_impact)}
                                </td>
                                <td class="mono" style=${signedStyle(d.delta_meta_war, 0.4)}>
                                  ${d.delta_meta_war != null ? `${d.delta_meta_war > 0 ? "+" : ""}${d.delta_meta_war.toFixed(2)}` : "—"}
                                </td>
                              </tr>
                            `)}
                          </tbody>
                        </table>
                      </div>
                    `}
              </div>

              <div class="card stack">
                <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                  <strong>Card Meta Trends Over Time</strong>
                  <div class="tabs" style="margin-bottom:0;">
                    ${[
                      ["drawn_wpa", "Drawn WPA"],
                      ["gih_wr", "GIH WR"],
                      ["inclusion_rate", "Inclusion %"],
                      ["played_wr", "Played WR"],
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
                <div class="row">
                  ${(overview.cards || []).slice(0, 14).map((c) => {
                    const active = trendCards.includes(c.card_key);
                    return html`
                      <button
                        class=${`tab-btn ${active ? "active" : ""}`}
                        onClick=${() =>
                          setTrendCards((prev) =>
                            prev.includes(c.card_key)
                              ? prev.filter((x) => x !== c.card_key)
                              : [...prev, c.card_key].slice(0, 8)
                          )}
                      >
                        ${c.display_name || c.card_key}
                      </button>
                    `;
                  })}
                </div>
                ${trendChart.runList.length > 0
                  ? html`
                      <${UPlotChart}
                        title=${`${trendMetric.replace("_", " ")} Over Time`}
                        seriesMap=${trendChart.seriesMap}
                        runList=${trendChart.runList}
                        xAxis="date"
                        smoothing=${0}
                        theme=${theme}
                        height=${240}
                      />
                    `
                  : html`<div class="muted">Not enough dated buckets yet for time trends.</div>`}
              </div>
            </div>
          `
        : null}
    </div>
  `;
}
