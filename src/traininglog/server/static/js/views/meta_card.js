import {
  apiFetch,
  formatPct,
  formatSignedPct,
} from "../api.js";
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

export function MetaCardView({
  cardKey,
  queryStr = "",
  theme,
  onBack,
  onOpenCard,
}) {
  const baseParams = useMemo(() => new URLSearchParams(queryStr || ""), [queryStr]);
  const [manifestOverride, setManifestOverride] = useState(
    baseParams.get("manifest_hash") || ""
  );
  const [filtersMeta, setFiltersMeta] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setManifestOverride(baseParams.get("manifest_hash") || "");
  }, [cardKey, queryStr]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const proj = baseParams.get("project");
        const fqs = proj ? `?project=${encodeURIComponent(proj)}` : "";
        const qs = new URLSearchParams(baseParams.toString());
        qs.delete("tab");
        qs.delete("q");
        if (manifestOverride) qs.set("manifest_hash", manifestOverride);

        const [fRes, dRes] = await Promise.all([
          apiFetch(`/api/meta/filters${fqs}`),
          apiFetch(`/api/meta/cards/${encodeURIComponent(cardKey)}?${qs.toString()}`),
        ]);
        if (!cancelled) {
          setFiltersMeta(fRes);
          setDetail(dRes);
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Failed loading card dossier");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [cardKey, queryStr, manifestOverride]);

  const turnChart = useMemo(() => {
    const tc = detail?.turn_curve || [];
    if (tc.length === 0) return { runList: [], seriesMap: {} };
    const xs = tc.map((t) => Number(t.turn));
    return {
      runList: [
        { id: "wr", name: "Win Rate %" },
        { id: "share", name: "Play Share %" },
      ],
      seriesMap: {
        wr: {
          x: xs,
          y: tc.map((t) => (t.win_rate != null ? t.win_rate * 100 : null)),
        },
        share: {
          x: xs,
          y: tc.map((t) => (t.share != null ? t.share * 100 : 0)),
        },
      },
    };
  }, [detail]);

  const trendChart = useMemo(() => {
    const tr = detail?.trend || [];
    if (tr.length < 2) return { runList: [], seriesMap: {} };
    const xs = tr.map((p) => Math.floor(Number(p.ts_ms) / 1000));
    return {
      runList: [
        { id: "gih", name: "GIH WR %" },
        { id: "wpa", name: "Drawn WPA %" },
        { id: "inc", name: "Inclusion %" },
      ],
      seriesMap: {
        gih: { x: xs, y: tr.map((p) => (p.gih_wr != null ? p.gih_wr * 100 : null)) },
        wpa: { x: xs, y: tr.map((p) => (p.drawn_wpa != null ? p.drawn_wpa * 100 : null)) },
        inc: { x: xs, y: tr.map((p) => p.inclusion_rate * 100) },
      },
    };
  }, [detail]);

  const card = detail?.card;
  const cat = card?.catalog;
  const funnel = detail?.funnel;
  const lanePref = card?.lane_preference;
  const seat = detail?.seat_split;

  return html`
    <div class="stack">
      <div class="space-between" style="flex-wrap:wrap; gap:8px;">
        <div class="row">
          <button class="btn sm" onClick=${() => onBack(queryStr)}>← Back to Meta</button>
          <div>
            <h2 style="margin:0; font-size:18px;">
              ${card ? card.display_name || card.card_key : cardKey}
            </h2>
            ${card && card.display_name !== card.card_key
              ? html`<div class="mono muted" style="font-size:11px;">${card.card_key}</div>`
              : null}
          </div>
        </div>
        ${(filtersMeta?.manifests || []).length > 0
          ? html`
              <select
                class="select"
                style="width:auto;"
                value=${manifestOverride}
                onChange=${(e) => setManifestOverride(e.target.value)}
              >
                <option value="all">All Patches</option>
                ${(filtersMeta.manifests || []).map(
                  (m) => html`<option value=${m.manifest_hash}>
                    ${m.patch_label ? `${m.patch_label} (${m.manifest_hash})` : m.manifest_hash}
                  </option>`
                )}
              </select>
            `
          : null}
      </div>

      ${loading
        ? html`<div class="card muted">Loading card meta dossier…</div>`
        : error
          ? html`<div class="card pill critical">${error}</div>`
          : html`
              ${cat
                ? html`
                    <div class="card stack" style="gap:6px;">
                      <div class="row">
                        ${cat.mana_cost != null ? html`<span class="pill accent">${cat.mana_cost} Mana</span>` : null}
                        ${cat.faction ? html`<span class="pill">${cat.faction}</span>` : null}
                        ${cat.card_type ? html`<span class="pill">${cat.card_type}</span>` : null}
                        ${cat.rarity ? html`<span class="pill">${cat.rarity}</span>` : null}
                        ${cat.attack != null || cat.health != null
                          ? html`<span class="pill ok mono">${cat.attack ?? "—"} / ${cat.health ?? "—"}</span>`
                          : null}
                        ${(cat.tags || []).map((t) => html`<span class="pill">#${t}</span>`)}
                      </div>
                      ${cat.text ? html`<div style="font-size:13px;">${cat.text}</div>` : null}
                    </div>
                  `
                : null}

              <div class="metrics-grid">
                <div class="stat-box card">
                  <div class="stat-label">GIH Win Rate</div>
                  <div class="stat-val mono">${formatPct(card.gih_wr)}</div>
                  <div class="muted mono" style="font-size:10.5px;">
                    ${card.gih_ci_low != null
                      ? `95% CI: ${formatPct(card.gih_ci_low)}–${formatPct(card.gih_ci_high)}`
                      : `${card.games_drawn} drawn`}
                  </div>
                </div>
                <div class="stat-box card">
                  <div class="stat-label">Drawn WPA (IWD)</div>
                  <div class="stat-val mono" style=${signedStyle(card.drawn_wpa)}>
                    ${formatSignedPct(card.drawn_wpa)}
                  </div>
                  <div class="muted mono" style="font-size:10.5px;">
                    Raw: ${formatSignedPct(card.drawn_wpa_raw)}
                  </div>
                </div>
                <div class="stat-box card">
                  <div class="stat-label">Played Lift</div>
                  <div class="stat-val mono" style=${signedStyle(card.played_lift)}>
                    ${formatSignedPct(card.played_lift)}
                  </div>
                  <div class="muted mono" style="font-size:10.5px;">
                    Play vs Hold: ${formatSignedPct(card.play_vs_hold_delta)}
                  </div>
                </div>
                <div class="stat-box card">
                  <div class="stat-label">Critic Turn WPA</div>
                  <div class="stat-val mono" style=${signedStyle(card.critic_play_wpa, 0.01)}>
                    ${formatSignedPct(card.critic_play_wpa, 2)}
                  </div>
                  <div class="muted mono" style="font-size:10.5px;">
                    Draw: ${formatSignedPct(card.critic_draw_wpa, 2)} (${detail.kpis?.critic_perspective || "n/a"})
                  </div>
                </div>
                <div class="stat-box card">
                  <div class="stat-label">Est. Impact (β)</div>
                  <div class="stat-val mono" style=${signedStyle(card.isolated_impact)}>
                    ${formatSignedPct(card.isolated_impact)}
                  </div>
                  <div class="muted mono" style="font-size:10.5px;">
                    Rating & deckmate adjusted
                  </div>
                </div>
                <div class="stat-box card">
                  <div class="stat-label">Meta WAR</div>
                  <div class="stat-val mono" style=${signedStyle(card.meta_war, 0.5)}>
                    ${card.meta_war != null ? `${card.meta_war > 0 ? "+" : ""}${card.meta_war.toFixed(2)}` : "—"}
                  </div>
                  <div class="muted mono" style="font-size:10.5px;">
                    Incl: ${formatPct(card.inclusion_rate)} · ${card.avg_copies.toFixed(2)}x
                  </div>
                </div>
              </div>

              <div class="grid-2">
                <div class="card stack">
                  <strong>Draw & Play Funnel</strong>
                  <div class="funnel-list">
                    ${[
                      ["In Deck", funnel.in_deck, funnel.in_deck_wr, 1.0],
                      ["Opening Hand", funnel.opening_hand, funnel.oh_wr, funnel.in_deck ? funnel.opening_hand / funnel.in_deck : 0],
                      ["Drawn (GIH)", funnel.drawn, funnel.gih_wr, funnel.in_deck ? funnel.drawn / funnel.in_deck : 0],
                      ["Played", funnel.played, funnel.played_wr, funnel.in_deck ? funnel.played / funnel.in_deck : 0],
                      ["Not Seen (GNS)", funnel.not_seen, funnel.gns_wr, funnel.in_deck ? funnel.not_seen / funnel.in_deck : 0],
                    ].map(
                      ([lbl, cnt, wr, ratio]) => html`
                        <div class="funnel-row">
                          <span>${lbl}</span>
                          <div class="prob-bar-track">
                            <div
                              class="prob-bar-fill"
                              style=${`width:${Math.max(2, Math.min(100, ratio * 100))}%`}
                            ></div>
                          </div>
                          <span class="mono">${cnt}g · ${formatPct(wr)}</span>
                        </div>
                      `
                    )}
                  </div>
                  ${funnel.mulligan
                    ? html`
                        <div class="muted mono" style="font-size:11.5px;">
                          Mulligan Keep: ${formatPct(funnel.mulligan.keep_rate)} (${funnel.mulligan.kept}/${funnel.mulligan.offered}) ·
                          Kept WR ${formatPct(funnel.mulligan.kept_wr)} vs Tossed ${formatPct(funnel.mulligan.tossed_wr)}
                        </div>
                      `
                    : null}
                </div>

                <div class="card stack">
                  <strong>P1 vs P2 Seat Split (When Drawn)</strong>
                  <div class="grid-2">
                    <div class="stat-box">
                      <div class="stat-label">Player 1 (First)</div>
                      <div class="stat-val mono">${formatPct(seat.p1_wr)}</div>
                      <div class="muted mono" style="font-size:11px;">${seat.p1_games} games</div>
                    </div>
                    <div class="stat-box">
                      <div class="stat-label">Player 2 (Second)</div>
                      <div class="stat-val mono">${formatPct(seat.p2_wr)}</div>
                      <div class="muted mono" style="font-size:11px;">${seat.p2_games} games</div>
                    </div>
                  </div>
                  <div class="muted mono" style="font-size:12px;">
                    Seat Delta (P1 − P2): <span style=${signedStyle(seat.seat_delta)}>${formatSignedPct(seat.seat_delta)}</span> ·
                    Draw→Play Rate: ${formatPct(card.draw_to_play_rate)}
                  </div>
                  <div class="muted mono" style="font-size:12px;">
                    Avg First Turn: ${card.avg_first_turn != null ? `T${card.avg_first_turn.toFixed(1)}` : "—"} ·
                    Avg Rounds When Played: ${detail.avg_rounds_when_played != null ? detail.avg_rounds_when_played.toFixed(1) : "—"}
                  </div>
                </div>
              </div>

              <div class="card stack">
                <div class="space-between" style="flex-wrap:wrap; gap:8px;">
                  <div>
                    <strong>4-Lane Preference & Optimal Placement</strong>
                    <div class="muted" style="font-size:11.5px;">
                      Lane Concentration: ${((lanePref?.concentration || 0) * 100).toFixed(0)}% ·
                      Lane Plays: ${lanePref?.lane_plays || 0}
                      ${lanePref?.global_plays > 0 ? ` · Global (non-lane) Plays: ${lanePref.global_plays}` : ""}
                    </div>
                  </div>
                  ${lanePref?.favored_lane != null
                    ? html`<span class="pill accent">Most Played: L${lanePref.favored_lane}</span>`
                    : null}
                </div>

                ${lanePref?.has_misplay_gap
                  ? html`
                      <div class="misplay-banner">
                        <strong>Optimal Lane Gap Detected:</strong>
                        Agents/players place this card most often in <strong>L${lanePref.favored_lane}</strong>,
                        but it achieves <strong>${formatSignedPct(lanePref.misplay_gap)}</strong> higher shrunk WR in
                        <strong>L${lanePref.best_lane}</strong>.
                      </div>
                    `
                  : null}

                ${lanePref?.lane_plays > 0
                  ? html`
                      <div class="lane-grid">
                        ${lanePref.lanes.map((ln) => {
                          const isFav = ln.lane === lanePref.favored_lane;
                          const isBest = ln.lane === lanePref.best_lane;
                          return html`
                            <div class=${`lane-box ${isFav ? "fav" : ""} ${isBest ? "best" : ""}`}>
                              <div class="space-between">
                                <strong>${ln.label}</strong>
                                <span class="mono" style="font-size:11px;">${formatPct(ln.share)}</span>
                              </div>
                              <div class="progress-track" style="margin:6px 0;">
                                <div class="progress-fill" style=${`width:${Math.round((ln.share || 0) * 100)}%`}></div>
                              </div>
                              <div class="mono" style="font-size:12px;">
                                WR: ${formatPct(ln.win_rate)}
                                <span style=${`margin-left:4px; ${signedStyle(ln.wr_delta, 0.01)}`}>
                                  (${formatSignedPct(ln.wr_delta)})
                                </span>
                              </div>
                              <div class="muted mono" style="font-size:11px;">
                                ${ln.plays} plays${ln.critic_wpa != null ? ` · Critic ${formatSignedPct(ln.critic_wpa, 2)}` : ""}
                              </div>
                            </div>
                          `;
                        })}
                      </div>
                    `
                  : html`<div class="muted">No lane-targeted plays recorded (global/non-lane card).</div>`}

                ${(detail.lane_contexts || []).length > 0
                  ? html`
                      <div class="table-wrap" style="margin-top:6px;">
                        <table>
                          <thead>
                            <tr>
                              <th>Board Context</th>
                              <th>Plays</th>
                              <th>Share</th>
                              <th>Win Rate</th>
                              <th>Critic WPA</th>
                            </tr>
                          </thead>
                          <tbody>
                            ${detail.lane_contexts.map(
                              (ctx) => html`
                                <tr>
                                  <td style="font-weight:600;">${ctx.context}</td>
                                  <td class="mono">${ctx.plays}</td>
                                  <td class="mono">${formatPct(ctx.share)}</td>
                                  <td class="mono">${formatPct(ctx.win_rate)}</td>
                                  <td class="mono" style=${signedStyle(ctx.critic_wpa, 0.01)}>
                                    ${formatSignedPct(ctx.critic_wpa, 2)}
                                  </td>
                                </tr>
                              `
                            )}
                          </tbody>
                        </table>
                      </div>
                    `
                  : null}
              </div>

              <div class="grid-2">
                <div class="card stack">
                  <strong>High Synergies (Both Played & Same-Lane)</strong>
                  ${(detail.synergies?.played || []).length === 0 && (detail.synergies?.deck || []).length === 0
                    ? html`<div class="muted">No statistically significant synergies yet.</div>`
                    : html`
                        <div class="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Partner</th>
                                <th>Played Lift</th>
                                <th>Same-Lane</th>
                                <th>Games</th>
                              </tr>
                            </thead>
                            <tbody>
                              ${(detail.synergies?.played || detail.synergies?.deck || []).map((s) => {
                                const laneMatch = (detail.synergies?.same_lane || []).find(
                                  (l) => l.partner === s.partner
                                );
                                return html`
                                  <tr>
                                    <td>
                                      <a href="javascript:void(0)" onClick=${() => onOpenCard(s.partner)}>${s.partner}</a>
                                    </td>
                                    <td class="mono" style=${signedStyle(s.lift, 0.01)}>${formatSignedPct(s.lift)}</td>
                                    <td class="mono" style=${signedStyle(laneMatch?.lift, 0.01)}>
                                      ${laneMatch ? formatSignedPct(laneMatch.lift) : "—"}
                                    </td>
                                    <td class="mono muted">${s.games}</td>
                                  </tr>
                                `;
                              })}
                            </tbody>
                          </table>
                        </div>
                      `}
                </div>

                <div class="card stack">
                  <strong>Counters & Anti-Synergies</strong>
                  ${(detail.counters?.worst_against || []).length === 0 &&
                  (detail.counters?.best_against || []).length === 0
                    ? html`<div class="muted">Not enough 1v1 opponent matchup samples yet.</div>`
                    : html`
                        <div class="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Opponent Card</th>
                                <th>Type</th>
                                <th>WR</th>
                                <th>Δ vs Exp</th>
                                <th>Games</th>
                              </tr>
                            </thead>
                            <tbody>
                              ${[
                                ...(detail.counters?.worst_against || []).slice(0, 5).map((c) => ({ ...c, kind: "Countered By" })),
                                ...(detail.counters?.best_against || []).slice(0, 5).map((c) => ({ ...c, kind: "Strong Into" })),
                              ].map(
                                (c) => html`
                                  <tr>
                                    <td>
                                      <a href="javascript:void(0)" onClick=${() => onOpenCard(c.opponent)}>${c.opponent}</a>
                                    </td>
                                    <td>
                                      <span class=${`pill ${c.lift < 0 ? "danger" : "ok"}`}>${c.kind}</span>
                                    </td>
                                    <td class="mono">${formatPct(c.win_rate)}</td>
                                    <td class="mono" style=${signedStyle(c.lift, 0.01)}>${formatSignedPct(c.lift)}</td>
                                    <td class="mono muted">${c.games}</td>
                                  </tr>
                                `
                              )}
                            </tbody>
                          </table>
                        </div>
                      `}
                </div>
              </div>

              <div class="grid-2">
                <div class="card stack">
                  <strong>Turn-by-Turn Timing & Tempo Curve</strong>
                  ${turnChart.runList.length > 0
                    ? html`
                        <${UPlotChart}
                          title="Play Share % & Win Rate % by Turn Played"
                          seriesMap=${turnChart.seriesMap}
                          runList=${turnChart.runList}
                          xAxis="turn"
                          smoothing=${0}
                          theme=${theme}
                          height=${210}
                        />
                      `
                    : html`<div class="muted">No turn-level play timing recorded.</div>`}
                </div>

                <div class="card stack">
                  <strong>Top Decks Featuring This Card</strong>
                  ${(detail.top_decks || []).length === 0
                    ? html`<div class="muted">No deck names recorded.</div>`
                    : html`
                        <div class="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Deck</th>
                                <th>Games</th>
                                <th>Share</th>
                                <th>Win Rate</th>
                              </tr>
                            </thead>
                            <tbody>
                              ${detail.top_decks.map(
                                (d) => html`
                                  <tr>
                                    <td style="font-weight:600;">${d.deck_name}</td>
                                    <td class="mono">${d.games}</td>
                                    <td class="mono">${formatPct(d.share)}</td>
                                    <td class="mono">${formatPct(d.win_rate)}</td>
                                  </tr>
                                `
                              )}
                            </tbody>
                          </table>
                        </div>
                      `}
                </div>
              </div>

              ${trendChart.runList.length > 0
                ? html`
                    <div class="card stack">
                      <strong>Historical Trend</strong>
                      <${UPlotChart}
                        title="GIH WR %, Drawn WPA %, and Inclusion % Over Time"
                        seriesMap=${trendChart.seriesMap}
                        runList=${trendChart.runList}
                        xAxis="date"
                        smoothing=${0}
                        theme=${theme}
                        height=${210}
                      />
                    </div>
                  `
                : null}
            `}
    </div>
  `;
}
