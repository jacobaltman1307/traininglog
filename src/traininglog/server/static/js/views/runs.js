import {
  apiFetch,
  formatNumber,
  formatPct,
  formatStep,
  formatTimeAgo,
} from "../api.js";
import { html, useMemo, useState } from "../../vendor/ui.mjs";

export function RunsView({
  runs,
  projects,
  selectedIds,
  onToggleSelect,
  onSetSelectedIds,
  onOpenRun,
  onOpenCompare,
  onRefresh,
}) {
  const [projectFilter, setProjectFilter] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [archFilter, setArchFilter] = useState("");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("started_at");
  const [sortDir, setSortDir] = useState("desc");

  const arches = useMemo(() => {
    const s = new Set();
    for (const r of runs) if (r.arch) s.add(r.arch);
    return Array.from(s).sort();
  }, [runs]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = runs.filter((r) => {
      if (projectFilter && r.project_name !== projectFilter) return false;
      if (stateFilter && r.state !== stateFilter) return false;
      if (archFilter && r.arch !== archFilter) return false;
      if (
        q &&
        !r.name.toLowerCase().includes(q) &&
        !r.id.toLowerCase().includes(q) &&
        !(r.notes || "").toLowerCase().includes(q) &&
        !(r.tags || []).some((t) => t.toLowerCase().includes(q))
      ) {
        return false;
      }
      return true;
    });

    list.sort((a, b) => {
      let av;
      let bv;
      if (sortBy === "step") {
        av = a.current_step || 0;
        bv = b.current_step || 0;
      } else if (sortBy === "winrate") {
        av = (a.summary || {})["game/win_rate_overall"] ?? -1;
        bv = (b.summary || {})["game/win_rate_overall"] ?? -1;
      } else if (sortBy === "sps") {
        av = a.current_sps || 0;
        bv = b.current_sps || 0;
      } else {
        av = a.started_at || 0;
        bv = b.started_at || 0;
      }
      return sortDir === "asc" ? av - bv : bv - av;
    });

    return list;
  }, [runs, projectFilter, stateFilter, archFilter, search, sortBy, sortDir]);

  function toggleSort(col) {
    if (sortBy === col) {
      setSortDir(sortDir === "desc" ? "asc" : "desc");
    } else {
      setSortBy(col);
      setSortDir("desc");
    }
  }

  async function deleteRun(run) {
    if (!window.confirm(`Delete run '${run.name}' and all its logged series?`)) {
      return;
    }
    try {
      await apiFetch(`/api/runs/${run.id}`, { method: "DELETE" });
      onRefresh();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    }
  }

  return html`
    <div class="stack">
      <div class="space-between" style="flex-wrap:wrap;">
        <div>
          <h2 style="margin:0; font-size:17px;">Experiments (${filtered.length})</h2>
          <div class="muted" style="font-size:12px;">
            Select runs to overlay curves & inspect hyperparameter diffs
          </div>
        </div>
        <div class="row">
          <button
            class=${`btn sm ${selectedIds.length > 0 ? "primary" : ""}`}
            onClick=${() => onOpenCompare(selectedIds)}
          >
            Compare${selectedIds.length > 0 ? ` (${selectedIds.length})` : ""}
          </button>
          ${selectedIds.length > 0
            ? html`<button class="btn sm" onClick=${() => onSetSelectedIds([])}>
                Clear
              </button>`
            : null}
        </div>
      </div>

      <div class="card">
        <div class="row">
          <div style="flex:2; min-width:170px;">
            <input
              class="input"
              placeholder="Search name, id, tag, notes…"
              value=${search}
              onInput=${(e) => setSearch(e.target.value)}
            />
          </div>
          <div style="flex:1; min-width:130px;">
            <select
              class="select"
              value=${projectFilter}
              onChange=${(e) => setProjectFilter(e.target.value)}
            >
              <option value="">All Projects</option>
              ${projects.map(
                (p) => html`<option value=${p.name}>${p.name} (${p.total_runs})</option>`
              )}
            </select>
          </div>
          <div style="flex:1; min-width:110px;">
            <select
              class="select"
              value=${stateFilter}
              onChange=${(e) => setStateFilter(e.target.value)}
            >
              <option value="">All States</option>
              <option value="running">running</option>
              <option value="stalled">stalled</option>
              <option value="finished">finished</option>
              <option value="crashed">crashed</option>
              <option value="killed">killed</option>
            </select>
          </div>
          <div style="flex:1; min-width:110px;">
            <select
              class="select"
              value=${archFilter}
              onChange=${(e) => setArchFilter(e.target.value)}
            >
              <option value="">All Archs</option>
              ${arches.map((a) => html`<option value=${a}>${a}</option>`)}
            </select>
          </div>
        </div>
      </div>

      ${filtered.length === 0
        ? html`<div class="card empty-state">No matching runs found.</div>`
        : html`
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th style="width:36px;">
                      <input
                        type="checkbox"
                        checked=${filtered.length > 0 &&
                        filtered.every((r) => selectedIds.includes(r.id))}
                        onChange=${(e) => {
                          if (e.target.checked) {
                            onSetSelectedIds(filtered.map((r) => r.id));
                          } else {
                            onSetSelectedIds([]);
                          }
                        }}
                      />
                    </th>
                    <th>Run</th>
                    <th>State</th>
                    <th>Arch</th>
                    <th style="cursor:pointer;" onClick=${() => toggleSort("step")}>
                      Step
                    </th>
                    <th style="cursor:pointer;" onClick=${() => toggleSort("winrate")}>
                      Win %
                    </th>
                    <th style="cursor:pointer;" onClick=${() => toggleSort("sps")}>
                      SPS
                    </th>
                    <th style="cursor:pointer;" onClick=${() => toggleSort("started_at")}>
                      Started
                    </th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  ${filtered.map((r) => {
                    const checked = selectedIds.includes(r.id);
                    const sum = r.summary || {};
                    const win = sum["game/win_rate_overall"];
                    return html`
                      <tr key=${r.id}>
                        <td>
                          <input
                            type="checkbox"
                            checked=${checked}
                            onChange=${() => onToggleSelect(r.id)}
                          />
                        </td>
                        <td style="cursor:pointer;" onClick=${() => onOpenRun(r.id)}>
                          <div style="font-weight:600; color:var(--accent);">${r.name}</div>
                          <div class="muted" style="font-size:11px;">
                            ${r.project_name}
                            ${r.group_name ? ` · ${r.group_name}` : ""}
                            ${(r.tags || []).slice(0, 3).map((t) => html` · #${t}`)}
                          </div>
                        </td>
                        <td>
                          <span class=${`pill ${r.state}`}>${r.state}</span>
                          ${r.open_anomaly_count > 0
                            ? html` <span class="pill critical">${r.open_anomaly_count}!</span>`
                            : null}
                        </td>
                        <td>${r.arch ? html`<span class="pill">${r.arch}</span>` : "—"}</td>
                        <td class="mono">${formatStep(r.current_step)}</td>
                        <td class="mono">${win != null ? formatPct(win) : "—"}</td>
                        <td class="mono">${formatNumber(r.current_sps)}</td>
                        <td class="muted">${formatTimeAgo(r.started_at)}</td>
                        <td>
                          <button
                            class="btn sm danger"
                            title="Delete run"
                            onClick=${() => deleteRun(r)}
                          >
                            Del
                          </button>
                        </td>
                      </tr>
                    `;
                  })}
                </tbody>
              </table>
            </div>
          `}
    </div>
  `;
}
