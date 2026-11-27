import {
  apiFetch,
  connectSSE,
  getStoredToken,
  setStoredToken,
  setUnauthorizedHandler,
} from "./api.js";
import { ActiveView } from "./views/active.js";
import { AlertsView } from "./views/alerts.js";
import { CardsView } from "./views/cards.js";
import { CompareView } from "./views/compare.js";
import { LeagueView } from "./views/league.js";
import { MetaCardView } from "./views/meta_card.js";
import { MetaView } from "./views/meta.js";
import { ReplaysView } from "./views/replays.js";
import { RunDetailView } from "./views/run_detail.js";
import { RunsView } from "./views/runs.js";
import { html, render, useEffect, useMemo, useState } from "../vendor/ui.mjs";

function parseHash() {
  const raw = (window.location.hash || "#/active").slice(1);
  const [pathPart, queryPart] = raw.split("?");
  const parts = pathPart.split("/").filter(Boolean);
  const params = new URLSearchParams(queryPart || "");
  if (parts[0] === "runs" && parts[1]) {
    return { route: "run_detail", runId: parts[1], compareIds: [] };
  }
  if (parts[0] === "runs") {
    return { route: "runs", runId: null, compareIds: [] };
  }
  if (parts[0] === "compare") {
    const ids = (params.get("ids") || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    return { route: "compare", runId: null, compareIds: ids };
  }
  if (parts[0] === "league") {
    return { route: "league", runId: null, compareIds: [] };
  }
  if (parts[0] === "meta" && parts[1] === "card" && parts[2]) {
    return {
      route: "meta_card",
      cardKey: decodeURIComponent(parts.slice(2).join("/")),
      queryStr: queryPart || "",
      runId: null,
      compareIds: [],
    };
  }
  if (parts[0] === "meta") {
    return {
      route: "meta",
      queryStr: queryPart || "",
      runId: null,
      compareIds: [],
    };
  }
  if (parts[0] === "replays") {
    return { route: "replays", runId: params.get("run_id") || null, compareIds: [] };
  }
  if (parts[0] === "cards") {
    return { route: "cards", runId: params.get("run_id") || null, compareIds: [] };
  }
  if (parts[0] === "alerts") {
    return { route: "alerts", runId: null, compareIds: [] };
  }
  return { route: "active", runId: null, compareIds: [] };
}

function App() {
  const [loc, setLoc] = useState(parseHash);
  const [runs, setRuns] = useState([]);
  const [projects, setProjects] = useState([]);
  const [selectedIds, setSelectedIds] = useState(() => parseHash().compareIds);
  const [sseStatus, setSseStatus] = useState("connecting");
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("traininglog_theme") || "dark";
    } catch {
      return "dark";
    }
  });
  const [showTokenModal, setShowTokenModal] = useState(false);
  const [tokenInput, setTokenInput] = useState(getStoredToken);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("traininglog_theme", theme);
    } catch {
      // ignore
    }
  }, [theme]);

  useEffect(() => {
    const onHash = () => {
      const next = parseHash();
      setLoc(next);
      if (next.compareIds.length > 0) setSelectedIds(next.compareIds);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  async function refreshAll() {
    try {
      const [runsRes, projRes] = await Promise.all([
        apiFetch("/api/runs"),
        apiFetch("/api/projects"),
      ]);
      setRuns(runsRes.runs || []);
      setProjects(projRes.projects || []);
    } catch {
      // auth modal handles 401
    }
  }

  useEffect(() => {
    setUnauthorizedHandler(() => setShowTokenModal(true));
    refreshAll();

    const close = connectSSE(
      (ev) => {
        if (ev.type === "run_created" || ev.type === "run_deleted") {
          refreshAll();
        } else if (ev.type === "run_updated" && ev.data) {
          setRuns((prev) =>
            prev.map((r) => (r.id === ev.data.id ? ev.data : r))
          );
        } else if (ev.type === "run_metrics" && ev.data && ev.data.run) {
          const updated = ev.data.run;
          setRuns((prev) =>
            prev.map((r) => (r.id === updated.id ? updated : r))
          );
        } else if (ev.type === "anomaly" || ev.type === "anomaly_resolved") {
          refreshAll();
        }
      },
      { onStatus: setSseStatus }
    );
    return close;
  }, []);

  function navigate(hash) {
    window.location.hash = hash;
  }

  function toggleSelect(rid) {
    setSelectedIds((prev) =>
      prev.includes(rid) ? prev.filter((x) => x !== rid) : [...prev, rid]
    );
  }

  function openCompare(ids) {
    setSelectedIds(ids);
    navigate(`#/compare?ids=${encodeURIComponent(ids.join(","))}`);
  }

  const openAlertsCount = useMemo(
    () => runs.reduce((acc, r) => acc + (r.open_anomaly_count || 0), 0),
    [runs]
  );

  const activeCount = useMemo(
    () => runs.filter((r) => r.state === "running" || r.state === "stalled").length,
    [runs]
  );

  function saveToken(e) {
    e.preventDefault();
    setStoredToken(tokenInput.trim());
    setShowTokenModal(false);
    refreshAll();
  }

  return html`
    <div class="shell">
      <aside class="sidebar">
        <div class="brand">
          <img src="/icon.svg" alt="" />
          <span>TrainingLog</span>
        </div>
        <nav class="sidebar-nav">
          <button
            class=${`side-btn ${loc.route === "active" ? "active" : ""}`}
            onClick=${() => navigate("#/active")}
          >
            <span>Active</span>
            ${activeCount > 0 ? html`<span class="pill running">${activeCount}</span>` : null}
          </button>
          <button
            class=${`side-btn ${loc.route === "runs" || loc.route === "run_detail" ? "active" : ""}`}
            onClick=${() => navigate("#/runs")}
          >
            <span>Runs</span>
            <span class="pill">${runs.length}</span>
          </button>
          <button
            class=${`side-btn ${loc.route === "compare" ? "active" : ""}`}
            onClick=${() => navigate("#/compare")}
          >
            <span>Compare</span>
            ${selectedIds.length > 0 ? html`<span class="pill accent">${selectedIds.length}</span>` : null}
          </button>
          <button
            class=${`side-btn ${loc.route === "league" ? "active" : ""}`}
            onClick=${() => navigate("#/league")}
          >
            <span>League</span>
          </button>
          <button
            class=${`side-btn ${loc.route === "meta" || loc.route === "meta_card" ? "active" : ""}`}
            onClick=${() => navigate("#/meta")}
          >
            <span>Meta</span>
          </button>
          <button
            class=${`side-btn ${loc.route === "replays" ? "active" : ""}`}
            onClick=${() => navigate("#/replays")}
          >
            <span>Replays</span>
          </button>
          <button
            class=${`side-btn ${loc.route === "cards" ? "active" : ""}`}
            onClick=${() => navigate("#/cards")}
          >
            <span>Cards</span>
          </button>
          <button
            class=${`side-btn ${loc.route === "alerts" ? "active" : ""}`}
            onClick=${() => navigate("#/alerts")}
          >
            <span>Alerts</span>
            ${openAlertsCount > 0 ? html`<span class="pill critical">${openAlertsCount}</span>` : null}
          </button>
        </nav>
        <div class="muted" style="font-size:11px; padding:8px 10px;">
          Local RL Tracker · 4-Lane Card Game
        </div>
      </aside>

      <div class="main-col">
        <header class="topbar">
          <div class="brand">
            <img src="/icon.svg" alt="" />
            <span>TrainingLog</span>
          </div>
          <div class="topbar-actions">
            <span class=${`pill ${sseStatus === "live" ? "running" : "warning"}`}>
              <span class=${`dot ${sseStatus === "live" ? "pulse" : ""}`}></span>
              ${sseStatus}
            </span>
            <button
              class="btn sm"
              onClick=${() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              ${theme === "dark" ? "Light" : "Dark"}
            </button>
            <button
              class="btn sm"
              title="API Token"
              onClick=${() => setShowTokenModal(true)}
            >
              Auth
            </button>
          </div>
        </header>

        <main class="content">
          ${loc.route === "active"
            ? html`<${ActiveView}
                runs=${runs}
                onOpenRun=${(id) => navigate(`#/runs/${id}`)}
                onOpenAlerts=${() => navigate("#/alerts")}
                onRefresh=${refreshAll}
              />`
            : null}

          ${loc.route === "runs"
            ? html`<${RunsView}
                runs=${runs}
                projects=${projects}
                selectedIds=${selectedIds}
                onToggleSelect=${toggleSelect}
                onSetSelectedIds=${setSelectedIds}
                onOpenRun=${(id) => navigate(`#/runs/${id}`)}
                onOpenCompare=${openCompare}
                onRefresh=${refreshAll}
              />`
            : null}

          ${loc.route === "run_detail"
            ? html`<${RunDetailView}
                runId=${loc.runId}
                theme=${theme}
                onBack=${() => navigate("#/runs")}
                onCompareRun=${(id) => openCompare([id])}
                onRefreshGlobal=${refreshAll}
                onOpenReplays=${(rid) => navigate(`#/replays?run_id=${encodeURIComponent(rid)}`)}
                onOpenCards=${(rid) => navigate(`#/cards?run_id=${encodeURIComponent(rid)}`)}
              />`
            : null}

          ${loc.route === "compare"
            ? html`<${CompareView}
                runs=${runs}
                selectedIds=${selectedIds}
                onSetSelectedIds=${setSelectedIds}
                onOpenRun=${(id) => navigate(`#/runs/${id}`)}
                theme=${theme}
              />`
            : null}

          ${loc.route === "league"
            ? html`<${LeagueView}
                projects=${projects}
                theme=${theme}
                onOpenRun=${(id) => navigate(`#/runs/${id}`)}
                onOpenCards=${() => navigate("#/cards")}
              />`
            : null}

          ${loc.route === "meta"
            ? html`<${MetaView}
                projects=${projects}
                theme=${theme}
                queryStr=${loc.queryStr || ""}
                onOpenCard=${(cardKey, qStr) =>
                  navigate(
                    `#/meta/card/${encodeURIComponent(cardKey)}${qStr ? `?${qStr}` : ""}`
                  )}
                onSyncQuery=${(qStr) => {
                  const nextHash = `#/meta${qStr ? `?${qStr}` : ""}`;
                  if (window.location.hash !== nextHash) {
                    history.replaceState(null, "", nextHash);
                  }
                }}
              />`
            : null}

          ${loc.route === "meta_card"
            ? html`<${MetaCardView}
                cardKey=${loc.cardKey}
                queryStr=${loc.queryStr || ""}
                theme=${theme}
                onBack=${(qStr) => navigate(`#/meta${qStr ? `?${qStr}` : ""}`)}
                onOpenCard=${(cardKey) =>
                  navigate(
                    `#/meta/card/${encodeURIComponent(cardKey)}${loc.queryStr ? `?${loc.queryStr}` : ""}`
                  )}
              />`
            : null}

          ${loc.route === "replays"
            ? html`<${ReplaysView}
                projects=${projects}
                runs=${runs}
                theme=${theme}
                initialRunId=${loc.runId || ""}
                onOpenCards=${() => navigate("#/cards")}
                onOpenRun=${(id) => navigate(`#/runs/${id}`)}
              />`
            : null}

          ${loc.route === "cards"
            ? html`<${CardsView}
                projects=${projects}
                runs=${runs}
                theme=${theme}
                initialRunId=${loc.runId || ""}
                onOpenReplays=${() => navigate("#/replays")}
                onOpenRun=${(id) => navigate(`#/runs/${id}`)}
              />`
            : null}

          ${loc.route === "alerts"
            ? html`<${AlertsView}
                projects=${projects}
                onOpenRun=${(id) => navigate(`#/runs/${id}`)}
                onRefreshGlobal=${refreshAll}
              />`
            : null}
        </main>
      </div>

      <nav class="bottom-nav">
        <button
          class=${`nav-btn ${loc.route === "active" ? "active" : ""}`}
          onClick=${() => navigate("#/active")}
        >
          <span>Active</span>
        </button>
        <button
          class=${`nav-btn ${loc.route === "runs" || loc.route === "run_detail" || loc.route === "compare" ? "active" : ""}`}
          onClick=${() => navigate("#/runs")}
        >
          <span>Runs</span>
        </button>
        <button
          class=${`nav-btn ${loc.route === "league" ? "active" : ""}`}
          onClick=${() => navigate("#/league")}
        >
          <span>League</span>
        </button>
        <button
          class=${`nav-btn ${loc.route === "meta" || loc.route === "meta_card" ? "active" : ""}`}
          onClick=${() => navigate("#/meta")}
        >
          <span>Meta</span>
        </button>
        <button
          class=${`nav-btn ${loc.route === "replays" || loc.route === "cards" ? "active" : ""}`}
          onClick=${() => navigate("#/replays")}
        >
          <span>Replays</span>
        </button>
        <button
          class=${`nav-btn ${loc.route === "alerts" ? "active" : ""}`}
          onClick=${() => navigate("#/alerts")}
        >
          <span>Alerts</span>
          ${openAlertsCount > 0 ? html`<span class="nav-badge">${openAlertsCount}</span>` : null}
        </button>
      </nav>

      ${showTokenModal
        ? html`
            <div class="modal-backdrop" onClick=${() => setShowTokenModal(false)}>
              <div class="modal-card stack" onClick=${(e) => e.stopPropagation()}>
                <div class="space-between">
                  <strong>API Token</strong>
                  <button class="btn sm" onClick=${() => setShowTokenModal(false)}>
                    Close
                  </button>
                </div>
                <p class="muted" style="margin:0; font-size:12px;">
                  Enter the server's <code class="mono">TRAININGLOG_API_TOKEN</code> if authentication is enabled.
                </p>
                <form class="stack" onSubmit=${saveToken}>
                  <input
                    class="input mono"
                    type="password"
                    placeholder="Bearer token (leave blank to clear)"
                    value=${tokenInput}
                    onInput=${(e) => setTokenInput(e.target.value)}
                  />
                  <div class="row">
                    <button class="btn primary sm" type="submit">Save Token</button>
                  </div>
                </form>
              </div>
            </div>
          `
        : null}
    </div>
  `;
}

render(html`<${App} />`, document.getElementById("app"));

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
