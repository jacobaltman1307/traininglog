/** HTTP client, SSE stream helper, and human-friendly RL formatting utilities. */

const TOKEN_STORAGE_KEY = "traininglog_api_token";
let onUnauthorizedCallback = null;

export function getStoredToken() {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function setStoredToken(token) {
  try {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch {
    // ignore storage restrictions
  }
}

export function setUnauthorizedHandler(fn) {
  onUnauthorizedCallback = fn;
}

export function withTokenQuery(url) {
  const token = getStoredToken();
  if (!token) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

export async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getStoredToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (
    options.body &&
    typeof options.body === "string" &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }

  const resp = await fetch(path, { ...options, headers });
  if (resp.status === 401) {
    if (onUnauthorizedCallback) onUnauthorizedCallback();
    throw new Error("Unauthorized (API token required)");
  }
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try {
      const errJson = await resp.json();
      if (errJson && errJson.detail) msg = String(errJson.detail);
    } catch {
      // ignore non-json error body
    }
    throw new Error(msg);
  }
  return resp.json();
}

export function connectSSE(onEvent, { runId = null, onStatus = null } = {}) {
  let es = null;
  let closed = false;
  let retryTimer = null;

  function open() {
    if (closed) return;
    const base = runId
      ? `/api/events?run_id=${encodeURIComponent(runId)}`
      : "/api/events";
    const url = withTokenQuery(base);
    es = new EventSource(url);
    if (onStatus) onStatus("connecting");

    const handleFrame = (ev) => {
      try {
        const parsed = JSON.parse(ev.data);
        onEvent(parsed);
      } catch {
        // ignore keepalive or malformed data
      }
    };

    es.addEventListener("ready", () => {
      if (onStatus) onStatus("live");
    });

    for (const type of [
      "run_created",
      "run_updated",
      "run_deleted",
      "run_metrics",
      "console_batch",
      "anomaly",
      "anomaly_resolved",
      "command_updated",
      "match_recorded",
      "checkpoint_updated",
      "plasticity_recorded",
      "replay_recorded",
      "cards_updated",
    ]) {
      es.addEventListener(type, handleFrame);
    }

    es.onerror = () => {
      if (onStatus) onStatus("reconnecting");
      if (es) {
        es.close();
        es = null;
      }
      if (!closed) {
        retryTimer = setTimeout(open, 2500);
      }
    };
  }

  open();

  return () => {
    closed = true;
    if (retryTimer) clearTimeout(retryTimer);
    if (es) es.close();
  };
}

export function formatStep(step) {
  if (step === null || step === undefined || Number.isNaN(step)) return "—";
  const abs = Math.abs(step);
  if (abs >= 1e9) return `${(step / 1e9).toFixed(abs >= 10e9 ? 1 : 2)}B`;
  if (abs >= 1e6) return `${(step / 1e6).toFixed(abs >= 100e6 ? 0 : 1)}M`;
  if (abs >= 1e3) return `${(step / 1e3).toFixed(abs >= 100e3 ? 0 : 1)}k`;
  return String(Math.round(step));
}

export function formatNumber(val) {
  if (val === null || val === undefined || !Number.isFinite(val)) return "—";
  const abs = Math.abs(val);
  if (abs === 0) return "0";
  if (abs < 0.001 || abs >= 1e6) return val.toExponential(2);
  if (abs < 1) return val.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  if (abs < 100) return val.toFixed(2).replace(/\.00$/, "");
  return Math.round(val).toLocaleString();
}

export function formatPct(val) {
  if (val === null || val === undefined || !Number.isFinite(val)) return "—";
  const pct = val <= 1.0 && val >= 0.0 ? val * 100 : val;
  return `${pct.toFixed(1)}%`;
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "—";
  }
  const s = Math.max(0, Math.round(seconds));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (d > 0) return `${d}d ${String(h).padStart(2, "0")}h`;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return `${sec}s`;
}

export function formatTimeAgo(tsMs) {
  if (!tsMs) return "—";
  const diffSec = Math.max(0, Math.round((Date.now() - tsMs) / 1000));
  if (diffSec < 5) return "just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86400)}d ago`;
}

export function computeETA(run) {
  if (
    !run ||
    run.state !== "running" ||
    !run.total_timesteps ||
    !run.current_sps ||
    run.current_sps <= 0
  ) {
    return null;
  }
  const rem = Math.max(0, run.total_timesteps - (run.current_step || 0));
  return rem / run.current_sps;
}

export function computeEMA(ys, alpha) {
  if (!alpha || alpha <= 0) return ys;
  const clamped = Math.min(0.99, Math.max(0, alpha));
  const out = new Array(ys.length);
  let prev = null;
  for (let i = 0; i < ys.length; i++) {
    const v = ys[i];
    if (v === null || v === undefined || !Number.isFinite(v)) {
      out[i] = null;
    } else if (prev === null) {
      prev = v;
      out[i] = v;
    } else {
      prev = clamped * prev + (1 - clamped) * v;
      out[i] = prev;
    }
  }
  return out;
}
