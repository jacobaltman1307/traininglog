import { apiFetch, formatStep, formatTimeAgo } from "../api.js";
import { html, useEffect, useState } from "../../vendor/ui.mjs";

const EVENT_CHOICES = [
  "crashed",
  "nan_inf",
  "stalled",
  "kl_spike",
  "entropy_collapse",
  "sps_drop",
  "high_dormancy",
  "rank_collapse",
  "new_best_elo",
  "finished",
];

export function AlertsView({ projects, onOpenRun, onRefreshGlobal }) {
  const [anomalies, setAnomalies] = useState([]);
  const [rules, setRules] = useState([]);
  const [events, setEvents] = useState([]);
  const [showResolved, setShowResolved] = useState(false);
  const [testResult, setTestResult] = useState(null);

  // New rule form
  const [name, setName] = useState("");
  const [projectId, setProjectId] = useState("");
  const [channel, setChannel] = useState("ntfy");
  const [url, setUrl] = useState("");
  const [token, setToken] = useState("");
  const [userKey, setUserKey] = useState("");
  const [chatId, setChatId] = useState("");
  const [selectedEvents, setSelectedEvents] = useState([
    "crashed",
    "nan_inf",
    "stalled",
    "kl_spike",
  ]);
  const [klThreshold, setKlThreshold] = useState("0.1");
  const [entropyFloor, setEntropyFloor] = useState("0.05");
  const [cooldownSec, setCooldownSec] = useState("300");
  const [creating, setCreating] = useState(false);

  async function loadAll() {
    try {
      const resolvedParam = showResolved ? "" : "?resolved=0";
      const [anomRes, ruleRes, evRes] = await Promise.all([
        apiFetch(`/api/anomalies${resolvedParam}`),
        apiFetch("/api/alerts/rules"),
        apiFetch("/api/alerts/events?limit=30"),
      ]);
      setAnomalies(anomRes.anomalies || []);
      setRules(ruleRes.rules || []);
      setEvents(evRes.events || []);
    } catch {
      // ignore transient error
    }
  }

  useEffect(() => {
    loadAll();
  }, [showResolved]);

  async function resolveAnomaly(id) {
    await apiFetch(`/api/anomalies/${id}/resolve`, { method: "POST" });
    await loadAll();
    if (onRefreshGlobal) onRefreshGlobal();
  }

  function toggleEvent(ev) {
    if (selectedEvents.includes(ev)) {
      setSelectedEvents(selectedEvents.filter((x) => x !== ev));
    } else {
      setSelectedEvents([...selectedEvents, ev]);
    }
  }

  async function createRule(e) {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    try {
      const target = {};
      if (channel === "ntfy") {
        target.url = url.trim();
        if (token.trim()) target.token = token.trim();
      } else if (channel === "discord" || channel === "webhook") {
        target.url = url.trim();
      } else if (channel === "pushover") {
        target.token = token.trim();
        target.user = userKey.trim();
      } else if (channel === "telegram") {
        target.bot_token = token.trim();
        target.chat_id = chatId.trim();
      }

      const thresholds = {};
      if (klThreshold) thresholds.kl_threshold = parseFloat(klThreshold);
      if (entropyFloor) thresholds.entropy_floor = parseFloat(entropyFloor);

      await apiFetch("/api/alerts/rules", {
        method: "POST",
        body: JSON.stringify({
          name: name.trim(),
          project_id: projectId || null,
          channel,
          target,
          events: selectedEvents,
          thresholds,
          cooldown_sec: parseInt(cooldownSec || "300", 10),
          enabled: true,
        }),
      });
      setName("");
      setUrl("");
      setToken("");
      setUserKey("");
      setChatId("");
      await loadAll();
    } catch (err) {
      alert(`Failed to create rule: ${err.message}`);
    } finally {
      setCreating(false);
    }
  }

  async function toggleRuleEnabled(rule) {
    await apiFetch(`/api/alerts/rules/${rule.id}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled: !rule.enabled }),
    });
    await loadAll();
  }

  async function deleteRule(ruleId) {
    await apiFetch(`/api/alerts/rules/${ruleId}`, { method: "DELETE" });
    await loadAll();
  }

  async function sendTest(ruleId) {
    setTestResult("Sending…");
    try {
      const res = await apiFetch(`/api/alerts/rules/${ruleId}/test`, {
        method: "POST",
      });
      const ev = res.event;
      setTestResult(
        ev.status === "sent"
          ? `Test alert sent (HTTP ${ev.response_code})`
          : `Test failed: ${ev.error || ev.status}`
      );
      await loadAll();
    } catch (err) {
      setTestResult(`Test failed: ${err.message}`);
    }
  }

  return html`
    <div class="stack">
      <div class="space-between">
        <div>
          <h2 style="margin:0; font-size:17px;">Health Anomalies & Alerts</h2>
          <div class="muted" style="font-size:12px;">
            Automatic RL health checks & push notifications over LAN/VPN
          </div>
        </div>
        <label class="row" style="font-size:12px; cursor:pointer;">
          <input
            type="checkbox"
            checked=${showResolved}
            onChange=${(e) => setShowResolved(e.target.checked)}
          />
          Show resolved
        </label>
      </div>

      <div class="card stack">
        <strong>Anomalies (${anomalies.length})</strong>
        ${anomalies.length === 0
          ? html`<div class="muted" style="font-size:12px;">
              No open anomalies. All runs look healthy.
            </div>`
          : html`
              <div class="stack" style="gap:8px;">
                ${anomalies.map(
                  (a) => html`
                    <div
                      key=${a.id}
                      class="space-between"
                      style="padding:10px; border-radius:10px; background:var(--bg); border:1px solid var(--border); flex-wrap:wrap;"
                    >
                      <div style="min-width:0; flex:1;">
                        <div class="row" style="gap:6px; margin-bottom:4px;">
                          <span class=${`pill ${a.severity}`}>${a.severity}</span>
                          <span class="pill">${a.kind}</span>
                          <a
                            href=${`#/runs/${a.run_id}`}
                            style="font-weight:600;"
                            onClick=${(e) => {
                              e.preventDefault();
                              onOpenRun(a.run_id);
                            }}
                          >
                            ${a.run_name || a.run_id}
                          </a>
                          ${a.step != null
                            ? html`<span class="muted mono" style="font-size:11px;">
                                @ ${formatStep(a.step)}
                              </span>`
                            : null}
                          <span class="muted" style="font-size:11px;">
                            ${formatTimeAgo(a.created_at)}
                          </span>
                        </div>
                        <div style="font-size:13px;">${a.message}</div>
                      </div>
                      ${!a.resolved
                        ? html`<button
                            class="btn sm"
                            onClick=${() => resolveAnomaly(a.id)}
                          >
                            Resolve
                          </button>`
                        : html`<span class="pill running">resolved</span>`}
                    </div>
                  `
                )}
              </div>
            `}
      </div>

      <div class="card stack">
        <div class="space-between">
          <strong>Notification Rules (${rules.length})</strong>
          ${testResult
            ? html`<span class="pill accent">${testResult}</span>`
            : null}
        </div>

        <form class="stack" onSubmit=${createRule}>
          <div class="row">
            <div style="flex:2; min-width:160px;">
              <input
                class="input"
                placeholder="Rule name (e.g. Phone ntfy)"
                value=${name}
                onInput=${(e) => setName(e.target.value)}
                required
              />
            </div>
            <div style="flex:1; min-width:130px;">
              <select
                class="select"
                value=${channel}
                onChange=${(e) => setChannel(e.target.value)}
              >
                <option value="ntfy">ntfy</option>
                <option value="discord">Discord Webhook</option>
                <option value="pushover">Pushover</option>
                <option value="telegram">Telegram</option>
                <option value="webhook">Generic Webhook</option>
              </select>
            </div>
            <div style="flex:1; min-width:130px;">
              <select
                class="select"
                value=${projectId}
                onChange=${(e) => setProjectId(e.target.value)}
              >
                <option value="">All Projects</option>
                ${projects.map(
                  (p) => html`<option value=${p.id}>${p.name}</option>`
                )}
              </select>
            </div>
          </div>

          <div class="row">
            ${channel === "ntfy" || channel === "discord" || channel === "webhook"
              ? html`<div style="flex:2; min-width:200px;">
                  <input
                    class="input"
                    placeholder=${channel === "ntfy"
                      ? "ntfy URL (e.g. https://ntfy.sh/my-training)"
                      : "Webhook URL"}
                    value=${url}
                    onInput=${(e) => setUrl(e.target.value)}
                    required
                  />
                </div>`
              : null}
            ${channel === "ntfy" || channel === "pushover" || channel === "telegram"
              ? html`<div style="flex:1; min-width:150px;">
                  <input
                    class="input"
                    placeholder=${channel === "ntfy"
                      ? "Bearer token (optional)"
                      : channel === "telegram"
                        ? "Bot Token"
                        : "App Token"}
                    value=${token}
                    onInput=${(e) => setToken(e.target.value)}
                    required=${channel !== "ntfy"}
                  />
                </div>`
              : null}
            ${channel === "pushover"
              ? html`<div style="flex:1; min-width:150px;">
                  <input
                    class="input"
                    placeholder="Pushover User Key"
                    value=${userKey}
                    onInput=${(e) => setUserKey(e.target.value)}
                    required
                  />
                </div>`
              : null}
            ${channel === "telegram"
              ? html`<div style="flex:1; min-width:150px;">
                  <input
                    class="input"
                    placeholder="Chat ID"
                    value=${chatId}
                    onInput=${(e) => setChatId(e.target.value)}
                    required
                  />
                </div>`
              : null}
          </div>

          <div class="row" style="gap:6px;">
            ${EVENT_CHOICES.map(
              (ev) => html`
                <label class="pill" style="cursor:pointer;" key=${ev}>
                  <input
                    type="checkbox"
                    checked=${selectedEvents.includes(ev)}
                    onChange=${() => toggleEvent(ev)}
                  />
                  ${ev}
                </label>
              `
            )}
          </div>

          <div class="row">
            <label class="row" style="font-size:12px;">
              <span class="muted">KL spike &gt;</span>
              <input
                class="input mono"
                style="width:76px;"
                value=${klThreshold}
                onInput=${(e) => setKlThreshold(e.target.value)}
              />
            </label>
            <label class="row" style="font-size:12px;">
              <span class="muted">Entropy &lt;</span>
              <input
                class="input mono"
                style="width:76px;"
                value=${entropyFloor}
                onInput=${(e) => setEntropyFloor(e.target.value)}
              />
            </label>
            <label class="row" style="font-size:12px;">
              <span class="muted">Cooldown (s)</span>
              <input
                class="input mono"
                style="width:76px;"
                value=${cooldownSec}
                onInput=${(e) => setCooldownSec(e.target.value)}
              />
            </label>
            <button class="btn primary sm" type="submit" disabled=${creating}>
              ${creating ? "Adding…" : "Add Rule"}
            </button>
          </div>
        </form>

        ${rules.length > 0
          ? html`
              <div class="stack" style="gap:6px; margin-top:6px;">
                ${rules.map(
                  (r) => html`
                    <div
                      key=${r.id}
                      class="space-between"
                      style="padding:8px 10px; border-radius:8px; background:var(--bg); border:1px solid var(--border); flex-wrap:wrap;"
                    >
                      <div>
                        <div class="row" style="gap:6px;">
                          <strong>${r.name}</strong>
                          <span class="pill accent">${r.channel}</span>
                          <span class=${`pill ${r.enabled ? "running" : ""}`}>
                            ${r.enabled ? "enabled" : "paused"}
                          </span>
                        </div>
                        <div class="muted" style="font-size:11px; margin-top:2px;">
                          Events: ${(r.events || []).join(", ")} · Cooldown ${r.cooldown_sec}s
                        </div>
                      </div>
                      <div class="row">
                        <button class="btn sm" onClick=${() => sendTest(r.id)}>
                          Send Test
                        </button>
                        <button class="btn sm" onClick=${() => toggleRuleEnabled(r)}>
                          ${r.enabled ? "Pause" : "Enable"}
                        </button>
                        <button class="btn sm danger" onClick=${() => deleteRule(r.id)}>
                          Del
                        </button>
                      </div>
                    </div>
                  `
                )}
              </div>
            `
          : null}
      </div>

      ${events.length > 0
        ? html`
            <div class="card stack">
              <strong>Recent Deliveries</strong>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Rule</th>
                      <th>Event</th>
                      <th>Status</th>
                      <th>HTTP</th>
                      <th>When</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${events.map(
                      (ev) => html`
                        <tr key=${ev.id}>
                          <td>${ev.rule_name || ev.rule_id}</td>
                          <td class="mono">${ev.event_type}</td>
                          <td>
                            <span class=${`pill ${ev.status === "sent" ? "running" : "critical"}`}>
                              ${ev.status}
                            </span>
                            ${ev.error ? html` <span class="muted" style="font-size:11px;">${ev.error}</span>` : null}
                          </td>
                          <td class="mono">${ev.response_code || "—"}</td>
                          <td class="muted">${formatTimeAgo(ev.sent_at)}</td>
                        </tr>
                      `
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          `
        : null}
    </div>
  `;
}
