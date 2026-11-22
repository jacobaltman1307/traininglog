/** uPlot chart wrapper, lightweight Canvas Sparkline, and Histogram viewer. */

import { html, useEffect, useRef } from "../vendor/ui.mjs";
import { computeEMA, formatDuration, formatNumber, formatStep } from "./api.js";

export const SERIES_COLORS = [
  "#e59838",
  "#34d399",
  "#fbbf24",
  "#fb7185",
  "#a78bfa",
  "#f97316",
  "#a3e635",
  "#e879f9",
];

function cssVar(name, fallback) {
  try {
    return (
      getComputedStyle(document.documentElement).getPropertyValue(name).trim() ||
      fallback
    );
  } catch {
    return fallback;
  }
}

function alignMultiRunSeries(seriesMap, runOrder) {
  const activeRuns = runOrder.filter(
    (rid) => seriesMap[rid] && seriesMap[rid].x && seriesMap[rid].x.length > 0
  );
  if (activeRuns.length === 0) {
    return { x: [], aligned: {}, activeRuns: [] };
  }
  if (activeRuns.length === 1) {
    const rid = activeRuns[0];
    return {
      x: seriesMap[rid].x,
      aligned: { [rid]: seriesMap[rid].y },
      activeRuns,
    };
  }

  const xSet = new Set();
  const maps = {};
  for (const rid of activeRuns) {
    const s = seriesMap[rid];
    const m = new Map();
    for (let i = 0; i < s.x.length; i++) {
      xSet.add(s.x[i]);
      m.set(s.x[i], s.y[i]);
    }
    maps[rid] = m;
  }

  const xSorted = Array.from(xSet).sort((a, b) => a - b);
  const aligned = {};
  for (const rid of activeRuns) {
    const m = maps[rid];
    aligned[rid] = xSorted.map((xv) => (m.has(xv) ? m.get(xv) : null));
  }
  return { x: xSorted, aligned, activeRuns };
}

export function UPlotChart({
  title,
  seriesMap,
  runList,
  xAxis = "step",
  smoothing = 0,
  logScale = false,
  syncKey = null,
  height = 230,
  theme = "dark",
}) {
  const wrapRef = useRef(null);
  const plotRef = useRef(null);

  useEffect(() => {
    const container = wrapRef.current;
    if (!container || typeof window.uPlot !== "function") return;

    const runOrder = runList.map((r) => r.id);
    const { x, aligned, activeRuns } = alignMultiRunSeries(seriesMap, runOrder);
    if (x.length === 0) {
      if (plotRef.current) {
        plotRef.current.destroy();
        plotRef.current = null;
      }
      return;
    }

    const nanSteps = [];
    for (const rid of activeRuns) {
      const s = seriesMap[rid];
      if (s && Array.isArray(s.non_finite_steps)) {
        for (const st of s.non_finite_steps) nanSteps.push(st);
      }
    }

    const data = [x];
    const seriesOpts = [
      {
        label:
          xAxis === "relative_time"
            ? "Elapsed"
            : xAxis === "wall_time"
              ? "Time"
              : xAxis === "turn"
                ? "Turn"
                : "Step",
        value: (_, v) =>
          v == null
            ? "—"
            : xAxis === "step"
              ? formatStep(v)
              : xAxis === "turn"
                ? `T${Math.round(v)}`
                : xAxis === "relative_time"
                  ? formatDuration(v)
                  : new Date(v * 1000).toLocaleTimeString(),
      },
    ];

    const useSmooth = smoothing > 0.01;
    const runMeta = new Map(runList.map((r, idx) => [r.id, { run: r, idx }]));

    for (const rid of activeRuns) {
      const meta = runMeta.get(rid) || { run: { name: rid }, idx: 0 };
      const color = SERIES_COLORS[meta.idx % SERIES_COLORS.length];
      let rawY = aligned[rid];
      if (logScale) {
        rawY = rawY.map((v) => (v !== null && v > 0 ? v : null));
      }

      if (useSmooth) {
        data.push(rawY);
        seriesOpts.push({
          label: `${meta.run.name} (raw)`,
          stroke: `${color}40`,
          width: 1,
          spanGaps: true,
          points: { show: false },
          value: (_, v) => formatNumber(v),
        });

        const emaY = computeEMA(rawY, smoothing);
        data.push(emaY);
        seriesOpts.push({
          label: meta.run.name,
          stroke: color,
          width: 2,
          spanGaps: true,
          points: { show: x.length <= 2 },
          value: (_, v) => formatNumber(v),
        });
      } else {
        data.push(rawY);
        seriesOpts.push({
          label: meta.run.name,
          stroke: color,
          width: 2,
          spanGaps: true,
          points: { show: x.length <= 2 },
          value: (_, v) => formatNumber(v),
        });
      }
    }

    const textColor = cssVar("--muted", "#a19a8f");
    const gridColor = cssVar("--grid", "rgba(161, 154, 143, 0.13)");
    const width = Math.max(260, container.clientWidth || 320);

    const opts = {
      width,
      height,
      cursor: {
        sync: syncKey ? { key: syncKey } : undefined,
        drag: { x: true, y: false },
      },
      scales: {
        x: { time: xAxis === "wall_time" },
        y: { distr: logScale ? 3 : 1 },
      },
      axes: [
        {
          stroke: textColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "11px system-ui, sans-serif",
          values: (_, splits) =>
            splits.map((v) => {
              if (v == null) return "";
              if (xAxis === "step") return formatStep(v);
              if (xAxis === "turn") return `T${Math.round(v)}`;
              if (xAxis === "relative_time") return formatDuration(v);
              return new Date(v * 1000).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              });
            }),
        },
        {
          stroke: textColor,
          grid: { stroke: gridColor, width: 1 },
          ticks: { stroke: gridColor, width: 1 },
          font: "11px system-ui, sans-serif",
          size: 56,
          values: (_, splits) => splits.map((v) => formatNumber(v)),
        },
      ],
      series: seriesOpts,
      hooks: {
        draw: [
          (u) => {
            if (xAxis !== "step" || nanSteps.length === 0) return;
            const ctx = u.ctx;
            const { top, height: bh, left, width: bw } = u.bbox;
            ctx.save();
            ctx.strokeStyle = cssVar("--danger", "#f87171");
            ctx.fillStyle = cssVar("--danger", "#f87171");
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            for (const s of nanSteps) {
              const px = u.valToPos(s, "x", true);
              if (px >= left && px <= left + bw) {
                ctx.beginPath();
                ctx.moveTo(px, top);
                ctx.lineTo(px, top + bh);
                ctx.stroke();
                ctx.font = "bold 10px system-ui, sans-serif";
                ctx.fillText("NaN", px + 3, top + 12);
              }
            }
            ctx.restore();
          },
        ],
      },
    };

    if (plotRef.current) {
      plotRef.current.destroy();
      plotRef.current = null;
    }
    const plot = new window.uPlot(opts, data, container);
    plotRef.current = plot;

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width);
        if (w > 60 && plotRef.current) {
          plotRef.current.setSize({ width: w, height });
        }
      }
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      if (plotRef.current) {
        plotRef.current.destroy();
        plotRef.current = null;
      }
    };
  }, [seriesMap, runList, xAxis, smoothing, logScale, syncKey, height, theme]);

  const hasData = runList.some(
    (r) => seriesMap[r.id] && seriesMap[r.id].x && seriesMap[r.id].x.length > 0
  );

  return html`
    <div class="uplot-card">
      <div class="uplot-head">
        <div class="uplot-title mono" title=${title}>${title}</div>
      </div>
      ${hasData
        ? html`<div ref=${wrapRef} style="width:100%;"></div>`
        : html`<div class="muted" style="padding: 28px 0; text-align:center; font-size:12px;">
            No points yet
          </div>`}
    </div>
  `;
}

export function Sparkline({ ys, color = "#e59838", height = 34 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 100;
    const h = height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const pts = (ys || []).filter((v) => v !== null && Number.isFinite(v));
    if (pts.length < 2) return;

    const minV = Math.min(...pts);
    const maxV = Math.max(...pts);
    const span = maxV - minV || 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    pts.forEach((v, idx) => {
      const px = (idx / (pts.length - 1)) * (w - 4) + 2;
      const py = h - 3 - ((v - minV) / span) * (h - 6);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }, [ys, color, height]);

  return html`<canvas
    ref=${canvasRef}
    style=${`width: 100%; height: ${height}px; display: block;`}
  ></canvas>`;
}

export function HistogramChart({ snapshot }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !snapshot) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 320;
    const h = 150;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const counts = snapshot.counts || [];
    if (counts.length === 0) return;
    const maxC = Math.max(...counts, 1);
    const padB = 22;
    const barW = (w - 16) / counts.length;

    ctx.fillStyle = cssVar("--accent", "#e59838");
    counts.forEach((c, idx) => {
      const bh = ((c / maxC) * (h - padB - 10));
      const x = 8 + idx * barW;
      const y = h - padB - bh;
      ctx.fillRect(x + 1, y, Math.max(2, barW - 2), bh);
    });

    ctx.fillStyle = cssVar("--muted", "#a19a8f");
    ctx.font = "11px system-ui, sans-serif";
    const bins = snapshot.bins || [];
    if (bins.length >= 2) {
      ctx.fillText(formatNumber(bins[0]), 8, h - 6);
      const endLabel = formatNumber(bins[bins.length - 1]);
      ctx.fillText(endLabel, w - ctx.measureText(endLabel).width - 8, h - 6);
    }
  }, [snapshot]);

  return html`<canvas
    ref=${canvasRef}
    style="width: 100%; height: 150px; display: block;"
  ></canvas>`;
}
