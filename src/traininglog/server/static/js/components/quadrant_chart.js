/** Lightweight HiDPI Canvas quadrant scatter plot for Meta overview. */

import { html, useEffect, useMemo, useRef, useState } from "../../vendor/ui.mjs";

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

export function QuadrantChart({
  cards,
  yMetric = "drawn_wpa",
  xMetric = "inclusion_rate",
  theme = "dark",
  height = 280,
  onSelectCard,
}) {
  const canvasRef = useRef(null);
  const [hoverInfo, setHoverInfo] = useState(null);
  const pointsRef = useRef([]);

  const metricLabel = useMemo(() => {
    if (yMetric === "isolated_impact") return "Est. Impact (β)";
    if (yMetric === "meta_war") return "Meta WAR";
    if (yMetric === "gih_wr") return "GIH Win Rate";
    return "Drawn WPA (IWD)";
  }, [yMetric]);

  const baselineY = yMetric === "gih_wr" ? 0.5 : 0.0;

  const validPoints = useMemo(() => {
    return (cards || [])
      .filter((c) => Number.isFinite(c[xMetric]) && Number.isFinite(c[yMetric]))
      .map((c) => ({
        card_key: c.card_key,
        label: c.display_name || c.card_key,
        x: Number(c[xMetric]),
        y: Number(c[yMetric]),
        n: Number(c.games_drawn || c.games_in_deck || 0),
      }));
  }, [cards, xMetric, yMetric]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const parent = canvas.parentElement;
      const width = Math.max(260, parent ? parent.clientWidth - 2 : 520);
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const padL = 52;
      const padR = 20;
      const padT = 22;
      const padB = 34;
      const plotW = Math.max(40, width - padL - padR);
      const plotH = Math.max(40, height - padT - padB);

      const muted = cssVar("--muted", "#a19a8f");
      const text = cssVar("--text", "#f5f2eb");
      const grid = cssVar("--grid", "rgba(161, 154, 143, 0.13)");
      const okCol = cssVar("--ok", "#34d399");
      const dangerCol = cssVar("--danger", "#f87171");
      const accentCol = cssVar("--accent", "#e59838");

      if (validPoints.length === 0) {
        ctx.fillStyle = muted;
        ctx.font = "12px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Not enough data for selected metric", width / 2, height / 2);
        pointsRef.current = [];
        return;
      }

      const xs = validPoints.map((p) => p.x).sort((a, b) => a - b);
      const ys = validPoints.map((p) => p.y);
      const medX = xs[Math.floor(xs.length / 2)];

      let minX = Math.max(0, Math.min(...xs) * 0.9);
      let maxX = Math.min(1, Math.max(...xs) * 1.08);
      if (maxX - minX < 0.05) {
        minX = Math.max(0, medX - 0.05);
        maxX = Math.min(1, medX + 0.05);
      }

      const maxAbsDev = Math.max(
        yMetric === "meta_war" ? 2.0 : 0.04,
        ...ys.map((y) => Math.abs(y - baselineY))
      );
      const minY = baselineY - maxAbsDev * 1.18;
      const maxY = baselineY + maxAbsDev * 1.18;

      const toPx = (x) => padL + ((x - minX) / (maxX - minX)) * plotW;
      const toPy = (y) => padT + (1 - (y - minY) / (maxY - minY)) * plotH;

      // Grid lines
      ctx.strokeStyle = grid;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const gy = padT + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(padL, gy);
        ctx.lineTo(padL + plotW, gy);
        ctx.stroke();

        const valY = maxY - ((maxY - minY) * i) / 4;
        ctx.fillStyle = muted;
        ctx.font = "10.5px ui-monospace, monospace";
        ctx.textAlign = "right";
        const lbl =
          yMetric === "meta_war"
            ? valY.toFixed(1)
            : `${(valY * 100).toFixed(1)}%`;
        ctx.fillText(lbl, padL - 6, gy + 3.5);
      }

      for (let i = 0; i <= 4; i++) {
        const gx = padL + (plotW * i) / 4;
        ctx.beginPath();
        ctx.moveTo(gx, padT);
        ctx.lineTo(gx, padT + plotH);
        ctx.stroke();

        const valX = minX + ((maxX - minX) * i) / 4;
        ctx.fillStyle = muted;
        ctx.font = "10.5px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(`${(valX * 100).toFixed(0)}%`, gx, padT + plotH + 15);
      }

      // Quadrant divider lines (baseline Y & median X)
      const basePy = toPy(baselineY);
      const medPx = toPx(medX);
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = accentCol;
      ctx.globalAlpha = 0.65;
      ctx.beginPath();
      ctx.moveTo(padL, basePy);
      ctx.lineTo(padL + plotW, basePy);
      ctx.moveTo(medPx, padT);
      ctx.lineTo(medPx, padT + plotH);
      ctx.stroke();
      ctx.restore();

      // Subtle quadrant labels
      ctx.fillStyle = muted;
      ctx.globalAlpha = 0.7;
      ctx.font = "600 10.5px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("HIDDEN GEMS", padL + 6, padT + 13);
      ctx.fillText("NICHE / WEAK", padL + 6, padT + plotH - 6);
      ctx.textAlign = "right";
      ctx.fillText("META STAPLES", padL + plotW - 6, padT + 13);
      ctx.fillText("OVERPLAYED TRAPS", padL + plotW - 6, padT + plotH - 6);
      ctx.globalAlpha = 1.0;

      // Axis titles
      ctx.fillStyle = muted;
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        xMetric === "draw_to_play_rate" ? "Draw → Play Rate" : "Deck Inclusion Rate",
        padL + plotW / 2,
        height - 4
      );

      // Scatter dots
      const maxN = Math.max(1, ...validPoints.map((p) => p.n));
      const drawnPts = [];
      for (const p of validPoints) {
        const px = toPx(p.x);
        const py = toPy(p.y);
        const r = 4.5 + 4.5 * Math.sqrt(p.n / maxN);
        const diff = p.y - baselineY;
        const fill =
          diff > (yMetric === "meta_war" ? 0.3 : 0.01)
            ? okCol
            : diff < -(yMetric === "meta_war" ? 0.3 : 0.01)
              ? dangerCol
              : accentCol;

        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.globalAlpha = 0.86;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = cssVar("--surface", "#151412");
        ctx.stroke();

        // Compact label next to point when few cards
        if (validPoints.length <= 28) {
          ctx.fillStyle = text;
          ctx.globalAlpha = 0.85;
          ctx.font = "10.5px system-ui, sans-serif";
          ctx.textAlign = px > padL + plotW * 0.72 ? "right" : "left";
          const dx = px > padL + plotW * 0.72 ? -(r + 4) : r + 4;
          ctx.fillText(p.label, px + dx, py + 3.5);
          ctx.globalAlpha = 1.0;
        }
        drawnPts.push({ ...p, px, py, r });
      }
      pointsRef.current = drawnPts;
    };

    draw();
    window.addEventListener("resize", draw);
    return () => window.removeEventListener("resize", draw);
  }, [validPoints, yMetric, xMetric, theme, height, baselineY]);

  function findHit(e) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = null;
    let bestDist = 20;
    for (const pt of pointsRef.current) {
      const d = Math.hypot(pt.px - mx, pt.py - my);
      if (d <= Math.max(10, pt.r + 5) && d < bestDist) {
        best = pt;
        bestDist = d;
      }
    }
    return best;
  }

  return html`
    <div class="quadrant-wrap">
      <canvas
        ref=${canvasRef}
        class="quadrant-canvas"
        onMouseMove=${(e) => {
          const hit = findHit(e);
          setHoverInfo(hit);
        }}
        onMouseLeave=${() => setHoverInfo(null)}
        onClick=${(e) => {
          const hit = findHit(e);
          if (hit && onSelectCard) onSelectCard(hit.card_key);
        }}
      ></canvas>
      ${hoverInfo
        ? html`
            <div
              class="quadrant-tooltip"
              style=${`left:${Math.min(Math.max(10, hoverInfo.px + 12), 420)}px; top:${Math.max(8, hoverInfo.py - 12)}px;`}
            >
              <div style="font-weight:700;">${hoverInfo.label}</div>
              <div class="mono" style="font-size:11px;">
                Inclusion: ${(hoverInfo.x * 100).toFixed(1)}% · ${metricLabel}:
                ${yMetric === "meta_war"
                  ? hoverInfo.y.toFixed(2)
                  : `${(hoverInfo.y * 100).toFixed(1)}%`}
              </div>
              <div class="muted" style="font-size:10.5px;">
                ${hoverInfo.n} drawn games · click to open dossier
              </div>
            </div>
          `
        : null}
    </div>
  `;
}
