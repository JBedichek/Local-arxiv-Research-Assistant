/* A bar/line chart SVG from a grounded chart visual -- shared by the Learn lesson view and the
 * standalone sub-lesson topic page, so both render the same numbers the same way. */

import { escapeHtml } from "./dom.js";
import { renderMath } from "./tex.js";

const md = (s) => renderMath(escapeHtml(s ?? ""));

export function chartSvg(v) {
  const W = 520, H = 220, pad = 40, max = Math.max(...v.points.map((p) => p.value), 0) || 1;
  const bw = (W - pad * 2) / v.points.length;
  const bars = v.points.map((p, i) => {
    const h = Math.max(2, (p.value / max) * (H - pad * 2));
    const x = pad + i * bw + bw * 0.15, y = H - pad - h;
    return v.chart === "line"
      ? `<circle cx="${x + bw * 0.35}" cy="${y}" r="4" class="pt"/>`
      : `<rect x="${x}" y="${y}" width="${bw * 0.7}" height="${h}" rx="3" class="bar"/>`
      + `<text x="${x + bw * 0.35}" y="${y - 4}" text-anchor="middle" class="val">${p.value}</text>`
      + `<text x="${x + bw * 0.35}" y="${H - pad + 14}" text-anchor="middle" class="lab">${escapeHtml(String(p.label).slice(0, 14))}</text>`;
  }).join("");
  const line = v.chart === "line" ? `<polyline class="ln" points="${v.points.map((p, i) => `${pad + i * bw + bw * 0.5},${H - pad - Math.max(2, (p.value / max) * (H - pad * 2))}`).join(" ")}"/>`
    + v.points.map((p, i) => `<text x="${pad + i * bw + bw * 0.5}" y="${H - pad + 14}" text-anchor="middle" class="lab">${escapeHtml(String(p.label).slice(0, 14))}</text>`).join("") : "";
  return `<div class="learn-card"><p><b>${md(v.title)}</b> <span class="hint">${md(v.y_label || "")} · every number is in its claim: ${v.claims.map(escapeHtml).join(", ")}</span></p>
    <div class="svg-wrap"><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img">
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" class="axis"/>${line}${bars}</svg></div></div>`;
}
