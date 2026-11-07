// Small SVG charts for the home screen's 학습 그래프 (7 days).

const W = 340, H = 220, L = 34, R = 10, T = 14, B = 42;
const DAYS = ["일", "월", "화", "수", "목", "금", "토"];

function niceMax(v, steps = 4) {
  if (v <= 0) return steps * 5;
  const raw = v / steps;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  return step * steps;
}

function axes(days, max, unit, ticks = 4, highlight = -1) {
  let g = "";
  for (let k = 0; k <= ticks; k++) {
    const y = T + (H - T - B) * (1 - k / ticks);
    g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#e8edf3" stroke-width="1"/>`;
    const v = (max * k) / ticks;
    g += `<text x="${L - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#9aa4b2">${Number.isInteger(v) ? v : v.toFixed(1)}</text>`;
  }
  if (unit) g += `<text x="${L - 30}" y="${T - 4}" font-size="9.5" fill="#9aa4b2">${unit}</text>`;
  const step = (W - L - R) / days.length;
  days.forEach((iso, i) => {
    const d = new Date(iso + "T00:00:00");
    const x = L + step * (i + 0.5);
    const on = i === highlight;
    g += `<text x="${x}" y="${H - B + 18}" text-anchor="middle" font-size="12" font-weight="700" fill="${on ? "#0a6bff" : "#4b5567"}">${DAYS[d.getDay()]}</text>`;
    g += `<text x="${x}" y="${H - B + 32}" text-anchor="middle" font-size="9.5" fill="${on ? "#0a6bff" : "#9aa4b2"}">${String(d.getMonth() + 1).padStart(2, "0")}.${String(d.getDate()).padStart(2, "0")}</text>`;
  });
  return { g, step };
}

export function lineChart({ days, values, unit = "", color = "#2f7bff", fill = "#6aa7ff", maxValue, highlight = -1 }) {
  const real = values.filter((v) => v !== null && v !== undefined);
  const max = maxValue || niceMax(Math.max(0, ...real));
  const { g, step } = axes(days, max, unit, 4, highlight);
  const pts = values.map((v, i) => (v === null || v === undefined ? null : [L + step * (i + 0.5), T + (H - T - B) * (1 - v / max)]));
  const id = "g" + Math.random().toString(36).slice(2, 8);
  let path = "", area = "", runs = [], cur = [];
  for (const p of pts) { if (p) cur.push(p); else if (cur.length) { runs.push(cur); cur = []; } }
  if (cur.length) runs.push(cur);
  for (const r of runs) {
    path += "M" + r.map((p) => p.join(",")).join("L");
    if (r.length > 1) area += `M${r[0][0]},${H - B}L` + r.map((p) => p.join(",")).join("L") + `L${r[r.length - 1][0]},${H - B}Z`;
  }
  const dots = pts.map((p, i) => (p ? `<circle cx="${p[0]}" cy="${p[1]}" r="${i === highlight ? 5 : 4}" fill="${color}" stroke="#fff" stroke-width="2"/>` : "")).join("");
  const svg = `<svg viewBox="0 0 ${W} ${H}" role="img">
    <defs><linearGradient id="${id}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${fill}" stop-opacity=".55"/><stop offset="1" stop-color="${fill}" stop-opacity="0"/></linearGradient></defs>
    ${g}<path d="${area}" fill="url(#${id})"/><path d="${path}" fill="none" stroke="${color}" stroke-width="2.4" stroke-linejoin="round"/>${dots}</svg>`;
  const anchor = highlight >= 0 && pts[highlight] ? { x: (pts[highlight][0] / W) * 100, y: (pts[highlight][1] / H) * 100 } : null;
  return { svg, anchor };
}

export function stackedBars({ days, series, unit = "" }) {
  const totals = days.map((_, i) => series.reduce((s, x) => s + (x.values[i] || 0), 0));
  const max = niceMax(Math.max(0, ...totals));
  const { g, step } = axes(days, max, unit);
  const bw = Math.min(22, step * 0.55);
  let bars = "";
  days.forEach((_, i) => {
    let y = H - B;
    const x = L + step * (i + 0.5) - bw / 2;
    series.forEach((s, k) => {
      const v = s.values[i] || 0;
      if (!v) return;
      const h = ((H - T - B) * v) / max;
      y -= h;
      const top = k === series.length - 1 || series.slice(k + 1).every((n) => !(n.values[i] || 0));
      bars += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" rx="${top ? 4 : 0}" fill="${s.color}"/>`;
    });
  });
  const legend = series.map((s) => `<span style="display:inline-flex;align-items:center;gap:5px;margin:0 7px;font-size:11.5px;color:#4b5567;font-weight:600"><i style="width:10px;height:10px;border-radius:3px;background:${s.color};display:inline-block"></i>${s.name}</span>`).join("");
  return { svg: `<svg viewBox="0 0 ${W} ${H}" role="img">${g}${bars}</svg>`, legend };
}
