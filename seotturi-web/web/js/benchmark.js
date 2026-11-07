// Benchmark page: learning curves, accent-zone results, per-region errors and the full table,
// re-read every minute (the engine benchmarks each new checkpoint of the 5-hourly update cycle),
// and the Whisper accent classifier's experiments (/api/accent).

const $ = (s, r = document) => r.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const ZONES = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "전라", "제주"];
const SHORT = { "서울·경기": "서울경기", "강원": "강원", "충청": "충청", "대구·경북": "대구경북", "부산·울산·경남": "부울경", "전라": "전라", "제주": "제주", "-": "기타" };
const REGIONS = ["강원", "경상", "제주", "충청"];
const RAMP = ["#f1f6fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim();
const tip = $("#tip");

let data = null, status = null;

function label(b) {
  if (b.run === "base") return "Gemma 4 (학습 전)";
  if (b.run === "lora_v1") return "v1 (4개 권역)";
  if (b.run === "lora_v2r") return b.final ? "v2r 최종" : `v2r ${b.step.toLocaleString()} 스텝`;
  if (b.run === "lora_v3") return b.final ? "v3 최종" : `v3 ${b.step.toLocaleString()} 스텝`;
  if (b.run === "lora_r64c") return "r64c (5단계)";
  if (b.run === "lora_s6c") return "s6c (6단계)";
  return b.id;
}
// "lora_s6c@final", "lora_v3@750" -> the same label as the item
const labelOfId = (id) => {
  const [run, v] = id.split("@");
  return label({ id, run, final: v === "final", step: v === "final" ? 0 : Number(v) || 0 });
};
const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);
const ci = (c) => (c ? `${(c[0] * 100).toFixed(1)}–${(c[1] * 100).toFixed(1)}` : "");
const sentences = (b) => b.cumulative_step * 16;
const hm = (ts) => { const d = new Date(ts * 1000); return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; };
const fmtCount = (n) => (n >= 10000 ? `${+(n / 10000).toFixed(n % 10000 ? 1 : 0)}만` : n.toLocaleString());
// Expected end of training: when the trainer last reported its speed + the hours it said were left,
// rounded to 10 minutes since the step time drifts by a few percent.
const endOf = (t) => {
  const d = new Date(Math.round((t.updated + t.eta_hours * 3600) / 600) * 600 * 1000);
  return `${d.getMonth() + 1}/${d.getDate()}(${"일월화수목금토"[d.getDay()]}) ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
};

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

function niceMax(v, steps = 5) {
  if (v <= 0) return steps;
  const raw = v / steps, mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((m) => m * mag).find((s) => s >= raw * 0.999) * steps;
}

function showTip(x, y, title, rows) {
  tip.replaceChildren();
  const t = document.createElement("div");
  t.className = "t";
  t.textContent = title;
  tip.append(t);
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "r";
    if (r.color) { const i = document.createElement("i"); i.style.background = r.color; row.append(i); }
    const b = document.createElement("b");
    b.textContent = r.value;
    const s = document.createElement("span");
    s.textContent = r.name;
    row.append(b, s);
    tip.append(row);
  }
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(window.innerWidth - w - 12, x + 14)}px`;
  tip.style.top = `${Math.max(8, y - h - 12)}px`;
}
const hideTip = () => { tip.hidden = true; };

// ---- line chart over training volume (one y-axis; CI as whiskers) -------------------------
function lineChart(host, { items, series, yMax, refLine, servedId, unitPct = true }) {
  host.replaceChildren();
  const W = Math.max(320, host.clientWidth), H = 300, L = 44, R = 18, T = 26, B = 46;
  const xs = items.map(sentences);
  const xMax = niceMax(Math.max(...xs, 1) * 1.04, 4);
  const X = (v) => L + ((W - L - R) * v) / xMax;
  const Y = (v) => T + (H - T - B) * (1 - v / yMax);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": series.map((s) => s.name).join(", ") + " 학습 곡선" });
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted");
  for (let k = 0; k <= 5; k++) {
    const v = (yMax * k) / 5, y = Y(v);
    svg.append(el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: k ? grid : axis, "stroke-width": 1 }));
    svg.append(el("text", { x: L - 8, y: y + 4, "text-anchor": "end", "font-size": 11, fill: muted }, unitPct ? `${Math.round(v * 100)}%` : v));
  }
  for (let k = 0; k <= 4; k++) {
    const v = (xMax * k) / 4;
    svg.append(el("text", { x: X(v), y: H - B + 18, "text-anchor": "middle", "font-size": 11, fill: muted }, fmtCount(v)));
  }
  svg.append(el("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", "font-size": 11.5, fill: muted }, "누적 학습 문장 수"));
  if (refLine) {
    svg.append(el("line", { x1: L, x2: W - R, y1: Y(refLine.value), y2: Y(refLine.value), stroke: muted, "stroke-width": 1 }));
    svg.append(el("text", { x: W - R, y: Y(refLine.value) - 6, "text-anchor": "end", "font-size": 11, fill: muted }, refLine.label));
  }
  const pts = items.map((b, i) => ({ b, x: X(xs[i]) }));
  for (const s of series) {
    const have = pts.filter((p) => s.get(p.b));
    if (!have.length) continue;
    svg.append(el("path", { d: "M" + have.map((p) => `${p.x},${Y(s.get(p.b).value)}`).join("L"), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    for (const p of have) {
      const m = s.get(p.b);
      if (m.ci) {
        svg.append(el("line", { x1: p.x, x2: p.x, y1: Y(m.ci[0]), y2: Y(m.ci[1]), stroke: s.color, "stroke-width": 1.5, opacity: 0.45 }));
        for (const c of m.ci) svg.append(el("line", { x1: p.x - 4, x2: p.x + 4, y1: Y(c), y2: Y(c), stroke: s.color, "stroke-width": 1.5, opacity: 0.45 }));
      }
      svg.append(el("circle", { cx: p.x, cy: Y(m.value), r: 4.5, fill: s.color, stroke: "#fff", "stroke-width": 2 }));
    }
  }
  const served = pts.find((p) => p.b.id === servedId);
  if (served) {
    const tops = series.map((s) => s.get(served.b)).filter(Boolean).map((m) => Y(m.ci ? m.ci[1] : m.value));
    if (tops.length) svg.append(el("text", { x: served.x, y: Math.max(12, Math.min(...tops) - 10), "text-anchor": "middle", "font-size": 11, "font-weight": 700, fill: css("--ink-2") }, "서비스 중"));
  }
  // crosshair + tooltip: snaps to the nearest measured model
  const cross = el("line", { y1: T, y2: H - B, stroke: axis, "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent", tabindex: 0 });
  let focusIdx = pts.length - 1;
  const show = (p, cx, cy) => {
    cross.setAttribute("x1", p.x); cross.setAttribute("x2", p.x); cross.setAttribute("visibility", "visible");
    showTip(cx, cy, `${label(p.b)} · 누적 ${fmtCount(sentences(p.b))} 문장`, series.map((s) => {
      const m = s.get(p.b);
      return { color: s.color, name: s.name, value: m ? `${unitPct ? pct(m.value) : m.value.toFixed(1)}${m.ci ? ` (${ci(m.ci)})` : ""}` : "측정 안 함" };
    }));
  };
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect(), x = ((e.clientX - r.left) * W) / r.width;
    const p = pts.reduce((a, q) => (Math.abs(q.x - x) < Math.abs(a.x - x) ? q : a));
    show(p, e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  hit.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    focusIdx = Math.max(0, Math.min(pts.length - 1, focusIdx + (e.key === "ArrowRight" ? 1 : -1)));
    const r = svg.getBoundingClientRect();
    show(pts[focusIdx], r.left + (pts[focusIdx].x * r.width) / W, r.top + 40);
  });
  hit.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  svg.append(hit);
  host.append(svg);
}

// ---- grouped bars: CER per regional set for one model --------------------------------------
function regionBars(host, b) {
  host.replaceChildren();
  const pr = b.metrics.asr.per_region;
  const W = Math.max(320, host.clientWidth), H = 280, L = 44, R = 10, T = 22, B = 34;
  const vals = REGIONS.flatMap((r) => [pr[r].cer_dialect, pr[r].cer_standard]);
  const yMax = niceMax(Math.max(...vals) * 1.1, 4);
  const Y = (v) => T + (H - T - B) * (1 - v / yMax);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "권역별 문자 오류율" });
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted"), ink2 = css("--ink-2");
  for (let k = 0; k <= 4; k++) {
    const v = (yMax * k) / 4, y = Y(v);
    svg.append(el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: k ? grid : axis, "stroke-width": 1 }));
    svg.append(el("text", { x: L - 8, y: y + 4, "text-anchor": "end", "font-size": 11, fill: muted }, `${Math.round(v * 100)}%`));
  }
  const band = (W - L - R) / REGIONS.length, bw = Math.min(24, band * 0.28);
  const series = [{ key: "cer_dialect", name: "방언 받아쓰기", color: css("--s1") }, { key: "cer_standard", name: "표준어 변환", color: css("--s2") }];
  REGIONS.forEach((reg, i) => {
    const cx = L + band * (i + 0.5);
    svg.append(el("text", { x: cx, y: H - B + 20, "text-anchor": "middle", "font-size": 12.5, "font-weight": 600, fill: ink2 }, `${reg} (${pr[reg].n})`));
    series.forEach((s, k) => {
      const v = pr[reg][s.key], x = cx + (k === 0 ? -bw - 1 : 1), y = Y(v), h = Y(0) - y, r = Math.min(4, h);
      const path = el("path", { d: `M${x},${Y(0)}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${Y(0)}Z`, fill: s.color, tabindex: 0 });
      const on = (cxp, cyp) => { path.setAttribute("opacity", 0.8); showTip(cxp, cyp, `${reg} · ${s.name}`, [{ color: s.color, name: "문자 오류율", value: pct(v) }]); };
      path.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
      path.addEventListener("focus", () => { const r2 = path.getBoundingClientRect(); on(r2.right, r2.top); });
      for (const ev of ["pointerleave", "blur"]) path.addEventListener(ev, () => { path.removeAttribute("opacity"); hideTip(); });
      svg.append(path);
      svg.append(el("text", { x: x + bw / 2, y: y - 5, "text-anchor": "middle", "font-size": 10.5, fill: ink2 }, (v * 100).toFixed(1)));
    });
  });
  host.append(svg);
}

// ---- confusion matrix (rows: speaker's zone, columns: predicted zone) ------------------------
function confusion(host, z, zones = ZONES, withOther = true) {
  host.replaceChildren();
  const rows = zones.filter((g) => z.confusion[g]);
  const cols = withOther ? [...zones, "-"] : zones;
  const t = document.createElement("table");
  const head = t.insertRow();
  head.append(Object.assign(document.createElement("th"), { textContent: "정답 \\ 예측" }));
  for (const c of cols) head.append(Object.assign(document.createElement("th"), { textContent: SHORT[c] }));
  head.append(Object.assign(document.createElement("th"), { textContent: "F1" }));
  for (const g of rows) {
    const tr = t.insertRow();
    tr.append(Object.assign(document.createElement("th"), { className: "row", textContent: SHORT[g] }));
    const counts = z.confusion[g], n = Object.values(counts).reduce((a, c) => a + c, 0);
    for (const c of cols) {
      const k = c === "-" ? Object.entries(counts).filter(([p]) => !zones.includes(p)).reduce((a, [, v]) => a + v, 0) : counts[c] || 0;
      const p = n ? k / n : 0;
      const td = tr.insertCell();
      td.textContent = k || "";
      td.tabIndex = 0;
      const step = Math.min(RAMP.length - 1, Math.round(p * (RAMP.length - 1)));
      td.style.background = RAMP[step];
      td.style.color = step >= 4 ? "#fff" : css("--ink");
      if (c === g) td.style.boxShadow = "inset 0 0 0 2px #0f1b33";
      const on = (x, y) => showTip(x, y, `정답 ${g} → 예측 ${c === "-" ? "기타·형식 밖" : c}`, [{ name: `${n}문장 중`, value: `${k}개 (${(p * 100).toFixed(0)}%)` }]);
      td.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
      td.addEventListener("focus", () => { const r = td.getBoundingClientRect(); on(r.right, r.top); });
      for (const ev of ["pointerleave", "blur"]) td.addEventListener(ev, hideTip);
    }
    const f = tr.insertCell();
    f.className = "f1";
    f.textContent = (z.per_zone_f1[g] * 100).toFixed(0);
  }
  host.append(t);
}

function kpis(b, base) {
  const a = b.metrics.asr, zone = b.metrics.zone_plain, ba = base && base.metrics.asr;
  const tiles = [
    { label: "방언 받아쓰기 CER", v: a.cer_dialect.value, c: a.cer_dialect.ci, b: ba && ba.cer_dialect.value, lower: true, p: true },
    { label: "표준어 변환 CER", v: a.cer_standard.value, c: a.cer_standard.ci, b: ba && ba.cer_standard.value, lower: true, p: true },
    { label: "chrF++ (표준어)", v: a.chrfpp_standard, b: ba && ba.chrfpp_standard, lower: false, p: false },
    { label: "사투리 어휘 교정률", v: a.dialect_word_recall, b: ba && ba.dialect_word_recall, lower: false, p: true },
    { label: "정규화 오류감소율", v: a.err, b: ba && ba.err, lower: false, p: true },
    zone ? { label: "억양 지역 Macro-F1", v: zone.macro_f1.value, c: zone.macro_f1.ci, lower: false, p: true, note: "무작위 수준 14%" }
      : { label: "억양 지역 Macro-F1", v: null, note: "v2r 체크포인트부터 측정" },
  ];
  $("#kpis").replaceChildren(...tiles.map(tile));
}

// One KPI tile: t = { label, v, c (CI), b (value compared with), lower, p (percent), digits, vs, note }
function tile(t) {
  const d = document.createElement("div");
  d.className = "kpi";
  const digits = t.digits || 1;
  const val = t.v === null || t.v === undefined ? "—" : t.p ? (t.v * 100).toFixed(digits) : t.v.toFixed(digits);
  let delta = "", cls = "flat";
  if (t.v !== null && t.v !== undefined && t.b !== undefined && t.b !== null) {
    const diff = t.p ? (t.v - t.b) * 100 : t.v - t.b;
    const good = t.lower ? diff < 0 : diff > 0;
    cls = Math.abs(diff) < 0.5 * 10 ** -digits ? "flat" : good ? "good" : "bad";
    delta = `${diff < 0 ? "▼" : "▲"} ${Math.abs(diff).toFixed(digits)}${t.p ? "%p" : ""} ${t.vs || "학습 전 대비"}`;
  }
  d.innerHTML = `<div class="label"></div><div class="value"></div><div class="ci"></div><div class="delta ${cls}"></div>`;
  d.querySelector(".label").textContent = t.label;
  d.querySelector(".value").innerHTML = `${val}${val !== "—" && t.p ? "<small>%</small>" : ""}`;
  d.querySelector(".ci").textContent = t.c ? `95% CI ${ci(t.c)}` : t.note || "";
  d.querySelector(".delta").textContent = delta || (t.c && t.note ? t.note : "");
  return d;
}

function table(items, servedId, history) {
  const verdict = Object.fromEntries((history || []).map((h) => [h.candidate, h.result]));
  const cols = ["모델", "누적 학습 문장", "방언 CER", "표준어 CER", "표준어 sWER", "표준어 WER", "chrF++", "ERR", "교정률", "형식 준수", "억양 F1", "억양 정확도", "RTF", "측정"];
  const t = $("#table");
  t.replaceChildren();
  const head = t.createTHead().insertRow();
  for (const c of cols) head.append(Object.assign(document.createElement("th"), { textContent: c }));
  const body = t.createTBody();
  for (const b of items) {
    const a = b.metrics.asr, z = b.metrics.zone_plain;
    const tr = body.insertRow();
    if (b.id === servedId) tr.className = "served";
    const name = tr.insertCell();
    name.textContent = label(b);
    if (b.id === servedId) name.append(Object.assign(document.createElement("span"), { className: "badge", textContent: "서비스 중" }));
    else if (verdict[b.id] === "kept") name.append(Object.assign(document.createElement("span"), { className: "badge kept", textContent: "보류" }));
    const cell = (v, extra) => {
      const td = tr.insertCell();
      td.textContent = v;
      if (extra) td.append(Object.assign(document.createElement("div"), { className: "ci", textContent: extra }));
    };
    cell(sentences(b).toLocaleString());
    cell(pct(a.cer_dialect.value), ci(a.cer_dialect.ci));
    cell(pct(a.cer_standard.value), ci(a.cer_standard.ci));
    cell(pct(a.swer_standard.value), ci(a.swer_standard.ci));
    cell(pct(a.wer_standard.value));
    cell(a.chrfpp_standard.toFixed(1));
    cell(a.err === null ? "—" : `${(a.err * 100).toFixed(1)}%`);
    cell(pct(a.dialect_word_recall, 0));
    cell(pct(a.format_ok, 0));
    cell(z ? pct(z.macro_f1.value) : "—", z ? ci(z.macro_f1.ci) : "");
    cell(z ? pct(z.accuracy.value) : "—");
    cell(b.metrics.rtf === undefined ? "—" : b.metrics.rtf.toFixed(2));
    cell(b.time ? hm(b.time) : "—");
  }
}

function meta(d, s) {
  const chips = [];
  const set = d.set || {};
  chips.push(`<span><i></i>시험 문장 ${set.asr || 160} + 억양 ${set.zone || 105} (학습 미사용)</span>`);
  const last = d.items.filter((b) => b.run !== "base").slice(-1)[0];
  if (last) chips.push(`<span>최근 측정 ${hm(last.time)} · ${label(last)}</span>`);
  if (d.progress) chips.push(`<span><i class="run"></i>측정 중 · ${labelOfId(d.progress.id)} ${d.progress.done}/${d.progress.total}</span>`);
  if (d.next_check) chips.push(`<span>다음 확인 ${hm(d.next_check)} (${d.update_hours}시간 주기)</span>`);
  if (d.training) chips.push(`<span>학습 ${((100 * d.training.step) / d.training.total).toFixed(1)}% · ${d.training.step.toLocaleString()}/${d.training.total.toLocaleString()} 스텝</span>`);
  if (d.training && d.training.eta_hours != null && d.training.updated) {
    // The trainer reports every ~2 minutes; silence for 30 minutes means it is not running.
    if (Date.now() / 1000 - d.training.updated > 1800) {
      chips.push(`<span class="eta"><i class="off"></i>학습 멈춤 · 마지막 기록 ${hm(d.training.updated)}</span>`);
    } else {
      chips.push(`<span class="eta"><i class="run"></i>학습 종료 예정 <b>${endOf(d.training)}</b> · 약 ${Math.round(d.training.eta_hours)}시간 남음</span>`);
    }
  }
  // Only the static copy (GitHub Pages) adds snapshot_time, when it shows saved results.
  if (d.snapshot_time) chips.unshift(`<span><i class="off"></i>AI 서버 꺼짐 · ${hm(d.snapshot_time)} 기록 기준</span>`);
  $("#meta").innerHTML = chips.join("");
  $("#hours").textContent = d.update_hours || 5;
}

// ---- accent classifier (Whisper): every experiment on its held-out tests (/api/accent) -------------
// Written by accent-id/bench_export.py after each run; the app's accent verdict comes from the served one.
const ACC_ZONES = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "제주"];
const ACC_BADGE = { served: ["서비스 중", ""], was_served: ["이전 서비스", "kept"], member: ["앙상블 구성", "member"],
  replaced: ["교체됨", "off"], not_adopted: ["미채택", "off"], reviewed: ["검토만", "off"],
  new: ["새 실험", ""], experiment: ["실험", "off"] };
let acc = null, accTest = "test_plain", accPick = null, accTick = 0;

// Since 10/9 the app runs a four-region system outside these six-zone items (acc.current); the history then
// highlights the last six-zone run served.
const accHighlight = () => (acc.current ? acc.previous_served : acc.served);
const accServed = () => acc.items.find((i) => i.id === accHighlight()) || acc.items[acc.items.length - 1];
const accTestOf = (id) => acc.tests.find((t) => t.id === id);
const rate = ([k, n]) => (n ? k / n : null);

function gemmaOf() {
  if (!acc || !acc.gemma || !acc.gemma.length) return null;
  return acc.gemma.find((g) => data && g.id === data.served) || acc.gemma[acc.gemma.length - 1];
}
const gemmaLabel = (g) => { const [run, step] = g.id.split("@"); return label({ run, step: +step, final: step === "final" }); };

function accBadge(it) {
  const [text, cls] = ACC_BADGE[it.status] || ["", ""];
  return text ? Object.assign(document.createElement("span"), { className: `badge ${cls}`.trim(), textContent: text }) : null;
}

function accMeta() {
  const s = accServed();
  const chip = (text, dot) => {
    const c = document.createElement("span");
    if (dot) c.append(document.createElement("i"));
    c.append(text);
    return c;
  };
  $("#acc-meta").replaceChildren(
    acc.current ? chip(`서비스 중 · ${acc.current.name} · ${hm(acc.current.since)}부터`, true) : chip(`서비스 중 · ${s.name} · 표준어 보정 +${s.bias}`, true),
    chip(`실험 ${acc.items.length}개 · 시험 ${acc.tests.length}종 ${acc.tests.reduce((a, t) => a + t.n, 0).toLocaleString()}문장`),
    chip(`마지막 갱신 ${hm(acc.generated)} · 실험이 끝날 때마다 갱신`));
}

function accKpis() {
  const s = accServed(), first = acc.items[0], t = s.tests.test_plain, f = first.tests.test_plain, g = gemmaOf();
  const vs = "첫 1단계 대비";
  const tiles = [
    { label: "억양 Macro-F1", v: t.macro_f1.value, c: t.macro_f1.ci, b: f.macro_f1.value, p: true, vs },
    { label: "정확도", v: t.accuracy.value, c: t.accuracy.ci, b: f.accuracy.value, p: true, vs },
    { label: "표준어·지역 구분 AUC", v: t.auc, b: f.auc, digits: 3, vs, note: "0.5 무작위 · 1 완벽" },
    { label: "서울·경기 → 지역 오판", v: rate(t.standard_called_regional), b: rate(f.standard_called_regional), p: true, lower: true, vs,
      note: `서울·경기 ${t.standard_called_regional[1]}문장 중 ${t.standard_called_regional[0]}개` },
    { label: "지역 → 표준어 놓침", v: rate(t.regional_called_standard), b: rate(f.regional_called_standard), p: true, lower: true, vs,
      note: `지역 ${t.regional_called_standard[1].toLocaleString()}문장 중 ${t.regional_called_standard[0]}개` },
  ];
  if (g) tiles.push({ label: "Gemma 벤치 90문장 F1", v: s.tests.bench.macro_f1.value, c: s.tests.bench.macro_f1.ci, b: g.macro_f1.value, p: true,
    vs: "Gemma 대비" });
  $("#acc-kpis").replaceChildren(...tiles.map(tile));
}

// ---- the system served since 10/9: four regions + a standard-accent detector (bench_export.py current) ----
function accNow() {
  const cur = acc.current;
  $("#acc-now").hidden = !cur;
  for (const n of document.querySelectorAll(".acc-served-label")) n.textContent = cur ? "직전 서비스" : "서비스 중";
  if (!cur) {
    $("#acc-intro-now").textContent = "6개 권역(서울·경기, 강원, 충청, 대구·경북, 부산·울산·경남, 제주)을 문장 하나만 듣고 고르는 과제이며, 무작위로 고르면 Macro-F1 약 17%입니다.";
    return;
  }
  $("#acc-intro-now").textContent = `${hm(cur.since)}부터는 서울·경기를 학습하지 않은 4지역 모델(강원·충청·경상·제주)이 어느 지역인지를, 따로 만든 표준 억양 판별기가 지역 억양이 있는지를 판정합니다.`;
  $("#acc-history-when").textContent = `· ${hm(cur.since)}까지 서비스`;
  const row = (n) => cur.conditions.find((c) => c.name === n);
  const o = row("원본"), g = row("무음화"), vs = "이전 서비스 대비";
  const tiles = [
    { label: "4지역 Macro-F1", v: cur.tests.test_plain.macro_f1, p: true, note: "사투리 단어 없는 문장 · 무작위 25%" },
    { label: "표준어·지역 구분 AUC", v: o.detector.served.auc, b: o.detector.previous.auc, digits: 3, vs },
    { label: "서울·경기 → 지역 오판", v: o.detector.served.standard_called_regional, b: o.detector.previous.standard_called_regional, p: true, lower: true, vs },
    { label: "지역 억양 잡아냄", v: o.detector.served.regional_caught, b: o.detector.previous.regional_caught, p: true, vs },
    { label: "무음화에서 지역 맞힘", v: g.region.served, b: g.region.previous, p: true, vs },
    { label: "무음화에서 서울·경기 오판", v: g.detector.served.standard_called_regional, b: g.detector.previous.standard_called_regional, p: true, lower: true, vs },
  ];
  $("#acc-now-kpis").replaceChildren(...tiles.map(tile));
  $("#acc-now-note").textContent = `서비스 중: ${cur.name} · 지역 ${cur.region_run}, 판별기 ${cur.detector_run} (${cur.detector_members.map((m) => `${m.run} ${m.weight}`).join(" + ")}, 서울·경기 보정 ${cur.detector_bias ? `+${cur.detector_bias}` : "없음"}) · 무음화 = 단어 사이를 −40 dB로 낮춤(브라우저 잡음 제거 흉내) · 비교 기준은 10/9까지 서비스한 6권역 앙상블`;
  const f3 = (x) => (x === null || x === undefined ? "—" : x.toFixed(3));
  const build = (id, cols, cells) => {
    const t = $(id);
    t.replaceChildren();
    const head = t.createTHead().insertRow();
    for (const c of cols) head.append(Object.assign(document.createElement("th"), { textContent: c }));
    const body = t.createTBody();
    for (const c of cur.conditions) {
      const tr = body.insertRow(), name = tr.insertCell();
      name.append(c.name, Object.assign(document.createElement("div"), { className: "desc", textContent: c.desc }));
      for (const [v, now] of cells(c)) Object.assign(tr.insertCell(), { textContent: v, className: now ? "now" : "" });
    }
  };
  build("#acc-now-region", ["조건", "이전 서비스 (6권역, 서울 제외)", "4지역 · 변형 없이 학습", "4지역 · 서비스 중"],
    (c) => [[pct(c.region.previous)], [pct(c.region.noaug)], [pct(c.region.served), true]]);
  build("#acc-now-detector", ["조건", "AUC 이전", "AUC 두 칸", "AUC 6권역 변형", "AUC 서비스 중", "잡아냄 지금", "서울 오판 지금", "잡아냄 이전", "서울 오판 이전"],
    (c) => {
      const x = (k, f) => (c.detector[k] ? c.detector[k][f] : null);
      return [[f3(x("previous", "auc"))], [f3(x("binary", "auc"))], [f3(x("zone6_aug", "auc"))], [f3(x("served", "auc")), true],
        [pct(x("served", "regional_caught"), 0), true], [pct(x("served", "standard_called_regional"), 0), true],
        [pct(x("previous", "regional_caught"), 0)], [pct(x("previous", "standard_called_regional"), 0)]];
    });
  $("#acc-now-notes").replaceChildren(...cur.notes.map((n) => Object.assign(document.createElement("li"), { textContent: n })));
  accRuns();
}

// Every four-region run and standard detector in accent-id runs/ (bench_export.py other_runs, newest last).
function accRuns() {
  const list = (id, rows, cols, cells) => {
    const t = $(id);
    t.replaceChildren();
    const head = t.createTHead().insertRow();
    for (const c of cols) head.append(Object.assign(document.createElement("th"), { textContent: c }));
    const body = t.createTBody();
    for (const r of rows || []) {
      const tr = body.insertRow(), name = tr.insertCell();
      if (r.status === "served") tr.className = "served";
      name.append(r.name);
      const b = accBadge(r);
      if (b) name.append(b);
      name.append(Object.assign(document.createElement("div"), { className: "ci", textContent: r.id }));
      for (const v of cells(r)) tr.insertCell().textContent = v;
      tr.insertCell().textContent = r.time ? hm(r.time) : "—";
    }
  };
  const f3 = (x) => (x === null || x === undefined ? "—" : x.toFixed(3));
  list("#acc-runs4", acc.runs4, ["4지역 실험", "단어 없는 문장 F1", "같은 문장 F1", "단어 포함 F1", "원본", "사이트+잡음제거", "무음화", "배경만 (낮을수록 좋음)", "측정"],
    (r) => [pct(r.tests.test_plain), pct(r.tests.test_same), pct(r.tests.test_dialect),
      ...["원본", "사이트+잡음제거", "무음화", "배경만"].map((c) => pct(r.conditions[c]))]);
  list("#acc-detectors", acc.detectors, ["표준 억양 판별기", "서울·경기 보정", "AUC 원본", "AUC 사이트 전처리", "AUC 잡음제거", "AUC 무음화", "AUC 앞 2초", "잡아냄 (원본)", "서울 오판 (원본)", "측정"],
    (r) => {
      const c = (n) => r.conditions[n] || {};
      return [r.bias ? `+${r.bias}` : "없음", ...["원본", "사이트 전처리", "사이트+잡음제거", "무음화", "앞 2초"].map((n) => f3(c(n).auc)),
        pct(c("원본").regional_caught, 0), pct(c("원본").standard_called_regional, 0)];
    });
}

function accPickItem(id) {
  accPick = id;
  renderAccent();
}

// ---- horizontal bars: Macro-F1 of every experiment on one test (CI whiskers, dev as a ring) --------
function accBars(host) {
  host.replaceChildren();
  const items = acc.items, test = accTestOf(accTest);
  const W = Math.max(300, host.clientWidth), narrow = W < 420;
  const rowH = 34, L = narrow ? 116 : 140, R = 14, T = 6, B = 28, H = T + B + rowH * items.length;
  const X = (v) => L + (W - L - R) * v;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `실험별 Macro-F1, ${test.name}` });
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted"), ink = css("--ink"), ink2 = css("--ink-2");
  for (let k = 0; k <= 5; k++) {
    const x = X(k / 5);
    svg.append(el("line", { x1: x, x2: x, y1: T, y2: H - B, stroke: k ? grid : axis, "stroke-width": 1 }));
    svg.append(el("text", { x, y: H - B + 17, "text-anchor": "middle", "font-size": 11, fill: muted }, `${k * 20}%`));
  }
  svg.append(el("line", { x1: X(1 / 6), x2: X(1 / 6), y1: T, y2: H - B, stroke: muted, "stroke-width": 1, "stroke-dasharray": "3 3" }));
  items.forEach((it, i) => {
    const r = it.tests[accTest], m = r.macro_f1, y = T + rowH * i + rowH / 2, served = it.id === accHighlight();
    const g = el("g", { tabindex: 0, role: "button", "aria-label": `${it.name} ${pct(m.value)}`, style: "cursor:pointer;outline:none" });
    g.append(el("rect", { x: 0, y: y - rowH / 2 + 1, width: W, height: rowH - 2, rx: 8, fill: it.id === accPick ? "#eef5ff" : "transparent" }));
    g.append(el("text", { x: L - 10, y: y + 4, "text-anchor": "end", "font-size": narrow ? 11.5 : 12.5, "font-weight": served ? 800 : 600, fill: served ? ink : ink2 }, it.short || it.name));
    g.append(el("rect", { x: X(0), y: y - 9, width: Math.max(1, X(m.value) - X(0)), height: 18, rx: 4, fill: served ? css("--s3") : "#a9bbd3" }));
    g.append(el("text", { x: X(0) + 7, y: y + 4, "font-size": 11.5, "font-weight": 750, fill: "#fff" }, (m.value * 100).toFixed(1)));
    g.append(el("line", { x1: X(m.ci[0]), x2: X(m.ci[1]), y1: y, y2: y, stroke: ink, "stroke-width": 1.3, opacity: 0.6 }));
    for (const c of m.ci) g.append(el("line", { x1: X(c), x2: X(c), y1: y - 5, y2: y + 5, stroke: ink, "stroke-width": 1.3, opacity: 0.6 }));
    if (it.dev_macro_f1 !== null) g.append(el("circle", { cx: X(it.dev_macro_f1), cy: y, r: 4.5, fill: "#fff", stroke: css("--s1"), "stroke-width": 2 }));
    const on = (cx, cy) => showTip(cx, cy, it.name, [
      { color: served ? css("--s3") : "#a9bbd3", name: `${test.name} Macro-F1`, value: `${pct(m.value)} (${ci(m.ci)})` },
      { name: "정확도", value: pct(r.accuracy.value) },
      { color: css("--s1"), name: "검증(dev) Macro-F1", value: it.dev_macro_f1 === null ? "— (앙상블)" : pct(it.dev_macro_f1) },
      { name: "표준어 보정", value: it.bias ? `+${it.bias}` : "없음" },
      { name: it.desc, value: "" },
    ]);
    g.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
    g.addEventListener("focus", () => { const b = g.getBoundingClientRect(); on(b.left + b.width * 0.6, b.top); });
    for (const ev of ["pointerleave", "blur"]) g.addEventListener(ev, hideTip);
    g.addEventListener("click", () => accPickItem(it.id));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); accPickItem(it.id); } });
    svg.append(g);
  });
  host.append(svg);
}

// ---- learning curve: phase 1 trained on more data (log-scale x), dev vs the held-out tests ----------
function accCurve(host) {
  host.replaceChildren();
  const pts = acc.curve;
  const W = Math.max(320, host.clientWidth), H = 300, L = 44, R = 18, T = 22, B = 46;
  const series = [
    { key: "dev", name: "검증(dev)", color: css("--s1"), dash: "5 4" },
    { key: "test_plain", name: "사투리 단어 없는 문장", color: css("--s3") },
    { key: "test_same", name: "같은 문장, 다른 권역", color: css("--s2") },
  ];
  const vals = pts.flatMap((p) => series.map((s) => p[s.key]));
  const lo = Math.floor((Math.min(...vals) - 0.03) * 10) / 10, hi = Math.ceil((Math.max(...vals) + 0.03) * 10) / 10;
  const lx = Math.log2, a = lx(pts[0].clips) - 0.35, b = lx(pts[pts.length - 1].clips) + 0.35;
  const X = (v) => L + ((W - L - R) * (lx(v) - a)) / (b - a);
  const Y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "학습 데이터 양에 따른 억양 분류기 Macro-F1" });
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted");
  for (let v = lo, k = 0; v <= hi + 1e-9; v += 0.1, k++) {
    svg.append(el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: k ? grid : axis, "stroke-width": 1 }));
    svg.append(el("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, fill: muted }, `${Math.round(v * 100)}%`));
  }
  for (const p of pts) svg.append(el("text", { x: X(p.clips), y: H - B + 18, "text-anchor": "middle", "font-size": 11, fill: muted }, fmtCount(p.clips)));
  svg.append(el("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", "font-size": 11.5, fill: muted }, "학습 클립 수 (로그 눈금)"));
  for (const s of series) {
    svg.append(el("path", { d: "M" + pts.map((p) => `${X(p.clips)},${Y(p[s.key])}`).join("L"), fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round", ...(s.dash ? { "stroke-dasharray": s.dash } : {}) }));
    for (const p of pts) svg.append(el("circle", { cx: X(p.clips), cy: Y(p[s.key]), r: 4.5, fill: s.dash ? "#fff" : s.color, stroke: s.dash ? s.color : "#fff", "stroke-width": 2 }));
  }
  const cross = el("line", { y1: T, y2: H - B, stroke: axis, "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent", tabindex: 0 });
  let focusIdx = pts.length - 1;
  const show = (p, cx, cy) => {
    cross.setAttribute("x1", X(p.clips)); cross.setAttribute("x2", X(p.clips)); cross.setAttribute("visibility", "visible");
    showTip(cx, cy, `${p.label} · 클립 ${p.clips.toLocaleString()}개 · 화자 ${p.persons.toLocaleString()}명${p.seeds > 1 ? ` · ${p.seeds}회 평균` : ""}`,
      series.map((s) => ({ color: s.color, name: s.name, value: pct(p[s.key]) })));
  };
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect(), x = ((e.clientX - r.left) * W) / r.width;
    show(pts.reduce((m, q) => (Math.abs(X(q.clips) - x) < Math.abs(X(m.clips) - x) ? q : m)), e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  hit.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    focusIdx = Math.max(0, Math.min(pts.length - 1, focusIdx + (e.key === "ArrowRight" ? 1 : -1)));
    const r = svg.getBoundingClientRect();
    show(pts[focusIdx], r.left + (X(pts[focusIdx].clips) * r.width) / W, r.top + 40);
  });
  hit.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  svg.append(hit);
  host.append(svg);
}

// ---- grouped bars: Gemma's 지역: line vs the served classifier, per zone, on the same 90 clips -------
function accGemma(host) {
  host.replaceChildren();
  const g = gemmaOf(), w = accServed().tests.bench;
  if (!g) { host.innerHTML = '<div class="empty-chart">Gemma 억양 벤치마크 결과가 아직 없어요.</div>'; return; }
  const groups = [...ACC_ZONES.map((z) => ({ name: SHORT[z], full: z, a: g.per_zone_f1[z], b: w.per_zone_f1[z] })),
    { name: "Macro-F1", full: "Macro-F1", a: g.macro_f1.value, b: w.macro_f1.value, strong: true }];
  const W = Math.max(320, host.clientWidth), H = 280, L = 40, R = 8, T = 22, B = 34;
  const Y = (v) => T + (H - T - B) * (1 - v);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "같은 90문장에서 Gemma와 억양 분류기의 권역별 F1" });
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted"), ink2 = css("--ink-2");
  for (let k = 0; k <= 5; k++) {
    const v = k / 5, y = Y(v);
    svg.append(el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: k ? grid : axis, "stroke-width": 1 }));
    svg.append(el("text", { x: L - 8, y: y + 4, "text-anchor": "end", "font-size": 11, fill: muted }, `${k * 20}%`));
  }
  const band = (W - L - R) / groups.length, bw = Math.min(20, band * 0.32);
  const series = [{ key: "a", name: `Gemma 4 ${gemmaLabel(g)} '지역:' 줄`, color: "#c3cddb" }, { key: "b", name: "Whisper 억양 분류기", color: css("--s3") }];
  groups.forEach((grp, i) => {
    const cx = L + band * (i + 0.5);
    svg.append(el("text", { x: cx, y: H - B + 20, "text-anchor": "middle", "font-size": W < 520 ? 10.5 : 12, "font-weight": grp.strong ? 800 : 600, fill: ink2 }, grp.name));
    series.forEach((s, k) => {
      const v = grp[s.key] || 0, x = cx + (k === 0 ? -bw - 1 : 1), y = Y(v), h = Y(0) - y, r = Math.min(4, h);
      const path = el("path", { d: `M${x},${Y(0)}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${Y(0)}Z`, fill: s.color, tabindex: 0 });
      const on = (px, py) => { path.setAttribute("opacity", 0.8); showTip(px, py, `${grp.full} · ${s.name}`, [{ color: s.color, name: grp.strong ? "Macro-F1 (90문장)" : "F1 (15문장)", value: pct(v) }]); };
      path.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
      path.addEventListener("focus", () => { const b2 = path.getBoundingClientRect(); on(b2.right, b2.top); });
      for (const ev of ["pointerleave", "blur"]) path.addEventListener(ev, () => { path.removeAttribute("opacity"); hideTip(); });
      svg.append(path);
      svg.append(el("text", { x: x + bw / 2, y: y - 5, "text-anchor": "middle", "font-size": 10, fill: ink2 }, Math.round(v * 100)));
    });
  });
  host.append(svg);
}

function accTable() {
  const t = $("#acc-table");
  t.replaceChildren();
  const cols = ["실험", "학습 클립", "표준어 보정", "검증(dev)", ...acc.tests.map((x) => x.short || x.name), "표준·지역 AUC", "서울·경기→지역", "지역→표준어", "측정"];
  const head = t.createTHead().insertRow();
  for (const c of cols) head.append(Object.assign(document.createElement("th"), { textContent: c }));
  const body = t.createTBody();
  for (const it of acc.items) {
    const tr = body.insertRow();
    tr.tabIndex = 0;
    tr.className = [it.id === accHighlight() ? "served" : "", it.id === accPick ? "sel" : ""].join(" ").trim();
    const name = tr.insertCell();
    name.append(it.name);
    const b = accBadge(it);
    if (b) name.append(b);
    name.append(Object.assign(document.createElement("div"), { className: "desc", textContent: it.note ? `${it.desc} · ${it.note}` : it.desc }));
    const cell = (v, extra) => {
      const td = tr.insertCell();
      td.textContent = v;
      if (extra) td.append(Object.assign(document.createElement("div"), { className: "ci", textContent: extra }));
    };
    cell(it.train ? it.train.clips.toLocaleString() : "—", it.train ? `화자 ${it.train.persons.toLocaleString()}` : "앙상블");
    cell(it.bias ? `+${it.bias}` : "—");
    cell(it.dev_macro_f1 === null ? "—" : pct(it.dev_macro_f1));
    for (const x of acc.tests) cell(pct(it.tests[x.id].macro_f1.value), ci(it.tests[x.id].macro_f1.ci));
    const p = it.tests.test_plain;
    cell(p.auc === null ? "—" : p.auc.toFixed(3));
    cell(pct(rate(p.standard_called_regional)), `${p.standard_called_regional[0]}/${p.standard_called_regional[1]}`);
    cell(pct(rate(p.regional_called_standard)), `${p.regional_called_standard[0]}/${p.regional_called_standard[1]}`);
    cell(it.time ? hm(it.time) : "—");
    tr.addEventListener("click", () => accPickItem(it.id));
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") accPickItem(it.id); });
  }
}

// ---- accent classifier versions: how each was trained (accent-id README, bench_export.py descriptions) ----
const ACC_VERSIONS = [
  { id: "phase1", name: "1단계", when: "10/7 02:25~04:00 서비스", task: "6개 권역 중 하나 (서울·경기 = 표준)",
    encoder: "고정 · 33개 층 평균·표준편차를 층 가중합", from: "Whisper-large-v3", data: "사투리 단어 없는 문장 5.5만 (화자 1.8만)",
    aug: "없음", mix: "단독 · 서비스 때 서울·경기 보정 +1.0", verdict: "첫 서비스, 데이터를 늘린 1단계로 교체" },
  { id: "phase2", name: "2단계", when: "10/7 · 앙상블 안에서만 사용", task: "6개 권역",
    encoder: "17~32층 LoRA 미세조정 + 어텐션 풀링", from: "Whisper-large-v3", data: "1단계와 같은 5.5만, 3 epoch",
    aug: "속도·잡음·대역 제한·울림", mix: "단독 (보정 없음)", verdict: "단독으론 1단계보다 낮지만 표준·지역 구분은 더 나음" },
  { id: "ensemble", name: "1+2단계 앙상블", when: "10/7 04:00~05:57 서비스", task: "6개 권역",
    encoder: "두 모델의 확률을 섞음", from: "1단계 + 2단계", data: "—", aug: "—", mix: "1단계 0.6 + 2단계 0.4 · 보정 +1.0",
    verdict: "섞은 쪽이 둘 다보다 나음" },
  { id: "ensemble_big", name: "1.7배 앙상블", when: "10/7 05:57~10/9 19:34 서비스", task: "6개 권역",
    encoder: "두 모델의 확률을 섞음", from: "1단계를 1.7배 데이터로 다시 학습 + 2단계", data: "1단계용 9.5만 문장 (화자 3만) · 3.6배로 늘려도 거의 안 오름",
    aug: "2단계만", mix: "1단계(1.7배) 0.6 + 2단계 0.4 · 보정 +1.25", verdict: "6개 권역 최고 성능" },
  { id: "phase3", name: "3단계", when: "10/7 · 실험", task: "6개 권역",
    encoder: "33개 층 전체 + 17~32층 LoRA", from: "1단계 판정 층에서 출발 (LP-FT)", data: "19.9만 문장 (화자 3.9만, 3.6배), 1 epoch",
    aug: "없음", mix: "단독", verdict: "검증은 0.84까지 올랐지만 시험은 앙상블보다 낮아 쓰지 않음" },
  { id: "region4", name: "4지역 모델", when: "10/9 19:33부터 서비스", task: "4개 지역 (강원·충청·경상·제주) · 서울·경기는 학습 안 함",
    encoder: "고정 (1단계와 같은 방식)", from: "Whisper-large-v3", data: "정답 = 녹음된 AI Hub 세트 · 원본 + 녹음 환경을 바꾼 복사본 2개",
    aug: "다른 세트 배경, 잡음 제거, 단어 사이 무음화, Opus 압축, 마이크 특성, 음량, 1.5~4초로 자르기", mix: "단독",
    verdict: "녹음 환경 변형 덕에 무음화·잡음 제거에도 견디고, 배경만으로 맞히는 비율이 줄어듦" },
  { id: "detector_mix", name: "표준 억양 판별기", when: "10/9 19:33부터 서비스", task: "지역 억양이 있는지 (서울·경기일 확률만 사용)",
    encoder: "두 6권역 모델의 확률을 섞음", from: "1.7배 앙상블 + 6권역 녹음 환경 변형 모델", data: "6권역 변형 모델: 1.7배 데이터 + 변형 복사본 2개 (28만)",
    aug: "구성원 하나가 녹음 환경 변형으로 학습", mix: "0.5 + 0.5 · 서울·경기 보정 +2.7 (검증 세트에서 지역 억양을 90% 이상 잡는 선)",
    verdict: "표준/지역 두 칸 판별기가 실패해 대신 씀" },
];

function accVersions() {
  const cur = acc.current;
  const now = new Set(cur ? [cur.region_run, cur.detector_run] : [acc.served]);
  for (const n of document.querySelectorAll("#acc-lineage .lin-node[data-run]")) n.classList.toggle("now", now.has(n.dataset.run));
  const std2 = (acc.detectors || []).find((d) => d.id === "std2");
  if (std2 && std2.conditions && std2.conditions["원본"]) $("#acc-std2-auc").textContent = std2.conditions["원본"].auc.toFixed(3);
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const numbers = (v) => {  // each kind of model is scored by what it does
    const it = acc.items.find((i) => i.id === v.id);
    if (it && it.tests && it.tests.test_plain) {
      const t = it.tests.test_plain, [fa, n] = t.standard_called_regional || [];
      return `6권역 Macro-F1 <b>${pct(t.macro_f1.value)}</b>${fa !== undefined ? `<br>서울·경기 오판 ${Math.round((100 * fa) / n)}% (${it.bias ? `보정 +${it.bias}` : "보정 없이"})` : ""}`;
    }
    const r4 = (acc.runs4 || []).find((r) => r.id === v.id);
    if (r4) {
      const c = r4.conditions || {};
      return `4지역 Macro-F1 <b>${pct(r4.tests.test_plain)}</b><br>무음화 ${pct(c["무음화"])} · 배경만 ${pct(c["배경만"])} (무작위 25%, 낮을수록 좋음)`;
    }
    const d = (acc.detectors || []).find((x) => x.id === v.id);
    if (d && d.conditions && d.conditions["원본"]) {
      const c = d.conditions["원본"];
      return `AUC <b>${c.auc.toFixed(3)}</b><br>지역 억양 잡아냄 ${pct(c.regional_caught, 0)} · 서울·경기 오판 ${pct(c.standard_called_regional, 0)}`;
    }
    return "—";
  };
  const rows = [
    ["시기 · 서비스", (v) => esc(now.has(v.id) ? `${v.when.replace("부터 서비스", "부터")} · 지금 서비스 중` : v.when)],
    ["맡은 일", (v) => esc(v.task)],
    ["인코더", (v) => esc(v.encoder)],
    ["출발점 · 구성", (v) => esc(v.from)],
    ["학습 데이터", (v) => esc(v.data)],
    ["녹음 환경 변형", (v) => esc(v.aug)],
    ["섞는 비율 · 보정", (v) => esc(v.mix)],
    ["주요 수치", numbers],
    ["결과", (v) => esc(v.verdict)],
  ];
  const cls = (v) => (now.has(v.id) ? ' class="now"' : "");
  $("#acc-versions-table").innerHTML = `<colgroup><col class="lab">${ACC_VERSIONS.map(() => "<col>").join("")}</colgroup>
    <thead><tr><th></th>${ACC_VERSIONS.map((v) => `<th${cls(v)}>${esc(v.name)}${now.has(v.id) ? '<span class="badge">서비스 중</span>' : ""}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([name, get]) => `<tr><th scope="row">${name}</th>${ACC_VERSIONS.map((v) => `<td${cls(v)}>${get(v)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

// ---- answer level: the served system on 1-8 sentences of one speaker (accent-id session_eval.py) ----
const SESSION_CONDS = [["원본", "--s1", "원본 녹음"], ["사이트 전처리", "--s2", "사이트 전처리 후"]];

function sessionChart(host, s, ks) {
  host.replaceChildren();
  const W = Math.max(320, host.clientWidth), H = 270, L = 44, R = 64, T = 18, B = 50;
  const yMin = 0.7, yMax = 1.0, target = 0.9;
  const X = (k) => L + ((W - L - R) * (k - ks[0])) / (ks[ks.length - 1] - ks[0]);
  const Y = (v) => T + (H - T - B) * (1 - (v - yMin) / (yMax - yMin));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "문장 수에 따른 4지역 Macro-F1, 원본 녹음과 사이트 전처리 후" });
  const grid = css("--grid"), axis = css("--axis"), muted = css("--muted");
  for (let v = yMin; v <= yMax + 1e-9; v += 0.05) {
    const y = Y(v);
    svg.append(el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: v - yMin < 1e-9 ? axis : grid, "stroke-width": 1 }));
    svg.append(el("text", { x: L - 8, y: y + 4, "text-anchor": "end", "font-size": 11, fill: muted }, v.toFixed(2)));
  }
  const secs = (k) => s.conditions["원본"][k].median_seconds;
  for (const k of ks) {
    svg.append(el("text", { x: X(k), y: H - B + 17, "text-anchor": "middle", "font-size": 11, fill: css("--ink-2"), "font-weight": 600 }, `${k}문장`));
    svg.append(el("text", { x: X(k), y: H - B + 31, "text-anchor": "middle", "font-size": 10.5, fill: muted }, `${Math.round(secs(k))}초`));
  }
  svg.append(el("line", { x1: L, x2: W - R, y1: Y(target), y2: Y(target), stroke: muted, "stroke-width": 1, "stroke-dasharray": "4 4" }));
  svg.append(el("text", { x: L + 6, y: Y(target) - 6, "font-size": 11, fill: muted }, `목표 ${target}`));
  const ends = [];
  for (const [cond, color] of SESSION_CONDS) {
    const c = css(color), pts = ks.map((k) => ({ k, m: s.conditions[cond][k] }));
    svg.append(el("path", { d: "M" + pts.map((p) => `${X(p.k)},${Y(p.m.region4_macro_f1)}`).join("L"), fill: "none", stroke: c, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    for (const p of pts) {
      const [lo, hi] = p.m.region4_macro_f1_ci, x = X(p.k);
      svg.append(el("line", { x1: x, x2: x, y1: Y(lo), y2: Y(hi), stroke: c, "stroke-width": 1.5, opacity: 0.45 }));
      svg.append(el("circle", { cx: x, cy: Y(p.m.region4_macro_f1), r: 4.5, fill: c, stroke: "#fff", "stroke-width": 2 }));
    }
    const last = pts[pts.length - 1];
    ends.push({ y: Y(last.m.region4_macro_f1), text: `${cond === "원본" ? "원본" : "전처리"} ${last.m.region4_macro_f1.toFixed(2)}` });
  }
  // direct labels at the right end, moved apart when they would touch
  ends.sort((a, b) => a.y - b.y);
  if (ends.length === 2 && ends[1].y - ends[0].y < 14) { const mid = (ends[0].y + ends[1].y) / 2; ends[0].y = mid - 7; ends[1].y = mid + 7; }
  for (const e of ends) svg.append(el("text", { x: X(ks[ks.length - 1]) + 9, y: e.y + 4, "font-size": 11, fill: css("--ink-2"), "font-weight": 600 }, e.text));
  const cross = el("line", { y1: T, y2: H - B, stroke: axis, "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  const hit = el("rect", { x: L - 10, y: T, width: W - L - R + 20, height: H - T - B, fill: "transparent", tabindex: 0 });
  let focusIdx = ks.length - 1;
  const show = (i, cx, cy) => {
    const k = ks[i], x = X(k);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
    showTip(cx, cy, `${k}문장 · 말한 길이 중앙값 ${secs(k)}초`, SESSION_CONDS.map(([cond, color, name]) => {
      const m = s.conditions[cond][k];
      return { color: css(color), name, value: `${m.region4_macro_f1.toFixed(3)} (${m.region4_macro_f1_ci.map((v) => v.toFixed(3)).join("–")})` };
    }));
  };
  hit.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect(), x = ((e.clientX - r.left) * W) / r.width;
    let best = 0;
    ks.forEach((k, i) => { if (Math.abs(X(k) - x) < Math.abs(X(ks[best]) - x)) best = i; });
    show(best, e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  hit.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    focusIdx = Math.max(0, Math.min(ks.length - 1, focusIdx + (e.key === "ArrowRight" ? 1 : -1)));
    const r = svg.getBoundingClientRect();
    show(focusIdx, r.left + (X(ks[focusIdx]) * r.width) / W, r.top + r.height / 3);
  });
  hit.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  svg.append(hit);
  host.append(svg);
}

function accSessions() {
  const s = acc.sessions, card = $("#acc-sessions-card");
  card.hidden = !s;
  if (!s) return;
  const ks = Object.keys(s.conditions["원본"]).map(Number).sort((a, b) => a - b);
  const regionalN = Object.entries(s.speakers).filter(([z]) => z !== "서울·경기").reduce((a, [, n]) => a + n, 0);
  const at = (cond, k) => s.conditions[cond][k];
  const first = at("사이트 전처리", ks[0]).region4_macro_f1, k6 = ks.find((k) => at("사이트 전처리", k).region4_macro_f1 >= 0.9);
  $("#acc-sessions-sub").textContent = `앱은 문장 하나가 아니라 답변 전체를 모아 판정해요. 학습에 쓰지 않은 AI Hub 검증 분할의 지역 화자 ${regionalN.toLocaleString()}명이 `
    + `사투리 단어 없는 문장 ${ks[ks.length - 1]}개씩 말한 것을 앱처럼 길이로 평균해, 서비스 중인 4지역 모델이 맞히는지 쟀어요 `
    + `(한 문장 ${first.toFixed(2)}${k6 ? ` → ${k6}문장부터 0.9 이상` : ""}). 세로선은 화자를 다시 뽑은 95% 신뢰구간이에요.`;
  sessionChart($("#acc-sessions"), s, ks);
  const rules = s.rules["사이트 전처리"] || {};
  const p0 = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  const head = ["문장 수", "말한 길이 (녹음, 중앙값)", "4지역 Macro-F1 (원본)", "4지역 Macro-F1 (사이트)", "표준·지역 AUC", "서울·경기를 지역으로", "앱 판정 보류 (이전 → 지금)", "보류 아닐 때 정확도", "앱이 서울·경기를 지역으로"];
  const rows = ks.map((k) => {
    const a = at("원본", k), b = at("사이트 전처리", k), r = rules[k] || {}, now = r["2step-renorm"], before = r.now;
    return `<tr><td>${k}문장</td><td>${a.median_seconds}초</td><td>${a.region4_macro_f1.toFixed(3)}</td><td class="now">${b.region4_macro_f1.toFixed(3)}</td>`
      + `<td>${b.detector_auc.toFixed(3)}</td><td>${p0(b.seoul_called_regional)}</td>`
      + `<td>${before && now ? `${p0(before.unsure)} → ${p0(now.unsure)}` : "—"}</td><td>${now ? p0(now.accuracy_when_sure) : "—"}</td><td>${now ? p0(now.seoul_called_regional) : "—"}</td></tr>`;
  }).join("");
  $("#acc-sessions-table").innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
  $("#acc-sessions-note").textContent = `"서울·경기를 지역으로"는 서울·경기 화자 ${s.speakers["서울·경기"] || 0}명 중 판별기가 지역 억양(확률 0.5 이상)으로 본 비율이에요(사람 수가 적어 대략적인 값). `
    + `앱 판정은 서울·경기와 4지역 중 하나를 고르거나 보류해요. 10/12부터 "지역 억양인가 → 어느 지역인가" 두 단계로 판정해서, 지역을 맞게 골라도 두 확률을 곱하면 기준(0.6)에 못 미쳐 `
    + `보류되던 답변이 줄었어요. 대신 서울·경기 화자를 지역 억양으로 판정하는 경우가 늘었고, 기준값은 실제 브라우저 녹음으로 다시 정할 예정이에요.`;
}

function renderAccent() {
  if (!acc) return;
  accVersions();
  accSessions();
  if (!acc.items.some((i) => i.id === accPick)) accPick = accHighlight();
  const test = accTestOf(accTest) || acc.tests[0], s = accServed(), item = acc.items.find((i) => i.id === accPick);
  accTest = test.id;
  accMeta();
  accNow();
  accKpis();
  $("#acc-note").textContent = `${acc.current ? "10/9까지 서비스" : "서비스 중"}: ${s.name} (${s.desc}) · 위 수치는 사투리 단어 없는 문장 ${accTestOf("test_plain").n.toLocaleString()}개(6개 권역 × 200) 기준 · 비교 기준은 첫 1단계`;
  const seg = $("#acc-test");
  seg.replaceChildren(...acc.tests.map((x, i) => {
    const btn = Object.assign(document.createElement("button"), { type: "button", textContent: x.name, className: x.id === accTest ? "on" : "" });
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(x.id === accTest));
    btn.addEventListener("click", () => { accTest = x.id; renderAccent(); $("#acc-test").children[i].focus(); });
    return btn;
  }));
  $("#acc-bars-sub").textContent = `${test.desc} · ${test.n.toLocaleString()}문장 · 가로선은 95% 신뢰구간 · 점선은 무작위 수준(17%) · 막대를 누르면 혼동 행렬이 바뀌어요`;
  accBars($("#acc-bars"));
  accCurve($("#acc-curve"));
  confusion($("#acc-cm"), item.tests[accTest], ACC_ZONES, false);
  $("#acc-cm-sub").textContent = `${item.name}${item.bias ? ` (표준어 보정 +${item.bias})` : ""} · ${test.name} ${test.n.toLocaleString()}문장 · 행은 화자의 실제 권역, 열은 분류기가 고른 권역 · 진할수록 많이 고름 · 대각선이 정답`;
  const g = gemmaOf();
  $("#acc-gemma-sub").textContent = g
    ? `같은 90문장(6개 권역 × 15, 전라 제외) · 권역별 F1 · Gemma 4 ${gemmaLabel(g)} Macro-F1 ${pct(g.macro_f1.value)} vs 억양 분류기 ${pct(s.tests.bench.macro_f1.value)}`
    : "";
  accGemma($("#acc-gemma"));
  accTable();
}

async function loadAccent() {
  try {
    const a = await fetch("/api/accent").then((r) => r.json());
    if (a && a.items && a.items.length) {
      acc = a;
      renderAccent();
    }
  } catch (e) {
    if (!acc) $("#acc-note").textContent = "억양 분류기 결과를 불러오지 못했어요.";
  }
}

// ---- model versions: how each one was trained (gemma4-dialect run scripts and train_args.json) ----------
const VERSIONS = [
  { run: "base", name: "Gemma 4", when: "—", from: "—", data: "—", format: "지시문만 (학습 없음)", adapter: "없음", train: "—",
    verdict: "출발점", served: "" },
  { run: "lora_v1", name: "v1", when: "1·2단계 · 10/4~10/5", from: "Gemma 4",
    data: "강원 2.5만 발화로 먼저 학습한 뒤, 경상·제주·충청 7.5만 + 강원 0.75만(앞서 배운 것을 잊지 않게 다시)", format: "방언 / 표준어",
    adapter: "LoRA rank 16", train: "lr 1e-4 · 1,563 + 5,155스텝", verdict: "기준 모델", served: "10/6~10/11 서비스" },
  { run: "lora_v2r", name: "v2r", when: "3단계 · 10/6~10/8", from: "v1에서 이어서",
    data: "새 발화 46만 + 서울·경기·전라 화자 4.4만 중 약 10만까지", format: "지역 / 방언 / 표준어 (억양 지역 줄 추가)",
    adapter: "v1과 같은 rank 16", train: "lr 1e-4 · 31,456스텝 중 6,500에서 중단",
    verdict: "4,000문장 시험에서 v1과 같음 · 억양 지역 줄도 거의 못 배움 (억양은 Whisper 분류기로 넘김)",
    served: "1,000스텝이 10/6~10/7 잠시 서비스 (옛 교체 기준), 되돌림" },
  { run: "lora_v3", name: "v3", when: "4단계 · 10/8~10/9", from: "v1에서 이어서",
    data: "새 발화 6.4만(제주 40%·강원 25%·충청 25%·경상 10%) 중 1.6만까지", format: "방언 / 표준어",
    adapter: "v1과 같은 rank 16", train: "lr 2e-5 · 4,000스텝 중 1,000에서 중단",
    verdict: "4,000문장 시험에서 v1과 같음 → 같은 크기 어댑터는 한계", served: "서비스 안 함" },
  { run: "lora_r64c", name: "r64c", when: "5단계 · 10/10~10/11", from: "v1을 가중치에 합친 모델",
    data: "새 발화 4.8만 (권역당 1.2만)", format: "방언 / 표준어", adapter: "새 rank 64 → v1과 합쳐 rank 80",
    train: "lr 2e-5 · 3,000스텝", verdict: "4,000문장 시험에서 v1보다 나음 (표준어 CER 0.153→0.151)", served: "10/11 00:35~21:25 서비스" },
  { run: "lora_s6c", name: "s6c", when: "6단계 · 10/11", from: "r64c를 가중치에 합친 모델",
    data: "새 발화 6.4만 (권역당 1.6만)", format: "방언 / 표준어", adapter: "새 rank 64 → r64c와 합쳐 rank 144",
    train: "lr 2e-5 · 4,000스텝", verdict: "4,000문장 시험에서 r64c보다 나음 (0.151→0.147, 방언 0.145→0.140)", served: "10/11 21:25부터 서비스" },
];

function versions(items, servedId) {
  const servedRun = (items.find((b) => b.id === servedId) || {}).run;
  for (const n of document.querySelectorAll("#lineage .lin-node")) n.classList.toggle("now", n.dataset.run === servedRun);
  const cer = (run, key) => {
    const bs = items.filter((b) => b.run === run);
    if (!bs.length) return "—";
    const one = bs.find((b) => b.id === servedId) || (bs.length === 1 ? bs[0] : null);
    if (one) return pct(one.metrics.asr[key].value);
    const v = bs.map((b) => b.metrics.asr[key].value);
    return `${pct(Math.min(...v))}~${pct(Math.max(...v))}`;
  };
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  // versions side by side (columns), what differs between them as rows
  const rows = [
    ["단계·기간", (v) => esc(v.when)],
    ["출발점", (v) => esc(v.from)],
    ["새로 학습한 데이터", (v) => esc(v.data)],
    ["답 형식", (v) => esc(v.format)],
    ["어댑터", (v) => esc(v.adapter)],
    ["학습률 · 스텝", (v) => esc(v.train)],
    ["표준어 CER (160문장)", (v) => `<b>${cer(v.run, "cer_standard")}</b>`],
    ["방언 CER (160문장)", (v) => `<b>${cer(v.run, "cer_dialect")}</b>`],
    ["결과", (v) => esc(v.verdict)],
    ["서비스", (v) => esc(v.run === servedRun ? "지금 서비스 중" : v.served)],
  ];
  const cls = (v) => (v.run === servedRun ? ' class="now"' : "");
  $("#versions-table").innerHTML = `<colgroup><col class="lab">${VERSIONS.map(() => "<col>").join("")}</colgroup>
    <thead><tr><th></th>${VERSIONS.map((v) => `<th${cls(v)}>${esc(v.name)}${v.run === servedRun ? '<span class="badge">서비스 중</span>' : ""}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([name, get]) => `<tr><th scope="row">${name}</th>${VERSIONS.map((v) => `<td${cls(v)}>${get(v)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function render() {
  if (!data || !data.items.length) return;
  const items = data.items;
  const served = items.find((b) => b.id === data.served) || items[items.length - 1];
  const base = items.find((b) => b.run === "base");
  meta(data, status);
  $("#kpi-title").textContent = `지금 서비스 중인 모델 · ${label(served)}`;
  $("#kpi-note").textContent = `시험 문장 ${served.metrics.asr.n}개 기준 · 학습 전 Gemma 4와 비교 · ${(status && status.update_hours) || data.update_hours || 2}시간마다 새 체크포인트를 채점해, 두 줄 모두 나빠지지 않고 같은 문장끼리 비교해 더 나을 확률이 ${Math.round(((status && status.gate_confidence) || 0.95) * 100)}% 이상일 때만 교체합니다.`;
  kpis(served, base);
  versions(items, data.served);
  lineChart($("#curve"), {
    items, servedId: data.served,
    yMax: niceMax(Math.max(...items.flatMap((b) => [b.metrics.asr.cer_dialect.ci[1], b.metrics.asr.cer_standard.ci[1]])) * 1.05, 5),
    series: [
      { name: "방언 받아쓰기", color: css("--s1"), get: (b) => b.metrics.asr.cer_dialect },
      { name: "표준어 변환", color: css("--s2"), get: (b) => b.metrics.asr.cer_standard },
    ],
  });
  const zoned = items.filter((b) => b.metrics.zone_plain);
  if (zoned.length) {
    lineChart($("#zone"), {
      items: zoned, servedId: data.served, yMax: 1, refLine: { value: 1 / 7, label: "무작위 수준 14%" },
      series: [{ name: "Macro-F1", color: css("--s3"), get: (b) => b.metrics.zone_plain.macro_f1 }],
    });
    const latest = zoned[zoned.length - 1];
    confusion($("#cm"), latest.metrics.zone_plain);
    $("#cm-sub").textContent = `${label(latest)} · 행은 화자의 실제 권역, 열은 모델이 고른 권역 · 진할수록 많이 고름 · 대각선이 정답`;
  } else {
    $("#zone").innerHTML = '<div class="empty-chart">억양 지역 판별은 "지역:" 줄을 학습하는 v2r 체크포인트부터 측정해요.<br>첫 결과는 다음 업데이트 주기에 나와요.</div>';
    $("#cm").innerHTML = '<div class="empty-chart">v2r 체크포인트를 측정하면 7개 권역의 혼동 행렬이 여기에 나와요.</div>';
    $("#cm-sub").textContent = "";
  }
  regionBars($("#regions"), served);
  $("#region-sub").textContent = `${label(served)} · 권역별 시험 문장 40개 · 낮을수록 좋아요`;
  table(items, data.served, status && status.history);
  renderAccent();  // the Gemma comparison follows the served checkpoint
}

async function load() {
  if (!acc || ++accTick % 10 === 0) loadAccent();  // changes only after a new run
  try {
    const [b, s] = await Promise.all([fetch("/api/bench").then((r) => r.json()), fetch("/api/status").then((r) => r.json())]);
    data = b;
    status = s;
    render();
  } catch (e) {
    $("#meta").innerHTML = "<span>결과를 불러오지 못했어요</span>";
  }
}

let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { render(); renderAccent(); }, 150); });
load();
setInterval(load, 60000);
