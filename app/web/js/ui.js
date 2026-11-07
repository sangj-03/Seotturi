// DOM helpers, icons, overlays and formatting shared by the screens.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const ICONS = {
  lock: '<rect x="5" y="10.5" width="14" height="10" rx="2.6" fill="currentColor"/><path d="M8.2 10.5V8a3.8 3.8 0 0 1 7.6 0v2.5" fill="none" stroke="currentColor" stroke-width="2"/>',
  back: '<path d="M15 5l-7 7 7 7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  close: '<path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
  chevron: '<path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>',
  bell: '<path d="M12 3a6 6 0 0 0-6 6v3.6l-1.6 2.7c-.4.7.1 1.7 1 1.7h13.2c.9 0 1.4-1 1-1.7L18 12.6V9a6 6 0 0 0-6-6Zm-2.6 15.5a2.7 2.7 0 0 0 5.2 0H9.4Z" fill="currentColor"/>',
  pencil: '<path d="M15.2 4.6l4.2 4.2L9 19.2 4 20l.8-5 10.4-10.4Z" fill="currentColor"/>',
  mic: '<rect x="8.5" y="2.5" width="7" height="12" rx="3.5" fill="currentColor"/><path d="M5 11.5a7 7 0 0 0 14 0M12 18.5V21.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="3" fill="currentColor"/>',
  check: '<path d="M5 12.5l4.3 4.3L19 7" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
  star: '<path d="M12 2.8l2.8 5.8 6.3.9-4.6 4.4 1.1 6.3L12 17.2l-5.6 3 1.1-6.3L2.9 9.5l6.3-.9L12 2.8Z" fill="currentColor"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="2.4"/><path d="M15.5 15.5L20 20" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/>',
  pin: '<path d="M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7Zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5Z" fill="currentColor"/>',
  doc: '<path d="M6 2.5h8l5 5V20a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 20V4a1.5 1.5 0 0 1 1-1.5Z" fill="currentColor"/><path d="M8.5 12h7M8.5 15.5h7" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/>',
  person: '<circle cx="9" cy="7.5" r="3.5" fill="currentColor"/><path d="M2.5 19c0-3.6 2.9-6.5 6.5-6.5s6.5 2.9 6.5 6.5c0 .6-.4 1-1 1h-11c-.6 0-1-.4-1-1Z" fill="currentColor"/><path d="M16.5 6.5c1.8.4 3 2 3 3.8M18.5 3.5c3 1 4.5 4 3.8 7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
  wave: '<path d="M3 10v4M6.5 7v10M10 9.5v5M13.5 6v12M17 10v4M20.5 8v8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  plus: '<path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>',
  folderPlus: '<path d="M7 4h4l2 2h5a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z" fill="currentColor" opacity=".55"/><rect x="2.5" y="7" width="15" height="13" rx="2.5" fill="currentColor"/><path d="M10 10.5v6M7 13.5h6" stroke="#0a6bff" stroke-width="2" stroke-linecap="round"/>',
  like: '<path d="M7 10.5v9.5H4.5a1.5 1.5 0 0 1-1.5-1.5V12a1.5 1.5 0 0 1 1.5-1.5H7Zm2 0l3.6-7c1.6 0 2.7 1.3 2.4 2.9L14.5 9H19a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.8 20H9v-9.5Z" fill="currentColor"/>',
  comment: '<path d="M4 4.5h16a1.5 1.5 0 0 1 1.5 1.5v10a1.5 1.5 0 0 1-1.5 1.5H9.5L5 21v-3.5H4A1.5 1.5 0 0 1 2.5 16V6A1.5 1.5 0 0 1 4 4.5Z" fill="currentColor"/>',
  bookmark: '<path d="M6.5 3h11A1.5 1.5 0 0 1 19 4.5V21l-7-4.5L5 21V4.5A1.5 1.5 0 0 1 6.5 3Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
  bookmarkOn: '<path d="M6.5 3h11A1.5 1.5 0 0 1 19 4.5V21l-7-4.5L5 21V4.5A1.5 1.5 0 0 1 6.5 3Z" fill="currentColor"/>',
  share: '<path d="M12 3v12M7.5 7.5L12 3l4.5 4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 12v6.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  download: '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>',
  sparkle: '<path d="M12 2.5l1.9 5.6 5.6 1.9-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.9L12 2.5Z" fill="currentColor"/><path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2-2.2-.8 2.2-.8L19 15Z" fill="currentColor"/>',
  upload: '<path d="M12 16V4M7 8.5L12 4l5 4.5M5 20h14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  info: '<circle cx="12" cy="12" r="9.5" fill="currentColor"/><path d="M12 11v6M12 7.5v.5" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  speaker: '<path d="M3.5 9.5H7l4.5-4v13L7 14.5H3.5a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1Z" fill="currentColor"/><path d="M15 9a4 4 0 0 1 0 6M17.6 6.4a7.6 7.6 0 0 1 0 11.2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  speakerOff: '<path d="M3.5 9.5H7l4.5-4v13L7 14.5H3.5a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1Z" fill="currentColor"/><path d="M15.5 9.5l5 5M20.5 9.5l-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  trash: '<path d="M5 7h14M10 7V5h4v2M7 7l1 13h8l1-13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
};

export function icon(name, cls = "") {
  return `<svg class="ic ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;
}

export function html(strings, ...values) {
  return strings.reduce((out, s, i) => out + s + (i < values.length ? values[i] ?? "" : ""), "");
}

// ---- toast / overlays ----
let toastTimer;
export function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), ms);
}

function overlay(kind, content, { onClose, dismissible = true } = {}) {
  const root = $("#overlay");
  const scrim = document.createElement("div");
  scrim.className = "scrim";
  const box = document.createElement("div");
  box.className = kind;
  box.innerHTML = (kind === "sheet" ? '<div class="grab"></div>' : "") + content;
  root.append(scrim, box);
  requestAnimationFrame(() => { scrim.classList.add("show"); box.classList.add("show"); });
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    scrim.classList.remove("show");
    box.classList.remove("show");
    setTimeout(() => { scrim.remove(); box.remove(); }, 300);
    onClose && onClose();
  };
  if (dismissible) scrim.addEventListener("click", close);
  return { el: box, close };
}
export const sheet = (content, opts) => overlay("sheet", content, opts);
export const modal = (content, opts) => overlay("modal", content, opts);
export function closeOverlays() {
  for (const el of $$("#overlay > *")) el.remove();
}

// ---- formatting ----
export function ymd(ts = Date.now() / 1000) {
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, "0")}.${String(d.getDate()).padStart(2, "0")}`;
}
export function hm(ts) {
  const d = new Date(ts * 1000);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
export function ago(ts) {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "방금";
  if (s < 3600) return `${Math.floor(s / 60)}분 전`;
  if (s < 86400) return `${Math.floor(s / 3600)}시간 전`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}일 전`;
  return ymd(ts);
}
export function untilText(ts) {
  const s = Math.max(0, ts - Date.now() / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}시간 ${m}분 후` : `${m}분 후`;
}
export function num(n) {
  return Number(n || 0).toLocaleString("ko-KR");
}

// Dialect transcript with the words the model changed (and optional quoted phrases) highlighted.
export function marked(analysis, quotes = []) {
  if (!analysis || !analysis.tokens) return "";
  let out = analysis.tokens.map((t) => (t.fix === null ? esc(t.t) : `<span class="hl-d">${esc(t.t)}</span>`)).join(" ");
  for (const q of quotes) {
    if (!q || q.length < 2) continue;
    const e = esc(q);
    const i = out.indexOf(e);
    if (i >= 0 && !out.slice(Math.max(0, i - 30), i).includes("<span class=\"hl-d\">" + e)) {
      out = out.slice(0, i) + `<span class="hl-c">${e}</span>` + out.slice(i + e.length);
    }
  }
  return out;
}

// Standard-Korean line: replaced words in blue, the coach's quoted phrases in orange.
export function standardMarked(analysis, standard, quotes = []) {
  let out = esc(standard);
  for (const q of quotes) {
    const e = esc(q || "");
    if (e.length >= 2 && out.includes(e)) out = out.replace(e, `\u0001${e}\u0002`);
  }
  for (const f of (analysis && analysis.fixes) || []) {
    if (!f.to) continue;
    const e = esc(f.to);
    out = out.replace(new RegExp(`(^|[\\s\u0001])(${e.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`), "$1<b>$2</b>");
  }
  return out.replace(/\u0001/g, '<span class="hl-c">').replace(/\u0002/g, "</span>");
}

export const MODE_LABEL = { interview: "모의 면접", presentation: "모의 발표", daily: "일상회화" };
export const ZONE_REGIONS = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "전라", "제주"];
