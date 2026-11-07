// Landing page: the poster's pie chart, the sticky nav and the live model panel.
import { chatter, ready } from "./alive.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const hm = (ts) => { const d = new Date(ts * 1000); return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; };
const md = (ts) => { const d = new Date(ts * 1000); return `${d.getMonth() + 1}/${d.getDate()}`; };

function pie() {
  const slices = [[48, "url(#p48)", 10], [31, "url(#p31)", 0], [21, "#c9ced6", 0]];
  const c = 115, r = 100;
  let a = -90, paths = "";
  for (const [pct, fill, out] of slices) {
    const sweep = (pct / 100) * 360;
    const mid = ((a + sweep / 2) * Math.PI) / 180;
    const dx = Math.cos(mid) * out * 0.4, dy = Math.sin(mid) * out * 0.4;
    const rr = r + out;
    const p = (deg) => [c + dx + rr * Math.cos((deg * Math.PI) / 180), c + dy + rr * Math.sin((deg * Math.PI) / 180)];
    const [x1, y1] = p(a), [x2, y2] = p(a + sweep);
    paths += `<path d="M${c + dx},${c + dy}L${x1.toFixed(1)},${y1.toFixed(1)}A${rr},${rr} 0 ${sweep > 180 ? 1 : 0} 1 ${x2.toFixed(1)},${y2.toFixed(1)}Z" fill="${fill}"/>`;
    a += sweep;
  }
  $("#pie").insertAdjacentHTML("afterbegin", `<svg viewBox="0 0 230 230" aria-label="취업 준비생의 사투리 인식: 48%, 31%, 21%">
    <defs><linearGradient id="p48" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="#0b1736"/><stop offset="1" stop-color="#2c4c8c"/></linearGradient>
    <linearGradient id="p31" x1="0" x2="1" y1="1" y2="0"><stop offset="0" stop-color="#8fbaff"/><stop offset="1" stop-color="#dce9ff"/></linearGradient></defs>${paths}</svg>`);
}

function fitPhone() {
  const box = document.querySelector(".phone-box");
  if (!box) return;
  const s = Math.max(0.62, Math.min(0.9, (window.innerHeight - 150) / 872));
  box.style.setProperty("--s", s.toFixed(3));
  box.querySelector(".phone").style.setProperty("--s", s.toFixed(3));
  document.querySelector(".hero-stage").style.height = `${Math.max(560, 872 * s + 40)}px`;
}

async function refresh() {
  let s;
  try {
    s = await (await fetch("/api/status")).json();
  } catch {
    s = { ready: false, error: "offline" };
  }
  const chip = $("#chip");
  chip.classList.toggle("ok", !!s.ready);
  chip.querySelector("span").textContent = s.ready
    ? `Gemma 4 · 서투리 LoRA ${s.model.id} 서비스 중`
    : s.demo ? "체험 모드 · Gemma 4 엔진이 꺼져 있어 AI 답은 예시로 보여 줘요"
    : s.error === "offline" ? "AI 엔진에 연결하지 못했어요" : "Gemma 4 모델을 불러오는 중이에요";
  $("#hours").textContent = s.update_hours || 5;
  if (s.model) {
    const m = s.model;
    $("#m-id").innerHTML = `${esc(m.run)} <small>${m.final ? "final" : "step " + m.step.toLocaleString()}</small>`;
    $("#m-desc").textContent = `Gemma 4 12B + LoRA (r=16) · 답 형식: ${m.format === "region" ? "지역 / 방언 / 표준어" : "방언 / 표준어"} · ${m.loaded_at ? md(m.loaded_at) + " " + hm(m.loaded_at) + " 적용" : ""}`;
    const sc = m.score;
    $("#m-cer").innerHTML = sc ? `${(sc.cer_standard * 100).toFixed(1)}<small>%</small>` : "측정 중";
    $("#m-cer-d").textContent = sc ? `방언 받아쓰기 ${(sc.cer_dialect * 100).toFixed(1)}% · chrF++ ${sc.chrfpp.toFixed(1)}${sc.zone_macro_f1 !== null && sc.zone_macro_f1 !== undefined ? ` · 억양 지역 F1 ${(sc.zone_macro_f1 * 100).toFixed(0)}%` : ""} · 시험 문장 ${sc.n}개` : "시험 문장 160개로 측정하고 있어요";
  }
  const t = s.training;
  if (t) {
    const pct = (100 * t.step) / t.total;
    $("#t-step").innerHTML = `${pct.toFixed(1)}<small>%</small>`;
    $("#t-bar").style.width = `${pct}%`;
    $("#t-desc").textContent = `${t.run} · ${t.step.toLocaleString()} / ${t.total.toLocaleString()} 스텝 · 약 ${Math.round(t.eta_hours)}시간 남음 · ${t.save_every.toLocaleString()} 스텝마다 체크포인트${t.checkpoints.length ? ` (최신 ${t.checkpoints[t.checkpoints.length - 1].toLocaleString()})` : ""}`;
  }
  if (s.next_check) {
    const left = Math.max(0, s.next_check - Date.now() / 1000);
    $("#u-next").innerHTML = `${hm(s.next_check)} <small>${Math.floor(left / 3600)}시간 ${Math.floor((left % 3600) / 60)}분 후</small>`;
    const lc = s.last_check;
    $("#u-last").textContent = `${s.update_hours}시간 주기 · 새 체크포인트를 검증 문장으로 채점해 더 정확할 때만 교체해요.${lc ? ` 마지막 확인 ${hm(lc.time)}: ${({ "no newer checkpoint": "새 체크포인트 없음" })[lc.result] || lc.result}` : ""}`;
  }
  if (s.history && s.history.length) {
    // A manual revert (result "reverted") has no benchmark scores.
    const cer = (b) => (b && b.cer_standard != null ? `${(b.cer_standard * 100).toFixed(1)}%` : "—");
    const verdict = { promoted: "교체", reverted: "되돌림" };
    $("#hist").innerHTML = s.history.slice().reverse().map((h) => `<tr><td>${md(h.time)} ${hm(h.time)}</td><td>${esc(h.candidate)}</td>
      <td>${cer(h.served_score)} → ${cer(h.candidate_score)}</td>
      <td class="${h.result === "promoted" ? "ok" : "keep"}">${verdict[h.result] || "보류"}</td></tr>`).join("");
  }
}

// The mascot says its speech-bubble line when it comes into view, and again every so often while
// it stays there.
function greet() {
  const mascot = $(".mascot"), speech = $(".speech");
  if (!mascot || !speech) return;
  let visible = false, last = 0;
  const say = () => {
    if (!visible || document.hidden || Date.now() - last < 6000) return;
    last = Date.now();
    speech.classList.remove("say");
    void speech.offsetWidth; // restart the pop
    speech.classList.add("say");
    chatter(mascot, 1700);
  };
  new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible) setTimeout(say, 400);
  }, { threshold: 0.4 }).observe(mascot);
  setInterval(say, 16000);
}

pie();
fitPhone();
window.addEventListener("resize", fitPhone);
window.addEventListener("scroll", () => $("#nav").classList.toggle("solid", window.scrollY > 40), { passive: true });
refresh();
setInterval(refresh, 30000);
ready.then(greet);
