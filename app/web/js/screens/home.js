// 홈: profile card, XP gauge, the journey map and the weekly 학습 그래프.
// The map is a layer the screen pans: one stop per tier from 부산 (bottom) to 서울 (in a sky band above the
// picture). Every stop can be opened, locked ones too; the card at the bottom browses them (‹ ›, swipe or a tap
// on a stop) and the map moves to the one shown.
import { api, me, on, status, store, uid } from "../api.js";
import { $, $$, esc, icon, num, toast } from "../ui.js";
import { lineChart, stackedBars } from "../charts.js";
import { openNotifications, openProfile, unreadCount } from "./profile.js";

let graphTab = "dialect";

// assets/map.jpg is 740x950; x, y are pixels of it (y < 0: the sky band above it, SKY pixels tall).
const MAP_W = 740, MAP_H = 950, SKY = 190;
const STOPS = [
  { x: 452, y: 700, place: "부산", text: "출발! 지금 말투 그대로에서 시작해요. 내 사투리 단어와 억양을 알아보는 단계예요." },
  { x: 240, y: 452, text: "자주 쓰는 사투리 단어를 표준어로 바꿔 말하기 시작하는 단계예요." },
  { x: 373, y: 236, text: "단어는 거의 표준어! 이제 억양과 말끝을 다듬어요." },
  { x: 228, y: 50, text: "면접·발표처럼 긴장되는 자리에서도 표준어를 유지하는 단계예요." },
  { x: 380, y: -105, place: "서울", text: "서울 도착! 어떤 자리에서도 자연스러운 표준어로 말해요." },
];
const XP_PER_PRACTICE = 25; // an interview or presentation: 10 + score / 10, +10 when passed
const pct = (st) => `left:${(st.x / MAP_W) * 100}%;top:${((st.y + SKY) / (MAP_H + SKY)) * 100}%`;

export async function render(view) {
  view.className = "";
  const draw = () => {
    const u = me;
    const t = u.tier;
    const remaining = t.max ? t.max - u.xp : 0;
    const tiers = (u.tiers || []).slice(0, STOPS.length);
    const cur = Math.min(t.index ?? 0, tiers.length - 1);
    const state = (i) => (i < cur ? "done" : i === cur ? "here" : "locked");
    const stops = tiers.map((tier, i) => `<button class="stop ${state(i)}" data-i="${i}" style="${pct(STOPS[i])}" aria-label="${i + 1}단계 ${esc(tier.name)}">
        <span class="dot">${state(i) === "done" ? icon("check") : state(i) === "locked" ? icon("lock") : i + 1}</span>
        <span class="tag">${STOPS[i].place ? `${STOPS[i].place} · ` : ""}${esc(tier.name)}</span></button>`).join("");
    view.innerHTML = `<div class="screen home">
      <div class="home-map" id="map"><div class="map-canvas" id="canvas">
        <div class="map-sky" style="height:${(SKY / (MAP_H + SKY)) * 100}%">${SKYLINE}</div>
        <img class="map-img" src="/assets/map.jpg" alt="" style="top:${(SKY / (MAP_H + SKY)) * 100}%">
        ${stops}
        <div class="me-pin" style="${pct(STOPS[cur])}"><img src="/assets/icon.png" alt=""><b>나</b></div>
      </div></div>
      <div class="sky" aria-hidden="true"><i class="cloud c1"></i><i class="cloud c2"></i><i class="cloud c3"></i>
        <div class="birds"><i></i><i></i><i></i></div><div class="birds b2"><i></i><i></i></div></div>
      <div class="wrap">
        <div class="profile-card">
          <div class="avatar"><img src="/assets/icon.png" alt=""><div class="xp">${num(u.xp)} XP</div></div>
          <div style="min-width:0">
            <span class="tier">${esc(t.name)}</span>
            <div class="name">${esc(u.nickname)}<small>${esc(u.tag)}</small></div>
            <button class="edit" id="edit">프로필 수정하기 ${icon("pencil")}</button>
          </div>
          <div class="side">
            <button class="icon-btn bell" id="bell" aria-label="알림">${icon("bell")}<span class="badge" hidden></span></button>
            <button class="icon-btn toggle-graph" id="toggle" aria-label="학습 그래프">${icon("chevron")}</button>
          </div>
        </div>
        <div class="xpbar"><div class="gauge">XP</div><div class="track"><div class="fill" style="width:0"></div>
          <div class="label">${t.next ? `${esc(t.next)}까지 ${num(remaining)} XP` : "표준어 마스터 달성!"}</div></div></div>
        <div id="graph"></div>
        ${store.get("seotturi.graphHintDone", false) ? "" : `<div class="home-hint" id="hint"><span class="bubble up">버튼을 눌러 이번주<br>학습 그래프를 살펴봐!!</span></div>`}
      </div>
      <img class="star-float" src="/assets/star.png" alt="" style="right:9%;top:46%">
      <div class="home-spacer"></div>
      <div class="home-cta stage-card" id="stage"></div>
    </div>`;
    requestAnimationFrame(() => { const f = $(".fill", view); if (f) f.style.width = `${Math.max(3, t.progress * 100)}%`; });
    $("#edit", view).onclick = () => openProfile(draw);
    $("#bell", view).onclick = async () => { await openNotifications(); badge(); };
    $("#toggle", view).onclick = toggle;
    badge();
    journey(tiers, cur, state);
  };

  // ---- the journey: the card shows one stop, the map pans to it ----
  let sel = 0, focus = null, ro = null;
  const journey = (tiers, cur, state) => {
    const map = $("#map", view), canvas = $("#canvas", view), card = $("#stage", view);
    const u = me;
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const fit = (point, animate) => {
      const W = map.clientWidth, H = map.clientHeight;
      if (!W || !H) return;
      // The map area left free by the cards: all of it on a wide screen (cards beside the map), else the band
      // between the profile card and the stage card. The map is scaled so that band shows about two stops,
      // and may move until its edge meets the band (the screen's background matches the sky and the sea).
      const m = map.getBoundingClientRect(), w = $(".wrap", view).getBoundingClientRect(), c = card.getBoundingClientRect();
      const beside = Math.min(c.right, m.right) - Math.max(c.left, m.left) < W * 0.25;  // cards next to the map, not over it
      const top = beside ? 0 : clamp(w.bottom - m.top, 0, H), bottom = beside ? H : clamp(c.top - m.top, top + 120, H);
      const s = Math.max(W / MAP_W, beside ? H / (MAP_H + SKY) : (bottom - top) / 430);
      const Wc = MAP_W * s, Hc = (MAP_H + SKY) * s;
      canvas.style.width = `${Wc}px`;
      canvas.style.height = `${Hc}px`;
      const X = clamp(W / 2 - point.x * s, W - Wc, 0);
      const ty = (top + bottom) / 2 - (point.y + SKY) * s;
      const Y = beside ? clamp(ty, H - Hc, 0) : clamp(ty, bottom - Hc, top);
      canvas.classList.toggle("still", !animate);
      canvas.style.transform = `translate(${X}px, ${Y}px)`;
    };
    const show = (i, animate = true) => {
      sel = Math.max(0, Math.min(tiers.length - 1, i));
      const tier = tiers[sel], st = STOPS[sel], k = state(sel);
      const need = Math.max(0, tier.min - u.xp);
      const nextTier = tiers[cur + 1];
      const chip = { done: "지나온 곳", here: "지금 여기", locked: `잠김 · ${num(tier.min)} XP에 열려요` }[k];
      const line = k === "locked" ? `앞으로 ${num(need)} XP · 면접·발표 연습 약 ${Math.ceil(need / XP_PER_PRACTICE)}번`
        : k === "here" ? (nextTier ? `다음 단계까지 ${num(nextTier.min - u.xp)} XP` : "여정을 모두 마쳤어요!")
        : `${num(tier.min)} XP에 지나왔어요`;
      const thumb = st.y < 0 ? `<div class="thumb seoul">${SKYLINE}<b>서울</b></div>`
        : `<div class="thumb ${k}" style="background-size:${MAP_W * 0.3}px ${MAP_H * 0.3}px;background-position:${-(st.x * 0.3 - 34)}px ${-(st.y * 0.3 - 34)}px">${k === "locked" ? `<i>${icon("lock")}</i>` : ""}</div>`;
      card.innerHTML = `<div class="stage-main">${thumb}
          <div class="stage-info"><span class="chip ${k}">${chip}</span>
            <b>${sel + 1}단계 · ${esc(tier.name)}${st.place ? ` <small>${st.place}</small>` : ""}</b>
            <span class="desc">${st.text}</span><span class="need">${line}</span></div></div>
        <div class="stage-nav">
          <button class="nav" id="prev" aria-label="이전 단계" ${sel === 0 ? "disabled" : ""}>${icon("back")}</button>
          <span class="dots">${tiers.map((_, j) => `<i class="${j === sel ? "on" : ""} ${state(j)}"></i>`).join("")}</span>
          <button class="nav next" id="next" aria-label="다음 단계" ${sel === tiers.length - 1 ? "disabled" : ""}>${icon("back")}</button>
          <a class="btn small" href="#/coach/setup">코칭 시작</a>
        </div>`;
      card.classList.remove("swap");
      void card.offsetWidth;
      card.classList.add("swap");
      for (const b of $$(".stop", canvas)) b.classList.toggle("sel", +b.dataset.i === sel);
      $("#prev", card).onclick = () => show(sel - 1);
      $("#next", card).onclick = () => show(sel + 1);
      focus = STOPS[sel];
      fit(focus, animate);
    };
    for (const b of $$(".stop", canvas)) b.onclick = () => show(+b.dataset.i);
    let x0 = null;  // swipe the card
    card.onpointerdown = (e) => { x0 = e.clientX; };
    card.onpointerup = (e) => {
      if (x0 !== null && Math.abs(e.clientX - x0) > 40) show(sel + (e.clientX < x0 ? 1 : -1));
      x0 = null;
    };
    // First the next destination, with the map between it and where the user stands.
    show(Math.min(cur + 1, tiers.length - 1), false);
    const a = STOPS[cur], b = STOPS[Math.min(cur + 1, tiers.length - 1)];
    focus = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    fit(focus, false);
    if (ro) ro.disconnect();
    ro = new ResizeObserver(() => fit(focus, false));
    ro.observe(map);
  };

  const badge = async () => {
    const b = $("#bell .badge", view);
    if (!b) return;
    const n = unreadCount(await status());
    b.hidden = !n;
    b.textContent = n;
  };

  let open = false;
  const toggle = async () => {
    open = !open;
    store.set("seotturi.graphHintDone", true);
    const hint = $("#hint", view);
    if (hint) hint.remove();
    $("#toggle", view).classList.toggle("open", open);
    const g = $("#graph", view);
    if (!open) { g.innerHTML = ""; return; }
    g.innerHTML = `<div class="graph-panel"><h3>학습 그래프</h3>
      <div class="seg" id="gtabs"><button data-k="dialect">사투리 지수</button><button data-k="practice">연습 로그</button><button data-k="review">복습 성공률</button></div>
      <div id="gbody"><div class="shimmer" style="height:240px;margin-top:14px"></div></div></div>`;
    let stats;
    try {
      stats = await api.get("/api/stats", { uid: uid() });
    } catch (e) {
      $("#gbody", view).innerHTML = `<div class="empty">${esc(e.message)}</div>`;
      return;
    }
    const show = () => {
      for (const b of $$("#gtabs button", view)) b.classList.toggle("on", b.dataset.k === graphTab);
      $("#gbody", view).innerHTML = graph(stats, graphTab);
      const help = $("#ghelp", view);
      if (help) help.onclick = () => toast(HELP[graphTab], 4200);
    };
    for (const b of $$("#gtabs button", view)) b.onclick = () => { graphTab = b.dataset.k; show(); };
    show();
  };

  draw();
  const off = on("me", draw);
  return () => { off(); if (ro) ro.disconnect(); };
}

// Seoul in the sky band above the map: a few towers and N Seoul Tower, drawn flat.
const SKYLINE = `<svg class="skyline" viewBox="0 0 200 60" preserveAspectRatio="xMidYMax meet" aria-hidden="true">
  <path d="M0 60V44h14V36h10v8h8V30h12v30M50 60V38h9V26h11v34M78 60V40h12v-8h10v28M106 60V34h14v26M126 60V42h10v-6h12v24M154 60V30h12v30M172 60V40h12v6h16v14Z"/>
  <path d="M96 60V24h-3l3-6V8l1.6-4L99 8v10l3 6h-3v36Z"/></svg>`;

const HELP = {
  dialect: "사투리 지수: AI 코칭에서 표준어로 교정된 사투리 표현(어절) 수예요. 낮을수록 표준어에 가까워요.",
  practice: "연습 로그: 하루 동안 모드별로 말한 시간(분)이에요.",
  review: "복습 성공률: 구문 복습에서 표준어로 바르게 읽은 비율이에요.",
};

function graph(s, tab) {
  const today = s.days.length - 1;
  let body = "", meta = "", callout = "";
  if (tab === "dialect") {
    const v = s.dialect_words;
    const { svg, anchor } = lineChart({ days: s.days, values: v, unit: "어절(개)", highlight: today, maxValue: Math.max(...v) < 4 ? 4 : undefined });
    const diff = Math.round(v[today] - s.prev_week_avg);
    const msg = s.totals.words === 0 ? "아직 기록이 없어요. AI 코칭을 시작해 보세요!"
      : diff > 0 ? `지난주 평균보다 ${diff}개 더 사용했어요!` : diff < 0 ? `지난주 평균보다 ${-diff}개 덜 사용했어요!` : "지난주 평균과 같아요!";
    meta = "사투리 지수 측정";
    callout = anchor ? `<span class="bubble callout" style="left:${Math.min(66, Math.max(34, anchor.x))}%;top:${Math.max(18, anchor.y - 4)}%">${esc(msg)}</span>` : "";
    body = svg;
  } else if (tab === "practice") {
    const p = s.practice_minutes;
    const { svg, legend } = stackedBars({
      days: s.days, unit: "분",
      series: [{ name: "면접", color: "#2f6bff", values: p.interview }, { name: "발표", color: "#4fb3ff", values: p.presentation },
        { name: "일상회화", color: "#55dcb9", values: p.daily }, { name: "복습", color: "#a9b8ff", values: p.review }],
    });
    const total = Object.values(p).flat().reduce((a, b) => a + b, 0);
    meta = "연습 로그 측정";
    body = `<div style="text-align:center;margin:-2px 0 6px">${legend}</div>${svg}<div class="muted small" style="text-align:center">이번 주 ${total.toFixed(1)}분 연습했어요</div>`;
  } else {
    const v = s.review_rate;
    const real = v.filter((x) => x !== null);
    const best = real.length ? v.indexOf(Math.max(...real)) : -1;
    const { svg, anchor } = lineChart({ days: s.days, values: v, unit: "%", color: "#22c39a", fill: "#5fe0c0", maxValue: 100, highlight: best });
    const avg = real.length ? Math.round(real.reduce((a, b) => a + b, 0) / real.length) : null;
    meta = "복습 성공률 측정";
    const msg = avg === 0 ? "아직 성공 기록이 없어요. 다시 도전해 봐요!" : real.length === 1 ? `이번 주 성공률은 ${avg}%예요!` : `이번 주 평균 ${avg}%, 이날이 최고예요!`;
    callout = anchor ? `<span class="bubble callout" style="left:${Math.min(66, Math.max(34, anchor.x))}%;top:${Math.max(18, anchor.y - 4)}%">${msg}</span>` : "";
    body = svg + (real.length ? "" : `<div class="muted small" style="text-align:center">학습록 › 구문 복습에서 문장을 읽어 보면 기록돼요</div>`);
  }
  return `<div class="graph-meta"><span>최근 7일 ${icon("chevron", "").replace("<svg", '<svg style="width:14px;height:14px;vertical-align:-2px"')}</span>
    <button id="ghelp" class="muted small" style="display:inline-flex;gap:4px;align-items:center">${meta} ${icon("info").replace("<svg", '<svg style="width:16px;height:16px;color:#9aa4b2"')}</button></div>
    <div class="chart">${callout}${body}</div>`;
}
