// 학습록: 단어 도감 (dialect words from AI Hub's labels listed in 우리말샘, scripts/build_dict.py), 구문 복습 (read your own
// corrected sentences aloud; Gemma 4 checks them), 피드백 모아보기 (past coaching feedback).
import { api, me, setMe, uid } from "../api.js";
import { $, $$, esc, icon, sheet, toast, ymd, MODE_LABEL } from "../ui.js";
import { encodeWav, Recorder, supported } from "../recorder.js";
import { feedbackBody } from "./coach.js";

const REGIONS = [["경상", "경상도"], ["강원", "강원도"], ["충청", "충청도"], ["전라", "전라도"], ["제주", "제주도"], ["", "전체 지역"]];
const FROM_PROFILE = { 부산: "경상", 대구: "경상", 강원: "강원", 충청: "충청", 전라: "전라", 제주: "제주" };
let tab = "words", sub = "active", region = null, lastQuery = "";

const norm = (t) => t.replace(/[.,?!~…·"'“”‘’()]/g, "");
function markWords(text, words, cls) {
  const targets = words.map((w) => norm(w).split(/\s+/)).filter((w) => w[0]);
  const toks = text.split(/\s+/);
  const on = new Array(toks.length).fill(false);
  for (const w of targets) {
    for (let i = 0; i + w.length <= toks.length; i++) {
      if (w.every((x, k) => norm(toks[i + k]) === x)) for (let k = 0; k < w.length; k++) on[i + k] = true;
    }
  }
  return toks.map((t, i) => (on[i] ? `<span class="${cls}">${esc(t)}</span>` : esc(t))).join(" ");
}

export async function render(view, which) {
  if (which) tab = ["words", "review", "feedback"].includes(which) ? which : "words";
  if (region === null) region = FROM_PROFILE[me.region] || "경상";
  view.innerHTML = `<div class="tabs-top"><h1>학습록</h1>
    <div class="seg" id="tabs"><button data-t="words">단어 도감</button><button data-t="review">구문 복습</button><button data-t="feedback">피드백 모아보기</button></div></div>
    <div id="body" style="padding:6px 16px 20px"></div>`;
  for (const b of $$("#tabs button", view)) {
    b.classList.toggle("on", b.dataset.t === tab);
    b.onclick = () => { location.hash = "#/journal/" + b.dataset.t; };
  }
  const body = $("#body", view);
  if (tab === "words") return words(body);
  if (tab === "review") return review(body);
  return feedbacks(body);
}

// ------------------------------------------------------------------------- 단어 도감
function wordCard(e) {
  return `<div class="word fade-in"><div class="w">${esc(e.dialect)}${e.standard ? ` <em>/${esc(e.standard)}</em>` : ""}</div>
    <div class="meta">${esc(e.region)}</div>
    <div class="ex">${e.example_dialect ? `“${markWords(e.example_dialect, [e.example_form || e.dialect], "hl-d")}”` : esc(e.dict_definition || "")}</div>
    <button class="star ${e.favorite ? "on" : ""}" data-key="${esc(JSON.stringify([e.region, e.dialect, e.standard]))}" aria-label="즐겨찾기">${icon("star")}</button></div>`;
}

async function words(body) {
  const name = (REGIONS.find(([k]) => k === region) || REGIONS[0])[1];
  body.innerHTML = `<div class="peek"><img src="/assets/icon.png" alt=""><span class="bubble">사투리 단어를 표준어로 찾아보아요!</span></div>
    <div class="region-box">
      <div class="bar"><span class="row" style="gap:6px">${icon("pin").replace("<svg", '<svg style="width:18px;height:18px"')}${name}</span>
        <label class="row" style="gap:2px;font-size:14px">지역 변경하기<select id="region">${REGIONS.map(([k, l]) => `<option value="${k}" ${k === region ? "selected" : ""}>${l}</option>`).join("")}</select>${icon("chevron").replace("<svg", '<svg style="width:16px;height:16px"')}</label></div>
      <div class="body"><div class="search"><input id="q" placeholder="원하는 단어를 검색해주세요." enterkeyhint="search" value="${esc(lastQuery)}">${icon("search")}</div><div class="tags" id="tags"></div></div>
    </div>
    <div id="results"></div>
    <div class="section-title">단어 즐겨찾기 <small>추가된 날짜순</small></div><div class="words" id="favs"></div>
    <div class="section-title">내가 쓴 사투리 <small>AI 코칭 기록</small></div><div id="mine"></div>
    <div class="section-title">${name === "전체 지역" ? "" : name + " "}자주 쓰는 표현 <small id="topn"></small></div><div class="words" id="top"></div>
    <button class="btn ghost block" id="more" hidden style="margin-top:4px">더 보기</button>
    <p class="muted small" style="margin:14px 2px 0;line-height:1.5">AI Hub 방언 발화 데이터에서 나온 표현 중 국립국어원 「우리말샘」에 그 지역 방언으로 실린 것만 모았어요. 출처: 국립국어원 우리말샘(CC BY-SA 2.0 KR)</p>`;
  // more of a list: the next page of /api/dict appended under it
  const more = (btn, list, params, total) => {
    let shown = list.children.length;
    btn.hidden = shown >= total;
    btn.onclick = async () => {
      btn.disabled = true;
      const r = await api.get("/api/dict", { ...params, uid: uid(), offset: shown, limit: 48 });
      list.insertAdjacentHTML("beforeend", r.items.map(wordCard).join(""));
      shown += r.items.length;
      btn.disabled = false;
      btn.hidden = shown >= r.total || !r.items.length;
    };
  };
  const load = async () => {
    const [top, favs, mine] = await Promise.all([
      api.get("/api/dict", { region, uid: uid(), limit: 24 }),
      api.get("/api/dict/favorites", { uid: uid() }),
      api.get("/api/dict/mine", { uid: uid() }),
    ]);
    $("#tags", body).innerHTML = (top.tags.length ? top.tags : top.items.slice(0, 8).map((e) => e.dialect)).map((t) => `<button data-tag="${esc(t)}">${esc(t)}</button>`).join("") + '<button data-tag="">···</button>';
    $("#top", body).innerHTML = top.items.map(wordCard).join("");
    $("#topn", body).textContent = `${top.total.toLocaleString()}개`;
    more($("#more", body), $("#top", body), { region }, top.total);
    $("#favs", body).outerHTML = favs.length ? `<div class="words" id="favs">${favs.map((e) => wordCard({ count: 0, example_dialect: "", ...e })).join("")}</div>`
      : `<div id="favs" class="empty">별표(★)를 누르면 단어가 여기에 모여요</div>`;
    $("#mine", body).innerHTML = mine.length ? `<div class="chips" style="justify-content:flex-start">${mine.map((w) => `<span class="pill">${esc(w.dialect)} → <b style="color:#0a6bff">${esc(w.standard)}</b> · ${w.count}회</span>`).join("")}</div>`
      : `<div class="empty">AI 코칭에서 교정된 표현이 여기에 쌓여요</div>`;
    for (const t of $$("#tags button", body)) t.onclick = () => { $("#q", body).value = t.dataset.tag; search(); };
  };
  let timer;
  const search = async () => {
    const q = $("#q", body).value.trim();
    lastQuery = q;
    const res = $("#results", body);
    if (!q) { res.innerHTML = ""; return; }
    const r = await api.get("/api/dict", { region, q, uid: uid(), limit: 60 });
    res.innerHTML = `<div class="section-title">검색 결과 <small>${r.total.toLocaleString()}건</small></div>` + (r.items.length ? `<div class="words" id="found">${r.items.map(wordCard).join("")}</div><button class="btn ghost block" id="more-found" hidden style="margin:4px 0 8px">더 보기</button>` : `<div class="empty">'${esc(q)}'에 맞는 표현이 없어요</div>`);
    if (r.items.length) more($("#more-found", res), $("#found", res), { region, q }, r.total);
  };
  $("#q", body).oninput = () => { clearTimeout(timer); timer = setTimeout(search, 250); };
  $("#region", body).onchange = (e) => { region = e.target.value; words(body); };
  body.onclick = async (e) => {
    const star = e.target.closest(".star");
    if (!star) return;
    const [r, d, s] = JSON.parse(star.dataset.key);
    const on = !star.classList.contains("on");
    await api.post("/api/dict/favorite", { uid: uid(), region: r, dialect: d, standard: s, on });
    toast(on ? "즐겨찾기에 추가했어요" : "즐겨찾기에서 뺐어요");
    load();
    if (lastQuery) search();
  };
  await load();
  if (lastQuery) search();
}

// ------------------------------------------------------------------------- 구문 복습
// When a sentence falls due again (server/analysis.py review_schedule, half-life regression).
const later = (t) => {
  const days = (t * 1000 - Date.now()) / 86400000;
  return days < 1 ? `${Math.max(1, Math.round(days * 24))}시간 후` : `${Math.round(days)}일 후`;
};

async function review(body) {
  const stats = await api.get("/api/stats", { uid: uid() });
  const rates = stats.review_rate.filter((x) => x !== null);
  const rate = rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null;
  const msg = rate === null ? "코칭에서 고친 문장을 소리 내어 복습해요!" : rate < 60 ? "요즘 복습률이 너무 안 좋아요! ㅠㅠ" : "복습 잘하고 있어요! 이대로 쭉!";
  body.innerHTML = `<div class="peek"><img src="/assets/icon.png" alt=""><span class="bubble">${msg}</span></div>
    <div class="seg blue" id="sub" style="margin:4px 0 14px"><button data-s="active">복습중</button><button data-s="done">복습완료</button><button data-s="fb">복습 피드백</button></div>
    <div id="list"></div>`;
  for (const b of $$("#sub button", body)) {
    b.classList.toggle("on", b.dataset.s === sub);
    b.onclick = () => { sub = b.dataset.s; review(body); };
  }
  const list = $("#list", body);
  if (sub === "fb") {
    const rows = await api.get("/api/sessions", { uid: uid() });
    list.innerHTML = rows.length ? rows.map((r) => `<details class="card acc"><summary>${ymd(r.created)} · ${esc(r.title)} ${icon("chevron").replace("<svg", '<svg style="width:18px;height:18px;color:#8b95a7"')}</summary>
      <div class="acc-body">요약: ${esc(r.summary || "AI 코치 총평이 없는 기록이에요.")}<br>사투리 비율 ${r.dialect_ratio}%${r.points !== null ? ` · 점수 ${r.points}점` : ""}<br><a href="#/coach/history/${r.id}" style="color:#0a6bff;font-weight:700">자세히 보기 →</a></div></details>`).join("")
      : `<div class="empty">아직 코칭 기록이 없어요</div>`;
    return;
  }
  const items = await api.get("/api/reviews", { uid: uid(), status: sub });
  const gen = sub === "active" ? `<button class="btn ghost block" id="gen" style="margin-bottom:14px">${icon("sparkle").replace("<svg", '<svg style="width:18px;height:18px"')} 내 사투리로 맞춤 복습 문장 만들기</button>` : "";
  list.innerHTML = gen + (items.length ? items.map((r) => `<div class="card review-card fade-in">
      <div class="t">${ymd(r.created)} · ${esc(r.title || "복습 문장")}${sub === "active" ? (r.due ? ` · <b style="color:#0a6bff">지금 복습할 차례</b>` : ` · 다음 복습 ${later(r.next_due)}`) : ""}</div>
      <div class="s">${markWords(r.dialect, r.words.map((w) => w.from), "hl-d")}</div>
      <div class="btns">${sub === "active" ? `<button class="btn ghost small" data-study="${r.id}">공부 시작</button>`
        : `<button class="btn gray small" disabled>공부 완료</button><button class="btn ghost small" data-reset="${r.id}">다시하기</button>`}</div></div>`).join("")
    : `<div class="empty">${sub === "active" ? "복습할 문장이 없어요.<br>AI 코칭에서 교정된 문장이 여기에 쌓여요!" : "아직 복습을 마친 문장이 없어요"}</div>`);
  const byId = Object.fromEntries(items.map((r) => [r.id, r]));
  for (const b of $$("[data-study]", list)) b.onclick = () => practice(byId[b.dataset.study], () => review(body));
  for (const b of $$("[data-reset]", list)) b.onclick = async () => { await api.post(`/api/reviews/${b.dataset.reset}/reset`, { uid: uid() }); sub = "active"; review(body); };
  const g = $("#gen", list);
  if (g) g.onclick = async () => {
    g.disabled = true;
    g.innerHTML = '<span class="spinner blue"></span> Gemma 4가 문장을 만드는 중… (30초 정도)';
    try {
      const r = await api.post("/api/reviews/generate", { uid: uid() });
      toast(r.demo ? `체험 모드: 예시 복습 문장 ${r.sentences.length}개를 넣었어요` : `맞춤 복습 문장 ${r.sentences.length}개를 만들었어요`);
      review(body);
    } catch (e) {
      toast(e.message, 3500);
      g.disabled = false;
      g.innerHTML = "내 사투리로 맞춤 복습 문장 만들기";
    }
  };
}

function practice(item, onDone) {
  const { el, close } = sheet(`<h2>표준어로 읽어 보세요</h2>
    <div class="muted small">${esc(item.title || "복습 문장")}</div>
    <div class="practice-target">${markWords(item.standard, item.words.map((w) => w.to), "hl-ok")}</div>
    <div class="muted small" style="text-align:center;line-height:1.6">이전에 말한 문장: ${markWords(item.dialect, item.words.map((w) => w.from), "hl-d")}</div>
    <div class="practice-mic"><button class="mic" id="pm" aria-label="녹음">${icon("mic")}</button><button class="mic-cancel" id="pcancel" hidden aria-label="녹음 취소">${icon("close")}<span>취소</span></button><div id="pcap" class="muted small">버튼을 누르고 문장을 읽은 뒤 다시 눌러요</div></div>
    <div id="verdict"></div>`, { onClose: () => { if (rec && recording) rec.stop(); } });
  let rec = null, recording = false, segs = [];
  const mic = $("#pm", el), cap = $("#pcap", el), cancelBtn = $("#pcancel", el);
  cancelBtn.onclick = async () => {
    if (!recording) return;
    recording = false;
    await rec.cancel();
    segs = [];
    mic.className = "mic";
    mic.innerHTML = icon("mic");
    cancelBtn.hidden = true;
    cap.textContent = "녹음을 취소했어요. 버튼을 누르면 처음부터 다시 녹음해요";
  };
  mic.onclick = async () => {
    if (!recording) {
      if (!supported() || !window.isSecureContext) { toast("이 주소에서는 마이크를 쓸 수 없어요 (HTTPS나 localhost 필요)"); return; }
      segs = [];
      rec = new Recorder({ onSegment: (s) => segs.push(s.samples) });
      try { await rec.start(); } catch { toast("마이크 권한이 필요해요"); return; }
      recording = true;
      cancelBtn.hidden = false;
      mic.classList.add("rec");
      mic.innerHTML = icon("stop");
      cap.textContent = "듣고 있어요 … 다 읽으면 버튼을 눌러요";
      return;
    }
    recording = false;
    cancelBtn.hidden = true;
    await rec.stop();
    mic.className = "mic busy";
    mic.innerHTML = '<span class="spinner"></span>';
    cap.textContent = "Gemma 4가 듣고 있어요 …";
    if (!segs.length) { mic.className = "mic"; mic.innerHTML = icon("mic"); cap.textContent = "말소리를 듣지 못했어요. 다시 해 볼까요?"; return; }
    const total = segs.reduce((a, s) => a + s.length, 0);
    const all = new Float32Array(total);
    let o = 0;
    for (const s of segs) { all.set(s, o); o += s.length; }
    try {
      const r = await api.audio(`/api/reviews/${item.id}/attempt`, encodeWav(all), { uid: uid() });
      mic.className = "mic";
      mic.innerHTML = icon("mic");
      cap.textContent = "다시 읽어 볼 수 있어요";
      if (r.empty) { $("#verdict", el).innerHTML = `<div class="verdict no">말소리를 듣지 못했어요.</div>`; return; }
      $("#verdict", el).innerHTML = r.demo
        ? `<div class="verdict ok"><b>체험 모드: 맞게 읽은 것으로 처리했어요 +${r.xp} XP</b><br>Gemma 4 엔진이 켜져 있으면 실제로 듣고 판정해요.</div>`
        : r.ok
        ? `<div class="verdict ok"><b>표준어로 정확하게 읽었어요! +${r.xp} XP</b><br>들은 문장: ${esc(r.said)}<br>${r.mastered ? "충분히 익혀서 복습완료로 옮겼어요." : `잊어버릴 때쯤 다시 알려 드릴게요 · 다음 복습 ${later(r.next_due)}`}</div>`
        : `<div class="verdict no"><b>${r.kept.length ? r.kept.map((w) => `'${esc(w.from)}'`).join(", ") + "(이)가 아직 사투리로 들려요." : "목표 문장과 조금 달라요."}</b><br>들은 문장: ${esc(r.said)}<br>다음 복습 ${later(r.next_due)}</div>`;
      if (r.ok) {
        api.post("/api/me", { uid: uid() }).then(setMe);
        setTimeout(() => { close(); onDone(); }, 1600);
      }
    } catch (e) {
      mic.className = "mic";
      mic.innerHTML = icon("mic");
      toast(e.message);
    }
  };
}

// ------------------------------------------------------------------------- 피드백 모아보기
async function feedbacks(body) {
  const rows = await api.get("/api/sessions", { uid: uid() });
  if (!rows.length) {
    body.innerHTML = `<div class="peek"><img src="/assets/icon.png" alt=""><span class="bubble">리포트를 바탕으로 차근차근 고쳐나가요!</span></div>
      <div class="empty">아직 코칭 기록이 없어요.<br><br><a class="btn small" href="#/coach/setup">첫 코칭 시작하기</a></div>`;
    return;
  }
  body.innerHTML = `<div class="peek"><img src="/assets/icon.png" alt=""><span class="bubble">리포트를 바탕으로 차근차근 고쳐나가요!</span></div>
    <div class="fb-select" style="margin:4px 0 12px"><select id="pick">${rows.map((r) => `<option value="${r.id}">${ymd(r.created)} · ${esc(r.title)} ${r.mode === "daily" ? "대화" : "답변"} 피드백</option>`).join("")}</select>${icon("chevron").replace("<svg", '<svg style="width:16px;height:16px;flex:none"')}</div>
    <div id="fb"></div>`;
  const show = async (id) => {
    const r = await api.get(`/api/sessions/${id}`, { uid: uid() });
    const d = r.data;
    $("#fb", body).innerHTML = feedbackBody({ segments: d.segments }, { state: "done", speech: d.speech, feedback: d.feedback, score: d.score, scriptMatch: d.script_match, scriptCoverage: d.script_coverage, progress: r.progress })
      + `<a class="btn ghost block" href="#/coach/history/${id}" style="margin-top:4px">${MODE_LABEL[r.mode]} 결과 보기</a>`;
  };
  $("#pick", body).onchange = (e) => show(e.target.value);
  await show(rows[0].id);
}
