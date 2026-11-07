// AI 코칭: new project (모드 → 세부 카테고리 → 피드백 → 성우·배경), preparation, the live
// session (Gemma 4 hears each phrase as soon as the speaker pauses), feedback and result.
import { api, me, setMe, status, stream, uid } from "../api.js";
import { $, $$, esc, icon, marked, modal, MODE_LABEL, standardMarked, toast, ymd } from "../ui.js";
import { encodeWav, Recorder, supported } from "../recorder.js";
import * as voice from "../voice.js";
import { confetti } from "../alive.js";
import { castOf, SITUATIONS, subject } from "../personas.js";

const MODES = [
  { id: "interview", title: "면접 모드", desc: "자주 나오는 면접 질문, AI랑 미리 연습해봐요!", img: "/assets/mode-interview.jpg", scene: "/assets/interviewer.jpg", who: "면접관" },
  { id: "presentation", title: "발표 모드", desc: "실제 발표처럼 말하면서 자신감을 기를 수 있어요!", img: "/assets/mode-presentation.jpg", scene: "/assets/audience.jpg", who: "청중" },
  { id: "daily", title: "일상회화 모드", desc: "실제로 대화하듯이 AI와 얘기하며 표준어로 말하는 감각을 키워보아요!", img: "/assets/mode-daily.jpg", scene: "/assets/friend.jpg", who: "대화 상대" },
];
const STEPS = {
  interview: ["모드", "세부 카테고리", "피드백", "성우·배경"],
  presentation: ["모드", "주제·대본", "피드백", "청중"],
  daily: ["모드", "상황", "피드백", "대화 상대"],
};
const CATEGORIES = {
  interview: ["자기소개", "지원 동기", "미래방향", "개인 경험", "원하는 질문"],
  daily: Object.keys(SITUATIONS),
};
const QUESTIONS = {
  자기소개: ["간단하게 자기소개 부탁드립니다.", "1분 동안 본인을 소개해 주세요.", "본인을 한 단어로 표현하고, 그 이유를 말씀해 주세요."],
  "지원 동기": ["우리 회사에 지원한 동기가 무엇인가요?", "이 직무를 선택한 이유를 말씀해 주세요.", "많은 회사 중 왜 저희 회사인가요?"],
  미래방향: ["입사 후 5년 뒤 어떤 모습이 되어 있을 것 같나요?", "입사하면 가장 먼저 이루고 싶은 목표는 무엇인가요?", "앞으로 어떤 전문가로 성장하고 싶나요?"],
  "개인 경험": ["팀 프로젝트에서 갈등을 해결한 경험을 말씀해 주세요.", "가장 크게 실패했던 경험과 그 뒤에 배운 점은 무엇인가요?", "끝까지 책임지고 해낸 경험을 하나 소개해 주세요."],
};
const FOCUS = ["발음·말투", "사투리·표준어 사용", "내용의 명확성", "논리적 구조", "태도·자세", "자신감 표현", "시간 관리", "키워드 사용", "전체적인 인상"];
const FOCUS_DEFAULT = ["발음·말투", "사투리·표준어 사용", "내용의 명확성", "논리적 구조", "시간 관리"];
const SCRIPT_TYPES = "txt · docx · pdf · pptx · hwpx";
const SCRIPT_MAX_MB = 50;  // a whole file (pdf, txt, or a zip this browser cannot read); app.py MAX_SCRIPT_BYTES
// The zip parts that hold a script's text (server/analysis.py script_text reads the same ones).
const SCRIPT_PARTS = /^(word\/document\.xml|ppt\/slides\/slide\d+\.xml|ppt\/notesSlides\/notesSlide\d+\.xml|Contents\/section\d+\.xml)$/;

// Slide decks are mostly pictures and video (서투리_최종 PPT.pptx: 307 MB, of which its text is 62 KB), so for
// docx, pptx and hwpx the browser reads only the text parts, straight from the zip's central directory, and
// sends those. null when it cannot (no DecompressionStream, zip64, unknown compression): send the file.
async function zipParts(file) {
  if (typeof DecompressionStream !== "function") return null;
  const bytes = async (a, b) => new Uint8Array(await file.slice(a, b).arrayBuffer());
  const view = (u) => new DataView(u.buffer, u.byteOffset, u.byteLength);
  const tail = await bytes(Math.max(0, file.size - 65557), file.size), t = view(tail);
  let e = tail.length - 22;
  while (e >= 0 && t.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) return null;
  const count = t.getUint16(e + 10, true), size = t.getUint32(e + 12, true), at = t.getUint32(e + 16, true);
  if (at === 0xffffffff || count === 0xffff) return null;
  const cd = await bytes(at, at + size), v = view(cd), utf8 = new TextDecoder();
  const parts = {};
  for (let p = 0, k = 0; k < count && p + 46 <= cd.length && v.getUint32(p, true) === 0x02014b50; k++) {
    const method = v.getUint16(p + 10, true), packed = v.getUint32(p + 20, true), offset = v.getUint32(p + 42, true);
    const nl = v.getUint16(p + 28, true), name = utf8.decode(cd.subarray(p + 46, p + 46 + nl));
    p += 46 + nl + v.getUint16(p + 30, true) + v.getUint16(p + 32, true);
    if (!SCRIPT_PARTS.test(name)) continue;
    if (method !== 0 && method !== 8) return null;
    const local = view(await bytes(offset, offset + 30));
    const start = offset + 30 + local.getUint16(26, true) + local.getUint16(28, true);
    let blob = file.slice(start, start + packed);
    if (method === 8) blob = await new Response(blob.stream().pipeThrough(new DecompressionStream("deflate-raw"))).blob();
    parts[name] = await blob.text();
  }
  return Object.keys(parts).length ? parts : null;
}

let P = null; // the project being set up / recorded / reviewed

function newProject() {
  P = { mode: null, category: null, custom: "", topic: "", script: "", focus: [...FOCUS_DEFAULT], extra: [], persona: null,
    step: 0, created: Date.now() / 1000, question: "", segments: [], history: [], fb: null };
}

const modeOf = (p) => MODES.find((m) => m.id === p.mode);
// The partner chosen in the setup (web/js/personas.js) and the scene shown behind them.
const personaOf = (p) => castOf(p.mode, p.category)[p.persona] || null;
const sceneOf = (p) => (personaOf(p) || {}).img || modeOf(p).scene;
function title(p) {
  if (p.mode === "interview") return `모의 면접: ${p.category === "원하는 질문" ? "직접 고른 질문" : p.category}`;
  if (p.mode === "presentation") return `모의 발표: ${p.topic || "자유 주제"}`;
  return `일상회화: ${p.category}`;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------------- landing
export async function landing(view) {
  view.className = "";
  view.innerHTML = `<div class="coach-home" style="min-height:100%">
    <div class="room" style="background-image:url(/assets/room-blur.jpg)"></div><div class="room-shade"></div>
    <div class="center">
      <span class="model-chip" id="chip"><span class="dot wait"></span>AI 상태 확인 중…</span>
      <button class="new-project" id="new">${icon("folderPlus")} 새 프로젝트 만들기</button>
    </div>
    <div class="recent" id="recent" style="padding-bottom:calc(var(--tab-h) + 34px)"></div>
  </div>`;
  $("#new", view).onclick = () => { newProject(); location.hash = "#/coach/setup"; };
  status(true).then((s) => {
    const chip = $("#chip", view);
    if (!chip) return;
    if (s.ready) chip.innerHTML = `<span class="dot"></span>Gemma 4 · 서투리 LoRA ${esc(s.model.id)} · 대기 ${s.queue}건`;
    else chip.innerHTML = `<span class="dot wait"></span>${s.demo ? "체험 모드 · Gemma 4 엔진이 꺼져 있어 AI 답은 예시예요" : s.error === "offline" ? "AI 엔진 연결 안 됨" : "AI 모델을 불러오는 중이에요"}`;
  });
  try {
    const rows = (await api.get("/api/sessions", { uid: uid() })).slice(0, 3);
    if (rows.length) {
      $("#recent", view).innerHTML = `<h3>최근 코칭</h3>` + rows.map((r) => `<a class="recent-item" href="#/coach/history/${r.id}">
        <div class="score ${r.passed === 0 ? "fail" : ""}">${r.points ?? "💬"}</div>
        <div style="min-width:0"><div class="t">${ymd(r.created)} · ${esc(r.title)}</div><div class="s">${esc(r.summary || `사투리 비율 ${r.dialect_ratio}%`)}</div></div></a>`).join("");
    }
  } catch { /* first visit */ }
}

// ---------------------------------------------------------------------------------- setup
export async function setup(view) {
  if (!P) newProject();
  const draw = () => {
    const steps = STEPS[P.mode || "interview"];
    const ok = valid();
    view.innerHTML = `<div class="setup" style="min-height:100%;display:flex;flex-direction:column">
      <div class="topbar"><button class="back" id="back">${icon("back")}</button><h1>${["AI 코칭 설정", "세부 카테고리", "피드백 설정", steps[3]][P.step]}</h1><span></span></div>
      <div style="padding:0 18px;flex:1">
        <div class="stepper">${steps.map((s, i) => `<div class="s ${i === P.step ? "on" : i < P.step ? "done" : ""}"><i></i>${s}</div>`).join("")}</div>
        ${[stepMode, stepCategory, stepFocus, stepPersona][P.step]()}
      </div>
      <div class="bottom-cta"><button class="btn block" id="next" ${ok ? "" : "disabled"}>${P.step === 3 ? "시작 준비하기" : "다음"}</button></div>
    </div>`;
    $("#back", view).onclick = () => { if (P.step) { P.step--; draw(); } else location.hash = "#/coach"; };
    $("#next", view).onclick = () => { if (P.step < 3) { P.step++; draw(); view.scrollTop = 0; } else location.hash = "#/coach/prepare"; };
    for (const b of $$("[data-mode]", view)) b.onclick = () => { if (P.mode !== b.dataset.mode) { P.mode = b.dataset.mode; P.category = null; P.persona = null; } draw(); };
    for (const b of $$("[data-cat]", view)) b.onclick = () => { if (P.category !== b.dataset.cat) { P.category = b.dataset.cat; if (P.mode === "daily") P.persona = null; } draw(); };
    for (const b of $$("[data-focus]", view)) b.onclick = () => {
      const f = b.dataset.focus;
      P.focus = P.focus.includes(f) ? P.focus.filter((x) => x !== f) : [...P.focus, f];
      draw();
    };
    for (const b of $$("[data-persona]", view)) b.onclick = () => { P.persona = +b.dataset.persona; draw(); };
    const bind = (id, key) => { const el = $(id, view); if (el) el.oninput = () => { P[key] = el.value; $("#next", view).disabled = !valid(); }; };
    bind("#custom", "custom"); bind("#topic", "topic"); bind("#script", "script");
    const sf = $("#scriptfile", view);
    if (sf) {  // a script file: the server reads its text (server/analysis.py script_text)
      const input = $("#scriptinput", view), info = $("#scriptinfo", view);
      sf.onclick = () => input.click();
      input.onchange = async () => {
        const f = input.files[0];
        input.value = "";
        if (!f) return;
        sf.disabled = true;
        info.textContent = `${f.name} 읽는 중…`;
        try {
          const parts = /\.(docx|pptx|hwpx)$/i.test(f.name) ? await zipParts(f).catch(() => null) : null;
          if (!parts && f.size > SCRIPT_MAX_MB * 1e6) throw new Error(`파일이 너무 커요. ${SCRIPT_MAX_MB}MB 이하로 올려 주세요.`);
          const r = parts ? await api.post("/api/script/extract", { name: f.name, parts })
            : await api.post("/api/script/extract", f, { name: f.name });
          P.script = r.text;
          $("#script", view).value = r.text;
          if (!P.topic.trim()) {  // no topic yet: the file name stands in
            P.topic = f.name.replace(/\.[^.]+$/, "").slice(0, 30);
            $("#topic", view).value = P.topic;
          }
          info.textContent = `${f.name} · ${r.chars.toLocaleString()}자 불러옴${r.truncated ? " (앞 2만 자까지)" : ""}`;
          $("#next", view).disabled = !valid();
        } catch (e) {
          info.textContent = SCRIPT_TYPES;
          toast(e.message, 4500);
        } finally {
          sf.disabled = false;
        }
      };
    }
    const add = $("#add-focus", view);
    if (add) add.onclick = () => {
      const v = (prompt("받고 싶은 피드백 항목을 입력해 주세요 (예: 존댓말 사용)") || "").trim().slice(0, 20);
      if (v && !P.extra.includes(v)) { P.extra.push(v); P.focus.push(v); }
      draw();
    };
  };
  const valid = () => [
    () => !!P.mode,
    () => P.mode === "presentation" ? P.topic.trim().length > 0 : !!P.category && (P.category !== "원하는 질문" || P.custom.trim().length > 3),
    () => P.focus.length > 0,
    () => P.persona !== null,
  ][P.step]();
  const modeCard = (m) => `<button class="mode-card ${P.mode === m.id ? "on" : P.step > 0 ? "dim" : ""}" data-mode="${m.id}" ${P.step > 0 ? "disabled" : ""}>
      <img src="${m.img}" alt=""><div><b>${m.title}</b><span>${m.desc}</span></div></button>`;
  const stepMode = () => `<h2>AI 코칭을 받고 싶은 모드를 선택해주세요</h2>${MODES.map(modeCard).join("")}`;
  const stepCategory = () => {
    const m = modeOf(P);
    if (P.mode === "presentation") {
      return `<h2>발표 주제와 대본을 알려주세요</h2>${modeCard(m)}
        <label class="field"><span>발표 주제</span><input class="input" id="topic" maxlength="30" placeholder="예: 서투리 서비스 소개" value="${esc(P.topic)}"></label>
        <label class="field"><span>대본 <small class="muted">· 선택, 넣으면 대본 일치도를 채점해요</small></span>
          <textarea class="textarea" id="script" placeholder="발표할 내용을 붙여 넣거나 파일로 올려 주세요">${esc(P.script)}</textarea></label>
        <div class="script-file"><button class="btn small gray" id="scriptfile" type="button">${icon("upload")} 파일로 올리기</button>
          <span class="muted small" id="scriptinfo">${SCRIPT_TYPES}</span>
          <input type="file" id="scriptinput" accept=".txt,.md,.docx,.pdf,.pptx,.hwpx,.hwp" hidden></div>`;
    }
    return `<h2>모드 속 세부 카테고리를 선택해주세요</h2>${modeCard(m)}
      <div class="chips">${CATEGORIES[P.mode].map((c) => `<button class="chip ${P.category === c ? "on" : ""}" data-cat="${c}">${c}</button>`).join("")}</div>
      ${P.category === "원하는 질문" ? `<label class="field" style="margin-top:18px"><span>연습하고 싶은 질문</span><input class="input" id="custom" maxlength="80" placeholder="예: 우리 회사의 단점은 무엇이라고 생각하나요?" value="${esc(P.custom)}"></label>` : ""}`;
  };
  const stepFocus = () => `<h2>어떤 부분에 대한 피드백을 받고 싶으신가요?</h2>
    <div class="checks">${[...FOCUS, ...P.extra].map((f) => `<button class="check ${P.focus.includes(f) ? "on" : ""}" data-focus="${esc(f)}"><i>${icon("check")}</i>${esc(f)}</button>`).join("")}
    <button class="check" id="add-focus"><i style="background:#e9edf3">${icon("plus").replace("<svg", '<svg style="opacity:1;color:#4b5567"')}</i>직접 추가</button></div>`;
  const stepPersona = () => {
    const who = { interview: "면접을 함께할 AI 면접관을 선택해 주세요", presentation: "발표를 들을 청중을 선택해 주세요" }[P.mode]
      || (SITUATIONS[P.category] || {}).pick || "대화할 상대를 선택해 주세요";
    return `<h2>${who}</h2>${castOf(P.mode, P.category).map((p, i) => `<button class="persona ${p.face ? "has-face" : ""} ${P.persona === i ? "on" : ""}" data-persona="${i}">
      ${p.face ? `<img class="face" src="${p.face}" alt="">` : ""}<div><b>${esc(p.name)}</b><span>${esc(p.desc)}</span></div></button>`).join("")}`;
  };
  draw();
}

// ---------------------------------------------------------------------------------- prepare
export async function prepare(view) {
  if (!P || P.persona === null) { location.replace("#/coach"); return; }
  const m = modeOf(P);
  const who = { interview: "면접관과 배경을", presentation: "청중과 발표장을", daily: "대화 상대를" }[P.mode];
  view.innerHTML = `<div class="prepare">
    <div class="topbar" style="padding-left:0;padding-right:0;background:none"><button class="back" id="back">${icon("back")}</button><h1>${STEPS[P.mode][3]}</h1><span></span></div>
    <div class="stepper" style="margin-bottom:4px">${STEPS[P.mode].map((s, i) => `<div class="s ${i === 3 ? "on" : "done"}"><i></i>${s}</div>`).join("")}</div>
    <div class="prep-card"><div class="img silhouette" style="background-image:url(${sceneOf(P)})"></div><div class="veil"></div>
      <div class="msg"><b id="pmsg">${who} 준비하고 있어요 …</b><div class="progress"><i id="bar"></i></div><small id="psub">곧 준비가 끝나니 조금만 기다려 주세요!</small></div></div>
    <div style="padding-top:14px"><button class="btn block" id="go" disabled>시작하기</button></div>
  </div>`;
  $("#back", view).onclick = () => { P.step = 3; location.hash = "#/coach/setup"; };
  let alive = true;
  const bar = $("#bar", view);
  const t0 = Date.now();
  for (;;) {
    const s = await status(true);
    const pct = Math.min(s.ready ? 100 : 90, ((Date.now() - t0) / 1800) * 100);
    if (!alive) return;
    bar.style.width = pct + "%";
    if (s.ready && pct >= 100) break;
    if (!s.ready) $("#psub", view).textContent = s.error === "offline" ? "AI 엔진에 연결하지 못했어요. 서버를 확인해 주세요." : "Gemma 4 모델을 GPU에 올리는 중이에요 (최대 5분)";
    await sleep(s.ready ? 250 : 4000);
  }
  if (!alive) return;
  P.question = P.mode === "interview"
    ? (P.category === "원하는 질문" ? P.custom.trim() : QUESTIONS[P.category][Math.floor(Math.random() * 3)])
    : P.mode === "presentation" ? `주제: ${P.topic}` : personaOf(P).opener;
  $(".img", view).classList.remove("silhouette");
  $(".veil", view).style.opacity = ".35";
  $("#pmsg", view).textContent = `준비가 완료되었어요!`;
  $("#psub", view).textContent = { interview: "이제부터 면접을 시작해볼까요?", presentation: "이제부터 발표를 시작해볼까요?", daily: "이제부터 대화를 시작해볼까요?" }[P.mode];
  $(".progress", view).remove();
  const go = $("#go", view);
  go.disabled = false;
  go.onclick = () => { voice.unlock(); P.segments = []; P.history = P.mode === "daily" ? [{ role: "ai", text: P.question }] : []; P.fb = null; location.hash = "#/coach/session"; };
  return () => { alive = false; };
}

// ---------------------------------------------------------------------------------- session
export async function session(view) {
  if (!P || !P.question) { location.replace("#/coach"); return; }
  return P.mode === "daily" ? chatSession(view) : talkSession(view);
}

function asrSubmit(item, blob, onChange) {
  return (async () => {
    for (let attempt = 0; attempt < 30; attempt++) {
      try {
        const r = await api.audio("/api/asr", blob, { uid: uid(), mode: (P && P.mode) || "" });
        if (r.empty) item.status = "empty";
        else Object.assign(item, r, { status: "done" });
        break;
      } catch (e) {
        if (e.status === 503 && attempt < 29) { item.note = "AI 모델 준비 중…"; onChange(); await sleep(6000); continue; }
        item.status = "error";
        item.error = e.message;
        break;
      }
    }
    onChange();
  })();
}

// The accent zone so far: the accent classifier's probabilities averaged over the sentences by length,
// shown from the first sentence on, however short (server/analysis.py zone_verdict does the same for the saved
// result): with the standard detector, regional or not first, then which region among the regions.
function zoneOf(segs) {
  const withProbs = segs.filter((s) => s.zone_probs);
  const secs = withProbs.reduce((a, s) => a + (s.duration || 0), 0);
  if (!withProbs.length || !secs) return null;
  const avg = {};
  for (const s of withProbs) for (const [z, p] of Object.entries(s.zone_probs)) avg[z] = (avg[z] || 0) + (p * (s.duration || 0)) / secs;
  const last = withProbs[withProbs.length - 1];
  const minConf = last.zone_min_conf ?? 0.6;
  if (last.zone_two_step && "서울·경기" in avg) {
    const regional = 1 - avg["서울·경기"];
    if (regional < (last.zone_regional_min ?? 0.5)) return `표준어에 가까움 ${Math.round(avg["서울·경기"] * 100)}%`;
    const [best, p] = Object.entries(avg).filter(([z]) => z !== "서울·경기").sort((a, b) => b[1] - a[1])[0];
    if (p / regional < minConf) return "지역 억양 · 지역은 확실하지 않음";
    return `${best} ${Math.round((p / regional) * 100)}%`;
  }
  const [best, p] = Object.entries(avg).sort((a, b) => b[1] - a[1])[0];
  if (p < minConf) return "확실하지 않음";
  return `${best === "서울·경기" ? "표준어에 가까움" : best} ${Math.round(p * 100)}%`;
}

function micUI(view) {
  return {
    set(state, caption = "") {
      const mic = $("#mic", view);
      mic.className = "mic" + ({ rec: " rec", busy: " busy", done: " done" }[state] || "");
      mic.innerHTML = state === "rec" ? `<span class="bars">${"<i></i>".repeat(5)}</span>` : state === "busy" ? '<span class="spinner"></span>'
        : state === "done" ? icon("check") : icon("mic");
      $("#cap", view).textContent = caption;
      const hint = $("#hint", view);
      if (hint) hint.hidden = state !== "idle";
      const cancel = $("#cancel", view);
      if (cancel) cancel.hidden = state !== "rec";
    },
    level(v) {
      const bars = $$("#mic .bars i", view);
      const k = [0.6, 0.85, 1, 0.85, 0.6];
      bars.forEach((b, i) => { b.style.height = `${8 + Math.min(32, v * 420 * k[i] * (0.7 + Math.random() * 0.6))}px`; });
    },
  };
}

function talkSession(view) {
  const m = modeOf(P);
  const persona = personaOf(P);
  view.innerHTML = `<div class="session">
    <div class="room" data-speaker style="background-image:url(${sceneOf(P)})"></div><div class="room-shade"></div>
    <button class="close" id="quit" aria-label="나가기">${icon("close")}</button>
    <div class="date-pill">${ymd(P.created)} · ${esc(title(P))}</div>
    ${P.mode === "presentation" && P.script ? '<button class="script-toggle" id="scriptbtn">대본</button>' : ""}
    <div class="question" id="qcard"><small>${esc(m.who)} · ${esc(persona.name)}</small>${esc(persona.hello ? persona.hello + " " : "")}${esc(P.question)}
      ${voice.supported() ? `<button class="speak" id="speak" aria-label="질문 다시 듣기">${icon(voice.enabled() ? "speaker" : "speakerOff")}</button>` : ""}</div>
    <div class="live" id="live" hidden></div>
    <div class="mic-area">
      <span class="bubble mic-hint" id="hint">버튼을 눌러 음성인식을<br>시작해보세요!</span>
      <button class="mic" id="mic" aria-label="녹음">${icon("mic")}</button><button class="mic-cancel" id="cancel" hidden aria-label="녹음 취소">${icon("close")}<span>취소</span></button>
      <div class="mic-caption" id="cap"></div>
      <button class="upload-link" id="upload">녹음 파일로 올리기</button>
      <input type="file" id="file" accept="audio/*,video/*" hidden>
    </div></div>`;
  const ui = micUI(view);
  let state = "idle", rec = null, timer = null, alive = true;
  ui.set("idle");
  const card = $("#qcard", view);
  const say = () => voice.speak(`${persona.hello ? persona.hello + " " : ""}${P.question}`, persona.name, {
    onstart: () => card.classList.add("speaking"),
    onend: () => card.classList.remove("speaking"),
  });
  const speakBtn = $("#speak", view);
  if (speakBtn) speakBtn.onclick = () => {
    if (!voice.enabled()) voice.setEnabled(true);
    speakBtn.innerHTML = icon("speaker");
    if (state === "idle") say();
  };
  setTimeout(() => { if (alive && state === "idle" && !P.segments.length) say(); }, 450);

  const renderLive = () => {
    if (!alive) return;
    const live = $("#live", view);
    const segs = P.segments.filter((s) => s.status !== "empty");
    live.hidden = !segs.length;
    const done = segs.filter((s) => s.status === "done");
    const zone = zoneOf(done);
    live.innerHTML = segs.map((s) => s.status === "done" ? `<p>${marked(s.analysis)}</p>`
      : s.status === "error" ? `<p class="wait">(이 부분은 인식하지 못했어요)</p>`
      : `<p class="wait"><span class="pending-dots">● ● ●</span>${s.note ? ` <small>${esc(s.note)}</small>` : ""}</p>`).join("")
      + `<div class="live-meta">${done[0] && done[0].demo ? "체험 모드 · 엔진이 꺼져 있어 예시 문장을 보여 줘요" : `Gemma 4 · ${esc((done[0] && done[0].model) || "서투리 LoRA")}`} · ${done.length}/${segs.length} 문장${zone ? ` · 억양 ${esc(zone)}` : ""}</div>`;
    live.scrollTop = live.scrollHeight;
    if (state === "finishing") finish();
  };

  const onSegment = (seg) => {
    const item = { seq: P.segments.length, status: "pending", duration: seg.end - seg.start, start: seg.start };
    P.segments.push(item);
    P.fb = null;
    renderLive();
    asrSubmit(item, encodeWav(seg.samples), renderLive);
  };

  const finish = () => {
    const segs = P.segments.filter((s) => s.status !== "empty");
    const pending = segs.filter((s) => s.status === "pending").length;
    if (pending) { ui.set("busy", `답변 피드백 생성중 … (${segs.length - pending}/${segs.length})`); return; }
    if (!segs.some((s) => s.status === "done")) {
      state = "idle";
      P.segments = [];
      ui.set("idle");
      $("#live", view).hidden = true;
      toast("말소리를 듣지 못했어요. 마이크 가까이에서 다시 말해 볼까요?");
      return;
    }
    state = "done";
    ui.set("done", "피드백 생성 완료!");
    setTimeout(() => { if (alive) location.hash = "#/coach/feedback"; }, 900);
  };

  const start = async () => {
    voice.stop(); // the question must not end up in the recording
    if (!supported() || !window.isSecureContext) {
      toast("이 주소에서는 마이크를 쓸 수 없어요 (HTTPS나 localhost 필요). 녹음 파일로 올려 주세요.", 4500);
      return;
    }
    rec = new Recorder({ onSegment, onLevel: (v) => ui.level(v) });
    try {
      await rec.start();
    } catch (e) {
      toast("마이크 권한이 필요해요. 브라우저 주소창에서 마이크를 허용해 주세요.", 4000);
      rec = null;
      return;
    }
    state = "rec";
    ui.set("rec", "듣고 있어요 …");
    timer = setTimeout(() => { if (state === "rec") stop(); toast("최대 5분까지 녹음할 수 있어요"); }, 5 * 60 * 1000);
  };
  const stop = async () => {
    clearTimeout(timer);
    state = "finishing";
    ui.set("busy", "답변 피드백 생성중 …");
    await rec.stop();
    finish();
  };
  // Cancel: drop everything said in this attempt, including clips still being transcribed (their results
  // land on items no longer in P.segments), and go back to the start.
  const cancel = async () => {
    if (state !== "rec") return;
    clearTimeout(timer);
    state = "idle";
    const r = rec;
    rec = null;
    await r.cancel();
    P.segments = [];
    $("#live", view).hidden = true;
    ui.set("idle");
    toast("녹음을 취소했어요. 다시 누르면 처음부터 녹음해요.");
  };
  $("#mic", view).onclick = () => { if (state === "idle") start(); else if (state === "rec") stop(); };
  $("#cancel", view).onclick = cancel;
  $("#upload", view).onclick = () => { if (state === "idle") $("#file", view).click(); };
  $("#file", view).onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    state = "finishing";
    ui.set("busy", "파일을 분석하고 있어요 …");
    const item = { seq: P.segments.length, status: "pending" };
    P.segments.push(item);
    renderLive();
    await asrSubmit(item, f, () => {});
    if (item.status === "done" && item.parts && item.parts.length > 1) {
      P.segments.splice(P.segments.indexOf(item), 1, ...item.parts.map((p) => ({ ...p, model: item.model, status: "done" })));
    }
    renderLive();
  };
  const sb = $("#scriptbtn", view);
  if (sb) sb.onclick = () => modal(`<h3 style="margin:0 0 10px">내 대본</h3><div style="white-space:pre-wrap;line-height:1.7;font-size:15px">${esc(P.script)}</div>`);
  $("#quit", view).onclick = () => {
    if ((state === "rec" || P.segments.length) && !confirm("연습을 그만둘까요? 지금까지 말한 내용은 저장되지 않아요.")) return;
    location.hash = "#/coach";
  };
  return () => {
    alive = false;
    clearTimeout(timer);
    voice.stop();
    if (rec && state === "rec") rec.stop();
  };
}

function chatSession(view) {
  const persona = personaOf(P);
  view.innerHTML = `<div class="session">
    <div class="room" data-speaker style="background-image:url(${sceneOf(P)})"></div><div class="room-shade"></div>
    <button class="close" id="quit" aria-label="나가기">${icon("close")}</button>
    <div class="date-pill" style="max-width:62%;overflow:hidden;text-overflow:ellipsis">${ymd(P.created)} · ${esc(title(P))}</div>
    <button class="end-chat" id="end" disabled>대화 끝내기</button>
    <div class="chat" id="chat"></div>
    <div class="mic-area">
      <span class="bubble mic-hint" id="hint">버튼을 눌러<br>대답해 보세요!</span>
      <button class="mic" id="mic" aria-label="녹음">${icon("mic")}</button><button class="mic-cancel" id="cancel" hidden aria-label="녹음 취소">${icon("close")}<span>취소</span></button>
      <div class="mic-caption" id="cap"></div>
    </div></div>`;
  const ui = micUI(view);
  let state = "idle", rec = null, alive = true, turn = [];
  ui.set("idle");

  const draw = (typing = false) => {
    if (!alive) return;
    const chat = $("#chat", view);
    chat.innerHTML = P.history.map((h, i) => h.role === "ai" ? `<div class="msg ai">${esc(h.text)}${voice.supported() ? `<button class="say" data-say="${i}" aria-label="다시 듣기">${icon("speaker")}</button>` : ""}</div>`
      : `<div class="msg me">${h.segs.map((s) => marked(s.analysis)).join(" ")}${h.segs.some((s) => s.analysis.fixes.length) ? `<span class="std">표준어 · ${esc(h.text)}</span>` : ""}</div>`).join("")
      + (typing ? '<div class="msg ai typing"><span class="pending-dots">● ● ●</span></div>' : "")
      + (turn.length ? `<div class="msg me" style="opacity:.75">${turn.map((s) => s.status === "done" ? marked(s.analysis) : '<span class="pending-dots">● ● ●</span>').join(" ")}</div>` : "");
    chat.scrollTop = chat.scrollHeight;
    $("#end", view).disabled = !P.history.some((h) => h.role === "user") || state !== "idle";
    for (const b of $$("[data-say]", chat)) b.onclick = () => {
      if (!voice.enabled()) voice.setEnabled(true);
      if (state === "idle") voice.speak(P.history[+b.dataset.say].text, persona.name);
    };
  };

  const onSegment = (seg) => {
    const item = { seq: P.segments.length, status: "pending", duration: seg.end - seg.start };
    turn.push(item);
    draw();
    asrSubmit(item, encodeWav(seg.samples), () => { draw(); if (state === "finishing") finishTurn(); });
  };

  const finishTurn = async () => {
    if (turn.some((s) => s.status === "pending")) { ui.set("busy", "듣고 이해하는 중 …"); return; }
    const segs = turn.filter((s) => s.status === "done");
    turn = [];
    if (!segs.length) { state = "idle"; ui.set("idle"); draw(); toast("말소리를 듣지 못했어요. 다시 말해 볼까요?"); return; }
    state = "reply";
    P.segments.push(...segs);
    P.history.push({ role: "user", text: segs.map((s) => s.standard).join(" "), segs });
    ui.set("busy", `${subject(persona.short)} 대답하는 중 …`);
    draw(true);
    try {
      const r = await api.post("/api/coach/reply", {
        persona: { name: persona.name, desc: persona.act || persona.desc },
        scenario: `${P.category} (상대 역할: ${persona.role || SITUATIONS[P.category].role})`,
        history: P.history.map((h) => ({ role: h.role, text: h.text })),
      });
      P.history.push({ role: "ai", text: r.text });
      if (alive) voice.speak(r.text, persona.name);
    } catch (e) {
      toast(e.message);
    }
    if (!alive) return;
    state = "idle";
    ui.set("idle", P.history.filter((h) => h.role === "user").length >= 10 ? "발화 10회 달성! 보너스 경험치를 받을 수 있어요" : "");
    draw();
  };

  $("#mic", view).onclick = async () => {
    if (state === "idle") {
      voice.stop();
      if (!supported() || !window.isSecureContext) { toast("이 주소에서는 마이크를 쓸 수 없어요 (HTTPS나 localhost 필요)", 4000); return; }
      rec = new Recorder({ onSegment, onLevel: (v) => ui.level(v) });
      try { await rec.start(); } catch { toast("마이크 권한이 필요해요."); rec = null; return; }
      state = "rec";
      ui.set("rec", "듣고 있어요 … 다 말하면 버튼을 눌러요");
      draw();
    } else if (state === "rec") {
      state = "finishing";
      ui.set("busy", "듣고 이해하는 중 …");
      await rec.stop();
      finishTurn();
    }
  };
  // Cancel this turn: its clips, including ones still being transcribed, are dropped.
  $("#cancel", view).onclick = async () => {
    if (state !== "rec") return;
    state = "idle";
    const r = rec;
    rec = null;
    await r.cancel();
    turn = [];
    ui.set("idle");
    draw();
    toast("녹음을 취소했어요.");
  };
  $("#end", view).onclick = () => { location.hash = "#/coach/feedback"; };
  $("#quit", view).onclick = () => {
    if (P.history.length > 1 && !confirm("대화를 그만둘까요? 기록은 저장되지 않아요.")) return;
    location.hash = "#/coach";
  };
  draw();
  setTimeout(() => { if (alive && P.history.length === 1) voice.speak(P.history[0].text, persona.name); }, 450);
  return () => { alive = false; voice.stop(); if (rec && state === "rec") rec.stop(); };
}

// ---------------------------------------------------------------------------------- feedback
function context(p) {
  const persona = personaOf(p) || {};
  return { title: title(p), persona: persona.name, persona_desc: persona.desc, question: p.mode === "interview" ? p.question : "",
    script: p.mode === "presentation" ? p.script : "", focus: p.focus, topic: p.topic, category: p.category };
}

// Fluency measured from the recording (server/analysis.py fluency_summary), with the native range.
function fluencyNote(f) {
  const rate = f.rate < 4 ? "조금 느려요" : f.rate > 5.5 ? "조금 빨라요" : "적당해요";
  const fill = f.fillers_per_min > 4 ? ` · 군말 ${f.fillers}번 ("그~", "어~" 등)` : f.fillers ? ` · 군말 ${f.fillers}번` : "";
  const gaps = f.long_gaps ? ` · 3초 넘는 침묵 ${f.long_gaps}번` : "";
  return `<div class="note">${icon("wave").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>유창성 ${f.score}점</b> (녹음에서 측정)<br>`
    + `말 속도 초당 ${f.rate}음절, ${rate} (표준어 화자 보통 4.0~5.5) · 쉼 없이 평균 ${f.mlr}음절${fill}${gaps}</span></div>`;
}

// 대본 일치도 sentence by sentence (server/analysis.py script_coverage): the script sentences not said.
function scriptNote(match, cov) {
  const missed = (cov && cov.missed) || [];
  const list = missed.slice(0, 5).map((m) => `<br><span class="muted small">· ${esc(m.length > 60 ? m.slice(0, 60) + "…" : m)}</span>`).join("");
  return `<div class="note">${icon("doc").replace("<svg", '<svg style="color:#f08300"')}<span><b>대본 일치도 ${match}%</b>${cov ? ` · 대본 ${cov.sentences.length}문장 중 ${cov.sentences.length - missed.length}문장을 말했어요` : ""}`
    + (missed.length ? `<br>빠뜨린 부분${missed.length > 5 ? ` (${missed.length}문장 중 5개)` : ""}${list}` : "") + `</span></div>`;
}

// This answer against the mean of the user's last sessions of the mode (server/app.py progress; MACH,
// Hoque et al. 2013: measured feedback compared over sessions).
const PROGRESS = { points: ["점수", "점"], dialect_ratio: ["사투리 비율", "%"], accent_strength: ["억양 강도", "%"],
  fluency: ["유창성", "점"], fillers_per_min: ["군말", "회/분"], script_match: ["대본 일치도", "%"] };
function progressNote(fb) {
  const pr = fb.progress, sp = fb.speech;
  if (!pr || !sp) return "";
  const fl = sp.fluency || {};
  const now = { points: fb.score && fb.score.points, dialect_ratio: sp.dialect_ratio, accent_strength: sp.accent_strength,
    fluency: fl.score, fillers_per_min: fl.fillers_per_min, script_match: fb.scriptMatch };
  const rows = Object.keys(PROGRESS).filter((k) => now[k] != null && pr.mean[k] != null).map((k) => {
    const d = now[k] - pr.mean[k];
    const better = Math.abs(d) < 0.05 ? null : pr.lower_is_better.includes(k) ? d < 0 : d > 0;
    const color = better === null ? "#8b95a7" : better ? "#0a8f4e" : "#c2410c";
    return `${PROGRESS[k][0]} ${pr.mean[k]} → <b style="color:${color}">${now[k]}${PROGRESS[k][1]}</b>`;
  });
  return rows.length ? `<div class="note">${icon("refresh").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>지난 ${pr.sessions}번 평균과 비교</b><br>${rows.join(" · ")}</span></div>` : "";
}

// Sounds a regional accent merges, measured in the recording (server/pron.py).
function pronNote(pr) {
  const rows = Object.values(pr).filter((v) => v.flagged);
  if (!rows.length) return "";
  return `<div class="note">${icon("speaker").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>발음</b> (녹음에서 측정)<br>`
    + rows.map((v) => `'${esc(v.name)}' 소리 ${v.n}번 중 ${v.flagged}번이 '${esc(v.merged)}'처럼 들렸어요${v.words.length ? ` · ${v.words.map(esc).join(", ")}` : ""}`).join("<br>")
    + `</span></div>`;
}

// The answer's sub-scores: word choice and facts by rubric (expected score over Gemma 4's probabilities of
// 1-5, server/analysis.py RUBRICS), fluency measured from the recording, delivery from Gemma 4 hearing it.
function scoresNote(fb) {
  const sc = fb.feedback && fb.feedback.scores;
  if (!sc || sc.word == null) return "";
  const fl = fb.speech && fb.speech.fluency;
  const parts = [`단어 선택 ${sc.word}`, sc.facts != null ? `정보·사실 ${sc.facts}` : "",
    fl ? `유창성 ${fl.score} (녹음 측정)` : sc.fluency != null ? `유창성 ${sc.fluency}` : "",
    sc.delivery != null ? `전달력 ${sc.delivery} (녹음을 듣고 평가, 점수 미반영)` : ""].filter(Boolean);
  return `<div class="note">${icon("star").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>답변 채점</b>${fb.feedback.rubric ? " (기준표 채점)" : ""}<br>${parts.join(" · ")}</span></div>`;
}

function slim(s) {
  return { dialect: s.dialect, standard: s.standard, region: s.region, zone_probs: s.zone_probs, analysis: s.analysis, duration: s.duration,
    fluency: s.fluency, start: s.start, pron: s.pron, clip: s.clip };
}

async function runFeedback(p) {
  const fb = p.fb = { state: "streaming", text: "", feedback: null, score: null, speech: null, error: null };
  const onUpdate = () => p.onFeedback && p.onFeedback();
  const segments = p.segments.filter((s) => s.status === "done").map(slim);
  try {
    for await (const ev of stream("/api/coach/feedback", { uid: uid(), mode: p.mode, context: context(p), segments })) {
      if (ev.speech) fb.speech = ev.speech;
      if (ev.delta) fb.text += ev.delta;
      if (ev.done) Object.assign(fb, { feedback: ev.feedback, score: ev.score, timeTip: ev.time_tip, scriptMatch: ev.script_match, scriptCoverage: ev.script_coverage, progress: ev.progress, error: ev.error, seconds: ev.seconds, demo: ev.demo });
      onUpdate();
    }
  } catch (e) {
    fb.error = e.message;
  }
  fb.state = "saving";
  onUpdate();
  try {
    const saved = await api.post("/api/sessions", { uid: uid(), mode: p.mode, context: context(p), segments, feedback: fb.feedback || {},
      history: p.mode === "daily" ? p.history.map((h) => ({ role: h.role, text: h.text })) : null });
    fb.saved = saved;
    fb.score = saved.score || fb.score;
    setMe(saved.user);
  } catch (e) {
    fb.error = fb.error || e.message;
  }
  fb.state = "done";
  onUpdate();
}

function bindSay(root, personaName) {
  for (const b of $$("[data-say-text]", root)) b.onclick = () => {
    if (!voice.enabled()) voice.setEnabled(true);
    voice.speak(b.dataset.sayText, personaName);
  };
}

function feedbackBody(d, fb) {
  const segs = d.segments;
  const speech = fb.speech;
  const tips = (fb.feedback && fb.feedback.tips) || [];
  const byN = {};
  for (const t of tips) (byN[t.n] ||= []).push(t);
  let k = 0;
  const items = segs.map((s, i) => {
    const mine = byN[i + 1] || [];
    const fixes = s.analysis.fixes.filter((f) => f.from);
    const accent = speech && speech.accent && speech.accent[i];  // counted in the 사투리 비율 for its accent
    const needs = fixes.length || mine.length || accent;
    const head = needs ? `개선필요 ${++k}` : `문장 ${i + 1} · 좋아요`;
    const quotes = mine.map((t) => t.quote);
    return `<div class="card fb-item fade-in"><h4>${head}</h4>
      <div class="sent">${marked(s.analysis, quotes)}</div>
      ${fixes.length || mine.length ? `<div class="std">표준어 · ${standardMarked(s.analysis, s.standard, quotes)}</div>` : ""}
      ${mine.map((t) => `<div class="tip c">${icon("doc")}<span>${esc(t.tip)}</span></div>`).join("")}
      ${fixes.map((f) => `<div class="tip d">${icon("person")}<span>'${esc(f.from)}' 대신 '${esc(f.to)}'(이)라고 말하면 표준어예요.</span></div>`).join("")}
      ${accent ? `<div class="tip d">${icon("pin")}<span>${esc(s.region && s.region !== "서울·경기" ? s.region : speech.zone || "지역")} 억양으로 들려서 이 문장은 억양 강도${speech.accent_share && speech.accent_share[i] != null ? ` ${speech.accent_share[i]}%` : ""}만큼 사투리 비율에 반영됐어요.</span></div>` : ""}
    </div>`;
  }).join("");
  const stats = speech ? `<div class="stat-row">
      <div class="stat"><b>${speech.dialect_ratio}%</b><span>사투리 비율</span></div>
      <div class="stat"><b>${speech.words.reduce((a, w) => a + w.count, 0)}</b><span>교정한 표현</span></div>
      <div class="stat"><b>${Math.round(segs.reduce((a, s) => a + (s.duration || 0), 0))}초</b><span>말한 시간</span></div></div>` : "";
  const demo = fb.demo || (fb.feedback && fb.feedback.demo) || segs.some((s) => s.demo);
  const notes = [
    demo ? `<div class="note" style="color:#8a5a00">${icon("info").replace("<svg", '<svg style="color:#d08700"')}<span><b>체험 모드</b><br>Gemma 4 엔진이 꺼져 있어서 받아쓰기는 예시 문장이고, 총평·조언도 예시예요. 사투리 비율과 점수는 예시 문장으로 계산했어요.</span></div>` : "",
    speech && speech.zone ? `<div class="note" style="color:#1d4ed8">${icon("pin").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>억양 · ${speech.zone === "서울·경기" && speech.zone_probs ? "표준어(서울·경기)에 가까워요" : esc(speech.zone)}</b> (${speech.accent_any_zone && speech.zone !== "서울·경기" ? "지역 억양 중 " : ""}${speech.zone_share}%)${speech.zone_probs && speech.zone !== "서울·경기" && !speech.accent_any_zone ? ` · 표준어 억양 ${speech.standard_share}%` : ""}<br>${esc(speech.zone_tip || "표준어에 가까운 억양이에요. 지금처럼 말하면 돼요!")}${speech.word_ratio !== undefined ? `<br><span class="muted small">단어 ${speech.word_ratio}%${speech.accent_strength != null ? ` · 억양 강도 ${speech.accent_strength}% (표준어 억양이 아닐 확률)` : ""}</span>` : ""}${speech.accent_words ? `<br><span class="muted small">사투리 비율 = (사투리 단어 ${speech.changed}어절 + 억양이 표준어가 아닌 문장의 나머지 어절 × 억양 강도 = ${speech.accent_words}어절) ÷ 전체 ${speech.total}어절</span>` : ""}</span></div>`
      : speech && speech.zone_unsure && speech.accent_words ? `<div class="note" style="color:#1d4ed8">${icon("pin").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>억양 · 지역 억양으로 들려요</b> (억양 강도 ${speech.accent_strength}%)<br>어느 지역 억양인지는 이번 답변만으로는 확실하지 않아요. 조금 더 길게 말하면 판별해 드려요.<br><span class="muted small">사투리 비율 = (사투리 단어 ${speech.changed}어절 + 억양이 표준어가 아닌 문장의 나머지 어절 × 억양 강도 = ${speech.accent_words}어절) ÷ 전체 ${speech.total}어절</span></span></div>`
      : speech && speech.zone_unsure ? `<div class="note">${icon("pin").replace("<svg", '<svg style="color:#8b95a7"')}<span>억양은 이번 답변만으로는 확실하지 않아요. 조금 더 길게 말하면 판별해 드려요.</span></div>`
      : speech && speech.zone_pending ? `<div class="note">${icon("pin").replace("<svg", '<svg style="color:#8b95a7"')}<span>억양은 8초 이상 말하면 판별해 드려요.</span></div>` : "",
    speech && speech.fluency ? fluencyNote(speech.fluency) : "",
    speech && speech.pron ? pronNote(speech.pron) : "",
    fb.state === "done" ? scoresNote(fb) : "",
    fb.state === "done" ? progressNote(fb) : "",
    fb.timeTip ? `<div class="note">${icon("info").replace("<svg", '<svg style="color:#8b95a7"')}<span>${esc(fb.timeTip)}</span></div>` : "",
    fb.feedback && fb.feedback.summary ? `<div class="note">${icon("sparkle").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>AI 코치 총평</b><br>${esc(fb.feedback.summary)}</span></div>` : "",
    fb.feedback && fb.feedback.question ? `<div class="note">${icon("comment").replace("<svg", '<svg style="color:#0a6bff"')}<span><b>청중 질문</b><br>${esc(fb.feedback.question)}</span>${voice.supported() ? `<button class="say" data-say-text="${esc(fb.feedback.question)}" aria-label="질문 듣기">${icon("speaker")}</button>` : ""}</div>` : "",
    fb.scriptMatch !== undefined && fb.scriptMatch !== null ? scriptNote(fb.scriptMatch, fb.scriptCoverage) : "",
  ].join("");
  const writing = fb.state === "streaming" ? `<div class="card writing"><div class="row"><span class="spinner blue"></span>AI 코치(Gemma 4)가 답변 내용을 살펴보고 있어요</div><pre>${esc(fb.text.slice(-400)) || "학습 중인 GPU를 함께 쓰고 있어서 1분 정도 걸릴 수 있어요…"}</pre></div>` : "";
  const err = fb.error ? `<div class="note" style="color:#a32323">${icon("info")}<span>${esc(fb.error)}</span></div>` : "";
  return `<div class="card fb-summary"><h3>사투리 교정 요약</h3>${stats}${notes}${err}</div>${writing}${items}`;
}

export async function feedback(view) {
  if (!P || !P.segments.some((s) => s.status === "done")) { location.replace("#/coach"); return; }
  const p = P;
  const label = { interview: "면접 결과보기", presentation: "발표 결과보기", daily: "대화 결과보기" }[p.mode];
  view.innerHTML = `<div style="min-height:100%;display:flex;flex-direction:column">
    <div class="fb-head"><div class="fb-select">${ymd(p.created)} · ${esc(title(p))} ${p.mode === "daily" ? "대화" : "답변"} 피드백</div></div>
    <div class="fb-list" id="list" style="flex:1"></div>
    <div class="bottom-cta"><button class="btn block" id="result" disabled>${label}</button></div></div>`;
  let alive = true;
  const segs = p.segments.filter((s) => s.status === "done");
  const draw = () => {
    if (!alive) return;
    const keep = view.scrollTop;
    $("#list", view).innerHTML = feedbackBody({ segments: segs }, p.fb);
    bindSay($("#list", view), (personaOf(p) || {}).name);
    view.scrollTop = keep;
    const btn = $("#result", view);
    btn.disabled = p.fb.state !== "done";
    btn.innerHTML = p.fb.state === "done" ? label : `<span class="spinner"></span> 피드백 작성 중`;
  };
  p.onFeedback = draw;
  if (!p.fb) runFeedback(p);
  draw();
  $("#result", view).onclick = () => showResult(view, p, title(p));
  return () => { alive = false; voice.stop(); if (p.onFeedback === draw) p.onFeedback = null; };
}

// ---------------------------------------------------------------------------------- result
async function showResult(view, p, heading) {
  const fb = p.fb;
  const calc = document.createElement("div");
  calc.className = "calc";
  calc.innerHTML = `<div class="ring"><img src="/assets/calculator.jpg" alt=""></div><div><span style="color:#0a6bff">${p.mode === "daily" ? "대화 기록" : "면접 기준"}</span>${p.mode === "daily" ? "을 정리하고 있어요!" : "으로<br>점수를 계산 중이에요!"}</div>`;
  $("#shell").append(calc);
  await sleep(1500);
  calc.remove();
  const s = fb.score;
  const xp = fb.saved ? fb.saved.xp : 0;
  const daily = p.mode === "daily" || !s;
  const passed = s && s.passed;
  const what = { interview: "면접", presentation: "발표", daily: "대화" }[p.mode];
  const { el, close } = modal(`<div class="result">
    <div class="head"><h3>${what} 결과</h3><button id="rx" aria-label="닫기">${icon("close").replace("<svg", '<svg style="width:20px;height:20px"')}</button></div>
    ${daily ? `<div class="stamp">완료</div>` : `<div class="stamp ${passed ? "" : "fail"}">${passed ? "합격" : "불합격"}</div>`}
    <div class="result-art"><div class="scorecard">
      ${daily ? `<b>발화 <em>${p.history.filter((h) => h.role === "user").length}회</em></b><span>사투리 비율 ${fb.speech ? fb.speech.dialect_ratio : 0}%<br>오늘도 표준어 한 걸음!</span>`
        : `<b>${what} 점수: <em>${s.points}점</em></b><span>${passed ? "축하드려요 🎉<br>이대로 가면 합격이에요!" : "조금만 더 연습하면<br>합격할 수 있어요!"}</span>`}
    </div></div>
    ${daily ? "" : `<div class="breakdown"><div class="stat"><b>${s.standard}</b><span>표준어 점수</span></div><div class="stat"><b>${s.answer ?? "-"}</b><span>답변 점수</span></div></div>
      <div class="muted small">표준어 점수 + 답변 점수 = ${s.total}/200 · 합격 기준 140</div>`}
    <div class="xpgain">+${xp} XP${fb.saved && fb.saved.tier_up ? ` · 티어 업! ${esc(fb.saved.user.tier.name)}` : ""}</div>
    <div class="actions"><button class="btn gray" id="rfb">피드백 보기</button><a class="btn" href="#/home">홈으로</a><button class="btn ghost" id="rdl" aria-label="결과 저장" style="padding:0">${icon("download").replace("<svg", '<svg style="width:22px;height:22px"')}</button></div>
  </div>`);
  if (daily || passed) setTimeout(() => confetti($("#overlay")), 250);
  $("#rx", el).onclick = close;
  $("#rfb", el).onclick = close;
  $("#rdl", el).onclick = () => downloadCard(p, heading);
}

function downloadCard(p, heading) {
  const fb = p.fb, s = fb.score;
  const c = document.createElement("canvas");
  c.width = 720; c.height = 900;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 900);
  grad.addColorStop(0, "#0a6bff"); grad.addColorStop(1, "#7db8ff");
  g.fillStyle = grad; g.fillRect(0, 0, 720, 900);
  g.fillStyle = "#fff";
  g.beginPath(); g.roundRect(50, 150, 620, 680, 36); g.fill();
  g.font = "800 54px Pretendard, sans-serif"; g.fillText("서투리", 60, 100);
  g.fillStyle = "#111a2e"; g.font = "700 30px Pretendard, sans-serif"; g.fillText(`${ymd(p.created)} · ${heading}`, 90, 230);
  g.font = "900 120px Pretendard, sans-serif"; g.fillStyle = "#0a6bff";
  g.fillText(s ? `${s.points}점` : "완료", 90, 390);
  g.font = "700 34px Pretendard, sans-serif"; g.fillStyle = s && !s.passed ? "#6b778a" : "#0a6bff";
  g.fillText(s ? (s.passed ? "합격" : "불합격") : "일상회화", 90, 450);
  g.fillStyle = "#4b5567"; g.font = "500 28px Pretendard, sans-serif";
  const lines = [
    s ? `표준어 점수 ${s.standard} · 답변 점수 ${s.answer ?? "-"}` : "",
    fb.speech ? `사투리 비율 ${fb.speech.dialect_ratio}% · 교정 표현 ${fb.speech.words.length}개` : "",
    fb.speech && fb.speech.zone ? `억양: ${fb.speech.zone}` : "",
    ...(fb.speech ? fb.speech.words.slice(0, 4).map((w) => `'${w.from}' → '${w.to}'`) : []),
  ].filter(Boolean);
  lines.forEach((t, i) => g.fillText(t, 90, 530 + i * 46));
  g.fillStyle = "#8b95a7"; g.font = "500 22px Pretendard, sans-serif"; g.fillText("Gemma 4 · 서투리 AI 코치", 90, 790);
  const a = document.createElement("a");
  a.download = `서투리_${ymd(p.created)}.png`;
  a.href = c.toDataURL("image/png");
  a.click();
}

// ---------------------------------------------------------------------------------- history
export async function history(view, sid) {
  const nav = window.history;
  const r = await api.get(`/api/sessions/${sid}`, { uid: uid() });
  const d = r.data;
  const p = { mode: r.mode, created: r.created, history: d.history || [], topic: d.context.topic, category: d.context.category,
    segments: d.segments.map((s) => ({ ...s, status: "done" })),
    fb: { state: "done", speech: d.speech, feedback: d.feedback, score: d.score, scriptMatch: d.script_match, scriptCoverage: d.script_coverage, progress: r.progress, saved: { xp: r.xp } } };
  view.innerHTML = `<div style="min-height:100%;display:flex;flex-direction:column">
    <div class="topbar"><button class="back" id="back">${icon("back")}</button><h1>코칭 기록</h1><span></span></div>
    <div class="fb-head" style="position:static;padding-top:0"><div class="fb-select">${ymd(r.created)} · ${esc(r.title)}</div></div>
    <div class="fb-list" style="flex:1">${feedbackBody({ segments: p.segments }, p.fb)}</div>
    <div class="bottom-cta"><button class="btn block" id="result">결과 다시 보기</button></div></div>`;
  bindSay(view, d.context.persona);
  $("#back", view).onclick = () => (nav.length > 1 ? nav.back() : (location.hash = "#/coach"));
  $("#result", view).onclick = () => showResult(view, p, r.title);
}

export { feedbackBody, MODES };
