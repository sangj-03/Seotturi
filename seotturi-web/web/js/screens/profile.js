// Profile edit sheet and the bell's notifications (model updates of the Gemma 4 engine).
import { api, me, setMe, status, store, uid } from "../api.js";
import { $, esc, icon, sheet, toast, ymd, hm, untilText } from "../ui.js";
import * as voice from "../voice.js";

export const REGIONS = [
  ["", "선택 안 함"], ["서울", "서울·경기·인천"], ["강원", "강원도"], ["충청", "충청도 (대전·세종)"], ["전라", "전라도 (광주)"],
  ["대구", "대구·경북"], ["부산", "부산·울산·경남"], ["제주", "제주도"],
];

export function modelBox(s) {
  if (!s || !s.ready) {
    return `<div class="model-box"><div class="k">서투리 AI</div><div class="v">${s && s.error === "offline" ? "엔진에 연결하지 못했어요" : "Gemma 4 모델을 불러오는 중이에요…"}</div></div>`;
  }
  const m = s.model || {};
  const t = s.training;
  const pct = t ? Math.min(100, (100 * t.step) / t.total) : 0;
  const sc = m.score;
  return `<div class="model-box">
    <div class="k">지금 쓰는 AI</div>
    <div class="v">${esc(s.base)} · 서투리 LoRA <span style="opacity:.8">${esc(m.id || "")}</span></div>
    <div class="k">표준어 변환 문자 오류율 (CER) · 시험 문장 160개</div>
    <div class="v">${sc ? (sc.cer_standard * 100).toFixed(1) + "%" : "측정 중…"}${sc && sc.zone_macro_f1 !== null && sc.zone_macro_f1 !== undefined ? ` · 억양 지역 F1 ${(sc.zone_macro_f1 * 100).toFixed(0)}%` : ""}</div>
    ${t ? `<div class="k">학습 진행 (${esc(t.run)})</div><div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div>
    <div class="k">${t.step.toLocaleString()} / ${t.total.toLocaleString()} 스텝 · 약 ${Math.round(t.eta_hours)}시간 남음</div>` : ""}
    <div class="k" style="margin-top:8px">${s.update_hours}시간마다 새 체크포인트를 검증해 자동 업데이트 · 다음 확인 ${s.next_check ? hm(s.next_check) + " (" + untilText(s.next_check) + ")" : "-"}</div>
  </div>`;
}

export async function openProfile(onSaved) {
  const s = await status();
  const { el, close } = sheet(`<h2>프로필 수정</h2>
    <label class="field"><span>닉네임</span><input class="input" id="pf-nick" maxlength="12" value="${esc(me.nickname)}"></label>
    <label class="field"><span>내 출신 지역 <small class="muted">· 단어 도감 지역에 쓰여요</small></span>
      <select class="select" id="pf-region">${REGIONS.map(([v, l]) => `<option value="${v}" ${v === (me.region || "") ? "selected" : ""}>${l}</option>`).join("")}</select></label>
    <div class="toggle-row"><div><b>내 음성으로 서투리 AI 키우기</b><span>켜면 코칭 음성과 인식 결과가 익명으로 저장되어 다음 학습 데이터 후보가 돼요. 끄면 음성은 분석 직후 버려져요.</span></div>
      <button class="switch ${me.contribute ? "on" : ""}" id="pf-contrib" aria-label="음성 기여"></button></div>
    ${voice.supported() ? `<div class="toggle-row"><div><b>AI 질문을 음성으로 듣기</b><span>면접관·청중·대화 상대가 질문과 대답을 소리 내어 읽어 줘요. 서버의 AI 음성(Supertonic 3)으로 인물마다 다른 목소리를 들려줘요.</span></div>
      <button class="switch ${voice.enabled() ? "on" : ""}" id="pf-voice" aria-label="AI 음성"></button></div>` : ""}
    ${modelBox(s)}
    <button class="btn block" id="pf-save">저장</button>
    <button class="btn block gray" id="pf-reset" style="margin-top:8px;height:44px;font-size:14px">이 기기 기록 초기화 (새 사용자로 시작)</button>`);
  const sw = $("#pf-contrib", el);
  sw.onclick = () => sw.classList.toggle("on");
  const vs = $("#pf-voice", el);
  if (vs) vs.onclick = () => {
    vs.classList.toggle("on");
    voice.setEnabled(vs.classList.contains("on"));
    if (vs.classList.contains("on")) { voice.unlock(); voice.speak("안녕하세요, 서투리 AI 코치예요.", "따뜻한 멘토형"); }
  };
  $("#pf-save", el).onclick = async () => {
    try {
      setMe(await api.patch("/api/me", { uid: uid(), nickname: $("#pf-nick", el).value, region: $("#pf-region", el).value, contribute: sw.classList.contains("on") }));
      toast("프로필을 저장했어요");
      close();
      onSaved && onSaved();
    } catch (e) {
      toast(e.message);
    }
  };
  $("#pf-reset", el).onclick = () => {
    if (!confirm("이 기기에서 쓰던 기록 대신 새 사용자로 시작할까요? (서버의 기존 기록은 지워지지 않아요)")) return;
    store.set("seotturi.uid", "");
    location.reload();
  };
}

export function unreadCount(s) {
  const seen = store.get("seotturi.notiSeen", 0);
  return (s && s.history ? s.history.filter((h) => h.time > seen).length : 0);
}

export async function openNotifications() {
  const s = await status(true);
  const items = (s.history || []).slice().reverse();
  const rows = items.map((h) => {
    const promoted = h.result === "promoted";
    const c = h.candidate_score, p = h.served_score;
    return `<div class="noti"><div class="ic">${icon(promoted ? "sparkle" : "info")}</div><div>
      <b>${promoted ? `서투리 AI가 ${esc(h.candidate)}(으)로 업데이트됐어요` : `${esc(h.candidate)} 체크포인트는 보류했어요`}</b>
      <span>${ymd(h.time)} ${hm(h.time)} · 검증 오류율 ${(p.cer_standard * 100).toFixed(1)}% → ${(c.cer_standard * 100).toFixed(1)}%${promoted ? "" : " (기존 모델이 더 정확해요)"}</span></div></div>`;
  }).join("");
  sheet(`<h2>알림</h2>${modelBox(s)}
    ${rows || `<div class="empty">아직 모델 업데이트 기록이 없어요.<br>학습 중인 Gemma 4 체크포인트를 ${s.update_hours || 5}시간마다 검증해서<br>더 정확하면 자동으로 바꿔 끼워요.</div>`}`);
  store.set("seotturi.notiSeen", Date.now() / 1000);
}
