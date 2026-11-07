// /record: the accent test set. Same recorder as the practice screens (recorder.js, browser noise
// suppression on), the whole take saved with the clip cuts the app would have sent (server/app.py).

import { Recorder, encodeWav, supported } from "./recorder.js";

const PLACES = ["서울", "경기", "인천", "강원", "충북", "충남", "대전", "세종", "경북", "대구", "경남", "부산", "울산",
  "전북", "전남", "광주", "제주", "해외·기타"];
const MAX_SECONDS = 120;
const $ = (id) => document.getElementById(id);
const key = new URLSearchParams(location.search).get("key") || "";

let items = [], version = null, sid = null, at = 0;
let rec = null, blob = null, cuts = [], timer = null;

function fill(sel) {
  sel.innerHTML = `<option value="">선택</option>` + PLACES.map((p) => `<option>${p}</option>`).join("");
}

function show(id) {
  for (const s of ["intro", "take", "done", "nokey"]) $(s).hidden = s !== id;
}

async function post(url, body, type) {
  const r = await fetch(url, { method: "POST", body, headers: { "content-type": type } });
  if (!r.ok) {
    let msg = `저장하지 못했어요 (${r.status}).`;
    try { msg = (await r.json()).detail || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

function render() {
  const it = items[at];
  $("prog").style.width = `${(at / items.length) * 100}%`;
  $("step").textContent = it.kind === "s" ? `문장 ${at + 1} / ${items.length} · 소리 내어 읽어 주세요`
    : `자유 말하기 ${at - items.filter((x) => x.kind === "s").length + 1} · 최대 2분`;
  $("text").textContent = it.text;
  $("rec").textContent = "녹음 시작";
  $("play").disabled = $("next").disabled = true;
  $("takeErr").textContent = "";
  blob = null;
}

async function toggle() {
  $("takeErr").textContent = "";
  if (rec) return stop();
  cuts = [];
  rec = new Recorder({
    onSegment: (s) => cuts.push([s.start, s.end]),
    onLevel: (v) => { $("lvl").style.width = `${Math.min(100, v * 400)}%`; },
  });
  try {
    await rec.start();
  } catch {
    rec = null;
    $("takeErr").textContent = "마이크를 쓸 수 없어요. 브라우저에서 마이크 권한을 허용해 주세요.";
    return;
  }
  $("rec").textContent = "녹음 끝내기";
  $("play").disabled = $("next").disabled = true;
  timer = setTimeout(stop, MAX_SECONDS * 1000);
}

async function stop() {
  clearTimeout(timer);
  const r = rec;
  rec = null;
  await r.stop();
  $("lvl").style.width = "0";
  $("rec").textContent = "다시 녹음";
  const samples = r.buf.slice(0, r.buf.n);
  if (samples.length < 16000 * 0.8) {
    $("takeErr").textContent = "녹음이 너무 짧아요. 다시 녹음해 주세요.";
    return;
  }
  blob = encodeWav(samples);
  $("play").disabled = $("next").disabled = false;
}

async function next() {
  if (!blob) return;
  $("next").disabled = true;
  const it = items[at];
  try {
    await post(`/api/testset/clip?key=${encodeURIComponent(key)}&sid=${sid}&item=${it.id}` +
      `&segments=${encodeURIComponent(JSON.stringify(cuts))}`, blob, "audio/wav");
  } catch (e) {
    $("takeErr").textContent = e.message;
    $("next").disabled = false;
    return;
  }
  at += 1;
  if (at >= items.length) {
    show("done");
    return;
  }
  render();
}

async function begin() {
  $("introErr").textContent = "";
  const v = (id) => $(id).value.trim();
  if (!v("raised") || !v("current")) { $("introErr").textContent = "자란 곳과 지금 사는 곳을 골라 주세요."; return; }
  if (!$("consent").checked) { $("introErr").textContent = "녹음 저장에 동의해 주세요."; return; }
  if (!supported()) { $("introErr").textContent = "이 브라우저에서는 녹음할 수 없어요. 최신 크롬이나 사파리로 열어 주세요."; return; }
  $("begin").disabled = true;
  try {
    const r = await post(`/api/testset/session?key=${encodeURIComponent(key)}`, JSON.stringify({
      name: v("name"), raised: v("raised"), current: v("current"), birthplace: v("birthplace"), years: v("years"),
      age: v("age"), sex: v("sex"), self_rating: v("self"), consent: true, version,
    }), "application/json");
    sid = r.sid;
  } catch (e) {
    $("introErr").textContent = e.message;
    $("begin").disabled = false;
    return;
  }
  show("take");
  render();
}

async function init() {
  if (!key) { show("nokey"); return; }
  for (const id of ["raised", "current", "birthplace"]) fill($(id));
  const t = await (await fetch("/assets/testset.json", { cache: "no-cache" })).json();
  version = t.version;
  items = [...t.sentences.map((text, i) => ({ id: `s${String(i).padStart(2, "0")}`, kind: "s", text })),
           ...t.free.map((text, i) => ({ id: `f${i}`, kind: "f", text }))];
  $("begin").onclick = begin;
  $("rec").onclick = toggle;
  $("play").onclick = () => { if (blob) new Audio(URL.createObjectURL(blob)).play(); };
  $("next").onclick = next;
}

init();
