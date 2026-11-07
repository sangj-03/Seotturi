// The AI side speaks. Lines are synthesised on the server by Supertonic 3 (on the CPU, cached;
// see server/tts.py and /api/tts) with a voice per persona, so every device hears the same
// natural voice. If that fails, the browser's own speech synthesis reads the line instead.
import { store } from "./api.js";
import { toast } from "./ui.js";
import { byName } from "./personas.js";

const synth = typeof window !== "undefined" ? window.speechSynthesis : null;
const audio = typeof Audio !== "undefined" ? new Audio() : null;
const KEY = "seotturi.voice";
const MALE = /injoon|hyunsu|bongjin|gookmin|minjun|male|남성|x-koc|x-kod/i;
const FEMALE = /sunhi|yuna|heami|jimin|seohyeon|soonbok|yujin|sora|female|여성|x-ism|x-kob/i;
const QUALITY = [/natural|neural/i, /online/i, /premium|enhanced|고품질/i, /google/i];

// Fallback only: persona -> [voice gender, rate, pitch] for the browser's synthesis; personas not
// listed get the gender and speed of their server voice (web/js/personas.js).
const PROFILE = {
  "따뜻한 멘토형": ["F", 0.95, 1.05], "깐깐한 부장님형": ["M", 0.92, 0.85], "차분한 인사팀장형": ["M", 0.85, 0.95],
  "유쾌한 스타트업 리더형": ["M", 1.08, 1.1], "냉철한 임원형": ["F", 1.0, 0.9], "우호적인 동료들": ["F", 1.0, 1.05],
  "날카로운 심사위원": ["M", 0.95, 0.9], "바쁜 임원진": ["M", 1.1, 0.9], "다정한 친구 지민": ["F", 1.0, 1.1],
  "장난꾸러기 친구 도윤": ["M", 1.1, 1.1], "무뚝뚝한 친구 서진": ["M", 0.95, 0.9],
};

let voices = [];
let current = null; // {id, onend} of the line being spoken
// Tells the page when a line starts and stops being heard (js/alive.js moves the speaker's mouth).
const announce = (speaking) => window.dispatchEvent(new CustomEvent("voice", { detail: { speaking } }));
let ids = 0;
let warned = false;

function load() {
  voices = synth ? synth.getVoices() : [];
}
if (synth) {
  load();
  synth.onvoiceschanged = load;
}
if (audio) audio.preload = "auto";

export function supported() {
  return !!(audio || synth);
}
export function enabled() {
  return store.get(KEY, true);
}
export function setEnabled(on) {
  store.set(KEY, !!on);
  if (!on) stop();
}

function finish(id) {
  if (current && current.id === id) {
    const end = current.onend;
    current = null;
    announce(false);
    if (end) end();
  }
}

export function stop() {
  if (audio) audio.pause();
  if (synth) synth.cancel();
  if (current) finish(current.id);
}

// Call inside a click: iOS only lets audio start from a user gesture; after that the same
// element (and speech synthesis) may play without one.
export function unlock() {
  if (audio) {
    try {
      audio.src = "/assets/silent.mp3";
      const p = audio.play();
      if (p) p.catch(() => {});
    } catch { /* not supported */ }
  }
  if (synth) {
    try {
      const u = new SpeechSynthesisUtterance(" ");
      u.volume = 0;
      synth.speak(u);
    } catch { /* not supported */ }
  }
}

export async function speak(text, persona, { onstart, onend } = {}) {
  if (!enabled() || !text) { if (onend) onend(); return false; }
  stop();
  const id = ++ids;
  current = { id, onend };
  if (audio) {
    try {
      await playServer(text, persona, id, onstart);
      finish(id);
      return true;
    } catch {
      if (!current || current.id !== id) return false; // stopped meanwhile
    }
  }
  return browserSpeak(text, persona, id, onstart);
}

function playServer(text, persona, id, onstart) {
  return new Promise((resolve, reject) => {
    audio.onplaying = () => { if (current && current.id === id) { announce(true); if (onstart) onstart(); } };
    audio.onwaiting = () => { if (current && current.id === id) announce(false); };
    audio.onended = () => resolve();
    audio.onerror = () => reject(new Error("audio"));
    const who = byName(persona); // its voice goes along; the server maps bare names of older pages itself
    const v = who ? `&voice=${who.voice[0]}&speed=${who.voice[1]}` : "";
    audio.src = `/api/tts?persona=${encodeURIComponent(persona || "")}${v}&text=${encodeURIComponent(text)}`;
    const p = audio.play();
    if (p) p.catch(reject);
  });
}

function pick(gender) {
  let best = null, bestScore = -1;
  for (const v of voices) {
    if (!/^ko/i.test(v.lang || "")) continue;
    let s = 100;
    QUALITY.forEach((re, i) => { if (re.test(v.name)) s += 20 - i * 3; });
    if (gender === "M" && MALE.test(v.name)) s += 15;
    if (gender === "F" && FEMALE.test(v.name)) s += 15;
    if (v.localService === false) s += 3;
    if (s > bestScore) { best = v; bestScore = s; }
  }
  return best;
}

async function browserSpeak(text, persona, id, onstart) {
  if (!synth) { finish(id); return false; }
  if (!voices.length) {
    await new Promise((r) => setTimeout(r, 500));
    load();
  }
  const p = byName(persona);
  const [gender, rate, pitch] = PROFILE[p ? p.name : persona] || (p ? [p.voice[0][0], p.voice[1], 1] : ["F", 1, 1]);
  const v = pick(gender);
  if (!v || !current || current.id !== id) {
    if (!v && !warned) toast("음성을 재생하지 못했어요. 잠시 후 다시 들어 보세요.", 3500);
    warned = warned || !v;
    finish(id);
    return false;
  }
  const u = new SpeechSynthesisUtterance(text);
  u.voice = v;
  u.lang = v.lang;
  u.rate = rate;
  u.pitch = gender === "M" && !MALE.test(v.name) ? pitch * 0.85 : pitch;
  u.onstart = () => { if (current && current.id === id) { announce(true); if (onstart) onstart(); } };
  u.onend = u.onerror = () => finish(id);
  current.utterance = u; // keep it referenced: Chrome drops callbacks of collected utterances
  synth.speak(u);
  return true;
}
