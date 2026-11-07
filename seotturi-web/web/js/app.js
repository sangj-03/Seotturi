// App shell: splash, boot, hash router, tab bar.
import { boot, status } from "./api.js";
import { $, $$, closeOverlays, esc } from "./ui.js";
import * as home from "./screens/home.js";
import * as coach from "./screens/coach.js";
import * as journal from "./screens/journal.js";
import * as community from "./screens/community.js";
import "./alive.js";

const framed = new URLSearchParams(location.search).has("frame") || window.self !== window.top;
if (framed) {
  document.body.classList.add("framed");
  $("#statusbar").hidden = false;
  const clock = () => {
    const d = new Date();
    $("#clock").textContent = `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
  };
  clock();
  setInterval(clock, 20000);
}

// [pattern, render(view, ...groups), tab to highlight (null hides the bar), light status bar]
const routes = [
  [/^#\/home$/, home.render, "home"],
  [/^#\/coach$/, coach.landing, "coach", true],
  [/^#\/coach\/setup$/, coach.setup, null],
  [/^#\/coach\/prepare$/, coach.prepare, null],
  [/^#\/coach\/session$/, coach.session, null, true],
  [/^#\/coach\/feedback$/, coach.feedback, null],
  [/^#\/coach\/history\/(\w+)$/, coach.history, null],
  [/^#\/journal(?:\/(\w+))?$/, journal.render, "journal"],
  [/^#\/community(?:\/(\w+))?$/, community.render, "community"],
  [/^#\/post\/(\d+)$/, community.post, null],
  [/^#\/write(?:\/(\w+))?$/, community.write, null],
];

let cleanup = null;
let seq = 0;
let currentTab = null;

// On a wide window outside the phone frame the tab bar is a sidebar (css/app.css), which stays
// visible on every screen; sub-screens highlight the section they belong to.
const wide = window.matchMedia("(min-width: 900px)");
const section = (hash) => (/^#\/coach/.test(hash) ? "coach" : /^#\/(post|write)/.test(hash) ? "community" : null);
const showTabbar = () => { $("#tabbar").hidden = !currentTab && !(wide.matches && !framed); };
wide.addEventListener("change", showTabbar);

async function route() {
  const hash = location.hash || "#/home";
  const match = routes.find(([re]) => re.test(hash));
  if (!match) {
    location.replace("#/home");
    return;
  }
  const [re, render, tab, light] = match;
  const my = ++seq;
  if (cleanup) {
    try { cleanup(); } catch (e) { console.error(e); }
    cleanup = null;
  }
  closeOverlays();
  const view = $("#view");
  view.className = tab ? "has-tab" : "";
  view.scrollTop = 0;
  view.innerHTML = "";
  currentTab = tab;
  showTabbar();
  $("#tabbar").classList.toggle("on-map", tab === "home");
  const active = tab || section(hash);
  for (const a of $$("#tabbar a")) a.classList.toggle("active", a.dataset.tab === active);
  $("#statusbar").classList.toggle("light", !!light);
  try {
    const c = await render(view, ...hash.match(re).slice(1));
    if (my === seq) cleanup = c || null;
    else if (c) c();
  } catch (e) {
    console.error(e);
    if (my === seq) view.innerHTML = `<div class="empty" style="padding-top:120px">화면을 불러오지 못했어요.<br><small>${esc(e.message)}</small><br><br><button class="btn small" onclick="location.reload()">다시 시도</button></div>`;
  }
}

async function start() {
  const splash = document.createElement("div");
  splash.className = "splash";
  splash.innerHTML = '<div class="logo" role="img" aria-label="서투리"></div><p>표현은 선명하게, 기회는 넓게</p><img class="scene" src="/assets/splash-scene.jpg" alt="">';
  $("#shell").append(splash);
  const t0 = Date.now();
  for (let attempt = 0; ; attempt++) {
    try {
      await boot();
      break;
    } catch (e) {
      if (attempt >= 2) {
        splash.querySelector("p").innerHTML = `서버에 연결하지 못했어요.<br><small>${esc(e.message)}</small>`;
        return;
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  }
  status();
  window.addEventListener("hashchange", route);
  await route();
  setTimeout(() => {
    splash.classList.add("hide");
    setTimeout(() => splash.remove(), 500);
  }, Math.max(0, 1300 - (Date.now() - t0)));
}

start();
