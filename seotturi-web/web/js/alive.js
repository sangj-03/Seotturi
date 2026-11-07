// Characters that blink, talk and react. scripts/alive_frames.py cuts the changed part of every
// frame (eyes half closed / closed, mouth a little / wide open) out of the character images; here
// the patches are pasted onto the decoded originals and the frames are swapped into each <img> or
// background that shows one of those images, on the screens rendered now and later.
//
//   blink  every few seconds; in a scene with several people each face keeps its own rhythm
//   talk   while an AI line plays (voice.js fires "voice") the [data-speaker] element moves its
//          mouth; chatter(el, ms) does it for a moment
//   poke   a tap on a character makes it hop, blink and chatter
//   wave   an image with a rig (the landing page mascot) gets its raised hand as a separate layer
//          that css/alive.css turns about the wrist
// The idle motion (breathing, floating, clouds, birds) is in css/alive.css. Nothing here runs when
// the system asks for reduced motion.

const ASSETS = new URL("../assets/", import.meta.url);
const SELECTOR = "img, [style*='background'], .result-art"; // .result-art: its background comes from app.css
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
const rand = (a, b) => a + Math.random() * (b - a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let manifest = null; // {v, images: {"icon.png": {size: [w, h], blink: [[[file, x, y], [..]] per face], talk: [[file, x, y], [..]], rig?}}}
const frames = new Map(); // "icon.png full blink" -> Promise of frame URLs
const actors = new Map(); // element -> its state

const seen = typeof IntersectionObserver === "function" ? new IntersectionObserver((entries) => {
  for (const e of entries) { const a = actors.get(e.target); if (a) a.visible = e.isIntersecting; }
}) : null;

// Which character image an element shows, if any (works under a sub-path such as GitHub Pages).
function imageOf(el) {
  let src;
  if (el.tagName === "IMG") src = el.getAttribute("src");
  else {
    const m = /url\(["']?([^"')]+)/.exec(el.style.backgroundImage || getComputedStyle(el).backgroundImage);
    src = m && m[1];
  }
  if (!src || src.startsWith("blob:")) return null;
  const url = new URL(src, location.href);
  const name = decodeURIComponent(url.pathname.split("/assets/").pop());
  return manifest.images[name] ? { name, url: url.href } : null;
}

function load(src) {
  const im = new Image();
  im.src = src;
  return im.decode().then(() => im);
}
const patch = (file) => load(new URL(`alive/${file}?v=${manifest.v}`, ASSETS));

// An <img> gets whole frames (the patch pasted onto the original). A background gets the patch on a
// transparent layer the size of the image, laid over the original with the same background-size,
// so the original keeps showing until the layer is decoded.
function compose(a, kind) {
  const mode = a.bg ? "layer" : "full";
  const key = `${a.name} ${mode} ${kind}`;
  if (!frames.has(key)) frames.set(key, build(a, mode, kind));
  return frames.get(key);
}

async function build(a, mode, kind) {
  const { size, blink, talk, rig } = manifest.images[a.name];
  const base = mode === "full" ? await load(a.url) : null;
  if (base && (base.naturalWidth !== size[0] || base.naturalHeight !== size[1])) {
    throw new Error(`${a.name} changed size: run scripts/alive_frames.py`);
  }
  const body = base && rig ? [await patch(rig.body[0]), rig.body[1], rig.body[2]] : null; // the hand is a layer of its own
  const paste = async (spec) => {
    const layers = [body, spec && [await patch(spec[0]), spec[1], spec[2]]].filter(Boolean);
    const c = document.createElement("canvas");
    [c.width, c.height] = size;
    const g = c.getContext("2d");
    if (base) g.drawImage(base, 0, 0);
    for (const [im, x, y] of layers) {
      g.clearRect(x, y, im.naturalWidth, im.naturalHeight); // keep the patch's own alpha
      g.drawImage(im, x, y);
    }
    const blob = await new Promise((r) => c.toBlob(r));
    c.width = c.height = 0;
    if (!blob) throw new Error("canvas");
    const url = URL.createObjectURL(blob);
    await load(url); // decoded before it is first shown
    return url;
  };
  if (kind === "rest") return paste(null);
  return kind === "blink" ? Promise.all(blink.map((face) => Promise.all(face.map(paste)))) : Promise.all(talk.map(paste));
}

// The raised hand becomes an <img> over the character, sized and placed in percent of the image
// and turned about the wrist by css/alive.css; the image itself shows the body without it from now on.
async function rigUp(a) {
  const { size: [w, h], rig } = manifest.images[a.name];
  const [rest, hand] = await Promise.all([compose(a, "rest"), patch(rig.hand[0])]);
  if (!here(a)) return;
  const [, x, y] = rig.hand, [px, py] = rig.pivot;
  hand.className = "alive-hand";
  hand.alt = "";
  hand.style.cssText = `left:${(x / w) * 100}%;top:${(y / h) * 100}%;width:${(hand.naturalWidth / w) * 100}%;`
    + `transform-origin:${((px - x) / hand.naturalWidth) * 100}% ${((py - y) / hand.naturalHeight) * 100}%`;
  const box = document.createElement("span");
  box.className = "alive-rig";
  a.el.replaceWith(box);
  box.append(a.el, hand);
  a.orig = rest;
  show(a);
}

function adopt(el) {
  if (actors.has(el)) return actors.get(el);
  if (!manifest) return null;
  const img = imageOf(el);
  if (!img) return null;
  const bg = el.tagName !== "IMG";
  const a = { el, bg, name: img.name, url: img.url, orig: bg ? el.style.backgroundImage : el.getAttribute("src"),
    blink: null, talk: null, face: -1, eyes: 0, mouth: 0, talking: false, talkLoop: false, visible: true, timers: [] };
  actors.set(el, a);
  if (seen) seen.observe(el);
  a.ready = !bg && manifest.images[img.name].rig ? rigUp(a) : Promise.resolve();
  a.ready.then(() => compose(a, "blink")).then((urls) => {
    a.blink = urls;
    urls.forEach((_, face) => schedule(a, face, rand(400, 2400)));
  }).catch((e) => { console.warn("alive:", e.message); drop(a); });
  return a;
}

function drop(a) {
  a.timers.forEach(clearTimeout);
  a.talking = false;
  actors.delete(a.el);
  if (seen) seen.unobserve(a.el);
}

const here = (a) => (a.el.isConnected ? true : (drop(a), false));

// Eyes win over the mouth: a blink in the middle of a line shows the closed-mouth blink frame.
function show(a) {
  const url = a.face >= 0 && a.eyes ? a.blink[a.face][a.eyes - 1] : a.mouth && a.talk ? a.talk[a.mouth - 1] : null;
  if (a.bg) a.el.style.backgroundImage = url ? `url("${url}"), url("${a.url}")` : a.orig;
  else a.el.setAttribute("src", url || a.orig);
}

function schedule(a, face, ms = rand(2400, 6500)) {
  clearTimeout(a.timers[face]);
  a.timers[face] = setTimeout(() => blink(a, face), ms);
}

async function blink(a, face, again = false) {
  if (!here(a) || !a.blink) return;
  if (!a.visible || document.hidden) return schedule(a, face);
  if (a.face >= 0) return schedule(a, face, rand(250, 700)); // another face of the scene is blinking
  a.face = face;
  for (const [eyes, ms] of [[1, 45], [2, rand(70, 110)], [1, 45], [0, 0]]) {
    a.eyes = eyes;
    show(a);
    if (ms) await sleep(ms);
  }
  a.face = -1;
  if (!again && Math.random() < 0.15) { // now and then a double blink
    await sleep(rand(110, 180));
    return blink(a, face, true);
  }
  schedule(a, face);
}

// Lip flaps of uneven size and length, now and then a pause between words.
async function mouth(a) {
  a.talkLoop = true;
  try {
    await a.ready;
    if (!a.talk) a.talk = await compose(a, "talk");
    while (a.talking && here(a)) {
      a.mouth = a.mouth ? (Math.random() < 0.55 ? 0 : 3 - a.mouth) : Math.random() < 0.6 ? 2 : 1;
      show(a);
      await sleep(a.mouth ? rand(70, 130) : Math.random() < 0.15 ? rand(180, 320) : rand(50, 110));
    }
  } catch { /* this image has no talk frames */ }
  a.mouth = 0;
  a.talkLoop = false;
  if (a.el.isConnected) show(a);
}

export function talk(el, on) {
  const a = still ? null : adopt(el);
  if (!a || !manifest.images[a.name].talk.length) return;
  a.talking = on;
  if (on && !a.talkLoop) mouth(a);
}

export function chatter(el, ms = 900) {
  talk(el, true);
  const a = actors.get(el);
  if (!a) return;
  clearTimeout(a.chatter);
  a.chatter = setTimeout(() => talk(el, false), ms);
}

function poke(e) {
  const el = e.target.closest && e.target.closest(SELECTOR);
  const a = el && actors.get(el);
  if (!a || el.closest("[data-speaker]")) return;
  (el.parentElement.classList.contains("alive-rig") ? el.parentElement : el).animate([
    { scale: "1", translate: "0 0" },
    { scale: "1.07 .93", translate: "0 0", offset: 0.18 },
    { scale: ".96 1.05", translate: "0 -9%", offset: 0.45 },
    { scale: "1.03 .97", translate: "0 0", offset: 0.75 },
    { scale: "1", translate: "0 0" },
  ], { duration: 560, easing: "ease-out" });
  if (a.blink && a.face < 0) blink(a, 0, true);
  chatter(el, rand(500, 800));
}

// Pass/finish celebration over the result card.
export function confetti(host, n = 42) {
  if (still || !host) return;
  const box = document.createElement("div");
  box.className = "confetti";
  const colors = ["#0a6bff", "#4fb3ff", "#ffc53d", "#ff7a9a", "#4fd8b4", "#ffffff"];
  const fall = host.clientHeight + 40;
  for (let i = 0; i < n; i++) {
    const p = document.createElement("i");
    p.style.cssText = `left:${rand(2, 98)}%;background:${colors[i % colors.length]};--fall:${fall}px;--dx:${rand(-70, 70)}px;`
      + `--spin:${rand(-720, 720)}deg;animation-duration:${rand(1500, 2600)}ms;animation-delay:${rand(0, 350)}ms`;
    box.append(p);
  }
  host.append(box);
  setTimeout(() => box.remove(), 3200);
}

async function start() {
  if (still) return;
  const r = await fetch(new URL("alive/manifest.json", ASSETS), { cache: "no-cache" });
  if (!r.ok) throw new Error(`manifest ${r.status}`);
  manifest = await r.json();
  const scan = (root) => {
    if (root.matches(SELECTOR)) adopt(root);
    for (const el of root.querySelectorAll(SELECTOR)) adopt(el);
  };
  scan(document.body);
  new MutationObserver((records) => {
    for (const r of records) for (const n of r.addedNodes) if (n.nodeType === 1) scan(n);
  }).observe(document.body, { childList: true, subtree: true });
  addEventListener("voice", (e) => { for (const el of document.querySelectorAll("[data-speaker]")) talk(el, e.detail.speaking); });
  document.addEventListener("pointerdown", poke, { passive: true });
}

export const ready = start().catch((e) => console.warn("alive:", e.message));
