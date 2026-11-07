// Server calls, the anonymous user id and a tiny event bus.

const UID_KEY = "seotturi.uid";
const memory = {};

export const store = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch {
      return key in memory ? memory[key] : fallback;
    }
  },
  set(key, value) {
    memory[key] = value;
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: keep it in memory */ }
  },
};

export let me = null;

export function setMe(profile) {
  me = profile;
  store.set(UID_KEY, profile.uid);
  emit("me", profile);
}

export function uid() {
  return (me && me.uid) || store.get(UID_KEY, "");
}

async function request(method, url, body, headers = {}) {
  const opts = { method, headers: { ...headers } };
  if (body instanceof Blob) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["content-type"] = "application/json";
  }
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = new Error((data && (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))) || `요청 실패 (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function withQuery(url, params) {
  if (!params) return url;
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") q.set(k, v);
  const s = q.toString();
  return s ? `${url}${url.includes("?") ? "&" : "?"}${s}` : url;
}

export const api = {
  get: (url, params) => request("GET", withQuery(url, params)),
  post: (url, body, params) => request("POST", withQuery(url, params), body),
  patch: (url, body) => request("PATCH", url, body),
  del: (url, params) => request("DELETE", withQuery(url, params)),
  audio: (url, blob, params) => request("POST", withQuery(url, params), blob, { "content-type": blob.type || "application/octet-stream" }),
};

export async function boot() {
  setMe(await api.post("/api/me", { uid: store.get(UID_KEY, "") }));
  return me;
}

// Server-sent events over a POST request; yields parsed `data:` objects.
export async function* stream(url, body, signal) {
  const res = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal });
  if (!res.ok || !res.body) throw new Error(`요청 실패 (${res.status})`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          try { yield JSON.parse(line.slice(6)); } catch { /* partial line */ }
        }
      }
    }
  }
}

// ---- model status, polled once and shared ----
let statusCache = null;
let statusAt = 0;
export async function status(force = false) {
  if (!force && statusCache && Date.now() - statusAt < 15000) return statusCache;
  try {
    statusCache = await api.get("/api/status");
  } catch {
    statusCache = { ready: false, error: "offline" };
  }
  statusAt = Date.now();
  emit("status", statusCache);
  return statusCache;
}

// ---- events ----
const listeners = {};
export function on(name, fn) {
  (listeners[name] ||= new Set()).add(fn);
  return () => listeners[name].delete(fn);
}
export function emit(name, value) {
  for (const fn of listeners[name] || []) fn(value);
}
