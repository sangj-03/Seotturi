// Static copy of the 서투리 site (GitHub Pages, built by scripts/build_static.py).
// The pages fetch /api/status, /api/bench and /api/accent. Here those go to the live server on the DGX when it
// answers within 4 s, and otherwise to the last snapshot in data/*.json, marked as offline.
// The app embedded in the landing page only works while the DGX is on, so it is loaded only then.
(() => {
  const C = window.SEOTTURI_STATIC; // { base, live }
  const realFetch = window.fetch.bind(window);
  let live = null; // null: not checked yet, true/false: last result (rechecked after a minute)

  async function fromLive(path) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 4000);
    try {
      const r = await realFetch(C.live + path, { mode: "cors", signal: ctl.signal });
      return r.ok ? r : null;
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  async function api(path) {
    if (live !== false) {
      const r = await fromLive(path);
      live = !!r;
      if (r) return r;
      setTimeout(() => { live = null; }, 60000);
    }
    const body = await (await realFetch(`${C.base}/data/${path.slice(5)}.json`, { cache: "no-cache" })).json();
    if (path === "/api/status") Object.assign(body, { ready: false, error: "offline" });
    return new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
  }

  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    return url === "/api/status" || url === "/api/bench" || url === "/api/accent" ? api(url) : realFetch(input, init);
  };

  document.addEventListener("DOMContentLoaded", async () => {
    const frame = document.querySelector("iframe[data-live-src]");
    if (!frame) return;
    await api("/api/status");
    if (live) {
      frame.src = frame.dataset.liveSrc;
      return;
    }
    for (const a of document.querySelectorAll("a[data-live-link]")) {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        alert("AI 서버가 잠시 꺼져 있어요. 서버가 다시 켜지면 앱을 쓸 수 있어요.");
      });
    }
    const note = document.createElement("div");
    note.className = "app-offline";
    note.innerHTML = `<img src="${C.base}/assets/mascot-wave.png" alt="">
      <b>AI 서버가 잠시 꺼져 있어요</b>
      <p>서투리 AI는 학습용 서버에서 돌아가요. 서버가 다시 켜지면 이 자리에서 바로 써 볼 수 있어요.</p>
      <p>그동안 서비스 소개와 <a href="${C.base}/benchmark/">AI 학습 벤치마크</a>는 그대로 볼 수 있어요.</p>`;
    frame.replaceWith(note);
  });
})();
