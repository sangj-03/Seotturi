// 커뮤니티: 자유게시판 / Q&A게시판 / 내 스크랩, post view with likes, scraps and comments.
import { api, uid } from "../api.js";
import { $, $$, ago, esc, icon, toast } from "../ui.js";

let board = "free", sort = "new", query = "";
const BOARD = { free: "자유게시판", qna: "Q&A게시판", scrap: "내 스크랩" };

function counts(p) {
  return `<span class="counts"><span>${icon("comment")}${p.comments}</span><span>${icon("like")}${p.likes.toLocaleString()}</span></span>`;
}

function postRow(p) {
  return `<a class="post" href="#/post/${p.id}"><div class="t">${esc(p.title)}${p.sample ? '<span class="sample">예시</span>' : ""}</div>
    <div class="b">${esc(p.body)}</div>
    <div class="m"><span>${esc(p.author)} · ${ago(p.created)} · 조회 ${p.views.toLocaleString()}</span>${counts(p)}</div></a>`;
}

export async function render(view, which) {
  if (which && BOARD[which]) board = which;
  view.innerHTML = `<div class="tabs-top"><h1>커뮤니티</h1>
    <div class="seg" id="tabs">${Object.entries(BOARD).map(([k, l]) => `<button data-b="${k}" class="${k === board ? "on" : ""}">${l}</button>`).join("")}</div>
    ${board !== "scrap" ? `<div class="search outline" style="margin-top:12px"><input id="q" placeholder="검색어를 입력해주세요." enterkeyhint="search" value="${esc(query)}">${icon("search")}</div>` : ""}</div>
    <div id="body" style="padding:4px 16px 20px"></div>`;
  for (const b of $$("#tabs button", view)) b.onclick = () => { location.hash = "#/community/" + b.dataset.b; };
  let fab = null;
  if (board !== "scrap") {
    fab = document.createElement("a");
    fab.className = "fab";
    fab.href = `#/write/${board}`;
    fab.setAttribute("aria-label", "글쓰기");
    fab.innerHTML = icon("pencil");
    $("#shell").append(fab);
  }
  const body = $("#body", view);
  const load = async () => {
    if (board === "scrap") {
      const rows = await api.get("/api/scraps", { uid: uid() });
      body.innerHTML = rows.length ? rows.map(postRow).join("") : `<div class="empty">스크랩한 글이 없어요</div>`;
      return;
    }
    const [hot, rows] = await Promise.all([
      query ? Promise.resolve([]) : api.get("/api/posts/hot", { uid: uid() }),
      api.get("/api/posts", { board, uid: uid(), sort, q: query }),
    ]);
    const medal = (i) => i < 3 ? `<span class="medal m${i + 1}">${i + 1}</span>` : `<span class="medal mn">${i + 1}</span>`;
    body.innerHTML = (hot.length ? `<div class="section-title" style="margin-top:14px">이주의 인기글</div><div class="hot">${hot.map((p, i) => `<a class="hot-item" href="#/post/${p.id}">${medal(i)}
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(p.title)}</span>${counts(p)}</a>`).join("")}</div>` : "")
      + `<div class="section-title">${query ? `'${esc(query)}' 검색 결과` : "게시글"}<select id="sort" class="muted small" style="border:0;background:none;font-weight:600">
          <option value="new" ${sort === "new" ? "selected" : ""}>최신순</option><option value="hot" ${sort === "hot" ? "selected" : ""}>인기순</option></select></div>`
      + (rows.length ? rows.map(postRow).join("") : `<div class="empty">게시글이 없어요. 첫 글을 남겨 보세요!</div>`);
    $("#sort", body).onchange = (e) => { sort = e.target.value; load(); };
  };
  const q = $("#q", view);
  let timer;
  if (q) q.oninput = () => { clearTimeout(timer); timer = setTimeout(() => { query = q.value.trim(); load(); }, 300); };
  await load();
  return () => fab && fab.remove();
}

export async function post(view, id) {
  let p = await api.get(`/api/posts/${id}`, { uid: uid() });
  let parent = null;
  const bar = document.createElement("form");
  bar.className = "comment-bar";
  bar.innerHTML = `<input id="cin" placeholder="댓글을 입력하세요" maxlength="800" autocomplete="off"><button>등록</button>`;
  $("#shell").append(bar);
  const draw = () => {
    const top = p.comment_list.filter((c) => !c.parent);
    const replies = (cid) => p.comment_list.filter((c) => c.parent === cid);
    const cm = (c, reply) => `<div class="comment ${reply ? "reply" : ""}"><img src="/assets/icon.png" alt=""><div style="flex:1;min-width:0">
        <b>${esc(c.author)}</b><span class="tier">${esc(c.tier || "")} · ${ago(c.created)}</span><p>${esc(c.body)}</p>
        ${reply ? "" : `<button class="muted small" data-reply="${c.id}" data-name="${esc(c.author)}" style="margin-top:6px">답글 달기</button>`}</div></div>`;
    view.innerHTML = `<div class="topbar"><button class="back" id="back">${icon("back")}</button><h1>${BOARD[p.board] || "게시글"}</h1>
        ${p.mine ? `<button class="back" id="del" aria-label="삭제">${icon("trash")}</button>` : "<span></span>"}</div>
      <div class="post-view">
        <div class="author"><img src="/assets/icon.png" alt=""><div><b>${esc(p.author)}</b><span>${esc(p.tier || "")} · ${ago(p.created)} · 조회 ${p.views.toLocaleString()}</span></div></div>
        <h2>${esc(p.title)} ${p.sample ? '<span class="sample">예시 글</span>' : ""}</h2>
        <div class="body">${esc(p.body)}</div>
        <button class="like-btn ${p.liked ? "on" : ""}" id="like">${icon("like")} ${p.likes.toLocaleString()}</button>
        <div class="post-actions"><button id="scrap" class="${p.scrapped ? "on" : ""}">${icon(p.scrapped ? "bookmarkOn" : "bookmark")} 스크랩</button><button id="share">${icon("share")} 공유</button></div>
        <div>${top.map((c) => cm(c, false) + replies(c.id).map((r) => cm(r, true)).join("")).join("") || '<div class="empty">첫 댓글을 남겨 보세요</div>'}</div>
      </div>`;
    $("#back", view).onclick = () => (window.history.length > 1 ? window.history.back() : (location.hash = "#/community"));
    $("#like", view).onclick = async () => { const r = await api.post(`/api/posts/${id}/like`, { uid: uid() }); Object.assign(p, { likes: r.likes, liked: r.liked }); draw(); };
    $("#scrap", view).onclick = async () => { const r = await api.post(`/api/posts/${id}/scrap`, { uid: uid() }); p.scrapped = r.scrapped; draw(); toast(r.scrapped ? "스크랩했어요" : "스크랩을 취소했어요"); };
    $("#share", view).onclick = async () => {
      const url = `${location.origin}/app#/post/${id}`;
      try {
        if (navigator.share) await navigator.share({ title: p.title, url });
        else { await navigator.clipboard.writeText(url); toast("링크를 복사했어요"); }
      } catch { /* cancelled */ }
    };
    const del = $("#del", view);
    if (del) del.onclick = async () => { if (confirm("이 글을 삭제할까요?")) { await api.del(`/api/posts/${id}`, { uid: uid() }); location.hash = "#/community/" + p.board; } };
    for (const b of $$("[data-reply]", view)) b.onclick = () => { parent = +b.dataset.reply; const i = $("#cin", bar); i.placeholder = `${b.dataset.name}님에게 답글`; i.focus(); };
  };
  bar.onsubmit = async (e) => {
    e.preventDefault();
    const i = $("#cin", bar);
    if (!i.value.trim()) return;
    try {
      p = await api.post(`/api/posts/${id}/comments`, { uid: uid(), body: i.value, parent });
      i.value = "";
      i.placeholder = "댓글을 입력하세요";
      parent = null;
      draw();
    } catch (err) { toast(err.message); }
  };
  draw();
  return () => bar.remove();
}

export async function write(view, which) {
  const b = which && BOARD[which] && which !== "scrap" ? which : "free";
  view.innerHTML = `<div class="topbar"><button class="back" id="back">${icon("close")}</button><h1>글쓰기</h1><button id="send" style="color:#0a6bff;font-weight:800">등록</button></div>
    <div style="padding:8px 18px">
      <label class="field"><span>게시판</span><select class="select" id="board"><option value="free" ${b === "free" ? "selected" : ""}>자유게시판</option><option value="qna" ${b === "qna" ? "selected" : ""}>Q&A게시판</option></select></label>
      <label class="field"><span>제목</span><input class="input" id="title" maxlength="60" placeholder="제목을 입력하세요"></label>
      <label class="field"><span>내용</span><textarea class="textarea" id="text" style="min-height:260px" maxlength="3000" placeholder="면접 후기, 표준어 연습 팁, 상경 생활 이야기 무엇이든 좋아요"></textarea></label>
    </div>`;
  $("#back", view).onclick = () => (window.history.length > 1 ? window.history.back() : (location.hash = "#/community"));
  $("#send", view).onclick = async () => {
    try {
      const r = await api.post("/api/posts", { uid: uid(), board: $("#board", view).value, title: $("#title", view).value, body: $("#text", view).value });
      location.replace(`#/post/${r.id}`);
    } catch (e) { toast(e.message); }
  };
}
