"""서투리 web server: the website, the app's API and a proxy to the Gemma 4 engine.

    /data/seotturi/.venv/bin/python server/app.py        (or ./run.sh start)

Serves web/ (landing page at /, the app at /app) and /api/* on SEOTTURI_HOST:SEOTTURI_PORT
(127.0.0.1:8787 by default). Speech goes to the engine (server/engine.py on 127.0.0.1:8788);
users, sessions, review items and the community board live in data/app.db (SQLite).
Recordings are analysed in memory and never stored, unless a user opts in to contributing
them to training (data/contrib/).
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict, defaultdict, deque
from datetime import date, datetime, timedelta

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A  # noqa: E402
import demo as D  # noqa: E402
import engine_files as EF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # repository root
WEB = os.path.join(ROOT, "web")
DATA = os.path.join(ROOT, "data")
DB = os.path.join(DATA, "app.db")
ENGINE = "http://127.0.0.1:" + os.environ.get("SEOTTURI_ENGINE_PORT", "8788")
TTS = "http://127.0.0.1:" + os.environ.get("SEOTTURI_TTS_PORT", "8789")
ACCENT = "http://127.0.0.1:" + os.environ.get("SEOTTURI_ACCENT_PORT", "8790")
TTS_CACHE = os.path.join(DATA, "tts_cache")
STATE = os.path.join(ROOT, "state")  # the engine's files: served adapter, history, benchmark results
# 체험 모드 (server/demo.py): while the engine does not answer at all, AI answers are labelled examples, so the
# site can be tried without the GPU. SEOTTURI_DEMO=0 gives the old errors instead.
DEMO = os.environ.get("SEOTTURI_DEMO", "1") != "0"
UPDATE_HOURS = float(os.environ.get("SEOTTURI_UPDATE_HOURS", "2"))  # the engine's update cycle, for display
# The accent zone comes from the accent service (server/accent.py: a Whisper-based classifier trained in
# training/accent-id) and is only shown once its held-out macro-F1 on speech without dialect words
# reaches this value. Gemma's own 지역 line is not used: on such speech it names the zone from the words.
ZONE_MIN_F1 = float(os.environ.get("SEOTTURI_ZONE_MIN_F1", "0.5"))
_zone = {"ok": False, "at": 0.0, "std": False}  # std: the accent service runs the standard-vs-regional detector
# The app sends each persona's Supertonic 3 voice and speed with the line (web/js/personas.js);
# this map serves pages that send only the persona name (older cached pages, the static copy).
# Picked by measured pitch: M5 is a deep male voice, M1 the brightest male, F2 the brightest
# female, F4 the flattest female intonation.
VOICES = {
    "따뜻한 멘토형": ("F1", 1.0), "깐깐한 부장님형": ("M5", 0.95), "차분한 교수형": ("M3", 0.9), "차분한 인사팀장형": ("M3", 0.9),
    "유쾌한 스타트업 리더형": ("M1", 1.12), "냉철한 임원형": ("F4", 1.05),
    "우호적인 동료들": ("F3", 1.05), "날카로운 심사위원": ("M2", 1.0), "바쁜 임원진": ("F5", 1.12),
    "다정한 친구 지민": ("F2", 1.05), "장난꾸러기 친구 도윤": ("M4", 1.12), "무뚝뚝한 친구 서진": ("M2", 0.95),
}
TTS_VOICES = {f"{g}{i}" for g in "FM" for i in range(1, 6)}  # the presets of server/tts.py
TIERS = [("사투리 마스터", 0), ("사투리 덜스터", 150), ("표투리", 350), ("표준어 덜스터", 600), ("표준어 마스터", 1000)]
ADJ = ["별난", "용감한", "수줍은", "씩씩한", "느긋한", "재빠른", "다정한", "엉뚱한", "반짝이는", "꿋꿋한"]
ANIMALS = ["백고미", "반달고미", "너구리", "수달", "다람쥐", "펭귄", "고래", "올빼미", "토끼", "해달"]
# Regions of the users' profile -> regional sets of the training data (the word book's regions).
DICT_REGION = {"부산": "경상", "울산": "경상", "경남": "경상", "대구": "경상", "경북": "경상", "경상": "경상",
               "강원": "강원", "제주": "제주", "대전": "충청", "세종": "충청", "충남": "충청", "충북": "충청", "충청": "충청",
               "광주": "전라", "전남": "전라", "전북": "전라", "전라": "전라"}

app = FastAPI(title="seotturi-web")
http = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=5.0))
DICT = []


# ---- limits: the site may be public, and its GPU is shared with the training run --------------
# (methods, path, bucket, requests, seconds) per client; a client is the first X-Forwarded-For
# address (set by the Tailscale proxy) or the socket's address.
RULES = [
    ("POST", r"^/api/(asr|reviews/\d+/attempt)$", "asr", 90, 600),
    ("POST", r"^/api/(coach/feedback|coach/reply|reviews/generate)$", "llm", 30, 600),
    ("POST", r"^/api/posts(/\d+/comments)?$", "write", 20, 600),
    ("POST", r"^/api/me$", "user", 20, 600),
    ("POST PATCH DELETE", r"^/api/", "misc", 300, 600),
]
HITS = defaultdict(deque)
MAX_INFLIGHT = int(os.environ.get("SEOTTURI_MAX_INFLIGHT", "24"))  # engine calls in progress, all users
INFLIGHT = 0
MAX_UPLOAD_BYTES = 25_000_000
MAX_UPLOAD_SECONDS = 180


def client_key(request):
    xff = request.headers.get("x-forwarded-for")
    return xff.split(",")[0].strip() if xff else (request.client.host if request.client else "?")


def allow(bucket, key, limit, window):
    hits = HITS[(bucket, key)]
    now = time.monotonic()
    while hits and now - hits[0] > window:
        hits.popleft()
    if len(hits) >= limit:
        return False
    hits.append(now)
    if len(HITS) > 20000:  # forget idle clients
        for k in [k for k, v in HITS.items() if not v or now - v[-1] > 3600]:
            del HITS[k]
    return True


@app.middleware("http")
async def rate_limit(request, call_next):
    path = request.url.path
    if request.method in ("POST", "PATCH", "DELETE") and path.startswith("/api/"):
        key = client_key(request)
        for methods, rx, bucket, limit, window in RULES:
            if request.method in methods.split() and re.match(rx, path):
                if not allow(bucket, key, limit, window):
                    return JSONResponse({"detail": "요청이 너무 많아요. 잠시 후 다시 시도해 주세요."}, status_code=429)
                break
    return await call_next(request)


# The static copy of the site (GitHub Pages, scripts/build_static.py) reads these two read-only
# endpoints from the visitor's browser to show whether the AI is up. Nothing else is cross-origin.
STATIC_ORIGINS = {o.strip() for o in os.environ.get("SEOTTURI_STATIC_ORIGINS", "https://sangj-03.github.io").split(",")
                  if o.strip()}


@app.middleware("http")
async def static_site_cors(request, call_next):
    response = await call_next(request)
    if request.url.path.endswith((".js", ".css")):  # revalidate (ETag) instead of a heuristic cache lifetime,
        response.headers["Cache-Control"] = "no-cache"  # so module imports never mix old and new files
    origin = request.headers.get("origin")
    if request.method == "GET" and request.url.path in ("/api/status", "/api/bench", "/api/accent") and origin in STATIC_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return response


class GpuSlot:
    """Counts engine calls in progress; beyond MAX_INFLIGHT new ones are turned away (503)."""

    def __enter__(self):
        global INFLIGHT
        if INFLIGHT >= MAX_INFLIGHT:
            raise HTTPException(503, "지금 이용자가 많아요. 잠시 후 다시 시도해 주세요.")
        INFLIGHT += 1

    def __exit__(self, *exc):
        global INFLIGHT
        INFLIGHT -= 1


# ---- database ---------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, nickname TEXT, tag TEXT, region TEXT,
    xp INTEGER DEFAULT 0, contribute INTEGER DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, user_id TEXT, mode TEXT, title TEXT, created REAL,
    seconds REAL, points INTEGER, passed INTEGER, xp INTEGER, data TEXT);
CREATE TABLE IF NOT EXISTS usages(id INTEGER PRIMARY KEY, user_id TEXT, session_id TEXT, dialect TEXT,
    standard TEXT, n INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY, user_id TEXT, session_id TEXT, title TEXT,
    dialect TEXT, standard TEXT, words TEXT, status TEXT, tries INTEGER DEFAULT 0, hits INTEGER DEFAULT 0,
    created REAL, done REAL, last TEXT, practiced REAL);
CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, user_id TEXT, review_id INTEGER, ok INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS favorites(user_id TEXT, region TEXT, dialect TEXT, standard TEXT, created REAL,
    PRIMARY KEY(user_id, region, dialect, standard));
CREATE TABLE IF NOT EXISTS posts(id INTEGER PRIMARY KEY, board TEXT, user_id TEXT, author TEXT, tier TEXT,
    title TEXT, body TEXT, created REAL, views INTEGER DEFAULT 0, base_likes INTEGER DEFAULT 0,
    sample INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS comments(id INTEGER PRIMARY KEY, post_id INTEGER, user_id TEXT, author TEXT,
    tier TEXT, body TEXT, created REAL, parent INTEGER);
CREATE TABLE IF NOT EXISTS likes(post_id INTEGER, user_id TEXT, PRIMARY KEY(post_id, user_id));
CREATE TABLE IF NOT EXISTS scraps(post_id INTEGER, user_id TEXT, created REAL, PRIMARY KEY(post_id, user_id));
"""


def db():
    con = sqlite3.connect(DB, timeout=15)
    con.row_factory = sqlite3.Row
    return con


def q(sql, args=(), one=False):
    con = db()
    try:
        rows = [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()
    return (rows[0] if rows else None) if one else rows


def x(sql, args=()):
    con = db()
    try:
        with con:
            return con.execute(sql, args).lastrowid
    finally:
        con.close()


def init_db():
    os.makedirs(DATA, exist_ok=True)
    con = db()
    with con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
        if "practiced" not in [r[1] for r in con.execute("PRAGMA table_info(reviews)")]:  # added 2026-10-07
            con.execute("ALTER TABLE reviews ADD COLUMN practiced REAL")
    con.close()
    if not q("SELECT 1 FROM posts LIMIT 1"):
        seed_posts()


def seed_posts():
    """Example posts from the deck's community screens, marked as examples in the app."""
    now = time.time()
    posts = [
        ("free", "반달고미", "사투리 준마스터", "취업 진짜 너무 힘들다..", "서류 열 번째 떨어졌어요. 다들 어떻게 버티세요? 면접까지만 가면 자신 있는데…", 2719, 3),
        ("free", "수상한 도롱뇽", "표준어 덜스터", "이제 진짜 서울인 다 됐다 !!", "어제 면접에서 '어디 출신이세요?' 소리 한 번도 안 들었어요. 서투리 3주차 후기입니다 ㅎㅎ", 2104, 5),
        ("free", "별난 백고미", "표투리", "현대자동차 면접 후기 알려드림", "1차 실무면접은 직무 질문 위주였고, 2차는 인성 위주였어요. 긴장해서 말끝이 올라갈 때마다 서투리 연습한 게 생각났네요.", 2003, 8),
        ("free", "느긋한 수달", "사투리 덜스터", "종로 국밥집 추천", "상경하고 나서 제일 그리운 게 돼지국밥… 종로 근처 괜찮은 곳 있으면 추천 부탁해요.", 1834, 10),
        ("qna", "씩씩한 펭귄", "사투리 마스터", "면접 복장 뭐 입어야 돼?", "비즈니스 캐주얼이라고 하면 정장 재킷은 꼭 입어야 하나요?", 1769, 12),
        ("free", "반달고미", "사투리 준마스터", "아니 근데 서울", "국밥은 원래 사골육수 베이스임??? 내가 사는 본가는 거의 다 돼지국밥이었는데…", 12, 0.1),
        ("free", "수상한 도롱뇽", "표준어 덜스터", "서울 버스", "승하차 때 카드 찍어야 되는 거 나만 지금 알았나… 여태까지 돈을 두 번 냈네", 35, 0.2),
        ("free", "반달고미", "사투리 준마스터", "서울에서 어디로", "놀러 가야 될지 추천받음. 상경한 지 얼마 안 돼서 아는 데가 없어요", 78, 0.4),
        ("free", "엉뚱한 너구리", "표투리", "요새 드라마", "드라마에 나오는 사투리 너무 어색하지 않아요? 다들 열심히 하는 건 알지만 ㅋㅋ", 70, 0.5),
        ("free", "다정한 고래", "사투리 덜스터", "소개팅 꿀팁", "주말에 소개팅인데 ㅠㅠ 처음이라 긴장돼요. 말투 신경 쓰이는데 그냥 편하게 해도 될까요?", 78, 0.6),
        ("free", "느긋한 수달", "표준어 덜스터", "상경한 지 어느새", "서울살이 한 지 어느새 10년 차… 아직도 전화할 때 엄마랑 통화하면 바로 돌아와요", 70, 0.8),
        ("free", "반달고미", "사투리 준마스터", "이 정도면 삼성 가능??", "한국대학교 정보 학점 3.8이고, 삼성 SW 아카데미 활동, 미래융합 소프트웨어 챌린지 수상, 네이버 인턴 경험 있음", 11, 0.9),
        ("free", "재빠른 다람쥐", "표투리", "취준 자취생 생활비...", "취준하면서 자취하는데 생활비가 생각보다 너무 많이 나와요. 다들 한 달에 얼마 쓰세요?", 28, 1.2),
        ("qna", "수줍은 토끼", "사투리 마스터", "'~했나?'랑 '~했어?' 억양 차이가 뭐예요?", "경상도에서는 끝을 올리는데 표준어는 어떻게 말해야 자연스러운지 궁금해요.", 46, 1.5),
        ("qna", "꿋꿋한 해달", "사투리 덜스터", "면접에서 사투리 쓰면 진짜 감점인가요?", "회사마다 다르다는데 실제로 경험해 보신 분 계신가요?", 64, 2.0),
        ("qna", "다정한 고래", "표투리", "충청도 말끝 늘이는 거 어떻게 고쳐요?", "발표할 때마다 '~했는디유'가 튀어나와요. 연습 방법 공유해 주세요!", 39, 2.5),
    ]
    con = db()
    with con:
        for board, author, tier, title, body, likes, days in posts:
            cur = con.execute("INSERT INTO posts(board,user_id,author,tier,title,body,created,views,base_likes,sample)"
                              " VALUES(?,?,?,?,?,?,?,?,?,1)",
                              (board, "", author, tier, title, body, now - days * 86400, likes * 3, likes))
            if title.startswith("이 정도면 삼성"):
                pid = cur.lastrowid
                c1 = con.execute("INSERT INTO comments(post_id,user_id,author,tier,body,created,parent) VALUES(?,?,?,?,?,?,NULL)",
                                 (pid, "", "익명1", "표투리", "나쁘지 않은데 정보처리기사 자격증 없으면 꼭 따세요 !!", now - 0.8 * 86400)).lastrowid
                con.execute("INSERT INTO comments(post_id,user_id,author,tier,body,created,parent) VALUES(?,?,?,?,?,?,?)",
                            (pid, "", "반달고미 (글쓴이)", "사투리 준마스터", "아하 감사합니다!!", now - 0.7 * 86400, c1))
                con.execute("INSERT INTO comments(post_id,user_id,author,tier,body,created,parent) VALUES(?,?,?,?,?,?,NULL)",
                            (pid, "", "익명2", "표준어 준마스터", "잘 모르겠네요", now - 0.6 * 86400))
    con.close()


# ---- users ------------------------------------------------------------------------------------
def tier_of(xp):
    i = max(k for k, (_, lo) in enumerate(TIERS) if xp >= lo)
    name, lo = TIERS[i]
    hi = TIERS[i + 1][1] if i + 1 < len(TIERS) else None
    return {"name": name, "index": i, "min": lo, "max": hi, "next": TIERS[i + 1][0] if hi else None,
            "progress": 1.0 if hi is None else round((xp - lo) / (hi - lo), 3)}


def profile(u):
    return {"uid": u["id"], "nickname": u["nickname"], "tag": u["tag"], "region": u["region"], "xp": u["xp"],
            "contribute": bool(u["contribute"]), "tier": tier_of(u["xp"]),
            "tiers": [{"name": n, "min": lo} for n, lo in TIERS]}


def user(uid):
    u = q("SELECT * FROM users WHERE id=?", (uid,), one=True) if uid else None
    if not u:
        raise HTTPException(401, "unknown user")
    return u


def valid_uid(uid):
    return isinstance(uid, str) and re.fullmatch(r"[A-Za-z0-9_-]{8,64}", uid)


@app.post("/api/me")
def me(body: dict):
    uid = body.get("uid")
    if not valid_uid(uid):
        uid = uuid.uuid4().hex
    u = q("SELECT * FROM users WHERE id=?", (uid,), one=True)
    if not u:
        x("INSERT INTO users(id,nickname,tag,region,xp,contribute,created) VALUES(?,?,?,?,0,0,?)",
          (uid, f"{random.choice(ADJ)} {random.choice(ANIMALS)}", "#%04X" % random.randrange(0x10000), None, time.time()))
        u = q("SELECT * FROM users WHERE id=?", (uid,), one=True)
    return profile(u)


@app.patch("/api/me")
def update_me(body: dict):
    u = user(body.get("uid"))
    nickname = (body.get("nickname") or u["nickname"]).strip()[:12] or u["nickname"]
    region = body.get("region", u["region"])
    contribute = int(bool(body.get("contribute", u["contribute"])))
    x("UPDATE users SET nickname=?, region=?, contribute=? WHERE id=?", (nickname, region, contribute, u["id"]))
    return profile(user(u["id"]))


# ---- engine -----------------------------------------------------------------------------------
class EngineOffline(HTTPException):
    """The engine does not answer at all (not running), as opposed to answering that it is still loading."""


async def engine_asr(audio, priority=0, uid="", mode="", say=None):
    """The engine's 방언/표준어 answer for one clip; in 체험 모드 an example sentence (demo.asr)."""
    try:
        with GpuSlot():
            r = await http.post(f"{ENGINE}/asr", params={"priority": priority}, content=A.wav_bytes(audio),
                                headers={"content-type": "audio/wav"})
    except httpx.HTTPError as e:
        if DEMO:
            return D.asr(uid, mode, say)
        raise EngineOffline(503, "AI 엔진에 연결하지 못했어요. 잠시 후 다시 시도해 주세요.") from e
    if r.status_code == 503:
        raise HTTPException(503, "AI 모델을 불러오는 중이에요. 잠시 후 다시 시도해 주세요.")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:300])
    return r.json()


# The classifier tells 대구·경북 from 부산·울산·경남 poorly (test F1 0.53 / 0.61) but Gyeongsang from the rest
# well (0.85), so the app shows one 경상 zone: macro-F1 on speech without dialect words 0.690 -> 0.774 (5 zones).
MERGE_ZONES = {"대구·경북": "경상", "부산·울산·경남": "경상"}


async def accent_probs(audio):
    """{zone: probability} for one clip from the accent service (Gyeongsang merged), or None when it is not
    available."""
    try:
        r = await http.post(f"{ACCENT}/accent", content=A.wav_bytes(audio), headers={"content-type": "audio/wav"},
                            timeout=20)
        if r.status_code == 200:
            d = r.json()
            probs = {}
            for z, p in zip(d["zones"], d["probs"]):
                z = MERGE_ZONES.get(z, z)
                probs[z] = round(probs.get(z, 0.0) + p, 4)
            return probs
    except (httpx.HTTPError, ValueError):
        pass
    return None


async def pron_check(audio, text):
    """Sounds regional accents merge, scored in what was said (accent service /pron, server/pron.py), or None."""
    if not text:
        return None
    try:
        r = await http.post(f"{ACCENT}/pron", params={"text": text[:400]}, content=A.wav_bytes(audio),
                            headers={"content-type": "audio/wav"}, timeout=20)
        if r.status_code == 200:
            return r.json()["results"]
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return None


RUBRIC_MIN_MASS = 0.5  # below this the model did not answer with a digit: keep the coach's own number
# The last clips sent for transcription (WAV bytes by id), so the coach can listen to an answer when scoring
# 전달력: speech LLMs judging the audio rate speaking better than a text LLM reading the transcript (Ma et al.
# Interspeech 2025). About 10 s each, so 400 clips hold roughly an hour of speech (~120 MB).
CLIPS, MAX_CLIPS, LISTEN_SECONDS = OrderedDict(), 400, 90
# Off: checked 2026-10-08 on 30 plain-speech clips (a one-off check, not included), plain Gemma 4 12B's
# 전달력 from the audio did not follow the fluency measured from the same clips (Pearson r = -0.17, n = 12)
# and did not tell Seoul from Gyeongsang speakers either (39.5 vs 43.3), so it would show noise. The clips and
# the audio rubric stay ready for a model that is trained or validated for speaking assessment.
LISTEN_SCORING = False


def keep_clip(audio):
    cid = uuid.uuid4().hex[:16]
    CLIPS[cid] = A.wav_bytes(audio)
    while len(CLIPS) > MAX_CLIPS:
        CLIPS.popitem(last=False)
    return cid


def answer_audio(segments):
    """Base64 WAV of the answer's clips still kept, in order, up to LISTEN_SECONDS of speech."""
    out, total = [], 0.0
    for s in segments:
        wav = CLIPS.get(s.get("clip") or "")
        if wav and total + float(s.get("duration") or 0) <= LISTEN_SECONDS:
            out.append(base64.b64encode(wav).decode())
            total += float(s.get("duration") or 0)
    return out


async def rubric_scores(mode, ctx, sentences, llm, audio=None):
    """Replace the coach's word and facts numbers with rubric expectations (analysis.RUBRICS, engine /score).
    facts is scored only when the coach found job or technical content in the answer (its SCORE line);
    delivery (전달력) only with the answer's audio, which Gemma 4 then hears."""
    if not llm or not sentences:
        return llm, None
    crit = ["word"] + (["facts"] if llm.get("facts") is not None else []) + (["delivery"] if audio else [])
    try:
        with GpuSlot():
            r = await http.post(f"{ENGINE}/score", json={"prompts": [
                A.rubric_messages(mode, ctx, sentences, c, audio if c == "delivery" else None) for c in crit]}, timeout=120)
        if r.status_code != 200:
            return llm, None
        res = r.json()["results"]
    except (httpx.HTTPError, ValueError, KeyError, HTTPException):
        return llm, None
    out, dist = dict(llm), {}
    for c, x in zip(crit, res):
        if x["mass"] >= RUBRIC_MIN_MASS:
            out[c], dist[c] = A.expected_score(x["probs"]), x["probs"]
    return out, {"llm": llm, "probs": dist} if dist else None


async def engine_generate(messages, max_new_tokens, temperature=0.0):
    try:
        with GpuSlot():
            r = await http.post(f"{ENGINE}/generate", json={"messages": messages, "max_new_tokens": max_new_tokens,
                                                            "temperature": temperature})
    except httpx.HTTPError as e:
        raise EngineOffline(503, "AI 엔진에 연결하지 못했어요.") from e
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:300])
    return r.json()


GEMMA_RUNS = os.environ.get("SEOTTURI_GEMMA_RUNS", os.path.join(REPO, "training", "gemma4-dialect", "runs"))


def training_now():
    """Progress of the run a stage script is training (its name in <runs>/training_now), for runs the engine does
    not watch: a LoRA trained on top of a merged adapter (stages 5-7) is served only once folded with it."""
    try:
        with open(os.path.join(GEMMA_RUNS, "training_now"), encoding="utf-8") as f:
            run = f.read().strip()
        path = os.path.join(GEMMA_RUNS, run, "speed.json")
        with open(path, encoding="utf-8") as f:
            sp = json.load(f)
    except (OSError, ValueError):
        return None
    total = sp["step"] + round(sp["eta_hours"] * 3600 / max(sp["sec_per_step"], 1e-6))
    steps = sorted((int(d.rsplit("-", 1)[1]), d) for d in os.listdir(os.path.join(GEMMA_RUNS, run))
                   if d.startswith("checkpoint-") and d.rsplit("-", 1)[1].isdigit())
    if steps:
        try:
            with open(os.path.join(GEMMA_RUNS, run, steps[-1][1], "trainer_state.json"), encoding="utf-8") as f:
                total = json.load(f).get("max_steps") or total
        except (OSError, ValueError):
            pass
    return {"run": run, "step": sp["step"], "total": total, "sec_per_step": sp["sec_per_step"],
            "eta_hours": sp["eta_hours"], "updated": os.path.getmtime(path), "checkpoints": [s for s, _ in steps]}


@app.get("/api/bench")
async def bench():
    """Benchmark results of every checkpoint the update cycle has scored (engine /bench), with the progress of
    a stage script's run when the engine watches none."""
    try:
        r = await http.get(f"{ENGINE}/bench", timeout=10)
        out = r.json()
    except (httpx.HTTPError, ValueError):
        return offline_bench()
    if not out.get("training"):
        out["training"] = training_now()
    return out


ACCENT_BENCH = os.path.join(os.environ.get("SEOTTURI_ACCENT_DIR", os.path.join(REPO, "training", "accent-id")), "runs", "benchmark.json")


@app.get("/api/accent")
def accent_bench():
    """The accent classifier's experiments on its held-out tests (written by accent-id/bench_export.py)."""
    try:
        with open(ACCENT_BENCH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"items": [], "error": "no accent benchmark yet"}


@app.get("/benchmark")
def benchmark_page():
    return FileResponse(os.path.join(WEB, "benchmark.html"), headers={"Cache-Control": "no-cache"})


@app.get("/api/whoami")
def whoami(request: Request):
    """The address the rate limits key on (the caller's own), for checking the proxy setup."""
    return {"client": client_key(request)}


def offline_bench():
    """/api/bench while the engine is off: the same benchmark files it serves (state/bench)."""
    a = EF.read_state(os.path.join(STATE, "engine_state.json"))["active"]
    return {"items": EF.sorted_summaries(EF.bench_summaries(os.path.join(STATE, "bench"))), "progress": None,
            "served": a["id"] if a else None, "next_check": None, "update_hours": UPDATE_HOURS, "training": training_now(),
            "set": EF.bench_set(os.path.join(STATE, "benchmark.jsonl")), "error": "engine offline", "demo": DEMO}


def offline_status():
    """/api/status while the engine is off: the last served adapter, its score and the update history from the
    engine's files, so the landing page keeps showing them."""
    st = EF.read_state(os.path.join(STATE, "engine_state.json"))
    benches = EF.bench_summaries(os.path.join(STATE, "bench"))
    a, model = st["active"], None
    if a:
        model = {**a, "loaded_at": None, "score": EF.compact(benches[a["id"]]) if a["id"] in benches else None}
    return {"ready": False, "error": "engine offline", "demo": DEMO, "busy": None, "queue": 0, "model": model,
            "bench": [EF.compact(b) for b in EF.sorted_summaries(benches)], "bench_progress": None,
            "base": "Gemma 4 12B (google/gemma-4-12B-it)", "update_hours": UPDATE_HOURS, "next_check": None,
            "last_check": None, "history": st["history"][-12:], "training": training_now()}


@app.get("/api/status")
async def status():
    try:
        r = await http.get(f"{ENGINE}/status", timeout=5)
        s = r.json()
    except (httpx.HTTPError, ValueError):
        s = offline_status()
    if s.get("model"):
        s["model"].pop("path", None)
    s["server_time"] = time.time()
    return s


def save_contribution(audio, result, uid):
    """Opt-in only: keep the clip and the model's answer for future training data."""
    day = datetime.now().strftime("%Y%m%d")
    d = os.path.join(DATA, "contrib", day)
    os.makedirs(d, exist_ok=True)
    name = uuid.uuid4().hex
    with open(os.path.join(d, name + ".wav"), "wb") as f:
        f.write(A.wav_bytes(audio))
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"user": uid[:8], "time": time.time(), **{k: result[k] for k in
                   ("dialect", "standard", "region", "model")}}, f, ensure_ascii=False)


async def zone_reliable():
    if time.time() - _zone["at"] > 60:
        try:
            s = (await http.get(f"{ACCENT}/status", timeout=5)).json()
            _zone["ok"] = bool(s.get("ready")) and (s.get("score", {}).get("test_plain_macro_f1") or 0) >= ZONE_MIN_F1
            _zone["std"] = _zone["ok"] and bool(s.get("std_model"))
        except (httpx.HTTPError, ValueError):
            _zone["ok"] = _zone["std"] = False
        _zone["at"] = time.time()
    return _zone["ok"]


async def transcribe(data, uid="", priority=0, mode="", say=None):
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "파일이 너무 커요. 25MB 이하로 올려 주세요.")
    try:
        audio = await asyncio.to_thread(A.decode_audio, data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "오디오를 읽지 못했어요.") from e
    if len(audio) > MAX_UPLOAD_SECONDS * 16000:
        raise HTTPException(413, f"녹음은 {MAX_UPLOAD_SECONDS // 60}분 이하로 올려 주세요.")
    clip, voiced = A.preprocess(audio)
    if voiced < 0.25:
        return {"empty": True, "duration": round(len(audio) / 16000, 2)}
    chunks = A.split_audio(clip)
    clip_ids = [keep_clip(c) for c in chunks]
    reliable = await zone_reliable()
    res, probs = await asyncio.gather(asyncio.gather(*[engine_asr(c, priority, uid, mode, say) for c in chunks]),
                                      asyncio.gather(*[accent_probs(c) if reliable else asyncio.sleep(0) for c in chunks]))
    for r in res:  # an answer outside the format: keep what was said
        if not r["dialect"] and not r["standard"]:
            r["dialect"] = r["standard"] = r["raw"].strip()
        r["dialect"] = r["dialect"] or r["standard"]
        r["standard"] = r["standard"] or r["dialect"]
    prons = await asyncio.gather(*[pron_check(c, r["dialect"]) for r, c in zip(res, chunks)])
    dialect = " ".join(r["dialect"] for r in res).strip()
    standard = " ".join(r["standard"] for r in res).strip()
    for r, p in zip(res, probs):
        r["zone_probs"] = p or None
        r["region"] = max(p, key=p.get) if p else None
    zone_probs = A.mean_probs([(p, len(c)) for p, c in zip(probs, chunks)])
    zone = max(zone_probs, key=zone_probs.get) if zone_probs else None
    timing = [A.fluency_clip(c, r["dialect"]) for r, c in zip(res, chunks)]
    fluency = {k: (round(sum(t[k] for t in timing), 2) if isinstance(timing[0][k], float) else sum(t[k] for t in timing))
               for k in timing[0]}
    out = {"dialect": dialect, "standard": standard, "region": zone, "zone_probs": zone_probs, "zone_tip": A.zone_tip(zone),
           "zone_min_conf": A.ZONE_MIN_CONF, "zone_two_step": _zone["std"], "zone_regional_min": A.REGIONAL_MIN,
           "fluency": fluency, "pron": A.join_pron(res, prons),
           "clip": clip_ids[0] if len(res) == 1 else None,
           "analysis": A.diff(dialect, standard), "duration": round(len(clip) / 16000, 2),
           "model": res[0]["model"], "format": res[0]["format"],
           "latency": round(max(r["seconds"] + r["wait"] for r in res), 1), "clips": len(res),
           "demo": any(r.get("demo") for r in res)}
    if len(res) > 1:  # a long upload: keep each clip as its own sentence for the feedback
        out["parts"] = [{"dialect": r["dialect"], "standard": r["standard"], "region": r.get("region"), "zone_probs": r["zone_probs"],
                         "analysis": A.diff(r["dialect"], r["standard"]), "duration": round(len(c) / 16000, 2), "fluency": t,
                         "pron": pr, "clip": cid} for r, c, t, pr, cid in zip(res, chunks, timing, prons, clip_ids)]
    if uid and valid_uid(uid):
        u = q("SELECT contribute FROM users WHERE id=?", (uid,), one=True)
        if u and u["contribute"]:
            await asyncio.to_thread(save_contribution, clip, out, uid)
    return out


@app.post("/api/asr")
async def api_asr(request: Request, uid: str = "", mode: str = ""):
    return await transcribe(await request.body(), uid, mode=mode)


MAX_SCRIPT_BYTES, MAX_SCRIPT_CHARS = 50_000_000, 20_000  # coach.js SCRIPT_MAX_MB


@app.post("/api/script/extract")
async def script_extract(request: Request, name: str = ""):
    """A presentation script uploaded as a file (txt, docx, pdf, pptx, hwpx) -> its text. For docx, pptx and
    hwpx the browser usually sends only the text parts, as JSON {"name", "parts": {zip path: xml}}
    (coach.js zipParts): slide decks run to hundreds of MB of pictures and video."""
    too_big = HTTPException(413, f"파일이 너무 커요. {MAX_SCRIPT_BYTES // 1_000_000}MB 이하로 올려 주세요.")
    if int(request.headers.get("content-length") or 0) > MAX_SCRIPT_BYTES:
        raise too_big
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_SCRIPT_BYTES:
            raise too_big
    if not data:
        raise HTTPException(400, "빈 파일이에요.")
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = json.loads(data)
            name, data = str(body.get("name") or name), A.zip_of(body["parts"])
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(400, "파일을 읽지 못했어요.") from e
    data = bytes(data)
    try:
        text = await asyncio.to_thread(A.script_text, data, name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not text:
        raise HTTPException(400, "파일에서 글자를 찾지 못했어요. 슬라이드나 PDF가 그림으로만 되어 있으면 대본을 직접 붙여 넣어 주세요.")
    return {"text": text[:MAX_SCRIPT_CHARS], "chars": len(text), "truncated": len(text) > MAX_SCRIPT_CHARS}


@app.get("/api/tts")
async def speech(request: Request, text: str = "", persona: str = "", voice: str = "", speed: float = 1.0):
    """The persona's line as MP3, from Supertonic 3 on the CPU (server/tts.py), cached on disk."""
    text = " ".join(text.split())[:300]
    if not text:
        raise HTTPException(422, "text is required")
    if voice in TTS_VOICES:
        speed = round(min(1.3, max(0.8, speed)), 2)
    else:
        voice, speed = VOICES.get(persona, ("F1", 1.0))
    key = hashlib.sha1(f"{voice}|{speed}|{text}".encode()).hexdigest()
    path = os.path.join(TTS_CACHE, key[:2], key + ".mp3")
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg", headers=headers)
    local = request.client and request.client.host == "127.0.0.1" and "x-forwarded-for" not in request.headers
    if not local and not allow("tts", client_key(request), 60, 600):
        raise HTTPException(429, "요청이 너무 많아요. 잠시 후 다시 시도해 주세요.")
    try:
        r = await http.post(f"{TTS}/tts", json={"text": text, "voice": voice, "speed": speed}, timeout=90)
    except httpx.HTTPError as e:
        raise HTTPException(503, "음성 서버에 연결하지 못했어요.") from e
    if r.status_code != 200:
        raise HTTPException(503 if r.status_code == 503 else 502, "음성을 만들지 못했어요.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, path)
    return FileResponse(path, media_type="audio/mpeg", headers=headers)


def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# MACH (Hoque et al. UbiComp 2013): interview practice with a summary of measured behaviour after each answer,
# compared over sessions, improved expert-rated interviews in a one-week trial. Lower is better for these:
PROGRESS_LOWER = ("dialect_ratio", "accent_strength", "fillers_per_min")


def progress(uid, mode, before=None, n=3):
    """This user's measurements over their last n sessions of the mode (before a time), averaged."""
    if not (uid and valid_uid(uid)):
        return None
    rows = q("SELECT points, data FROM sessions WHERE user_id=? AND mode=? AND created<? ORDER BY created DESC LIMIT ?",
             (uid, mode, before or time.time() + 1, n))
    vals = defaultdict(list)
    for r in rows:
        d = json.loads(r["data"])
        sp, fl = d.get("speech") or {}, (d.get("speech") or {}).get("fluency") or {}
        for k, v in (("points", r["points"]), ("dialect_ratio", sp.get("dialect_ratio")),
                     ("accent_strength", sp.get("accent_strength")), ("fluency", fl.get("score")), ("rate", fl.get("rate")),
                     ("fillers_per_min", fl.get("fillers_per_min")), ("script_match", d.get("script_match"))):
            if v is not None:
                vals[k].append(v)
    return {"sessions": len(rows), "mean": {k: round(sum(v) / len(v), 1) for k, v in vals.items()},
            "lower_is_better": PROGRESS_LOWER} if rows else None


@app.post("/api/coach/feedback")
async def coach_feedback(body: dict):
    """Content coaching for one answer, streamed: {"speech"}, {"delta"}..., {"done", "feedback", "score"}."""
    mode = body.get("mode") if body.get("mode") in A.MODE_FOCUS else "interview"
    ctx = body.get("context") or {}
    segments = [s for s in body.get("segments") or [] if s.get("analysis")][:80]
    seconds = sum(float(s.get("duration") or 0) for s in segments)
    await zone_reliable()  # refreshes _zone['std'] at most once a minute
    speech = A.summarize_speech(segments, mode, accent_without_zone=_zone["std"])
    sentences = [s.get("standard") or s.get("dialect") for s in segments if s.get("standard") or s.get("dialect")]
    cov = await asyncio.to_thread(A.script_coverage, ctx["script"], sentences) if ctx.get("script") else None
    messages = A.feedback_messages(mode, ctx, sentences, speech["words"], seconds, speech, cov and cov["missed"])

    async def events():
        yield sse({"speech": speech})
        text, error, demo = "", None, False
        slot = GpuSlot()
        try:
            slot.__enter__()
        except HTTPException as e:
            yield sse({"done": True, "feedback": A.parse_feedback(""), "score": None, "error": e.detail})
            return
        try:
            async with http.stream("POST", f"{ENGINE}/generate",
                                   json={"messages": messages, "max_new_tokens": 300, "stream": True}) as r:
                if r.status_code != 200:
                    error = "AI 코치가 아직 준비되지 않았어요." if r.status_code == 503 else f"engine {r.status_code}"
                else:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        ev = json.loads(line[6:])
                        if "delta" in ev:
                            text += ev["delta"]
                            yield sse({"delta": ev["delta"]})
                        elif "error" in ev:
                            error = ev["error"]
                        elif ev.get("done"):
                            break
        except httpx.HTTPError:
            if DEMO and not text:  # the engine is off: an example of the coach's answer (server/demo.py)
                demo = True
                for piece in D.chunks(D.feedback_text(mode, sentences, speech)):
                    text += piece
                    yield sse({"delta": piece})
                    await asyncio.sleep(0.03)
            else:
                error = "AI 엔진에 연결하지 못했어요."
        finally:
            slot.__exit__()
        fb = A.parse_feedback(text)
        if demo:
            fb["demo"] = True  # saved with the session, so its history shows the label too
        if mode != "daily" and not error and not demo:
            fb["scores"], fb["rubric"] = await rubric_scores(mode, ctx, sentences, fb["scores"],
                                                             answer_audio(segments) if LISTEN_SCORING else None)
        sm = cov["score"] if cov else None
        score = None if mode == "daily" else A.final_score(speech["standard_score"], A.answer_score(
            mode, fb["scores"], sm, (speech.get("fluency") or {}).get("score")))
        tt = A.time_tip(mode, seconds) if mode != "daily" and "시간 관리" in (ctx.get("focus") or []) else None
        yield sse({"done": True, "feedback": fb, "score": score, "script_match": sm, "script_coverage": cov,
                   "time_tip": tt, "error": error, "seconds": round(seconds, 1), "demo": demo,
                   "progress": progress(body.get("uid"), mode)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/coach/reply")
async def coach_reply(body: dict):
    msgs = A.reply_messages(body.get("persona") or {}, body.get("scenario") or "", body.get("history") or [])
    try:
        r = await engine_generate(msgs, 90, temperature=0.7)
    except EngineOffline:
        if not DEMO:
            raise
        return {"text": D.reply(body.get("scenario") or "", body.get("history") or []), "seconds": 0.0, "demo": True}
    text = re.sub(r"^[^:：]{1,8}[:：]\s*", "", r["text"].strip()).strip("\"'“” ")
    return {"text": text, "seconds": r.get("seconds")}


# ---- sessions ---------------------------------------------------------------------------------
@app.post("/api/sessions")
def save_session(body: dict):
    u = user(body.get("uid"))
    mode = body.get("mode") if body.get("mode") in A.MODE_FOCUS else "interview"
    segments = [s for s in body.get("segments") or [] if s.get("analysis")][:120]
    ctx = body.get("context") or {}
    fb = body.get("feedback") or {}
    speech = A.summarize_speech(segments, mode, accent_without_zone=_zone["std"])
    seconds = sum(float(s.get("duration") or 0) for s in segments)
    sentences = [s.get("standard") or s.get("dialect") or "" for s in segments]
    cov = A.script_coverage(ctx["script"], sentences) if ctx.get("script") else None
    sm = cov["score"] if cov else None
    score = None if mode == "daily" else A.final_score(speech["standard_score"], A.answer_score(
        mode, fb.get("scores"), sm, (speech.get("fluency") or {}).get("score")))
    if mode == "daily":  # XP per utterance: a turn may span several clips
        history = body.get("history") or []
        turns = sum(1 for h in history if h.get("role") == "user") or len(segments)
        gained = 2 * turns + (10 if turns >= 10 else 0)
    else:
        gained = 10 + score["points"] // 10 + (10 if score["passed"] else 0)
    sid = uuid.uuid4().hex[:12]
    now = time.time()
    data = {"context": ctx, "segments": segments, "feedback": fb, "speech": speech, "score": score,
            "script_match": sm, "script_coverage": cov, "history": body.get("history")}
    con = db()
    with con:
        con.execute("INSERT INTO sessions(id,user_id,mode,title,created,seconds,points,passed,xp,data) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (sid, u["id"], mode, ctx.get("title", ""), now, seconds, score["points"] if score else None,
                     int(score["passed"]) if score else None, gained, json.dumps(data, ensure_ascii=False)))
        for w in speech["words"]:
            con.execute("INSERT INTO usages(user_id,session_id,dialect,standard,n,created) VALUES(?,?,?,?,?,?)",
                        (u["id"], sid, w["from"], w["to"], w["count"], now))
        for s in segments:
            if s["analysis"]["fixes"]:
                con.execute("INSERT INTO reviews(user_id,session_id,title,dialect,standard,words,status,created)"
                            " VALUES(?,?,?,?,?,?, 'active', ?)",
                            (u["id"], sid, ctx.get("title", ""), s["dialect"], s["standard"],
                             json.dumps(s["analysis"]["fixes"], ensure_ascii=False), now))
        con.execute("UPDATE users SET xp = xp + ? WHERE id=?", (gained, u["id"]))
    con.close()
    before = tier_of(u["xp"])["name"]
    after = user(u["id"])
    return {"id": sid, "xp": gained, "score": score, "user": profile(after),
            "tier_up": tier_of(after["xp"])["name"] != before}


@app.get("/api/sessions")
def list_sessions(uid: str):
    rows = q("SELECT id,mode,title,created,seconds,points,passed,xp,data FROM sessions WHERE user_id=? ORDER BY created DESC LIMIT 50",
             (user(uid)["id"],))
    for r in rows:
        d = json.loads(r.pop("data"))
        r["summary"] = (d.get("feedback") or {}).get("summary", "")
        r["dialect_ratio"] = d["speech"]["dialect_ratio"]
    return rows


@app.get("/api/sessions/{sid}")
def get_session(sid: str, uid: str):
    r = q("SELECT * FROM sessions WHERE id=? AND user_id=?", (sid, user(uid)["id"]), one=True)
    if not r:
        raise HTTPException(404, "not found")
    r["data"] = json.loads(r["data"])
    r["progress"] = progress(r["user_id"], r["mode"], before=r["created"])
    return r


def local_day(ts):
    return datetime.fromtimestamp(ts).date()


@app.get("/api/stats")
def stats(uid: str):
    u = user(uid)
    today = date.today()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    since = time.mktime(days[0].timetuple())
    idx = {d: i for i, d in enumerate(days)}
    words = [0] * 7
    for r in q("SELECT n, created FROM usages WHERE user_id=? AND created>=?", (u["id"], since)):
        words[idx[local_day(r["created"])]] += r["n"]
    practice = {m: [0.0] * 7 for m in ("interview", "presentation", "daily", "review")}
    for r in q("SELECT mode, seconds, created FROM sessions WHERE user_id=? AND created>=?", (u["id"], since)):
        practice[r["mode"]][idx[local_day(r["created"])]] += (r["seconds"] or 0) / 60
    tries, hits = [0] * 7, [0] * 7
    for r in q("SELECT ok, created FROM attempts WHERE user_id=? AND created>=?", (u["id"], since)):
        i = idx[local_day(r["created"])]
        tries[i] += 1
        hits[i] += r["ok"]
        practice["review"][i] += 0.2
    prev = q("SELECT COALESCE(SUM(n),0) AS s FROM usages WHERE user_id=? AND created>=? AND created<?",
             (u["id"], since - 7 * 86400, since), one=True)["s"]
    return {"days": [d.isoformat() for d in days], "dialect_words": words, "prev_week_avg": round(prev / 7, 1),
            "practice_minutes": {k: [round(v, 1) for v in vals] for k, vals in practice.items()},
            "review_rate": [round(100 * h / t) if t else None for h, t in zip(hits, tries)],
            "totals": {"sessions": q("SELECT COUNT(*) AS c FROM sessions WHERE user_id=?", (u["id"],), one=True)["c"],
                       "words": q("SELECT COALESCE(SUM(n),0) AS s FROM usages WHERE user_id=?", (u["id"],), one=True)["s"]}}


# ---- review -----------------------------------------------------------------------------------
@app.get("/api/reviews")
def list_reviews(uid: str, status: str = "active"):
    """Review sentences; active ones due now first (lowest predicted recall), then by when they fall due."""
    rows = q("SELECT * FROM reviews WHERE user_id=? AND status=? ORDER BY created DESC LIMIT 200",
             (user(uid)["id"], "done" if status == "done" else "active"))
    now = time.time()
    for r in rows:
        r["words"] = json.loads(r["words"] or "[]")
        r["last"] = json.loads(r["last"]) if r["last"] else None
        r.update(A.review_schedule(r["hits"], r["tries"] - r["hits"], r["practiced"], now))
    if status != "done":
        rows.sort(key=lambda r: (not r["due"], r["recall"] if r["due"] else r["next_due"]))
    return rows[:60]


@app.post("/api/reviews/{rid}/attempt")
async def review_attempt(rid: int, request: Request, uid: str):
    u = user(uid)
    r = q("SELECT * FROM reviews WHERE id=? AND user_id=?", (rid, u["id"]), one=True)
    if not r:
        raise HTTPException(404, "not found")
    res = await transcribe(await request.body(), uid, say=r["standard"])  # 체험 모드 hears the sentence as read
    if res.get("empty"):
        return {"empty": True}
    said = res["dialect"]
    error = A.cer_text(r["standard"], said)
    words = json.loads(r["words"] or "[]")
    kept = [w for w in words if w["from"] and A.contains_word(said, w["from"])]
    ok = error <= 0.2 and not kept
    now = time.time()
    plan = A.review_schedule(r["hits"] + ok, r["tries"] + 1 - r["hits"] - ok, now, now)
    last = {"said": said, "standard": res["standard"], "error": round(error, 3), "kept": kept, "ok": ok,
            "analysis": res["analysis"], "region": res.get("region"), **plan}
    con = db()
    with con:
        con.execute("UPDATE reviews SET tries=tries+1, hits=hits+?, status=?, done=?, last=?, practiced=? WHERE id=?",
                    (int(ok), "done" if plan["mastered"] else "active", now if plan["mastered"] else None,
                     json.dumps(last, ensure_ascii=False), now, rid))
        con.execute("INSERT INTO attempts(user_id,review_id,ok,created) VALUES(?,?,?,?)", (u["id"], rid, int(ok), now))
        if ok:
            con.execute("UPDATE users SET xp = xp + 5 WHERE id=?", (u["id"],))
    con.close()
    return {**last, "xp": 5 if ok else 0, "model": res["model"], "demo": res.get("demo", False)}


@app.post("/api/reviews/{rid}/reset")
def review_reset(rid: int, body: dict):
    u = user(body.get("uid"))
    x("UPDATE reviews SET status='active', done=NULL, practiced=NULL WHERE id=? AND user_id=?", (rid, u["id"]))
    return {"ok": True}


@app.post("/api/reviews/generate")
async def review_generate(body: dict):
    """Deck slide 14: sentences built from the user's own most frequent dialect words."""
    u = user(body.get("uid"))
    rows = q("SELECT dialect, standard, SUM(n) AS c FROM usages WHERE user_id=? GROUP BY dialect, standard "
             "ORDER BY c DESC LIMIT 8", (u["id"],))
    if not rows:
        raise HTTPException(409, "아직 교정 기록이 없어요. AI 코칭을 먼저 해 보세요!")
    words = [{"from": r["dialect"], "to": r["standard"], "count": r["c"]} for r in rows]
    try:
        r = await engine_generate(A.review_messages(words, u["region"]), 160, temperature=0.6)
        sents, demo = A.parse_sentences(r["text"]), False
    except EngineOffline:
        if not DEMO:
            raise
        sents, demo = D.review_sentences(words), True
    now = time.time()
    for s in sents:
        fixes = [w for w in words if A.contains_word(s["dialect"], w["from"])] or words[:1]
        x("INSERT INTO reviews(user_id,session_id,title,dialect,standard,words,status,created) VALUES(?,?,?,?,?,?, 'active', ?)",
          (u["id"], "", "맞춤 복습 문장", s["dialect"], s["standard"],
           json.dumps([{"from": w["from"], "to": w["to"]} for w in fixes], ensure_ascii=False), now))
    return {"sentences": sents, "words": words, "demo": demo}


# ---- word book --------------------------------------------------------------------------------
# data/dialect_dict.json also keeps an example sentence of the region's AI Hub speakers for each word they said
# (example_dialect, example_standard, example_form). AI Hub's terms do not allow showing its transcripts to others,
# so the site leaves them out and the word card shows the 우리말샘 definition (web/js/screens/journal.js).
AIHUB_FIELDS = ("example_dialect", "example_standard", "example_form")


def load_dict():
    global DICT
    path = os.path.join(DATA, "dialect_dict.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            DICT = [{k: v for k, v in e.items() if k not in AIHUB_FIELDS} for e in json.load(f)]


@app.get("/api/dict")
def word_book(region: str = "", term: str = Query("", alias="q"), uid: str = "", limit: int = 60, offset: int = 0):
    reg = DICT_REGION.get(region, region if region in ("강원", "경상", "전라", "제주", "충청") else "")
    items = [e for e in DICT if not reg or e["region"] == reg]
    if term:
        term = term.strip()
        items = [e for e in items if term in e["dialect"] or term in e["standard"]]
        items.sort(key=lambda e: (not e["dialect"].startswith(term), -e["count"]))
    favs = set()
    if valid_uid(uid):
        favs = {(r["region"], r["dialect"], r["standard"]) for r in
                q("SELECT region, dialect, standard FROM favorites WHERE user_id=?", (uid,))}
    offset = max(0, offset)
    out = [{**e, "favorite": (e["region"], e["dialect"], e["standard"]) in favs}
           for e in items[offset:offset + max(1, min(limit, 200))]]
    return {"region": reg, "total": len(items), "items": out,
            "tags": [e["dialect"] for e in DICT if e["region"] == reg][:8] if reg else []}


@app.get("/api/dict/favorites")
def favorites(uid: str):
    u = user(uid)
    rows = q("SELECT region, dialect, standard, created FROM favorites WHERE user_id=? ORDER BY created DESC", (u["id"],))
    index = {(e["region"], e["dialect"], e["standard"]): e for e in DICT}
    return [{**index.get((r["region"], r["dialect"], r["standard"]), r), "favorite": True, "created": r["created"]}
            for r in rows]


@app.post("/api/dict/favorite")
def set_favorite(body: dict):
    u = user(body.get("uid"))
    key = (u["id"], body.get("region") or "", body.get("dialect") or "", body.get("standard") or "")
    if body.get("on"):
        x("INSERT OR IGNORE INTO favorites(user_id,region,dialect,standard,created) VALUES(?,?,?,?,?)", key + (time.time(),))
    else:
        x("DELETE FROM favorites WHERE user_id=? AND region=? AND dialect=? AND standard=?", key)
    return {"ok": True}


@app.get("/api/dict/mine")
def my_words(uid: str):
    u = user(uid)
    return q("SELECT dialect, standard, SUM(n) AS count, MAX(created) AS last FROM usages WHERE user_id=? "
             "GROUP BY dialect, standard ORDER BY count DESC LIMIT 50", (u["id"],))


# ---- community --------------------------------------------------------------------------------
def post_row(r, uid):
    likes = q("SELECT COUNT(*) AS c FROM likes WHERE post_id=?", (r["id"],), one=True)["c"]
    comments = q("SELECT COUNT(*) AS c FROM comments WHERE post_id=?", (r["id"],), one=True)["c"]
    mine = bool(valid_uid(uid) and q("SELECT 1 FROM likes WHERE post_id=? AND user_id=?", (r["id"], uid), one=True))
    scrap = bool(valid_uid(uid) and q("SELECT 1 FROM scraps WHERE post_id=? AND user_id=?", (r["id"], uid), one=True))
    return {"id": r["id"], "board": r["board"], "author": r["author"], "tier": r["tier"], "title": r["title"],
            "body": r["body"], "created": r["created"], "views": r["views"], "likes": r["base_likes"] + likes,
            "comments": comments, "liked": mine, "scrapped": scrap, "sample": bool(r["sample"]),
            "mine": bool(uid and r["user_id"] == uid)}


@app.get("/api/posts")
def list_posts(board: str = "free", uid: str = "", sort: str = "new", term: str = Query("", alias="q")):
    term = term.strip()
    sql, args = "SELECT * FROM posts WHERE board=?", [board]
    if term:
        sql += " AND (title LIKE ? OR body LIKE ?)"
        args += [f"%{term}%", f"%{term}%"]
    rows = [post_row(r, uid) for r in q(sql + " ORDER BY created DESC LIMIT 100", args)]
    if sort == "hot":
        rows.sort(key=lambda p: -p["likes"])
    return rows


@app.get("/api/posts/hot")
def hot_posts(uid: str = ""):
    rows = [post_row(r, uid) for r in q("SELECT * FROM posts WHERE created>=? ORDER BY created DESC LIMIT 200",
                                        (time.time() - 14 * 86400,))]
    rows.sort(key=lambda p: -(p["likes"] + 3 * p["comments"]))
    return rows[:5]


@app.get("/api/scraps")
def my_scraps(uid: str):
    u = user(uid)
    return [post_row(r, uid) for r in q("SELECT p.* FROM posts p JOIN scraps s ON s.post_id=p.id WHERE s.user_id=? "
                                        "ORDER BY s.created DESC", (u["id"],))]


@app.get("/api/posts/{pid}")
def get_post(pid: int, uid: str = ""):
    r = q("SELECT * FROM posts WHERE id=?", (pid,), one=True)
    if not r:
        raise HTTPException(404, "not found")
    x("UPDATE posts SET views=views+1 WHERE id=?", (pid,))
    p = post_row(r, uid)
    p["views"] += 1
    p["comment_list"] = [{**c, "mine": bool(uid and c["user_id"] == uid), "user_id": None}
                         for c in q("SELECT * FROM comments WHERE post_id=? ORDER BY created", (pid,))]
    return p


@app.post("/api/posts")
def create_post(body: dict):
    u = user(body.get("uid"))
    title, text = (body.get("title") or "").strip()[:60], (body.get("body") or "").strip()[:3000]
    if not title or not text:
        raise HTTPException(422, "제목과 내용을 입력해 주세요.")
    board = body.get("board") if body.get("board") in ("free", "qna") else "free"
    pid = x("INSERT INTO posts(board,user_id,author,tier,title,body,created) VALUES(?,?,?,?,?,?,?)",
            (board, u["id"], u["nickname"], tier_of(u["xp"])["name"], title, text, time.time()))
    return {"id": pid}


@app.delete("/api/posts/{pid}")
def delete_post(pid: int, uid: str):
    u = user(uid)
    x("DELETE FROM posts WHERE id=? AND user_id=?", (pid, u["id"]))
    return {"ok": True}


@app.post("/api/posts/{pid}/like")
def like_post(pid: int, body: dict):
    u = user(body.get("uid"))
    if q("SELECT 1 FROM likes WHERE post_id=? AND user_id=?", (pid, u["id"]), one=True):
        x("DELETE FROM likes WHERE post_id=? AND user_id=?", (pid, u["id"]))
    else:
        x("INSERT INTO likes(post_id,user_id) VALUES(?,?)", (pid, u["id"]))
    r = q("SELECT * FROM posts WHERE id=?", (pid,), one=True)
    return post_row(r, u["id"])


@app.post("/api/posts/{pid}/scrap")
def scrap_post(pid: int, body: dict):
    u = user(body.get("uid"))
    if q("SELECT 1 FROM scraps WHERE post_id=? AND user_id=?", (pid, u["id"]), one=True):
        x("DELETE FROM scraps WHERE post_id=? AND user_id=?", (pid, u["id"]))
    else:
        x("INSERT INTO scraps(post_id,user_id,created) VALUES(?,?,?)", (pid, u["id"], time.time()))
    return post_row(q("SELECT * FROM posts WHERE id=?", (pid,), one=True), u["id"])


@app.post("/api/posts/{pid}/comments")
def add_comment(pid: int, body: dict):
    u = user(body.get("uid"))
    text = (body.get("body") or "").strip()[:800]
    if not text:
        raise HTTPException(422, "댓글을 입력해 주세요.")
    post = q("SELECT user_id FROM posts WHERE id=?", (pid,), one=True)
    if not post:
        raise HTTPException(404, "not found")
    author = u["nickname"] + (" (글쓴이)" if post["user_id"] == u["id"] else "")
    x("INSERT INTO comments(post_id,user_id,author,tier,body,created,parent) VALUES(?,?,?,?,?,?,?)",
      (pid, u["id"], author, tier_of(u["xp"])["name"], text, time.time(), body.get("parent")))
    return get_post(pid, u["id"])


# ---- accent test set (/record) -----------------------------------------------------------------
# Team members with a known home region read the same standard sentences (web/assets/testset.json) and
# talk freely through the site's own recorder, browser noise suppression and all, so the accent models
# can be scored on what the app really receives (accent-id/eval_browser.py). Only reachable with the key
# in data/testset_key; each take is kept as data/testset/<session>/<item>.wav with its browser clip cuts.
TESTSET = os.path.join(DATA, "testset")
TESTSET_KEY = os.path.join(DATA, "testset_key")
TESTSET_ITEMS, MAX_TESTSET_BYTES = 40, 12_000_000
SID_RE, ITEM_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$"), re.compile(r"^(s\d{2}|f\d)$")


def testset_allowed(key):
    try:
        with open(TESTSET_KEY, encoding="utf-8") as f:
            want = f.read().strip()
    except OSError:
        return False
    return bool(want) and hmac.compare_digest(key.encode(), want.encode())


@app.post("/api/testset/session")
async def testset_session(request: Request, key: str = ""):
    if not testset_allowed(key):
        raise HTTPException(404)
    body = await request.json()
    if not body.get("consent"):
        raise HTTPException(400, "녹음 저장에 동의해 주세요.")
    meta = {k: str(body.get(k) or "")[:80] for k in ("name", "birthplace", "raised", "current", "years", "age", "sex", "self_rating")}
    if not meta["raised"] or not meta["current"]:
        raise HTTPException(400, "자란 곳과 지금 사는 곳을 골라 주세요.")
    sid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    os.makedirs(os.path.join(TESTSET, sid))
    with open(os.path.join(TESTSET, sid, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "consent": True, "created": time.time(), "version": body.get("version"),
                   "user_agent": request.headers.get("user-agent", "")[:300]}, f, ensure_ascii=False, indent=1)
    return {"sid": sid}


@app.post("/api/testset/clip")
async def testset_clip(request: Request, key: str = "", sid: str = "", item: str = "", segments: str = "[]"):
    if not testset_allowed(key):
        raise HTTPException(404)
    d = os.path.join(TESTSET, sid)
    if not SID_RE.match(sid) or not ITEM_RE.match(item) or not os.path.isdir(d):
        raise HTTPException(400, "잘못된 녹음 정보예요.")
    data = await request.body()
    if len(data) > MAX_TESTSET_BYTES:
        raise HTTPException(413, "녹음이 너무 길어요. 2분 안으로 해 주세요.")
    if data[:4] != b"RIFF" or A.read_wav(data) is None:
        raise HTTPException(400, "오디오를 읽지 못했어요.")
    if len([n for n in os.listdir(d) if n.endswith(".wav")]) >= TESTSET_ITEMS and not os.path.exists(os.path.join(d, item + ".wav")):
        raise HTTPException(400, "이 세션은 녹음이 가득 찼어요.")
    try:
        cuts = [[round(float(a), 3), round(float(b), 3)] for a, b in json.loads(segments)[:200]]
    except (ValueError, TypeError):
        cuts = []
    with open(os.path.join(d, item + ".wav"), "wb") as f:
        f.write(data)
    with open(os.path.join(d, item + ".json"), "w", encoding="utf-8") as f:
        json.dump({"item": item, "segments": cuts, "bytes": len(data), "time": time.time()}, f)
    return {"ok": True}


@app.get("/record")
def record_page():
    return FileResponse(os.path.join(WEB, "record.html"), headers={"Cache-Control": "no-cache"})


# ---- pages ------------------------------------------------------------------------------------
@app.get("/app")
def app_page():
    return FileResponse(os.path.join(WEB, "app.html"), headers={"Cache-Control": "no-cache"})


@app.get("/")
def landing_page():
    return FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=WEB), name="web")


def main():
    init_db()
    load_dict()
    # load the script-coverage encoder now rather than on the first presentation's feedback
    threading.Thread(target=A.script_coverage, args=("서투리 발표를 시작하겠습니다.", ["발표를 시작합니다"]), daemon=True).start()
    host = os.environ.get("SEOTTURI_HOST", "127.0.0.1")
    port = int(os.environ.get("SEOTTURI_PORT", "8787"))
    cert, key = os.environ.get("SEOTTURI_SSL_CERT"), os.environ.get("SEOTTURI_SSL_KEY")
    uvicorn.run(app, host=host, port=port, log_level="warning",
                **({"ssl_certfile": cert, "ssl_keyfile": key} if cert and key else {}))


if __name__ == "__main__":
    main()
