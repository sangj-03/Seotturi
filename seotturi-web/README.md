# 서투리 웹사이트

The 서투리 service as a working website: the poster's design as the landing page, the final
deck's screens as the app, and the Gemma 4 dialect model that is still training
(`/data/seotturi/gemma4-dialect`) as the AI coach. The model listens to the user's voice,
writes down what was said and converts it to standard Korean; a separate accent classifier
(`/data/seotturi/accent-id`, Whisper-large-v3 encoder + a small head) tells from the sound which
accent zone the speaker comes from, also when every word is standard. The app turns that into
corrections, scores and practice.

```bash
cd /data/seotturi/seotturi-web
./run.sh start          # engine (GPU, ~5 min to load) + web server
./run.sh status         # served adapter, its validation score, next update check
./run.sh restart-web    # after editing server/app.py or web/ (the model stays loaded)
./run.sh stop
```

Open http://localhost:8787 (landing page with the app in a phone frame) or
http://localhost:8787/app (the app alone, phone layout). From elsewhere: through Tailscale Funnel
(see [Public access](#public-access-tailscale-funnel)). Browsers only allow the microphone on `localhost` or HTTPS.

**Running unattended.** The servers do not depend on a terminal or a login: `run.sh` starts each one with
`setsid nohup` under `scripts/supervise.sh`, which restarts it after a crash, and logind keeps
user processes after logout (`KillUserProcesses=no`). To start them after every reboot, add a crontab line
`@reboot sleep 60 && cd /data/seotturi/seotturi-web && ./run.sh start >> logs/boot.log 2>&1`; `tailscaled`
restores the Funnel by itself. The AI answers about 5 minutes after the engine starts.

Start and restart the servers from cron (a one-off crontab line), not from a desktop terminal: on 2026-10-12 a
memory peak made systemd-oomd stop the whole scope of the terminal the engine had been restarted from, engine
and supervisor included, while the servers started by cron kept running.

## How it fits together

| Process | Port | What it does |
| --- | --- | --- |
| `server/tts.py` (`.venv-tts`) | 127.0.0.1:8789 | Supertonic 3 speech synthesis on the CPU (4 threads, nice 5) |
| `server/accent.py` | 127.0.0.1:8790 | accent-zone classifier (Whisper-large-v3 encoder + head, `/data/seotturi/accent-id`), GPU, 5 % memory cap |
| `server/engine.py` | 127.0.0.1:8788 | Holds Gemma 4 12B + the dialect LoRA on the GPU. `POST /asr` (WAV, at most 30 s) answers `지역/방언/표준어`; `POST /generate` is plain Gemma 4 (adapter switched off) for coaching text. One worker thread owns the GPU; waiting clips are batched (up to 4). |
| `server/app.py` | 127.0.0.1:8787 | Website (`web/`), the app's API (`/api/*`), SQLite in `data/app.db`. Splits long uploads at pauses, scores answers, streams the coach's feedback. |

While recording, a cancel button (✕, left of the mic) in the session, the daily conversation and the review
sheet drops everything said in that attempt, including clips still being transcribed, and starts over
(`Recorder.cancel()` in `web/js/recorder.js`).

The browser cuts the recording at pauses (700 ms; at most 15 s per clip, since the model was
trained on 1–20 s utterances) and sends each clip as soon as the speaker pauses, so the first
sentences are already transcribed while the user is still talking.

### The model and its 2-hour updates

- At start the engine serves the last promoted adapter (`state/engine_state.json`; `lora_s6c@final` since
  2026-10-11), or `SEOTTURI_FALLBACK` (`runs/lora_v1/final`, two-line format) when there is none.
- Every 2 hours (`SEOTTURI_UPDATE_HOURS`) it looks for the newest complete checkpoint of the runs in
  `SEOTTURI_WATCH` (default `lora_r64c:basic,lora_s6c:basic`; a checkpoint counts once `trainer_state.json`
  exists and has been quiet for a minute, `final` once `train_args.json` exists). The Trainer keeps only a few
  checkpoints, so the candidate is first copied to `state/adapters/`.
- The candidate is loaded next to the served adapter and both are scored on the fixed benchmark
  (below). It replaces the served adapter only when it is worse on neither the dialect nor the
  standard-Korean CER (`SEOTTURI_GATE_TOLERANCE`, 0 since 2026-10-07: an earlier 0.02 tolerance on the
  standard line alone let lora_v2r@1000 replace lora_v1 with a worse dialect CER, so v1 was put back) and a
  paired blockwise bootstrap over recordings (`scripts/bench_score.py --compare`) gives the two lines' summed CER
  a probability of being better of at least `SEOTTURI_GATE_CONFIDENCE` (0.95); otherwise it is recorded as
  kept-back and the next checkpoint is tried 2 hours later. Scores are cached per adapter, so a check only
  costs the candidate's scoring. The served adapters `lora_r64c` and `lora_s6c` were promoted by hand after
  the 4,000-utterance test set in gemma4-dialect, since their gain on the 160 benchmark sentences was too small
  for the gate.
- Swapping is a PEFT `load_adapter` / `set_adapter` / `delete_adapter` on the loaded model
  (about 1.5 s); no restart, requests keep being served. User requests always run before
  scoring batches.
- The history (time, candidate, both scores, decision) is in `state/engine_state.json`, on the
  landing page's AI Engine section, and in the app's bell.

Runs and formats are configurable: `SEOTTURI_WATCH=lora_v2r:region` (comma-separated, earliest
stage first; add the next stage's run here), `SEOTTURI_FALLBACK=lora_v1:basic`.

### Benchmark (every update cycle)

Each checkpoint the update cycle looks at is benchmarked once on a fixed set
(`state/benchmark.jsonl`, `scripts/build_benchmark.py`, seed 20261006), and the result decides the
swap (neither CER worse than the served adapter's, format rate not lower). Results are
in `state/bench/<id>.json` (summary) and `.jsonl` (every prediction), shown at `/benchmark` and
served by `/api/bench`.

- **Set:** 160 utterances of AI Hub's Validation split (never trained on), 40 per regional set,
  32 with dialect words + 8 without, one per recording (157 recordings); 105 plain utterances
  (no dialect words) from the intonation test set, 15 per accent zone, for zone identification.
- **Metrics** (`scripts/bench_score.py`, run in `.venv-tts` for sacreBLEU): corpus-level CER of the
  dialect and the standard line (spaces and punctuation removed; KsponSpeech / K-Wav2vec 2.0),
  WER and space-normalised sWER (Bang et al. 2020), chrF++ of the standard line (Popović 2017,
  sacreBLEU signature kept; Post 2018), ERR = word-error reduction of the standard line relative
  to leaving the model's own dialect transcript as is (van der Goot et al. 2021; Kuparinen et al.
  2023), dialect-word correction recall, zone accuracy and macro-F1 on plain speech (AESRC 2020;
  NADI), 95% confidence intervals by blockwise bootstrap over recordings (Bisani & Ney 2004;
  Liu & Peng 2020), RTF.
- **Baselines:** Gemma 4 without the adapter and `lora_v1` were scored from the training
  project's `evaluate.py` predictions restricted to the same 160 utterances (no GPU needed).
- **Cost:** about 20 minutes of GPU per checkpoint (265 clips, batches of 8); the training run's
  steps take about twice as long meanwhile, so roughly 10 minutes of training per 2-hour cycle.
- **Accent classifier (Whisper) section** (`/benchmark#accent`, added 2026-10-07): `/api/accent` serves
  `accent-id/runs/benchmark.json`, written by `accent-id/bench_export.py` (run it after each new accent run):
  every experiment re-scored from its test logits on the four held-out tests as it was or would be served
  (ensemble weights, standard bonus), 95% intervals by bootstrap over recordings, Seoul-Gyeonggi false alarms
  and missed regional accents, the learning curve, and Gemma's `지역:` line on the same 90 bench clips.

### Sharing the GPU with the training run

Training keeps the GB10 at ~96%, so serving is slower than it will be after training:
measured 9–20 s for a 3–6 s clip and ~3 tokens/s for coaching text (a full interview answer's
feedback takes 40–60 s). The engine takes ~25 GB, caps itself at 40 GB
(`SEOTTURI_ENGINE_MEM_GB`), and sets `oom_score_adj=1000` so that if memory ever runs out the
kernel kills the engine, not the training run. Requests slow training down only while they run.

**Loading the engine while the training runs can exhaust memory** (2026-10-07): loading Gemma 4 briefly needs
~20 GB on top of the engine's ~25 GB, and with stage 3 training (~40 GB), the driver project's vLLM (~8 GB)
and the accent service (~3 GB) up, `./run.sh restart-engine` drove memory pressure to 99% and the kernel
killed the half-loaded engine. Order that works: stop the training
(`kill -- -$(ps -o sid= -p $(pgrep -f 'bash ./keep_stage3_region.sh') | tr -d ' ')` in gemma4-dialect), restart the
engine, wait for `loaded ...` in logs/engine.log, then start `keep_stage3_region.sh` again (it resumes from the
newest checkpoint). The training's own loading peak (~43 GB) fits next to a loaded engine.

## Scoring (from the deck)

- 사투리 비율 = (changed 어절 + accent 어절) × 100 / all 어절 of the model's 방언 line; 표준어 점수 = 100 − 사투리 비율.
  Accent 어절 (user's choice, 2026-10-07): when the answer's accent zone is not 서울·경기, every clip the accent
  classifier places outside 서울·경기 counts as dialect in full, i.e. its 어절 that are not dialect words already
  (`analysis.accent_sentences`). On the accent tests the served model (bonus 1.25) places 33-35% of 서울·경기 clips
  outside 서울·경기, so a standard speaker can lose points this way; raising `served_standard_bias` lowers that
  (2.25: 19-25%) at the cost of missing more regional accents (detected 69-84% instead of 83-92%).
- 답변 점수 from the coach (plain Gemma 4): word choice × 0.5 + fluency × 0.5, or
  word × 0.25 + fluency × 0.25 + facts × 0.5 when the answer carries job/technical facts.
  For presentations with a script: word × 0.25 + fluency × 0.25 + 대본 일치도 × 0.5.
- 면접 점수 = 표준어 점수 + 답변 점수 (200, pass 140), shown out of 100.
- XP: 10 + score/10 (+10 when passed) per interview or presentation; 2 per utterance in daily
  conversation (+10 at 10 utterances); 5 per review sentence read correctly. Tiers: 사투리 마스터 0,
  사투리 덜스터 150, 표투리 350, 표준어 덜스터 600, 표준어 마스터 1000.

The coach never hears the audio; it gets the standard-Korean sentences plus the dialect ratio
and the accent zone, and is told not to judge pronunciation beyond that.

**Accent zone.** Gemma's own `지역:` line names the zone mostly from the words: on speech without
dialect words it reaches macro-F1 0.216 (7 zones), so the app never shows it. The zone comes instead
from `server/accent.py` (127.0.0.1:8790), which serves the classifier trained in
`/data/seotturi/accent-id` (six zones: 서울·경기 = standard, 강원, 충청, 대구·경북, 부산·울산·경남, 제주;
Jeolla is not covered; app.py merges the two Gyeongsang zones into 경상, which the classifier tells apart poorly). app.py sends every clip to it alongside the Gemma transcription, averages the
zone probabilities over an answer by length, and shows a zone from the first sentence, however short
(`MIN_ZONE_SECONDS` in server/analysis.py, 0 since 2026-10-07; it was 8 s), when the served
classifier's held-out macro-F1 on speech without dialect words reaches `SEOTTURI_ZONE_MIN_F1` (0.5);
otherwise no zone is shown, as before. The served run is the link `accent-id/runs/served`; switch it and
run `./run.sh restart-accent` (seconds, the Gemma engine keeps running).

## App screens (deck flow)

The app is designed for a phone, and inside the landing page's phone frame (`/app?frame=1`) it
stays that way. Opened on its own in a window at least 900 px wide, it fills the window instead
(the `@media (min-width: 900px)` block at the end of `css/app.css`): the tab bar becomes a left
sidebar that stays on every screen, pages sit in a centred column (1,040 px, forms 820 px), word
cards spread into several columns, and the home map stays at the right at its own resolution.
`app.html` and `index.html` load their CSS and JS with a `?v=` tag; bump it after changing them so
browsers do not keep an old copy.

| Tab | Screens |
| --- | --- |
| 홈 | Profile card (tier, nickname, XP), XP gauge, journey map (one stop per tier from 부산 to 서울; every stop, locked ones too, opens in the card below with what it means and the XP still needed; ‹ ›, swipe or a tap on a stop pans the map there), 학습 그래프 (사투리 지수 = corrected 어절 per day, 연습 로그 = minutes per mode, 복습 성공률), bell with model updates |
| AI 코칭 | 새 프로젝트 → 모드 (면접 / 발표 / 일상회화; 발표 대본은 붙여 넣거나 txt·docx·pdf·pptx·hwpx 파일로 올림, `/api/script/extract`: pptx는 발표자 노트 우선) → 세부 카테고리 → 피드백 항목 → 성우·배경 → preparation → session (live transcript with dialect words marked) → feedback (개선필요 cards: 📄 content, 👥 dialect) → 면접 결과 (합격/불합격, score, XP, PNG download) |
| 학습록 | 단어 도감 (85,557 entries: every 지역어 headword of 국립국어원 우리말샘 for 강원, 경상, 전라, 제주 and 충청, with how often AI Hub speakers of that region said it and one of their sentences, or the dictionary definition when they never did; casual forms said everywhere, such as 쫌 and 걍, are removed by comparing with Seoul-born speakers; `scripts/build_dict.py`; favourites; "내가 쓴 사투리"), 구문 복습 (read your corrected sentences aloud; Gemma 4 checks them; 맞춤 복습 문장 generated from your own dialect words), 피드백 모아보기 |
| 커뮤니티 | 자유게시판 / Q&A게시판 / 내 스크랩, 이주의 인기글, posts, likes, scraps, threaded comments. The posts from the deck are seeded and marked 예시. |

The interviewer, audience and conversation partner speak their lines aloud. `server/tts.py` runs
Supertonic 3 (Supertone, OpenRAIL; 10 preset voices, Korean) on the CPU in its own venv
(`.venv-tts`, model in /data/seotturi/models/supertonic3): about 1.6 s per line, MP3 at 48 kbps,
cached in `data/tts_cache/` by `/api/tts` (the fixed interview questions and openers are
pre-generated). Each persona's voice and speed are in `web/js/personas.js`, picked by measured pitch
(median F0 per preset in its header); `/api/tts` takes them as `voice`/`speed` and falls back to
`VOICES` in app.py for pages that send only a persona name. `web/js/voice.js` plays them, stops
as soon as the user records, has replay buttons and a switch in the profile, and falls back to the
browser's speech synthesis if the server cannot answer. Cache misses count against a 60 per 10
minutes per-client limit.

Conversation partners (`web/js/personas.js`): 5 interviewers, 3 audiences, and for 일상회화 a cast per
situation, so the partner fits the scene: 친구와 수다 (친구 지민·도윤·서진), 카페에서 주문 (바리스타 하은,
바쁜 알바생 준호, 꼼꼼한 카페 사장님), 길 묻기 (친절한 대학생, 바쁜 직장인, 동네 토박이 할머니), 직장 동료와 대화
(입사 동기 수아, 꼼꼼한 팀장님, 신입 후배 민재), 자유 주제 (말동무 서투리 + the friends). Each has a card text, an
`act` line that tells Gemma how to play them (`/api/coach/reply`), an opener, a voice, a scene (`img`, shown
behind the preparation and the session) and a card avatar (`face`). The scenes in `web/assets/personas/`
were drawn on the DGX with Qwen-Image-Edit-2511 (Apache-2.0) plus its 8-step Lightning LoRA, each edited from
one of the app's own character images (friend, interviewer, mascot) so the 3D style matches; the pipeline
and prompts are in `/data/seotturi/img-gen` (`jobs.json`, `touchup.json`, `gen.py` stores the 20B
transformer as FP8 to stay near 25 GB on the shared GPU; `finalize.py` writes the scene and avatar,
`specs_block.py` the eye and mouth boxes for `scripts/alive_frames.py`). 깐깐한 부장님, 우호적인 동료들 and
도윤 keep the original interviewer, audience and friend images.

The characters move (`web/js/alive.js`, `web/css/alive.css`, in the app and on the landing page).
Every character image blinks every few seconds, each person in the audience on their own; the
session's interviewer, audience or friend moves their mouth while the line plays (`voice.js` fires a
`voice` event); a tap makes a character hop and chatter; the landing page mascot waves and says its
speech-bubble line when it scrolls into view; clouds and birds cross the home map; a passed result
gets confetti. `scripts/alive_frames.py` draws the frames (eye and mouth boxes per image in
`SPECS`) and saves only the changed rectangle of each, 190 KB in all, in `web/assets/alive/`; the
browser pastes them onto the original, so a frame differs from the image nowhere else. Run it again
after replacing a character image. Nothing moves when the system asks for reduced motion.

Users are anonymous: a random id in the browser's localStorage, no login.

## Data and privacy

Audio is decoded and analysed in memory and dropped. Only when a user turns on "내 음성으로
서투리 AI 키우기" in the profile are clips saved to `data/contrib/YYYYMMDD/` (WAV + the model's
answer, user id truncated), as candidates for later training data. Text results (transcripts,
feedback, scores) are stored per user in `data/app.db`.

## Public access (Tailscale Funnel)

The web server itself listens on localhost only. From 2026-10-06 it was published **to the whole
internet** through Tailscale Funnel, with Tailscale's Let's Encrypt certificate:

    sudo tailscale funnel --bg 8787        # https://<machine>.<tailnet>.ts.net

The engine (8788) is never exposed; only the web server is. Because anyone with the link can use
the GPU that the training run needs, `server/app.py` limits use per client (the first
`X-Forwarded-For` address, which the Tailscale proxy sets to the visitor's address), per 10 minutes:
90 speech clips, 30 coach calls, 20 posts/comments, 20 new users, 300 other writes; at most 24
engine calls in flight for everyone (`SEOTTURI_MAX_INFLIGHT`); uploads up to 25 MB / 3 minutes;
ffmpeg only reads the upload itself (`-protocol_whitelist pipe`) with common audio demuxers.

```bash
# back to tailnet-only
sudo tailscale funnel --https=443 off && sudo tailscale serve --bg 8787
# close completely
sudo tailscale serve reset
# community moderation (posts are anonymous)
/data/seotturi/.venv/bin/python -I scripts/moderate.py posts | comments | delete-post ID | delete-comment ID
```

Note that the word book shows example sentences taken verbatim from the AI Hub transcripts;
check the AI Hub data terms before keeping them on a public site.

## Static copy (GitHub Pages)

So the site can be visited while this machine is off, the landing page and the benchmark page
can also be published as a static site on the `gh-pages` branch of this repository (with Pages turned on for
that branch: https://sangj-03.github.io/Seotturi/).

```bash
python3 scripts/build_static.py            # build into static_site/
SEOTTURI_LIVE_URL=https://<machine>.<tailnet>.ts.net \
python3 scripts/build_static.py --publish  # build and force-push one commit to gh-pages (only if the site changed)
```

- **Live or saved data.** `scripts/static/shim.js` answers the pages' `/api/status`,
  `/api/bench` and `/api/accent` calls from this server (`--live`) when it replies within 4 s, and otherwise from
  the snapshot in `data/*.json`, shown as "AI 서버 꺼짐 · … 기록 기준". Without a live address the copy shows the snapshot only. The snapshot is taken from
  `127.0.0.1:8787` at build time; if the server is down then, the last one (`state/static_snapshot/`) is reused.
- **The app.** The app needs the GPU, so the phone frame on the landing page loads the live app
  only when this server answers, and otherwise shows a notice; "앱 열기" links go to the live server.
- **Cross-origin.** `server/app.py` sends `Access-Control-Allow-Origin` for `GET /api/status`,
  `GET /api/bench` and `GET /api/accent` to `SEOTTURI_STATIC_ORIGINS` (default `https://sangj-03.github.io`) only.
- **Updates.** Run `--publish` again (for example from cron) to refresh it; timestamps alone do not trigger a
  push, so the branch keeps a single commit and changes only when results or training progress change.

## Files

| Path | Purpose |
| --- | --- |
| `server/engine.py` | GPU engine, batching, adapter hot-swap, update gate |
| `server/app.py` | web server and API |
| `server/tts.py` | speech service (Supertonic 3) |
| `server/accent.py` | accent-zone service (classifier from `/data/seotturi/accent-id`) |
| `server/analysis.py` | word-level corrections, scores, fluency, script coverage, coach prompts and parsing, audio helpers |
| `server/pron.py` | pronunciation check (phone recognition and forced alignment, served by `accent.py` at `/pron`) |
| `web/index.html`, `web/css/landing.css`, `web/js/landing.js` | landing page (poster) |
| `web/app.html`, `web/css/app.css`, `web/js/` | the app (deck): `recorder.js` (16 kHz capture, pause detection), `voice.js` (spoken questions), `screens/*.js` |
| `web/js/alive.js`, `web/css/alive.css` | blinking, talking and waving characters, idle motion |
| `web/js/personas.js`, `web/assets/personas/` | interviewers, audiences, the daily-conversation cast per situation; their scenes and avatars |
| `scripts/extract_assets.py` | cuts logo, mascot, scenes out of the poster and deck into `web/assets/` |
| `scripts/alive_frames.py` | blink and talk frames and the waving hand, into `web/assets/alive/` |
| `scripts/fetch_opendict.py`, `scripts/build_dict.py` | 우리말샘 dialect headwords (API key in `~/.opendict_key`) and the word book `data/dialect_dict.json` built with the AI Hub labels |
| `scripts/moderate.py` | list and delete community posts and comments |
| `scripts/build_benchmark.py`, `scripts/bench_score.py` | benchmark set and metrics |
| `web/benchmark.html`, `web/js/benchmark.js`, `web/css/benchmark.css` | benchmark page |
| `web/record.html`, `web/js/record.js`, `web/assets/testset.json` | accent test set recording page (`/record?key=…`, key in `data/testset_key`) |
| `scripts/build_static.py`, `scripts/static/` | static copy of the landing and benchmark pages for GitHub Pages |
| `scripts/supervise.sh` | restarts a server when it exits (used by `run.sh`) |
| `state/`, `data/`, `logs/` | engine state and adapter snapshots, app database and word book, logs (not in Git) |
