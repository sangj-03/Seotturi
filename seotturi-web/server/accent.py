"""Accent zone of a speaker from how they sound, for the 서투리 app.

    /data/seotturi/.venv/bin/python server/accent.py        (127.0.0.1:8790, started by run.sh)

POST /accent   a WAV clip (16 kHz mono, 16-bit) -> {"zones": [...], "probs": [...], "model", "seconds"}
POST /pron?text=...   a WAV clip and what was said -> {"results": [...]} (pron.py: sounds regional accents merge)
GET  /status   -> {"ready", "model", "zones", "score", "pron"}

The classifier is trained in /data/seotturi/accent-id (README there): Whisper-large-v3's encoder plus a
small head, six zones (서울·경기 = the standard accent, 강원, 충청, 대구·경북, 부산·울산·경남, 제주; Jeolla
is not covered). Gemma 4 keeps transcription and feedback; this service only answers "which zone does
this voice sound like". The served run is accent-id/runs/served (a link to the chosen run), and "score" is
that run's held-out result, which app.py compares with SEOTTURI_ZONE_MIN_F1 before showing any zone.

A process of its own, so a better classifier can be swapped in with ./run.sh restart-accent while the
Gemma engine keeps running. One clip at a time on the GPU (about 0.1 s each), with a small memory cap
because the GPU is shared with the engine.
"""

import io
import json
import logging
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

os.environ.setdefault("HF_HOME", "/data/huggingface")  # where whisper-large-v3 is cached (run.sh runs offline)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ACCENT_DIR = os.environ.get("SEOTTURI_ACCENT_DIR", "/data/seotturi/accent-id")
RUN = os.environ.get("SEOTTURI_ACCENT_RUN", os.path.join(ACCENT_DIR, "runs", "served"))
# A standard-vs-regional detector (accent-id region4.py --std2), used when this link exists. With it, a run
# without a 서울·경기 zone (region4: 강원/충청/경상/제주) answers P(zone) = P(regional) * P(zone | regional) and
# 서울·경기 = 1 - P(regional), so the app reads the same six-zone-style probabilities as before.
STD_RUN = os.environ.get("SEOTTURI_ACCENT_STD_RUN", os.path.join(ACCENT_DIR, "runs", "served_std"))
PORT = int(os.environ.get("SEOTTURI_ACCENT_PORT", "8790"))
GPU_SHARE = float(os.environ.get("SEOTTURI_ACCENT_GPU_SHARE", "0.05"))
# Added to the standard zone's log-probability. Telling a standard speaker they have an accent is the worse
# mistake for a coach, and the classifier confuses 서울·경기 mostly with 충청. The value belongs to the served
# run: its ensemble.json "served_standard_bias" (accent-id runs/ensemble_big: +1.25 keeps Seoul-Gyeonggi
# test_plain clips called regional at 34.5 %, as runs/ensemble had with +1.0, with 8.2 % of regional clips
# called standard), else 1.0; SEOTTURI_ACCENT_STANDARD_BIAS overrides both.
STANDARD_BIAS = 1.0  # set in load()
STD_BIAS = 0.0  # the detector's own bonus on 서울·경기 (its ensemble.json "served_standard_bias", calibrate_std.py)
MAX_BYTES = 2_000_000  # 60 s of 16 kHz 16-bit audio

log = logging.getLogger("accent")
lock = threading.Lock()
state = {"ready": False, "error": None, "model": None, "score": {}, "zones": [], "pron": False, "std_model": None}
enc = None  # the one frozen Whisper encoder every head reads (common.Tree); LoRA runs keep their own
clf = None  # common.Tree of the served run
std = None  # common.Tree of the standard detector, or None
zones_in = []  # the served run's own zones
phones = None  # pron.Scorer, loaded after the classifier


def test_metrics(run):
    """(test_plain, bench) entries of a run's metrics.json (region4 / std2 runs prefix their split names)."""
    with open(os.path.join(run, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    pick = lambda end: next(v for k, v in m.items() if k.endswith(end) and isinstance(v, dict))
    return pick("test_plain"), pick("bench")


def run_bias(run, default):
    spec = os.path.join(run, "ensemble.json")
    if not os.path.exists(spec):
        return default
    with open(spec, encoding="utf-8") as f:
        return float(json.load(f).get("served_standard_bias", default))


def detector_auc(run):
    """Held-out standard-vs-regional AUC of a detector: metrics.json, else accent-id conditions.py --binary."""
    try:
        return test_metrics(run)[0]["standard_vs_regional_auc"]
    except (OSError, StopIteration, KeyError):
        try:
            with open(os.path.join(run, "conditions_binary.json"), encoding="utf-8") as f:
                return json.load(f)["conditions"]["원본"]["auc"]
        except (OSError, KeyError, ValueError):
            return None


def load():
    global enc, clf, std, zones_in, STANDARD_BIAS, STD_BIAS
    try:
        sys.path.insert(0, ACCENT_DIR)
        import torch
        import common
        torch.cuda.set_per_process_memory_fraction(GPU_SHARE)
        t0 = time.time()
        enc = common.Encoder()
        run = os.path.realpath(RUN)
        clf = common.Tree(run, enc)
        zones_in = clf.zones
        tp, bench = test_metrics(run)
        STANDARD_BIAS = (float(os.environ.get("SEOTTURI_ACCENT_STANDARD_BIAS", run_bias(run, 1.0)))
                         if "서울·경기" in zones_in else 0.0)
        zones, std_auc = zones_in, tp.get("standard_vs_regional_auc")
        srun = os.path.realpath(STD_RUN)
        if "서울·경기" not in zones_in and os.path.isdir(srun):
            std = common.Tree(srun, enc)
            assert std.zones == ["표준", "지역"] or "서울·경기" in std.zones, std.zones
            STD_BIAS = run_bias(srun, 0.0) if "서울·경기" in std.zones else 0.0
            zones = ["서울·경기"] + zones_in
            std_auc = detector_auc(srun)
            state.update(std_model=os.path.basename(srun), std_bias=STD_BIAS)
        state.update(ready=True, model=os.path.basename(run), zones=zones, standard_bias=STANDARD_BIAS, score={
            "test_plain_macro_f1": tp["macro_f1"], "test_plain_accuracy": tp["accuracy"],
            "standard_auc": std_auc, "bench_macro_f1": bench["macro_f1"],
            "persons_macro_f1": tp["persons"]["macro_f1"]})
        log.info("accent classifier %s (+ %s) loaded in %.1f s: %s", state["model"], state["std_model"],
                 time.time() - t0, state["score"])
    except Exception as e:  # noqa: BLE001  the app keeps working without zones
        state["error"] = repr(e)
        log.exception("could not load the accent classifier")
    global phones
    try:
        import pron
        phones = pron.Scorer()
        state["pron"] = True
        log.info("phone recogniser loaded for pronunciation checks")
    except Exception:  # noqa: BLE001  the app keeps working without them
        log.exception("could not load the phone recogniser")


def read_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expected 16 kHz mono 16-bit WAV")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def classify(audio):
    """The served run's zone probabilities; with the detector, (P(regional), zone probabilities).
    One encoder pass feeds every frozen-encoder head of both runs."""
    x = enc.pooled([audio])
    p = clf.probs([audio], x)[0]
    if std is None:
        return p
    q = std.probs([audio], x)[0]
    if std.zones == ["표준", "지역"]:
        return float(q[1]), p
    s = q[std.zones.index("서울·경기")] * np.exp(STD_BIAS)
    return float(1 - s / (s + 1 - q[std.zones.index("서울·경기")])), p


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/status":
            return self._json(200, state)
        self._json(404, {"detail": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/pron?"):
            return self._pron()
        if self.path != "/accent":
            return self._json(404, {"detail": "not found"})
        if not state["ready"]:
            return self._json(503, {"detail": state["error"] or "loading"})
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BYTES:
            return self._json(413, {"detail": "clip too long"})
        try:
            audio = read_wav(self.rfile.read(n))
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"detail": str(e)})
        t0 = time.time()
        regional = None
        with lock:
            p = classify(audio)
            if std is not None:
                regional = p[0]
                p = np.concatenate([[1 - regional], regional * p[1]])  # 서울·경기 first, as state["zones"]
        if "서울·경기" in zones_in:
            p = p * np.exp(np.eye(len(p))[0] * STANDARD_BIAS)  # zone 0 is 서울·경기 (common.STANDARD)
            p = p / p.sum()
        self._json(200, {"zones": state["zones"], "probs": [round(float(x), 4) for x in p], "model": state["model"],
                         "regional": None if regional is None else round(float(regional), 4),
                         "seconds": round(time.time() - t0, 3)})

    def _pron(self):
        if phones is None:
            return self._json(503, {"detail": "loading"})
        from urllib.parse import parse_qs, urlparse
        text = (parse_qs(urlparse(self.path).query).get("text") or [""])[0][:400]
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BYTES:
            return self._json(413, {"detail": "clip too long"})
        try:
            audio = read_wav(self.rfile.read(n))
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"detail": str(e)})
        t0 = time.time()
        with lock:
            results = phones.score(audio, text)
        self._json(200, {"results": results, "seconds": round(time.time() - t0, 3)})

    def log_message(self, fmt, *args):  # keep the log to errors
        pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    threading.Thread(target=load, daemon=True).start()
    log.info("listening on 127.0.0.1:%d", PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
