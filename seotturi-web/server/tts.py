"""Speech for the AI side of 서투리: Supertonic 3 (Supertone, OpenRAIL) on the CPU.

    .venv-tts/bin/python server/tts.py        (127.0.0.1:8789, started by run.sh)

POST /tts {"text", "voice": F1-F5 | M1-M5, "speed": 0.7-2.0} -> audio/mpeg (mono, 24 kHz, 48 kbps)

Runs in its own venv (.venv-tts) so the training environment stays untouched, and on the CPU
(4 ONNX threads) so the GPU stays with the training run and the Gemma engine. One synthesis at a
time; app.py caches the results on disk. Model files: /data/seotturi/models/supertonic3.
"""

import json
import logging
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("SEOTTURI_TTS_MODEL", "/data/seotturi/models/supertonic3")
PORT = int(os.environ.get("SEOTTURI_TTS_PORT", "8789"))
THREADS = int(os.environ.get("SEOTTURI_TTS_THREADS", "4"))
MAX_CHARS = 400

log = logging.getLogger("tts")
lock = threading.Lock()
tts = None
styles = {}


def load():
    global tts
    os.environ.setdefault("SUPERTONIC_CACHE_DIR", MODEL_DIR)
    from supertonic import TTS
    t0 = time.time()
    tts = TTS(model_dir=MODEL_DIR, auto_download=False, intra_op_num_threads=THREADS, inter_op_num_threads=1)
    for name in tts.voice_style_names:
        styles[name] = tts.get_voice_style(voice_name=name)
    log.info("Supertonic 3 loaded in %.1f s: %s", time.time() - t0, sorted(styles))


def synthesize(text, voice, speed):
    """Korean text -> MP3 bytes."""
    import numpy as np
    with lock:
        wav, _ = tts.synthesize(text, voice_style=styles[voice], lang="ko", speed=speed)
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(tts.sample_rate), "-ac", "1", "-i", "pipe:0",
                        "-ar", "24000", "-codec:a", "libmp3lame", "-b:a", "48k", "-f", "mp3", "pipe:1"],
                       input=pcm, capture_output=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-300:])
    return p.stdout


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            self._send(200, json.dumps({"ready": tts is not None, "voices": sorted(styles)}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        if self.path != "/tts":
            return self._send(404, b"{}")
        if tts is None:
            return self._send(503, b'{"detail": "loading"}')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0)) or 0))
            text = " ".join(str(body.get("text", "")).split())[:MAX_CHARS]
            voice = body.get("voice", "F1")
            speed = min(2.0, max(0.7, float(body.get("speed", 1.0))))
            if not text or voice not in styles:
                return self._send(422, b'{"detail": "text and a known voice are required"}')
            t0 = time.time()
            mp3 = synthesize(text, voice, speed)
            log.info("%s x%.2f %d chars -> %d bytes in %.2f s", voice, speed, len(text), len(mp3), time.time() - t0)
            self._send(200, mp3, "audio/mpeg")
        except Exception as e:  # noqa: BLE001  reported to app.py
            log.exception("synthesis failed")
            self._send(500, json.dumps({"detail": repr(e)}).encode())

    def log_message(self, *args):  # requests are logged by synthesize
        pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:  # under memory pressure the kernel should pick this process, not the training run
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    os.nice(5)  # the CPU is shared with the training run's data loaders
    load()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
