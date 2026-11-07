"""Gemma 4 dialect engine for the 서투리 web app.

Holds Gemma 4 12B and the dialect LoRA adapter on the GPU and serves, on 127.0.0.1:8788
(for app.py only):

* POST /asr       a WAV clip (at most 30 s) -> 지역 / 방언 / 표준어 from the served adapter
* POST /generate  chat messages -> text, with the adapter switched off (plain Gemma 4 coach)
* POST /score     chat prompts (text, and audio as base64 WAV parts) -> the probability of each score token
                  after each (G-Eval), adapter off
* GET  /status    served adapter, its validation score, update schedule, training progress
* POST /update    look for a newer checkpoint now

Every SEOTTURI_UPDATE_HOURS (2) it looks for the newest complete checkpoint of the running
training (runs/lora_v2r/checkpoint-*, then final), copies it to state/adapters/, scores it and
the served adapter on the fixed benchmark and swaps it in only when it is worse on neither line
(dialect and standard-Korean CER), keeps the answer format and, clip by clip, is better over both
lines with 95% probability (paired bootstrap). One worker thread owns the GPU; user requests
always run before update scoring.

    /data/seotturi/.venv/bin/python server/engine.py
"""

import asyncio
import base64
import concurrent.futures
import glob
import io
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import defaultdict
from math import gcd

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # repository root
GEMMA_DIR = os.environ.get("SEOTTURI_GEMMA_DIR", os.path.join(REPO, "training", "gemma4-dialect"))
sys.path.insert(0, GEMMA_DIR)
import common  # noqa: E402  the training project's prompts and parsing, so serving matches training
import engine_files as EF  # noqa: E402  the state files, shared with app.py for when the engine is off

RUNS = os.path.join(GEMMA_DIR, "runs")
# Runs whose checkpoints may be served, earliest stage first, each with the answer format it trains on.
# 2026-10-08: lora_v2r (region format) stopped at step 6,250 without beating lora_v1; lora_v3 is a short
# fine-tune of lora_v1 in the basic format (gemma4-dialect/run_stage4_v3.sh).
# 2026-10-11: lora_v3 tied lora_v1 on the held-out data_test; lora_r64 (a rank-64 LoRA trained on top of the
# merged lora_v1, run_stage5_r64.sh) beat it there and on data_all/valid, and is served as lora_r64c, the two
# folded into one rank-80 adapter (combine_adapters.py) so it can still be switched off for the plain coach.
# 2026-10-11: lora_s6 (another rank-64 LoRA on the merged lora_r64c, run_stage6.sh), folded with it into
# lora_s6c, beat lora_r64c on data_test (4,000 clips, decoded as the site does): standard CER 0.151 -> 0.147,
# dialect 0.145 -> 0.140, p_better 1.00 (runs/eval_test_guard/report_s6c.txt). Served since then by the
# user's decision; the 160-clip gate below is too small to resolve a gain of that size.
WATCH = [tuple(x.split(":")) for x in os.environ.get("SEOTTURI_WATCH", "lora_r64c:basic,lora_s6c:basic").split(",")]
# Served until the first watched checkpoint passes the gate.
FALLBACK = tuple(os.environ.get("SEOTTURI_FALLBACK", "lora_v1:basic").split(":"))
UPDATE_HOURS = float(os.environ.get("SEOTTURI_UPDATE_HOURS", "2"))
# How much worse (CER) a candidate may be and still replace the served adapter. 0 since 2026-10-07 (user's
# choice: never swap in a worse model); the old 0.02 on the standard line alone let lora_v2r@1000 replace
# lora_v1 with a dialect CER 0.03 worse.
GATE_TOLERANCE = float(os.environ.get("SEOTTURI_GATE_TOLERANCE", "0"))
GATE_MANIFEST = os.path.join(GEMMA_DIR, "data_all", "valid_origin.jsonl")
# Utterances 100-111 of each region: held out from training and outside the trainer's eval subset (first 40).
GATE_SKIP, GATE_PER_REGION = 100, 12
MAX_BATCH = 4
MAX_SECONDS = 30  # Gemma 4 audio input: 750 tokens x 40 ms
STATE_DIR = os.path.join(ROOT, "state")
STATE_FILE = os.path.join(STATE_DIR, "engine_state.json")
# Benchmark: a fixed set of held-out clips (scripts/build_benchmark.py) scored by scripts/bench_score.py
# in the speech venv (sacreBLEU). Every checkpoint the update cycle looks at is benchmarked once.
BENCH_FILE = os.path.join(STATE_DIR, "benchmark.jsonl")
BENCH_DIR = os.path.join(STATE_DIR, "bench")
SCORER = [os.path.join(ROOT, ".venv-tts", "bin", "python"), "-I", os.path.join(ROOT, "scripts", "bench_score.py")]
# Training steps (16 utterances each) before each run: stage 1 Gangwon 1,563, stage 2 (lora_v1) 5,155.
STEP_OFFSET = {"base": 0, "lora_v1": 1563, "lora_v2r": 1563 + 5155, "lora_v3": 1563 + 5155,
               "lora_r64c": 1563 + 5155 + 3000, "lora_s6c": 1563 + 5155 + 3000 + 4000}
MEM_GB = float(os.environ.get("SEOTTURI_ENGINE_MEM_GB", "40"))
# Promote a candidate only when the paired bootstrap (scripts/bench_score.py --compare) gives it at least
# this probability of a lower CER over both lines together (Bisani & Ney 2004), on top of the point rule
# above: with 160 clips two equally good adapters would otherwise swap on noise.
GATE_CONFIDENCE = float(os.environ.get("SEOTTURI_GATE_CONFIDENCE", "0.95"))
# Decoding guard for the answers (ASR, benchmark, self-test), written into each benchmark summary: cached
# scores made with another decoding are redone, so the gate compares adapters decoded the same way.
DECODE = "guard1"
# The answer starts as trained ("방언:" or "지역:"): lora_v2r@1000-2000 began 5% of their benchmark
# answers with stray text ("T:", "_thought", a number) and lost the 방언 line.
ANSWER_START = {"basic": "방언:", "region": "지역:"}
# A runaway repetition: a unit of up to 20 characters 8+ times in a row (Whisper re-decodes looping
# segments, Radford et al. 2022). In 2 of 167,352 reference lines ("음음음음 음음음음"); in 6 of lora_v1's
# 1,199 evaluation answers ("똥 똥 똥 ...") and 3 of lora_v2r@2000's 160 benchmark clips ("다다다...").
LOOP = re.compile(r"(\S.{0,19}?)(?:\s*\1){7,}")

log = logging.getLogger("engine")


class Version:
    """A servable adapter: one checkpoint (or the final adapter) of a training run."""

    def __init__(self, run, step, path, fmt, final=False):
        self.run, self.step, self.path, self.fmt, self.final = run, int(step), path, fmt, bool(final)

    @property
    def id(self):
        return f"{self.run}@{'final' if self.final else self.step}"

    @property
    def key(self):
        runs = [r for r, _ in WATCH]
        return (runs.index(self.run) + 1 if self.run in runs else 0, self.step, self.final)

    def to_dict(self):
        return {"id": self.id, "run": self.run, "step": self.step, "path": self.path, "format": self.fmt,
                "final": self.final}

    @classmethod
    def from_dict(cls, d):
        return cls(d["run"], d["step"], d["path"], d["format"], d["final"])


def checkpoints(run):
    """Complete checkpoints of a run as (step, dir). trainer_state.json is the last file the Trainer
    writes, so its presence (and a minute of quiet) means the adapter is fully on disk."""
    out = []
    for d in glob.glob(os.path.join(RUNS, run, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        st = os.path.join(d, "trainer_state.json")
        if (m and os.path.exists(os.path.join(d, "adapter_model.safetensors")) and os.path.exists(st)
                and time.time() - os.path.getmtime(st) > 60):
            out.append((int(m.group(1)), d))
    return sorted(out)


def latest_version():
    best = None
    for run, fmt in WATCH:
        steps = checkpoints(run)
        final = os.path.join(RUNS, run, "final")
        if os.path.exists(os.path.join(final, "train_args.json")):  # train.py writes it after the adapter
            v = Version(run, steps[-1][0] if steps else 0, final, fmt, final=True)
        elif steps:
            v = Version(run, steps[-1][0], steps[-1][1], fmt)
        else:
            continue
        if best is None or v.key > best.key:
            best = v
    return best


def fallback_version():
    run, fmt = FALLBACK
    steps = checkpoints(run)
    return Version(run, steps[-1][0] if steps else 0, os.path.join(RUNS, run, "final"), fmt, final=True)


def snapshot(v):
    """Copy a watched adapter out of its run (the Trainer keeps only the last 3 checkpoints)."""
    dst = os.path.join(STATE_DIR, "adapters", v.id.replace("@", "-"))
    if not os.path.exists(os.path.join(dst, "adapter_model.safetensors")):
        tmp = dst + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(os.path.join(v.path, f), tmp)
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)
    return dst


def read_wav(data):
    """PCM WAV bytes -> float32 mono at 16 kHz."""
    with wave.open(io.BytesIO(data)) as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        from scipy.signal import resample_poly
        g = gcd(rate, 16000)
        x = resample_poly(x, 16000 // g, rate // g).astype(np.float32)
    return x


def bench_path(version_id):
    return os.path.join(BENCH_DIR, version_id.replace("@", "-") + ".json")


def answer_budget(seconds):
    """max_new_tokens for a clip: training answers reach 72 tokens below 5 s and 149 at 10-18 s, so 160
    for every clip was tight for long fast speech. Loops are stopped by StopLoops, not by this budget."""
    return int(min(512, max(64, 32 + 16 * seconds)))


def unloop(line):
    """The line cut after the first two copies of a runaway repetition, and whether it was cut."""
    m = LOOP.search(line)
    if not m:
        return line, False
    two = re.compile(re.escape(m.group(1)) + r"\s*" + re.escape(m.group(1))).match(line, m.start())
    return line[:two.end()].strip(), True


def bench_summaries():
    return EF.bench_summaries(BENCH_DIR)


compact = EF.compact  # the headline numbers of a benchmark summary (also kept in the update history)


class Job:
    _seq = itertools.count()

    def __init__(self, kind, priority=0, **payload):
        self.kind, self.priority, self.payload = kind, priority, payload
        self.seq = next(Job._seq)
        self.future = concurrent.futures.Future()
        self.created, self.started = time.time(), None
        self.cancelled = False


class Engine:
    def __init__(self):
        self.ready, self.error, self.busy = False, None, None
        self.cv = threading.Condition()
        self.pending = []
        self.model = self.processor = None
        self.active = self.active_name = None
        self.loaded_at = None
        self.names = itertools.count()
        self.lookup = False  # set by the background self-test (or SEOTTURI_PROMPT_LOOKUP=0/1)
        self.stats = defaultdict(float)
        self.state = self._read_state()
        self.next_check, self.last_check = None, None
        self.gate_rows = self._gate_rows()
        self.bench_rows = ([json.loads(line) for line in open(BENCH_FILE, encoding="utf-8")]
                           if os.path.exists(BENCH_FILE) else [])
        self.bench_progress = None
        threading.Thread(target=self._boot, name="gpu-worker", daemon=True).start()

    # ---- state ---------------------------------------------------------------------------
    def _read_state(self):
        return EF.read_state(STATE_FILE)

    def _save_state(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_FILE)

    def _gate_rows(self):
        by_region = defaultdict(list)
        for r in common.read_manifest(GATE_MANIFEST):
            by_region[r["region"]].append(r)
        return [r for rows in by_region.values() for r in rows[GATE_SKIP:GATE_SKIP + GATE_PER_REGION]]

    # ---- boot ----------------------------------------------------------------------------
    def _initial_version(self):
        a = self.state.get("active")
        if a and os.path.exists(os.path.join(a["path"], "adapter_model.safetensors")):
            return Version.from_dict(a)
        return fallback_version()

    def _boot(self):
        try:
            from peft import PeftModel
            from transformers import AutoModelForMultimodalLM, AutoProcessor
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(0.9, MEM_GB * 1e9 / total))
            t0 = time.time()
            self.processor = AutoProcessor.from_pretrained(common.MODEL_PATH)
            self.processor.tokenizer.padding_side = "left"
            base = AutoModelForMultimodalLM.from_pretrained(common.MODEL_PATH, dtype=torch.bfloat16,
                                                            device_map="cuda")
            v = self._initial_version()
            name = f"a{next(self.names)}"
            self.model = PeftModel.from_pretrained(base, v.path, adapter_name=name)
            self.model.eval()
            self.active, self.active_name, self.loaded_at = v, name, time.time()
            self.state["active"] = v.to_dict()
            self._save_state()
            log.info("loaded %s in %.0f s", v.id, time.time() - t0)
        except Exception as e:  # noqa: BLE001  reported through /status
            log.exception("boot failed")
            self.error = repr(e)
            return
        self.ready = True
        self.submit(Job("bench", priority=2))
        self.submit(Job("selftest", priority=3))
        self.check_now("startup")
        self.next_check = time.time() + UPDATE_HOURS * 3600
        threading.Thread(target=self._updater, name="updater", daemon=True).start()
        while True:
            self._run(self._take())

    def _selftest(self):
        """Decide on prompt-lookup decoding. It copies n-grams already in the sequence (the 표준어
        line mostly repeats the 방언 line), so it needs fewer decoding steps, but its multi-token
        verification is not bit-identical to plain greedy decoding in bf16. Keep it only if, on 8
        held-out clips, it is faster and its standard-Korean CER against the references is not
        worse by more than 0.02. Runs in the background; plain decoding is used until it is done."""
        mode = os.environ.get("SEOTTURI_PROMPT_LOOKUP", "auto")
        if mode in ("0", "1"):
            self.lookup = mode == "1"
            return {"lookup": self.lookup, "forced": True}
        rows = self.gate_rows[::6][:8]
        t_plain = t_fast = 0.0
        e_plain, e_fast = [], []
        for r in rows:
            self._serve_users()
            self.model.set_adapter(self.active_name)
            enc = common.prompt_inputs(self.processor, common.load_audio(r["audio_path"]), self.active.fmt)
            t = time.time()
            a = self._generate_ids(enc, self.active.fmt, r["duration"])[0]
            t_plain += time.time() - t
            t = time.time()
            try:
                b = self._generate_ids(enc, self.active.fmt, r["duration"], prompt_lookup_num_tokens=10)[0]
            except Exception:  # noqa: BLE001
                log.exception("prompt lookup decoding failed; plain decoding only")
                self.lookup = False
                return {"lookup": False, "error": "failed"}
            t_fast += time.time() - t
            e_plain.append(common.cer(r["standard_form"], common.parse_answer(a)[1]))
            e_fast.append(common.cer(r["standard_form"], common.parse_answer(b)[1]))
        cp, cf = sum(e_plain) / len(rows), sum(e_fast) / len(rows)
        self.lookup = t_fast < 0.9 * t_plain and cf <= cp + 0.02
        res = {"lookup": self.lookup, "plain_s": round(t_plain, 1), "lookup_s": round(t_fast, 1),
               "plain_cer": round(cp, 4), "lookup_cer": round(cf, 4), "n": len(rows)}
        self.stats.update({f"selftest_{k}": v for k, v in res.items() if k != "lookup"})
        log.info("prompt-lookup self-test: %s", res)
        return res

    # ---- scheduling ----------------------------------------------------------------------
    def submit(self, job):
        with self.cv:
            self.pending.append(job)
            self.cv.notify()
        return job

    def _take(self, max_priority=None):
        """Next job, or several waiting ASR jobs of the same priority as one batch.
        With max_priority set, return [] instead of waiting when nothing that urgent is queued."""
        with self.cv:
            while True:
                ready = [j for j in self.pending if max_priority is None or j.priority <= max_priority]
                if ready or max_priority is not None:
                    break
                self.cv.wait()
            if not ready:
                return []
            ready.sort(key=lambda j: (j.priority, j.seq))
            batch = [ready[0]]
            if batch[0].kind == "asr":
                batch += [j for j in ready[1:] if j.kind == "asr" and j.priority == batch[0].priority][:MAX_BATCH - 1]
            for j in batch:
                self.pending.remove(j)
            return batch

    def _run(self, batch):
        batch = [j for j in batch if not j.cancelled and j.future.set_running_or_notify_cancel()]
        if not batch:
            return
        for j in batch:
            j.started = time.time()
        prev, self.busy = self.busy, batch[0].kind
        try:
            if batch[0].kind == "asr":
                results = self._asr(batch)
            elif batch[0].kind == "generate":
                results = [self._chat(batch[0])]
            elif batch[0].kind == "bench":
                results = [self._bench(self.active_name, self.active)]
            elif batch[0].kind == "score":
                results = [self._score(batch[0])]
            elif batch[0].kind == "selftest":
                results = [self._selftest()]
            else:
                results = [self._update(batch[0])]
            for j, r in zip(batch, results):
                j.future.set_result(r)
        except Exception as e:  # noqa: BLE001  handed to the waiting request
            log.exception("%s job failed", batch[0].kind)
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            for j in batch:
                if not j.future.done():
                    j.future.set_exception(e)
        finally:
            self.busy = prev

    def _serve_users(self):
        """Run waiting user requests (priority 0) in the middle of a long background job."""
        while True:
            batch = self._take(max_priority=0)
            if not batch:
                return
            self._run(batch)

    # ---- model calls ---------------------------------------------------------------------
    def _generate_ids(self, enc, fmt, seconds, prefix=None, **kw):
        """Greedy answers for a batch of prompts (`seconds`: the longest clip), each continuing `prefix`
        (default ANSWER_START[fmt]) and returned with it. A sequence whose last line runs away (LOOP) stops
        there; _finish repairs it."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        tok = self.processor.tokenizer
        prefix = ANSWER_START[fmt] if prefix is None else prefix
        enc = {k: v.to("cuda") for k, v in enc.items()}
        ids = torch.tensor(tok(prefix, add_special_tokens=False)["input_ids"], device="cuda")
        ids = ids.expand(enc["input_ids"].shape[0], -1)
        enc["input_ids"] = torch.cat([enc["input_ids"], ids], 1)
        enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(ids)], 1)
        enc["mm_token_type_ids"] = torch.cat([enc["mm_token_type_ids"], torch.zeros_like(ids)], 1)
        n = enc["input_ids"].shape[1]

        class StopLoops(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                texts = tok.batch_decode(input_ids[:, n:], skip_special_tokens=True)
                return torch.tensor([bool(LOOP.search(t.rsplit("\n", 1)[-1])) for t in texts],
                                    dtype=torch.bool, device=input_ids.device)

        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=answer_budget(seconds), do_sample=False,
                                      stopping_criteria=StoppingCriteriaList([StopLoops()]), **kw)
        return [prefix + self.processor.decode(s[n:], skip_special_tokens=True) for s in out]

    def _finish(self, text, fmt, seconds, enc_of):
        """(answer, repaired): a runaway line is cut after its first two copies (unloop), and when that cut
        the 방언 line or the 표준어 line is missing, the model writes the 표준어 line once more after the
        fixed 지역/방언 lines. `enc_of()` gives the clip's single prompt; it is only built for a repair."""
        dia, std = common.parse_answer(text)
        dia, cut = unloop(dia)
        std, cut_std = unloop(std)
        if not (cut or cut_std or (dia and not std)):
            return text, False
        head = (f"지역: {common.parse_region(text)}\n" if fmt == "region" else "") + f"방언: {dia}\n"
        if dia and (cut or not std):
            again = self._generate_ids(enc_of(), fmt, seconds, prefix=head + "표준어:")[0]
            std = unloop(common.parse_answer(again)[1])[0]
        self.stats["repaired"] += 1
        return head + f"표준어: {std}", True

    def _left_pad(self, ps):
        n = max(p["input_ids"].shape[1] for p in ps)
        pad = self.processor.tokenizer.pad_token_id or 0
        ids, att, tts = [], [], []
        for p in ps:
            k = n - p["input_ids"].shape[1]
            ids.append([pad] * k + p["input_ids"][0].tolist())
            att.append([0] * k + p["attention_mask"][0].tolist())
            tts.append([0] * k + p["mm_token_type_ids"][0].tolist())
        batch = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att),
                 "mm_token_type_ids": torch.tensor(tts)}
        batch["input_features"], batch["input_features_mask"] = common._pad_features(
            [p["input_features"] for p in ps], [p["input_features_mask"] for p in ps])
        return batch

    def _asr(self, jobs):
        self.model.set_adapter(self.active_name)
        v, t0 = self.active, time.time()
        encs = [common.prompt_inputs(self.processor, j.payload["audio"], v.fmt) for j in jobs]
        secs = [len(j.payload["audio"]) / 16000 for j in jobs]
        texts = None
        if len(jobs) == 1 and self.lookup:
            try:
                texts = self._generate_ids(encs[0], v.fmt, secs[0], prompt_lookup_num_tokens=10)
            except Exception:  # noqa: BLE001
                log.exception("prompt lookup decoding failed; plain decoding from now on")
                self.lookup = False
        if texts is not None:
            pass
        elif len(jobs) == 1:
            texts = self._generate_ids(encs[0], v.fmt, secs[0])
        else:
            texts = self._generate_ids(self._left_pad(encs), v.fmt, max(secs))
        done = [self._finish(t, v.fmt, s, lambda e=e: e) for t, s, e in zip(texts, secs, encs)]
        dt = time.time() - t0
        self.stats["asr"] += len(jobs)
        self.stats["asr_seconds"] += dt
        out = []
        for j, (text, repaired) in zip(jobs, done):
            dia, std = common.parse_answer(text)
            out.append({"raw": text, "dialect": dia, "standard": std,
                        "region": (common.parse_region(text) or None) if v.fmt == "region" else None,
                        "format": v.fmt, "model": v.id, "seconds": round(dt, 2), "batch": len(jobs),
                        "wait": round(j.started - j.created, 2), "repaired": repaired})
        return out

    def _chat(self, job):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer
        p = job.payload

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), job.cancelled, dtype=torch.bool, device=input_ids.device)

        class Push(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text and p.get("on_text"):
                    p["on_text"](text)

        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in p["messages"]]
        enc = self.processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                                 add_generation_prompt=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        kw = {"max_new_tokens": p["max_new_tokens"], "stopping_criteria": StoppingCriteriaList([Cancel()]),
              "streamer": Push(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)}
        if p["temperature"] > 0:
            kw.update(do_sample=True, temperature=p["temperature"], top_p=0.95, top_k=64)
        else:
            kw.update(do_sample=False)
        t0 = time.time()
        with torch.inference_mode(), self.model.disable_adapter():
            out = self.model.generate(**enc, **kw)
        n = enc["input_ids"].shape[1]
        self.stats["generate"] += 1
        return {"text": self.processor.decode(out[0][n:], skip_special_tokens=True).strip(),
                "tokens": int(out.shape[1] - n), "seconds": round(time.time() - t0, 2)}

    def _score(self, job):
        """G-Eval (Liu et al. 2023): for each prompt, the probability of each choice token (e.g. "1".."5") as
        the first token of the answer, with the adapter off, so the app can take the expected score instead of
        one sampled number. "mass" is how much probability the choices had together (low: no digit answer)."""
        tok = self.processor.tokenizer
        ids = [tok.encode(c, add_special_tokens=False) for c in job.payload["choices"]]
        if any(len(i) != 1 for i in ids):
            raise ValueError("every choice must be a single token")
        ids = [i[0] for i in ids]
        out = []
        for msgs in job.payload["prompts"]:
            m = [{"role": x["role"], "content": [
                {"type": "audio", "audio": read_wav(base64.b64decode(p["wav"]))} if p.get("type") == "audio" else p
                for p in (x["content"] if isinstance(x["content"], list) else [{"type": "text", "text": x["content"]}])]}
                for x in msgs]
            enc = self.processor.apply_chat_template(m, tokenize=True, return_dict=True, return_tensors="pt",
                                                     add_generation_prompt=True)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.inference_mode(), self.model.disable_adapter():
                logits = self.model(**enc, logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(logits, dim=-1)[ids]
            out.append({"probs": [round(float(x), 4) for x in p / p.sum().clamp_min(1e-12)], "mass": round(float(p.sum()), 4)})
        self.stats["score"] += len(out)
        return {"results": out}

    def _bench(self, name, version):
        """Benchmark an adapter on the fixed set and score it (cached per adapter id in state/bench/).
        Clips are generated in batches of 8 and decoded as users' clips are (_generate_ids, _finish);
        waiting user requests run between batches. A cached score made with another DECODE is redone."""
        path = bench_path(version.id)
        pred_path = path[:-len(".json")] + ".jsonl"
        if os.path.exists(path) and os.path.exists(pred_path):
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("decode") == DECODE:
                return cached
        rows = [r for r in self.bench_rows if r["kind"] == "asr" or version.fmt == "region"]
        if not rows:
            raise RuntimeError("no benchmark set: run scripts/build_benchmark.py")
        os.makedirs(BENCH_DIR, exist_ok=True)
        preds, t0 = [], time.time()
        self.bench_progress = {"id": version.id, "done": 0, "total": len(rows), "started": t0}
        try:
            for i in range(0, len(rows), 8):
                self._serve_users()
                self.model.set_adapter(name)
                chunk = rows[i:i + 8]
                t = time.time()
                texts = self._generate_ids(common.generation_batch(self.processor, chunk, version.fmt), version.fmt,
                                           max(r["duration"] for r in chunk))
                done = [self._finish(text, version.fmt, r["duration"], lambda r=r: common.prompt_inputs(
                    self.processor, common.load_audio(r["audio_path"]), version.fmt)) for r, text in zip(chunk, texts)]
                share = (time.time() - t) / len(chunk)
                for r, (text, repaired) in zip(chunk, done):
                    dia, std = common.parse_answer(text)
                    preds.append({"kind": r["kind"], "id": r["id"], "block": r["id"].split(".")[0],
                                  "region": r["region"], "origin": r.get("origin"), "has_dialect": r["has_dialect"],
                                  "ref_dialect": r["dialect_form"], "ref_standard": r["standard_form"],
                                  "hyp_raw": text, "hyp_dialect": dia, "hyp_standard": std,
                                  "hyp_zone": (common.parse_region(text) or None) if version.fmt == "region" else None,
                                  "duration": r["duration"], "seconds": round(share, 3), "repaired": repaired})
                self.bench_progress["done"] = len(preds)
        finally:
            self.model.set_adapter(self.active_name)
            self.bench_progress = None
        with open(pred_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        out = subprocess.run(SCORER + [pred_path, "--format", version.fmt], capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            raise RuntimeError(f"bench_score failed: {out.stderr[-500:]}")
        summary = {"id": version.id, "run": version.run, "step": version.step, "final": version.final,
                   "format": version.fmt, "cumulative_step": STEP_OFFSET.get(version.run, 0) + version.step,
                   "time": time.time(), "seconds": round(time.time() - t0), "n": len(preds), "source": "engine",
                   "decode": DECODE, "repaired": sum(p["repaired"] for p in preds), "metrics": json.loads(out.stdout)}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        log.info("benchmark %s: %s", version.id, compact(summary))
        return summary

    def _update(self, job):
        cand, path = job.payload["version"], job.payload["path"]
        name = f"a{next(self.names)}"
        rec = {"time": time.time(), "candidate": cand.id, "served": self.active.id}
        self.model.load_adapter(path, adapter_name=name)
        try:
            cur = self._bench(self.active_name, self.active)
            new = self._bench(name, Version(cand.run, cand.step, path, cand.fmt, cand.final))
            paired = self._compare(cur, new)
        except Exception:
            self.model.set_adapter(self.active_name)
            self.model.delete_adapter(name)
            raise
        cm, nm = cur["metrics"]["asr"], new["metrics"]["asr"]
        better = (nm["format_ok"] >= cm["format_ok"]
                  and nm["cer_standard"]["value"] <= cm["cer_standard"]["value"] + GATE_TOLERANCE
                  and nm["cer_dialect"]["value"] <= cm["cer_dialect"]["value"] + GATE_TOLERANCE
                  and paired["cer_both"]["p_better"] >= GATE_CONFIDENCE)
        if better:
            old = self.active_name
            self.active = Version(cand.run, cand.step, path, cand.fmt, cand.final)
            self.active_name, self.loaded_at = name, time.time()
            self.model.set_adapter(name)
            self.model.delete_adapter(old)
            self.state["active"] = self.active.to_dict()
            self._prune_snapshots()
        else:
            self.model.set_adapter(self.active_name)
            self.model.delete_adapter(name)
            self.state["rejected"].append(cand.id)
        torch.cuda.empty_cache()
        rec.update(result="promoted" if better else "kept", served_score=compact(cur), candidate_score=compact(new),
                   paired={k: paired[k] for k in ("cer_dialect", "cer_standard", "cer_both", "format_fail")},
                   confidence=GATE_CONFIDENCE)
        self.state["history"].append(rec)
        self._save_state()
        log.info("update: %s", rec)
        return rec

    def _compare(self, cur, new):
        """Paired bootstrap of the candidate's benchmark answers against the served adapter's."""
        files = [bench_path(b["id"])[:-len(".json")] + ".jsonl" for b in (new, cur)]
        out = subprocess.run(SCORER + [files[0], "--compare", files[1]], capture_output=True, text=True, timeout=900)
        if out.returncode != 0:
            raise RuntimeError(f"bench_score --compare failed: {out.stderr[-500:]}")
        return json.loads(out.stdout)

    def _prune_snapshots(self):
        keep = {os.path.abspath(self.active.path)}
        snaps = sorted(glob.glob(os.path.join(STATE_DIR, "adapters", "*")), key=os.path.getmtime)
        for d in snaps[:-3]:
            if os.path.abspath(d) not in keep:
                shutil.rmtree(d, ignore_errors=True)

    # ---- updates -------------------------------------------------------------------------
    def check_now(self, reason):
        """Queue an update when the training has a complete checkpoint newer than the served adapter."""
        self.last_check = {"time": time.time(), "reason": reason}
        try:
            v = latest_version()
            if v is None or v.key <= self.active.key:
                self.last_check["result"] = "no newer checkpoint"
                return None
            if v.id in self.state["rejected"]:
                self.last_check["result"] = f"{v.id} already scored worse; waiting for the next checkpoint"
                return None
            with self.cv:
                running = any(j.kind == "update" for j in self.pending) or self.busy == "update"
            if running:
                self.last_check["result"] = "an update is already running"
                return None
            path = snapshot(v)
            self.last_check["result"] = f"scoring {v.id}"
            return self.submit(Job("update", priority=2, version=v, path=path))
        except Exception as e:  # noqa: BLE001
            log.exception("update check failed")
            self.last_check["result"] = f"error: {e!r}"
            return None

    def _updater(self):
        while True:
            if time.time() >= self.next_check:
                self.next_check = time.time() + UPDATE_HOURS * 3600
                self.check_now("schedule")
            time.sleep(30)

    def training(self):
        """Progress of the newest watched run, from train.py's speed.json and the Trainer state."""
        for run, _ in reversed(WATCH):
            path = os.path.join(RUNS, run, "speed.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                sp = json.load(f)
            total = sp["step"] + round(sp["eta_hours"] * 3600 / max(sp["sec_per_step"], 1e-6))
            steps = checkpoints(run)
            if steps:
                with open(os.path.join(steps[-1][1], "trainer_state.json")) as f:
                    ts = json.load(f)
                total = ts.get("max_steps") or total
            return {"run": run, "step": sp["step"], "total": total, "sec_per_step": sp["sec_per_step"],
                    "eta_hours": sp["eta_hours"], "updated": os.path.getmtime(path),
                    "save_every": 1000, "checkpoints": [s for s, _ in steps]}
        return None

    def status(self):
        a = self.active
        benches = bench_summaries()
        served = benches.get(a.id) if a else None
        return {"ready": self.ready, "error": self.error, "busy": self.busy,
                "queue": len(self.pending),
                "model": None if a is None else {**a.to_dict(), "loaded_at": self.loaded_at,
                                                 "score": compact(served) if served else None},
                "bench": [compact(b) for b in sorted(benches.values(), key=lambda b: (b["cumulative_step"], b["final"]))],
                "bench_progress": self.bench_progress,
                "base": "Gemma 4 12B (google/gemma-4-12B-it)",
                "update_hours": UPDATE_HOURS, "next_check": self.next_check, "last_check": self.last_check,
                "gate_confidence": GATE_CONFIDENCE, "decode": DECODE,
                "history": self.state["history"][-12:], "training": self.training(),
                "prompt_lookup": self.lookup,
                "stats": {k: round(v, 2) for k, v in self.stats.items()}}


engine = None
app = FastAPI(title="seotturi-engine")


def _need_ready():
    if engine.error:
        raise HTTPException(500, f"engine failed to start: {engine.error}")
    if not engine.ready:
        raise HTTPException(503, "model is loading")


@app.get("/status")
def status():
    return engine.status()


@app.get("/bench")
def bench():
    """Every benchmark summary with all metrics, oldest training state first."""
    items = sorted(bench_summaries().values(), key=lambda b: (b["cumulative_step"], b["final"]))
    return {"items": items, "progress": engine.bench_progress, "served": engine.active.id if engine.active else None,
            "next_check": engine.next_check, "update_hours": UPDATE_HOURS, "training": engine.training(),
            "set": {"asr": sum(r["kind"] == "asr" for r in engine.bench_rows),
                    "zone": sum(r["kind"] == "zone" for r in engine.bench_rows)}}


@app.post("/update")
def update():
    _need_ready()
    engine.check_now("manual")
    return engine.last_check


@app.post("/asr")
async def asr(request: Request, priority: int = 0):
    _need_ready()
    try:
        audio = read_wav(await request.body())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"not a PCM WAV file: {e}") from e
    if len(audio) > MAX_SECONDS * 16000:
        raise HTTPException(413, f"clip longer than {MAX_SECONDS} s; split it first")
    if len(audio) < 0.3 * 16000:
        raise HTTPException(422, "clip shorter than 0.3 s")
    job = engine.submit(Job("asr", priority, audio=audio))
    try:
        return await asyncio.wait_for(asyncio.wrap_future(job.future), timeout=900)
    except asyncio.TimeoutError as e:
        job.cancelled = True
        raise HTTPException(504, "timed out in the queue") from e


@app.post("/score")
async def score(request: Request):
    """{"prompts": [[{"role", "content"}], ...] (at most 4), "choices": ["1", ..., "5"]} -> {"results": [{"probs", "mass"}]}"""
    _need_ready()
    body = await request.json()
    job = engine.submit(Job("score", int(body.get("priority", 0)), prompts=body["prompts"][:4],
                            choices=body.get("choices") or ["1", "2", "3", "4", "5"]))
    return await asyncio.wait_for(asyncio.wrap_future(job.future), timeout=300)


@app.post("/generate")
async def generate(request: Request):
    """{"messages": [{"role", "content"}], "max_new_tokens", "temperature", "stream"}"""
    _need_ready()
    body = await request.json()
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    job = engine.submit(Job("generate", int(body.get("priority", 0)), messages=body["messages"],
                            max_new_tokens=max(1, min(int(body.get("max_new_tokens", 400)), 1200)),
                            temperature=float(body.get("temperature", 0)),
                            on_text=lambda t: loop.call_soon_threadsafe(q.put_nowait, t)))
    fut = asyncio.wrap_future(job.future)
    if not body.get("stream"):
        try:
            return await asyncio.wait_for(fut, timeout=900)
        finally:
            job.cancelled = True

    async def events():
        try:
            while True:
                getter = asyncio.ensure_future(q.get())
                done, _ = await asyncio.wait({getter, fut}, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    yield f"data: {json.dumps({'delta': getter.result()}, ensure_ascii=False)}\n\n"
                    continue
                getter.cancel()
                if fut.done():
                    while not q.empty():
                        yield f"data: {json.dumps({'delta': q.get_nowait()}, ensure_ascii=False)}\n\n"
                    if fut.exception():
                        yield f"data: {json.dumps({'error': repr(fut.exception())})}\n\n"
                    else:
                        yield f"data: {json.dumps({'done': True, **fut.result()}, ensure_ascii=False)}\n\n"
                    return
                if await request.is_disconnected():
                    return
        finally:
            job.cancelled = True

    return StreamingResponse(events(), media_type="text/event-stream")


def main():
    global engine
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:  # if memory runs out, the kernel should kill this server, never the training run
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        log.warning("could not raise oom_score_adj")
    engine = Engine()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("SEOTTURI_ENGINE_PORT", "8788")),
                log_level="warning")


if __name__ == "__main__":
    main()
