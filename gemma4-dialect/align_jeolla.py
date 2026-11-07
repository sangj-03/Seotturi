"""Cut the Jeolla set (AI Hub 120) into utterances. Its labels are plain text, one line per utterance
("speaker<TAB>text", dialect/standard pairs inline as "(방언)/(표준)"), without times or speaker origin, so
the times come from aligning each recording's lines to Whisper-large-v3's word timestamps:

  1. Whisper (Korean, segment timestamps) over the whole recording -> segments with start/end; each
     segment's characters spread evenly over its time (word timestamps keep every cross-attention and ran
     out of an 18 GB cap even at batch 2 on the shared GPU)
  2. the label lines (dialect side) and the Whisper text, both reduced to Hangul and digits, aligned
     character by character (difflib); a line runs from its first to its last matched character
  3. kept only when >= --min-match of its characters match and both of its ends are matched

    python align_jeolla.py --audio <원천데이터 zip> --labels <라벨링데이터 zip> --out data_jeolla [--limit N]

Writes <out>/clips/<recording>/<id>.wav (16 kHz mono) and <out>/align.jsonl (id, recording, speaker,
start, end, dialect, standard, has_dialect, match). Resumable: recordings already in align.jsonl are skipped.
"""

import argparse
import difflib
import io
import json
import os
import re
import sys
import time
import wave
import zipfile

import numpy as np

sys.path.insert(0, "/data/seotturi/seotturi-web/server")
from analysis import read_wav  # noqa: E402  PCM WAV -> float32 mono 16 kHz

PAIR = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")
KEEP = re.compile(r"[가-힣0-9]")


def sides(text):
    """(dialect line, standard line) of one label line; anonymised names and places dropped."""
    text = re.sub(r"[@#][^\s#]*#?", " ", text)  # @상호명, #이름#
    text = re.sub(r"\(\(([^()]*)\)\)", r"\1", text)  # ((unsure)) -> unsure
    dialect = PAIR.sub(lambda m: m.group(1), text)
    standard = PAIR.sub(lambda m: m.group(2), text)
    clean = lambda s: re.sub(r"\s+", " ", s.replace("~", "")).strip()  # noqa: E731
    return clean(dialect), clean(standard), bool(PAIR.search(text))


def chars(s):
    return "".join(KEEP.findall(s))


def labels_of(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith(".txt"):
                lines = []
                for line in z.read(n).decode("utf-8", "replace").splitlines():
                    spk, _, text = line.partition("\t")
                    if text.strip():
                        lines.append((spk.strip(), text.strip()))
                out[os.path.basename(n)[:-4]] = lines
    return out


def align(lines, words, min_match):
    """lines: [(spk, text)], words: [(text, start, end)] (words or segments) -> [(index, start, end, match)]
    of the lines kept; inside one word or segment the characters are spread evenly over its time."""
    tchars, tline = [], []
    for i, (_, text) in enumerate(lines):
        c = chars(sides(text)[0])
        tchars.append(c)
        tline += [i] * len(c)
    T = "".join(tchars)
    A, t0, t1 = "", [], []
    for w, s, e in words:
        c = chars(w)
        if not c or s is None or e is None or e <= s:
            continue
        A += c
        t0 += [s + (e - s) * k / len(c) for k in range(len(c))]
        t1 += [s + (e - s) * (k + 1) / len(c) for k in range(len(c))]
    if not T or not A:
        return []
    matched = {}  # transcript char -> asr char
    for blk in difflib.SequenceMatcher(None, T, A, autojunk=False).get_matching_blocks():
        for d in range(blk.size):
            matched[blk.a + d] = blk.b + d
    out, pos = [], 0
    for i, c in enumerate(tchars):
        idx = range(pos, pos + len(c))
        pos += len(c)
        if len(c) < 3:
            continue
        hit = [j for j in idx if j in matched]
        if len(hit) / len(c) < min_match or hit[0] > idx[0] + 1 or hit[-1] < idx[-1] - 1:
            continue
        s, e = t0[matched[hit[0]]], t1[matched[hit[-1]]]
        if e <= s or (e - s) > 1.5 * len(c) or len(c) / (e - s) > 15:  # implausible speaking rate: a bad match
            continue
        out.append((i, s, e, round(len(hit) / len(c), 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="recordings to process (0 = all)")
    ap.add_argument("--min-match", type=float, default=0.7)
    ap.add_argument("--gpu-share", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4, help="Whisper chunks per batch")
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/data/huggingface")
    import torch
    from transformers import pipeline
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda")
    labels = labels_of(args.labels)
    os.makedirs(args.out, exist_ok=True)
    meta = os.path.join(args.out, "align.jsonl")
    done = set()
    if os.path.exists(meta):
        done = {json.loads(line)["recording"] for line in open(meta, encoding="utf-8")}
    stats = os.path.join(args.out, "recordings.jsonl")  # per recording: lines, kept, seconds
    if os.path.exists(stats):
        done |= {json.loads(line)["recording"] for line in open(stats, encoding="utf-8")}
    n = 0
    with zipfile.ZipFile(args.audio) as z, open(meta, "a", encoding="utf-8") as fm, open(stats, "a", encoding="utf-8") as fs:
        for name in z.namelist():
            rec = os.path.basename(name)[:-4]
            if not name.lower().endswith(".wav") or rec not in labels or rec in done:
                continue
            if args.limit and n >= args.limit:
                break
            t0 = time.time()
            x = read_wav(z.read(name))
            if x is None:
                continue
            r = asr({"raw": x, "sampling_rate": 16000}, chunk_length_s=30, batch_size=args.batch, return_timestamps=True,
                    generate_kwargs={"language": "ko", "task": "transcribe"})
            seg = r["chunks"]  # a missing end (last segment of a chunk) runs to the next start
            words = [(c["text"], c["timestamp"][0],
                      c["timestamp"][1] if c["timestamp"][1] is not None else (seg[k + 1]["timestamp"][0] if k + 1 < len(seg) else len(x) / 16000))
                     for k, c in enumerate(seg)]
            lines = labels[rec]
            kept = align(lines, words, args.min_match)
            d = os.path.join(args.out, "clips", rec)
            os.makedirs(d, exist_ok=True)
            for i, s, e, m in kept:
                a, b = max(0, int((s - 0.3) * 16000)), min(len(x), int((e + 0.4) * 16000))  # segment times are coarse
                cid = f"{rec}.{i}"
                with wave.open(os.path.join(d, cid + ".wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    w.writeframes((np.clip(x[a:b], -1, 1) * 32767).astype("<i2").tobytes())
                dia, std, has = sides(lines[i][1])
                fm.write(json.dumps({"id": cid, "recording": rec, "speaker": lines[i][0], "start": round(s, 2), "end": round(e, 2),
                                     "dialect": dia, "standard": std, "has_dialect": has, "match": m}, ensure_ascii=False) + "\n")
            fs.write(json.dumps({"recording": rec, "lines": len(lines), "kept": len(kept), "seconds": round(len(x) / 16000, 1),
                                 "asr_words": len(words), "took": round(time.time() - t0, 1)}) + "\n")
            fm.flush(); fs.flush()
            n += 1
            print(f"{rec}: {len(kept)}/{len(lines)} lines kept, {len(x) / 16000 / 60:.1f} min audio in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
