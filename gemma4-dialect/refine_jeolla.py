"""Tighten align_jeolla.py's cuts. Whisper's segment times only place a line roughly (verify_jeolla.py: own-line
CER median 0.57, with clips of 18-22 s for 4 s lines), so each line's phones (seotturi-web pron.py g2p) are
force-aligned (CTC, the Korean phone recogniser pron.py uses) inside its rough span plus 1.5 s on both sides;
the line then runs from its first to its last aligned phone. Kept when the mean log-posterior of its phones
reaches --min-score and its speaking rate is plausible.

    python refine_jeolla.py --audio <원천데이터 zip> --out data_jeolla

Reads <out>/align.jsonl, writes <out>/refined.jsonl (start, end, score) and <out>/clips_refined/<recording>/<id>.wav.
"""

import argparse
import json
import os
import sys
import wave
import zipfile

import numpy as np

sys.path.insert(0, "/data/seotturi/seotturi-web/server")
from analysis import read_wav  # noqa: E402
import pron  # noqa: E402

from align_jeolla import chars  # noqa: E402

FRAME = 0.02  # wav2vec2 frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=1.5, help="seconds searched on both sides of the rough span")
    ap.add_argument("--min-score", type=float, default=-1.5, help="mean log-posterior of the line's phones")
    ap.add_argument("--gpu-share", type=float, default=0.05)
    args = ap.parse_args()
    import torch
    import torchaudio.functional as F
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)
    scorer = pron.Scorer()
    rows = [json.loads(line) for line in open(os.path.join(args.out, "align.jsonl"), encoding="utf-8")]
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r["recording"], []).append(r)
    path = os.path.join(args.out, "refined.jsonl")
    done = {json.loads(line)["recording"] for line in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    kept = total = 0
    with zipfile.ZipFile(args.audio) as z, open(path, "a", encoding="utf-8") as fo:
        names = {os.path.basename(n)[:-4]: n for n in z.namelist() if n.lower().endswith(".wav")}
        for rec, rs in by_rec.items():
            if rec in done or rec not in names:
                continue
            x = read_wav(z.read(names[rec]))
            d = os.path.join(args.out, "clips_refined", rec)
            os.makedirs(d, exist_ok=True)
            for r in rs:
                total += 1
                n = len(chars(r["dialect"]))
                a = max(0.0, r["start"] - args.pad)
                b = min(len(x) / 16000, r["end"] + args.pad, a + n / 2.5 + 2 * args.pad + 2)
                phones = [p for p in pron.g2p(r["dialect"]) if p[0] in scorer.vocab and p[0] != "|"]
                if not phones:
                    continue
                lp = scorer.log_probs(x[int(a * 16000):int(b * 16000)])
                ids = torch.tensor([[scorer.vocab[p[0]] for p in phones]], device=lp.device)
                if lp.shape[0] < 2 * ids.shape[1]:
                    continue
                try:
                    frames, scores = F.forced_align(lp[None], ids, blank=scorer.blank)
                except RuntimeError:
                    continue
                spans = F.merge_tokens(frames[0], scores[0].exp(), blank=scorer.blank)
                if len(spans) != len(phones):
                    continue
                score = float(np.mean([lp[s.start:s.end, s.token].mean().item() for s in spans]))
                s, e = a + spans[0].start * FRAME, a + spans[-1].end * FRAME
                if score < args.min_score or not (n / 12 <= e - s <= n / 2.5 + 1.0):
                    continue
                kept += 1
                lo, hi = max(0, int((s - 0.1) * 16000)), min(len(x), int((e + 0.15) * 16000))
                with wave.open(os.path.join(d, r["id"] + ".wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                    w.writeframes((np.clip(x[lo:hi], -1, 1) * 32767).astype("<i2").tobytes())
                fo.write(json.dumps({**r, "start": round(s, 2), "end": round(e, 2), "rough": [r["start"], r["end"]],
                                     "score": round(score, 3)}, ensure_ascii=False) + "\n")
            fo.flush()
            print(f"{rec}: kept {kept}/{total} so far", flush=True)


if __name__ == "__main__":
    main()
