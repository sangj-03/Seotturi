"""Check align_jeolla.py's cuts: Whisper transcribes a random sample of clips again, and its text is compared
with the clip's own label line (dialect side) and, as a control, with a neighbouring kept line of the same
recording. Right cuts give a much lower character error rate against their own line than against the
neighbour; a cut shifted onto the wrong speech does not.

    python verify_jeolla.py data_jeolla [--n 200]
"""

import argparse
import json
import os
import random

import numpy as np

from align_jeolla import chars


def cer(ref, hyp):
    r, h = chars(ref), chars(hyp)
    if not r:
        return None
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return prev[-1] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--meta", default="align.jsonl", help="refined.jsonl for refine_jeolla.py's cuts")
    ap.add_argument("--clips", default="clips", help="clips_refined for refine_jeolla.py's cuts")
    ap.add_argument("--gpu-share", type=float, default=0.08)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/data/huggingface")
    import torch
    from transformers import pipeline
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)
    rows = [json.loads(line) for line in open(os.path.join(args.out, args.meta), encoding="utf-8")]
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r["recording"], []).append(r)
    random.seed(0)
    sample = random.sample(rows, min(args.n, len(rows)))
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda")
    paths = [os.path.join(args.out, args.clips, r["recording"], r["id"] + ".wav") for r in sample]
    hyps = [o["text"] for o in asr(paths, batch_size=4, generate_kwargs={"language": "ko", "task": "transcribe", "max_new_tokens": 160})]  # clips are short; a runaway repeat would fill memory
    own, other = [], []
    for r, h in zip(sample, hyps):
        rec = by_rec[r["recording"]]
        k = rec.index(r)
        nb = rec[k + 1] if k + 1 < len(rec) else rec[k - 1]
        own.append(cer(r["dialect"], h))
        other.append(cer(nb["dialect"], h))
    if "score" in sample[0]:  # refine_jeolla.py: CER by alignment score, to choose --min-score
        sc = np.array([r["score"] for r in sample])
        for q in (0.25, 0.5, 0.75):
            print(f"  score >= {np.quantile(sc, q):.2f} (top {1 - q:.0%}): own-line CER median "
                  f"{np.median([c for c, v in zip(own, sc) if c is not None and v >= np.quantile(sc, q)]):.2f}")
    own, other = np.array([x for x in own if x is not None]), np.array([x for x in other if x is not None])
    dur = np.array([r["end"] - r["start"] for r in rows])
    print(f"clips {len(rows)} from {len(by_rec)} recordings, median {np.median(dur):.1f} s (p10 {np.percentile(dur, 10):.1f}, p90 {np.percentile(dur, 90):.1f}), "
          f"with dialect words {np.mean([r['has_dialect'] for r in rows]):.0%}")
    print(f"sample {len(own)}: CER vs own line median {np.median(own):.2f} (<0.3: {np.mean(own < 0.3):.0%}, <0.5: {np.mean(own < 0.5):.0%}) | "
          f"vs neighbour line median {np.median(other):.2f} (<0.5: {np.mean(other < 0.5):.0%})")
    worst = sorted(zip(own, sample, hyps), key=lambda t: -t[0])[:5]
    for c, r, h in worst:
        print(f"  CER {c:.2f} {r['id']} [{r['start']}-{r['end']}] label: {r['dialect'][:40]} | whisper: {h[:40]}")


if __name__ == "__main__":
    main()
