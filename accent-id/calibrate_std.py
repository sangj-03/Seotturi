"""The standard detector's bonus (accent.py adds it to log P(서울·경기) before P(regional) = 1 - P(서울·경기)),
chosen on the dev set, clean clips and their augmented copies (region4.py) together: the smallest share of
Seoul-Gyeonggi clips called regional while at least --caught of the regional clips are still called regional.
Writes "served_standard_bias" into the run's ensemble.json and the per-clip scores to dev_scores.npz.

    python calibrate_std.py runs/detector_mix
"""

import argparse
import json
import os

import numpy as np
import torch

import augment
from common import Encoder, Tree, load_wav

HERE = os.path.dirname(os.path.abspath(__file__))


def rows_of(name):
    with open(os.path.join(HERE, "data", f"{name}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def regional(p_seoul, bias):
    s = p_seoul * np.exp(bias)
    return 1 - s / (s + 1 - p_seoul)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--caught", type=float, default=0.90, help="regional dev clips that must stay called regional")
    ap.add_argument("--gpu-share", type=float, default=0.2)
    args = ap.parse_args()
    run = os.path.join(HERE, args.run)
    out = os.path.join(run, "dev_scores.npz")
    if os.path.exists(out):
        d = np.load(out)
        ps, seoul, aug = d["p_seoul"], d["seoul"], d["aug"]
    else:
        torch.cuda.set_per_process_memory_fraction(args.gpu_share)
        enc = Encoder()
        det = Tree(run, enc)
        k = det.zones.index("서울·경기")
        rows = rows_of("dev") + rows_of("r4_dev_aug") + rows_of("std_dev_aug")
        ps = []
        for i in range(0, len(rows), 16):
            b = rows[i:i + 16]
            audios = [augment.apply(load_wav(r["path"]), r["aug"]) if r.get("aug") else load_wav(r["path"]) for r in b]
            ps.append(det.probs(audios, enc.pooled(audios))[:, k])
            if i % 800 == 0:
                print(f"{i}/{len(rows)}", flush=True)
        ps = np.concatenate(ps)
        seoul = np.array([r["zone"] == "서울·경기" for r in rows])
        aug = np.array([bool(r.get("aug")) for r in rows])
        np.savez(out, p_seoul=ps, seoul=seoul, aug=aug)
    print(f"dev: {int((~seoul).sum())} regional, {int(seoul.sum())} Seoul-Gyeonggi clips ({int(aug.sum())} augmented)")
    best = None
    for b in np.arange(-2.0, 4.01, 0.05):
        r = regional(ps, b) >= 0.5
        caught, false = r[~seoul].mean(), r[seoul].mean()
        if caught >= args.caught and (best is None or false < best[2]):
            best = (round(float(b), 2), caught, false)
    b, caught, false = best
    for part, m in (("clean", ~aug), ("augmented", aug)):
        r = regional(ps[m], b) >= 0.5
        print(f"  bias {b:+.2f} {part:9s}: regional caught {r[~seoul[m]].mean():.0%}, Seoul called regional {r[seoul[m]].mean():.0%}")
    print(f"chosen bias {b:+.2f}: regional caught {caught:.0%}, Seoul called regional {false:.0%} (dev, clean + augmented)")
    spec = os.path.join(run, "ensemble.json")
    with open(spec, encoding="utf-8") as f:
        e = json.load(f)
    e["served_standard_bias"] = b
    e["bias_chosen_on"] = (f"calibrate_std.py: dev clean + augmented, smallest Seoul-Gyeonggi false alarm with regional caught >= "
                           f"{args.caught:.0%} (caught {caught:.1%}, false alarm {false:.1%})")
    with open(spec, "w", encoding="utf-8") as f:
        json.dump(e, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
