"""A 4-region run (region4.py) under the site's recording conditions: the same 985 test_plain clips and
conditions as the served model's baseline (a one-off probe, not included; 2026-10-09), so the
numbers compare directly. The denoiser here (ffmpeg afftdn) is one training never used.

    python conditions.py runs/region4

Baseline (served ensemble, Seoul probability dropped), balanced recall over 강원/충청/경상/제주, chance 0.25:
원본 0.865, 사이트 전처리 0.815, 사이트+잡음제거 0.693, 무음화 0.550, 말소리만 0.792, 앞 2초 0.710.
"""

import argparse
import collections
import json
import os
import subprocess
import sys

import numpy as np
import torch

import augment
from common import ZONES, load_classifier, load_wav
from region4 import REGIONS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/data/seotturi/seotturi-web/server")
import analysis as A  # noqa: E402  the site's own preprocessing

BASELINE = {"원본": 0.865, "사이트 전처리": 0.815, "사이트+잡음제거": 0.693, "무음화": 0.550, "말소리만": 0.792, "앞 2초": 0.710,
            "배경만": 0.562}  # 배경만: the 520 clips with >= 0.5 s of background (background_probe.py); lower is better


def afftdn(x):
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                        "-af", "afftdn=nr=20:nf=-40", "-f", "s16le", "pipe:1"],
                       input=(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes(), capture_output=True)
    return np.frombuffer(p.stdout, "<i2").astype(np.float32) / 32768


def nonempty(x):
    return x if len(x) >= 3200 else np.zeros(3200, np.float32)


COND = {
    "원본": lambda x: x,
    "사이트 전처리": lambda x: A.preprocess(x)[0],
    "사이트+잡음제거": lambda x: A.preprocess(afftdn(x))[0],
    "무음화": lambda x: augment.gate(x, -40, 3),
    "말소리만": augment.speech_only,
    "앞 2초": lambda x: A.preprocess(x)[0][:32000],
    "배경만": augment.background,  # None (clip skipped) when under 0.5 s of background
}


def regional_score(p, zones, bias):
    """P(regional accent) from a 6-zone run (1 - P(서울·경기), after the service's standard bonus) or a std2 run."""
    if "서울·경기" in zones:
        s = p[:, zones.index("서울·경기")] * np.exp(bias)
        return 1 - s / (s + 1 - p[:, zones.index("서울·경기")])
    return p[:, zones.index("지역")]


def binary(clf, zones, args):
    """Standard vs regional: the 985 regional clips above plus test_plain's 200 Seoul-Gyeonggi clips."""
    rows = [json.loads(line) for line in open(os.path.join(HERE, "data", "test_plain.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["zone"] == "서울·경기"
            or {"대구·경북": "경상", "부산·울산·경남": "경상"}.get(r["zone"], r["zone"]) == r["dataset"]]
    regional = np.array([r["zone"] != "서울·경기" for r in rows])
    audio = [load_wav(r["path"]) for r in rows]
    out = {}
    for name, f in COND.items():
        ys = [f(x) for x in audio]
        keep = np.array([y is not None for y in ys])
        xs = [nonempty(y) for y in ys if y is not None]
        p = np.concatenate([clf.probs(xs[i:i + 16]) for i in range(0, len(xs), 16)])
        s, g = regional_score(p, zones, args.standard_bias), regional[keep]
        a, b = s[g], s[~g]
        auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
        out[name] = {"n": int(keep.sum()), "auc": round(auc, 3), "regional_caught": round(float((a >= 0.5).mean()), 3),
                     "standard_called_regional": round(float((b >= 0.5).mean()), 3)}
        print(f"{name:10s} AUC {auc:.3f}  regional caught {out[name]['regional_caught']:.0%}  "
              f"Seoul called regional {out[name]['standard_called_regional']:.0%}  (n {keep.sum()}, Seoul {int((~g).sum())})", flush=True)
    with open(os.path.join(HERE, args.run, "conditions_binary.json"), "w", encoding="utf-8") as fo:
        json.dump({"standard_bias": args.standard_bias, "conditions": out}, fo, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gpu-share", type=float, default=0.12)
    ap.add_argument("--binary", action="store_true", help="standard vs regional (std2 runs, or a 6-zone run's P(서울·경기))")
    ap.add_argument("--standard-bias", type=float, default=0.0, help="the service's bonus for a 6-zone run (served: 1.25)")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)
    clf = load_classifier(os.path.join(HERE, args.run))
    zones = list(getattr(clf, "info", {}).get("zones") or ZONES)
    if args.binary or zones == ["표준", "지역"]:
        return binary(clf, zones, args)
    rows = [json.loads(line) for line in open(os.path.join(HERE, "data", "test_plain.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["zone"] != "서울·경기"
            and {"대구·경북": "경상", "부산·울산·경남": "경상"}.get(r["zone"], r["zone"]) == r["dataset"]]
    gold = np.array([REGIONS.index(r["dataset"]) for r in rows])
    audio = [load_wav(r["path"]) for r in rows]
    out = {}
    for name, f in COND.items():
        ys = [f(x) for x in audio]
        keep = np.array([y is not None for y in ys])
        xs = [nonempty(y) for y in ys if y is not None]
        pred = np.concatenate([clf.probs(xs[i:i + 16]).argmax(1) for i in range(0, len(xs), 16)])
        g = gold[keep]
        rec = {REGIONS[k]: round(float((pred[g == k] == k).mean()), 2) for k in range(4)}
        bal = float(np.mean(list(rec.values())))
        out[name] = {"n": int(keep.sum()), "balanced_recall": round(bal, 3), "per_region": rec,
                     "predicted": dict(collections.Counter(REGIONS[i] for i in pred))}
        print(f"{name:10s} {bal:.3f} (served {BASELINE[name]:.3f})  {rec}  predicted {out[name]['predicted']}", flush=True)
    with open(os.path.join(HERE, args.run, "conditions.json"), "w", encoding="utf-8") as fo:
        json.dump({"n": len(rows), "baseline_served": BASELINE, "conditions": out}, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
