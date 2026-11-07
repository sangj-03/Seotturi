"""The accent models on the browser test set: recordings made through the site's own recorder at /record
(seotturi-web data/testset/<session>/: meta.json, s00..s19.wav read sentences, f0/f1.wav free speech, each
with the clip cuts the app would have sent). Every take goes through what the app does with it (the
browser's clips -> analysis.preprocess -> split_audio), and each speaker's clips are pooled by length, as
the app pools one answer.

    python eval_browser.py                     # all sessions; writes runs/browser_eval.json

Models: the served six-zone ensemble (runs/ensemble_big, standard bonus 1.25 as in the service) and the
4-region run with the standard detector (runs/region4 + runs/detector_mix: P(zone) = P(regional) *
P(zone | regional), 서울·경기 = 1 - P(regional)). Truth: the region the speaker grew up in.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

from common import Encoder, Tree, load_classifier, load_wav

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = "/data/seotturi/seotturi-web"
sys.path.insert(0, os.path.join(WEB, "server"))
import analysis as A  # noqa: E402

PLACE = {"서울": "서울·경기", "경기": "서울·경기", "인천": "서울·경기", "강원": "강원", "충북": "충청", "충남": "충청",
         "대전": "충청", "세종": "충청", "경북": "경상", "대구": "경상", "경남": "경상", "부산": "경상", "울산": "경상",
         "제주": "제주", "전북": "전라", "전남": "전라", "광주": "전라"}
MERGE = {"대구·경북": "경상", "부산·울산·경남": "경상"}


def clips_of(d, item):
    """What the app would analyse for one take: the browser's clips, preprocessed and split."""
    x = load_wav(os.path.join(d, item + ".wav"))
    cuts = json.load(open(os.path.join(d, item + ".json"), encoding="utf-8")).get("segments") or [[0, len(x) / 16000]]
    out = []
    for a, b in cuts:
        clip, voiced = A.preprocess(x[int(a * 16000):int(b * 16000)])
        if voiced >= 0.25:
            out += [c for c in A.split_audio(clip) if len(c) >= 3200]
    return out


class Served:
    def __init__(self, bias=1.25):
        self.clf, self.bias = load_classifier(os.path.join(HERE, "runs", "ensemble_big")), bias

    def probs(self, audios):
        p = self.clf.probs(audios)
        p[:, 0] *= np.exp(self.bias)  # zone 0 is 서울·경기
        p /= p.sum(1, keepdims=True)
        zones = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "제주"]
        out = []
        for row in p:
            m = collections.defaultdict(float)
            for z, v in zip(zones, row):
                m[MERGE.get(z, z)] += float(v)
            out.append(dict(m))
        return out


class Hierarchical:
    """What seotturi-web server/accent.py serves with SEOTTURI_ACCENT_RUN=region and _STD_RUN=std: one shared
    encoder, P(regional) from the detector (its 서울·경기 probability after the calibrated bonus, or a std2 head)."""

    def __init__(self, region="runs/region4", std="runs/detector_mix"):
        self.enc = Encoder()
        self.region, self.std = Tree(os.path.join(HERE, region), self.enc), Tree(os.path.join(HERE, std), self.enc)
        spec = os.path.join(HERE, std, "ensemble.json")
        self.bias = json.load(open(spec, encoding="utf-8")).get("served_standard_bias", 0.0) if os.path.exists(spec) else 0.0

    def probs(self, audios):
        x = self.enc.pooled(audios)
        p, q = self.region.probs(audios, x), self.std.probs(audios, x)
        if self.std.zones == ["표준", "지역"]:
            r = q[:, 1]
        else:
            k = self.std.zones.index("서울·경기")
            s = q[:, k] * np.exp(self.bias)
            r = 1 - s / (s + 1 - q[:, k])
        return [{"서울·경기": float(1 - ri), **{z: float(ri * v) for z, v in zip(self.region.zones, pi)}} for pi, ri in zip(p, r)]


def pooled(model, clips):
    if not clips:
        return None
    ps = [q for i in range(0, len(clips), 16) for q in model.probs(clips[i:i + 16])]
    return A.mean_probs([(p, len(c)) for p, c in zip(ps, clips)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(WEB, "data", "testset"))
    ap.add_argument("--gpu-share", type=float, default=0.2)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)
    sessions = sorted(s for s in os.listdir(args.dir) if os.path.exists(os.path.join(args.dir, s, "meta.json")))
    if not sessions:
        print("no sessions yet in", args.dir)
        return
    models = {"served": Served(), "new": Hierarchical()}
    rows = []
    for sid in sessions:
        d = os.path.join(args.dir, sid)
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        items = sorted(n[:-4] for n in os.listdir(d) if n.endswith(".wav"))
        parts = {"sentences": [c for i in items if i.startswith("s") for c in clips_of(d, i)],
                 "free": [c for i in items if i.startswith("f") for c in clips_of(d, i)]}
        row = {"session": sid, "name": meta.get("name"), "truth": PLACE.get(meta.get("raised"), "기타"),
               "current": PLACE.get(meta.get("current"), "기타"), "self": meta.get("self_rating"), "items": len(items)}
        for m, model in models.items():
            for part, clips in parts.items():
                p = pooled(model, clips)
                if p:
                    zone = max(p, key=p.get)
                    row[f"{m}_{part}"] = {"zone": zone, "conf": round(p[zone], 3), "regional": round(1 - p["서울·경기"], 3)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print("\nspeakers:", dict(collections.Counter(r["truth"] for r in rows)))
    for m in models:
        for part in ("sentences", "free"):
            k = f"{m}_{part}"
            got = [r for r in rows if k in r]
            reg = [r for r in got if r["truth"] in ("강원", "충청", "경상", "제주")]
            std = [r for r in got if r["truth"] == "서울·경기"]
            line = f"{k:16s}"
            if reg:
                line += (f" regional speakers {len(reg)}: zone right {np.mean([r[k]['zone'] == r['truth'] for r in reg]):.0%},"
                         f" called regional {np.mean([r[k]['regional'] >= 0.5 for r in reg]):.0%}")
            if std:
                line += f" | standard speakers {len(std)}: called regional {np.mean([r[k]['regional'] >= 0.5 for r in std]):.0%}"
            print(line)
    with open(os.path.join(HERE, "runs", "browser_eval.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
