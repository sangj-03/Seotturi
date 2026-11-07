"""Do test voices also occur in training? AI Hub has no person ids, so this compares voices.

    python leakage.py embed     # data/voices.npz: speaker embeddings of every train slot and test clip
    python leakage.py report runs/phase1   # scores on test clips whose voice has no close match in training

Embeddings: microsoft/wavlm-base-plus-sv, two clips per training slot (averaged), every test clip alone.
Reference points from the same model: two clips of one slot (the same person in one recording) and the two
speakers of one recording (different people, same phone and room). A test clip counts as "possibly seen"
when some training slot of the same zone and sex is at least as similar as the threshold; the report gives
the scores on the remaining clips for a range of thresholds, next to the scores on all clips.
"""

import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TESTS = ("test_plain", "test_same", "test_dialect")


def read(split):
    with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed():
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    from common import load_wav
    torch.cuda.set_per_process_memory_fraction(0.08)  # the GPU is shared with the app's engine
    name = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = WavLMForXVector.from_pretrained(name).cuda().eval()
    train = read("train")
    by_slot = collections.defaultdict(list)
    for r in train:
        by_slot[r["slot"]].append(r)
    items = [("train", s, r) for s, rs in by_slot.items() for r in rs[:2]]
    items += [(t, r["id"], r) for t in TESTS for r in read(t)]
    items.sort(key=lambda x: x[2]["duration"])
    vecs = {}
    with torch.no_grad():
        for i in range(0, len(items), 16):
            batch = items[i:i + 16]
            feats = fe([load_wav(r["path"])[:8 * 16000] for _, _, r in batch], sampling_rate=16000, padding=True,
                       return_tensors="pt")
            e = torch.nn.functional.normalize(model(**{k: v.cuda() for k, v in feats.items()}).embeddings.float(), dim=-1)
            for (kind, key, _), v in zip(batch, e.cpu().numpy()):
                vecs.setdefault((kind, key), []).append(v)
            if i % 8000 == 0:
                print(f"{i}/{len(items)}", flush=True)
    keys = sorted(vecs)
    np.savez(os.path.join(DATA, "voices.npz"), keys=np.array([f"{k}|{s}" for k, s in keys]),
             first=np.stack([vecs[k][0] for k in keys]), mean=np.stack([np.mean(vecs[k], 0) for k in keys]),
             n=np.array([len(vecs[k]) for k in keys]))
    print(f"wrote {len(keys)} voices", flush=True)


def unit(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def report(run):
    sys.path.insert(0, HERE)
    import torch
    from train_head import report as score
    z = np.load(os.path.join(DATA, "voices.npz"))
    keys = [k.split("|", 1) for k in z["keys"]]
    idx = {(k, s): i for i, (k, s) in enumerate(keys)}
    first, mean, n = z["first"], unit(z["mean"]), z["n"]
    train = read("train")
    meta = {r["slot"]: (r["zone"], r["sex"], r["recording"]) for r in train}
    # reference: the two people of one recording (different people, same phone and room)
    by_rec = collections.defaultdict(list)
    for (k, s), i in idx.items():
        if k == "train":
            by_rec[meta[s][2]].append(i)
    diff = [float(mean[a] @ mean[b]) for ids in by_rec.values() if len(ids) == 2 for a, b in [ids]]
    print(f"reference: the two people of one recording, median similarity {np.median(diff):.3f} "
          f"(90th percentile {np.quantile(diff, 0.9):.3f})")
    groups = collections.defaultdict(list)
    for (k, s), i in idx.items():
        if k == "train":
            groups[meta[s][:2]].append(i)
    gm = {g: mean[ids] for g, ids in groups.items()}
    metrics = json.load(open(os.path.join(HERE, run, "metrics.json"), encoding="utf-8"))
    preds = np.load(os.path.join(HERE, run, "test_logits.npz")) if os.path.exists(os.path.join(HERE, run, "test_logits.npz")) else None
    for t in TESTS:
        rows = read(t)
        best = np.array([float((gm.get((r["zone"], r["sex"]), np.zeros((1, mean.shape[1]))) @ unit(first[idx[(t, r["id"])]])).max())
                         for r in rows])
        print(f"\n{t}: nearest training voice of the same zone and sex, median {np.median(best):.3f}, "
              f"share ≥0.95 {np.mean(best >= 0.95):.1%}, ≥0.97 {np.mean(best >= 0.97):.1%}")
        if preds is None:
            continue
        logits = torch.from_numpy(preds[t])
        y = np.array([r["y"] for r in rows])
        for thr in (1.01, 0.97, 0.95, 0.93, 0.90):
            keep = best < thr
            if keep.sum() < 50:
                continue
            m = score([r for r, k in zip(rows, keep) if k], y[keep], logits[keep])
            label = "all clips" if thr > 1 else f"nearest < {thr:.2f}"
            print(f"  {label:16s} n={int(keep.sum()):5d} macro-F1 {m['macro_f1']} acc {m['accuracy']} AUC {m['standard_vs_regional_auc']}")


if __name__ == "__main__":
    {"embed": lambda: embed(), "report": lambda: report(sys.argv[2])}[sys.argv[1]]()
