"""Does more training data help the phase 1 head? Two learning curves on the cached features:

    python learning_curve.py                # runs/curve/results.json and a table

  persons  a fraction of the training persons (all their clips), the same fraction in every zone:
           would more speakers help?
  clips    at most N clips per person (every person kept): would more clips of the same speakers help?

Each point trains the phase 1 head (train_head.py's model, loss, schedule and early stopping on dev)
with two seeds and scores the held-out tests. The full-data, seed-0 point repeats runs/phase1.
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from common import ZONES, Head
from train_head import HERE, load, logits_of, moments, report


def fit(Xtr, ytr, idx, dv_rows, Xdv, ydv, seed, epochs=40, patience=6, batch=256, lr=1e-3, wd=0.05):
    """train_head.main's training, on the training rows `idx` only."""
    torch.manual_seed(seed)
    idx = np.sort(idx)
    y = ytr[idx]
    mu, sd = moments(Xtr[idx] if len(idx) < len(ytr) else Xtr)
    head = Head(hidden=256, bottleneck=64, dropout=0.3)
    head.mu.copy_(torch.from_numpy(mu).float()); head.sd.copy_(torch.from_numpy(sd).float())
    head.cuda()
    counts = np.bincount(y, minlength=len(ZONES))
    weight = torch.tensor(len(y) / (len(ZONES) * np.maximum(counts, 1)), dtype=torch.float32).cuda()
    lossf = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(y) + batch - 1) // batch), pct_start=0.1)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep = -1, None, 0, 0
    for ep in range(epochs):
        head.train()
        order = rng.permutation(len(idx))
        for i in range(0, len(order), batch):
            sel = idx[np.sort(order[i:i + batch])]
            loss = lossf(head(torch.from_numpy(np.asarray(Xtr[sel])).cuda()), torch.from_numpy(ytr[sel]).cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        f1 = report(dv_rows, ydv, logits_of(head, Xdv))["macro_f1"]
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head, best, ep + 1


def persons_subset(rows, frac, seed):
    """Rows of `frac` of the training persons, drawn per zone."""
    rng = np.random.default_rng(seed)
    by_zone = collections.defaultdict(set)
    for r in rows:
        by_zone[r["y"]].add(r["person"])
    keep = set()
    for zone, persons in sorted(by_zone.items()):
        persons = sorted(persons)
        keep.update(rng.choice(persons, max(1, round(len(persons) * frac)), replace=False).tolist())
    return np.array([i for i, r in enumerate(rows) if r["person"] in keep])


def clips_subset(rows, cap, seed):
    """At most `cap` clips of every training person."""
    rng = np.random.default_rng(seed)
    by_person = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_person[r["person"]].append(i)
    out = []
    for p in sorted(by_person):
        ids = by_person[p]
        out.extend(ids if len(ids) <= cap else rng.choice(ids, cap, replace=False).tolist())
    return np.array(sorted(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="runs/curve")
    args = ap.parse_args()
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    tr_rows, Xtr, ytr = load("train")
    dv_rows, Xdv, ydv = load("dev")
    tests = {s: load(s) for s in ("test_plain", "test_same", "test_dialect", "bench")}
    points = [("persons", f) for f in (0.25, 0.5, 1.0)] + [("clips", c) for c in (3, 6)]
    path = os.path.join(out, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["curve"], r["value"], r["seed"]) for r in results}
    for curve, value in points:
        for seed in range(args.seeds):
            if (curve, value, seed) in done:
                continue
            idx = persons_subset(tr_rows, value, seed) if curve == "persons" else clips_subset(tr_rows, value, seed)
            t0 = time.time()
            head, dev_f1, epochs = fit(Xtr, ytr, idx, dv_rows, Xdv, ydv, seed)
            r = {"curve": curve, "value": value, "seed": seed, "clips": int(len(idx)),
                 "persons": len({tr_rows[i]["person"] for i in idx}), "epochs": epochs, "dev_macro_f1": dev_f1}
            for s, (rows, X, y) in tests.items():
                m = report(rows, y, logits_of(head, X))
                r[s] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "auc": m["standard_vs_regional_auc"],
                        "persons_macro_f1": m["persons"]["macro_f1"]}
            r["seconds"] = round(time.time() - t0)
            results.append(r)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            print(f"{curve:7s} {value:>5} seed {seed}: {r['clips']:6d} clips, {r['persons']:5d} persons, {epochs:2d} epochs, "
                  f"dev {dev_f1:.3f} | test_plain {r['test_plain']['macro_f1']:.3f} test_same {r['test_same']['macro_f1']:.3f} "
                  f"bench {r['bench']['macro_f1']:.3f} | {r['seconds']} s", flush=True)
    print("\nmean over seeds (macro-F1): curve value | clips persons | dev test_plain test_same test_dialect bench")
    for curve, value in points:
        rs = [r for r in results if r["curve"] == curve and r["value"] == value]
        if rs:
            m = lambda k: np.mean([r[k]["macro_f1"] if isinstance(r[k], dict) else r[k] for r in rs])  # noqa: E731
            print(f"{curve:7s} {value:>5} | {int(m('clips')):6d} {int(m('persons')):5d} | {m('dev_macro_f1'):.3f} "
                  f"{m('test_plain'):.3f} {m('test_same'):.3f} {m('test_dialect'):.3f} {m('bench'):.3f}")


if __name__ == "__main__":
    main()
