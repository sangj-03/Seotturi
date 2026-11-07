"""Phase 1: train the head on pooled features of the frozen encoder, then score it on every test set.

    python train_head.py --out runs/phase1

Model selection uses the dev set (persons held out of training); the test sets are only scored at the
end. Writes runs/<out>/head.pt (what common.Classifier loads) and metrics.json.

Scores, all on persons never seen in training:
  test_plain    AI Hub Validation clips without dialect words, 200 per zone: the main number
  test_same     one sentence said by speakers of several zones: accent alone decides
  test_dialect  the Gemma validation clips, dialect words included
  bench         the engine's zone benchmark (the served Gemma adapter scores macro-F1 0.216 on it, 7 zones)
For each: macro-F1 over the six zones, accuracy, the standard-vs-regional AUC, how many standard
speakers are called regional, accuracy with Gyeongsang as one zone, and per-person accuracy when all
of a person's clips are pooled (what the app does with several sentences).
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from common import DIM, LAYERS, STANDARD, ZONES, Head

REGIONS4 = ["강원", "충청", "경상", "제주"]  # --zones region4 (region4.py): no standard zone
STD2 = ["표준", "지역"]  # --zones std2 (region4.py --std2): standard-vs-regional detector

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FEAT = os.path.join(HERE, "features")
COARSE = [0, 1, 2, 3, 3, 4]  # Gyeongsang (대구·경북, 부산·울산·경남) as one zone


def load(split):
    with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    X = np.load(os.path.join(FEAT, f"{split}.npy"), mmap_mode="r")
    with open(os.path.join(FEAT, f"{split}.ids.json")) as f:
        assert json.load(f) == [r["id"] for r in rows], f"{split}: features and data/{split}.jsonl differ, rerun features.py"
    return rows, X, np.array([r["y"] for r in rows])


def moments(X, chunk=4096):
    s = np.zeros((LAYERS, DIM)); ss = np.zeros((LAYERS, DIM))
    for i in range(0, len(X), chunk):
        x = np.asarray(X[i:i + chunk], dtype=np.float64)
        s += x.sum(0)
        ss += (x * x).sum(0)
    mu = s / len(X)
    return mu, np.sqrt(np.maximum(ss / len(X) - mu * mu, 1e-8))


@torch.no_grad()
def logits_of(head, X, bs=1024):
    head.eval()
    out = [head(torch.from_numpy(np.asarray(X[i:i + bs])).cuda()).float().cpu() for i in range(0, len(X), bs)]
    return torch.cat(out)


def f1s(y, p, k):
    f = []
    for c in range(k):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return f


def auc(score, pos):
    a, b = score[pos], score[~pos]
    if not len(a) or not len(b):
        return None
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def report(rows, y, logits):
    prob = torch.softmax(logits, -1).numpy()
    pred = prob.argmax(1)
    zones = sorted(set(y.tolist()))
    f = f1s(y, pred, len(ZONES))
    regional = y != STANDARD if STANDARD is not None else np.ones(len(y), bool)
    called = pred != STANDARD if STANDARD is not None else np.ones(len(y), bool)
    out = {"n": int(len(y)), "macro_f1": round(float(np.mean([f[k] for k in zones])), 3),
           "accuracy": round(float((pred == y).mean()), 3),
           "per_zone_f1": {ZONES[k]: round(f[k], 3) for k in zones},
           "standard_vs_regional_auc": None if STANDARD not in zones else round(auc(1 - prob[:, STANDARD], regional), 3),
           "regional_caught": f"{int((called & regional).sum())}/{int(regional.sum())}",
           "standard_called_regional": f"{int((called & ~regional).sum())}/{int((~regional).sum())}",
           "coarse_accuracy": None if COARSE is None else round(float((np.array(COARSE)[pred] == np.array(COARSE)[y]).mean()), 3)}
    conf = collections.Counter(zip(y.tolist(), pred.tolist()))
    out["confusion"] = {ZONES[t]: {ZONES[p]: conf[(t, p)] for p in range(len(ZONES)) if conf[(t, p)]} for t in zones}
    # one decision per person from all their clips (sum of log-probabilities)
    lp = collections.defaultdict(lambda: np.zeros(len(ZONES)))
    truth = {}
    for r, l in zip(rows, np.log(prob + 1e-9)):
        lp[r["person"]] += l
        truth[r["person"]] = r["y"]
    persons = list(lp)
    py = np.array([truth[p] for p in persons]); pp = np.array([lp[p].argmax() for p in persons])
    out["persons"] = {"n": len(persons), "clips_per_person": round(len(rows) / len(persons), 1),
                      "macro_f1": round(float(np.mean([f1s(py, pp, len(ZONES))[k] for k in zones])), 3),
                      "accuracy": round(float((pp == py).mean()), 3)}
    ages = collections.defaultdict(list)
    for r, ok in zip(rows, pred == y):
        ages[r.get("age") or "?"].append(ok)
    out["accuracy_by_age"] = {a: f"{np.mean(v):.2f} (n={len(v)})" for a, v in sorted(ages.items())}
    return out


def fit_temperature(logits, y):
    best = (1.0, 1e9)
    for t in np.arange(0.5, 4.01, 0.05):
        nll = torch.nn.functional.cross_entropy(logits / t, torch.from_numpy(y)).item()
        best = min(best, (float(t), nll), key=lambda x: x[1])
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase1")
    ap.add_argument("--train", default="train", help="training split (data/<name>.jsonl + features/<name>.npy)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--bottleneck", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zones", choices=["zone6", "region4", "std2"], default="zone6",
                    help="region4: 강원/충청/경상/제주 by recording set, no standard zone (region4.py)")
    ap.add_argument("--dev", default="dev")
    ap.add_argument("--tests", default="test_plain,test_same,test_dialect,bench")
    args = ap.parse_args()
    global ZONES, STANDARD, COARSE
    if args.zones == "region4":
        ZONES, STANDARD, COARSE = REGIONS4, None, None
    elif args.zones == "std2":
        ZONES, STANDARD, COARSE = STD2, 0, None
    tests = args.tests.split(",")
    torch.manual_seed(args.seed)
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    tr_rows, Xtr, ytr = load(args.train)
    dv_rows, Xdv, ydv = load(args.dev)
    print(f"train {len(ytr)} clips, dev {len(ydv)}", flush=True)
    mu, sd = moments(Xtr)
    config = {"hidden": args.hidden, "bottleneck": args.bottleneck, "dropout": args.dropout, "n_out": len(ZONES)}
    head = Head(**config)
    head.mu.copy_(torch.from_numpy(mu).float()); head.sd.copy_(torch.from_numpy(sd).float())
    head.cuda()
    counts = np.bincount(ytr, minlength=len(ZONES))
    weight = torch.tensor(len(ytr) / (len(ZONES) * np.maximum(counts, 1)), dtype=torch.float32).cuda()
    lossf = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=args.smoothing)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * ((len(ytr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(args.seed)
    best, best_state, bad, history = -1, None, 0, []
    for epoch in range(args.epochs):
        head.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), args.batch):
            idx = np.sort(order[i:i + args.batch])
            x = torch.from_numpy(np.asarray(Xtr[idx])).cuda()
            loss = lossf(head(x), torch.from_numpy(ytr[idx]).cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        dev = report(dv_rows, ydv, logits_of(head, Xdv))
        history.append({"epoch": epoch, "loss": round(total / len(ytr), 4), "dev_macro_f1": dev["macro_f1"]})
        print(f"epoch {epoch}: loss {total / len(ytr):.4f}, dev macro-F1 {dev['macro_f1']:.3f} "
              f"(acc {dev['accuracy']:.3f}, std-vs-regional AUC {dev['standard_vs_regional_auc']}) {time.time() - t0:.0f}s", flush=True)
        if dev["macro_f1"] > best:
            best, bad = dev["macro_f1"], 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    head.load_state_dict(best_state)
    t = fit_temperature(logits_of(head, Xdv), ydv)
    head.temperature.fill_(t)
    metrics = {"dev": report(dv_rows, ydv, logits_of(head, Xdv)), "temperature": t, "history": history,
               "layer_weights": [round(float(w), 4) for w in torch.softmax(head.mix.detach().cpu(), 0)]}
    test_logits = {}
    for split in tests:
        rows, X, y = load(split)
        test_logits[split] = logits_of(head, X)
        metrics[split] = report(rows, y, test_logits[split])
        m = metrics[split]
        print(f"{split}: macro-F1 {m['macro_f1']} acc {m['accuracy']} AUC {m['standard_vs_regional_auc']} "
              f"std→regional {m['standard_called_regional']} coarse {m['coarse_accuracy']} persons {m['persons']}", flush=True)
    torch.save({"config": config, "state": {k: v.cpu() for k, v in head.state_dict().items()},
                "info": {"zones": ZONES, "args": vars(args), "test_plain_macro_f1": metrics[tests[0]]["macro_f1"],
                         "bench_macro_f1": metrics[tests[-1]]["macro_f1"]}}, os.path.join(out, "head.pt"))
    np.savez(os.path.join(out, "test_logits.npz"), **{k: v.numpy() for k, v in test_logits.items()})  # for leakage.py
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: metrics[k] for k in tests[:1]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
