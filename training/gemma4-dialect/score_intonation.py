"""Score evaluate.py predictions on the intonation test set (see prepare_intonation.py).

    python score_intonation.py data_intonation/test.jsonl base=runs/intonation_base v1=runs/intonation_v1

question  Share of questions transcribed with a final "?" and of statements transcribed
          without one. Their mean (balanced accuracy) is 50% for a model that ignores
          intonation. Also reported only for utterances whose words were heard right,
          so recognition errors do not hide the intonation result.
region    Character error rate per accent zone on sentences said in three or more zones,
          and the accent penalty: how much worse a zone does than the average of the
          other zones on the same sentence (0 = accent makes no difference).
zone      For models trained with `--format region`: how often the `지역:` line names the
          speaker's zone on the plain speech of the region and origin subsets, where
          no dialect word gives it away (1 in 7 by chance).
"""

import json
import sys
from collections import Counter, defaultdict

from prepare_intonation import STRIP, is_question


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def balanced(rows):
    q = [r for r in rows if r["is_question"]]
    s = [r for r in rows if not r["is_question"]]
    if not q or not s:
        return None
    hit = sum(r["pred_q"] for r in q) / len(q)
    false = sum(r["pred_q"] for r in s) / len(s)
    return {"n": len(rows), "q_hit": hit, "s_false": false, "balanced": (hit + 1 - false) / 2}


def score(test, preds):
    rows = []
    for t in test:
        p = preds.get(t["id"])
        if p is None:
            continue
        rows.append({**t, "zone": t.get("origin") or t["region"], "pred_region": p.get("pred_region"),
                     "pred_q": is_question(p["dialect"]),
                     "heard": STRIP.sub("", p["dialect"]) == t["group"],
                     "err": edits(t["group"], STRIP.sub("", p["dialect"])), "len": len(t["group"])})
    out = {"missing": len(test) - len(rows)}

    q = [r for r in rows if r["subset"] == "question"]
    out["question"] = {
        "all": balanced(q),
        "heard_right": balanced([r for r in q if r["heard"]]),
        "one_word": balanced([r for r in q if r["n_words"] == 1]),
        "two_plus_words": balanced([r for r in q if r["n_words"] >= 2]),
        "regions": {g: balanced([r for r in q if r["region"] == g]) for g in sorted({r["region"] for r in q})},
    }

    reg = [r for r in rows if r["subset"] == "region"]
    by_group = defaultdict(list)
    for r in reg:
        by_group[r["group"]].append(r)
    penalty = defaultdict(list)
    for rs in by_group.values():
        if len(rs) < 2:
            continue
        for r in rs:
            others = [x["err"] / x["len"] for x in rs if x is not r]
            penalty[r["zone"]].append(r["err"] / r["len"] - sum(others) / len(others))
    out["region"] = {g: {"n": len(rs), "cer": sum(r["err"] for r in rs) / sum(r["len"] for r in rs),
                         "exact": sum(r["heard"] for r in rs) / len(rs),
                         "penalty": sum(penalty[g]) / len(penalty[g]) if penalty[g] else 0.0}
                     for g in sorted({r["zone"] for r in reg})
                     for rs in [[r for r in reg if r["zone"] == g]]}

    z = [r for r in rows if r["subset"] in ("region", "origin") and r.get("origin")]
    if any(r["pred_region"] for r in z):
        confusion = defaultdict(Counter)
        for r in z:
            confusion[r["origin"]][r["pred_region"] or "-"] += 1
        out["zone"] = {"n": len(z), "accuracy": sum(r["pred_region"] == r["origin"] for r in z) / len(z),
                       "per_zone": {g: (c[g] / sum(c.values()), sum(c.values())) for g, c in sorted(confusion.items())},
                       "confusion": {g: dict(c.most_common()) for g, c in sorted(confusion.items())}}
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def main():
    test = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    res = {}
    for name, d in runs:
        path = d if d.endswith(".jsonl") else f"{d}/predictions.jsonl"
        res[name] = score(test, {p["id"]: p for p in map(json.loads, open(path, encoding="utf-8"))})
    names = [n for n, _ in runs]

    print("## Question vs statement (same words; 50% = intonation ignored)\n")
    print("| | " + " | ".join(names) + " |")
    print("| --- |" + " ---: |" * len(names))
    q0 = res[names[0]]["question"]
    lines = [("all", "all"), ("words heard right", "heard_right"), ("one word", "one_word"),
             ("two or more words", "two_plus_words")] + [(g, g) for g in q0["regions"]]
    for label, key in lines:
        cells = []
        for n in names:
            b = res[n]["question"]["regions"].get(key) if key in q0["regions"] else res[n]["question"][key]
            cells.append("–" if b is None else f"{pct(b['balanced'])} ({pct(b['q_hit'])} / {pct(b['s_false'])}, n={b['n']})")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nCells: balanced accuracy (questions given ?, statements given ?, n). Rows below 'two or more words' are datasets.\n")

    print("## Same sentence, different accent zones\n")
    print("| zone | " + " | ".join(f"CER {n}" for n in names) + " | "
          + " | ".join(f"accent penalty {n}" for n in names) + " | n |")
    print("| --- |" + " ---: |" * (2 * len(names) + 1))
    for g in res[names[0]]["region"]:
        r = [res[n]["region"][g] for n in names]
        print(f"| {g} | " + " | ".join(f"{x['cer']:.3f}" for x in r) + " | "
              + " | ".join(f"{x['penalty']:+.3f}" for x in r) + f" | {r[0]['n']} |")
    print("\nAccent penalty: CER minus the other zones' average CER on the same sentence (0 = no accent effect).")

    zoned = [n for n in names if "zone" in res[n]]
    if zoned:
        print("\n## Naming the speaker's zone from plain speech (1 in 7 by chance)\n")
        print("| zone | " + " | ".join(zoned) + " | n |")
        print("| --- |" + " ---: |" * (len(zoned) + 1))
        first = res[zoned[0]]["zone"]
        for g, (_, n) in first["per_zone"].items():
            print(f"| {g} | " + " | ".join(pct(res[m]["zone"]["per_zone"][g][0]) for m in zoned) + f" | {n} |")
        print(f"| all | " + " | ".join(pct(res[m]["zone"]["accuracy"]) for m in zoned) + f" | {first['n']} |")
        last = zoned[-1]
        print(f"\nConfusion for {last} (true zone: predicted counts):\n")
        for g, c in res[last]["zone"]["confusion"].items():
            print(f"- {g}: " + ", ".join(f"{p} {k}" for p, k in c.items()))

    missing = {n: res[n]["missing"] for n in names if res[n]["missing"]}
    if missing:
        print(f"\nMissing predictions: {missing}")


if __name__ == "__main__":
    main()
