"""Compare evaluate.py runs on the same manifest, per region, with intervals that say whether a difference is real.

    python score_eval.py runs/eval_full/v1 runs/eval_full/v3_500 ...   [--base v1] [--out runs/eval_full/report.json]

Three character error rates for each line (dialect transcript, standard conversion):
  cer      characters, spaces and punctuation ignored (as evaluate.py and the engine benchmark)
  jamo     the same over jamo, so a near spelling (햄수다 / 헴수다) costs one jamo, not a syllable
  near     the output's words that are one jamo away from the aligned reference word (햄수다/헴수다, 불면/불믄)
           count as right, then characters: dialect spelling is not standardised (Ali et al. 2017, WERd;
           Ali, Khalifa & Habash 2019), and AI Hub's transcribers spell one dialect word several ways.
           (Mapping words to one canonical spelling per variant group was tried first and made some pairs
           worse: one side moved, the other did not.)
Intervals: block bootstrap over recordings (1,000 resamples). Against --base: the share of resamples in which
the run is better (fewer edits), on the same utterances.
"""

import argparse
import collections
import difflib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRIP = re.compile(r"[^가-힣0-9]")
B = 1000


def jamo(s):
    out = []
    for ch in s:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out += [chr(0x1100 + o // 588), chr(0x1161 + o % 588 // 28)] + ([chr(0x11A7 + o % 28)] if o % 28 else [])
        else:
            out.append(ch)
    return "".join(out)


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def words(s):
    return [w for w in (STRIP.sub("", t) for t in s.split()) if w]


def near_fix(ref, hyp):
    """hyp with every word one jamo from its aligned reference word replaced by that word."""
    r, h = words(ref), words(hyp)
    out = list(h)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for a, j in zip(r[i1:i2], range(j1, j2)):
                ja, jh = jamo(a), jamo(h[j])
                if edits(ja, jh) <= 1 and len(ja) >= 3:
                    out[j] = a
    return " ".join(out)


def counts(ref, hyp, kind):
    if kind == "near":
        hyp = near_fix(ref, hyp)
    r, h = STRIP.sub("", ref), STRIP.sub("", hyp)
    if kind == "jamo":
        r, h = jamo(r), jamo(h)
    return edits(r, h), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base", default=None, help="run name (folder name) the others are compared with")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "eval_full", "report.json"))
    args = ap.parse_args()
    runs = {os.path.basename(r.rstrip("/")): {p["id"]: p for p in map(json.loads, open(os.path.join(r, "predictions.jsonl"), encoding="utf-8"))}
            for r in args.runs}
    ids = sorted(set.intersection(*(set(v) for v in runs.values())))
    first = runs[next(iter(runs))]
    regions = sorted({first[i]["region"] for i in ids})
    rng = np.random.default_rng(0)
    report = {"n": len(ids), "regions": {}}
    base = args.base or next(iter(runs))
    for region in ["전체"] + regions:
        sel = [i for i in ids if region == "전체" or first[i]["region"] == region]
        blocks = collections.defaultdict(list)
        for k, i in enumerate(sel):
            blocks[i.split(".")[0]].append(k)
        blocks = [np.array(v) for v in blocks.values()]
        samples = [np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))]) for _ in range(B)]
        out = {}
        for line in ("dialect", "standard"):
            for kind in ("cer", "jamo", "near"):
                mats = {}
                for name, preds in runs.items():
                    e = np.array([counts(preds[i][f"ref_{line}"], preds[i][line] or "", kind) for i in sel], dtype=float)
                    mats[name] = e
                key = f"{line}_{kind}"
                out[key] = {}
                for name, e in mats.items():
                    boot = [e[s, 0].sum() / e[s, 1].sum() for s in samples]
                    row = {"value": round(e[:, 0].sum() / e[:, 1].sum(), 4),
                           "ci": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]}
                    if name != base:
                        b = mats[base]
                        row["p_better"] = round(float(np.mean([e[s, 0].sum() < b[s, 0].sum() for s in samples])), 3)
                    out[key][name] = row
        report["regions"][region] = {"n": len(sel), **out}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    names = list(runs)
    for line in ("dialect", "standard"):
        print(f"\n[{'방언 줄' if line == 'dialect' else '표준어 줄'}] value [95% CI] (p_better vs {base})")
        for region, r in report["regions"].items():
            cells = []
            for kind in ("cer", "jamo", "near"):
                cells.append(f"{kind}: " + " | ".join(
                    f"{n} {r[f'{line}_{kind}'][n]['value']:.3f}" + (f" ({r[f'{line}_{kind}'][n]['p_better']:.2f})" if n != base else "")
                    for n in names))
            print(f"  {region} (n {r['n']}) " + "  ||  ".join(cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
