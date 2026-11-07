"""Benchmark metrics for the 서투리 dialect model.

    .venv-tts/bin/python -I scripts/bench_score.py predictions.jsonl [--format basic|region] > summary.json

Runs in .venv-tts because chrF++ comes from sacreBLEU, the reference implementation. One JSON object
per prediction line:
  {"kind": "asr"|"zone", "id", "block": recording id, "region": regional set, "origin": speaker's
   accent zone or null, "has_dialect", "ref_dialect", "ref_standard", "hyp_dialect", "hyp_standard",
   "hyp_zone", "duration": seconds of audio, "seconds": generation time (optional)}
"asr" rows are AI Hub Validation utterances with dialect and standard references; "zone" rows are
plain speech (no dialect words) for accent-zone identification.

Metrics (the website's benchmark page lists the sources):
  CER    syllable-level Levenshtein distance / reference length, spaces and punctuation removed,
         summed over the set (corpus level) - the main Korean ASR metric (KsponSpeech; Kim et al. 2021)
  WER    eojeol-level; sWER: WER after re-spacing the hypothesis to the reference's spacing, because
         Korean spacing is inconsistent (Bang et al. 2020; Kim et al. 2021)
  chrF++ of the standard line: character 6-grams + word 2-grams, beta 2 (Popović 2017; sacreBLEU)
  ERR    word error reduction of the standard line relative to leaving the model's own dialect
         transcript as is (leave-as-is baseline; van der Goot et al. 2021; Kuparinen et al. 2023)
  교정률  share of the references' dialect->standard word changes found in the standard line
  zone   accuracy (AESRC 2020) and macro-F1 over zones (NADI shared tasks), with a confusion matrix
  95% confidence intervals: blockwise bootstrap over recordings, 1,000 resamples
         (Bisani & Ney 2004; Liu et al. 2020: utterances of one recording are not independent)
  RTF    generation time / audio duration

    bench_score.py candidate.jsonl --compare served.jsonl   (the engine's update gate, see compare())
"""

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from sacrebleu.metrics import CHRF

PUNCT = re.compile(r"[.,?!~…·\"'“”‘’()\[\]{}<>/:;\-]")
ZONES = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "전라", "제주"]
ITERS = 1000


def norm(s):
    return " ".join(PUNCT.sub(" ", s or "").split())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def chars(ref, hyp):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return lev(r, h), len(r)


def words(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    return lev(r, h), len(r)


def respace(ref, hyp):
    """The hypothesis with the reference's spacing (for sWER): align the space-free character
    sequences and start a word wherever the aligned reference character starts one."""
    rw = norm(ref).split()
    r = "".join(rw)
    starts, k = set(), 0
    for w in rw:
        starts.add(k)
        k += len(w)
    h = norm(hyp).replace(" ", "")
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    align, i, j = [None] * m, n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            align[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif d[i, j] == d[i, j - 1] + 1:
            j -= 1
        else:
            i -= 1
    out = []
    for j, ch in enumerate(h):
        if j and align[j] is not None and align[j] in starts:
            out.append(" ")
        out.append(ch)
    return "".join(out)


def swords(ref, hyp):
    return words(ref, respace(ref, hyp))


def ratio_ci(blocks, num, den):
    """Corpus-level ratio sum(num)/sum(den) and its 95% blockwise-bootstrap interval."""
    agg = defaultdict(lambda: [0.0, 0.0])
    for b, x, y in zip(blocks, num, den):
        agg[b][0] += x
        agg[b][1] += y
    arr = np.array(list(agg.values()))
    value = float(arr[:, 0].sum() / max(arr[:, 1].sum(), 1))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(arr), size=(ITERS, len(arr)))
    s = arr[idx].sum(axis=1)
    vals = s[:, 0] / np.maximum(s[:, 1], 1)
    return {"value": round(value, 4), "ci": [round(float(np.percentile(vals, 2.5)), 4),
                                             round(float(np.percentile(vals, 97.5)), 4)]}


def changes(dialect, standard):
    """Word-level dialect -> standard changes of a reference pair."""
    a, b = norm(dialect).split(), norm(standard).split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op in ("replace", "insert") and j2 > j1:
            out.append(b[j1:j2])
    return out


def macro_f1(gold, pred):
    labels = [z for z in ZONES if z in gold]
    f1 = {}
    for z in labels:
        tp = sum(g == z and p == z for g, p in zip(gold, pred))
        fp = sum(g != z and p == z for g, p in zip(gold, pred))
        fn = sum(g == z and p != z for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[z] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (sum(f1.values()) / len(f1) if f1 else 0.0), f1


def zone_metrics(rows):
    rows = [r for r in rows if r.get("origin")]
    if not rows:
        return None
    gold = [r["origin"] for r in rows]
    pred = [r.get("hyp_zone") or "-" for r in rows]
    macro, per = macro_f1(gold, pred)
    blocks = sorted({r["block"] for r in rows})
    by_block = defaultdict(list)
    for i, r in enumerate(rows):
        by_block[r["block"]].append(i)
    rng = np.random.default_rng(0)
    accs, f1s = [], []
    for _ in range(ITERS):
        pick = [i for b in rng.choice(blocks, size=len(blocks)) for i in by_block[b]]
        g, p = [gold[i] for i in pick], [pred[i] for i in pick]
        accs.append(sum(x == y for x, y in zip(g, p)) / len(pick))
        f1s.append(macro_f1(g, p)[0])
    acc = sum(g == p for g, p in zip(gold, pred)) / len(rows)
    return {"n": len(rows),
            "accuracy": {"value": round(acc, 4), "ci": [round(float(np.percentile(accs, 2.5)), 4),
                                                        round(float(np.percentile(accs, 97.5)), 4)]},
            "macro_f1": {"value": round(macro, 4), "ci": [round(float(np.percentile(f1s, 2.5)), 4),
                                                          round(float(np.percentile(f1s, 97.5)), 4)]},
            "per_zone_f1": {z: round(v, 4) for z, v in per.items()},
            "confusion": {g: dict(Counter(p for gg, p in zip(gold, pred) if gg == g).most_common())
                          for g in [z for z in ZONES if z in gold]}}


def compare(base_rows, cand_rows, iters=10000):
    """Paired comparison of two models on the same "asr" clips, for the engine's update gate. For each
    measure: both values, the difference candidate - base, its 95% interval and the probability of
    improvement p_better (share of resamples where the candidate is lower; Bisani & Ney 2004), from one
    blockwise bootstrap over recordings that resamples both models together. Pairing removes how hard each
    clip is, so the interval of the difference is much narrower than the two separate intervals suggest."""
    base = {r["id"]: r for r in base_rows if r["kind"] == "asr"}
    cand = {r["id"]: r for r in cand_rows if r["kind"] == "asr"}
    if set(base) != set(cand):
        raise SystemExit(f"different clips: {len(set(base) ^ set(cand))} ids in only one of the files")
    blocks = sorted({r["block"] for r in base.values()})
    pos = {b: k for k, b in enumerate(blocks)}
    # per recording: edits of base, edits of candidate, reference length (dialect, standard), format failures, clips
    agg = np.zeros((len(blocks), 9))
    for i, b in base.items():
        c, k = cand[i], pos[b["block"]]
        for j, side in enumerate(("dialect", "standard")):
            eb, n = chars(b[f"ref_{side}"], b[f"hyp_{side}"])
            agg[k, 3 * j:3 * j + 3] += (eb, chars(c[f"ref_{side}"], c[f"hyp_{side}"])[0], n)
        agg[k, 6:9] += (not (b["hyp_dialect"] and b["hyp_standard"]), not (c["hyp_dialect"] and c["hyp_standard"]), 1)
    cols = {"cer_dialect": ([0], [1], [2]), "cer_standard": ([3], [4], [5]), "cer_both": ([0, 3], [1, 4], [2, 5]),
            "format_fail": ([6], [7], [8])}
    rng = np.random.default_rng(0)
    sums = agg[rng.integers(0, len(blocks), size=(iters, len(blocks)))].sum(axis=1)
    total = agg.sum(axis=0)
    out = {"n": len(base), "recordings": len(blocks), "iters": iters}
    for name, (eb, ec, n) in cols.items():
        vb, vc = total[eb].sum() / max(total[n].sum(), 1), total[ec].sum() / max(total[n].sum(), 1)
        d = (sums[:, ec].sum(axis=1) - sums[:, eb].sum(axis=1)) / np.maximum(sums[:, n].sum(axis=1), 1)
        out[name] = {"base": round(float(vb), 4), "candidate": round(float(vc), 4), "diff": round(float(vc - vb), 4),
                     "ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                     "p_better": round(float((d < 0).mean()), 4), "p_worse": round(float((d > 0).mean()), 4)}
    return out


def asr_metrics(rows):
    blocks = [r["block"] for r in rows]
    out = {"n": len(rows)}
    for side in ("dialect", "standard"):
        c = [chars(r[f"ref_{side}"], r[f"hyp_{side}"]) for r in rows]
        w = [words(r[f"ref_{side}"], r[f"hyp_{side}"]) for r in rows]
        sw = [swords(r[f"ref_{side}"], r[f"hyp_{side}"]) for r in rows]
        out[f"cer_{side}"] = ratio_ci(blocks, [x for x, _ in c], [y for _, y in c])
        out[f"wer_{side}"] = ratio_ci(blocks, [x for x, _ in w], [y for _, y in w])
        out[f"swer_{side}"] = ratio_ci(blocks, [x for x, _ in sw], [y for _, y in sw])
    chrf = CHRF(word_order=2)
    out["chrfpp_standard"] = round(chrf.corpus_score([norm(r["hyp_standard"]) for r in rows],
                                                     [[norm(r["ref_standard"]) for r in rows]]).score, 2)
    out["chrf_signature"] = str(chrf.get_signature())
    # leave-as-is: the model's own dialect transcript used as the standard sentence
    lai = sum(words(r["ref_standard"], r["hyp_dialect"])[0] for r in rows)
    sysw = sum(words(r["ref_standard"], r["hyp_standard"])[0] for r in rows)
    out["err"] = round((lai - sysw) / lai, 4) if lai else None
    hits = total = 0
    for r in rows:
        hyp = Counter(norm(r["hyp_standard"]).split())
        for target in changes(r["ref_dialect"], r["ref_standard"]):
            total += 1
            need = Counter(target)
            hits += all(hyp[t] >= k for t, k in need.items())
    out["dialect_word_recall"] = round(hits / total, 4) if total else None
    out["dialect_word_changes"] = total
    out["format_ok"] = round(sum(bool(r["hyp_dialect"] and r["hyp_standard"]) for r in rows) / len(rows), 4)
    by_region = defaultdict(list)
    for r in rows:
        by_region[r["region"]].append(r)
    out["per_region"] = {}
    for reg, rs in sorted(by_region.items()):
        cd = [chars(r["ref_dialect"], r["hyp_dialect"]) for r in rs]
        cs = [chars(r["ref_standard"], r["hyp_standard"]) for r in rs]
        out["per_region"][reg] = {"n": len(rs),
                                  "cer_dialect": round(sum(x for x, _ in cd) / max(1, sum(y for _, y in cd)), 4),
                                  "cer_standard": round(sum(x for x, _ in cs) / max(1, sum(y for _, y in cs)), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--format", choices=["basic", "region"], default="basic")
    ap.add_argument("--compare", metavar="BASE", help="instead of the summary: paired comparison of the "
                    "predictions (candidate) with BASE's predictions on the same clips")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.predictions, encoding="utf-8")]
    if args.compare:
        base = [json.loads(line) for line in open(args.compare, encoding="utf-8")]
        json.dump(compare(base, rows), sys.stdout, ensure_ascii=False)
        return
    asr = [r for r in rows if r["kind"] == "asr"]
    summary = {"asr": asr_metrics(asr) if asr else None}
    if args.format == "region":
        summary["zone_plain"] = zone_metrics([r for r in rows if r["kind"] == "zone"])
        summary["zone_all"] = zone_metrics([r for r in rows if r.get("origin")])
    timed = [r for r in rows if r.get("seconds") is not None]
    if timed:
        summary["rtf"] = round(sum(r["seconds"] for r in timed) / max(1e-6, sum(r["duration"] for r in timed)), 3)
    json.dump(summary, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
