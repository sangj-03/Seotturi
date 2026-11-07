"""Where do the CER points come from? Error breakdown of one evaluate.py run on data_test.

    python error_analysis.py runs/eval_test_guard/s6c         # writes <run>/error_analysis.{md,json}

Both answer lines are aligned to their reference character by character (spaces and punctuation removed, as
in common.cer), and every substitution, deletion and insertion is charged to one cause, first match wins:

  loop        the whole clip, when the answer is far longer than the reference (a repetition loop or text
              the reference does not have) or the guard had to repair it
  filler      a reference word written with "~" (뭐~, 그~, 어~: fillers and drawn-out words)
  edge        the first or last word of the reference, or text added before or after it: where a clip cut
              inside speech shows up
  converted   a word that differs between the two references: on the dialect line a dialect word, on the
              standard line a word the model had to convert
  other       everything else: ordinary words in the middle of the sentence

Besides each cause's share of the CER, the report gives the error rate inside each cause (its errors per
reference character it covers): edge words are about a third of the text, so a third of the errors there is
no sign of a cut by itself.

The audio of every clip is checked for a cut inside speech (the first or last 60 ms as loud as speech), and
the report compares those clips with the rest. "CER without X" removes that cause's errors from the count,
i.e. what is left if X were solved completely: an upper bound on what fixing X can buy, not a forecast.
"""

import collections
import json
import os
import sys
import wave

import numpy as np

from common import _STRIP

HERE = os.path.dirname(os.path.abspath(__file__))
CAUSES = ["loop", "filler", "edge", "converted", "other"]
NAMES = {"loop": "반복·환각 문장", "filler": "군말 (뭐~ 그~ 어~)", "edge": "문장 첫·끝 단어", "converted": "사투리 단어 / 표준어로 바꾼 단어",
         "other": "그 밖의 단어"}


def chars_of(text):
    """Normalized characters of a text and, for each, the index of its word."""
    out, word = [], []
    for w, tok in enumerate(text.split()):
        for ch in _STRIP.sub("", tok):
            out.append(ch)
            word.append(w)
    return "".join(out), word


def align(ref, hyp):
    """Edit operations turning ref into hyp: list of (op, ref_index), insertions at the ref index they precede."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("S", i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", i - 1))
            i -= 1
        else:
            ops.append(("I", i))  # inserted before ref[i] (i == n: after the end)
            j -= 1
    return ops


def cut_inside_speech(path):
    """(starts inside speech, ends inside speech) from the loudness of the first and last 60 ms."""
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    f = 320  # 20 ms at 16 kHz
    if len(a) < 10 * f:
        return False, False
    rms = np.sqrt(np.mean(a[: len(a) // f * f].reshape(-1, f) ** 2, axis=1))
    thr = max(3 * np.percentile(rms, 10), 0.15 * np.percentile(rms, 95), 1e-3)
    return bool(rms[:3].mean() > thr), bool(rms[-3:].mean() > thr)


def categories(r, line):
    """The cause of an error in each reference word, first match wins ("loop" is decided per clip)."""
    tokens = r[f"ref_{line}"].split()
    other = {_STRIP.sub("", t) for t in r["ref_standard" if line == "dialect" else "ref_dialect"].split()}
    cats = []
    for w, t in enumerate(tokens):
        if t.endswith("~"):
            cats.append("filler")
        elif w in (0, len(tokens) - 1):
            cats.append("edge")
        elif _STRIP.sub("", t) not in other:
            cats.append("converted")  # a word that differs between the lines: dialect word / converted word
        else:
            cats.append("other")
    return cats


def analyse(rows, line):
    """Per clip, for one line ("dialect" or "standard"): reference length, errors by cause and reference
    characters by cause (how much of the text each cause covers)."""
    out = []
    for r in rows:
        ref, word = chars_of(r[f"ref_{line}"])
        hyp, _ = chars_of(r[line] or "")
        cats = categories(r, line)
        loop = len(hyp) > 1.6 * len(ref) + 5 or r.get("repaired")
        chars = collections.Counter("loop" if loop else cats[w] for w in word)
        errs = collections.Counter()
        for op, i in align(ref, hyp):
            if loop:
                errs["loop"] += 1
            elif op == "I" and (i == 0 or i == len(ref)):
                errs["edge"] += 1
            else:
                errs[cats[word[i] if op != "I" else word[i - 1]]] += 1
        out.append({"n": len(ref), "errs": errs, "chars": chars})
    return out


def scan_recordings(args):
    """Utterances of the wanted recordings in one label zip, in recording order."""
    import zipfile
    zpath, wanted = args
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".json") or os.path.basename(name)[:-5] not in wanted:
                continue
            d = json.loads(z.read(name).decode("utf-8-sig"))
            out[d["id"]] = [(u["id"], (u.get("dialect_form") or "").strip()) for u in d.get("utterance", [])]
    return out


def cer_of(ref, hyp):
    a, b = _STRIP.sub("", ref), _STRIP.sub("", hyp)
    return sum(1 for _ in align(a, b)) / max(len(a), 1)


def label_check(rows, datasets=("118", "119", "121", "122")):
    """Per clip: "other" when the answer matches other utterances of the recording (one to three consecutive
    ones near it) far better than its own reference: the label times point at other speech. "spill" when it
    matches its own text together with the next or previous one better: the clip holds part of a neighbour."""
    from multiprocessing import Pool
    from prepare_data import find_zips, split_of
    wanted = {r["id"].split(".")[0] for r in rows}
    zips = [z for ds in datasets for z in find_zips(f"/data/aihub/{ds}", "라벨링데이터") if split_of(z) == "valid"]
    recs = {}
    with Pool(8) as p:
        for part in p.imap_unordered(scan_recordings, [(z, wanted) for z in zips]):
            recs.update(part)
    flags = []
    for r in rows:
        utts = recs.get(r["id"].split(".")[0], [])
        idx = next((k for k, (uid, _) in enumerate(utts) if uid == r["id"]), None)
        hyp, own = r["dialect"] or "", r["ref_dialect"]
        c_own = cer_of(own, hyp)
        if idx is None or c_own < 0.3:
            flags.append("ok")
            continue
        # other utterances: any run of one to three consecutive neighbours within six places, not the clip's own
        lo, hi = max(0, idx - 6), min(len(utts), idx + 7)
        windows = [" ".join(utts[k][1] for k in range(a, b)) for a in range(lo, hi) for b in range(a + 1, min(hi, a + 3) + 1)
                   if not (a <= idx < b)]
        c_other = min((cer_of(t, hyp) for t in windows if t.strip()), default=9)
        prev_t = utts[idx - 1][1] if idx > 0 else ""
        next_t = utts[idx + 1][1] if idx + 1 < len(utts) else ""
        c_spill = min(cer_of(own + " " + next_t, hyp) if next_t else 9, cer_of(prev_t + " " + own, hyp) if prev_t else 9)
        if c_other < 0.5 and c_other < c_own - 0.3:
            flags.append("other")
        elif c_spill < c_own - 0.1:
            flags.append("spill")
        else:
            flags.append("ok")
    return flags


def summary(items, mask=None):
    sel = [x for k, x in enumerate(items) if mask is None or mask[k]]
    n = sum(x["n"] for x in sel)
    errs, chars = collections.Counter(), collections.Counter()
    for x in sel:
        errs.update(x["errs"])
        chars.update(x["chars"])
    total = sum(errs.values())
    return {"clips": len(sel), "chars": n, "cer": total / max(n, 1),
            "by_cause": {c: errs[c] / max(n, 1) for c in CAUSES},
            "share": {c: errs[c] / max(total, 1) for c in CAUSES},
            "char_share": {c: chars[c] / max(n, 1) for c in CAUSES},
            "rate": {c: errs[c] / chars[c] if chars[c] else None for c in CAUSES}}


def main():
    run = sys.argv[1]
    rows = [json.loads(l) for l in open(os.path.join(run, "predictions.jsonl"), encoding="utf-8")]
    manifest = {}
    for l in open(os.path.join(HERE, "data_test", "test.jsonl"), encoding="utf-8"):
        m = json.loads(l)
        manifest[m["id"]] = m
    cut = []
    for r in rows:
        m = manifest[r["id"]]
        path = m["audio"] if os.path.isabs(m["audio"]) else os.path.join(HERE, "data_test", m["audio"])
        cut.append(cut_inside_speech(path))
    regions = sorted({r["region"] for r in rows})
    flags = label_check(rows)
    report = {"run": run, "lines": {}, "labels": {}}
    for g in regions + ["all"]:
        sel = [f for r, f in zip(rows, flags) if g == "all" or r["region"] == g]
        report["labels"][g] = {k: sel.count(k) / len(sel) for k in ("other", "spill", "ok")}
    for line in ("dialect", "standard"):
        items = analyse(rows, line)
        res = {"all": summary(items)}
        for g in regions:
            res[g] = summary(items, [r["region"] == g for r in rows])
            res[g + " · 잘린 클립"] = summary(items, [r["region"] == g and (a or b) for r, (a, b) in zip(rows, cut)])
            res[g + " · 온전한 클립"] = summary(items, [r["region"] == g and not (a or b) for r, (a, b) in zip(rows, cut)])
        per_clip = sorted((sum(x["errs"].values()) for x in items), reverse=True)
        res["worst_10pct_share"] = sum(per_clip[: len(per_clip) // 10]) / max(sum(per_clip), 1)
        dur = [manifest[r["id"]]["duration"] for r in rows]
        for g in regions:  # short clips: little context, and their first and last words are most of the text
            for name, lo, hi in (("3초 미만", 0, 3), ("3~6초", 3, 6), ("6초 이상", 6, 99)):
                res[f"{g} · {name}"] = summary(items, [r["region"] == g and lo <= d < hi for r, d in zip(rows, dur)])
        for g in regions + ["all"]:  # the same scores on clips whose audio matches their label
            res[f"{'전체' if g == 'all' else g} · 라벨이 맞는 클립"] = summary(items, [(g == "all" or r["region"] == g) and f == "ok" for r, f in zip(rows, flags)])
        if line == "standard":  # the conversion alone: clips whose dialect line came out right
            perfect = [r["cer_dialect"] == 0 for r in rows]
            res["dialect_perfect"] = summary(items, perfect)
        report["lines"][line] = res
    report["cut_share"] = {g: float(np.mean([(a or b) for r, (a, b) in zip(rows, cut) if r["region"] == g])) for g in regions}
    with open(os.path.join(run, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    md = [f"# 오류 분석 · {run}", ""]
    for line, label in (("dialect", "방언 줄"), ("standard", "표준어 줄")):
        res = report["lines"][line]
        md += [f"## {label}", "", "| 범위 | 클립 | CER | " + " | ".join(f"{NAMES[c]}" for c in CAUSES) + " |",
               "| --- | --- | --- | " + " | ".join("---" for _ in CAUSES) + " |"]
        for key in ["all"] + [k for k in res if k not in ("all", "worst_10pct_share", "dialect_perfect")] + (["dialect_perfect"] if "dialect_perfect" in res else []):
            s = res[key]
            name = {"all": "전체", "dialect_perfect": "방언 줄이 완벽한 클립"}.get(key, key)
            md.append(f"| {name} | {s['clips']} | {s['cer']:.3f} | " + " | ".join(f"{s['by_cause'][c]:.3f} ({s['share'][c]:.0%})" for c in CAUSES) + " |")
        md += ["", f"오류가 많은 10% 클립이 전체 오류의 {res['worst_10pct_share']:.0%}를 차지합니다.", "",
               f"### {label} · 범주 안의 오류율 (범주가 차지하는 글자 비중)", "",
               "| 범위 | " + " | ".join(NAMES[c] for c in CAUSES) + " |", "| --- | " + " | ".join("---" for _ in CAUSES) + " |"]
        for key in ["all"] + regions:
            s = res[key]
            md.append(f"| {'전체' if key == 'all' else key} | " + " | ".join(
                "—" if s["rate"][c] is None else f"{s['rate'][c]:.3f} ({s['char_share'][c]:.0%})" for c in CAUSES) + " |")
        md.append("")
    md += ["## 말하는 도중에 잘린 클립 비율", ""] + [f"- {g}: {v:.0%}" for g, v in report["cut_share"].items()]
    md += ["", "## 라벨이 오디오와 어긋난 클립 (모델 답이 같은 녹음의 이웃 문장과 맞음)", ""]
    md += [f"- {'전체' if g == 'all' else g}: 다른 문장 {v['other']:.1%} · 옆 문장이 섞임 {v['spill']:.1%}" for g, v in report["labels"].items()]
    with open(os.path.join(run, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
