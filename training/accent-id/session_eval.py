"""Answer-level accent scores: what the app actually decides from several sentences of one speaker.

    python session_eval.py cut      # data_sessions/: 8 plain clips per speaker from AI Hub's Validation split
    python session_eval.py score    # runs/session_eval/report.{md,json}: served region4 + detector, 1-8 clips
    python session_eval.py rules    # runs/session_eval/rules.{md,json}: the app's final verdict under other rules

The app does not judge one sentence: it averages the clips of an answer (8 s or more) by length
(seotturi-web/server/analysis.py mean_probs). The test sets so far score single clips (test_plain: about one
clip per speaker), which is the hardest and least realistic case. Here every speaker brings 8 sentences without
dialect words, and the served system (accent.py: region4 for the region, detector_mix for standard vs regional,
one shared encoder) is scored on the first 1, 2, 3, 4, 6 and 8 of them, averaged by length as the app does, on
the clips as cut and as the site prepares them. Scored: the four-region choice of the regional speakers (the
number the benchmark page shows, 81.2 % on single clips), the detector, and the app's final five-way verdict
(서울·경기 or a region, "unsure" below analysis.ZONE_MIN_CONF).

Speakers: Validation-split recordings of the four sets (강원 118, 경상 119, 제주 121, 충청 122), so none was
heard in training; a speaker is one recording x speaker number with 8+ clips of 2-15 s without dialect words,
250 per set. As in region4's data (accent-id prepare.py), only speakers whose birthplace and residence fall in
the same zone, Jeolla left out: regional speakers are labelled by the set they were recorded in, and the
Seoul-Gyeonggi speakers (all 100 there are) score the standard detector.
"""

import collections
import json
import os
import random
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # repository root
GEMMA = os.path.join(REPO, "training", "gemma4-dialect")
sys.path.insert(0, GEMMA)
from prepare_data import BAD_TEXT, REGIONS, cut_recording, find_zips, index_recordings, split_of, zone_of  # noqa: E402

OUT = os.path.join(HERE, "data_sessions")
RUN_DIR = os.path.join(HERE, "runs", "session_eval")
SETS = ("118", "119", "121", "122")
REGION4 = ["강원", "충청", "경상", "제주"]
CLIPS, PER_REGION, SEOUL_MAX, SEED = 8, 250, 200, 0
STEPS = (1, 2, 3, 4, 6, 8)  # answers of this many sentences are scored
# speakers as in region4's data (accent-id prepare.py): birthplace and residence in the same zone, Jeolla out;
# regional = one of these, labelled by the set they were recorded in
REGIONAL = {"강원", "충청", "대구·경북", "부산·울산·경남", "제주"}


def scan(args):
    """Validation utterances of one label zip with their speaker slot."""
    zpath, region = args
    out = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            d = json.loads(z.read(name).decode("utf-8-sig"))
            spk = {str(s.get("id")): s for s in d.get("speaker", [])}
            for u in d.get("utterance", []):
                dia, std = (u.get("dialect_form") or "").strip(), (u.get("standard_form") or "").strip()
                start, end = float(u["start"]), float(u["end"])
                if not dia or not std or BAD_TEXT.search(dia) or BAD_TEXT.search(std) or not 2 <= end - start <= 15:
                    continue
                if any(e.get("isDialect") for e in u.get("eojeolList", [])) or dia != std:
                    continue  # sentences without dialect words only: the accent has to do the work
                s = spk.get(str(u.get("speaker_id")), {})
                bp, res = (s.get("birthplace") or "").strip(), (s.get("principal_residence") or "").strip()
                out.append({"id": u["id"], "recording": d["id"], "slot": f"{d['id']}#{u.get('speaker_id')}",
                            "start": start, "end": end, "split": "valid", "region": region, "dialect_form": dia,
                            "standard_form": std, "has_dialect": False, "origin": zone_of(bp, res),
                            "birthplace": bp, "residence": res})
    return out


def cut():
    rng = random.Random(SEED)
    jobs_scan = [(z, REGIONS[ds]) for ds in SETS for z in find_zips(f"/data/aihub/{ds}", "라벨링데이터") if split_of(z) == "valid"]
    utts = []
    with Pool(4) as p:
        for part in p.imap_unordered(scan, jobs_scan):
            utts += part
    by_slot = collections.defaultdict(list)
    for u in utts:
        by_slot[u["slot"]].append(u)
    slots = {s: us for s, us in by_slot.items() if len(us) >= CLIPS}
    chosen = []
    for region in REGION4:  # regional speakers, labelled by the set they were recorded in
        cand = sorted(s for s, us in slots.items() if us[0]["region"] == region and us[0]["origin"] in REGIONAL)
        rng.shuffle(cand)
        chosen += [(s, region) for s in cand[:PER_REGION]]
    seoul = sorted(s for s, us in slots.items() if us[0]["origin"] == "서울·경기")
    rng.shuffle(seoul)
    chosen += [(s, "서울·경기") for s in seoul[:SEOUL_MAX]]
    picked = []
    for s, label in chosen:
        us = sorted(slots[s], key=lambda u: u["start"])
        rng.shuffle(us)
        for k, u in enumerate(us[:CLIPS]):
            picked.append({**u, "label": label, "order": k})
    print("speakers:", collections.Counter(label for _, label in chosen), "clips:", len(picked), flush=True)
    index = {}
    for ds in SETS:
        index.update(index_recordings(f"/data/aihub/{ds}"))
    by_rec = collections.defaultdict(list)
    for u in picked:
        by_rec[u["recording"]].append(u)
    jobs = [(index[r], us, OUT, 0.0) for r, us in by_rec.items() if r in index]
    meta = {u["id"]: u for u in picked}
    rows = []
    with Pool(8) as p:
        for k, (part, err) in enumerate(p.imap_unordered(cut_recording, jobs)):
            if err:
                print("skip", err, flush=True)
            for r in part:
                m = meta[r["id"]]
                rows.append({**r, "slot": m["slot"], "label": m["label"], "order": m["order"]})
            if k % 200 == 0:
                print(f"cut {k}/{len(jobs)} recordings", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sessions.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: (r["slot"], r["order"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} clips", flush=True)


def macro_f1(y, p, labels):
    f = []
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), {c: round(v, 3) for c, v in zip(labels, f)}


def auc(pos, neg):
    a, b = np.asarray(pos), np.asarray(neg)
    return float(((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()))


CONDITIONS = ("원본", "사이트 전처리")  # the AI Hub clip as cut, and as the site prepares a recording (analysis.preprocess)


def answer(rs, cond):
    """One answer's verdict inputs, as the app forms them: the clips' probabilities averaged by length.
    Returns (four-region probabilities, P(regional), the app's five-way probabilities 서울·경기 + regions)."""
    w = np.array([r["scores"][cond]["sec"] for r in rs])
    p4 = np.array([r["scores"][cond]["p4"] for r in rs])
    reg = np.array([r["scores"][cond]["regional"] for r in rs])
    app = np.concatenate([(1 - reg)[:, None], reg[:, None] * p4], axis=1)  # accent.py: per clip
    avg = lambda a: (a * w.reshape(-1, *([1] * (a.ndim - 1)))).sum(0) / w.sum()  # analysis.mean_probs
    return avg(p4), float(avg(reg)), avg(app), float(w.sum())


def score():
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(REPO, "app", "server"))
    import analysis as A  # the site's preprocessing and its verdict threshold
    import common
    torch.cuda.set_per_process_memory_fraction(0.1)
    rows = [json.loads(l) for l in open(os.path.join(OUT, "sessions.jsonl"), encoding="utf-8")]
    enc = common.Encoder()
    region = common.Tree(os.path.realpath(os.path.join(HERE, "runs", "served")), enc)
    std_dir = os.path.realpath(os.path.join(HERE, "runs", "served_std"))
    detector = common.Tree(std_dir, enc)
    with open(os.path.join(std_dir, "ensemble.json"), encoding="utf-8") as f:
        std_bias = json.load(f).get("served_standard_bias", 0.0)
    assert region.zones == REGION4, region.zones
    k_seoul = detector.zones.index("서울·경기")
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        raw = [common.load_wav(os.path.join(OUT, r["audio"])) for r in batch]
        site = [y if len(y) >= 8000 else a for a, y in ((a, A.preprocess(a)[0]) for a in raw)]
        for cond, audios in zip(CONDITIONS, (raw, site)):
            x = enc.pooled(audios)
            p4, q = region.probs(audios, x), detector.probs(audios, x)
            for r, au, a, b in zip(batch, audios, p4, q):
                s = b[k_seoul] * np.exp(std_bias)  # accent.py classify(): the detector's bonus on 서울·경기
                r.setdefault("scores", {})[cond] = {"p4": [round(float(v), 5) for v in a], "sec": len(au) / 16000,
                                                    "regional": float(1 - s / (s + 1 - b[k_seoul]))}
        if i % 640 == 0:
            print(f"scored {i}/{len(rows)}", flush=True)
    by_slot = collections.defaultdict(list)
    for r in rows:
        by_slot[r["slot"]].append(r)
    for rs in by_slot.values():
        rs.sort(key=lambda r: r["order"])
    app_zones = ["서울·경기"] + REGION4
    report = {"clips": len(rows), "speakers": dict(collections.Counter(v[0]["label"] for v in by_slot.values())),
              "system": {"region": os.path.basename(os.path.realpath(os.path.join(HERE, "runs", "served"))),
                         "detector": os.path.basename(std_dir)},
              "std_bias": std_bias, "min_conf": A.ZONE_MIN_CONF, "conditions": {}}
    rng = np.random.default_rng(SEED)
    for cond in CONDITIONS:
        out = report["conditions"][cond] = {}
        for k in STEPS:
            y, pred, reg_scores, std_scores, secs, y5, p5 = [], [], [], [], [], [], []
            for rs in by_slot.values():
                if len(rs) < k:
                    continue
                p4, regional, app, sec = answer(rs[:k], cond)
                secs.append(sec)
                label = rs[0]["label"]
                verdict = app_zones[int(np.argmax(app))]
                y5.append(label)
                p5.append(verdict if app.max() >= A.ZONE_MIN_CONF else "보류")
                if label == "서울·경기":
                    std_scores.append(regional)
                else:
                    y.append(label)
                    pred.append(REGION4[int(np.argmax(p4))])
                    reg_scores.append(regional)
            mf1, per = macro_f1(y, pred, REGION4)
            boot = []  # 95% interval over speakers
            for _ in range(1000):
                idx = rng.integers(0, len(y), len(y))
                boot.append(macro_f1([y[i] for i in idx], [pred[i] for i in idx], REGION4)[0])
            app_f1, app_per = macro_f1(y5, p5, app_zones)
            sure = [(a, b) for a, b in zip(y5, p5) if b != "보류"]
            out[k] = {
                "speakers": len(y) + len(std_scores), "median_seconds": round(float(np.median(secs)), 1),
                "region4_macro_f1": round(mf1, 3),
                "region4_macro_f1_ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                "region4_accuracy": round(float(np.mean([a == b for a, b in zip(y, pred)])), 3), "per_region_f1": per,
                "detector_auc": round(auc(reg_scores, std_scores), 3),
                "regional_caught": round(float(np.mean(np.array(reg_scores) >= 0.5)), 3),
                "seoul_called_regional": round(float(np.mean(np.array(std_scores) >= 0.5)), 3),
                "app_macro_f1": round(app_f1, 3), "app_per_zone_f1": app_per,
                "app_unsure": round(1 - len(sure) / len(y5), 3),
                "app_accuracy_when_sure": round(float(np.mean([a == b for a, b in sure])), 3) if sure else None,
            }
            print(cond, k, out[k], flush=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(os.path.join(RUN_DIR, "clips.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "slot", "label", "order", "duration", "scores")}, ensure_ascii=False) + "\n")
    md = ["# 답변 단위 억양 판정 (서비스 중인 4지역 모델 + 표준 억양 판별기)", "",
          f"화자 {report['speakers']} · 화자당 사투리 단어 없는 문장 {CLIPS}개 · AI Hub 검증 분할 (학습 미사용)", ""]
    for cond in CONDITIONS:
        md += [f"## {cond}", "",
               "| 문장 수 | 길이(중앙값) | 4지역 Macro-F1 [95% CI] | 4지역 정확도 | 판별기 AUC | 지역 억양 잡아냄 | 서울·경기 오판 | 앱 최종 Macro-F1 (5칸) | 판정 보류 | 보류 아닐 때 정확도 |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for k, v in report["conditions"][cond].items():
            ci = v["region4_macro_f1_ci"]
            md.append(f"| {k} | {v['median_seconds']}초 | {v['region4_macro_f1']:.3f} [{ci[0]:.3f}–{ci[1]:.3f}] | {v['region4_accuracy']:.3f} | "
                      f"{v['detector_auc']:.3f} | {v['regional_caught']:.0%} | {v['seoul_called_regional']:.0%} | {v['app_macro_f1']:.3f} | "
                      f"{v['app_unsure']:.0%} | {v['app_accuracy_when_sure']:.0%} |")
        md.append("")
    with open(os.path.join(RUN_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


MAX_SEOUL_FA = 0.15


def rules():
    """The app's final verdict under other decision rules, from the scores score() saved (no GPU):
      now      the app today: length-weighted mean of the five-way probabilities, unsure below ZONE_MIN_CONF
      2step    regional or not from the mean P(regional) (>= 0.5), then the region from the mean four-region
               probabilities, unsure when their top is below 0.6: the threshold sees P(region | regional),
               not P(regional) x P(region | regional)
      2step-log  the same with the four-region probabilities averaged in log space (independent evidence)
      2step-renorm  the same from the app's averaged five-way probabilities alone: regional = 1 - P(서울·경기),
               region = the regions' probabilities renormalised, i.e. the clips' four-region probabilities
               averaged with weights length x P(regional) (what the app can compute without new fields)
      tuned    2step-log with the two thresholds chosen on the other half of the speakers (2-fold by recording):
               the best five-way Macro-F1 that calls at most MAX_SEOUL_FA of the Seoul-Gyeonggi speakers
               regional (telling a standard speaker they have an accent is the costly mistake), so the number
               is not tuned on what it is reported on"""
    rows = [json.loads(l) for l in open(os.path.join(RUN_DIR, "clips.jsonl"), encoding="utf-8")]
    by_slot = collections.defaultdict(list)
    for r in rows:
        by_slot[r["slot"]].append(r)
    for rs in by_slot.values():
        rs.sort(key=lambda r: r["order"])
    zones5 = ["서울·경기"] + REGION4
    min_conf = json.load(open(os.path.join(RUN_DIR, "report.json"), encoding="utf-8"))["min_conf"]

    def feats(rs, cond):
        w = np.array([r["scores"][cond]["sec"] for r in rs])
        p4 = np.array([r["scores"][cond]["p4"] for r in rs])
        reg = np.array([r["scores"][cond]["regional"] for r in rs])
        app = np.concatenate([(1 - reg)[:, None], reg[:, None] * p4], axis=1)
        mean = lambda a: (a * w[:, None]).sum(0) / w.sum()
        logp = (np.log(np.clip(p4, 1e-6, 1)) * w[:, None]).sum(0) / w.sum()
        geo = np.exp(logp - logp.max())
        return {"app": mean(app), "reg": float((reg * w).sum() / w.sum()), "p4": mean(p4), "p4log": geo / geo.sum()}

    def decide(f, rule, t_reg=0.5, tau=0.6):
        if rule == "now":
            return zones5[int(np.argmax(f["app"]))] if f["app"].max() >= min_conf else "보류"
        if rule == "2step-renorm":  # from the averaged five-way probabilities alone, as the app has them
            app = f["app"]
            if 1 - app[0] < t_reg:
                return "서울·경기"
            p = app[1:] / app[1:].sum()
            return REGION4[int(np.argmax(p))] if p.max() >= tau else "보류"
        if f["reg"] < t_reg:
            return "서울·경기"
        p = f["p4log"] if rule.endswith("log") else f["p4"]
        return REGION4[int(np.argmax(p))] if p.max() >= tau else "보류"

    def measure(items, rule, t_reg=0.5, tau=0.6):
        y = [lab for lab, _ in items]
        pred = [decide(f, rule, t_reg, tau) for _, f in items]
        f1 = macro_f1(y, pred, zones5)[0]
        sure = [(a, b) for a, b in zip(y, pred) if b != "보류"]
        seoul = [b for a, b in zip(y, pred) if a == "서울·경기"]
        return {"macro_f1": round(f1, 3), "unsure": round(1 - len(sure) / len(y), 3),
                "accuracy_when_sure": round(float(np.mean([a == b for a, b in sure])), 3) if sure else None,
                "seoul_called_regional": round(float(np.mean([b in REGION4 for b in seoul])), 3)}

    out = {}
    grid_reg, grid_tau = np.arange(0.3, 0.96, 0.05), np.arange(0.3, 0.86, 0.05)
    for cond in CONDITIONS:
        out[cond] = {}
        for k in STEPS:
            items = [(rs[0]["label"], feats(rs[:k], cond), rs[0]["slot"].split("#")[0]) for rs in by_slot.values() if len(rs) >= k]
            res = {rule: measure([(a, b) for a, b, _ in items], rule) for rule in ("now", "2step", "2step-log", "2step-renorm")}
            # 2-fold by recording: thresholds from one half, scored on the other, verdicts pooled
            recs = sorted({c for _, _, c in items})
            half = {c: i % 2 for i, c in enumerate(np.random.default_rng(SEED).permutation(recs))}
            pooled, chosen = [], []
            for fold in (0, 1):
                tune = [(a, b) for a, b, c in items if half[c] != fold]
                test = [(a, b) for a, b, c in items if half[c] == fold]
                ok = [(tr, ta) for tr in grid_reg for ta in grid_tau
                      if measure(tune, "2step-log", tr, ta)["seoul_called_regional"] <= MAX_SEOUL_FA]
                best = max(ok or [(0.5, 0.6)], key=lambda t: measure(tune, "2step-log", *t)["macro_f1"])
                chosen.append([round(float(best[0]), 2), round(float(best[1]), 2)])
                pooled += [(a, decide(f, "2step-log", *best)) for a, f in test]
            y, pred = [a for a, _ in pooled], [b for _, b in pooled]
            sure = [(a, b) for a, b in pooled if b != "보류"]
            res["tuned"] = {"macro_f1": round(macro_f1(y, pred, zones5)[0], 3), "unsure": round(1 - len(sure) / len(y), 3),
                            "accuracy_when_sure": round(float(np.mean([a == b for a, b in sure])), 3),
                            "seoul_called_regional": round(float(np.mean([b in REGION4 for a, b in pooled if a == "서울·경기"])), 3),
                            "thresholds": chosen}
            out[cond][k] = res
            print(cond, k, json.dumps(res, ensure_ascii=False), flush=True)
    with open(os.path.join(RUN_DIR, "rules.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    names = {"now": "지금 앱 규칙", "2step": "두 단계", "2step-log": "두 단계 + 로그 평균", "2step-renorm": "두 단계 (앱 평균 확률로)",
             "tuned": "두 단계 + 기준 조정(서울 오판 ≤15%, 교차)"}
    md = ["# 앱 최종 판정(5칸: 서울·경기 + 4지역) 규칙 비교", "",
          "칸 = 5칸 Macro-F1 (보류는 틀린 것으로) · 보류 비율 · 보류 아닐 때 정확도 · 서울·경기 오판", ""]
    for cond in CONDITIONS:
        md += [f"## {cond}", "", "| 문장 수 | " + " | ".join(names.values()) + " |", "| --- |" + " --- |" * len(names)]
        for k, res in out[cond].items():
            md.append(f"| {k} | " + " | ".join(
                f"{v['macro_f1']:.3f} · 보류 {v['unsure']:.0%} · 정확도 {v['accuracy_when_sure']:.0%} · 서울 오판 {v['seoul_called_regional']:.0%}"
                for v in (res[r] for r in names)) + " |")
        md.append("")
    with open(os.path.join(RUN_DIR, "rules.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    {"cut": cut, "score": score, "rules": rules}[sys.argv[1]]()
