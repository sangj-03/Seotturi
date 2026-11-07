"""Accent classifier results for the 서투리 benchmark page (seotturi-web: /api/accent, /benchmark#accent).

    python bench_export.py          # runs/benchmark.json

Every experiment kept in runs/ is scored again from its test_logits.npz on the four held-out tests
(AI Hub Validation) the way it was or would be served (ensemble weights in log-probability, standard
bonus), with 95 % intervals from a block bootstrap over recordings (1,000 resamples, as the page does
for the Gemma results). Also written: dev macro-F1 and training-set sizes, the learning curve
(runs/curve plus the 1.7x and 3.6x runs) and Gemma's `지역:` line on the same bench clips
(seotturi-web state/bench/lora_v2r-*.jsonl).

Refreshed by cron every 10 minutes with --if-changed (nothing happens unless a run's results, the served
links or the Gemma bench files are newer than runs/benchmark.json). Runs not listed in ITEMS still appear:
six-zone runs with test logits join the history as "새 실험", four-region runs (r4_* tests) and standard
detectors (conditions_binary.json) are listed in the current-system card. Give a run a name and a
description by adding it to ITEMS.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from common import STANDARD, ZONES

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
WEB_STATE = "/data/seotturi/seotturi-web/state"
TESTS = [  # id, name, description, short name (table header)
    ("test_plain", "사투리 단어 없는 문장", "6개 권역 × 200문장, 사투리 단어가 없는 문장만: 억양만 듣고 맞혀야 해요", "단어 없는 문장"),
    ("test_same", "같은 문장, 다른 권역", "같은 문장을 두 권역 이상의 화자가 말한 것: 내용으로는 권역을 알 수 없어요", "같은 문장"),
    ("test_dialect", "사투리 단어 포함", "Gemma 검증 문장 그대로, 사투리 단어도 들어 있어요", "단어 포함"),
    ("bench", "Gemma 억양 벤치마크와 같은 문장", "위 Gemma '지역:' 판별과 같은 문장(전라 15개 제외, 6개 권역 × 15)", "Gemma 벤치 90"),
]
# Chronological. members: (run, weight) mixed in log-probability; bias: added to 서울·경기's log-probability.
ITEMS = [
    {"id": "phase1", "name": "1단계", "short": "1단계", "members": [("phase1", 1.0)], "bias": 0.0, "data": "data",
     "desc": "Whisper-large-v3 인코더는 고정, 33개 층의 평균·표준편차 + 작은 분류 헤드", "status": "replaced",
     "note": "1.7배 데이터로 다시 학습한 1단계로 교체"},
    {"id": "phase2", "name": "2단계", "short": "2단계", "members": [("phase2", 1.0)], "bias": 0.0, "data": "data",
     "desc": "인코더 17~32층 LoRA 미세조정, 어텐션 풀링, 음성 증강 (3 epoch)", "status": "member",
     "note": "단독으로는 1단계보다 낮아 앙상블 안에서만 사용"},
    {"id": "ensemble", "name": "1+2단계 앙상블", "short": "1+2단계", "members": [("phase1", 0.6), ("phase2", 0.4)], "bias": 1.0,
     "desc": "1단계 0.6 + 2단계 0.4, 표준어 보정 +1.0", "status": "was_served", "note": "10/7 04:00~05:57 서비스"},
    {"id": "phase1_big", "name": "1단계 · 데이터 1.7배", "short": "1단계 (1.7배)", "members": [("phase1_big", 1.0)], "bias": 0.0,
     "data": "data_big", "desc": "1단계를 학습 클립 1.73배(모든 대상 화자)로 다시 학습", "status": "member"},
    {"id": "ensemble_big", "name": "1단계(1.7배)+2단계 앙상블", "short": "1단계(1.7배)+2단계", "members": [("phase1_big", 0.6), ("phase2", 0.4)],
     "bias": 1.25, "desc": "1단계(1.7배) 0.6 + 2단계 0.4, 표준어 보정 +1.25", "status": "was_served",
     "note": "10/7 05:57~10/9 19:34 서비스, 지금은 표준 억양 판별기의 절반"},
    {"id": "phase1_x2", "name": "1단계 · 데이터 3.6배", "short": "1단계 (3.6배)", "members": [("phase1_x2", 1.0)], "bias": 0.0,
     "data": "data_x2", "desc": "남은 새 화자를 모두 더해 클립 3.6배: 1.7배보다 거의 오르지 않음", "status": "not_adopted"},
    {"id": "phase3", "name": "3단계", "short": "3단계", "members": [("phase3", 1.0)], "bias": 0.0, "data": "data_x2",
     "desc": "33개 층 전체 + 1단계 헤드에서 시작, 17~32층 LoRA, 3.6배 데이터 1 epoch", "status": "not_adopted",
     "note": "dev는 0.84까지 올랐지만 시험은 서비스 중인 앙상블보다 낮음"},
    {"id": "mix3", "name": "3개 앙상블 (검토)", "short": "3개 앙상블", "members": [("phase1_big", 1 / 3), ("phase2", 1 / 3), ("phase3", 1 / 3)],
     "bias": 1.25, "desc": "1단계(1.7배)·2단계·3단계 1/3씩, 표준어 보정 +1.25", "status": "reviewed",
     "note": "test_plain만 조금 오르고 서울·경기 오판은 그대로라 채택하지 않음"},
    {"id": "zone6_aug", "name": "1단계 · 녹음 환경 변형", "short": "1단계 (변형)", "members": [("zone6_aug", 1.0)], "bias": 0.0,
     "data": "data/z6_train.jsonl", "desc": "1단계(1.7배) 클립마다 녹음 환경을 바꾼 복사본 2개(무음화·잡음 제거·음량·다른 세트 배경·마이크·압축·짧게 자르기)를 더해 학습",
     "status": "member", "note": "10/9부터 표준 억양 판별기의 절반(서울·경기일 확률만 사용)"},
]
B = 1000


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else None


class Scorer:
    """Macro-F1 and accuracy of one test split, with block-bootstrap intervals over recordings."""

    def __init__(self, rows):
        self.y = np.array([r["y"] for r in rows])
        self.zones = sorted(set(self.y.tolist()))
        rec = {}
        for i, r in enumerate(rows):
            rec.setdefault(r["recording"], []).append(i)
        blocks = [np.array(v) for v in rec.values()]
        rng = np.random.default_rng(0)
        self.samples = [np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))]) for _ in range(B)]

    def f1(self, y, p):
        """p may hold len(ZONES) for "no zone / another zone" (Gemma): wrong, but nobody's false positive."""
        k = len(ZONES)
        cm = np.bincount(y * (k + 1) + p, minlength=k * (k + 1)).reshape(k, k + 1)
        tp, pred, true = np.diag(cm[:, :k]), cm[:, :k].sum(0), cm.sum(1)
        f = np.where(pred + true > 0, 2 * tp / np.maximum(pred + true, 1), 0.0)
        return float(np.mean(f[self.zones])), f, cm

    def score(self, prob=None, pred=None):
        p = prob.argmax(1) if pred is None else pred
        y = self.y
        mf, f, cm = self.f1(y, p)
        boot_f = [self.f1(y[s], p[s])[0] for s in self.samples]
        boot_a = [float(np.mean(p[s] == y[s])) for s in self.samples]
        regional, called = y != STANDARD, p != STANDARD
        names = ZONES + ["-"]
        return {"n": int(len(y)),
                "macro_f1": {"value": round(mf, 4), "ci": [round(float(np.percentile(boot_f, q)), 4) for q in (2.5, 97.5)]},
                "accuracy": {"value": round(float(np.mean(p == y)), 4),
                             "ci": [round(float(np.percentile(boot_a, q)), 4) for q in (2.5, 97.5)]},
                "auc": auc(1 - prob[:, STANDARD], regional) if prob is not None and STANDARD in self.zones else None,
                "standard_called_regional": [int((called & ~regional).sum()), int((~regional).sum())],
                "regional_called_standard": [int((~called & regional).sum()), int(regional.sum())],
                "per_zone_f1": {ZONES[k]: round(float(f[k]), 4) for k in self.zones},
                "confusion": {ZONES[t]: {names[q]: int(cm[t, q]) for q in range(len(names)) if cm[t, q]} for t in self.zones}}


def auc(score, positive):
    """Area under the ROC curve (ties count half)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):  # average ranks of ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return round(float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4) if n1 and n0 else None


def probs(members, logits, split, bias):
    lp = sum(w * torch.log_softmax(torch.from_numpy(logits[r][split]).float(), -1) for r, w in members)
    lp[:, STANDARD] += bias
    return torch.softmax(lp, -1).numpy()


def train_size(data):
    rows = read_jsonl(os.path.join(HERE, data) if data.endswith(".jsonl") else os.path.join(HERE, data, "train.jsonl"))
    return {"clips": len(rows), "persons": len({r["person"] for r in rows})}


def curve(items):
    """Phase 1 trained on more or less data: runs/curve (persons subsets, 2 seeds) + the 1.7x and 3.6x runs."""
    res = json.load(open(os.path.join(RUNS, "curve", "results.json"), encoding="utf-8"))
    points = []
    for frac in sorted({r["value"] for r in res if r["curve"] == "persons"}):
        rs = [r for r in res if r["curve"] == "persons" and r["value"] == frac]
        m = lambda k: round(float(np.mean([r[k]["macro_f1"] for r in rs])), 4)  # noqa: E731
        points.append({"label": f"화자 {round(frac * 100)}%", "clips": int(np.mean([r["clips"] for r in rs])),
                       "persons": int(np.mean([r["persons"] for r in rs])), "seeds": len(rs),
                       "dev": round(float(np.mean([r["dev_macro_f1"] for r in rs])), 4),
                       **{s: m(s) for s, *_ in TESTS}})
    for run, label in (("phase1_big", "1.7배"), ("phase1_x2", "3.6배")):
        it = next(i for i in items if i["id"] == run)
        points.append({"label": label, "clips": it["train"]["clips"], "persons": it["train"]["persons"], "seeds": 1,
                       "dev": it["dev_macro_f1"], **{s: it["tests"][s]["macro_f1"]["value"] for s, *_ in TESTS}})
    return points


def gemma(bench_rows, scorer):
    """Gemma's `지역:` line on the same bench clips, per v2r checkpoint the engine benchmarked."""
    truth = {r["id"]: i for i, r in enumerate(bench_rows)}
    out = []
    for path in sorted(glob.glob(os.path.join(WEB_STATE, "bench", "lora_v2r-*.jsonl"))):
        hyp = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] == "zone" and r["id"] in truth:
                hyp[r["id"]] = r.get("hyp_zone")
        if len(hyp) != len(truth):
            continue
        pred = np.full(len(truth), len(ZONES))  # 전라 or no zone line: wrong
        for cid, i in truth.items():
            if hyp[cid] in ZONES:
                pred[i] = ZONES.index(hyp[cid])
        s = scorer.score(pred=pred)
        name = os.path.basename(path)[:-len(".jsonl")]
        run, step = name.rsplit("-", 1)
        out.append({"id": f"{run}@{step}", "macro_f1": s["macro_f1"], "accuracy": s["accuracy"],
                    "per_zone_f1": s["per_zone_f1"], "n": s["n"]})
    return out


CONDITIONS = [  # conditions.py: the same 985 test_plain clips (+ 200 Seoul-Gyeonggi for standard vs regional)
    ("원본", "AI Hub 녹음 그대로"),
    ("사이트 전처리", "사이트처럼 앞뒤 무음을 자르고 최대 음량 0.9로 키움"),
    ("사이트+잡음제거", "사이트 전처리 + 학습 때 못 본 잡음 제거(ffmpeg afftdn)"),
    ("무음화", "단어 사이 조용한 부분을 −40 dB로 (브라우저 잡음 제거 흉내)"),
    ("말소리만", "배경 구간을 잘라 내고 말소리만"),
    ("앞 2초", "앞 2초만 (짧은 답변)"),
    ("배경만", "말소리 없이 배경만 · 낮을수록(0.5·25%에 가까울수록) 배경에 덜 기댐"),
]


def current_system():
    """The served four-region run with its standard detector (accent.py STD_RUN): what the app uses since 10/9."""
    load = lambda *p: maybe(os.path.join(RUNS, *p)) or {}  # noqa: E731
    region = os.path.basename(os.path.realpath(os.path.join(RUNS, "served")))
    std = os.path.basename(os.path.realpath(os.path.join(RUNS, "served_std")))
    spec = load(std, "ensemble.json")
    m = load(region, "metrics.json")
    tests = {k[3:]: {"macro_f1": v["macro_f1"], "accuracy": v["accuracy"], "per_zone_f1": v["per_zone_f1"], "n": v["n"]}
             for k, v in m.items() if k.startswith("r4_")}
    rc, rc0 = load(region, "conditions.json"), load(f"{region}_noaug", "conditions.json")
    rc.setdefault("baseline_served", {}), rc.setdefault("conditions", {}), rc0.setdefault("conditions", {})
    dc = {name: load(run, "conditions_binary.json").get("conditions", {})
          for name, run in (("previous", "ensemble_big"), ("binary", "std2"), ("zone6_aug", "zone6_aug"), ("served", std))}
    rows = []
    for c, desc in CONDITIONS:
        rows.append({"name": c, "desc": desc,
                     "region": {"previous": rc["baseline_served"].get(c), "noaug": rc0["conditions"].get(c, {}).get("balanced_recall"),
                                "served": rc["conditions"].get(c, {}).get("balanced_recall")},
                     "detector": {k: v.get(c) for k, v in dc.items()}})
    return {"name": "4지역 + 표준 억양 판별기", "region_run": region, "detector_run": std,
            "since": os.lstat(os.path.join(RUNS, "served")).st_mtime, "zones": ["강원", "충청", "경상", "제주"],
            "detector_bias": spec.get("served_standard_bias"), "detector_bias_note": spec.get("bias_chosen_on", ""),
            "detector_members": spec.get("members", [{"run": std, "weight": 1.0}]), "tests": tests, "conditions": rows,
            "notes": [
                "서울·경기는 학습하지 않아요(표준어 기준). 지역 정답은 녹음된 AI Hub 세트예요: 제주 세트의 경상 출신은 제주에 살며 제주 말을 써요.",
                "말소리 없이 배경만 들려줘도 이전 모델은 지역을 맞혔어요(4지역 57%, 무작위 25%). 그래서 학습 클립마다 녹음 환경을 바꾼 복사본 2개를 더했어요.",
                "지역 억양이 있는지는 따로 판별해요. 표준·지역 두 칸짜리 판별기는 AUC 0.735로 실패해서, 녹음 환경을 바꿔 학습한 6권역 모델과 이전 서비스 앙상블을 반반 섞어 서울·경기일 확률만 써요. 기준은 검증 세트로 정했어요.",
                "사투리 비율: 어느 지역인지 확실하지 않아도 지역 억양일 확률이 50% 이상이면 억양 강도만큼 반영해요.",
                "알려진 위험: 시험 세트에서 서울·경기 화자의 23~31%를 지역 억양으로 판정해요. 실제 브라우저 녹음으로 확인할 예정이에요.",
            ]}


# Page names of the four-region runs and standard detectors (other runs show their folder name).
NAMES = {"region4": "4지역 · 녹음 환경 변형", "region4_noaug": "4지역 · 변형 없이", "std2": "두 칸(표준·지역) 판별기",
         "std2_noaug": "두 칸 판별기 · 변형 없이", "ensemble_big": "이전 서비스 앙상블(6권역)", "zone6_aug": "6권역 · 녹음 환경 변형",
         "detector_mix": "섞은 판별기 (이전 앙상블 + 6권역 변형)"}
WATCH = ("metrics.json", "test_logits.npz", "conditions.json", "conditions_binary.json", "ensemble.json")


def maybe(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def sessions():
    """Answer-level scores of the served system (session_eval.py): 1-8 sentences of one speaker, averaged by
    length as the app does, and the app's final verdict under the decision rules compared there."""
    rep = maybe(os.path.join(RUNS, "session_eval", "report.json"))
    if not rep:
        return None
    return {"speakers": rep["speakers"], "clips": rep["clips"], "system": rep.get("system"),
            "conditions": rep["conditions"], "rules": maybe(os.path.join(RUNS, "session_eval", "rules.json")) or {},
            "time": mtime(os.path.join(RUNS, "session_eval", "report.json"))}


def newest_input():
    """The newest file the page is built from (run results, served links, Gemma bench files, this script)."""
    paths = [p for w in WATCH for p in glob.glob(os.path.join(RUNS, "*", w))]
    paths += [os.path.join(RUNS, "session_eval", n) for n in ("report.json", "rules.json")]
    paths += glob.glob(os.path.join(WEB_STATE, "bench", "lora_v2r-*.jsonl")) + [os.path.abspath(__file__)]
    times = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    times += [os.lstat(os.path.join(RUNS, n)).st_mtime for n in ("served", "served_std") if os.path.lexists(os.path.join(RUNS, n))]
    return max(times)


def run_dirs():
    """Real run folders in runs/, oldest first (the served / served_std links are not runs)."""
    names = [n for n in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, n)) and not os.path.islink(os.path.join(RUNS, n))]
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(RUNS, n)))


def discover_six(known):
    """Six-zone runs with test logits that ITEMS does not list yet: shown as "새 실험" until named there."""
    out = []
    for d in run_dirs():
        npz = os.path.join(RUNS, d, "test_logits.npz")
        if d in known or not os.path.exists(npz) or not os.path.exists(os.path.join(RUNS, d, "metrics.json")):
            continue
        z = np.load(npz)
        if all(t in z.files and z[t].ndim == 2 and z[t].shape[1] == len(ZONES) for t, *_ in TESTS):
            out.append({"id": d, "name": f"{d} (새 실험)", "short": d, "members": [(d, 1.0)], "bias": 0.0, "status": "new",
                        "desc": "자동으로 추가된 실험 · 이름과 설명은 bench_export.py ITEMS에 적으면 바뀌어요"})
    return out


def other_runs(region, std, previous):
    """Four-region runs (r4_* tests) and standard detectors (conditions_binary.json), newest last."""
    runs4, detectors = [], []
    members = {m["run"] for m in (maybe(os.path.join(RUNS, std, "ensemble.json")) or {}).get("members", [])}
    for d in run_dirs():
        m, rc = maybe(os.path.join(RUNS, d, "metrics.json")) or {}, maybe(os.path.join(RUNS, d, "conditions.json")) or {}
        if any(k.startswith("r4_") for k in m):
            runs4.append({"id": d, "name": NAMES.get(d, d), "status": "served" if d == region else "experiment",
                          "time": os.path.getmtime(os.path.join(RUNS, d, "metrics.json")),
                          "tests": {k[3:]: v["macro_f1"] for k, v in m.items() if k.startswith("r4_") and isinstance(v, dict)},
                          "conditions": {c: v.get("balanced_recall") for c, v in rc.get("conditions", {}).items()}})
        b = maybe(os.path.join(RUNS, d, "conditions_binary.json"))
        if b:
            detectors.append({"id": d, "name": NAMES.get(d, d), "status": "served" if d == std else "was_served" if d == previous else "member" if d in members else "experiment",
                              "time": os.path.getmtime(os.path.join(RUNS, d, "conditions_binary.json")),
                              "bias": b.get("standard_bias"), "conditions": b.get("conditions", {})})
    return runs4, detectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-changed", action="store_true", help="do nothing unless an input is newer than runs/benchmark.json")
    args = ap.parse_args()
    path = os.path.join(RUNS, "benchmark.json")
    if args.if_changed and os.path.exists(path) and newest_input() <= os.path.getmtime(path):
        return
    rows = {s: read_jsonl(os.path.join(HERE, "data", f"{s}.jsonl")) for s, *_ in TESTS}
    scorers = {s: Scorer(rows[s]) for s in rows}
    known = {it["id"] for it in ITEMS} | {r for it in ITEMS for r, _ in it["members"]}
    todo = ITEMS + discover_six(known)
    runs = sorted({r for it in todo for r, _ in it["members"]})
    logits = {r: np.load(os.path.join(RUNS, r, "test_logits.npz")) for r in runs}
    served = os.path.basename(os.path.realpath(os.path.join(RUNS, "served")))
    spec = os.path.join(RUNS, served, "ensemble.json")
    served_bias = json.load(open(spec)).get("served_standard_bias") if os.path.exists(spec) else None
    items = []
    for it in todo:
        single = len(it["members"]) == 1
        run_dir = os.path.join(RUNS, it["id"])
        if not os.path.isdir(run_dir):  # a mix that was only reviewed: dated by its newest member
            run_dir = os.path.join(RUNS, it["members"][-1][0])
        metrics = json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))
        status = it["status"]
        if status == "served" and (it["id"] != served or it["bias"] != served_bias):
            raise SystemExit(f"runs/served is {served} with bias {served_bias}; update ITEMS")
        item = {"id": it["id"], "name": it["name"], "short": it["short"], "desc": it["desc"], "note": it.get("note", ""), "status": status,
                "members": [{"run": r, "weight": round(w, 4)} for r, w in it["members"]], "bias": it["bias"],
                "time": mtime(os.path.join(run_dir, "metrics.json")),
                "dev_macro_f1": metrics["dev"]["macro_f1"] if single and "dev" in metrics else None,
                "train": train_size(it["data"]) if it.get("data") else None,
                "tests": {s: scorers[s].score(probs(it["members"], logits, s, it["bias"])) for s in rows}}
        items.append(item)
        print(f"{it['id']:13s} " + " ".join(f"{s} {item['tests'][s]['macro_f1']['value']:.3f}" for s in rows), flush=True)
    out = {"generated": time.time(), "zones": ZONES, "served": served, "bootstrap": B,
           "previous_served": next((i["id"] for i in reversed(items) if i["status"] == "was_served"), None),
           "current": current_system() if served not in {i["id"] for i in items} else None,
           "tests": [{"id": s, "name": name, "short": short, "desc": desc, "n": len(rows[s]),
                      "recordings": len({r["recording"] for r in rows[s]}), "persons": len({r["person"] for r in rows[s]})}
                     for s, name, desc, short in TESTS],
           "items": items, "curve": curve(items), "gemma": gemma(rows["bench"], scorers["bench"]), "sessions": sessions()}
    out["runs4"], out["detectors"] = other_runs(served, os.path.basename(os.path.realpath(os.path.join(RUNS, "served_std"))),
                                                out["previous_served"])
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print(f"wrote {path}: {len(items)} items, curve {len(out['curve'])} points, gemma {[g['id'] for g in out['gemma']]}")


if __name__ == "__main__":
    main()
