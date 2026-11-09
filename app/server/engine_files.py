"""The engine's files under app/state, read without the engine: served adapter, update history, benchmark results.

engine.py writes them and builds /status and /bench from them; app.py reads the same files when the engine is
not running, so the landing page and /benchmark keep showing the last results. Only the standard library, so the
web server can import it without torch.
"""
import glob
import json
import os

# The zones of the benchmark (scripts/bench_score.py, web/js/benchmark.js).
ZONES = {"서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "전라", "제주"}


def read_state(path):
    """engine_state.json: the served adapter (active), update history, cached scores, rejected checkpoints."""
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {}
    for k, v in (("history", []), ("scores", {}), ("rejected", []), ("active", None)):
        s.setdefault(k, v)
    return s


def bench_summaries(bench_dir):
    """Every benchmark summary (bench/<id>.json) by adapter id, zone answers outside ZONES counted as "-"."""
    out = {}
    for p in glob.glob(os.path.join(bench_dir, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                b = json.load(f)
            out[b["id"]] = fold_off_format(b)
        except (OSError, ValueError, KeyError):
            continue
    return out


def fold_off_format(b):
    """Count every zone answer outside ZONES as "-" in the confusion matrices, as /benchmark already shows them.
    When the model broke the answer format, the zone it gave can run on into its transcript of an AI Hub clip, and
    AI Hub's terms do not allow showing that to others."""
    for v in (b.get("metrics") or {}).values():
        conf = v.get("confusion") if isinstance(v, dict) else None
        if not isinstance(conf, dict):
            continue
        for gold, preds in list(conf.items()):
            if isinstance(preds, dict):
                folded = {}
                for p, n in preds.items():
                    k = p if p in ZONES else "-"
                    folded[k] = folded.get(k, 0) + n
                conf[gold] = folded
    return b


def sorted_summaries(benches):
    """Oldest training state first."""
    return sorted(benches.values(), key=lambda b: (b["cumulative_step"], b["final"]))


def compact(b):
    """The headline numbers of a benchmark summary (also kept in the update history)."""
    m = b["metrics"]
    asr = m.get("asr") or {}
    zone = m.get("zone_plain") or {}
    return {"id": b["id"], "run": b["run"], "step": b["step"], "final": b["final"], "format": b["format"],
            "cumulative_step": b["cumulative_step"], "time": b["time"],
            "cer_dialect": asr.get("cer_dialect", {}).get("value"), "cer_standard": asr.get("cer_standard", {}).get("value"),
            "cer_standard_ci": asr.get("cer_standard", {}).get("ci"), "cer_dialect_ci": asr.get("cer_dialect", {}).get("ci"),
            "swer_standard": asr.get("swer_standard", {}).get("value"), "chrfpp": asr.get("chrfpp_standard"),
            "err": asr.get("err"), "recall": asr.get("dialect_word_recall"), "format_ok": asr.get("format_ok"),
            "zone_macro_f1": (zone.get("macro_f1") or {}).get("value"), "zone_accuracy": (zone.get("accuracy") or {}).get("value"),
            "rtf": m.get("rtf"), "n": asr.get("n")}


def bench_set(path):
    """How many benchmark clips of each kind (asr, zone) state/benchmark.jsonl holds."""
    out = {"asr": 0, "zone": 0}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                kind = json.loads(line).get("kind")
                if kind in out:
                    out[kind] += 1
    except (OSError, ValueError):
        pass
    return out
