"""Fix the benchmark set (state/benchmark.jsonl) once, so every checkpoint is scored on the same clips.

    /data/seotturi/.venv/bin/python -I scripts/build_benchmark.py

asr  : 40 utterances per regional set (강원, 경상, 제주, 충청) from AI Hub's Validation split
       (data_all/valid_origin.jsonl, never trained on), 32 with dialect words and 8 without (the
       training mix), one per recording, outside the trainer's eval subset (first 40 per region)
       and the old 48-clip gate (rows 100-111).
zone : 15 plain utterances (no dialect words) per accent zone from the intonation test set's
       "origin" subset (data_intonation/test.jsonl), one per recording where possible.
"""

import json
import os
import random
from collections import defaultdict

GEMMA = "/data/seotturi/gemma4-dialect"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state", "benchmark.jsonl")
SEED = 20261006


def rows_of(path):
    base = os.path.dirname(path)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        r["audio_path"] = os.path.join(base, r["audio"])
        yield r


def pick(rows, n, rng, per_block=1):
    """n rows, at most per_block per recording, recordings in random order."""
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["id"].split(".")[0]].append(r)
    order = sorted(blocks)
    rng.shuffle(order)
    out = []
    for take in range(per_block):
        for b in order:
            if len(out) < n and len(blocks[b]) > take:
                out.append(blocks[b][take])
    return out


def main():
    rng = random.Random(SEED)
    out = []
    by_region = defaultdict(list)
    for r in rows_of(f"{GEMMA}/data_all/valid_origin.jsonl"):
        by_region[r["region"]].append(r)
    for region in sorted(by_region):
        rows = [r for i, r in enumerate(by_region[region]) if i >= 40 and not 100 <= i < 112]
        chosen = pick([r for r in rows if r["has_dialect"]], 32, rng) + pick([r for r in rows if not r["has_dialect"]], 8, rng)
        out += [{"kind": "asr", **r} for r in chosen]
    by_zone = defaultdict(list)
    for r in rows_of(f"{GEMMA}/data_intonation/test.jsonl"):
        if r.get("subset") == "origin":
            by_zone[r["origin"]].append(r)
    for zone in sorted(by_zone):
        chosen = pick(by_zone[zone], 15, rng)
        if len(chosen) < 15:
            chosen = pick(by_zone[zone], 15, rng, per_block=2)
        out += [{"kind": "zone", **r} for r in chosen]
    keep = ("kind", "id", "region", "origin", "has_dialect", "dialect_form", "standard_form", "audio_path", "duration")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n")
    asr = [r for r in out if r["kind"] == "asr"]
    zone = [r for r in out if r["kind"] == "zone"]
    print(f"{len(asr)} asr clips ({sum(r['duration'] for r in asr) / 60:.1f} min, "
          f"{len({r['id'].split('.')[0] for r in asr})} recordings), {len(zone)} zone clips "
          f"({sum(r['duration'] for r in zone) / 60:.1f} min) -> {OUT}")


if __name__ == "__main__":
    main()
