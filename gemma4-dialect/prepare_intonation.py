"""Build an intonation test set: utterances whose words alone do not give the answer.

All subsets come from AI Hub's Validation split (never used for training) and
skip the utterances already in data_all/valid.jsonl. "Zone" is the speaker's accent
zone (prepare_data.ZONES), not the dataset the utterance came from.

  question  The same words said as a question and as a statement ("그래?" / "그래."),
            the same number of each per sentence. Only intonation tells them apart,
            so score_intonation.py checks whether the model ends its transcript with
            "?" exactly when the speaker asked.
  region    The same plain sentence (no dialect words) said in three or more zones,
            one utterance per zone. The text is identical, so a gap in accuracy
            between zones comes from accent and intonation, not vocabulary.
  origin    Plain utterances (no dialect words), the same number per zone and at most
            a few per recording. A model with a `지역:` answer line has to name the
            zone from accent and intonation alone.

    python prepare_intonation.py --dry-run            # counts only
    python prepare_intonation.py --out data_intonation
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

from prepare_data import REGIONS, cut_recording, index_recordings, load_utterances

STRIP = re.compile(r"[\s.,?!~'\"“”‘’·…-]")  # same as common._STRIP


def words_key(text):
    return STRIP.sub("", text)


def is_question(text):
    return text.rstrip().endswith("?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aihub", default="/data/aihub")
    ap.add_argument("--out", default="data_intonation")
    ap.add_argument("--datasets", default="118,119,121,122")
    ap.add_argument("--exclude", default="data_all/valid.jsonl")
    ap.add_argument("--question-per-side", type=int, default=2,
                    help="max questions and max statements kept per sentence and region")
    ap.add_argument("--question-per-region", type=int, default=600)
    ap.add_argument("--region-min-regions", type=int, default=3)
    ap.add_argument("--region-min-words", type=int, default=2)
    ap.add_argument("--origin-per-zone", type=int, default=200)
    ap.add_argument("--origin-per-recording", type=int, default=5)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    excluded = set()
    for path in filter(None, args.exclude.split(",")):
        excluded |= {json.loads(line)["id"] for line in open(path, encoding="utf-8")}

    utts, where = [], {}
    for ds in args.datasets.split(","):
        ds_dir = os.path.join(args.aihub, ds)
        recs = index_recordings(ds_dir)
        for u in load_utterances(ds_dir, args.min_sec, args.max_sec).get("valid", []):
            # one sentence per utterance: at most a final "?"
            if u["id"] in excluded or u["recording"] not in recs or "?" in u["dialect_form"].rstrip()[:-1]:
                continue
            if not words_key(u["dialect_form"]):
                continue
            utts.append({**u, "region": REGIONS[ds], "split": "valid"})
            where[u["recording"]] = recs[u["recording"]]
        print(f"{REGIONS[ds]}: {len(utts)} usable validation utterances so far", flush=True)

    # pairs are made within one region, so a question is compared with a statement in the same accent
    groups = defaultdict(lambda: ([], []))  # (region, sentence) -> (statements, questions)
    for u in utts:
        groups[u["region"], words_key(u["dialect_form"])][is_question(u["dialect_form"])].append(u)
    sets = defaultdict(list)
    for (region, key), (st, qu) in sorted(groups.items()):
        k = min(len(st), len(qu), args.question_per_side)
        if k:
            sets[region].append([{**u, "subset": "question", "group": key} for u in rng.sample(st, k) + rng.sample(qu, k)])
    chosen = []
    for region, ss in sorted(sets.items()):
        rng.shuffle(ss)
        n = 0
        for s in ss:
            if n + len(s) > args.question_per_region:
                continue
            chosen += s
            n += len(s)

    taken = {u["id"] for u in chosen}
    plain = [u for u in utts if not u["has_dialect"] and u["origin"] and u["id"] not in taken
             and len(u["dialect_form"].split()) >= args.region_min_words]
    by_text = defaultdict(lambda: defaultdict(list))
    for u in plain:
        by_text[words_key(u["dialect_form"])][u["origin"]].append(u)
    for key, zones in sorted(by_text.items()):
        if len(zones) >= args.region_min_regions:
            chosen += [{**rng.choice(us), "subset": "region", "group": key} for _, us in sorted(zones.items())]

    taken = {u["id"] for u in chosen}
    by_zone = defaultdict(list)
    for u in plain:
        if u["id"] not in taken:
            by_zone[u["origin"]].append(u)
    for zone, us in sorted(by_zone.items()):
        rng.shuffle(us)
        per_rec, n = Counter(), 0
        for u in us:
            if n == args.origin_per_zone:
                break
            if per_rec[u["recording"]] < args.origin_per_recording:
                chosen.append({**u, "subset": "origin", "group": words_key(u["dialect_form"])})
                per_rec[u["recording"]] += 1
                n += 1

    count = Counter((u["subset"], u["region"] if u["subset"] == "question" else u["origin"]) for u in chosen)
    for subset in ("question", "region", "origin"):
        rows = [u for u in chosen if u["subset"] == subset]
        print(f"{subset}: {len(rows)} utterances, {len({u['group'] for u in rows})} sentences, "
              f"{sum(is_question(u['dialect_form']) for u in rows)} questions, "
              f"by {'dataset' if subset == 'question' else 'zone'} "
              f"{dict(sorted((r, n) for (s, r), n in count.items() if s == subset))}", flush=True)
    if args.dry_run:
        return

    meta = {u["id"]: u for u in chosen}
    by_rec = defaultdict(list)
    for u in chosen:
        by_rec[u["recording"]].append(u)
    jobs = [(where[r], us, args.out) for r, us in by_rec.items()]
    rows, errors = [], []
    with Pool(args.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(cut_recording, jobs), 1):
            rows += r
            if err:
                errors.append(err)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} recordings cut", flush=True)

    for row in rows:
        u = meta[row["id"]]
        row.update(subset=u["subset"], group=u["group"], is_question=is_question(row["dialect_form"]),
                   n_words=len(row["dialect_form"].split()))
    rows.sort(key=lambda r: (r["subset"], r["group"], r["region"], r["id"]))
    with open(os.path.join(args.out, "test.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"test: {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.1f} h", flush=True)
    with open(os.path.join(args.out, "prepare_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "counts": {f"{s}/{r}": n for (s, r), n in sorted(count.items())},
                   "errors": errors}, f, ensure_ascii=False, indent=2)
    if errors:
        print(f"{len(errors)} recordings failed; see prepare_stats.json", flush=True)


if __name__ == "__main__":
    main()
