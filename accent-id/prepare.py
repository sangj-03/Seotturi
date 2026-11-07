"""Build the accent-zone data set: clips -> speaker, person and zone, then the splits.

    python prepare.py index      # data/clips.jsonl: every cut clip with its speaker's metadata and zone
    python prepare.py split      # data/{train,dev,test_plain,test_same,test_dialect,bench}.jsonl

Clips: the WAVs already cut for the Gemma training (/data/seotturi/gemma4-dialect), so nothing is cut
again. Training clips come from AI Hub's Training split, test clips from its Validation split.

Zone: the speaker's accent zone, given only when birthplace and principal residence fall in the same
zone (gemma4-dialect/prepare_data.zone_of). Jeolla is left out of this classifier (six zones).

Speakers: AI Hub numbers the two speakers inside each recording ("1", "2") and has no person ids, so a
person who recorded several conversations has several unrelated "slots", and nothing in the labels joins
them: profiles (age, sex, occupation, birthplace, residences, education) repeat across thousands of slots,
and speaker-verification embeddings rate the two people of one conversation almost as alike as one person
(shared phone and room). The splits are therefore by recording: train and dev never share a recording, and
the tests come from AI Hub's Validation split, a different set of recordings. leakage.py checks afterwards
whether test voices also occur in training.
"""

import argparse
import collections
import json
import os
import random
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

GEMMA = "/data/seotturi/gemma4-dialect"
sys.path.insert(0, GEMMA)
from prepare_data import BAD_TEXT, find_zips, split_of, zone_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ZONES = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "제주"]
DATASETS = ("118", "119", "121", "122")
MANIFESTS = {  # manifest -> AI Hub split its clips come from
    "data_more/train_origin.jsonl": "train", "data_zone_extra/train.jsonl": "train",
    "data_all/train.jsonl": "train", "data_118/train.jsonl": "train", "data_rest/train.jsonl": "train",
    "data_intonation/test.jsonl": "valid", "data_all/valid_origin.jsonl": "valid",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def scan_labels(args):
    """utterance id -> speaker slot and metadata, for the ids we have clips of."""
    zpath, wanted = args
    out = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            d = json.loads(z.read(name).decode("utf-8-sig"))
            speakers = {str(s.get("id")): s for s in d.get("speaker", [])}
            for u in d.get("utterance", []):
                if u["id"] not in wanted:
                    continue
                sid = str(u.get("speaker_id"))
                s = speakers.get(sid, {})
                bp, res = (s.get("birthplace") or "").strip(), (s.get("principal_residence") or "").strip()
                out[u["id"]] = {"recording": d["id"], "slot": f"{d['id']}#{sid}", "zone": zone_of(bp, res),
                                "age": s.get("age") or "", "sex": s.get("sex") or "", "birthplace": bp, "residence": res,
                                "occupation": s.get("occupation") or "", "education": s.get("education") or "",
                                "current": (s.get("current_residence") or "").strip()}
    return out


def index(_):
    clips = {}
    for rel, split in MANIFESTS.items():
        base = os.path.dirname(os.path.join(GEMMA, rel))
        for r in read_jsonl(os.path.join(GEMMA, rel)):
            path = r["audio"] if os.path.isabs(r["audio"]) else os.path.join(base, r["audio"])
            c = clips.setdefault(r["id"], {"id": r["id"], "path": path, "split": split, "duration": r["duration"],
                                           "plain": not r["has_dialect"], "text": r["standard_form"],
                                           "dataset": r["region"]})
            if rel.startswith("data_intonation"):  # keep the test subsets (origin / region / question)
                c["subset"], c["group"] = r["subset"], r.get("group")
    missing = [i for i, c in clips.items() if not os.path.exists(c["path"])]
    for i in missing:
        del clips[i]
    print(f"{len(clips)} clips ({len(missing)} listed without a file)", flush=True)
    zips = [z for ds in DATASETS for z in find_zips(f"/data/aihub/{ds}", "라벨링데이터") if split_of(z)]
    wanted = set(clips)
    meta = {}
    with Pool(12) as p:
        for part in p.imap_unordered(scan_labels, [(z, wanted) for z in zips]):
            meta.update(part)
    rows = []
    for i, c in clips.items():
        m = meta.get(i)
        if m and m["zone"] in ZONES:
            rows.append({**c, **m})
    os.makedirs(DATA, exist_ok=True)
    write_jsonl(os.path.join(DATA, "clips.jsonl"), rows)
    by = collections.defaultdict(lambda: [0, set()])
    for r in rows:
        by[(r["split"], r["zone"])][0] += 1
        by[(r["split"], r["zone"])][1].add(r["slot"])
    for (split, zone), (n, slots) in sorted(by.items()):
        print(f"  {split:5s} {zone:9s} clips {n:7d} slots {len(slots):6d}")


def pick_clips(rs, cap, plain_share, rng):
    """At most `cap` clips of one person, about `plain_share` of them without dialect words."""
    rs = rs[:]
    rng.shuffle(rs)
    plain = [r for r in rs if r["plain"]]
    dial = [r for r in rs if not r["plain"]]
    n_plain = min(len(plain), max(round(cap * plain_share), cap - len(dial)))
    return plain[:n_plain] + dial[:cap - n_plain]


def split(args):
    rows = read_jsonl(os.path.join(DATA, "clips.jsonl"))
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    for r in rows:
        r["person"] = r["slot"]  # AI Hub has no person ids (see the module docstring)
        r["y"] = ZONES.index(r["zone"])
    rng = random.Random(args.seed)

    # Test sets: AI Hub Validation split.
    valid = [r for r in rows if r["split"] == "valid"]
    test_plain = [r for r in valid if r.get("subset") == "origin"]       # no dialect words, 200 per zone
    region = [r for r in valid if r.get("subset") == "region"]           # one sentence said in several zones
    zones_per_group = collections.defaultdict(set)
    for r in region:
        zones_per_group[r["group"]].add(r["zone"])
    test_same = [r for r in region if len(zones_per_group[r["group"]]) >= 2]
    test_dialect = [r for r in valid if "subset" not in r]               # the Gemma validation set, dialect words included
    bench_ids = {json.loads(l)["id"] for l in open("/data/seotturi/seotturi-web/state/bench/lora_v2r-1000.jsonl")
                 if json.loads(l)["kind"] != "asr"}
    bench = [r for r in test_plain if r["id"] in bench_ids]              # the engine's zone benchmark, Jeolla left out

    # Training pool: Training split, 2-15 s, nobody who also speaks in a test clip, and by default only
    # clips without dialect words: the clips cut for the Gemma training hold dialect words in 81-85 % of the
    # regional clips but 18 % of the Seoul-Gyeonggi ones, so with them "no dialect words" would mean
    # "standard" and the classifier would learn the words instead of the accent.
    held = {r["person"] for r in valid}
    pool = [r for r in rows if r["split"] == "train" and r["person"] not in held and 2 <= r["duration"] <= 15
            and (r["plain"] or not args.plain_only)]
    by_person = collections.defaultdict(list)
    for r in pool:
        by_person[r["person"]].append(r)
    zone_persons = collections.defaultdict(list)
    recordings = collections.defaultdict(set)
    for p, rs in by_person.items():
        zone_persons[rs[0]["zone"]].append(p)
        recordings[rs[0]["zone"]].add(rs[0]["recording"])
    # Dev takes whole recordings, so the two people of one conversation are never split between dev and train.
    dev_recs = set()
    for zone in ZONES:
        recs = sorted(recordings[zone])
        rng.shuffle(recs)
        want = min(args.dev_max, max(20, round(len(zone_persons[zone]) * args.dev_share)))
        have = 0
        for rec in recs:
            if have >= want:
                break
            dev_recs.add(rec)
            have += sum(1 for p in zone_persons[zone] if by_person[p][0]["recording"] == rec)
    dev, train = [], []
    cell = lambda p: (by_person[p][0]["age"], by_person[p][0]["sex"])
    chosen = {}
    for zone, ps in zone_persons.items():
        ps.sort()
        rng.shuffle(ps)
        for p in ps:
            if by_person[p][0]["recording"] in dev_recs:
                dev += pick_clips(by_person[p], args.dev_cap, args.plain_share, rng)
        chosen[zone] = [p for p in ps if by_person[p][0]["recording"] not in dev_recs]
    # Same age and sex mix in every zone, set by the zone with the fewest speakers (Seoul-Gyeonggi), so that
    # age or sex cannot stand in for accent.
    ref = collections.Counter(cell(p) for p in chosen["서울·경기"])
    for zone, ps in chosen.items():
        if zone == "서울·경기":
            keep = ps
            cap = args.seoul_cap
        else:
            per_cell = collections.defaultdict(list)
            for p in ps:
                per_cell[cell(p)].append(p)
            keep = [p for c, n in ref.items() for p in per_cell.get(c, [])[:args.k * n]]
            cap = args.cap
        for p in keep:
            train += pick_clips(by_person[p], cap, args.plain_share, rng)
    for name, part in (("train", train), ("dev", dev), ("test_plain", test_plain), ("test_same", test_same),
                       ("test_dialect", test_dialect), ("bench", bench)):
        write_jsonl(os.path.join(out, f"{name}.jsonl"), part)
        n = collections.Counter(r["zone"] for r in part)
        persons = collections.defaultdict(set)
        for r in part:
            persons[r["zone"]].add(r["person"])
        plain = sum(r["plain"] for r in part) / max(1, len(part))
        print(f"{name:12s} {len(part):7d} clips, plain {plain:.0%}: " +
              ", ".join(f"{z} {n[z]}/{len(persons[z])}명" for z in ZONES))
    assert not {r["recording"] for r in train} & {r["recording"] for r in dev}
    ages = collections.defaultdict(collections.Counter)
    seen = set()
    for r in train:
        if r["person"] not in seen:
            seen.add(r["person"])
            ages[r["zone"]][r["age"]] += 1
    for z in ZONES:
        tot = sum(ages[z].values())
        print(f"  train ages {z:9s} " + ", ".join(f"{a} {v / tot:.0%}" for a, v in sorted(ages[z].items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["index", "split"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-share", type=float, default=0.1, help="share of each zone's training persons kept for dev")
    ap.add_argument("--dev-cap", type=int, default=6, help="clips per dev person")
    ap.add_argument("--dev-max", type=int, default=200, help="dev persons per zone at most")
    ap.add_argument("--k", type=int, default=10, help="persons per age-sex cell, as a multiple of Seoul-Gyeonggi's")
    ap.add_argument("--seoul-cap", type=int, default=20, help="clips per Seoul-Gyeonggi training person")
    ap.add_argument("--cap", type=int, default=12, help="clips per training person, other zones")
    ap.add_argument("--plain-share", type=float, default=0.7, help="share of clips without dialect words (with --mixed)")
    ap.add_argument("--mixed", dest="plain_only", action="store_false", help="also train on clips with dialect words")
    ap.add_argument("--out", default="data", help="folder for the split files (dev and tests depend only on --seed)")
    args = ap.parse_args()
    {"index": index, "split": split}[args.step](args)


if __name__ == "__main__":
    main()
