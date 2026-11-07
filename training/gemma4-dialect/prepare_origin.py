"""Add each speaker's accent zone to the manifests, and top up the zones with few speakers.

The `지역:` answer line is the speaker's accent zone (prepare_data.ZONES), not the dataset
the utterance came from: the Chungcheong set, for example, also records speakers born
in Seoul or Daegu.

    python prepare_origin.py data_more/train.jsonl data_all/valid.jsonl --extra-out data_zone_extra

writes `<manifest>_origin.jsonl` next to each manifest with `origin`, `birthplace` and
`residence` added. Utterances without a zone are dropped from the `--drop-unknown`
manifests (training) and kept in the others so scores stay comparable with earlier runs.
Seoul-Gyeonggi and Jeolla speakers are rare in data_more (about 2,700 and 1,600
utterances), so `--extra-out` cuts more of their unused Training-split utterances.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

from prepare_data import REGIONS, cut_recording, index_recordings, load_utterances

USED = ("data_118/train.jsonl,data_rest/train.jsonl,data_all/train.jsonl,data_more/train.jsonl,"
        "data_all/valid.jsonl,data_intonation/test.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--aihub", default="/data/aihub")
    ap.add_argument("--datasets", default="118,119,121,122")
    ap.add_argument("--drop-unknown", default="data_more/train.jsonl",
                    help="comma-separated manifests whose utterances without a zone are dropped")
    ap.add_argument("--used", default=USED, help="manifests whose utterances must not be picked as extra")
    ap.add_argument("--extra-out", default=None)
    ap.add_argument("--extra-zones", default="서울·경기,전라")
    ap.add_argument("--extra-per-zone", type=int, default=22000)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = {m: [json.loads(line) for line in open(m, encoding="utf-8")] for m in args.manifests}
    wanted = {r["id"] for rs in rows.values() for r in rs}
    used = set()
    for path in filter(None, args.used.split(",")):
        if os.path.exists(path):
            used |= {json.loads(line)["id"] for line in open(path, encoding="utf-8")}
    zones = set(filter(None, args.extra_zones.split(",")))

    meta, extra_pool, where = {}, defaultdict(list), {}
    for ds in args.datasets.split(","):
        ds_dir = os.path.join(args.aihub, ds)
        recs = index_recordings(ds_dir) if args.extra_out else {}
        for split, us in load_utterances(ds_dir, args.min_sec, args.max_sec).items():
            for u in us:
                if u["id"] in wanted:
                    meta[u["id"]] = (u["origin"], u["birthplace"], u["residence"])
                elif (args.extra_out and split == "train" and u["origin"] in zones
                      and u["id"] not in used and u["recording"] in recs):
                    extra_pool[u["origin"]].append({**u, "region": REGIONS[ds], "split": "train"})
                    where[u["recording"]] = recs[u["recording"]]
        print(f"{REGIONS[ds]}: {len(meta)} of {len(wanted)} manifest utterances matched", flush=True)

    drop = set(filter(None, args.drop_unknown.split(",")))
    for m, rs in rows.items():
        table, kept = Counter(), []
        for r in rs:
            origin, bp, res = meta.get(r["id"], (None, "", ""))
            r.update(origin=origin, birthplace=bp, residence=res)
            table[r["region"], origin or "-"] += 1
            if origin or m not in drop:
                kept.append(r)
        out = m[:-len(".jsonl")] + "_origin.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n{out}: {len(kept)} of {len(rs)} utterances")
        for region in sorted({k[0] for k in table}):
            cells = sorted(((o, n) for (g, o), n in table.items() if g == region), key=lambda x: -x[1])
            print(f"  {region} set: " + ", ".join(f"{o} {n}" for o, n in cells), flush=True)

    if not args.extra_out:
        return
    rng = random.Random(args.seed)
    chosen = []
    for z in sorted(zones):
        pool_z = extra_pool[z]
        chosen += rng.sample(pool_z, min(len(pool_z), args.extra_per_zone))
        print(f"extra {z}: {min(len(pool_z), args.extra_per_zone)} of {len(pool_z)} unused utterances", flush=True)
    by_rec = defaultdict(list)
    for u in chosen:
        by_rec[u["recording"]].append(u)
    jobs = [(where[r], us, args.extra_out) for r, us in by_rec.items()]
    out_rows, errors = [], []
    with Pool(args.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(cut_recording, jobs), 1):
            out_rows += r
            if err:
                errors.append(err)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} recordings cut", flush=True)
    out_rows.sort(key=lambda x: x["id"])
    with open(os.path.join(args.extra_out, "train.jsonl"), "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"extra train: {len(out_rows)} clips, {sum(r['duration'] for r in out_rows) / 3600:.1f} h, "
          f"{dict(Counter(r['origin'] for r in out_rows))}", flush=True)
    with open(os.path.join(args.extra_out, "prepare_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "errors": errors}, f, ensure_ascii=False, indent=2)
    if errors:
        print(f"{len(errors)} recordings failed; see prepare_stats.json", flush=True)


if __name__ == "__main__":
    main()
