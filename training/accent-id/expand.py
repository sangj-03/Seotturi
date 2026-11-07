"""Grow the training set with clips cut straight from the AI Hub recordings (data_big holds only clips that
were already cut for the Gemma training).

    python expand.py            # clips_x2/<zone>/<id>.wav, data_x2/*.jsonl, data/train_extra2.jsonl
    python features.py train_extra2 && python assemble_x2.py && python train_head.py --train train_x2 --out runs/phase1_x2

Candidates are AI Hub Training utterances that data_big could have used (six zones by birthplace and residence,
no dialect words, 2-15 s, clean text, the age-sex cells of the Seoul-Gyeonggi persons) and that are not in
data_big's train or dev (dev recordings stay out entirely). Added:
  every person (recording slot) not yet in training, at most --per-new clips each: all the new speakers that are
  left (about 9,300; Seoul-Gyeonggi has none left);
  Seoul-Gyeonggi's remaining clips of the persons already in training, so the zone keeps its share.
dev and the tests are copied from data/ unchanged.
"""

import argparse
import collections
import json
import os
import random
import sys
import zipfile
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "gemma4-dialect"))
from prepare_data import BAD_TEXT, find_zips, index_recordings, read_wav, split_of, write_wav, zone_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "제주"]
DATASETS = {"118": "강원", "119": "경상", "121": "제주", "122": "충청"}


KEEP = {}  # filled in each worker: used ids, dev recordings, persons in training, Seoul-Gyeonggi's age-sex cells


def init(keep):
    KEEP.update(keep)


def scan(job):
    """Candidate utterances of one label zip (Training split): eligible, not used, and either from a person
    not yet in training or from Seoul-Gyeonggi (the rest is never chosen, so it is not kept in memory)."""
    zpath, dataset = job
    out = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            d = json.loads(z.read(name).decode("utf-8-sig"))
            sp = {str(s.get("id")): s for s in d.get("speaker", [])}
            for u in d.get("utterance", []):
                dia, std = (u.get("dialect_form") or "").strip(), (u.get("standard_form") or "").strip()
                try:
                    start, end = float(u["start"]), float(u["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (dia and std and not BAD_TEXT.search(dia) and not BAD_TEXT.search(std) and 2 <= end - start <= 15):
                    continue
                if any(e.get("isDialect") for e in u.get("eojeolList", [])) or dia != std:
                    continue
                sid = str(u.get("speaker_id"))
                s = sp.get(sid, {})
                bp, res = (s.get("birthplace") or "").strip(), (s.get("principal_residence") or "").strip()
                zone = zone_of(bp, res)
                slot = f"{d['id']}#{sid}"
                if (zone not in ZONES or u["id"] in KEEP["used"] or d["id"] in KEEP["dev_recs"]
                        or (s.get("age") or "", s.get("sex") or "") not in KEEP["cells"]
                        or (slot in KEEP["train_slots"] and zone != "서울·경기")):
                    continue
                out.append({"id": u["id"], "start": start, "end": end, "text": std, "dataset": dataset,
                            "recording": d["id"], "slot": slot, "zone": zone, "age": s.get("age") or "",
                            "sex": s.get("sex") or "", "birthplace": bp, "residence": res,
                            "occupation": s.get("occupation") or "", "education": s.get("education") or "",
                            "current": (s.get("current_residence") or "").strip()})
    return out


def cut(job):
    """Cut the chosen utterances of one recording to 16 kHz mono WAVs. Runs in a worker."""
    import torch
    import torchaudio.functional as AF
    torch.set_num_threads(1)
    (zpath, member), utts, out_dir = job
    try:
        audio, sr = read_wav(zpath, member)
    except Exception as e:  # noqa: BLE001  a broken recording should not stop the run
        return [], f"{member}: {e}"
    rows = []
    for u in utts:
        seg = audio[int(u["start"] * sr):int(u["end"] * sr)]
        if len(seg) < sr * 0.5:
            continue
        seg = AF.resample(torch.from_numpy(seg), sr, 16000).numpy()
        path = os.path.join(out_dir, u["zone"], f"{u['id']}.wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_wav(path, seg)
        row = {k: v for k, v in u.items() if k not in ("start", "end")}
        rows.append({**row, "path": path, "split": "train", "duration": round(len(seg) / 16000, 3), "plain": True,
                     "person": u["slot"], "y": ZONES.index(u["zone"])})
    return rows, None


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_big", help="the split that is grown (its dev and tests are kept)")
    ap.add_argument("--out", default="data_x2")
    ap.add_argument("--clips", default="clips_x2")
    ap.add_argument("--per-new", type=int, default=10, help="clips per person not yet in training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    base = os.path.join(HERE, args.base)
    train, dev = read_jsonl(os.path.join(base, "train.jsonl")), read_jsonl(os.path.join(base, "dev.jsonl"))
    used = {r["id"] for r in train} | {r["id"] for r in dev}
    dev_recs = {r["recording"] for r in dev}
    train_slots = {r["person"] for r in train}
    cells = {(r["age"], r["sex"]) for r in train if r["zone"] == "서울·경기"}

    jobs = [(z, DATASETS[ds]) for ds in DATASETS for z in find_zips(f"/data/aihub/{ds}", "라벨링데이터") if split_of(z) == "train"]
    keep = {"used": used, "dev_recs": dev_recs, "train_slots": train_slots, "cells": cells}
    with Pool(args.workers, initializer=init, initargs=(keep,)) as pool:
        cand = list({u["id"]: u for part in pool.map(scan, jobs) for u in part}.values())
    print(f"{len(cand)} candidate utterances", flush=True)

    rng = random.Random(args.seed)
    by_slot = collections.defaultdict(list)
    for u in sorted(cand, key=lambda u: u["id"]):
        by_slot[u["slot"]].append(u)
    chosen = []
    for slot in sorted(by_slot):
        us = by_slot[slot]
        if slot not in train_slots:
            rng.shuffle(us)
            chosen += us[:args.per_new]
        elif us[0]["zone"] == "서울·경기":
            chosen += us
    print("chosen:", dict(collections.Counter(u["zone"] for u in chosen)), flush=True)

    where = {}
    for ds in DATASETS:
        where.update(index_recordings(f"/data/aihub/{ds}"))
    by_rec = collections.defaultdict(list)
    missing = 0
    for u in chosen:
        if u["recording"] in where:
            by_rec[u["recording"]].append(u)
        else:
            missing += 1
    out_dir = os.path.join(HERE, args.clips)
    rows, errors = [], []
    with Pool(args.workers) as pool:
        for i, (rs, err) in enumerate(pool.imap_unordered(cut, [(where[r], us, out_dir) for r, us in by_rec.items()]), 1):
            rows += rs
            if err:
                errors.append(err)
            if i % 500 == 0:
                print(f"{i}/{len(by_rec)} recordings cut", flush=True)
    rows.sort(key=lambda r: r["id"])
    print(f"cut {len(rows)} clips ({missing} without a recording, {len(errors)} recordings failed)", flush=True)

    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "train.jsonl"), train + rows)
    for name in ("dev", "test_plain", "test_same", "test_dialect", "bench"):
        write_jsonl(os.path.join(out, f"{name}.jsonl"), read_jsonl(os.path.join(HERE, "data", f"{name}.jsonl")))
    write_jsonl(os.path.join(HERE, "data", "train_extra2.jsonl"), rows)
    with open(os.path.join(out, "expand_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "added": len(rows), "errors": errors[:50]}, f, ensure_ascii=False, indent=1)
    persons = collections.defaultdict(set)
    for r in train + rows:
        persons[r["zone"]].add(r["person"])
    n = collections.Counter(r["zone"] for r in train + rows)
    print(f"{args.out}/train: {len(train) + len(rows)} clips: " + ", ".join(f"{z} {n[z]}/{len(persons[z])}명" for z in ZONES))


if __name__ == "__main__":
    main()
