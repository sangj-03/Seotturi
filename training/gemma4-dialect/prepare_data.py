"""Build a speech fine-tuning set from the AI Hub dialect datasets (118-122).

Each AI Hub recording is a 10-20 minute conversation (44.1 kHz stereo WAV inside
large zips), and its label JSON lists utterances with start/end times plus a
dialect transcription (`dialect_form`) and its standard-Korean version
(`standard_form`). This script samples utterances per region, cuts them out of
the recordings, converts them to 16 kHz mono (the model's input rate) and
writes `<out>/<split>.jsonl` manifests next to the clips.

    python prepare_data.py --train-per-region 20000 --valid-per-region 300
"""

import argparse
import io
import json
import os
import random
import re
import subprocess
import wave
import zipfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torchaudio.functional as AF

REGIONS = {"118": "강원", "119": "경상", "120": "전라", "121": "제주", "122": "충청"}
TARGET_SR = 16000
# Since 2026-10-14 the source zips hold FLAC instead of WAV (/data/aihub/flac_convert.py).
FLAC = "/data/opt/flac/bin/flac"
# Anonymization (#이름#), noise/filler tags (@...) and unresolved notation make
# poor targets, so utterances containing them are skipped.
BAD_TEXT = re.compile(r"[#@&{}()\[\]/*+=<>]")

# The dataset an utterance comes from is not its speaker's accent (the Chungcheong set
# also records speakers born in Seoul or Daegu), so the `지역:` answer line uses the
# speaker's accent zone: birthplace and principal residence must fall in the same zone.
# Gyeongsang is split into its two main dialect areas.
ZONES = {
    "서울": "서울·경기", "경기": "서울·경기", "인천": "서울·경기",
    "강원": "강원",
    "충남": "충청", "충북": "충청", "대전": "충청", "세종": "충청",
    "부산": "부산·울산·경남", "울산": "부산·울산·경남", "경남": "부산·울산·경남",
    "대구": "대구·경북", "경북": "대구·경북",
    "전남": "전라", "전북": "전라", "광주": "전라",
    "제주": "제주",
}


def zone_of(birthplace, residence):
    zb, zr = ZONES.get(birthplace), ZONES.get(residence)
    return zb if zb and zb == zr else None


def split_of(path):
    if "1.Training" in path:
        return "train"
    if "2.Validation" in path:
        return "valid"
    return None


def find_zips(ds_dir, kind):
    found = []
    for root, _, files in os.walk(ds_dir):
        for f in files:
            if f.endswith(".zip") and kind in root:
                found.append(os.path.join(root, f))
    return sorted(found)


def load_utterances(ds_dir, min_sec, max_sec):
    """Return {split: [utterance dict]} for one dataset."""
    out = defaultdict(list)
    for zpath in find_zips(ds_dir, "라벨링데이터"):
        split = split_of(zpath)
        if split is None:
            continue
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.endswith(".json"):
                    continue
                d = json.loads(z.read(name).decode("utf-8-sig"))
                rec = d["id"]
                speakers = {str(s.get("id")): s for s in d.get("speaker", [])}
                for u in d.get("utterance", []):
                    dia, std = u.get("dialect_form", "").strip(), u.get("standard_form", "").strip()
                    start, end = float(u["start"]), float(u["end"])
                    if not dia or not std or BAD_TEXT.search(dia) or BAD_TEXT.search(std):
                        continue
                    if not (min_sec <= end - start <= max_sec):
                        continue
                    has_dialect = any(e.get("isDialect") for e in u.get("eojeolList", [])) or dia != std
                    s = speakers.get(str(u.get("speaker_id")), {})
                    bp, res = (s.get("birthplace") or "").strip(), (s.get("principal_residence") or "").strip()
                    out[split].append({
                        "id": u["id"], "recording": rec, "start": start, "end": end,
                        "dialect_form": dia, "standard_form": std, "has_dialect": has_dialect,
                        "origin": zone_of(bp, res), "birthplace": bp, "residence": res,
                    })
    return out


def index_recordings(ds_dir):
    """Map recording id -> (zip path, member name) over the source-data zips."""
    index = {}
    for zpath in find_zips(ds_dir, "원천데이터"):
        try:
            with zipfile.ZipFile(zpath) as z:
                names = z.namelist()
        except zipfile.BadZipFile:  # still being merged by the downloader
            continue
        for name in names:
            if name.lower().endswith((".wav", ".flac")):
                index[os.path.splitext(os.path.basename(name))[0]] = (zpath, name)
    return index


def sample(utts, n, dialect_ratio, rng):
    dia = [u for u in utts if u["has_dialect"]]
    plain = [u for u in utts if not u["has_dialect"]]
    n_dia = min(len(dia), round(n * dialect_ratio))
    n_plain = min(len(plain), n - n_dia)
    return rng.sample(dia, n_dia) + rng.sample(plain, n_plain)


def cap_per_recording(utts, k, dialect_ratio, rng):
    """Keep at most k random utterances of each recording, so a test set spreads over many conversations.
    Each recording keeps the dialect share sample() asks for; a random cap would leave mostly plain speech."""
    by_rec = defaultdict(lambda: ([], []))
    for u in utts:
        by_rec[u["recording"]][0 if u["has_dialect"] else 1].append(u)
    out = []
    for dia, plain in by_rec.values():
        n_dia = min(len(dia), round(k * dialect_ratio))
        n_plain = min(len(plain), k - n_dia)
        n_dia = min(len(dia), k - n_plain)
        out += rng.sample(dia, n_dia) + rng.sample(plain, n_plain)
    return out


def read_wav(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        data = z.read(member)
    if member.lower().endswith(".flac"):  # decodes to a plain WAV with the original samples
        data = subprocess.run([FLAC, "-d", "-c", "-s", "-"], input=data, capture_output=True, check=True).stdout
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        frames = w.readframes(n)
    if width != 2:
        raise ValueError(f"{member}: expected 16-bit PCM, got {width * 8}-bit")
    audio = np.frombuffer(frames, dtype="<i2").reshape(-1, ch).astype(np.float32).mean(axis=1) / 32768.0
    return audio, sr


def write_wav(path, audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm.tobytes())


def cut_recording(job):
    """Cut every selected utterance out of one recording. Runs in a worker."""
    torch.set_num_threads(1)
    (zpath, member), utts, out_dir, pad = job
    try:
        audio, sr = read_wav(zpath, member)
    except Exception as e:  # a broken recording should not stop the whole run
        return [], f"{member}: {e}"
    rows = []
    for u in utts:
        seg = audio[max(0, int((u["start"] - pad) * sr)):int((u["end"] + pad) * sr)]
        if len(seg) < sr * 0.5:
            continue
        seg = AF.resample(torch.from_numpy(seg), sr, TARGET_SR).numpy()
        rel = os.path.join(u["split"], u["region"], f"{u['id']}.wav")
        os.makedirs(os.path.join(out_dir, os.path.dirname(rel)), exist_ok=True)
        write_wav(os.path.join(out_dir, rel), seg)
        rows.append({
            "audio": rel, "id": u["id"], "region": u["region"], "split": u["split"],
            "duration": round(len(seg) / TARGET_SR, 3), "has_dialect": u["has_dialect"],
            "dialect_form": u["dialect_form"], "standard_form": u["standard_form"],
            "origin": u.get("origin"), "birthplace": u.get("birthplace"), "residence": u.get("residence"),
        })
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aihub", default="/data/aihub")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
    ap.add_argument("--datasets", default="118,119,120,121,122")
    ap.add_argument("--train-per-region", type=int, default=20000)
    ap.add_argument("--valid-per-region", type=int, default=300)
    ap.add_argument("--dialect-ratio", type=float, default=0.8,
                    help="share of sampled utterances that contain at least one dialect word")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default="",
                    help="comma-separated manifests whose utterance ids must not be sampled again")
    ap.add_argument("--exclude-recordings", default="",
                    help="comma-separated manifests whose recordings (id before the first '.') must not be sampled")
    ap.add_argument("--max-per-recording", type=int, default=0, help="0 = no cap")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="seconds of audio kept before start and after end. In the Jeju set (121) about 30%% of "
                         "clips cut at the label times start or end inside speech (1-5%% elsewhere), but padding "
                         "the Jeju test clips made lora_r64c worse (CER 0.302 -> 0.310 at 0.3 s, 0.399 at 0.6 s; "
                         "runs/eval_pad/report.txt): the extra audio is mostly the other speaker, not cut-off words")
    ap.add_argument("--only", default="",
                    help="comma-separated manifests: cut exactly their utterances again (e.g. with --pad), "
                         "ignoring the per-region counts")
    args = ap.parse_args()
    only = set()
    for path in filter(None, args.only.split(",")):
        only |= {json.loads(line)["id"] for line in open(path, encoding="utf-8")}
    excluded = set()
    for path in filter(None, args.exclude.split(",")):
        excluded |= {json.loads(line)["id"] for line in open(path, encoding="utf-8")}
    if excluded:
        print(f"excluding {len(excluded)} already-used utterances", flush=True)
    excluded_recs = set()
    for path in filter(None, args.exclude_recordings.split(",")):
        excluded_recs |= {json.loads(line)["id"].split(".")[0] for line in open(path, encoding="utf-8")}
    if excluded_recs:
        print(f"excluding {len(excluded_recs)} already-used recordings", flush=True)

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    jobs, stats = [], {}
    for ds in args.datasets.split(","):
        region = REGIONS[ds]
        ds_dir = os.path.join(args.aihub, ds)
        utts = load_utterances(ds_dir, args.min_sec, args.max_sec)
        recs = index_recordings(ds_dir)
        for split, n in (("train", args.train_per_region), ("valid", args.valid_per_region)):
            available = [u for u in utts.get(split, []) if u["recording"] in recs and u["id"] not in excluded
                         and u["recording"] not in excluded_recs]
            if only:
                available = chosen = [u for u in available if u["id"] in only]
            else:
                if args.max_per_recording:
                    available = cap_per_recording(available, args.max_per_recording, args.dialect_ratio, rng)
                chosen = sample(available, n, args.dialect_ratio, rng)
            stats[f"{region}/{split}"] = {"labelled": len(utts.get(split, [])), "with_audio": len(available),
                                          "chosen": len(chosen)}
            by_rec = defaultdict(list)
            for u in chosen:
                by_rec[u["recording"]].append({**u, "split": split, "region": region})
            jobs += [(recs[r], us, args.out, args.pad) for r, us in by_rec.items()]
        print(region, {k: v for k, v in stats.items() if k.startswith(region)}, flush=True)

    rows, errors = defaultdict(list), []
    with Pool(args.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(cut_recording, jobs), 1):
            for row in r:
                rows[row["split"]].append(row)
            if err:
                errors.append(err)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} recordings cut", flush=True)

    for split, rs in rows.items():
        rs.sort(key=lambda x: x["id"])
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for row in rs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        hours = sum(r["duration"] for r in rs) / 3600
        print(f"{split}: {len(rs)} clips, {hours:.1f} h", flush=True)
    with open(os.path.join(args.out, "prepare_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "stats": stats, "errors": errors}, f, ensure_ascii=False, indent=2)
    if errors:
        print(f"{len(errors)} recordings failed; see prepare_stats.json", flush=True)


if __name__ == "__main__":
    main()
