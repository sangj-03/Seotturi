"""Pooled Whisper features for the data splits: features/<split>.npy ([N, 33, 2560] float16), rows in
the order of data/<split>.jsonl.

    python features.py train dev test_plain test_same test_dialect bench

Resumable: progress is kept in features/<split>.progress, and a finished split is skipped.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import augment
from common import DIM, LAYERS, Encoder, load_wav

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FEAT = os.path.join(HERE, "features")


def run(enc, split, bs):
    with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    final = os.path.join(FEAT, f"{split}.npy")
    if os.path.exists(final):
        print(f"{split}: done already", flush=True)
        return
    part, prog = final + ".part", os.path.join(FEAT, f"{split}.progress")
    start = int(open(prog).read()) if os.path.exists(prog) and os.path.exists(part) else 0
    mm = (np.lib.format.open_memmap(part, mode="r+") if start else
          np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(len(rows), LAYERS, DIM)))
    t0, done = time.time(), start
    with ThreadPoolExecutor(8) as pool:
        # rows with "aug" are augmented copies (region4.py): rebuilt from their stored parameters
        read = lambda b: list(pool.map(lambda r: augment.apply(load_wav(r["path"]), r["aug"]) if r.get("aug")
                                       else load_wav(r["path"]), b))
        starts = list(range(start, len(rows), bs))
        nxt = pool.submit(read, rows[starts[0]:starts[0] + bs]) if starts else None
        for k, i in enumerate(starts):
            audios = nxt.result()
            if k + 1 < len(starts):
                nxt = pool.submit(read, rows[starts[k + 1]:starts[k + 1] + bs])
            mm[i:i + len(audios)] = enc.pooled(audios).cpu().numpy()
            done = i + len(audios)
            if k % 50 == 0 or done == len(rows):
                mm.flush()
                with open(prog, "w") as f:
                    f.write(str(done))
                rate = (done - start) / max(1e-6, time.time() - t0)
                print(f"{split}: {done}/{len(rows)} ({rate:.1f} clips/s, {(len(rows) - done) / max(rate, 1e-6) / 60:.0f} min left)", flush=True)
    mm.flush()
    del mm
    with open(os.path.join(FEAT, f"{split}.ids.json"), "w") as f:
        json.dump([r["id"] for r in rows], f)  # train_head.py checks the rows still line up
    os.replace(part, final)
    os.remove(prog)
    print(f"{split}: wrote {final}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="+")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu-share", type=float, default=0.12, help="cap on this process's share of GPU memory")
    args = ap.parse_args()
    os.makedirs(FEAT, exist_ok=True)
    import torch
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)  # the GPU is shared with the app's engine
    enc = Encoder()
    for split in args.splits:
        run(enc, split, args.batch)


if __name__ == "__main__":
    main()
