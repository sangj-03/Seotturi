"""Features of data_big/train.jsonl (prepare.py split --k 1000 --seoul-cap 60 --cap 36 --out data_big): rows found
in features/train.npy are copied (recomputing them gives the same float16 values), the rest come from
features/train_extra.npy (features.py train_extra). Writes data/train_big.jsonl, features/train_big.npy, .ids.json.

    python assemble_big.py && python train_head.py --train train_big --out runs/phase1_big
    python assemble_big.py --split data_x2 --name train_x2 --sources train_big,train_extra2     # expand.py's split
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rows_of(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="data_big", help="folder whose train.jsonl is assembled")
ap.add_argument("--name", default="train_big", help="output: data/<name>.jsonl, features/<name>.npy")
ap.add_argument("--sources", default="train,train_extra", help="feature sets to take rows from")
args = ap.parse_args()
big = rows_of(os.path.join(HERE, args.split, "train.jsonl"))
src = {}
for name in args.sources.split(","):
    ids = json.load(open(os.path.join(HERE, "features", f"{name}.ids.json")))
    X = np.load(os.path.join(HERE, "features", f"{name}.npy"), mmap_mode="r")
    for i, cid in enumerate(ids):
        src.setdefault(cid, (X, i))
missing = [r["id"] for r in big if r["id"] not in src]
assert not missing, f"{len(missing)} clips have no features, e.g. {missing[:3]}"
X0 = next(iter(src.values()))[0]
out = np.lib.format.open_memmap(os.path.join(HERE, "features", f"{args.name}.npy.part"), mode="w+", dtype=np.float16,
                                shape=(len(big),) + X0.shape[1:])
for i, r in enumerate(big):
    X, j = src[r["id"]]
    out[i] = X[j]
out.flush()
del out
os.replace(os.path.join(HERE, "features", f"{args.name}.npy.part"), os.path.join(HERE, "features", f"{args.name}.npy"))
with open(os.path.join(HERE, "data", f"{args.name}.jsonl"), "w", encoding="utf-8") as f:
    for r in big:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(HERE, "features", f"{args.name}.ids.json"), "w") as f:
    json.dump([r["id"] for r in big], f)
print(f"{args.name}: {len(big)} clips")
