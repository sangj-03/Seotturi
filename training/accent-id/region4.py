"""Four-region data (user's choice, 2026-10-09): Seoul-Gyeonggi is the standard accent and is not trained;
the label is the AI Hub set a clip was recorded in (강원, 충청, 경상, 제주), not the birthplace (people who
moved speak the dialect of where they live: the Gyeongsang-born speakers of the Jeju set live in Jeju and
use Jeju forms as often as the natives). Speakers born in Seoul-Gyeonggi are left out of every split.

Writes, for assemble_big.py and features.py:
  data/r4_aug.jsonl, data/r4_dev_aug.jsonl   augmented copies (augment.py), features from features.py
  data_r4/<name>/train.jsonl                 the rows of each r4 split, assembled from existing features

    python region4.py
    python features.py r4_aug r4_dev_aug
    for s in train dev test_plain test_same test_dialect bench; do python assemble_big.py --split data_r4/$s --name r4_$s --sources ...; done
    python train_head.py --zones region4 --train r4_train --dev r4_dev --tests r4_test_plain,r4_test_same,r4_test_dialect,r4_bench --out runs/region4
"""

import argparse
import collections
import json
import os

import numpy as np

import augment

HERE = os.path.dirname(os.path.abspath(__file__))
REGIONS = ["강원", "충청", "경상", "제주"]


def rows_of(split):
    with open(os.path.join(HERE, "data", f"{split}.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [{**r, "y": REGIONS.index(r["dataset"])} for r in rows if r["zone"] != "서울·경기"]


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def copies(rows, n, seed, donors):
    """n augmented copies of every row; the mixed-in background always comes from another set."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        for r in rows:
            out.append({**r, "id": f"{r['id']}~a{k}", "src": r["id"], "aug": augment.draw(rng, donors[r["dataset"]])})
    return out


def all_rows(split):
    with open(os.path.join(HERE, "data", f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def std2(args):
    """Standard-vs-regional detector (user's choice after region4 told the two apart poorly, AUC 0.68):
    Seoul-Gyeonggi-born speakers are the standard reference only (y 0), everyone else regional (y 1).
    Reuses region4's regional copies; adds copies of the Seoul clips with the same augmentation."""
    lab = lambda rows, y=None: [{**r, "y": (0 if r["zone"] == "서울·경기" else 1) if y is None else y} for r in rows]
    train = all_rows("train_big")
    donors = {d: [r["path"] for r in train if r["zone"] != "서울·경기" and r["dataset"] != d] for d in REGIONS}
    seoul = [r for r in train if r["zone"] == "서울·경기"]
    std_aug = lab(copies(seoul, args.copies, args.seed + 10, donors))
    write(os.path.join(HERE, "data", "std_aug.jsonl"), std_aug)
    write(os.path.join(HERE, "data_std2", "train", "train.jsonl"), lab(train) + lab(all_rows("r4_aug"), 1) + std_aug)
    dev = all_rows("dev")
    std_dev_aug = lab(copies([r for r in dev if r["zone"] == "서울·경기"], 1, args.seed + 11, donors))
    write(os.path.join(HERE, "data", "std_dev_aug.jsonl"), std_dev_aug)
    write(os.path.join(HERE, "data_std2", "dev", "train.jsonl"), lab(dev) + lab(all_rows("r4_dev_aug"), 1) + std_dev_aug)
    for split in ("test_plain", "test_same", "test_dialect", "bench"):
        write(os.path.join(HERE, "data_std2", split, "train.jsonl"), lab(all_rows(split)))
    print(f"std2: Seoul {len(seoul)} + {len(std_aug)} copies, dev Seoul copies {len(std_dev_aug)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--std2", action="store_true", help="the standard-vs-regional detector's data instead")
    args = ap.parse_args()
    if args.std2:
        return std2(args)
    train = rows_of("train_big")
    donors = {d: [r["path"] for r in train if r["dataset"] != d] for d in REGIONS}
    aug = copies(train, args.copies, args.seed, donors)
    write(os.path.join(HERE, "data", "r4_aug.jsonl"), aug)
    write(os.path.join(HERE, "data_r4", "train", "train.jsonl"), train + aug)
    dev = rows_of("dev")
    dev_aug = copies(dev, 1, args.seed + 1, donors)  # model selection on clean and damaged clips alike
    write(os.path.join(HERE, "data", "r4_dev_aug.jsonl"), dev_aug)
    write(os.path.join(HERE, "data_r4", "dev", "train.jsonl"), dev + dev_aug)
    for split in ("test_plain", "test_same", "test_dialect", "bench"):
        write(os.path.join(HERE, "data_r4", split, "train.jsonl"), rows_of(split))
    print(f"train {len(train)} + {len(aug)} copies, dev {len(dev)} + {len(dev_aug)}",
          dict(collections.Counter(REGIONS[r["y"]] for r in train)))
    print("augment use:", dict(collections.Counter(k for r in aug for k in r["aug"])))


if __name__ == "__main__":
    main()
