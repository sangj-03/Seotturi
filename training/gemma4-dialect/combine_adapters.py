"""Fold a LoRA trained on top of a merged adapter (train.py --merge-adapter) and that adapter into one LoRA
the web engine can serve. The engine switches the adapter off for the plain Gemma coach, so the bottom
adapter cannot stay merged into the base weights. B1·A1·s1 + B2·A2·s2 = [s1·B1 | s2·B2]·[A1; A2]: a rank
r1 + r2 adapter with scaling 1 (lora_alpha = r). modules_to_save come from the top adapter, which trained
them further from the merged copy.

    python combine_adapters.py runs/lora_v1/final runs/lora_r64/final runs/lora_r64c/final
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bottom", help="the adapter that was merged (train.py --merge-adapter)")
    ap.add_argument("top", help="the adapter trained on top of it")
    ap.add_argument("out")
    args = ap.parse_args()
    dirs = (args.bottom, args.top)
    cfg = [json.load(open(os.path.join(d, "adapter_config.json"), encoding="utf-8")) for d in dirs]
    for k in ("target_modules", "modules_to_save"):
        assert sorted(cfg[0][k] or []) == sorted(cfg[1][k] or []), k
    for c in cfg:
        assert not (c.get("use_rslora") or c.get("use_dora") or c.get("rank_pattern") or c.get("alpha_pattern"))
    w = [load_file(os.path.join(d, "adapter_model.safetensors")) for d in dirs]
    assert w[0].keys() == w[1].keys()
    scale = [c["lora_alpha"] / c["r"] for c in cfg]
    out = {}
    for k in w[1]:
        if ".lora_A." in k:
            out[k] = torch.cat([w[0][k], w[1][k]], 0).contiguous()
        elif ".lora_B." in k:
            out[k] = torch.cat([w[0][k] * scale[0], w[1][k] * scale[1]], 1).contiguous()
        else:
            out[k] = w[1][k]
    r = cfg[0]["r"] + cfg[1]["r"]
    os.makedirs(args.out, exist_ok=True)
    save_file(out, os.path.join(args.out, "adapter_model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(args.out, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg[1], "r": r, "lora_alpha": r}, f, ensure_ascii=False, indent=2)
    # the web engine takes a run's final/ as complete once train_args.json is there
    with open(os.path.join(args.out, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump({"combined_from": list(dirs), "r": r, "lora_alpha": r}, f, ensure_ascii=False, indent=2)
    print(f"{args.out}: rank {r} from {dirs[0]} (r {cfg[0]['r']}) + {dirs[1]} (r {cfg[1]['r']})")


if __name__ == "__main__":
    main()
