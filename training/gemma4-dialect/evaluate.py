"""Score the base model or a LoRA adapter on held-out dialect speech.

    python evaluate.py --out runs/base_eval
    python evaluate.py --adapter runs/lora_v1/final --out runs/lora_v1_eval
    python evaluate.py --adapters v3_500=runs/lora_v3/checkpoint-500,v3_750=... --out runs/eval_full
        (several adapters on one loaded base model: one load instead of one per adapter; loading the 12B
        model briefly took the shared memory down to 3 GB beside the site's engine)
    python evaluate.py --guard ...   (decode as the web engine serves: see GUARD below)

Reports character error rate (spaces and punctuation ignored) for the dialect
transcript and for the standard-Korean conversion, overall and per region, and
writes every prediction to predictions.jsonl. With `--format region` it also scores
the `지역:` line against the speaker's accent zone (manifest field `origin`).
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, StoppingCriteria, StoppingCriteriaList

from common import (MODEL_PATH, cer, generation_batch, load_audio, parse_answer, parse_region, prompt_inputs,
                    read_manifest)

# GUARD: the web engine's decoding (seotturi-web/server/engine.py, DECODE "guard1"), kept the same here so
# --guard scores what the site serves. The answer starts as trained ("방언:"/"지역:"), the token budget grows
# with the clip, a line that repeats a unit of up to 20 characters 8+ times stops generation and is cut after
# two copies, and a cut or missing 표준어 line is written once more after the fixed 방언 line.
ANSWER_START = {"basic": "방언:", "region": "지역:"}
LOOP = re.compile(r"(\S.{0,19}?)(?:\s*\1){7,}")


def answer_budget(seconds):
    return int(min(512, max(64, 32 + 16 * seconds)))


def unloop(line):
    """The line cut after the first two copies of a runaway repetition, and whether it was cut."""
    m = LOOP.search(line)
    if not m:
        return line, False
    two = re.compile(re.escape(m.group(1)) + r"\s*" + re.escape(m.group(1))).match(line, m.start())
    return line[:two.end()].strip(), True


def generate_guarded(model, processor, enc, fmt, seconds, prefix=None):
    """Greedy answers continuing `prefix` (default ANSWER_START[fmt]), returned with it; a sequence whose last
    line runs away stops there."""
    tok = processor.tokenizer
    prefix = ANSWER_START[fmt] if prefix is None else prefix
    enc = {k: v.to("cuda") for k, v in enc.items()}
    ids = torch.tensor(tok(prefix, add_special_tokens=False)["input_ids"], device="cuda")
    ids = ids.expand(enc["input_ids"].shape[0], -1)
    enc["input_ids"] = torch.cat([enc["input_ids"], ids], 1)
    enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(ids)], 1)
    enc["mm_token_type_ids"] = torch.cat([enc["mm_token_type_ids"], torch.zeros_like(ids)], 1)
    n = enc["input_ids"].shape[1]

    class StopLoops(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            texts = tok.batch_decode(input_ids[:, n:], skip_special_tokens=True)
            return torch.tensor([bool(LOOP.search(t.rsplit("\n", 1)[-1])) for t in texts],
                                dtype=torch.bool, device=input_ids.device)

    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=answer_budget(seconds), do_sample=False,
                             stopping_criteria=StoppingCriteriaList([StopLoops()]))
    return [prefix + processor.decode(s[n:], skip_special_tokens=True) for s in out]


def finish_guarded(model, processor, text, fmt, row):
    """(answer, repaired) as the engine's _finish."""
    dia, std = parse_answer(text)
    dia, cut = unloop(dia)
    std, cut_std = unloop(std)
    if not (cut or cut_std or (dia and not std)):
        return text, False
    head = (f"지역: {parse_region(text)}\n" if fmt == "region" else "") + f"방언: {dia}\n"
    if dia and (cut or not std):
        enc = prompt_inputs(processor, load_audio(row["audio_path"]), fmt)
        again = generate_guarded(model, processor, enc, fmt, row["duration"], prefix=head + "표준어:")[0]
        std = unloop(parse_answer(again)[1])[0]
    return head + f"표준어: {std}", True


def zone_scores(ps):
    """Accuracy of the `지역:` line against the speaker's zone, overall, per zone and as a confusion table."""
    ps = [p for p in ps if p["origin"]]
    if not ps:
        return None
    confusion = defaultdict(lambda: defaultdict(int))
    for p in ps:
        confusion[p["origin"]][p["pred_region"] or "-"] += 1
    return {"n": len(ps),
            "accuracy": round(sum(p["pred_region"] == p["origin"] for p in ps) / len(ps), 4),
            "per_zone": {z: round(c[z] / sum(c.values()), 4) for z, c in sorted(confusion.items())},
            "confusion": {z: dict(sorted(c.items(), key=lambda x: -x[1])) for z, c in sorted(confusion.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/valid.jsonl")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapters", default=None, help="name=path,... evaluated in turn into <out>/<name>/")
    ap.add_argument("--format", choices=["basic", "region"], default="basic",
                    help="the answer format the adapter was trained with (train.py --format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-region", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--merge-adapter", default=None,
                    help="merge this adapter into the base weights first (for adapters trained with train.py --merge-adapter)")
    ap.add_argument("--guard", action="store_true", help="decode as the web engine does (see GUARD)")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    if args.merge_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.merge_adapter).merge_and_unload()
    rows = read_manifest(args.manifest, limit_per_region=args.per_region)
    if not args.adapters:
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        return run(model, processor, rows, args, args.adapter, args.out)
    from peft import PeftModel
    for k, spec in enumerate(args.adapters.split(",")):
        name, path = spec.split("=", 1)
        if k == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
        model.set_adapter(name)
        model.eval()
        run(model, processor, rows, args, path, os.path.join(args.out, name))


def run(model, processor, rows, args, adapter, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0, preds = time.time(), []
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            if args.guard:
                texts = generate_guarded(model, processor, generation_batch(processor, chunk, args.format),
                                         args.format, max(r["duration"] for r in chunk))
                done = [finish_guarded(model, processor, t, args.format, r) for t, r in zip(texts, chunk)]
            else:
                batch = {k: v.to("cuda") for k, v in generation_batch(processor, chunk, args.format).items()}
                with torch.no_grad():
                    out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
                n = batch["input_ids"].shape[1]
                done = [(processor.decode(seq[n:], skip_special_tokens=True), False) for seq in out]
            for r, (text, repaired) in zip(chunk, done):
                dia, std = parse_answer(text)
                p = {"id": r["id"], "region": r["region"], "has_dialect": r["has_dialect"],
                     "origin": r.get("origin"),
                     "pred_region": parse_region(text) if args.format == "region" else None,
                     "ref_dialect": r["dialect_form"], "ref_standard": r["standard_form"],
                     "raw": text, "dialect": dia, "standard": std, "repaired": repaired,
                     "cer_dialect": cer(r["dialect_form"], dia), "cer_standard": cer(r["standard_form"], std)}
                preds.append(p)
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"{len(preds)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    def summarize(ps):
        if not ps:  # e.g. no utterance with dialect words in the intonation set's plain subsets
            return {"n": 0}
        return {"n": len(ps),
                "cer_dialect": round(sum(p["cer_dialect"] for p in ps) / len(ps), 4),
                "cer_standard": round(sum(p["cer_standard"] for p in ps) / len(ps), 4),
                "format_ok": round(sum(bool(p["dialect"] and p["standard"]) for p in ps) / len(ps), 4),
                "repaired": sum(p["repaired"] for p in ps)}

    by_region = defaultdict(list)
    for p in preds:
        by_region[p["region"]].append(p)
    summary = {"adapter": adapter, "overall": summarize(preds),
               "dialect_utterances": summarize([p for p in preds if p["has_dialect"]]),
               "regions": {k: summarize(v) for k, v in sorted(by_region.items())},
               "seconds": round(time.time() - t0)}
    if args.format == "region":
        summary["zone"] = zone_scores(preds)
        # without dialect words the zone can only come from accent and intonation
        summary["zone_plain_speech"] = zone_scores([p for p in preds if not p["has_dialect"]])
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
