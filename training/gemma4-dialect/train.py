"""LoRA fine-tuning of Gemma 4 12B on dialect speech -> (dialect transcript, standard Korean).

Pilot (measure speed and memory):  python train.py --out runs/pilot --max-steps 100
Full run:                          python train.py --out runs/lora_v1 --epochs 1
"""

import argparse
import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainerCallback,
                          TrainingArguments)

from common import MODEL_PATH, TrainCollator, read_manifest


class CurrentSteps(TrainerCallback):
    """On --resume the Trainer takes save/eval/logging steps from the checkpoint's trainer_state.json, so a
    changed --save-steps was ignored (2026-10-07: 250 asked, 1000 used). Use this run's settings instead."""

    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps, state.eval_steps, state.logging_steps = args.save_steps, args.eval_steps, args.logging_steps


class SpeedCallback(TrainerCallback):
    """Write seconds/step and peak memory so a pilot run can size the full run."""

    def __init__(self, out):
        self.out, self.t0, self.first = out, None, None

    def on_step_begin(self, args, state, control, **kw):
        # skip 5 warm-up steps, counted from wherever the run starts (a resumed run starts at its checkpoint;
        # timing from global step 5 only never started on a resume, leaving speed.json stale since 2026-10-06)
        if self.first is None:
            self.first = state.global_step
        if self.t0 is None and state.global_step >= self.first + 5:
            self.t0, self.s0 = time.time(), state.global_step

    def on_log(self, args, state, control, logs=None, **kw):
        if self.t0 and state.global_step > self.s0:
            sps = (time.time() - self.t0) / (state.global_step - self.s0)
            info = {"step": state.global_step, "sec_per_step": round(sps, 2),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                    "eta_hours": round(sps * (state.max_steps - state.global_step) / 3600, 2)}
            with open(os.path.join(self.out, "speed.json"), "w") as f:
                json.dump(info, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl", help="one or more comma-separated manifests")
    ap.add_argument("--valid", default="data/valid.jsonl")
    ap.add_argument("--format", choices=["basic", "region"], default="basic",
                    help="region: answer starts with a `지역:` line (the speaker's accent zone, field `origin`)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    # 2 x 8 was the fastest split of 16 utterances per step on the GB10:
    # 10.6 s/step vs 11.6 (1 x 16), 12.0 (4 x 4) and 18.6 (8 x 2).
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--train-audio-projection", action=argparse.BooleanOptionalAction, default=True,
                    help="also fully train the small linear layer that projects audio into the LLM")
    ap.add_argument("--eval-per-region", type=int, default=40, help="validation-loss subset size")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --out")
    ap.add_argument("--init-adapter", default=None,
                    help="continue training an existing adapter (e.g. the Gangwon stage) instead of a fresh LoRA")
    ap.add_argument("--merge-adapter", default=None,
                    help="merge this adapter into the base weights first, then train a fresh LoRA on top (more capacity)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # The GPU shares the machine's memory with the app's engine and everything else. With batches of varying
    # audio length, PyTorch's cache kept reserving new blocks (44 GB reserved for ~31 GB in use) until the
    # kernel killed processes; TRAIN_GPU_MEM_GB caps it, so the cache is emptied and reused instead.
    cap = float(os.environ.get("TRAIN_GPU_MEM_GB", "0"))
    if cap:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1e9 / total))
        print(f"GPU memory cap {cap:.0f} GB of {total / 1e9:.0f} GB", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
    if args.merge_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.merge_adapter).merge_and_unload()
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_audio.embedding_projection"] if args.train_audio_projection else None,
    )
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    require = "origin" if args.format == "region" else None
    train_rows = read_manifest(args.train, require=require)
    eval_rows = (read_manifest(args.valid, limit_per_region=args.eval_per_region, require=require)
                 if os.path.exists(args.valid) else None)
    print(f"train {len(train_rows)} rows, eval {len(eval_rows) if eval_rows else 0} rows", flush=True)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.workers,
        remove_unused_columns=False,
        report_to="none",
        seed=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_rows, eval_dataset=eval_rows,
                      data_collator=TrainCollator(processor, args.format), callbacks=[CurrentSteps(), SpeedCallback(args.out)])
    trainer.train(resume_from_checkpoint=args.resume or None)
    final = os.path.join(args.out, "final")
    model.save_pretrained(final)
    json.dump(vars(args), open(os.path.join(final, "train_args.json"), "w"), ensure_ascii=False, indent=2)
    print("saved adapter to", final, flush=True)


if __name__ == "__main__":
    main()
