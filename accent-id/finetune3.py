"""Phase 3: fine-tune the Whisper encoder (LoRA on layers 17-32) starting from a trained phase 1 head (LP-FT),
reading all 33 hidden states, with a speaker-adversarial branch, on the larger data_x2 split.

    python finetune3.py --init runs/phase1_x2 --data data_x2 --out runs/phase3

What changes against phase 2 (finetune.py), each for a reason found there:
  pooling and head  phase 1's: mean and std over the speech frames of every one of the 33 hidden states,
                    normalised and mixed by the phase 1 head, which is loaded as it is (phase 2 read only
                    layers 16-32 although phase 1 puts 55 % of its layer weight below 17, and started from a
                    random head; Kumar et al. 2022: fine-tuning from a random head distorts the features).
  LP then FT        the first --lp-steps train the head alone (the encoder runs on 5/10/15 s windows here,
                    phase 1 features used 15 s: dev macro-F1 of the loaded head 0.849 -> 0.813), then LoRA joins
                    at a lower learning rate.
  speaker-adversarial  a gradient-reversal branch on the head's 64-wide bottleneck predicts the training
                    person, so that features which only name the voice are pushed out (dev scores well above
                    the tests showed voices being learned); its weight ramps up during the first part of FT.
  data              data_x2: 198,771 clips, 39,295 persons; dev and tests as before.
Augmentation, windows, LoRA placement, class weights and label smoothing are phase 2's.

Writes runs/<out>/model.pt (LoRA + head; Tuned3 serves it), metrics.json, test_logits.npz.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

from common import FPS, LAYERS, WHISPER, ZONES, Head
from finetune import WINDOWS, Batches, LoRALinear, predict, read, window_of  # noqa: F401
from train_head import fit_temperature, report

HERE = os.path.dirname(os.path.abspath(__file__))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Model3(nn.Module):
    def __init__(self, encoder, head, n_persons=0, first_tuned=16, r=16):
        super().__init__()
        self.enc, self.head, self.first_tuned = encoder, head, first_tuned
        for p in self.enc.parameters():
            p.requires_grad_(False)
        for layer in self.enc.layers[first_tuned:]:
            at = layer.self_attn
            at.q_proj, at.k_proj, at.v_proj, at.out_proj = (LoRALinear(at.q_proj, r), LoRALinear(at.k_proj, r),
                                                             LoRALinear(at.v_proj, r), LoRALinear(at.out_proj, r))
            layer.fc1, layer.fc2 = LoRALinear(layer.fc1, r), LoRALinear(layer.fc2, r)
        full = self.enc.embed_positions.weight.data
        self.positions = nn.ModuleDict({str(w): nn.Embedding.from_pretrained(full[:w * FPS].clone(), freeze=True)
                                        for w in WINDOWS})
        self.adv = nn.Linear(head.net[5].out_features, n_persons) if n_persons else None
        self.lam = 0.0

    def set_window(self, seconds):
        self.enc.embed_positions = self.positions[str(seconds)]
        self.enc.config.max_source_positions = seconds * FPS

    def pooled(self, feats, mask):
        """[B, 33, 2560]: mean and std over the speech frames of every hidden state (common.Encoder.pooled)."""
        self.set_window(feats.shape[-1] // (2 * FPS))
        hs = torch.stack(self.enc(feats, output_hidden_states=True).hidden_states, 1).float()  # [B, 33, T, D]
        m = mask[:, None, :, None]
        n = mask.sum(1)[:, None, None]
        mu = (hs * m).sum(2) / n
        sd = (((hs - mu[:, :, None]) ** 2 * m).sum(2) / (n - 1).clamp(min=1)).clamp(min=1e-8).sqrt()
        return torch.cat([mu, sd], -1)

    def forward(self, feats, mask, with_adv=False):
        h = self.head
        x = (self.pooled(feats, mask) - h.mu) / h.sd
        x = torch.einsum("bld,l->bd", x, torch.softmax(h.mix, 0))
        z = h.net[:7](x)  # the 64-wide bottleneck
        logits = h.net[7](z) / h.temperature
        if with_adv and self.adv is not None:
            return logits, self.adv(GradReverse.apply(z, self.lam))
        return logits

    def lora(self):
        return [p for n, p in self.enc.named_parameters() if p.requires_grad]


def load_head(run):
    ck = torch.load(os.path.join(HERE, run, "head.pt"), map_location="cpu")
    head = Head(**ck["config"])
    head.load_state_dict(ck["state"])
    head.temperature.fill_(1.0)  # fitted again at the end
    return head, ck["config"]


def build(head, n_persons=0, first_tuned=16, r=16):
    from transformers import WhisperModel
    enc = WhisperModel.from_pretrained(WHISPER, dtype=torch.bfloat16).encoder
    return Model3(enc.to(torch.bfloat16), head, n_persons=n_persons, first_tuned=first_tuned, r=r)


class Tuned3:
    """A phase 3 model for serving: audio clips -> zone probabilities (common.load_classifier)."""

    def __init__(self, path, device="cuda"):
        from transformers import WhisperFeatureExtractor
        ck = torch.load(path, map_location="cpu")
        a = ck["args"]
        self.model = build(Head(**ck["head_config"]), first_tuned=a["first_tuned"], r=a["rank"])
        self.model.load_state_dict(ck["state"], strict=False)
        self.model.to(device).eval()
        self.model.head.float()
        self.fes = {w: WhisperFeatureExtractor.from_pretrained(WHISPER, chunk_length=w) for w in WINDOWS}
        self.device = device
        self.info = ck.get("info", {})

    @torch.no_grad()
    def probs(self, audios):
        from common import SR, WINDOW
        out = []
        for a in audios:
            a = a[-WINDOW * SR:]
            w = window_of(len(a) / SR)
            x = self.fes[w]([a], sampling_rate=SR, return_tensors="pt").input_features
            mask = torch.zeros(1, x.shape[-1] // 2)
            mask[0, :max(2, min(mask.shape[1], math.ceil(len(a) / SR * FPS)))] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(x.to(self.device, torch.bfloat16), mask.to(self.device)).float()
            out.append(torch.softmax(logits, -1)[0].cpu().numpy())
        return np.stack(out)


def saved_state(model):
    """LoRA weights, the head (with its buffers) and the window tables are what differs from Whisper."""
    keep = {}
    for k, v in model.state_dict().items():
        if k.startswith("head.") or ".a.weight" in k or ".b.weight" in k:
            keep[k] = v.detach().to("cpu", copy=True)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/phase1_x2", help="phase 1 run whose head.pt starts the head")
    ap.add_argument("--data", default="data_x2")
    ap.add_argument("--out", default="runs/phase3")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lp-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA")
    ap.add_argument("--head-lr", type=float, default=3e-5)
    ap.add_argument("--lp-lr", type=float, default=3e-4, help="head during LP")
    ap.add_argument("--adv", type=float, default=0.1, help="final gradient-reversal weight (0 = off)")
    ap.add_argument("--adv-ramp", type=float, default=0.3, help="share of FT steps over which it ramps up")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--first-tuned", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--gpu-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--dev-limit", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    data = os.path.join(HERE, args.data)
    train, dev = read("train", data), read("dev", data)
    if args.dev_limit:
        dev = dev[:args.dev_limit]
    persons = {p: i for i, p in enumerate(sorted({r["person"] for r in train}))}
    head, head_config = load_head(args.init)
    model = build(head, n_persons=len(persons) if args.adv > 0 else 0, first_tuned=args.first_tuned, r=args.rank).cuda()
    model.head.float()
    from transformers import WhisperFeatureExtractor
    fes = {w: WhisperFeatureExtractor.from_pretrained(WHISPER, chunk_length=w) for w in WINDOWS}
    steps_per_epoch = math.ceil(len(train) / args.batch)
    steps = args.max_steps or int(args.epochs * steps_per_epoch)
    print(f"train {len(train)} clips / {len(persons)} persons, dev {len(dev)}; {steps} steps "
          f"(LP {args.lp_steps}); LoRA {sum(p.numel() for p in model.lora()):,}, head "
          f"{sum(p.numel() for p in model.head.parameters()):,}", flush=True)
    dev_y = np.array([r["y"] for r in dev])
    m0 = report(dev, dev_y, predict(model, fes, dev))
    print(f"  dev before training: macro-F1 {m0['macro_f1']} acc {m0['accuracy']} AUC {m0['standard_vs_regional_auc']}", flush=True)

    counts = np.bincount([r["y"] for r in train], minlength=len(ZONES))
    weight = torch.tensor(len(train) / (len(ZONES) * np.maximum(counts, 1)), dtype=torch.float32).cuda()
    lossf = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.smoothing)
    advf = nn.CrossEntropyLoss()
    groups = [{"params": model.lora(), "lr": args.lr, "weight_decay": 0.01, "name": "lora"},
              {"params": list(model.head.parameters()), "lr": args.head_lr, "weight_decay": 0.05, "name": "head"}]
    if model.adv is not None:
        groups.append({"params": list(model.adv.parameters()), "lr": 1e-3, "weight_decay": 0.0, "name": "adv"})
    opt = torch.optim.AdamW(groups)
    base = {g["name"]: g["lr"] for g in opt.param_groups}
    ft_steps = max(1, steps - args.lp_steps)

    def set_lrs(step):
        for g in opt.param_groups:
            if step < args.lp_steps:  # LP: the head alone (and the adversary, which only learns)
                g["lr"] = {"lora": 0.0, "head": args.lp_lr, "adv": base.get("adv", 0.0)}[g["name"]]
            else:
                t = (step - args.lp_steps) / ft_steps
                warm = min(1.0, (step - args.lp_steps + 1) / 300)
                g["lr"] = base[g["name"]] * warm * 0.5 * (1 + math.cos(math.pi * t))
        t = (step - args.lp_steps) / ft_steps
        model.lam = 0.0 if step < args.lp_steps else args.adv * min(1.0, t / max(args.adv_ramp, 1e-6))

    best, best_state, history, step, t0, run = -1.0, None, [], 0, time.time(), [0.0, 0.0]
    epoch = 0
    while step < steps:
        for idx, x, mask, y in Batches(fes, train, args.batch, True, args.seed + epoch):
            model.train()
            set_lrs(step)
            pid = torch.tensor([persons[train[i]["person"]] for i in idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = model(x.cuda().to(torch.bfloat16), mask.cuda(), with_adv=model.adv is not None)
            logits, adv_logits = res if isinstance(res, tuple) else (res, None)
            loss = lossf(logits.float(), y.cuda())
            adv_loss = advf(adv_logits.float(), pid) if adv_logits is not None else torch.zeros((), device="cuda")
            opt.zero_grad(set_to_none=True)
            (loss + adv_loss).backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            step += 1
            run[0] += loss.item(); run[1] += adv_loss.item()
            if step % 100 == 0:
                rate = step * args.batch / (time.time() - t0)
                print(f"step {step}/{steps} loss {run[0] / 100:.4f} adv {run[1] / 100:.3f} lam {model.lam:.3f} "
                      f"({rate:.1f} clips/s, {(steps - step) * args.batch / rate / 3600:.1f} h left)", flush=True)
                run = [0.0, 0.0]
            if step % args.eval_every == 0 or step == steps or step == args.lp_steps:
                m = report(dev, dev_y, predict(model, fes, dev))
                history.append({"step": step, "dev_macro_f1": m["macro_f1"], "dev_auc": m["standard_vs_regional_auc"]})
                print(f"  dev @ {step}: macro-F1 {m['macro_f1']} acc {m['accuracy']} AUC {m['standard_vs_regional_auc']}", flush=True)
                if m["macro_f1"] > best:
                    best, best_state = m["macro_f1"], saved_state(model)
                    torch.save({"args": vars(args), "head_config": head_config, "state": best_state, "step": step,
                                "dev_macro_f1": best}, os.path.join(out, "model.pt"))
            if step >= steps:
                break
        epoch += 1
    model.load_state_dict(best_state, strict=False)
    dev_logits = predict(model, fes, dev)
    model.head.temperature.fill_(fit_temperature(dev_logits, dev_y))
    metrics = {"dev": report(dev, dev_y, predict(model, fes, dev)), "temperature": float(model.head.temperature),
               "history": history, "dev_before": m0["macro_f1"],
               "layer_weights": [round(float(w), 4) for w in torch.softmax(model.head.mix.detach().cpu(), 0)]}
    test_logits = {}
    for split in ("test_plain", "test_same", "test_dialect", "bench"):
        rows = read(split, data)
        test_logits[split] = predict(model, fes, rows)
        metrics[split] = report(rows, np.array([r["y"] for r in rows]), test_logits[split])
        mm = metrics[split]
        print(f"{split}: macro-F1 {mm['macro_f1']} acc {mm['accuracy']} AUC {mm['standard_vs_regional_auc']} "
              f"std→regional {mm['standard_called_regional']} coarse {mm['coarse_accuracy']} persons {mm['persons']}", flush=True)
    best_state = saved_state(model)
    torch.save({"args": vars(args), "head_config": head_config, "state": best_state,
                "info": {"zones": ZONES, "test_plain_macro_f1": metrics["test_plain"]["macro_f1"],
                         "bench_macro_f1": metrics["bench"]["macro_f1"]}}, os.path.join(out, "model.pt"))
    np.savez(os.path.join(out, "test_logits.npz"), **{k: v.numpy() for k, v in test_logits.items()})
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
