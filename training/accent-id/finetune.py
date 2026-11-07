"""Phase 2: fine-tune the upper half of the Whisper encoder (LoRA) with attentive pooling, on raw audio
with augmentation, and score it on the same test sets as phase 1.

    python finetune.py --out runs/phase2

Model: Whisper-large-v3 encoder on a 5, 10 or 15 s window, the shortest that holds the clip (its position
embeddings are fixed sinusoids, so a shorter window is the first rows of the table; plain clips average 4.5 s,
so this halves the compute against a fixed 15 s). Layers 1-16 stay frozen; layers 17-32 get LoRA (rank 16 on
the attention and MLP projections). A learned mix of the upper hidden states feeds attentive statistics
pooling (weighted mean and std over the speech frames, weights from a small attention net) -> 256 -> 64 ->
6 zones. The encoder runs in bf16, the head in fp32.

Augmentation (per clip, on the fly): speed 0.9-1.1 (resampling, pitch moves with it), gain -6..+6 dB,
band limit (telephone-like 300-3400 Hz or laptop-like low-pass), additive noise at 5-30 dB SNR
(white/pink/brown), a short synthetic room reverb. Browser microphones are the target condition.

Selection: macro-F1 on the dev persons after every evaluation; the best state is kept. Writes
runs/<out>/model.pt (LoRA weights + head; common.load_finetuned rebuilds it), metrics.json.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn

from common import SR, STANDARD, WHISPER, WINDOW, ZONES, load_wav
from train_head import COARSE, f1s, fit_temperature, report  # noqa: F401  (same scoring as phase 1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FPS = 50
WINDOWS = (5, 10, 15)


def window_of(seconds):
    return next((w for w in WINDOWS if seconds <= w), WINDOWS[-1])


def read(split, data=DATA):
    with open(os.path.join(data, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, r, bias=False)
        self.b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.b(self.a(self.drop(x))) * self.scale


class AttentiveStats(nn.Module):
    """Weighted mean and std over frames; the weights come from a small attention net (Okabe et al. 2018)."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h, mask):  # h [B, T, D], mask [B, T] (1 = speech frame)
        w = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e4)
        w = torch.softmax(w, -1).unsqueeze(-1)
        mu = (w * h).sum(1)
        sd = ((w * (h - mu.unsqueeze(1)) ** 2).sum(1).clamp(min=1e-6)).sqrt()
        return torch.cat([mu, sd], -1)


class AccentModel(nn.Module):
    def __init__(self, encoder, first_tuned=16, r=16, hidden=256, bottleneck=64, dropout=0.3):
        super().__init__()
        self.enc = encoder
        self.first_tuned = first_tuned
        for p in self.enc.parameters():
            p.requires_grad_(False)
        for layer in self.enc.layers[first_tuned:]:
            at = layer.self_attn
            at.q_proj, at.k_proj, at.v_proj, at.out_proj = (LoRALinear(at.q_proj, r), LoRALinear(at.k_proj, r),
                                                             LoRALinear(at.v_proj, r), LoRALinear(at.out_proj, r))
            layer.fc1, layer.fc2 = LoRALinear(layer.fc1, r), LoRALinear(layer.fc2, r)
        n_mix = len(self.enc.layers) - first_tuned + 1  # hidden states after layers first_tuned..32
        d = self.enc.config.d_model
        self.mix = nn.Parameter(torch.zeros(n_mix))
        self.norm = nn.LayerNorm(d)
        self.pool = AttentiveStats(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, bottleneck), nn.GELU(), nn.Linear(bottleneck, len(ZONES)))
        self.register_buffer("temperature", torch.ones(()))
        full = self.enc.embed_positions.weight.data
        self.positions = nn.ModuleDict({str(w): nn.Embedding.from_pretrained(full[:w * FPS].clone(), freeze=True)
                                        for w in WINDOWS})

    def set_window(self, seconds):
        self.enc.embed_positions = self.positions[str(seconds)]
        self.enc.config.max_source_positions = seconds * FPS

    def forward(self, feats, mask):
        self.set_window(feats.shape[-1] // (2 * FPS))
        out = self.enc(feats, output_hidden_states=True)
        hs = torch.stack(out.hidden_states[self.first_tuned:], 0)  # [n_mix, B, T, D]
        h = torch.einsum("lbtd,l->btd", hs.float(), torch.softmax(self.mix, 0))
        return self.head(self.pool(self.norm(h), mask)) / self.temperature

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


def build(r=16, first_tuned=16, **head):
    from transformers import WhisperModel
    enc = WhisperModel.from_pretrained(WHISPER, dtype=torch.bfloat16).encoder
    return AccentModel(enc.to(torch.bfloat16), first_tuned=first_tuned, r=r, **head)


# ---------------------------------------------------------------------------------------------- audio
def colored_noise(n, kind, rng):
    white = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
    spec /= np.sqrt(f) if kind == "pink" else f
    x = np.fft.irfft(spec, n).astype(np.float32)
    return x / (x.std() + 1e-8)


def augment(a, rng):
    if rng.random() < 0.6:  # speed (and pitch) 0.9-1.1
        s = rng.uniform(0.9, 1.1)
        idx = np.arange(0, len(a) - 1, s)
        a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    if rng.random() < 0.4:  # band limit
        spec = np.fft.rfft(a)
        f = np.fft.rfftfreq(len(a), 1 / SR)
        if rng.random() < 0.5:
            spec[(f < 300) | (f > 3400)] *= 0.05
        else:
            spec[f > rng.uniform(4000, 7000)] *= 0.05
        a = np.fft.irfft(spec, len(a)).astype(np.float32)
    if rng.random() < 0.3:  # small room
        t = np.arange(int(SR * rng.uniform(0.15, 0.4))) / SR
        ir = rng.standard_normal(len(t)).astype(np.float32) * np.exp(-t / rng.uniform(0.03, 0.12)).astype(np.float32)
        ir[0] = 1.0
        a = np.convolve(a, ir / np.abs(ir).sum() * 4)[:len(a)].astype(np.float32)
    if rng.random() < 0.5:  # noise
        noise = colored_noise(len(a), rng.choice(["white", "pink", "brown"]), rng)
        snr = rng.uniform(5, 30)
        a = a + noise * (a.std() + 1e-6) / (10 ** (snr / 20))
    a = a * 10 ** (rng.uniform(-6, 6) / 20)
    return np.clip(a, -1, 1).astype(np.float32)


class Batches:
    """Batches of clips that share a window length, read, augmented and featurised in worker threads a
    few batches ahead. Yields (row indices, features, frame mask, labels)."""

    def __init__(self, fes, rows, bs, train, seed):
        self.fes, self.rows, self.bs, self.train, self.seed = fes, rows, bs, train, seed

    def make(self, idx, rng):
        w = window_of(max(min(self.rows[i]["duration"], WINDOW) for i in idx))
        audios = []
        for i in idx:
            a = load_wav(self.rows[i]["path"])[-WINDOW * SR:]
            if self.train:
                a = augment(a, rng)
            audios.append(a[-w * SR:])
        x = self.fes[w](audios, sampling_rate=SR, return_tensors="pt").input_features
        n = x.shape[-1] // 2
        mask = torch.zeros(len(audios), n)
        for k, a in enumerate(audios):
            mask[k, :max(2, min(n, math.ceil(len(a) / SR * FPS)))] = 1
        return idx, x, mask, torch.tensor([self.rows[i]["y"] for i in idx])

    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor
        rng = random.Random(self.seed)
        by_window = {}
        for i, r in enumerate(self.rows):
            by_window.setdefault(window_of(min(r["duration"], WINDOW)), []).append(i)
        chunks = []
        for idx in by_window.values():
            if self.train:
                rng.shuffle(idx)
            chunks += [idx[k:k + self.bs] for k in range(0, len(idx), self.bs)]
        if self.train:
            rng.shuffle(chunks)
        with ThreadPoolExecutor(6) as pool:
            futs = [pool.submit(self.make, c, np.random.default_rng(self.seed * 100003 + k)) for k, c in enumerate(chunks[:6])]
            for k in range(len(chunks)):
                if k + 6 < len(chunks):
                    futs.append(pool.submit(self.make, chunks[k + 6], np.random.default_rng(self.seed * 100003 + k + 6)))
                yield futs[k].result()
                futs[k] = None


class Tuned:
    """A phase 2 model for serving: audio clips -> zone probabilities (common.load_classifier)."""

    def __init__(self, path, device="cuda"):
        from transformers import WhisperFeatureExtractor
        ck = torch.load(path, map_location="cpu")
        a = ck["args"]
        self.model = build(r=a["rank"], first_tuned=a["first_tuned"])
        self.model.load_state_dict(ck["state"], strict=False)
        self.model.to(device).eval()
        self.fes = {w: WhisperFeatureExtractor.from_pretrained(WHISPER, chunk_length=w) for w in WINDOWS}
        self.device = device
        self.info = ck.get("info", {})

    @torch.no_grad()
    def probs(self, audios):
        out = []
        for a in audios:  # one clip per call in the service; each gets the shortest window that holds it
            a = a[-WINDOW * SR:]
            w = window_of(len(a) / SR)
            x = self.fes[w]([a], sampling_rate=SR, return_tensors="pt").input_features
            mask = torch.zeros(1, x.shape[-1] // 2)
            mask[0, :max(2, min(mask.shape[1], math.ceil(len(a) / SR * FPS)))] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(x.to(self.device, torch.bfloat16), mask.to(self.device)).float()
            out.append(torch.softmax(logits, -1)[0].cpu().numpy())
        return np.stack(out)


@torch.no_grad()
def predict(model, fes, rows, bs=16):
    model.eval()
    out = torch.zeros(len(rows), len(ZONES))
    for idx, x, mask, _ in Batches(fes, rows, bs, False, 0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = model(x.cuda().to(torch.bfloat16), mask.cuda()).float().cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--first-tuned", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000, help="steps between dev evaluations")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--gpu-share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=DATA, help="folder with train/dev/test jsonl (prepare.py split)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--dev-limit", type=int, default=0, help="score only this many dev clips (smoke test)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.gpu_share)  # the GPU is shared with the app's engine
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    from transformers import WhisperFeatureExtractor
    fes = {w: WhisperFeatureExtractor.from_pretrained(WHISPER, chunk_length=w) for w in WINDOWS}
    train, dev = read("train", args.data), read("dev", args.data)
    if args.dev_limit:
        dev = random.Random(0).sample(dev, args.dev_limit)
    model = build(r=args.rank, first_tuned=args.first_tuned).cuda()
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")]
    print(f"train {len(train)}, dev {len(dev)}; trainable LoRA {sum(p.numel() for p in lora):,}, head {sum(p.numel() for p in rest):,}", flush=True)
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr}, {"params": rest, "lr": args.head_lr}], weight_decay=0.01)
    steps = args.max_steps or args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.05)
    counts = np.bincount([r["y"] for r in train], minlength=len(ZONES))
    weight = torch.tensor(len(train) / (len(ZONES) * np.maximum(counts, 1)), dtype=torch.float32).cuda()
    lossf = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.smoothing)
    dev_y = np.array([r["y"] for r in dev])
    best, best_state, step, t0, run_loss, history = -1.0, None, 0, time.time(), 0.0, []
    for epoch in range(args.epochs):
        for _, x, mask, y in Batches(fes, train, args.batch, True, args.seed + epoch):
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.cuda().to(torch.bfloat16), mask.cuda())
            loss = lossf(logits.float(), y.cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            if step % 100 == 0:
                rate = step * args.batch / (time.time() - t0)
                print(f"step {step}/{steps} loss {run_loss / 100:.4f} ({rate:.1f} clips/s, {(steps - step) * args.batch / rate / 3600:.1f} h left)", flush=True)
                run_loss = 0.0
            if step % args.eval_every == 0 or step == steps:
                m = report(dev, dev_y, predict(model, fes, dev))
                history.append({"step": step, "dev_macro_f1": m["macro_f1"], "dev_auc": m["standard_vs_regional_auc"]})
                print(f"  dev @ {step}: macro-F1 {m['macro_f1']} acc {m['accuracy']} AUC {m['standard_vs_regional_auc']}", flush=True)
                if m["macro_f1"] > best:
                    best = m["macro_f1"]
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items() if "lora" in k or not k.startswith("enc.") or k.endswith((".a.weight", ".b.weight"))}
                    torch.save({"args": vars(args), "state": best_state, "step": step, "dev_macro_f1": best}, os.path.join(out, "model.pt"))
            if step >= steps:
                break
        if step >= steps:
            break
    model.load_state_dict(best_state, strict=False)
    dev_logits = predict(model, fes, dev)
    model.temperature.fill_(fit_temperature(dev_logits, dev_y))
    metrics = {"dev": report(dev, dev_y, predict(model, fes, dev)), "temperature": float(model.temperature), "history": history,
               "layer_weights": [round(float(w), 4) for w in torch.softmax(model.mix.detach().cpu(), 0)]}
    test_logits = {}
    for split in ("test_plain", "test_same", "test_dialect", "bench"):
        rows = read(split, args.data)
        if args.max_steps:
            rows = rows[:64]
        test_logits[split] = predict(model, fes, rows)
        metrics[split] = report(rows, np.array([r["y"] for r in rows]), test_logits[split])
        m = metrics[split]
        print(f"{split}: macro-F1 {m['macro_f1']} acc {m['accuracy']} AUC {m['standard_vs_regional_auc']} "
              f"std→regional {m['standard_called_regional']} coarse {m['coarse_accuracy']} persons {m['persons']}", flush=True)
    best_state["temperature"] = model.temperature.detach().cpu()
    torch.save({"args": vars(args), "state": best_state, "info": {"zones": ZONES, "test_plain_macro_f1": metrics["test_plain"]["macro_f1"],
                "bench_macro_f1": metrics["bench"]["macro_f1"]}}, os.path.join(out, "model.pt"))
    np.savez(os.path.join(out, "test_logits.npz"), **{k: v.numpy() for k, v in test_logits.items()})  # for leakage.py
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
