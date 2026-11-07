"""Shared pieces of the accent-zone classifier: the zones, audio, Whisper encoder features, the head.

The serving engine (seotturi-web/server/engine.py) imports this module, so the features and the model
it runs are exactly the ones trained here.

Features: Whisper-large-v3's encoder, frozen, on a 15 s window (the first 750 of its 1,500 position
embeddings; a clip longer than 15 s keeps its last 15 s, where the sentence-final intonation is).
For every one of its 33 hidden states (the embeddings and 32 layers) the frames that hold speech are
pooled into mean and standard deviation: [33, 2560] per clip.

Head: a learned softmax mix of the 33 layers -> LayerNorm -> 256 -> 64 -> 6 zones, with a temperature
fitted on the dev set so the probabilities can be shown as they are.
"""

import json
import math
import os
import wave

import numpy as np
import torch
from torch import nn

ZONES = ["서울·경기", "강원", "충청", "대구·경북", "부산·울산·경남", "제주"]
STANDARD = 0  # index of the standard-accent zone
WHISPER = "openai/whisper-large-v3"
SR = 16000
WINDOW = 15  # seconds the encoder sees
FPS = 50     # encoder frames per second
LAYERS, DIM = 33, 2 * 1280


def load_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


class Encoder:
    """Frozen Whisper-large-v3 encoder -> pooled hidden states of every layer."""

    def __init__(self, device="cuda", window=WINDOW):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(WHISPER, chunk_length=window)
        enc = WhisperModel.from_pretrained(WHISPER, dtype=torch.float16).encoder
        n = window * FPS
        if n != enc.config.max_source_positions:  # a shorter window: keep the first n positions
            pos = enc.embed_positions.weight.data[:n].clone()
            enc.embed_positions = nn.Embedding(n, pos.shape[1])
            enc.embed_positions.weight.data.copy_(pos)
            enc.embed_positions.requires_grad_(False)
            enc.config.max_source_positions = n
        self.enc = enc.to(device=device, dtype=torch.float16).eval()
        self.device, self.window = device, window

    @torch.no_grad()
    def pooled(self, audios):
        """float32 16 kHz arrays -> [B, 33, 2560] float16 (mean, std over the clip's frames, per layer)."""
        audios = [a[-self.window * SR:] for a in audios]
        x = self.fe(audios, sampling_rate=SR, return_tensors="pt").input_features.to(self.device, torch.float16)
        hs = torch.stack(self.enc(x, output_hidden_states=True).hidden_states, 1)  # [B, 33, T, 1280]
        out = []
        for i, a in enumerate(audios):
            n = max(2, min(hs.shape[2], math.ceil(len(a) / SR * FPS)))
            h = hs[i, :, :n].float()
            out.append(torch.cat([h.mean(1), h.std(1)], -1))
        return torch.stack(out).half()


class Head(nn.Module):
    """Pooled features [B, 33, 2560] -> zone logits."""

    def __init__(self, hidden=256, bottleneck=64, dropout=0.2, n_out=len(ZONES)):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(LAYERS))
        self.net = nn.Sequential(nn.LayerNorm(DIM), nn.Dropout(dropout), nn.Linear(DIM, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, bottleneck), nn.GELU(),
                                 nn.Linear(bottleneck, n_out))  # n_out 4 for region4.py's runs
        self.register_buffer("mu", torch.zeros(LAYERS, DIM))
        self.register_buffer("sd", torch.ones(LAYERS, DIM))
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, x):
        x = (x.float() - self.mu) / self.sd
        x = torch.einsum("bld,l->bd", x, torch.softmax(self.mix, 0))
        return self.net(x) / self.temperature


class Classifier:
    """Encoder + trained head: audio clips -> zone probabilities (what the engine runs)."""

    def __init__(self, head_path, device="cuda"):
        self.encoder = Encoder(device)
        ck = torch.load(head_path, map_location="cpu")
        self.head = Head(**ck["config"])
        self.head.load_state_dict(ck["state"])
        self.head.to(device).eval()
        self.info = ck.get("info", {})

    @torch.no_grad()
    def probs(self, audios):
        return torch.softmax(self.head(self.encoder.pooled(audios)), -1).cpu().numpy()


def load_classifier(run_dir, device="cuda"):
    """The classifier of a run: phase 1 (head.pt on the frozen encoder), phase 2 (model.pt, LoRA-tuned
    encoder, finetune.py) or an ensemble of runs (ensemble.json). Returns an object with
    .probs(audios) -> [N, 6] and .info."""
    if os.path.exists(os.path.join(run_dir, "ensemble.json")):
        return Ensemble(run_dir, device)
    if os.path.exists(os.path.join(run_dir, "model.pt")):
        import finetune
        return finetune.Tuned(os.path.join(run_dir, "model.pt"), device)
    return Classifier(os.path.join(run_dir, "head.pt"), device)


class Ensemble:
    """Classifiers mixed in log-probability with fixed weights, as listed in <run>/ensemble.json:
    {"members": [{"run": "phase1", "weight": 0.6}, ...]} (runs next to this one)."""

    def __init__(self, run_dir, device="cuda"):
        with open(os.path.join(run_dir, "ensemble.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        base = os.path.dirname(os.path.abspath(run_dir))
        self.members = [(load_classifier(os.path.join(base, m["run"]), device), m["weight"]) for m in self.info["members"]]

    def probs(self, audios):
        logp = sum(w * np.log(np.clip(c.probs(audios), 1e-9, 1.0)) for c, w in self.members)
        p = np.exp(logp - logp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)


class Tree:
    """A run (head.pt, model.pt or a nested ensemble.json) whose frozen-encoder heads all read one shared
    Encoder's pooled features, so several heads cost one encoder pass. accent.py serves the four-region run
    and the standard detector this way; eval scripts use it to score exactly what the service runs.
    probs(audios, x): x = encoder.pooled(audios) when the caller already has it."""

    def __init__(self, run_dir, encoder, device="cuda"):
        self.encoder, self.weight = encoder, 1.0
        spec = os.path.join(run_dir, "ensemble.json")
        if os.path.exists(spec):
            with open(spec, encoding="utf-8") as f:
                self.info = json.load(f)
            base = os.path.dirname(os.path.abspath(run_dir))
            self.members = [(Tree(os.path.join(base, m["run"]), encoder, device), m["weight"]) for m in self.info["members"]]
            self.zones = self.members[0][0].zones
            assert all(t.zones == self.zones for t, _ in self.members), "ensemble members disagree on zones"
            self.kind = "mix"
        elif os.path.exists(os.path.join(run_dir, "model.pt")):
            import finetune
            self.model = finetune.Tuned(os.path.join(run_dir, "model.pt"), device)  # LoRA: its own encoder
            self.info, self.zones, self.kind = self.model.info, list(ZONES), "tuned"
        else:
            ck = torch.load(os.path.join(run_dir, "head.pt"), map_location="cpu")
            self.head = Head(**ck["config"])
            self.head.load_state_dict(ck["state"])
            self.head.to(device).eval()
            self.info, self.kind = ck.get("info", {}), "head"
            self.zones = list(self.info.get("zones") or ZONES)

    @torch.no_grad()
    def probs(self, audios, x=None):
        if self.kind == "head":
            x = self.encoder.pooled(audios) if x is None else x
            return torch.softmax(self.head(x), -1).float().cpu().numpy()
        if self.kind == "tuned":
            return self.model.probs(audios)
        if x is None and any(t.kind != "tuned" for t, _ in self.members):
            x = self.encoder.pooled(audios)
        logp = sum(w * np.log(np.clip(t.probs(audios, x), 1e-9, 1.0)) for t, w in self.members)
        p = np.exp(logp - logp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)
