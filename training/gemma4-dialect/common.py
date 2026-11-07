"""Shared prompt format, audio loading, batching and scoring for train.py and evaluate.py."""

import json
import os
import re
import wave

import numpy as np
import torch

MODEL_PATH = "/data/huggingface/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
END_OF_TURN_ID = 106  # "<turn|>", one of the model's eos ids

PROMPTS = {
    "basic": (
        "다음 음성은 한국어 방언일 수 있습니다. 들은 그대로 받아쓰고, "
        "의미와 말투를 유지한 표준어로 바꿔 주세요. 형식:\n"
        "방언: <들은 그대로>\n표준어: <표준어 문장>"
    ),
    # The zone comes first, so the model names it from the sound before it writes the words.
    # The zone names are learned from the answers rather than listed here: listing them
    # doubled the prompt (107 tokens) and made each training step about 24% slower.
    "region": (
        "다음 음성은 한국어 방언일 수 있습니다. 말투로 화자의 지역을 고르고, 들은 그대로 받아쓰고, "
        "의미와 말투를 유지한 표준어로 바꿔 주세요. 형식:\n"
        "지역: <지역>\n방언: <들은 그대로>\n표준어: <표준어 문장>"
    ),
}
PROMPT = PROMPTS["basic"]


def answer_text(row, fmt="basic"):
    text = f"방언: {row['dialect_form']}\n표준어: {row['standard_form']}"
    return f"지역: {row['origin']}\n{text}" if fmt == "region" else text


def read_manifest(path, limit_per_region=None, require=None):
    """Read one or more comma-separated manifests; audio paths are relative to each manifest.
    `require` drops rows whose field of that name is empty (e.g. "origin" for the region format)."""
    rows = []
    for p in path.split(","):
        base = os.path.dirname(os.path.abspath(p))
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            r["audio_path"] = os.path.join(base, r["audio"])
            rows.append(r)
    if require:
        rows = [r for r in rows if r.get(require)]
    if limit_per_region:
        kept, seen = [], {}
        for r in rows:
            if seen.get(r["region"], 0) < limit_per_region:
                kept.append(r)
                seen[r["region"]] = seen.get(r["region"], 0) + 1
        rows = kept
    return rows


def load_audio(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def prompt_inputs(processor, audio, fmt="basic"):
    """Tokenize the user turn exactly as inference does (generation prompt included)."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPTS[fmt]}, {"type": "audio", "audio": audio}]}]
    return processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                         add_generation_prompt=True)


def _pad_features(feats, masks):
    n = max(f.shape[1] for f in feats)
    out_f = torch.zeros(len(feats), n, feats[0].shape[2], dtype=feats[0].dtype)
    out_m = torch.zeros(len(feats), n, dtype=torch.bool)
    for i, (f, m) in enumerate(zip(feats, masks)):
        out_f[i, :f.shape[1]] = f[0]
        out_m[i, :m.shape[1]] = m[0]
    return out_f, out_m


class TrainCollator:
    """Prompt (masked) + answer + <turn|> (trained), right-padded."""

    def __init__(self, processor, fmt="basic"):
        self.processor = processor
        self.tok = processor.tokenizer
        self.fmt = fmt

    def __call__(self, rows):
        ids, tts, labels, feats, masks = [], [], [], [], []
        for r in rows:
            p = prompt_inputs(self.processor, load_audio(r["audio_path"]), self.fmt)
            ans = self.tok(answer_text(r, self.fmt), add_special_tokens=False)["input_ids"] + [END_OF_TURN_ID]
            pid = p["input_ids"][0].tolist()
            ids.append(pid + ans)
            tts.append(p["mm_token_type_ids"][0].tolist() + [0] * len(ans))
            labels.append([-100] * len(pid) + ans)
            feats.append(p["input_features"])
            masks.append(p["input_features_mask"])
        n = max(len(x) for x in ids)
        pad = self.tok.pad_token_id or 0
        batch = {
            "input_ids": torch.tensor([x + [pad] * (n - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids]),
            "mm_token_type_ids": torch.tensor([x + [0] * (n - len(x)) for x in tts]),
            "labels": torch.tensor([x + [-100] * (n - len(x)) for x in labels]),
        }
        batch["input_features"], batch["input_features_mask"] = _pad_features(feats, masks)
        return batch


def generation_batch(processor, rows, fmt="basic"):
    """Left-padded prompts for batched generation."""
    ps = [prompt_inputs(processor, load_audio(r["audio_path"]), fmt) for r in rows]
    n = max(p["input_ids"].shape[1] for p in ps)
    pad = processor.tokenizer.pad_token_id or 0
    ids, att, tts = [], [], []
    for p in ps:
        k = n - p["input_ids"].shape[1]
        ids.append([pad] * k + p["input_ids"][0].tolist())
        att.append([0] * k + p["attention_mask"][0].tolist())
        tts.append([0] * k + p["mm_token_type_ids"][0].tolist())
    batch = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att),
             "mm_token_type_ids": torch.tensor(tts)}
    batch["input_features"], batch["input_features_mask"] = _pad_features(
        [p["input_features"] for p in ps], [p["input_features_mask"] for p in ps])
    return batch


def parse_answer(text):
    dia = re.search(r"방언:\s*(.*)", text)
    std = re.search(r"표준어:\s*(.*)", text)
    return (dia.group(1).strip() if dia else ""), (std.group(1).strip() if std else "")


def parse_region(text):
    m = re.search(r"지역:\s*(.*)", text)
    return m.group(1).strip() if m else ""


_STRIP = re.compile(r"[\s.,?!~'\"“”‘’·…-]")


def cer(ref, hyp):
    """Character error rate ignoring spaces and punctuation."""
    r, h = _STRIP.sub("", ref), _STRIP.sub("", hyp)
    if not r:
        return float(bool(h))
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)
