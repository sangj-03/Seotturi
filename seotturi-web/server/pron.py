"""Pronunciation of the sounds regional accents merge, for the 서투리 app.

Gyeongsang speakers say ㅞ as ㅔ in about 35 % and ㅓ as ㅡ in about 20.6 % of tokens (Lee, Kim & Chung 2023),
and often do not tell ㅆ from ㅅ. For each such sound in what was said, this module measures goodness of
pronunciation (Witt & Young 2000) with a Korean phone recogniser (wav2vec2 XLS-R 300M fine-tuned on MFA
phones of read native speech, slplab/wav2vec2-xls-r-300m_phone-mfa_korean, Apache-2.0; wav2vec2 for
mispronunciation detection: Xu et al. 2021): the canonical phones of the transcript are force-aligned to the
recording (CTC), and each target sound scores the mean log-posterior ratio of the expected phone over the
merged one on its frames (ㅓ EO vs ㅡ EU, ㅆ SS vs ㅅ S, ㅚ/ㅙ/ㅞ oE vs ㅔ E). A ratio below the threshold
counts as the merged sound heard.

Calibration on accent-id's plain-speech clips (no dialect words; 2,026 clips, at most two per recording;
a one-off calibration script, not included), thresholds at the 5th percentile of Seoul-Gyeonggi speakers:
  ㅚ·ㅙ·ㅞ -> ㅔ  Seoul 5.3 %, 대구·경북 8.2 %, 부산·울산·경남 11.8 %, 충청 0 %, 강원 0 %, 제주 4.1 % (Gyeongsang
                 only, as in Lee, Kim & Chung 2023) - the one check shown in the app;
  ㅓ -> ㅡ       Seoul 5.0 %, every other zone 7-11 %: not specific to Gyeongsang, not shown;
  ㅆ -> ㅅ       Seoul 5.2 %, 부산·울산·경남 3.3 %: no difference, not shown.
So it is a weak signal per sound (about twice the Seoul rate); an answer's result is shown only when its count
is unlikely under the Seoul rate (one-sided binomial test, p < 0.05).

g2p() turns Hangul into those phones with the main rules of standard pronunciation (연음, ㅎ 탈락·격음화,
받침 일곱 소리, 비음화, 유음화, ㄹ의 비음화, 경음화, 구개음화) inside each 어절.
"""

import glob
import os
import re

import numpy as np

MODEL = "/data/huggingface/hub/models--slplab--wav2vec2-xls-r-300m_phone-mfa_korean"

ONSETS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
CODAS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
         "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
ONSET_PHONE = {"ㄱ": "G", "ㄲ": "GG", "ㄴ": "N", "ㄷ": "D", "ㄸ": "DD", "ㄹ": "R", "ㅁ": "M", "ㅂ": "B", "ㅃ": "BB",
               "ㅅ": "S", "ㅆ": "SS", "ㅇ": "", "ㅈ": "J", "ㅉ": "JJ", "ㅊ": "CHh", "ㅋ": "Kh", "ㅌ": "Th", "ㅍ": "Ph",
               "ㅎ": "H"}
VOWEL_PHONE = dict(zip(VOWELS, ["A", "E", "iA", "iE", "EO", "E", "iEO", "iE", "O", "oA", "oE", "oE", "iO", "U", "uEO",
                                "oE", "uI", "iU", "EU", "euI", "I"]))
# 받침 일곱 소리 (after the double codas give up one consonant)
CODA_SOUND = {"ㄱ": "k", "ㄲ": "k", "ㅋ": "k", "ㄳ": "k", "ㄺ": "k", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N", "ㄷ": "t", "ㅅ": "t",
              "ㅆ": "t", "ㅈ": "t", "ㅊ": "t", "ㅌ": "t", "ㅎ": "t", "ㄹ": "L", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㅀ": "L",
              "ㅁ": "M", "ㄻ": "M", "ㅂ": "p", "ㅍ": "p", "ㅄ": "p", "ㄿ": "p", "ㅇ": "NG"}
# a double coda before a vowel: (what stays, what moves on)
SPLIT = {"ㄳ": ("ㄱ", "ㅆ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", ""), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"),
         "ㄽ": ("ㄹ", "ㅆ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", ""), "ㅄ": ("ㅂ", "ㅆ")}
ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ", "ㅂ": "ㅍ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
NASAL = {"k": "NG", "t": "N", "p": "M"}

# target: merged phone, names for the feedback, LPR threshold (Seoul-Gyeonggi 5th percentile), shown in the app
CHECKS = {"EO": ("EU", "ㅓ", "ㅡ", -1.16, False), "SS": ("S", "ㅆ", "ㅅ", -2.57, False),
          "oE": ("E", "ㅚ·ㅙ·ㅞ", "ㅔ", -6.44, True)}
BASE_RATE, ALPHA = 0.05, 0.05  # Seoul-Gyeonggi speakers fall below a threshold 5 % of the time


def syllables(word):
    """[[onset, vowel, coda], ...] of the Hangul syllables of one 어절 (other characters dropped)."""
    out = []
    for ch in word:
        k = ord(ch) - 0xAC00
        if 0 <= k < 11172:
            out.append([ONSETS[k // 588], VOWELS[(k % 588) // 28], CODAS[k % 28]])
    return out


def g2p(text):
    """Canonical phones of a sentence: [(phone, target or None, 어절 index, syllable index)], "|" between 어절.
    target is set on the sounds CHECKS scores: every ㅓ and ㅚ/ㅙ/ㅞ, and ㅆ written as such (not ㅆ from 경음화)."""
    out = []
    for wi, word in enumerate(text.split()):
        syl = syllables(word)
        if not syl:
            continue
        if out:
            out.append(("|", None, wi, -1))
        written_ss = [o == "ㅆ" for o, _, _ in syl]
        for i in range(len(syl) - 1):  # the next onset is changed by this coda
            o2 = syl[i + 1][0]
            c = syl[i][2]
            if not c:
                continue
            if o2 == "ㅇ":  # 연음 (and ㅎ 탈락)
                if c == "ㅇ":
                    continue
                if c in SPLIT:
                    stay, move = SPLIT[c]
                    syl[i][2] = stay
                    if move:
                        syl[i + 1][0] = move
                        written_ss[i + 1] = False  # 넋이 [넉씨]: this ㅆ is 경음화, not written
                    continue
                if c == "ㅎ":
                    syl[i][2] = ""
                    continue
                if c in ("ㄷ", "ㅌ") and syl[i + 1][1] == "ㅣ":  # 구개음화
                    c = "ㅈ" if c == "ㄷ" else "ㅊ"
                syl[i][2], syl[i + 1][0] = "", c
                written_ss[i + 1] = c == "ㅆ"
            elif c in ("ㅎ", "ㄶ", "ㅀ"):  # ㅎ + ㄱㄷㅈ -> 격음, ㅎ + ㅅ -> ㅆ, ㅎ + ㄴ -> ㄴ
                rest = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[c]
                if o2 in ASPIRATE:
                    syl[i][2], syl[i + 1][0] = rest, ASPIRATE[o2]
                elif o2 == "ㅅ":
                    syl[i][2], syl[i + 1][0] = rest, "ㅆ"
                elif c == "ㅎ" and o2 == "ㄴ":
                    syl[i][2] = "ㄴ"
            elif o2 == "ㅎ" and CODA_SOUND.get(c) in ("k", "t", "p"):  # 격음화
                if c in SPLIT:
                    stay, move = SPLIT[c]
                    syl[i][2], syl[i + 1][0] = stay, ASPIRATE.get(move, "ㅎ")
                else:
                    syl[i][2], syl[i + 1][0] = "", ASPIRATE.get({"ㅅ": "ㄷ", "ㅈ": "ㅈ"}.get(c, c), "ㅎ")
        phones = []  # (onset, vowel, coda sound) after neutralisation, then the boundary rules on sounds
        for o, v, c in syl:
            phones.append([o, v, CODA_SOUND.get(c, "")])
        for i in range(len(phones) - 1):
            c, o2 = phones[i][2], phones[i + 1][0]
            if c in NASAL and o2 in ("ㄴ", "ㅁ"):
                phones[i][2] = NASAL[c]
            elif c in NASAL and o2 == "ㄹ":
                phones[i][2], phones[i + 1][0] = NASAL[c], "ㄴ"
            elif c in ("M", "NG") and o2 == "ㄹ":
                phones[i + 1][0] = "ㄴ"
            elif c == "N" and o2 == "ㄹ":
                phones[i][2] = "L"
            elif c == "L" and o2 == "ㄴ":
                phones[i + 1][0] = "ㄹ"
            if phones[i][2] in ("k", "t", "p") and phones[i + 1][0] in TENSE:
                phones[i + 1][0] = TENSE[phones[i + 1][0]]
        for si, (o, v, c) in enumerate(phones):
            if ONSET_PHONE[o]:
                p = "L" if o == "ㄹ" and si and phones[si - 1][2] == "L" else ONSET_PHONE[o]
                out.append((p, "SS" if p == "SS" and written_ss[si] else None, wi, si))
            vp = VOWEL_PHONE[v]
            out.append((vp, vp if vp in ("EO", "oE") else None, wi, si))
            if c:
                out.append((c, None, wi, si))
    return out


class Scorer:
    """The phone recogniser (GPU when there is one) and the scoring of one clip."""

    def __init__(self, device=None):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        path = sorted(glob.glob(os.path.join(MODEL, "snapshots", "*")))[-1]
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Wav2Vec2ForCTC.from_pretrained(path, dtype=dtype).to(self.device).eval()
        import json
        with open(os.path.join(path, "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.blank = self.vocab["[PAD]"]

    def log_probs(self, audio):
        x = self.fe(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device, self.model.dtype)
        with self.torch.inference_mode():
            logits = self.model(x).logits[0].float()
        return self.torch.log_softmax(logits, dim=-1)

    def score(self, audio, text):
        """[{"word", "syllable", "target", "lpr"}] for the target sounds of `text` in `audio` (16 kHz mono)."""
        import torchaudio.functional as F
        phones = [p for p in g2p(text) if p[0] in self.vocab and p[0] != "|"]  # the model writes no word breaks
        if not any(p[1] for p in phones):
            return []
        lp = self.log_probs(audio)
        ids = self.torch.tensor([[self.vocab[p[0]] for p in phones]], device=lp.device)
        if lp.shape[0] < ids.shape[1]:  # too short for the transcript
            return []
        try:
            frames, _ = F.forced_align(lp[None], ids, blank=self.blank)
        except RuntimeError:
            return []
        spans = F.merge_tokens(frames[0], self.torch.zeros_like(frames[0], dtype=self.torch.float), blank=self.blank)
        if len(spans) != len(phones):
            return []
        words = text.split()
        out = []
        for (phone, target, wi, si), span in zip(phones, spans):
            if not target:
                continue
            other = self.vocab[CHECKS[target][0]]
            seg = lp[span.start:span.end]
            lpr = float((seg[:, self.vocab[phone]] - seg[:, other]).mean())
            out.append({"w": wi, "word": re.sub(r"[^\w]", "", words[wi]), "syllable": si, "target": target,
                        "lpr": round(lpr, 2)})
        return out


def flagged(results):
    """The scored sounds below their threshold: heard as the merged sound."""
    return [r for r in results if r["lpr"] < CHECKS[r["target"]][3]]


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def summary(results):
    """Per shown target: how many were scored and heard as the merged sound, with example words, when that
    many is unlikely for a Seoul-Gyeonggi speaker (else left out)."""
    out = {}
    for t, (merged, name, merged_name, _, shown) in CHECKS.items():
        rs = [r for r in results if r["target"] == t]
        bad = flagged(rs)
        if shown and bad and binom_tail(len(bad), len(rs), BASE_RATE) < ALPHA:
            out[t] = {"name": name, "merged": merged_name, "n": len(rs), "flagged": len(bad),
                      "words": list(dict.fromkeys(r["word"] for r in bad))[:5]}
    return out


if __name__ == "__main__":
    for s in ["거기 있어요", "어떻게 해야 돼요", "좋아요 쌀이 싸요", "외국에 갔었어요", "국수를 먹었어요", "신라 종로 협력"]:
        print(s, [(p, t) for p, t, _, _ in g2p(s)])
    print(np.__version__)
