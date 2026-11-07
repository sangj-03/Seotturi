"""서투리 feedback logic on top of the engine's answers.

The dialect model returns, per clip, what was said (방언) and its standard-Korean form (표준어),
and from stage 3 on the speaker's accent zone (지역). This module turns that into
the coaching shown in the app:

* word-level corrections between the two lines and the deck's 사투리 점수
  (non-standard 어절 x 100 / all 어절, where the other 어절 of a clip spoken with a regional accent
  count by that clip's accent strength), shown as 표준어 점수 = 100 - 사투리 비율, with its lexical
  and accent parts also reported apart;
* prompts for plain Gemma 4 (content tips and sub-scores, daily-conversation replies,
  personalised review sentences) in a line format that can be parsed while it streams;
* the deck's 면접 점수: 표준어 점수 + 답변 점수 (word choice, fluency and, when the answer
  carries job facts, factual content), 200 points with a pass mark of 140.
"""

import difflib
import html
import io
import os
import re
import subprocess
import tempfile
import zipfile
from collections import Counter

import numpy as np

import pron

PUNCT = re.compile(r"[.,?!~…·\"'“”‘’()\[\]{}<>/:;]")
STANDARD_ZONE = "서울·경기"

ZONE_TIPS = {
    "경상": "높낮이가 뚜렷하고 단어 첫머리에 힘이 실리는 경상 억양이 느껴져요. 표준어는 비교적 평탄하니 높낮이를 줄이고, "
            "강세를 고르게 나눠 문장 끝을 부드럽게 내려 보세요.",
    "부산·울산·경남": "높낮이가 뚜렷한 경남 억양이 느껴져요. 표준어는 비교적 평탄하니 높낮이를 줄이고 문장 끝을 부드럽게 내려 보세요.",
    "대구·경북": "단어 첫머리에 힘이 실리는 경북 억양이 느껴져요. 강세를 고르게 나누고 조금 천천히 말해 보세요.",
    "전라": "말끝을 길게 끌거나 올리는 전라 억양이 느껴져요. 문장 끝을 짧고 단정하게 맺어 보세요.",
    "충청": "말끝을 길게 늘이는 충청 억양이 느껴져요. 속도를 조금 올리고 끝소리를 또렷하게 맺어 보세요.",
    "강원": "말끝이 올라가는 강원 억양이 느껴져요. 평서문은 끝을 내려서 말해 보세요.",
    "제주": "제주 말투의 어휘와 어미가 들려요. '-어요', '-습니다' 같은 표준어 어미로 바꿔 말해 보세요.",
}


def norm(tok):
    return PUNCT.sub("", tok)


def diff(dialect, standard):
    """Corrections from the dialect line to the standard line, word by word.

    Returns the dialect tokens (each with the index of its correction or None), the
    corrections themselves and the counts behind the 사투리 비율."""
    a, b = dialect.split(), standard.split()
    an, bn = [norm(t) for t in a], [norm(t) for t in b]
    tokens, fixes = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=an, b=bn, autojunk=False).get_opcodes():
        src, dst = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if op == "equal" or (op == "replace" and norm(src).replace(" ", "") == norm(dst).replace(" ", "")):
            tokens += [{"t": t, "fix": None} for t in a[i1:i2]]
        elif op in ("replace", "delete"):
            fixes.append({"from": norm(src), "to": norm(dst)})
            tokens += [{"t": t, "fix": len(fixes) - 1} for t in a[i1:i2]]
        elif op == "insert" and tokens:
            # the standard form adds words: attach them to the word before
            prev = tokens[-1]
            if prev["fix"] is None:
                fixes.append({"from": norm(prev["t"]), "to": norm(prev["t"]) + " " + norm(dst)})
                prev["fix"] = len(fixes) - 1
            else:
                fixes[prev["fix"]]["to"] += " " + norm(dst)
    changed = sum(1 for t in tokens if t["fix"] is not None)
    return {"tokens": tokens, "fixes": fixes, "changed": changed, "total": len(a)}


# Show a zone only when the averaged probability of the top zone reaches this (selective classification).
# On accent-id's test_plain, one clip at a time with the served ensemble: 0.6 shows 83 % of verdicts, and
# those are right 85 % of the time instead of 79 % (Seoul speakers called regional: 31 % instead of 35 %).
ZONE_MIN_CONF = 0.6
MIN_ZONE_SECONDS = 0  # speech needed before a zone verdict; 0 = shown from the first sentence (was 8, user's choice)
# With the standard-vs-regional detector (accent.py STD_RUN) the zone is decided in two steps, the rule that did
# best on whole answers (accent-id session_eval.py rules, 2026-10-12): regional when the averaged P(서울·경기) is
# below REGIONAL_MIN (the same line that counts the accent in the 사투리 비율), then the region from the regions'
# probabilities renormalised, "unsure" when the top one is below ZONE_MIN_CONF. Thresholding the five-way
# top probability instead left 25-30 % of answers unsure with the right region on top (0.9 regional x 0.65 for
# the region is 0.585): two steps decide 90 % of them, right 85-87 % of the time from four sentences, but call
# more Seoul-Gyeonggi speakers regional (14-17 % of them instead of 4-7 %; 100 speakers, so roughly).
REGIONAL_MIN = 0.5


def zone_verdict(probs, two_step=False):
    """(zone, unsure, share %) from an answer's averaged zone probabilities. With two_step, the share of a
    region is its probability given that the accent is regional."""
    if two_step and STANDARD_ZONE in probs:
        regional = 1.0 - probs[STANDARD_ZONE]
        if regional < REGIONAL_MIN:
            return STANDARD_ZONE, False, round(100 * probs[STANDARD_ZONE])
        given = {z: p / regional for z, p in probs.items() if z != STANDARD_ZONE}
        zone = max(given, key=given.get)
        if given[zone] < ZONE_MIN_CONF:
            return None, True, None
        return zone, False, round(100 * given[zone])
    zone = max(probs, key=probs.get)
    if probs[zone] < ZONE_MIN_CONF:  # too close to call: say so instead of guessing
        return None, True, None
    return zone, False, round(probs[zone] * 100)


def mean_probs(items):
    """Weighted mean of {zone: probability} dicts given as [(probs or None, weight)]; None if there are none."""
    items = [(p, w) for p, w in items if p and w > 0]
    total = sum(w for _, w in items)
    if not total:
        return None
    return {z: round(sum(p[z] * w for p, w in items) / total, 4) for z in items[0][0]}


def zone_tip(zone):
    if not zone or zone == STANDARD_ZONE:
        return None
    return ZONE_TIPS.get(zone)


def accent_strength(probs):
    """How regional an accent sounds, 0-1: the classifier's probability of any zone but 서울·경기."""
    return 1.0 - probs.get(STANDARD_ZONE, 0.0) if probs else 0.0


def accent_sentences(segments, zone):
    """Per clip: does its accent count as dialect? Only when the answer's accent verdict is a regional zone,
    and then for the clips the classifier itself places outside 서울·경기 (so the zone shown and the
    사투리 비율 never disagree, and one stray clip of a standard speaker does not count)."""
    if not zone or zone == STANDARD_ZONE:
        return [False] * len(segments)
    return [bool(s.get("zone_probs")) and max(s["zone_probs"], key=s["zone_probs"].get) != STANDARD_ZONE for s in segments]


def join_pron(results, prons):
    """Pronunciation results of the clips of one upload as one list, word indices shifted to the joined
    dialect line (transcribe() joins the clips' lines with spaces)."""
    if not any(prons):
        return None
    out, base = [], 0
    for r, pr in zip(results, prons):
        out += [{**x, "w": x["w"] + base} for x in pr or []]
        base += len(r["dialect"].split())
    return out


def pron_summary(segments):
    """pron.summary over the sounds in words that are not dialect words (those get their own correction)."""
    sounds = []
    for s in segments:
        tokens = s["analysis"]["tokens"]
        sounds += [x for x in s.get("pron") or [] if x["w"] < len(tokens) and tokens[x["w"]]["fix"] is None]
    return pron.summary(sounds) or None if sounds else None


def accent_any_zone(segments, probs):
    """Per clip, with the standard-vs-regional detector (accent.py STD_RUN): the accent counts when the
    answer and the clip are more likely regional than standard, even if which region is still unsure (the
    detector answers "is it regional" far better than the four-region model alone, AUC 0.68)."""
    if accent_strength(probs) < 0.5:
        return [False] * len(segments)
    return [bool(s.get("zone_probs")) and accent_strength(s["zone_probs"]) >= 0.5 for s in segments]


def summarize_speech(segments, mode="interview", accent_without_zone=False):
    """Totals over the analysed clips of one answer: 사투리 비율, 표준어 점수, dominant zone.
    accent_without_zone: the zone probabilities come from the detector + four-region model (app.py)."""
    changed = sum(s["analysis"]["changed"] for s in segments)
    total = sum(s["analysis"]["total"] for s in segments)
    # Zone: the accent classifier's probabilities, averaged over the clips by length, once there is enough
    # speech; sessions saved before the classifier only have one zone per clip, counted instead.
    probs = mean_probs([(s.get("zone_probs"), float(s.get("duration") or 0)) for s in segments])
    heard = sum(float(s.get("duration") or 0) for s in segments if s.get("zone_probs"))
    pending = bool(probs) and heard < MIN_ZONE_SECONDS
    unsure = False
    if probs:
        zone, unsure, share = (None, False, None) if pending else zone_verdict(probs, accent_without_zone)
    else:
        zones = Counter(s.get("region") for s in segments if s.get("region"))
        zone = zones.most_common(1)[0][0] if zones else None
        share = round(zones[zone] / sum(zones.values()) * 100) if zone else None
    # A clip spoken with a regional accent counts as dialect (user's choice) by how strong its accent is: its
    # 어절 that are not dialect words already add 1 - P(서울·경기) each, not a full 어절 (dialect density as a
    # continuous measure, Johnson et al. 2022). Words and accent are also reported apart (Munro & Derwing
    # 1995: accent is partly independent of how understandable speech is). Only the classifier's verdicts count.
    if not probs or pending:
        accent = [False] * len(segments)
    else:
        accent = accent_any_zone(segments, probs) if accent_without_zone else accent_sentences(segments, zone)
    accent_words = round(sum((s["analysis"]["total"] - s["analysis"]["changed"]) * accent_strength(s["zone_probs"])
                             for s, a in zip(segments, accent) if a), 1)
    ratio = (changed + accent_words) / total if total else 0.0
    words = Counter((f["from"], f["to"]) for s in segments for f in s["analysis"]["fixes"] if f["from"])
    return {"changed": changed, "accent_words": accent_words, "accent": accent, "total": total,
            "accent_share": [round(100 * accent_strength(s["zone_probs"])) if a else None for s, a in zip(segments, accent)],
            "word_ratio": round(100 * changed / total) if total else 0,
            "accent_strength": round(100 * accent_strength(probs)) if probs and not pending else None,
            "dialect_ratio": round(ratio * 100), "standard_score": round(100 * (1 - ratio)), "zone": zone,
            "zone_share": share, "zone_probs": probs, "zone_pending": pending, "zone_unsure": unsure,
            "accent_any_zone": accent_without_zone,
            "fluency": fluency_summary(segments, mode), "pron": pron_summary(segments),
            "standard_share": round(probs[STANDARD_ZONE] * 100) if probs and STANDARD_ZONE in probs else None,
            "zone_tip": zone_tip(zone),
            "words": [{"from": d, "to": s, "count": n} for (d, s), n in words.most_common()]}


# ---- coaching prompts ---------------------------------------------------------------------

SYSTEM = ("너는 '서투리'의 AI 코치다. 서투리는 사투리를 틀린 말로 보지 않고, 면접·발표처럼 공식적인 자리에서 "
          "필요한 표준어 표현과 말하기를 익히도록 돕는 서비스다. 따뜻하지만 구체적으로, 짧게 코칭한다.")

MODE_FOCUS = {
    "interview": "면접 답변으로서 직무와 연결된 구체적 경험, 두괄식 구성, 근거와 결과가 드러나는지를 본다.",
    "presentation": "발표로서 구조(도입-본론-마무리), 핵심 메시지의 명확성, 연결어와 전달력을 본다.",
    "daily": "일상 대화로서 표준어로 자연스럽게 말하는 표현을 본다.",
}


def feedback_messages(mode, ctx, sentences, fixes, seconds, speech=None, missed=None):
    """Prompt for the content feedback. Output, one item per line:
    TIP|<문장 번호>|<인용>|<조언>, SCORE|<단어 선택>|<유창성>|<정보 사실 또는 ->, SUMMARY|<총평>,
    and for presentations QUESTION|<청중 질문>."""
    lines = [f"[상황] {ctx.get('title', '')}"]
    if ctx.get("persona"):
        lines.append(f"[상대] {ctx['persona']}: {ctx.get('persona_desc', '')}")
    if ctx.get("question"):
        lines.append(f"[질문] {ctx['question']}")
    if ctx.get("script"):
        lines.append(f"[준비한 대본] {ctx['script'][:1200]}")
    if ctx.get("focus"):
        lines.append(f"[사용자가 원하는 피드백] {', '.join(ctx['focus'])}")
    lines.append(f"[말한 시간] {round(seconds)}초")
    lines.append("[사용자 답변, 표준어로 옮긴 문장]")
    lines += [f"{i}. {s}" for i, s in enumerate(sentences, 1)]
    if speech:
        z = speech.get("zone")
        if z and speech.get("zone_probs"):  # the accent classifier (probabilities over six zones)
            zone = (f", 억양 판별(음성 분류기): 표준어(서울·경기) 억양에 가까움, 가능성 {speech['zone_share']}%"
                    if z == STANDARD_ZONE else
                    f", 억양 판별(음성 분류기): 지역 억양일 가능성 {speech['accent_strength']}%, "
                    f"지역 억양 중 {z} 억양일 가능성 {speech['zone_share']}%"
                    if speech.get("accent_any_zone") else
                    f", 억양 판별(음성 분류기): {z} 억양일 가능성 {speech['zone_share']}% "
                    f"(표준어 억양일 가능성 {speech['standard_share']}%)")
        elif speech.get("zone_unsure") and speech.get("accent_words"):  # regional, region not yet clear
            zone = ", 억양 판별: 지역 억양이지만 어느 지역인지는 확실하지 않음(지역 이름은 언급하지 않는다)"
        elif speech.get("zone_unsure"):
            zone = ", 억양 판별: 확실하지 않음(언급하지 않는다)"
        else:
            zone = f", 억양 판별: {z}" if z else ""
        accent = (f", 억양이 표준어가 아닌 문장의 나머지 어절을 억양 강도만큼 더해 {speech['accent_words']}어절"
                  if speech.get("accent_words") else "")
        strength = (f", 억양 강도 {speech['accent_strength']}%(표준어 억양이 아닐 확률)"
                    if speech.get("accent_strength") is not None
                    and (not speech.get("zone_unsure") or speech.get("accent_words")) else "")
        lines.append(f"[음성 모델의 말투 분석] 전체 {speech['total']}어절 중 {speech['changed']}어절이 사투리 단어{accent} "
                     f"(사투리 비율 {speech['dialect_ratio']}%, 단어만 {speech.get('word_ratio', 0)}%){strength}{zone}")
    fl = (speech or {}).get("fluency")
    if fl:
        lines.append(f"[말하기 측정, 녹음에서 잰 값] 말 속도 초당 {fl['rate']}음절(표준어 화자 보통 4.0~5.5), "
                     f"쉼 없이 이어 말한 길이 평균 {fl['mlr']}음절(보통 12), 군말 {fl['fillers']}번(분당 {fl['fillers_per_min']}, 보통 4 이하)"
                     + (f", 문장 사이 {LONG_GAP_S:.0f}초 넘는 침묵 {fl['long_gaps']}번" if fl["long_gaps"] else "")
                     + f" → 유창성 {fl['score']}점")
    pr = (speech or {}).get("pron")
    if pr:
        lines.append("[발음 측정, 녹음에서 잰 값] " + ", ".join(
            f"'{v['name']}' 소리 {v['n']}번 중 {v['flagged']}번이 '{v['merged']}'처럼 들림"
            + (f"({', '.join(v['words'][:3])})" if v["words"] else "") for v in pr.values()))
    if fixes:
        lines.append("[사투리 교정, 이미 따로 안내함] " + ", ".join(f"'{f['from']}'→'{f['to']}'" for f in fixes[:12]))
    if missed:
        lines.append("[대본에서 말하지 않은 문장, 이미 따로 안내함] " + " / ".join(m[:60] for m in missed[:6]))
    lines.append("")
    lines.append(f"[평가 기준] {MODE_FOCUS[mode]}")
    rules = [
        "아래 형식의 줄만 출력한다. 머리말, 번호 목록, 마크다운은 쓰지 않는다.",
        "TIP|문장 번호|그 문장에서 그대로 옮긴 고칠 부분(10자 이내)|한 문장 조언 (해요체)",
        "TIP 줄은 가장 중요한 것만 최대 3개. 사투리 교정은 이미 안내했으니 반복하지 않는다.",
        "너는 음성을 직접 듣지 못했다. 발음·억양은 위 말투 분석만 근거로 하고, 근거 없이 표준어 발음이 좋다고 칭찬하지 않는다.",
    ]
    if fl:
        rules.append("말 속도·쉼·군말에 대한 조언은 [말하기 측정] 값만 근거로 한다. 측정값이 보통 범위면 언급하지 않는다.")
    if pr:
        rules.append("모음·자음 발음에 대한 조언은 [발음 측정] 값만 근거로 하고, 들린 횟수가 적으면 언급하지 않는다.")
    if mode != "daily":
        rules.append("SCORE|단어 선택 점수|유창성 점수|정보·사실 점수 (각 0~100 정수. 답변에 직무·기술 정보가 없으면 세 번째는 -)")
    rules.append("SUMMARY|두 문장 이내 총평 (해요체, 잘한 점 하나와 다음에 할 일 하나)")
    if mode == "presentation":
        rules.append("QUESTION|청중이 할 법한 추가 질문 하나")
    lines.append("[출력 형식]")
    lines += [f"- {r}" for r in rules]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


# Rubric scoring (G-Eval, Liu et al. 2023; Prometheus, Kim et al. ICLR 2024): one criterion per prompt with a
# fixed description of every score level, answered with a single digit, and the score taken as the expectation
# over the probabilities of "1".."5" (engine /score) instead of a number the model writes, which LLM judges
# give in a few round values. The levels say what each score means, so a long answer is not a better one
# (verbosity bias, Zheng et al. 2023). Reference answers, Prometheus' other input, are not available here.
RUBRICS = {
    "word": ("단어 선택", "공식적인 자리에 맞는 정확한 어휘와 높임말", [
        "공식적인 자리에 맞지 않는 반말·비속어·유행어가 많다",
        "공식적인 자리에 어울리지 않는 표현이 여러 번 나온다",
        "대체로 무난하지만 구어적이거나 모호한 표현이 눈에 띈다",
        "상황에 맞는 높임말과 정확한 어휘를 쓰고, 어색한 표현이 한두 개뿐이다",
        "상황에 꼭 맞는 정확하고 간결한 어휘를 처음부터 끝까지 쓴다"]),
    "facts": ("정보·사실", "질문에 대한 구체적이고 정확한 정보와 근거", [
        "질문과 관련된 정보가 거의 없거나 틀린 정보가 있다",
        "관련은 있지만 근거나 구체적인 사례가 없다",
        "사례나 근거가 있지만 모호하거나 일부가 부정확하다",
        "구체적인 사례와 근거가 있고 대체로 정확하다",
        "구체적인 사례·근거·결과가 분명하고 정확하며 질문에 바로 답한다"]),
    # scored from the recording (Ma et al. Interspeech 2025); an accent itself is not marked down (Munro &
    # Derwing 1995: accent and comprehensibility are partly independent)
    "delivery": ("전달력", "듣는 사람이 알아듣기 쉬운 발음·속도·목소리와 자신감 (사투리 억양 자체는 감점하지 않는다)", [
        "웅얼거리거나 너무 빠르거나 작아서 알아듣기 어렵다",
        "알아듣기 어려운 부분이 여러 번 있다",
        "대체로 알아들을 수 있지만 발음이 흐리거나 속도·크기가 불안정한 부분이 있다",
        "발음이 또렷하고 속도와 목소리가 안정적이며 자신감이 느껴진다",
        "처음부터 끝까지 또렷하고 자연스러우며 자신감 있게 전달한다"]),
}


def rubric_messages(mode, ctx, sentences, criterion, audio=None):
    """A prompt asking for one digit, 1-5, on one criterion of RUBRICS; with `audio` (base64 WAV clips of the
    answer) the model hears the answer before the rubric."""
    name, what, levels = RUBRICS[criterion]
    lines = [f"[상황] {ctx.get('title', '')}"]
    if ctx.get("question"):
        lines.append(f"[질문] {ctx['question']}")
    lines.append("[답변, 표준어로 옮긴 문장]")
    lines += [f"{i}. {s}" for i, s in enumerate(sentences, 1)]
    lines += ["", f"[평가 항목] {name}: {what}", "[기준표]"] + [f"{i}점: {d}" for i, d in enumerate(levels, 1)]
    lines.append("답변의 길이와 말투(사투리 여부)는 이 항목에서 평가하지 않는다. 기준표에서 가장 맞는 점수를 숫자 하나로만 답한다.")
    system = {"role": "system", "content": "너는 면접·발표 답변을 기준표대로 채점하는 공정한 평가자다."}
    if not audio:
        return [system, {"role": "user", "content": "\n".join(lines)}]
    return [system, {"role": "user", "content": [{"type": "text", "text": "다음은 사용자의 답변 녹음이다."}]
                     + [{"type": "audio", "wav": a} for a in audio] + [{"type": "text", "text": "\n".join(lines)}]}]


def expected_score(probs):
    """0-100 from the probabilities of the scores 1..5: their expectation, rescaled."""
    e = sum((i + 1) * p for i, p in enumerate(probs))
    return round((e - 1) / 4 * 100)


def _int(x):
    m = re.search(r"\d+", x or "")
    return max(0, min(100, int(m.group()))) if m else None


def parse_feedback(text):
    tips, scores, summary, question = [], None, "", ""
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip().strip("-* ").split("|")]
        tag = parts[0].upper()
        if tag == "TIP" and len(parts) >= 4:
            tips.append({"n": _int(parts[1]), "quote": parts[2].strip("'\"‘’“” "), "tip": "|".join(parts[3:])})
        elif tag == "SCORE" and len(parts) >= 3:
            facts = parts[3] if len(parts) > 3 else "-"
            scores = {"word": _int(parts[1]), "fluency": _int(parts[2]),
                      "facts": None if facts.strip() in ("-", "", "없음") else _int(facts)}
        elif tag == "SUMMARY" and len(parts) >= 2:
            summary = "|".join(parts[1:])
        elif tag == "QUESTION" and len(parts) >= 2:
            question = "|".join(parts[1:])
    return {"tips": tips[:3], "scores": scores, "summary": summary, "question": question}


def answer_score(mode, llm, script_match=None, fluency=None):
    """The deck's 답변 점수 (0-100) from the coach's sub-scores; fluency comes from the recording
    (fluency_summary) when it could be measured, else from the coach, who only reads the transcript."""
    if not llm or llm.get("word") is None or (llm.get("fluency") is None and fluency is None):
        return None
    w, f = llm["word"], (fluency if fluency is not None else llm["fluency"])
    if mode == "presentation" and script_match is not None:
        return round(w * 0.25 + f * 0.25 + script_match * 0.5)
    if llm.get("facts") is not None:
        return round(w * 0.25 + f * 0.25 + llm["facts"] * 0.5)
    return round(w * 0.5 + f * 0.5)


def final_score(standard_score, answer):
    """면접 점수 = 표준어 점수 + 답변 점수 (200 points, pass 140), shown out of 100."""
    if answer is None:
        return {"total": standard_score * 2, "points": standard_score, "passed": standard_score >= 70,
                "answer": None, "standard": standard_score}
    total = standard_score + answer
    return {"total": total, "points": round(total / 2), "passed": total >= 140, "answer": answer,
            "standard": standard_score}


# 대본 일치도, sentence by sentence: how much of each script sentence was said anywhere in the talk, the larger
# of BERTScore recall (Zhang et al. ICLR 2020; klue/roberta-base layer 8, which also catches paraphrases) and
# character-bigram recall (exact wording through ASR errors), each mapped from "missed" (0) to "said" (1).
# Calibrated on the benchmark's ASR output against 16-sentence talks (one-off scripts, not included): at 0.5
# it counts 97 % of sentences read out and 70 % of heavily paraphrased ones as said, and 3 % of missed ones.
# Unlike the old whole-text SequenceMatcher ratio, word order and extra words do not lower it.
SCRIPT_ENCODER = "/data/huggingface/hub/models--klue--roberta-base"
SCRIPT_LAYER, BERT_RANGE, BIGRAM_RANGE, SAID = 8, (0.667, 0.74), (0.30, 0.60), 0.5
_encoder = {}


def _sentences(text):
    parts = re.split(r"(?<=[.?!。])\s+|\n+", text)
    return [p.strip() for p in parts if len(re.sub(r"\W", "", p)) >= 4]


def _token_vectors(texts):
    """Layer-8 token vectors (unit length) of each text, special tokens left out; CPU, loaded on first use."""
    import glob
    import threading
    import torch
    with _encoder.setdefault("lock", threading.Lock()):
        if "model" not in _encoder:
            from transformers import AutoModel, AutoTokenizer
            torch.set_num_threads(4)
            path = sorted(glob.glob(os.path.join(SCRIPT_ENCODER, "snapshots", "*")))[-1]
            _encoder["tok"] = AutoTokenizer.from_pretrained(path)
            _encoder["model"] = AutoModel.from_pretrained(path).eval()
        tok, model = _encoder["tok"], _encoder["model"]
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.inference_mode():
                hs = model(**enc, output_hidden_states=True).hidden_states[SCRIPT_LAYER]
            for h, m in zip(hs, enc["attention_mask"]):
                v = h[1:int(m.sum()) - 1]
                out.append(torch.nn.functional.normalize(v, dim=-1))
        return out


def _encoder_vectors(texts):
    """_token_vectors, or None when the encoder cannot be loaded; coverage then uses the bigram recall alone.
    (2026-11-07: transformers 5.13 refuses huggingface-hub 2.2 in this machine's shared environment.)"""
    if _encoder.get("failed"):
        return None
    try:
        return _token_vectors(texts)
    except Exception as e:  # noqa: BLE001  import or version clash, missing snapshot: keep the feature working
        _encoder["failed"] = f"{type(e).__name__}: {e}"[:300]
        print(f"script coverage: BERTScore encoder unavailable, bigram recall only ({_encoder['failed']})", flush=True)
        return None


def _bigrams(text):
    t = re.sub(r"\W", "", text)
    return Counter(t[i:i + 2] for i in range(len(t) - 1))


def script_coverage(script, spoken):
    """{"score": 대본 일치도 0-100 (script sentences weighted by length), "sentences": [{"text", "said"}],
    "missed": [sentences not said]} for the script and the spoken sentences (standard-Korean lines)."""
    sents, spoken = _sentences(script or ""), [s for s in spoken if s and s.strip()]
    if not sents:
        return None
    if not spoken:
        return {"score": 0, "sentences": [{"text": s, "said": 0.0} for s in sents], "missed": sents}
    vecs = _encoder_vectors(sents + spoken)
    if vecs is not None:
        import torch
        talk = torch.cat(vecs[len(sents):])
    heard = _bigrams(" ".join(spoken))
    clip = lambda x, lo, hi: min(1.0, max(0.0, (x - lo) / (hi - lo)))
    rows = []
    for i, s in enumerate(sents):
        v = vecs[i] if vecs is not None else []
        bert = float((v @ talk.T).max(dim=1).values.mean()) if len(v) else 0.0
        g = _bigrams(s)
        bigram = sum(min(n, heard[k]) for k, n in g.items()) / max(1, sum(g.values()))
        rows.append({"text": s, "said": round(max(clip(bert, *BERT_RANGE), clip(bigram, *BIGRAM_RANGE)), 2)})
    weight = [len(re.sub(r"\W", "", r["text"])) for r in rows]
    score = round(100 * sum(w * r["said"] for w, r in zip(weight, rows)) / sum(weight))
    return {"score": score, "sentences": rows, "missed": [r["text"] for r in rows if r["said"] < SAID]}


def script_match(script, spoken):
    """대본 일치도 (0-100) of script_coverage, or None without a script. `spoken`: a list of sentences."""
    cov = script_coverage(script, spoken)
    return cov["score"] if cov else None


def time_tip(mode, seconds):
    lo, hi = {"interview": (40, 90), "presentation": (60, 300)}.get(mode, (0, 10 ** 6))
    if seconds < lo:
        return f"{round(seconds)}초 동안 말했어요. 조금 짧아요. {lo}~{hi}초 정도가 적당해요."
    if seconds > hi:
        return f"{round(seconds)}초 동안 말했어요. 조금 길어요. 핵심만 {hi}초 안에 정리해 보세요."
    return f"{round(seconds)}초 동안 말했어요. 시간 배분이 적당해요."


def reply_messages(persona, scenario, history):
    """Daily-conversation partner. history: [{"role": "ai"|"user", "text"}], user turns in standard form."""
    system = (f"너는 사용자의 대화 상대 '{persona.get('name', '친구')}'이다. {persona.get('desc', '')} "
              f"상황: {scenario}. 표준어 구어체로 1~2문장만 자연스럽게 대답하고, 대화를 이어갈 짧은 질문을 덧붙인다. "
              "사투리, 이모지, 괄호 설명은 쓰지 않는다.")
    turns = []  # chat templates want user/assistant turns alternating, starting with the user
    for h in history[-10:]:
        role = "assistant" if h["role"] == "ai" else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += " " + h["text"]
        else:
            turns.append({"role": role, "content": h["text"]})
    if not turns or turns[0]["role"] == "assistant":  # the partner spoke first
        turns.insert(0, {"role": "user", "content": "(대화를 시작해 줘)"})
    if turns[-1]["role"] != "user":
        turns.append({"role": "user", "content": "(계속 이야기해 줘)"})
    return [{"role": "system", "content": system}] + turns


# Spaced review (Settles & Meeder 2016, Duolingo's half-life regression): the chance of still saying a sentence
# the standard way is p = 2^(-days since practice / h), with half-life h = 2^(θ·x) days over
# x = (1, √(1 + right readings), √(1 + wrong readings)). It falls due when p drops to one half, i.e. after h
# days. θ is set by hand until there are attempts to fit it on (HLR's loss over the attempts table): correct
# readings in a row give 1, 1.7, 2.8, 4.2, 6, 8.5, 11.6, 15.6 days; a miss shortens the next wait.
HLR_THETA = (-2.04, 2.5, -1.5)
MASTERED_DAYS = 14  # a sentence moves to 복습완료 once its half-life reaches two weeks


def half_life(right, wrong):
    t0, t1, t2 = HLR_THETA
    return 2 ** (t0 + t1 * (1 + right) ** 0.5 + t2 * (1 + wrong) ** 0.5)


def review_schedule(right, wrong, practiced, now):
    """{"half_life" (days), "recall" (predicted now), "next_due" (time), "due", "mastered"}; never practised: due."""
    h = half_life(right, wrong)
    if not practiced:
        return {"half_life": round(h, 2), "recall": 0.0, "next_due": now, "due": True, "mastered": False}
    days = max(0.0, (now - practiced) / 86400)
    return {"half_life": round(h, 2), "recall": round(2 ** (-days / h), 3), "next_due": practiced + h * 86400,
            "due": days >= h, "mastered": right > 0 and h >= MASTERED_DAYS}


def review_messages(words, region=None):
    """Personalised review sentences (deck: 개인 맞춤형 복습). Output lines: SENT|<사투리 문장>|<표준어 문장>"""
    pairs = ", ".join(f"'{w['from']}'→'{w['to']}'" for w in words[:8])
    where = f"{region} " if region else ""
    user = (f"사용자가 자주 쓰는 {where}사투리 표현: {pairs}\n"
            "이 표현을 넣은 짧은 일상·면접 문장 3개를 만들어라. 각 문장은 25자 이내.\n"
            "각 줄은 반드시 SENT로 시작한다. 형식: SENT|사투리 문장|같은 뜻의 자연스러운 표준어 문장\n"
            "이 형식의 줄 3개만 출력한다.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_sentences(text):
    """SENT|<사투리>|<표준어> lines; the model sometimes writes another label (일상, 면접) in front."""
    out = []
    for line in text.splitlines():
        parts = [p.strip().strip("\"'“”") for p in line.strip().strip("-* ").split("|")]
        if len(parts) >= 2 and parts[-2] and parts[-1] and parts[-2] != parts[-1] and "사투리" not in parts[-2]:
            out.append({"dialect": parts[-2], "standard": parts[-1]})
    return out[:3]


# ---- audio --------------------------------------------------------------------------------

def preprocess(x, sr=16000):
    """Trim long silence at both ends and bring quiet recordings to a normal level."""
    if len(x) == 0:
        return x, 0.0
    frame = int(0.02 * sr)
    n = len(x) // frame
    if n == 0:
        return x, 0.0
    rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    thr = max(float(np.percentile(rms, 15)) * 3, 0.004)
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return x[:0], 0.0
    a = max(0, voiced[0] - 15) * frame
    b = min(n, voiced[-1] + 16) * frame
    y = x[a:b]
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if 0 < peak < 0.5:
        y = y * min(0.9 / peak, 10.0)
    return y.astype(np.float32), len(voiced) * 0.02


# AI Hub writes a drawn-out one-syllable filler as "그~", "어~", "뭐~", and the model keeps that notation.
FILLER = re.compile(r"(?<!\S)[가-힣]~")
HANGUL = re.compile(r"[가-힣]")
PAUSE_S = 0.25  # a silence at least this long splits a run (a common threshold in L2 fluency research)


# Native reference (accent-id data, 1,195 Seoul-Gyeonggi clips without dialect words, same measures):
# speech rate 4.0-5.5 syllables/s (median 4.8), speaking-time ratio median 0.83, 12 syllables per run, 4.1
# fillers per minute; speakers of the other zones measure about the same, so this scores delivery, not dialect.
# Speech rate, mean length of run and phonation-time ratio predict rated fluency best (Kormos & Denes 2004).
FLUENCY_REF = {"rate": (4.0, 5.5), "phonation": 0.83, "mlr": 12.0, "fillers": 4.0}
LONG_GAP_S = 3.0  # a silence this long between sentences of one answer


def fluency_summary(segments, mode="interview"):
    """Fluency of one answer from the clips' timings (fluency_clip, measured when each clip was transcribed):
    0-100 from speech rate (35 %), syllables per run (25 %), speaking-time ratio (25 %) and fillers (15 %),
    each against the native reference. None for old sessions or under 6 s / 15 syllables of speech."""
    fl = [s["fluency"] for s in segments if isinstance(s.get("fluency"), dict)]
    secs = sum(f["seconds"] for f in fl)
    syl = sum(f["syllables"] for f in fl)
    if secs < 6 or syl < 15:
        return None
    speaking = sum(f["speaking"] for f in fl)
    runs = sum(f["runs"] for f in fl)
    fillers = sum(f["fillers"] for f in fl)
    rate, phon, mlr, fpm = syl / secs, speaking / secs, syl / max(1, runs), fillers / secs * 60
    lo, hi = FLUENCY_REF["rate"]
    clamp = lambda v: max(0.0, min(100.0, v))
    s_rate = 100.0 if lo <= rate <= hi else clamp(100 * (rate - 2.0) / (lo - 2.0)) if rate < lo else max(40.0, 100 - 40 * (rate - hi) / 2.0)
    s_mlr = clamp(100 * (mlr - 3.0) / (FLUENCY_REF["mlr"] - 3.0))
    s_phon = clamp(100 * (phon - 0.45) / (FLUENCY_REF["phonation"] - 0.45))
    s_fill = clamp(100 * (20.0 - fpm) / (20.0 - FLUENCY_REF["fillers"]))
    # long silences between the sentences of one recording (interview / presentation answers)
    gaps = 0
    if mode != "daily":
        timed = sorted((float(s["start"]), float(s.get("duration") or 0)) for s in segments if s.get("start") is not None)
        gaps = sum(1 for (a, d), (b, _) in zip(timed, timed[1:]) if LONG_GAP_S <= b - (a + d) <= 60)
    return {"score": round(0.35 * s_rate + 0.25 * s_mlr + 0.25 * s_phon + 0.15 * s_fill), "rate": round(rate, 2),
            "phonation": round(phon, 2), "mlr": round(mlr, 1), "fillers": fillers, "fillers_per_min": round(fpm, 1),
            "long_gaps": gaps, "seconds": round(secs, 1), "parts": {"rate": round(s_rate), "mlr": round(s_mlr),
            "phonation": round(s_phon), "fillers": round(s_fill)}}


def fluency_clip(x, text, sr=16000):
    """Timing of one clip for the fluency measures: its seconds, the time spent speaking (silences of
    PAUSE_S or more left out), syllables (Hangul characters), runs between such silences, and fillers."""
    syllables = len(HANGUL.findall(text or ""))
    fillers = len(FILLER.findall(text or ""))
    seconds = len(x) / sr
    frame = int(0.02 * sr)
    n = len(x) // frame
    if n < 10:
        return {"seconds": round(seconds, 2), "speaking": round(seconds, 2), "syllables": syllables, "runs": 1, "fillers": fillers}
    rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    rms = np.convolve(rms, np.ones(3) / 3, mode="same")
    lo, hi = np.percentile(rms, 10), np.percentile(rms, 90)
    silent = rms < max(lo + 0.1 * (hi - lo), 0.003)
    # silent stretches inside the clip (not at its ends) of PAUSE_S or more
    min_run = int(round(PAUSE_S / 0.02))
    pauses, run, start = [], 0, None
    for i, sil in enumerate(silent):
        if sil:
            run += 1
            start = i if run == 1 else start
        else:
            if run >= min_run and start > 0:
                pauses.append(run)
            run = 0
    lead = int(np.argmax(~silent)) if (~silent).any() else n
    tail = int(np.argmax(~silent[::-1])) if (~silent).any() else 0
    speaking = max(0.0, (n - lead - tail - sum(pauses)) * 0.02)
    return {"seconds": round(seconds, 2), "speaking": round(speaking, 2), "syllables": syllables,
            "runs": len(pauses) + 1, "fillers": fillers}


def split_audio(x, sr=16000, max_s=18.0, min_s=2.0):
    """Cut a long recording at its quietest moments into clips of at most max_s seconds."""
    if len(x) <= max_s * sr:
        return [x]
    frame = int(0.02 * sr)
    n = len(x) // frame
    rms = np.sqrt(np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    smooth = np.convolve(rms, np.ones(15) / 15, mode="same")
    cuts, start = [], 0
    while (n - start) * 0.02 > max_s:
        lo, hi = start + int(min_s / 0.02), start + int(max_s / 0.02)
        window = smooth[lo:hi] + np.linspace(0.002, 0, hi - lo)  # prefer later pauses: fewer, longer clips
        k = lo + int(np.argmin(window))
        cuts.append(k)
        start = k
    bounds = [0] + cuts + [n]
    return [x[a * frame:(b * frame if b < n else len(x))] for a, b in zip(bounds, bounds[1:]) if (b - a) * 0.02 >= 0.3]


def read_wav(data):
    """PCM WAV bytes -> float32 mono at 16 kHz (None when it is not a PCM WAV)."""
    import io
    import wave
    from math import gcd
    try:
        with wave.open(io.BytesIO(data)) as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    else:
        return None
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        from scipy.signal import resample_poly
        g = gcd(rate, 16000)
        x = resample_poly(x, 16000 // g, rate // g).astype(np.float32)
    return x


def decode_audio(data):
    """Any recording (browser WAV, m4a, webm, mp3 ...) -> float32 mono at 16 kHz, via ffmpeg if needed."""
    import subprocess
    x = read_wav(data) if data[:4] == b"RIFF" else None
    if x is not None:
        return x
    # Uploads are untrusted: only the pipe protocol (no playlists pulling in local files or URLs)
    # and only common audio/video demuxers.
    p = subprocess.run(["ffmpeg", "-v", "error", "-protocol_whitelist", "pipe",
                        "-format_whitelist", "wav,mp3,ogg,matroska,webm,mov,mp4,m4a,flac,aac,amr,caf",
                        "-i", "pipe:0", "-t", "600", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
                       input=data, capture_output=True, timeout=180)
    if p.returncode != 0:
        raise ValueError(p.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(p.stdout, "<i2").astype(np.float32) / 32768


def wav_bytes(x, sr=16000):
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


_STRIP = re.compile(r"[\s.,?!~'\"“”‘’·…-]")


def cer_text(ref, hyp):
    """Character error rate ignoring spaces and punctuation (as in the training project's evaluation)."""
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


def contains_word(sentence, word):
    toks, w = [norm(t) for t in sentence.split()], norm(word).split()
    return bool(w) and any(toks[i:i + len(w)] == w for i in range(len(toks) - len(w) + 1))


# ---- presentation scripts from files ---------------------------------------------------------

SCRIPT_TYPES = "txt, docx, pdf, pptx, hwpx"
_RUN = re.compile(r"<(?:w|a|hp):t(?:\s[^>]*)?>(.*?)</(?:w|a|hp):t>|<(?:w:br|a:br|hp:lineBreak)\b[^>]*/>|<(?:w|hp):tab\b[^>]*/>", re.S)


def _xml_text(xml, para_end):
    """Paragraphs of an Office/HWPX XML part: text runs joined, one line per paragraph."""
    xml = re.sub(r"<a:fld\b.*?</a:fld>", "", xml, flags=re.S)  # slide-number fields of notes pages
    lines = []
    for para in xml.split(para_end):
        parts = []
        for m in _RUN.finditer(para):
            tag = m.group(0)
            parts.append(m.group(1) if m.group(1) is not None else ("\t" if "tab" in tag[:8] else "\n"))
        lines.append(html.unescape(re.sub(r"<[^>]+>", "", "".join(parts))).strip())
    return "\n".join(lines)


def _zip_parts(z, pattern):
    names = [n for n in z.namelist() if re.fullmatch(pattern, n) and z.getinfo(n).file_size < 30_000_000]
    return sorted(names, key=lambda n: [int(x) for x in re.findall(r"\d+", n)] or [0])


# The parts script_text reads (coach.js SCRIPT_PARTS sends only these from a docx, pptx or hwpx).
SCRIPT_PART = re.compile(r"word/document\.xml|ppt/slides/slide\d+\.xml|ppt/notesSlides/notesSlide\d+\.xml"
                         r"|Contents/section\d+\.xml")


def zip_of(parts):
    """A zip of the text parts the browser sent ({zip path: xml}), for script_text."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, xml in list(parts.items())[:2000]:
            if isinstance(xml, str) and SCRIPT_PART.fullmatch(path):
                z.writestr(path, xml)
    return buf.getvalue()


def script_text(data, name=""):
    """The text of an uploaded presentation script. pptx: the speaker notes when there are any (that is
    where a script usually is), else the slides' text. Raises ValueError with a message for the user."""
    ext = os.path.splitext(name.lower())[1]
    if data[:4] == b"%PDF" or ext == ".pdf":
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "script.pdf")
            with open(path, "wb") as f:
                f.write(data)
            try:
                r = subprocess.run(["pdftotext", "-enc", "UTF-8", "-q", path, "-"], capture_output=True, timeout=30)
            except subprocess.TimeoutExpired as e:
                raise ValueError("PDF를 읽는 데 시간이 너무 걸려요.") from e
        text = r.stdout.decode("utf-8", "replace")
    elif data[:2] == b"PK":
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise ValueError("파일이 손상되었어요.") from e
        read = lambda n: z.read(n).decode("utf-8", "replace")
        if "word/document.xml" in z.namelist():
            text = _xml_text(read("word/document.xml"), "</w:p>")
        elif any(n.startswith("ppt/slides/") for n in z.namelist()):
            notes = [_xml_text(read(n), "</a:p>") for n in _zip_parts(z, r"ppt/notesSlides/notesSlide\d+\.xml")]
            slides = [_xml_text(read(n), "</a:p>") for n in _zip_parts(z, r"ppt/slides/slide\d+\.xml")]
            text = "\n\n".join(t for t in (notes if any(t.strip() for t in notes) else slides) if t.strip())
        elif any(n.startswith("Contents/section") for n in z.namelist()):
            text = "\n".join(_xml_text(read(n), "</hp:p>") for n in _zip_parts(z, r"Contents/section\d+\.xml"))
        else:
            raise ValueError(f"{SCRIPT_TYPES} 파일만 올릴 수 있어요.")
    elif ext == ".hwp" or data[:8] == bytes.fromhex("d0cf11e0a1b11ae1"):  # HWP 5 and old Office share this header
        if ext in (".ppt", ".doc"):
            raise ValueError("예전 형식(ppt·doc)은 읽을 수 없어요. pptx·docx나 PDF로 저장해서 올려 주세요.")
        raise ValueError("HWP 파일은 읽을 수 없어요. 한글에서 PDF나 HWPX로 저장해서 올려 주세요.")
    else:
        if data.count(b"\x00") > len(data) // 10 and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise ValueError(f"{SCRIPT_TYPES} 파일만 올릴 수 있어요.")
        for enc in ("utf-8-sig", "utf-16", "cp949"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = data.decode("utf-8", "replace")
    text = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()
