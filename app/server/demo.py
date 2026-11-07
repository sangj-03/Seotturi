"""체험 모드 (demo mode): what the app answers while the Gemma 4 engine is not running.

The site then still works end to end so it can be looked at without the GPU: speech in a session is answered
with an example sentence, the coach's feedback and the conversation partner's lines are fixed examples, and
review sentences come from the same examples. Everything else is real: the recording, the measured fluency, the
scores computed from the example text, accounts, saved sessions, the word book, the community and the voices.
The app labels every such answer "체험 모드". app.py falls back to it only when the engine does not answer at
all (SEOTTURI_DEMO=0 turns it off and restores the old "AI 엔진에 연결하지 못했어요" errors).

The sentences were written for this demo in Gyeongsang dialect (the app's journey map starts in Busan); they are
not taken from the AI Hub data.
"""
from collections import OrderedDict

MODEL = "체험 모드"  # shown where the app names the model that answered

# (said in dialect, the same sentence in standard Korean), in the order a speaker would say them
SENTENCES = {
    "interview": [
        ("저는 부산에서 나고 자란 김서투리입니더.", "저는 부산에서 나고 자란 김서투리입니다."),
        ("학교 댕길 때 동아리 회장을 맡아가 행사를 억수로 마이 했심더.", "학교 다닐 때 동아리 회장을 맡아서 행사를 정말 많이 했습니다."),
        ("그때 사람들하고 같이 일하는 기 제일 재밌었어예.", "그때 사람들하고 같이 일하는 게 제일 재미있었어요."),
        ("이 회사에 들어오면 고객한테 도움 되는 서비스를 만들고 싶심더.", "이 회사에 들어오면 고객한테 도움 되는 서비스를 만들고 싶습니다."),
        ("부족한 기 있으면 끝까지 배아가 꼭 해내겠심더.", "부족한 게 있으면 끝까지 배워서 꼭 해내겠습니다."),
    ],
    "presentation": [
        ("오늘 지가 발표할 주제는 사투리 교정 앱입니더.", "오늘 제가 발표할 주제는 사투리 교정 앱입니다."),
        ("요즘 취준생들 사이에서 사투리 때문에 걱정하는 사람이 억수로 많아예.", "요즘 취준생들 사이에서 사투리 때문에 걱정하는 사람이 정말 많아요."),
        ("그래가 저희는 목소리를 듣고 바로 표준어로 바까 주는 기능을 만들었심더.", "그래서 저희는 목소리를 듣고 바로 표준어로 바꿔 주는 기능을 만들었습니다."),
        ("마지막으로 앞으로 할 일을 말씀드리겠심더.", "마지막으로 앞으로 할 일을 말씀드리겠습니다."),
    ],
    "daily": [
        ("니 밥 뭇나?", "너 밥 먹었어?"),
        ("어제 친구들하고 늦게까지 놀다가 왔다 아이가.", "어제 친구들하고 늦게까지 놀다가 왔잖아."),
        ("와 이래 덥노, 아이스 아메리카노 한 잔 주이소.", "왜 이렇게 덥지, 아이스 아메리카노 한 잔 주세요."),
        ("거기 가려면 어데로 가야 되는교?", "거기 가려면 어디로 가야 되나요?"),
        ("내일 시간 되믄 같이 영화 보러 갈래?", "내일 시간 되면 같이 영화 보러 갈래?"),
    ],
}

TIPS = {
    "interview": ["첫 문장에서 지원 동기를 한 문장으로 먼저 말하면 답변이 더 또렷해져요.",
                  "마지막 문장에 입사 후 하고 싶은 일을 구체적으로 덧붙여 보세요."],
    "presentation": ["주제를 말한 뒤 발표 순서를 한 문장으로 알려 주면 청중이 따라오기 쉬워요.",
                     "마무리에서 핵심을 한 번 더 정리해 주면 기억에 남아요."],
    "daily": ["상대의 말에 짧게 반응한 뒤 질문을 하나 덧붙이면 대화가 자연스럽게 이어져요.",
              "문장 끝을 부드럽게 맺으면 더 친근하게 들려요."],
}

REPLIES = {  # by the situation's name (coach.js sends it as the scenario)
    "카페": ["네, 바로 준비해 드릴게요. 사이즈는 어떤 걸로 하시겠어요?", "드시고 가세요, 아니면 포장해 드릴까요?",
           "결제는 카드로 도와드릴까요?", "주문하신 음료 나왔습니다. 맛있게 드세요!"],
    "길": ["아, 거기요? 이 길로 쭉 가다가 두 번째 사거리에서 왼쪽으로 가시면 돼요.", "걸어서 10분쯤 걸려요. 혹시 지하철 타실 거예요?",
          "건너편 정류장에서 버스를 타시면 금방이에요.", "조심히 가세요!"],
    "직장": ["오늘 회의 자료는 다 준비됐어요?", "좋네요. 그럼 오후에 같이 한 번 더 볼까요?",
           "점심은 뭐 먹을까요? 근처에 새로 생긴 곳이 있대요.", "수고 많았어요. 내일 또 이야기해요."],
    "친구": ["진짜? 그래서 어떻게 됐어?", "와, 재미있었겠다. 누구랑 갔어?", "나도 가 보고 싶다. 다음에 같이 가자!",
           "좋아, 그럼 주말에 연락할게."],
}

_turn = OrderedDict()  # (user, mode) -> the next example sentence


def asr(uid="", mode="", say=None):
    """One clip 'transcribed': the next example sentence of the mode, in the engine's /asr answer format.
    say: the sentence to hear instead (a review sentence read out, which the demo takes as read correctly)."""
    if say:
        dialect = standard = say
    else:
        rows = SENTENCES.get(mode) or SENTENCES["interview"]
        key = (uid, mode)
        i = _turn.pop(key, 0)
        _turn[key] = i + 1
        while len(_turn) > 5000:
            _turn.popitem(last=False)
        dialect, standard = rows[i % len(rows)]
    return {"dialect": dialect, "standard": standard, "raw": f"방언: {dialect}\n표준어: {standard}", "model": MODEL,
            "format": "demo", "seconds": 0.0, "wait": 0.0, "demo": True}


def _quote(sentence, limit=10):
    """The first words of a sentence, at most `limit` characters (the coach's TIP quotes are that short)."""
    out = ""
    for w in sentence.split():
        if len(out) + len(w) + (1 if out else 0) > limit:
            break
        out = f"{out} {w}".strip()
    return out or sentence[:limit]


def feedback_text(mode, sentences, speech):
    """The coach's answer in the format analysis.parse_feedback reads (TIP / SCORE / SUMMARY / QUESTION lines)."""
    mode = mode if mode in TIPS else "interview"
    lines = []
    if sentences:
        picks = [1] if len(sentences) == 1 else [1, len(sentences)]
        for tip, n in zip(TIPS[mode], picks):
            lines.append(f"TIP|{n}|{_quote(sentences[n - 1])}|{tip}")
    speech = speech or {}
    if mode != "daily":
        word = max(50, min(95, round(92 - 0.6 * float(speech.get("dialect_ratio") or 0))))
        fluency = (speech.get("fluency") or {}).get("score")
        lines.append(f"SCORE|{word}|{fluency if fluency is not None else 80}|-")
    changed = speech.get("changed") or 0
    nxt = (f"다음에는 표시된 사투리 표현 {changed}개를 표준어로 바꿔 말해 보세요." if changed
           else "다음에는 문장 끝을 조금 더 또렷하게 맺어 보세요.")
    lines.append(f"SUMMARY|끝까지 이어서 말한 점이 좋았어요. {nxt}")
    if mode == "presentation":
        lines.append("QUESTION|이 기능은 경상도 말고 다른 지역 사투리도 알아들을 수 있나요?")
    return "\n".join(lines)


def chunks(text, size=12):
    """The feedback in small pieces, so the app shows it being written as it does with the engine."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def reply(scenario, history):
    """The conversation partner's next line, by situation."""
    key = next((k for k in REPLIES if k in (scenario or "")), "친구")
    n = sum(1 for h in history or [] if h.get("role") == "ai")
    lines = REPLIES[key]
    return lines[(n - 1) % len(lines)] if n else lines[0]


def review_sentences(words):
    """Three review sentences: examples that hold the user's own dialect words first, then the rest."""
    pool = [p for rows in SENTENCES.values() for p in rows]
    picked = []
    for w in words:
        for d, s in pool:
            if w.get("from") and w["from"] in d and (d, s) not in picked:
                picked.append((d, s))
                break
    for p in pool:
        if p not in picked:
            picked.append(p)
    return [{"dialect": d, "standard": s} for d, s in picked[:3]]
