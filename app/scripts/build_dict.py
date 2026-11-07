"""Build the 단어 도감 (dialect word book): every 우리말샘 dialect word, with how AI Hub speakers say it.

    ../.venv/bin/python -I scripts/build_dict.py

The entries are the 지역어 headwords of 국립국어원 우리말샘 for 강원, 경상, 전라, 제주 and 충청
(data/opendict_dialect.json from scripts/fetch_opendict.py), one per headword and standard word (‘사람’의
방언 -> 사램 / 사람). The AI Hub dialect recordings below give each one how often the region's speakers said
it and an example sentence of theirs; words they never said keep the dictionary definition instead (all of
전라, whose AI Hub labels have no standard forms). A sense is left out only when all of its AI Hub forms
failed the Seoul comparison below (쫌 ‘좀’, 걍 ‘그냥’: said everywhere, though the dictionary lists them).

Reads the label archives of AI Hub's 한국어 방언 발화 datasets as downloaded to /data/aihub (118 강원,
119 경상, 121 제주, 122 충청; Training and Validation, about 9.4 million utterances; 전라 has no standard
forms in its labels). Each utterance comes with the annotators' word-level marks (eojeolList: the 어절 as
said, its standard form, isDialect), so a dialect word is taken from them instead of aligning the two
transcripts. Only speakers born in the dataset's region count (a Seoul-born speaker in the Gyeongsang set
does not), and a pair needs MIN_COUNT occurrences.

The annotators also mark casual speech that is not regional (쫌, 걍, 이케, 그니까, 머...), so a pair is kept
only when it passes two tests. First, 국립국어원 우리말샘 lists it as a dialect word of that region
(data/opendict_dialect.json, from scripts/fetch_opendict.py; 지역어 headwords and their 「‘표준어’의 방언」
definitions). A dialect form matches
a headword when it is the headword, the headword plus a particle (사램한테 - 사램 ‘사람’), or, for a verb or
adjective, its stem plus an ending (묵고 - 묵다 ‘먹다’), and the pair's standard form starts with the same
standard word. Grammatical endings (-수다, -여) and forms the dictionary does not list are left out.
Second, since the dictionary also lists forms said everywhere (쫌 ‘좀’, 걍 ‘그냥’), the pair must be regional
in speech: a second pass compares it with speakers born and raised in 서울·경기·인천 who appear in the same
recordings (about 72,000 utterances, 560,000 어절). For a region's pair, only such speakers recorded in the
other regions' datasets count (in the Jeju recordings they pick up 미깡, 간 'went'). A pair is removed when
  - these speakers' 어절 are annotated with the same pair at least CASUAL_MIN times and the region's speakers
    use the pair less than REF_RATIO times as often (per 어절): 쫌→조금, 이케→이렇게, 그니까→그러니까 are
    marked as dialect for Seoul speakers too; or
  - they say the form with that meaning at least REF_MIN times and use the form itself (however annotated)
    at a rate within REF_RATIO of the region's: 근데 - a common Seoul word that some annotators changed.
Fewer uses are no evidence (우에→위에 by three Seoul speakers, 집이→집에 by one), and neither are uses from
one dataset only: Seoul speakers who now live in a region pick up its forms (하는디, 같은디 only by those
living in Jeju), while common Seoul forms turn up in several regions' recordings (쫌, 이케), whether or not
that dataset's annotators changed them (the Jeju annotators leave 쫌 as it is).
A form that Seoul speakers say only as another word is kept (아이가 'the child' vs 경상 아이가 '않니'). Once a
form is removed on the first ground (Seoul speakers' own pairs) for at least SAME_FORM of its uses in the
region, its other pairs there go too (충청 쫌→좀 with 쫌→조금); not for a minor sense (제주 핸→한 does not
take 핸→했어), nor on the second ground, where the form may be another word for them.
Forms too rare for that comparison are still removed when they differ from the standard form only in ways
common to casual speech everywhere (casual_variant): an ㄹ before 려 (키울려고), a tensed consonant (짜르기도,
쎄진, 꼬추; not a final 고 -> 꼬, the Gyeongsang ending of 뭐꼬, 한다꼬) or 뀌었 -> 꼈 (바꼈다는).
Writes data/dialect_dict.json: the kept pairs per region, most frequent first, each with one short example
sentence (and the 우리말샘 headword and definition); dialect_dict.removed.json lists what was left out.

Until 2026-10-08 the word book came from the 580,000 training utterances, word-aligned with difflib, with
the 700 most frequent pairs per region (2,800 entries).
"""

import glob
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict

AIHUB = "/data/aihub"
DATASETS = {"118": "강원", "119": "경상", "121": "제주", "122": "충청"}
BORN = {"강원": {"강원"}, "경상": {"부산", "대구", "울산", "경남", "경북"}, "제주": {"제주"},
        "충청": {"대전", "세종", "충남", "충북"}}
STANDARD = {"서울", "경기", "인천"}  # born and mostly lived here: the reference for standard speech
KNOWN = set().union(*BORN.values()) | STANDARD | {"전남", "전북", "광주"}
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(DATA, "dialect_dict.json")
OPENDICT = os.path.join(DATA, "opendict_dialect.json")
MIN_COUNT = 3
REF_MIN, CASUAL_MIN, REF_RATIO, SAME_FORM = 2, 5, 5.0, 0.3

PUNCT = re.compile(r"[.,?!~…·\"'“”‘’()\[\]{}<>/:;-]")
HANGUL = re.compile(r"^[가-힣]+( [가-힣]+)*$")


CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
        "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
TENSE = {("ㄱ", "ㄲ"), ("ㄷ", "ㄸ"), ("ㅂ", "ㅃ"), ("ㅅ", "ㅆ"), ("ㅈ", "ㅉ")}


def jamo(text):
    out = []
    for ch in text:
        k = ord(ch) - 0xAC00
        out += [CHO[k // 588], JUNG[k % 588 // 28]] + ([JONG[k % 28]] if k % 28 else []) if 0 <= k < 11172 else [ch]
    return out


def casual_variant(d, s):
    """Does the dialect form differ from the standard one only as casual speech does everywhere?"""
    if s.replace("뀌었", "꼈") == d or s.replace("뀌어", "껴") == d:  # 바뀌었다 -> 바꼈다
        return True
    a, b = jamo(d), jamo(s)
    if len(a) == len(b):  # only tensed consonants: 자르다 -> 짜르다
        diff = [(y, x) for x, y in zip(a, b) if x != y]
        if d.endswith("꼬") and s.endswith("고") and jamo(d[:-1]) == jamo(s[:-1]):
            return False  # 뭐꼬, 한다꼬: a regional ending, not casual tensing
        return bool(diff) and all(p in TENSE for p in diff)
    if len(a) == len(b) + 1:  # one ㄹ more before 려/러/라: 키우려고 -> 키울려고
        for i, x in enumerate(a):
            if x == "ㄹ" and a[:i] + a[i + 1:] == b and a[i + 1:i + 2] == ["ㄹ"]:
                return True
    return False


# what may follow a headword noun, and a one-syllable verb stem (a longer rest is likely another word:
# 뚜꺼운 is not 뚜다 ‘두다’ + an ending)
PARTICLES = {"이", "가", "을", "를", "은", "는", "에", "에서", "한테", "께", "도", "만", "로", "으로", "랑", "이랑", "하고",
             "까지", "부터", "요", "이요", "야", "이야", "의", "처럼", "보다", "마다", "두", "루", "으루", "하구", "꺼정",
             "까정", "보담", "에다", "이나", "나", "든지", "라도", "이라도", "밖에", "서", "더러", "보고", "이고", "고",
             "이지", "지", "이다", "다", "인데", "인디", "이여", "여"}
ENDINGS = {"고", "구", "어", "아", "여", "서", "는", "은", "을", "지", "다", "면", "믄", "니", "게", "기", "자", "네", "대",
           "더라", "드라", "었", "았", "였", "겠", "것", "거나", "다가", "니까", "이까", "냐", "요", "어요", "아요", "고요",
           "는데", "는디", "었어", "았어", "었다", "았다", "어서", "아서", "으면", "으믄", "을까", "까", "라", "래", "랴"}


def listed(heads, d, s):
    """(headword, definition) of 우리말샘 that makes dialect form d (standard s) a listed dialect word, or None.
    heads: the region's {headword: [[standard word, part of speech, definition], ...]}."""
    s = s.replace(" ", "")
    for k in range(len(d), 0, -1):
        h, rest = d[:k], d[k:]
        if rest == "" or rest in PARTICLES:
            for std, _, definition in heads.get(h, []):
                if std is None and rest == "":
                    return h, definition
                if std and (s == std or (s.startswith(std) and (s[len(std):] in PARTICLES or s[len(std):] == rest))):
                    return h, definition
        for std, _, definition in heads.get(h + "다", []):  # a verb or adjective: dialect stem + an ending
            if not (std and std.endswith("다") and len(std) > 1 and s.startswith(std[:-1])):
                continue
            if len(h) >= 2 or rest in ENDINGS or (rest[:1] in ENDINGS and len(rest) <= 3):
                return h + "다", definition
    return None


def save(out, removed, checks=()):
    os.makedirs(DATA, exist_ok=True)
    for path, data in ((OUT, out), (OUT[:-len(".json")] + ".removed.json", removed),
                       (OUT[:-len(".json")] + ".checks.json", list(checks))):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    per, gone = Counter(e["region"] for e in out), Counter(e["region"] for e in removed)
    print(f"kept {len(out):,} {dict(per)}, left out {len(removed):,} {dict(gone)}")
    for region in per:
        print(region, "kept", [f"{e['dialect']}→{e['standard']}" for e in out if e["region"] == region][:12])


def norm(tok):
    return PUNCT.sub("", tok or "").strip()


def utterances():
    """(region, speaker group, utterance) for every labelled utterance; group is "native", "standard" or None."""
    for ds, region in DATASETS.items():
        archives = sorted(glob.glob(f"{AIHUB}/{ds}/*/01.데이터/*/*라벨링데이터*/*.zip"))
        if not archives:
            sys.exit(f"no label archives under {AIHUB}/{ds}")
        for path in archives:
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if not name.endswith(".json"):
                        continue
                    try:
                        d = json.loads(zf.read(name).decode("utf-8-sig"))
                    except ValueError:
                        continue
                    speakers = {s.get("id"): s for s in d.get("speaker") or []}
                    for u in d.get("utterance") or []:
                        s = speakers.get(u.get("speaker_id")) or {}
                        born, lived = s.get("birthplace") or "", s.get("principal_residence") or ""
                        if born in BORN[region] or born not in KNOWN:
                            group = "native"
                        elif born in STANDARD and lived in STANDARD:
                            group = "standard"
                        else:
                            group = None
                        yield region, group, u
            print(f"  {region} {os.path.basename(path)} read", flush=True)


def words(u):
    """The utterance's 어절 (normalised) and their 2- and 3-어절 sequences, for counting a form's use."""
    w = [norm(e.get("eojeol")) for e in u.get("eojeolList") or []]
    w = [x for x in w if x]
    return w + [" ".join(w[i:i + 2]) for i in range(len(w) - 1)] + [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]


def main():
    t0 = time.time()
    count = defaultdict(Counter)
    example = {}
    print("pass 1: dialect pairs marked by the annotators")
    for region, group, u in utterances():
        if group != "native":
            continue
        for e in u.get("eojeolList") or []:
            if not e.get("isDialect"):
                continue
            a, b = norm(e.get("eojeol")), norm(e.get("standard"))
            if not a or not b or a.replace(" ", "") == b.replace(" ", ""):
                continue
            if not HANGUL.match(a) or not HANGUL.match(b):
                continue
            key = (a, b)
            count[region][key] += 1
            ex = (u.get("dialect_form") or "", u.get("standard_form") or "")
            old = example.get((region, key))
            if old is None or (8 <= len(ex[0]) < len(old[0])) or len(old[0]) < 8 <= len(ex[0]) <= 40:
                example[(region, key)] = ex
    pairs = {r: [(k, n) for k, n in c.most_common() if n >= MIN_COUNT] for r, c in count.items()}
    forms = {r: {a for (a, _), _ in ps} for r, ps in pairs.items()}
    every = set().union(*forms.values())
    print(f"pass 2: how often native and standard speakers use the {len(every):,} forms ({time.time() - t0:.0f} s)")
    used = defaultdict(Counter)        # region -> form -> uses by its native speakers
    ref_form = defaultdict(Counter)    # dataset region -> form -> uses by standard speakers
    ref_pair = defaultdict(Counter)    # dataset region -> (form, annotated standard) -> uses by standard speakers
    tokens, ref_tokens = Counter(), Counter()
    for region, group, u in utterances():
        if group is None:
            continue
        n = len(u.get("eojeolList") or [])
        if group == "native":
            tokens[region] += n
            for w in words(u):
                if w in forms[region]:
                    used[region][w] += 1
            continue
        ref_tokens[region] += n
        for w in words(u):
            if w in every:
                ref_form[region][w] += 1
        for e in u.get("eojeolList") or []:
            a = norm(e.get("eojeol"))
            if a in every:
                ref_pair[region][(a, norm(e.get("standard")))] += 1
    out, removed, checks = [], [], []
    for region, ps in sorted(pairs.items()):
        others = [r for r in DATASETS.values() if r != region]
        n_ref = max(1, sum(ref_tokens[r] for r in others))
        for (d, s), n in ps:
            rp = sum(ref_pair[r][(d, s)] for r in others)
            spread = sum(ref_form[r][d] > 0 for r in others) >= 2
            rf = sum(ref_form[r][d] for r in others)
            pair_ratio = (n / max(1, tokens[region])) / (rp / n_ref) if rp else None
            form_ratio = (used[region][d] / max(1, tokens[region])) / (rf / n_ref) if rf else None
            casual = (spread and rp >= CASUAL_MIN and pair_ratio < REF_RATIO) or casual_variant(d, s)
            common = spread and rp >= REF_MIN and rf >= REF_MIN and form_ratio < REF_RATIO
            ex_d, ex_s = example[(region, (d, s))]
            entry = {"region": region, "dialect": d, "standard": s, "count": n,
                     "example_dialect": ex_d, "example_standard": ex_s}
            if casual or common:
                removed.append({**entry, "why": ("variant" if casual_variant(d, s) else "casual") if casual else "common",
                                "standard_pair_uses": rp, "standard_form_uses": rf,
                                "ratio": round(pair_ratio, 2) if casual and pair_ratio else form_ratio and round(form_ratio, 2)})
            else:
                out.append(entry)
                if sum(e["region"] == region for e in out) <= 300:  # for checking the most visible entries
                    checks.append({"region": region, "pair": f"{d}→{s}", "count": n, "standard_pair_uses": rp,
                                   "standard_form_uses": rf, "pair_ratio": pair_ratio and round(pair_ratio, 2),
                                   "form_ratio": form_ratio and round(form_ratio, 2)})
    uses = Counter()
    for region, ps in pairs.items():
        for (d, _), n in ps:
            uses[(region, d)] += n
    gone_forms = {(e["region"], e["dialect"]) for e in removed
                  if e["why"] == "casual" and e["count"] >= SAME_FORM * uses[(e["region"], e["dialect"])]}
    removed += [{**e, "why": "same form"} for e in out if (e["region"], e["dialect"]) in gone_forms]
    out = [e for e in out if (e["region"], e["dialect"]) not in gone_forms]
    print(f"standard reference: {sum(ref_tokens.values()):,} 어절 {dict(ref_tokens)}")
    if not os.path.exists(OPENDICT):
        print(f"no {OPENDICT}: the word book is the AI Hub forms only (run scripts/fetch_opendict.py)")
        save(out, removed, checks)
        return
    with open(OPENDICT, encoding="utf-8") as f:
        dictionary = json.load(f)
    save(from_dictionary(dictionary, out, removed), removed, checks)


def from_dictionary(dictionary, kept, removed):
    """One entry per region, 우리말샘 headword and standard word, with the AI Hub speakers' uses of it (count,
    the example of its most used form); senses whose AI Hub forms all failed the Seoul comparison are left out."""
    spoken = {}
    for ok, entries in ((True, kept), (False, removed)):
        for e in entries:
            hit = listed(dictionary.get(e["region"], {}), e["dialect"], e["standard"])
            if not hit:
                continue
            sp = spoken.setdefault((e["region"], hit[0], hit[1]), {"kept": False, "count": 0, "best": None})
            if ok:
                sp["kept"] = True
                sp["count"] += e["count"]
                if sp["best"] is None or e["count"] > sp["best"]["count"]:
                    sp["best"] = e
    out, seen = [], set()
    for region, heads in sorted(dictionary.items()):
        for word, senses in heads.items():
            for std, pos, definition in senses:
                sp = spoken.get((region, word, definition))
                if sp and not sp["kept"]:
                    continue
                ending = pos == "어미"
                shown = f"-{word}" if ending else word
                ex = sp["best"] if sp else None
                # an ending's standard word is an ending too; a definition that is not ‘...’의 방언 takes the
                # standard form the AI Hub annotators gave (충청 겨 -> 거야)
                standard = (f"-{std}" if ending else std) if std else (ex["standard"] if ex else "")
                key = (region, shown, standard or definition)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"region": region, "dialect": shown, "standard": standard, "count": sp["count"] if sp else 0,
                            "example_dialect": ex["example_dialect"] if ex else "",
                            "example_standard": ex["example_standard"] if ex else "",
                            "example_form": ex["dialect"] if ex else "", "dict_definition": definition})
    out.sort(key=lambda e: (e["region"], -e["count"], e["dialect"].startswith("-"), e["dialect"]))  # endings after words
    return out


if __name__ == "__main__":
    main()
