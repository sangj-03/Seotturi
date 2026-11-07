"""Download 우리말샘 dialect headwords (국립국어원, opendict.korean.go.kr, CC BY-SA 2.0 KR) for the word book.

    ../.venv/bin/python -I scripts/fetch_opendict.py        (API key in ~/.opendict_key)

Every 지역어 (dialect) headword of the word book's regions: the open API has no wildcard, so headwords are
searched by first syllable - the 2,350 common syllables of KS X 1001 and the first syllables of the AI Hub
dialect forms (data/dialect_dict*.json) - with method=start, type3=dialect and each region's codes. Each
(region, syllable) result is cached in data/opendict/, so a rerun only asks for what is missing (about
12,000 calls the first time, three at a time). Writes data/opendict_dialect.json, {region: {headword:
[[standard word or None, part of speech, definition], ...]}}, which build_dict.py turns into the word book.
"""

import concurrent.futures
import glob
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "opendict")
OUT = os.path.join(DATA, "opendict_dialect.json")
API = "https://opendict.korean.go.kr/api/search"
# 우리말샘 region codes: 1 강원, 3 경남, 4 경북, 5 경상, 6 전남, 7 전라, 8 전북, 9 제주, 10 충남, 11 충북, 12 충청
REGIONS = {"강원": "1", "경상": "3,4,5", "전라": "6,7,8", "제주": "9", "충청": "10,11,12"}
SYLLABLES = [bytes([a, b]).decode("euc_kr") for a in range(0xB0, 0xC9) for b in range(0xA1, 0xFF)]  # KS X 1001
DEF = re.compile(r"[‘'“\"]([^’'”\"]+)[’'”\"]\s*의?\s*방언")  # ‘사람’의 방언 (sometimes ‘갇히다’ 방언)
CLEAN = re.compile(r"[-^\d\s·]")
WORKERS = 3
stop = threading.Event()


def search(key, q, region, start):
    """One page (up to 100) of dialect headwords of the region starting with q."""
    url = API + "?" + urllib.parse.urlencode({"key": key, "q": q, "req_type": "json", "advanced": "y", "method": "start",
                                              "type3": "dialect", "region": region, "num": 100, "start": start})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"), strict=False)  # some definitions hold control characters
            if "error" in d:
                print(f"우리말샘 API error: {d['error']}", flush=True)
                stop.set()
                return None
            return d["channel"]
        except (OSError, ValueError):
            time.sleep(3 * (attempt + 1))
    print(f"skipped {q} (region {region}, from {start})", flush=True)
    return {"item": [], "total": 0}


def fetch(key, region, codes, s):
    """All pages for one (region, first syllable), cached; False when the API refused."""
    path = os.path.join(CACHE, f"{region}_{ord(s):05x}.json")
    if os.path.exists(path):
        return True
    items, start, total = [], 1, 0
    while not stop.is_set():
        ch = search(key, s, codes, start)
        if ch is None:
            return False
        time.sleep(0.3)
        items += ch.get("item", [])
        total = int(ch.get("total", 0))
        if start + 100 > min(total, 1000):
            break
        start += 100
    if stop.is_set():
        return False
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"q": s, "total": total, "items": items}, f, ensure_ascii=False)
    os.replace(tmp, path)
    return True


def main():
    key = open(os.path.expanduser("~/.opendict_key")).read().strip()
    entries = []
    for name in ("dialect_dict.json", "dialect_dict.removed.json"):
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            entries += json.load(open(path, encoding="utf-8"))
    os.makedirs(CACHE, exist_ok=True)
    jobs = []
    for region, codes in REGIONS.items():
        firsts = set(SYLLABLES) | {e["dialect"][0] for e in entries if e["region"] == region and "가" <= e["dialect"][0] <= "힣"}
        jobs += [(region, codes, s) for s in sorted(firsts)
                 if not os.path.exists(os.path.join(CACHE, f"{region}_{ord(s):05x}.json"))]
    print(f"{len(jobs)} (region, syllable) searches to do", flush=True)
    t0, done = time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for ok in pool.map(lambda j: fetch(key, *j), jobs):
            done += 1
            if done % 250 == 0:
                print(f"{done}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    if stop.is_set():
        sys.exit("stopped by the API; run again later to continue")
    out = {}
    for path in glob.glob(os.path.join(CACHE, "*.json")):
        region = os.path.basename(path).split("_")[0]
        for it in json.load(open(path, encoding="utf-8"))["items"]:
            word = CLEAN.sub("", it["word"])
            for s in it["sense"]:
                m = DEF.search(s["definition"])
                row = [CLEAN.sub("", m.group(1)) if m else None, s.get("pos", ""), s["definition"]]
                rows = out.setdefault(region, {}).setdefault(word, [])
                if row not in rows:
                    rows.append(row)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, OUT)
    print({r: len(v) for r, v in out.items()}, "headwords ->", OUT)


if __name__ == "__main__":
    main()
