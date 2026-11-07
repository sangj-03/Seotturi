"""Write the eye/mouth boxes of the persona scenes into scripts/alive_frames.py (a marked block in SPECS).

    python specs_block.py id id ...        # recomputes the boxes from web/assets/personas/<id>.jpg
"""
import json, re, sys
from finalize import spec
ALIVE = "/data/seotturi/seotturi-web/scripts/alive_frames.py"
WEB = "/data/seotturi/seotturi-web/web/assets/"
BEGIN = "    # >>> personas (boxes from YuNet landmarks: /data/seotturi/img-gen/specs_block.py)\n"
END = "    # <<< personas\n"
store = "/data/seotturi/img-gen/specs.json"
try:
    specs = json.load(open(store))
except FileNotFoundError:
    specs = {}
from PIL import Image
for pid in sys.argv[1:]:
    path = f"{WEB}personas/{pid}.jpg"
    w = Image.open(path).width
    specs[f"personas/{pid}.jpg"] = spec(path, talker_x=w / 2)
json.dump(specs, open(store, "w"), indent=1)
# Hand fixes (overrides.json): {"<id>": {"boxes": [[x0, y0, x1, y1], [..]], "mouth": null | [..]}} for the talking face.
# Eyes behind glasses need boxes inside the lenses; an open smile with teeth gets no mouth (talk frames look doubled).
try:
    fixes = json.load(open("/data/seotturi/img-gen/overrides.json"))
except FileNotFoundError:
    fixes = {}
for pid, fix in fixes.items():
    name = f"personas/{pid}.jpg"
    if "faces" in fix:  # a whole spec by hand (the mascot's dot eyes, faces found in another image)
        specs[name] = {"faces": fix["faces"]}
    elif specs.get(name):
        face = dict(specs[name]["faces"][0])
        if "boxes" in fix:
            face["boxes"] = fix["boxes"]
        if "mouth" in fix:
            face.pop("mouth", None)
            if fix["mouth"]:
                face["mouth"] = fix["mouth"]
        specs[name] = {"faces": [face] + specs[name]["faces"][1:]}
lines = []
for name in sorted(n for n in specs if specs[n]):  # no face found and no hand spec: no frames
    faces = []
    for f in specs[name]["faces"]:
        parts = [f'"eyes": "{f["eyes"]}"', f'"boxes": [{", ".join(str(tuple(b)) for b in f["boxes"])}]']
        if "mouth" in f:
            parts.append(f'"mouth": {tuple(f["mouth"])}')
        faces.append("{" + ", ".join(parts) + "}")
    lines.append(f'    "{name}": {{"faces": [{", ".join(faces)}]}},\n')
src = open(ALIVE, encoding="utf-8").read()
block = BEGIN + "".join(lines) + END
if BEGIN in src:
    src = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda m: block, src, flags=re.S)
else:
    anchor = "\n}\n\n# A raised hand that waves"
    assert src.count(anchor) == 1
    src = src.replace(anchor, "\n" + block + "}\n\n# A raised hand that waves")
open(ALIVE, "w", encoding="utf-8").write(src)
print(f"{len(specs)} persona scenes in SPECS")
