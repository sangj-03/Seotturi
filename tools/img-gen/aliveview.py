"""Original vs. blink2 / talk2 frames of persona scenes, cropped around the face: python aliveview.py out.png id id ..."""
import os
import json, sys
from PIL import Image, ImageDraw
A = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "app", "web", "assets") + "/"
man = json.load(open(A + "alive/manifest.json"))["images"]
out, ids = sys.argv[1], sys.argv[2:]
tiles = []
for pid in ids:
    name = pid if "." in pid else f"personas/{pid}.jpg"
    e = man[name]
    base = Image.open(A + name).convert("RGB")
    frames = [("orig", base)]
    for label, spec in (("blink1", e["blink"][0][0]), ("blink2", e["blink"][0][1])) + ((("talk2", e["talk"][1]),) if e["talk"] else ()):
        f = base.copy()
        patch = Image.open(A + "alive/" + spec[0]).convert("RGBA")
        f.paste(patch, (spec[1], spec[2]), patch)
        frames.append((label, f))
    xs = [s[1] for s in e["blink"][0]] + [s[1] for s in e["talk"]]
    ys = [s[2] for s in e["blink"][0]] + [s[2] for s in e["talk"]]
    x0, y0 = min(xs) - 40, min(ys) - 40
    x1 = x0 + 330; y1 = y0 + 260
    row = Image.new("RGB", (len(frames) * 334, 280), "white")
    d = ImageDraw.Draw(row)
    for i, (label, f) in enumerate(frames):
        row.paste(f.crop((x0, y0, x1, y1)), (i * 334, 20)); d.text((i * 334 + 4, 4), f"{pid} {label}", fill="black")
    tiles.append(row)
W = max(t.width for t in tiles)
sheet = Image.new("RGB", (W, 280 * len(tiles)), (230, 230, 230))
for i, t in enumerate(tiles):
    sheet.paste(t, (0, i * 280))
sheet.save(out); print(out, sheet.size)
