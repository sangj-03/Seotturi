"""Zoomed face crop with a coordinate grid and the current eye/mouth boxes: python boxview.py id out.png"""
import json, sys
from PIL import Image, ImageDraw
pid, out = sys.argv[1], sys.argv[2]
specs = json.load(open("/data/seotturi/img-gen/specs.json"))
s = specs[f"personas/{pid}.jpg"]["faces"][0]
im = Image.open(f"/data/seotturi/seotturi-web/web/assets/personas/{pid}.jpg").convert("RGB")
bx = s["boxes"] + ([s["mouth"]] if "mouth" in s else [])
x0 = min(b[0] for b in bx) - 50; y0 = min(b[1] for b in bx) - 50; x1 = max(b[2] for b in bx) + 50; y1 = max(b[3] for b in bx) + 30
k = 3
c = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k), Image.LANCZOS)
d = ImageDraw.Draw(c)
for gx in range((x0 // 20 + 1) * 20, x1, 20):
    d.line([((gx - x0) * k, 0), ((gx - x0) * k, c.height)], fill=(0, 160, 255) if gx % 100 else (255, 0, 0), width=1)
    if gx % 100 == 0: d.text(((gx - x0) * k + 2, 2), str(gx), fill=(255, 0, 0))
for gy in range((y0 // 20 + 1) * 20, y1, 20):
    d.line([(0, (gy - y0) * k), (c.width, (gy - y0) * k)], fill=(0, 160, 255) if gy % 100 else (255, 0, 0), width=1)
    if gy % 100 == 0: d.text((2, (gy - y0) * k + 2), str(gy), fill=(255, 0, 0))
for b in bx:
    d.rectangle([((b[0] - x0) * k, (b[1] - y0) * k), ((b[2] - x0) * k, (b[3] - y0) * k)], outline=(0, 220, 0), width=3)
c.save(out)
print(out, (x0, y0, x1, y1))
