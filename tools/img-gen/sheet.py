"""Contact sheet of the drawn candidates: python sheet.py out.png id id ... (all seeds of each id, side by side)."""
import glob, os, sys
from PIL import Image, ImageDraw
out, ids = sys.argv[1], sys.argv[2:]
H = 380
rows = []
for pid in ids:
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), "out", pid, "*.png")))
    ims = [Image.open(f).convert("RGB") for f in files]
    ims = [im.resize((round(im.width * H / im.height), H)) for im in ims]
    w = sum(im.width for im in ims) + 8 * len(ims)
    row = Image.new("RGB", (max(w, 200), H + 22), "white")
    x = 0
    d = ImageDraw.Draw(row)
    for f, im in zip(files, ims):
        row.paste(im, (x, 22)); d.text((x + 4, 4), f"{pid} {os.path.basename(f)[:-4]}", fill="black"); x += im.width + 8
    rows.append(row)
W = max(r.width for r in rows)
sheet = Image.new("RGB", (W, sum(r.height for r in rows)), (235, 235, 235))
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height
sheet.save(out)
print(out, sheet.size)
