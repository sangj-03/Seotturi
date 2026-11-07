"""Put chosen drawings into the app: web/assets/personas/<id>.jpg (scene) and <id>.face.jpg (card avatar),
plus the eye and mouth boxes scripts/alive_frames.py needs to make them blink and talk.

    python finalize.py barista=s1 grandma=s0 ...      # prints the SPECS entries to add
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from avatar import avatar
from faces import detect

WEB = "/data/seotturi/seotturi-web/web/assets/"


def eye_box(img, x, y, d):
    """The box around one eye: the white of the eye found near the landmark, plus room for lids and lashes
    (inside a glasses rim too). Falls back to a box sized from the eye distance."""
    x0, y0 = int(max(0, x - 0.45 * d)), int(max(0, y - 0.4 * d))
    reg = img[y0:int(y + 0.4 * d), x0:int(x + 0.45 * d)].astype(float)
    white = (reg.min(axis=2) > 185) & (reg.max(axis=2) - reg.min(axis=2) < 40)
    lab, n = ndi.label(white)
    # The iris often splits the white of the eye into a left and a right part: take every white blob
    # whose centre is near the landmark.
    xs_all, ys_all = [], []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < 0.002 * d * d:
            continue
        if np.hypot(xs.mean() + x0 - x, ys.mean() + y0 - y) < 0.42 * d:
            xs_all.append(xs + x0); ys_all.append(ys + y0)
    if not xs_all:
        return (round(x - 0.32 * d), round(y - 0.27 * d), round(x + 0.32 * d), round(y + 0.27 * d))
    xs, ys = np.concatenate(xs_all), np.concatenate(ys_all)
    bx0, by0, bx1, by1 = min(xs.min(), x - 0.12 * d), min(ys.min(), y - 0.1 * d), max(xs.max(), x + 0.12 * d), max(ys.max(), y + 0.1 * d)
    return (int(bx0 - 0.07 * d), int(by0 - 0.1 * d), int(bx1 + 0.07 * d), int(by1 + 0.08 * d))


def boxes(f, img):
    """alive_frames.py boxes from YuNet landmarks: two eye boxes and the mouth (lip line) box."""
    (rx, ry), (lx, ly), _, (mrx, mry), (mlx, mly) = f["lm"]
    d = abs(lx - rx)
    eyes = [eye_box(img, x, y, d) for x, y in sorted([(rx, ry), (lx, ly)])]
    my, mx = (mry + mly) / 2, (mrx + mlx) / 2
    mouth = (round(mx - 0.55 * d), round(my - 0.2 * d), round(mx + 0.55 * d), round(my + 0.06 * d))
    return eyes, mouth


def spec(path, talker_x=None, max_faces=5):
    img = np.asarray(Image.open(path).convert("RGB"))
    fs = [f for f in detect(path) if f["score"] > 0.8][:max_faces]
    if not fs:
        return None
    big = max(f["box"][2] for f in fs)  # drop small faces such as the photo on an ID card
    fs = [f for f in fs if f["box"][2] >= 0.4 * big]
    talker = fs[0] if talker_x is None else min(fs, key=lambda f: abs(f["box"][0] + f["box"][2] / 2 - talker_x))
    faces = []
    for f in [talker] + [f for f in fs if f is not talker]:
        eyes, mouth = boxes(f, img)
        faces.append({"eyes": "human", "boxes": eyes, **({"mouth": mouth} if f is talker else {})})
    return {"faces": faces}


if __name__ == "__main__":
    out = {}
    for arg in sys.argv[1:]:
        pid, pick = arg.split("=")
        src = f"/data/seotturi/img-gen/out/{pid}/{pick}.png"
        im = Image.open(src).convert("RGB")
        dst = f"{WEB}personas/{pid}.jpg"
        im.save(dst, quality=86, optimize=True, progressive=True)
        w = im.width
        s = spec(dst, talker_x=w / 2)
        avatar(dst, pid, pick=w / 2)
        out[f"personas/{pid}.jpg"] = s
        print(pid, im.size, json.dumps(s))
    json.dump(out, open("/data/seotturi/img-gen/specs.json", "w"), ensure_ascii=False, indent=1)
