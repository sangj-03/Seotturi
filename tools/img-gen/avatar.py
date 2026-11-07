"""Square face avatars for the persona cards, cut around the YuNet eye landmarks."""
import os
import json, sys
from PIL import Image
from faces import detect
OUT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "app", "web", "assets"), "personas") + "/"
def avatar(src, pid, pick=None, size=200):
    fs = detect(src)
    f = fs[0] if pick is None else min(fs, key=lambda f: abs(f["box"][0] + f["box"][2] / 2 - pick))
    (rx, ry), (lx, ly) = f["lm"][0], f["lm"][1]
    d = abs(lx - rx); cx, cy = (rx + lx) / 2, (ry + ly) / 2 - 0.1 * d
    half = 1.95 * d
    im = Image.open(src).convert("RGB")
    box = (round(cx - half), round(cy - half), round(cx + half), round(cy + half))
    im.crop(box).resize((size, size), Image.LANCZOS).save(f"{OUT}{pid}.face.jpg", quality=90)
    return box
if __name__ == "__main__":
    A = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "app", "web", "assets") + "/"
    print(avatar(A + "interviewer.jpg", "strict-manager"))
    print(avatar(A + "friend.jpg", "doyun"))
    print(avatar(A + "audience.jpg", "friendly-team", pick=450))
