"""Cut the web assets out of the team's poster and final deck.

    ../.venv/bin/python -I scripts/extract_assets.py

Reads ~/Downloads/서투리포스터.jpg and ~/Downloads/서투리_최종 PPT.pptx (slide images and the
four screen recordings), writes PNG/JPEG files to web/assets/. Logo, icon, mascot and star are
cut out of their blue backgrounds; the rest are plain crops.
"""

import io
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

HOME = os.path.expanduser("~")
POSTER = os.path.join(HOME, "Downloads", "서투리포스터.jpg")
DECK = os.path.join(HOME, "Downloads", "서투리_최종 PPT.pptx")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "assets")


def deck_image(name):
    with zipfile.ZipFile(DECK) as z:
        return Image.open(io.BytesIO(z.read(f"ppt/media/{name}"))).convert("RGB")


def video_frame(name, seconds, tmp):
    """One frame of a screen recording in the deck (900 x 1814 phone mockup)."""
    src = os.path.join(tmp, name)
    if not os.path.exists(src):
        with zipfile.ZipFile(DECK) as z, open(src, "wb") as f:
            f.write(z.read(f"ppt/media/{name}"))
    dst = os.path.join(tmp, f"{name}_{seconds}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(seconds), "-i", src, "-frames:v", "1", dst],
                   check=True)
    return Image.open(dst).convert("RGB")


def cutout(img, step=12, min_area=400, keep=None, feather=0.8, sides="tblr", opening=1, drop=None, sky=False):
    """Make the background transparent: flood the smooth area connected to the image border.

    A pixel is "smooth" when it differs from its right and lower neighbours by less than `step`
    in every channel; smooth regions that touch the border are background (sky, flat fill).
    Objects have crisp rendered edges, so the flood stops there. `keep` limits the foreground to
    components overlapping that box (x0, y0, x1, y1), e.g. to drop a neighbouring speech bubble.
    `sides` names the borders that seed the flood; leave out a side the object is cut off at.
    `opening` strips thin lines (a phone edge touching the object); `drop` is a polygon whose
    inside is always background. `sky` also clears saturated sky-blue pixels the border flood
    could not reach (between the legs, where the object is cut off).
    """
    a = np.asarray(img).astype(np.int16)
    dx = np.abs(np.diff(a, axis=1, append=a[:, -1:])).max(axis=2)
    dy = np.abs(np.diff(a, axis=0, append=a[-1:])).max(axis=2)
    smooth = (dx < step) & (dy < step)
    lab, _ = ndimage.label(smooth)
    edges = {"t": lab[0], "b": lab[-1], "l": lab[:, 0], "r": lab[:, -1]}
    border = np.unique(np.concatenate([edges[k] for k in sides]))
    bg = np.isin(lab, border[border > 0])
    if drop is not None:
        from PIL import ImageDraw
        m = Image.new("L", img.size, 0)
        ImageDraw.Draw(m).polygon(drop, fill=1)
        bg |= np.asarray(m).astype(bool)
    fg = ndimage.binary_opening(~bg, iterations=opening)
    lab, n = ndimage.label(fg)
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    good = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if keep is not None:
        x0, y0, x1, y1 = keep
        inside = np.unique(lab[y0:y1, x0:x1])
        good = [g for g in good if g in inside]
    fg = ndimage.binary_fill_holes(np.isin(lab, good))
    if sky:
        r, b = a[..., 0], a[..., 2]
        fg &= ~ndimage.binary_dilation((b > 190) & (r < 150) & (b - r > 80), iterations=1)
    alpha = ndimage.gaussian_filter(fg.astype(np.float32), feather)
    alpha = np.clip((alpha - 0.15) / 0.7, 0, 1)
    # pull colours on the soft edge from the nearest solid pixel so no blue fringe is left
    solid = ndimage.binary_erosion(fg, iterations=2)
    idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    rgb = np.asarray(img)[idx[0], idx[1]]
    edge = (alpha < 1) | ~solid
    out = np.where(edge[..., None], rgb, np.asarray(img))
    rgba = np.dstack([out, (alpha * 255).astype(np.uint8)])
    res = Image.fromarray(rgba.astype(np.uint8), "RGBA")
    return res.crop(res.getbbox())


def save(img, name, width=None, quality=88):
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    path = os.path.join(OUT, name)
    if name.endswith(".jpg"):
        img.convert("RGB").save(path, quality=quality, optimize=True, progressive=True)
    else:
        img.save(path, optimize=True)
    print(f"{name:28s} {img.width}x{img.height}  {os.path.getsize(path) // 1024} KB")


def main():
    os.makedirs(OUT, exist_ok=True)
    poster = Image.open(POSTER).convert("RGB")  # 2000 x 3512
    title = deck_image("image1.png")  # 7680 x 4320, flat #0072FF behind icon and logo

    save(cutout(title.crop((560, 250, 1210, 900))), "icon.png", width=360)
    save(cutout(title.crop((1300, 300, 2840, 880)), step=14), "logo-white.png", width=900)
    # poster hero: the bubble above and the phone to the right are dropped, the bottom edge is a cut
    save(cutout(poster.crop((270, 1225, 975, 1998)), keep=(250, 100, 500, 600), sides="tlr", opening=2,
                drop=[(482, 0), (705, 0), (705, 240), (636, 240)], sky=True), "mascot-wave.png", width=600)
    save(cutout(poster.crop((1320, 1045, 1490, 1215))), "star.png")
    save(poster.crop((958, 2215, 1742, 2572)), "office-comic.jpg")
    save(poster.crop((1272, 2903, 1744, 3381)), "branding.jpg")

    with tempfile.TemporaryDirectory() as tmp:
        home = video_frame("media1.mov", 3.0, tmp)
        save(home.crop((80, 545, 820, 1495)), "map.jpg")
        coach = video_frame("media2.mov", 33.5, tmp)
        save(coach.crop((60, 165, 840, 1395)), "interviewer.jpg")
        modes = video_frame("media2.mov", 2.5, tmp)
        for name, top in (("mode-interview.jpg", 588), ("mode-presentation.jpg", 898), ("mode-daily.jpg", 1208)):
            save(modes.crop((121, top + 3, 355, top + 235)), name)
        calc = video_frame("media2.mov", 53.5, tmp)
        save(calc.crop((270, 685, 632, 1047)), "calculator.jpg")
        result = video_frame("media2.mov", 60.0, tmp)
        result = result.crop((98, 700, 803, 1440))
        bg = result.getpixel((20, 700))
        result.paste(bg, (560, 650, 705, 740))  # the recording's download button
        save(result, "result-pass.jpg")

    splash = deck_image("image7.png")  # slide 6: the splash screen, larger than the recording and without its cursor
    save(splash.crop((3100, 2650, 4580, 3815)), "splash-scene.jpg", width=900)

    # AI 코칭 start screen: the deck's blurred empty room, with its button painted over
    room = np.asarray(deck_image("image11.png").crop((60, 160, 840, 1540))).astype(np.float32)
    top, bot = 680, 810  # rows around the "새 프로젝트 만들기" button
    for y in range(top + 1, bot):
        w = (y - top) / (bot - top)
        room[y] = room[top] * (1 - w) + room[bot] * w
    x0, x1 = 430, 510  # the presenter's mouse cursor
    for x in range(x0 + 1, x1):
        w = (x - x0) / (x1 - x0)
        room[1200:1290, x] = room[1200:1290, x0] * (1 - w) + room[1200:1290, x1] * w
    save(Image.fromarray(room.astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), "room-blur.jpg", width=600, quality=80)

    ui = deck_image("image12.png")  # slide 9: the three coaching modes in phone mockups
    save(ui.crop((3135, 1720, 4545, 3230)), "audience.jpg", width=900)
    save(ui.crop((5415, 1720, 6825, 3230)), "friend.jpg", width=900)


if __name__ == "__main__":
    sys.exit(main())
