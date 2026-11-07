"""Make the extra frames that let the app's characters blink and talk.

    python3 scripts/alive_frames.py            # writes web/assets/alive/*.webp + manifest.json
    python3 scripts/alive_frames.py --preview  # also writes a contact sheet to check the frames

Each character image gets frames that differ from it only around the eyes or the mouth:
blink1 (half closed), blink2 (closed), talk1 (mouth a little open), talk2 (wide open).
The eyes are covered with a colour field fitted to the surrounding skin, and a closed-lid line is
drawn where the lids meet. Only the changed rectangle of each frame is saved (lossless WebP, a few
KB) with its position; web/js/alive.js pastes it onto the decoded original in the browser, so every
other pixel of a frame is exactly the original and swapping frames does not shimmer.

Eye and mouth positions are pixel boxes in the original image (x0, y0, x1, y1). "dot" eyes are
the mascot's solid black eyes; "human" eyes are segmented inside the box as the pixels that differ
from the skin at the box border (white, iris, lashes). Mouths are the dark lip line in the box.
"""

import argparse
import hashlib
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "web", "assets")
OUT = os.path.join(ASSETS, "alive")
SS = 4  # supersampling for anti-aliased drawing

# (x0, y0, x1, y1) boxes. A "face" is the unit that blinks together; a scene with several people
# gets one blink set per face so they do not blink in unison.
SPECS = {
    "icon.png": {"faces": [{"eyes": "dot", "boxes": [(114, 135, 131, 155), (174, 152, 192, 172)], "mouth": (126, 165, 169, 187)}]},
    "mascot-wave.png": {"faces": [{"eyes": "dot", "boxes": [(234, 131, 260, 159), (327, 132, 352, 159)], "mouth": (261, 171, 327, 195)}]},
    "star.png": {"faces": [{"eyes": "dot", "boxes": [(38, 46, 48, 58), (61, 54, 71, 67)], "mouth": (38, 64, 61, 78)}]},
    "mode-interview.jpg": {"faces": [{"eyes": "dot", "boxes": [(119, 54, 128, 64), (150, 55, 159, 66)], "mouth": (126, 69, 151, 78)}]},
    "mode-presentation.jpg": {"faces": [{"eyes": "dot", "boxes": [(70, 52, 79, 62), (97, 52, 104, 62)], "mouth": (78, 65, 97, 73)}]},
    "mode-daily.jpg": {"faces": [{"eyes": "dot", "boxes": [(56, 63, 66, 75), (84, 63, 93, 73)], "mouth": (66, 78, 88, 87)}]},
    "result-pass.jpg": {"faces": [{"eyes": "dot", "boxes": [(318, 345, 336, 363), (375, 345, 393, 363)], "mouth": (334, 367, 378, 386)}]},
    "splash-scene.jpg": {"faces": [{"eyes": "dot", "boxes": [(211, 187, 226, 203), (262, 184, 273, 201)], "mouth": (232, 212, 266, 227)},
                                   {"eyes": "dot", "boxes": [(338, 107, 347, 118), (359, 106, 366, 116)], "mouth": (343, 120, 364, 130)}]},
    "interviewer.jpg": {"faces": [{"eyes": "human", "boxes": [(298, 529, 372, 572), (415, 529, 492, 572)], "mouth": (338, 638, 452, 662),
                                   "mouth_kind": "frown"}]},
    "friend.jpg": {"faces": [{"eyes": "human", "boxes": [(346, 437, 418, 501), (481, 434, 557, 497)], "mouth": (392, 552, 520, 582)}]},
    "audience.jpg": {"faces": [
        {"eyes": "human", "boxes": [(378, 379, 433, 419), (461, 377, 517, 417)], "mouth": (404, 450, 492, 474)},  # front centre (talks)
        {"eyes": "human", "boxes": [(52, 393, 96, 425), (119, 393, 163, 425)]},     # top left
        {"eyes": "human", "boxes": [(717, 378, 759, 410), (785, 377, 827, 409)]},   # top right
        {"eyes": "human", "boxes": [(32, 702, 76, 736), (110, 703, 156, 737)]},     # bottom left
        {"eyes": "human", "boxes": [(723, 700, 768, 735), (807, 698, 853, 733)]},   # bottom right (glasses)
    ]},
    # >>> personas (boxes from YuNet landmarks: /data/seotturi/img-gen/specs_block.py)
    "personas/barista.jpg": {"faces": [{"eyes": "human", "boxes": [(390, 357, 469, 416), (541, 356, 621, 412)]}]},
    "personas/busy-execs.jpg": {"faces": [{"eyes": "human", "boxes": [(426, 418, 466, 451), (496, 421, 536, 451)], "mouth": (447, 472, 511, 487)}, {"eyes": "human", "boxes": [(149, 433, 182, 457), (211, 437, 237, 460)]}, {"eyes": "human", "boxes": [(731, 442, 757, 464), (772, 436, 799, 459)]}]},
    "personas/cafe-owner.jpg": {"faces": [{"eyes": "human", "boxes": [(378, 327, 466, 378), (532, 326, 626, 380)], "mouth": (432, 456, 567, 488)}]},
    "personas/calm-hr.jpg": {"faces": [{"eyes": "human", "boxes": [(302, 534, 384, 598), (426, 535, 513, 600)], "mouth": (335, 637, 464, 667)}]},
    "personas/cool-exec.jpg": {"faces": [{"eyes": "human", "boxes": [(307, 530, 374, 583), (430, 526, 502, 583)], "mouth": (355, 626, 469, 653)}]},
    "personas/grandma.jpg": {"faces": [{"eyes": "human", "boxes": [(417, 347, 465, 413), (544, 349, 602, 413)]}]},
    "personas/jimin.jpg": {"faces": [{"eyes": "human", "boxes": [(377, 447, 466, 516), (538, 443, 636, 515)]}]},
    "personas/junior.jpg": {"faces": [{"eyes": "human", "boxes": [(380, 324, 463, 399), (525, 324, 612, 400)]}]},
    "personas/office-worker.jpg": {"faces": [{"eyes": "human", "boxes": [(393, 336, 466, 403), (527, 335, 605, 404)], "mouth": (438, 460, 561, 489)}]},
    "personas/peer.jpg": {"faces": [{"eyes": "human", "boxes": [(370, 350, 463, 414), (535, 348, 626, 412)], "mouth": (437, 473, 576, 506)}]},
    "personas/rush-staff.jpg": {"faces": [{"eyes": "human", "boxes": [(411, 312, 468, 375), (529, 313, 592, 377)], "mouth": (443, 428, 559, 455)}]},
    "personas/seojin.jpg": {"faces": [{"eyes": "human", "boxes": [(379, 403, 460, 464), (540, 399, 618, 462)], "mouth": (429, 539, 568, 572)}]},
    "personas/sharp-judges.jpg": {"faces": [{"eyes": "human", "boxes": [(423, 763, 455, 797), (502, 762, 535, 800)], "mouth": (437, 836, 523, 856)}, {"eyes": "human", "boxes": [(727, 770, 772, 803), (801, 772, 834, 804)]}, {"eyes": "human", "boxes": [(119, 773, 159, 808), (188, 774, 233, 812)]}]},
    "personas/startup-leader.jpg": {"faces": [{"eyes": "human", "boxes": [(310, 520, 382, 587), (431, 520, 504, 587)]}]},
    "personas/student.jpg": {"faces": [{"eyes": "human", "boxes": [(399, 463, 469, 527), (545, 460, 623, 528)], "mouth": (437, 581, 575, 614)}]},
    "personas/team-lead.jpg": {"faces": [{"eyes": "human", "boxes": [(384, 312, 463, 378), (529, 311, 611, 379)], "mouth": (423, 450, 569, 485)}]},
    "personas/turi.jpg": {"faces": [{"eyes": "dot", "boxes": [(387, 347, 428, 391), (541, 348, 579, 392)], "mouth": (432, 412, 538, 451)}]},
    "personas/warm-mentor.jpg": {"faces": [{"eyes": "human", "boxes": [(317, 512, 382, 567), (432, 508, 499, 568)], "mouth": (359, 601, 469, 627)}]},
    # <<< personas
}

# A raised hand that waves: the hand is cut out as its own layer and turned about the wrist (pivot)
# by css/alive.css. Where the hand covered the sleeve (inside the convex hull of the sleeve's
# pixels), the body gets the sleeve's colour, so turning the hand shows more sleeve, not a hole.
# "seed" is a pixel on the hand and "box" bounds the search.
RIGS = {
    "mascot-wave.png": {"box": (470, 180, 600, 330), "seed": (546, 250), "pivot": (546, 330)},
}


def lum(rgb):
    return rgb[..., :3] @ np.array([0.299, 0.587, 0.114])


def dot_mask(img, box):
    """The mascot's solid black eye as a filled ellipse a little larger than the dark blob, so the
    soft edge and the white highlight inside it go too."""
    x0, y0, x1, y1 = box
    m = np.zeros(img.shape[:2], bool)
    m[y0:y1, x0:x1] = lum(img[y0:y1, x0:x1].astype(float)) < 120
    lab, n = ndi.label(m)
    if n > 1:  # keep the biggest blob (the eye), drop stray dark pixels
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    if not m.any():
        return m
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    ry, rx = (ys.max() - ys.min() + 1) / 2 * 1.25 + 1, (xs.max() - xs.min() + 1) / 2 * 1.25 + 1
    yy, xx = np.ogrid[: m.shape[0], : m.shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def ellipse_in(shape, box, grow=1.15):
    x0, y0, x1, y1 = box
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    ry, rx = (y1 - y0) / 2 * grow, (x1 - x0) / 2 * grow
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def human_mask(img, box):
    """Everything around the eye that is not skin: the white, the iris and the lash lines.
    Skin is measured in a ring just outside the box (dark frames and bright whites left out), and
    the mask is kept inside an ellipse over the box so brows and glasses frames stay untouched."""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - 6), max(0, y0 - 6), min(w, x1 + 6), min(h, y1 + 6)
    outer = img[Y0:Y1, X0:X1, :3].astype(float)
    ring = np.ones(outer.shape[:2], bool)
    ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    px = outer[ring]
    L = lum(px)
    skin = np.median(px[(L > 70) & (L < 235)] if ((L > 70) & (L < 235)).sum() > 10 else px, axis=0)
    crop = img[y0:y1, x0:x1, :3].astype(float)
    m = np.linalg.norm(crop - skin, axis=2) > 40
    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = m
    full &= ellipse_in(img.shape, box)
    full = ndi.binary_closing(full, structure=np.ones((3, 3)), iterations=2)
    full = ndi.binary_fill_holes(full)
    lab, n = ndi.label(full)
    if n > 1:
        sizes = ndi.sum(full, lab, range(1, n + 1))
        full = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_dilation(full, iterations=2)


def skin_fill(img, mask, ring_w=7, tol=40):
    """Cover the mask with a smooth colour field fitted to the skin around it (a quadratic surface
    per channel), ignoring ring pixels that are not skin-coloured, then feather the edge."""
    ring = ndi.binary_dilation(mask, iterations=ring_w) & ~ndi.binary_dilation(mask, iterations=1)
    ys, xs = np.nonzero(ring)
    px = img[ys, xs, :3].astype(float)
    # Reference colour from skin-like ring pixels: not dark (frames, lashes), not blown out, and
    # warm or neutral (R >= B), which holds for every skin tone here and for the white/yellow mascots.
    L = lum(px)
    skinlike = (L > 90) & (L < 248) & (px[:, 0] >= px[:, 2] - 8)
    med = np.median(px[skinlike] if skinlike.sum() >= 12 else px, axis=0)
    keep = np.linalg.norm(px - med, axis=1) < tol
    if keep.sum() >= 12:
        ys, xs, px = ys[keep], xs[keep], px[keep]
    cy, cx = ys.mean(), xs.mean()
    sc = max(1.0, float(max(ys.max() - ys.min(), xs.max() - xs.min())))
    def design(yv, xv):  # a plane: enough for skin shading over an eye, and it cannot overshoot
        return np.stack([np.ones_like(xv, dtype=float), (xv - cx) / sc, (yv - cy) / sc], 1)
    coef, *_ = np.linalg.lstsq(design(ys, xs), px, rcond=None)
    region = ndi.binary_dilation(mask, iterations=2)
    ry, rx = np.nonzero(region)
    lo, hi = np.percentile(px, 3, axis=0), np.percentile(px, 97, axis=0)
    pred = np.clip(design(ry, rx) @ coef, lo, hi)
    alpha = np.clip(ndi.gaussian_filter(mask.astype(float), 1.0) * 1.8, 0, 1)[ry, rx][:, None]
    out = img.astype(float).copy()
    out[ry, rx, :3] = pred * alpha + out[ry, rx, :3] * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def columns(mask):
    """Per column of a mask: x, top y, bottom y."""
    xs = np.where(mask.any(axis=0))[0]
    tops = np.array([np.where(mask[:, x])[0][0] for x in xs])
    bots = np.array([np.where(mask[:, x])[0][-1] for x in xs])
    return xs, tops, bots


def dark_color(img, mask):
    px = img[mask][:, :3].astype(float)
    order = np.argsort(lum(px))
    return tuple(int(v) for v in np.median(px[order[: max(3, len(order) // 6)]], axis=0))


def draw_layer(size, fn):
    """Draw with PIL at SS x size and return an RGBA float layer at the original size."""
    big = Image.new("RGBA", (size[1] * SS, size[0] * SS), (0, 0, 0, 0))
    fn(ImageDraw.Draw(big), SS)
    return np.asarray(big.resize((size[1], size[0]), Image.LANCZOS)).astype(float)


def over(img, layer):
    a = layer[..., 3:4] / 255.0
    out = img.astype(float).copy()
    out[..., :3] = layer[..., :3] * a + out[..., :3] * (1 - a)
    if out.shape[2] == 4:
        out[..., 3] = np.maximum(out[..., 3], layer[..., 3])
    return np.clip(out, 0, 255).astype(np.uint8)


def lid_line(xs, ys, ss):
    """Smooth the lid curve with a quadratic fit; returns supersampled points."""
    if len(xs) >= 3:
        c = np.polyfit(xs, ys, 2)
        ys = np.polyval(c, xs)
    return [((x + 0.5) * ss, (y + 0.5) * ss) for x, y in zip(xs, ys)]


def tapered(d, pts, width, color, ss):
    """A stroke that is thickest in the middle and thin at the corners, like a lash line."""
    n = len(pts)
    for i in range(n - 1):
        t = i / max(1, n - 2)
        w = max(1.0, width * (0.45 + 0.55 * np.sin(np.pi * t))) * ss
        d.line([pts[i], pts[i + 1]], fill=color, width=int(round(w)))
    for p, t in ((pts[0], 0), (pts[-1], 1)):
        r = max(0.5, width * 0.45 * ss / 2)
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)


def blink(img, face, level):
    """level 1: half closed, 2: closed."""
    res = img.copy()
    for box in face["boxes"]:
        if face["eyes"] == "dot":
            m = dot_mask(img, box)
            if not m.any():
                continue
            col = dark_color(img, m)
            xs, tops, bots = columns(m)
            cx, w = (xs[0] + xs[-1]) / 2, xs[-1] - xs[0] + 1
            h = bots.max() - tops.min() + 1
            cy = (tops.min() + bots.max()) / 2
            res = skin_fill(res, ndi.binary_dilation(m, iterations=1))
            if level == 1:  # squint: a flatter dot sitting low
                ry, rx = h * 0.24, w * 0.47
                layer = draw_layer(img.shape, lambda d, s: d.ellipse([(cx - rx) * s, (cy + h * 0.12 - ry) * s, (cx + rx) * s, (cy + h * 0.12 + ry) * s], fill=col + (255,)))
            else:  # closed: a short curve bowing down, like a resting eye
                pts = [((cx + w * 0.62 * u + 0.5) * SS, (cy + h * 0.18 + h * 0.16 * (1 - u * u) + 0.5) * SS) for u in np.linspace(-1, 1, 15)]
                layer = draw_layer(img.shape, lambda d, s: tapered(d, pts, max(1.6, w * 0.24), col + (255,), s))
            res = over(res, layer)
        else:
            m = human_mask(img, box)
            if not m.any():
                continue
            col = dark_color(img, m)
            xs, tops, bots = columns(m)
            h = np.maximum(1, bots - tops)
            closed = skin_fill(res, m)
            if level == 1:  # the upper lid comes halfway down: skin above the cut, the eye below it
                cut = tops + 0.48 * h
                cut_full = np.interp(np.arange(m.shape[1]), xs, cut, left=1e9, right=1e9)
                below = m & (np.arange(m.shape[0])[:, None] > cut_full[None, :])
                a = np.clip(ndi.gaussian_filter(below.astype(float), 0.7) * 1.4, 0, 1)[..., None]
                res = np.clip(res.astype(float) * a + closed.astype(float) * (1 - a), 0, 255).astype(np.uint8)
                pts = lid_line(xs[1:-1] if len(xs) > 6 else xs, cut[1:-1] if len(xs) > 6 else cut, SS)
                width = max(1.6, float(np.median(h)) * 0.13)
            else:  # fully closed: lids meet a little below the middle
                res = closed
                meet = tops + 0.62 * h
                pts = lid_line(xs[2:-2] if len(xs) > 8 else xs, (meet[2:-2] if len(xs) > 8 else meet), SS)
                width = max(1.5, float(np.median(h)) * 0.11)
            layer = draw_layer(img.shape, lambda d, s: tapered(d, pts, width, col + (255,), s))
            res = over(res, layer)
    return res


def talk(img, face, level):
    """level 1: a little open, 2: wide open. The lip line stays as the upper lip."""
    if "mouth" not in face:
        return None
    x0, y0, x1, y1 = face["mouth"]
    m = np.zeros(img.shape[:2], bool)
    m[y0:y1, x0:x1] = lum(img[y0:y1, x0:x1].astype(float)) < (110 if face["eyes"] == "dot" else 135)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    xs, tops, bots = columns(m)
    mid = (tops + bots) / 2
    if len(xs) >= 3:
        mid = np.polyval(np.polyfit(xs, mid, 2), xs)
    width = xs[-1] - xs[0] + 1
    line = dark_color(img, m)
    human = face["eyes"] == "human"
    depth = width * ((0.13 if level == 1 else 0.24) if human else (0.22 if level == 1 else 0.38))
    t = (xs - xs[0]) / max(1, width - 1)
    bulge = depth * np.sin(np.pi * t) ** 0.9
    if face.get("mouth_kind") == "frown":  # opens around the line, not below it
        top, bot = mid - bulge * 0.35, mid + bulge * 0.65
    else:
        top, bot = mid, mid + bulge
    cavity = (74, 30, 34) if human else (70, 30, 36)
    tongue = (196, 96, 104) if human else (222, 112, 122)

    def drawing(d, s):
        poly = [((x + 0.5) * s, (y + 0.5) * s) for x, y in zip(xs, top)] + [((x + 0.5) * s, (y + 0.5) * s) for x, y in zip(xs[::-1], bot[::-1])]
        d.polygon(poly, fill=cavity + (255,))
        cx, cy = (xs[0] + xs[-1]) / 2, (top + bot)[len(xs) // 2] / 2 + bulge.max() * 0.22
        rx, ry = width * 0.22, bulge.max() * 0.3
        if level == 2 or not human:
            d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry + bulge.max() * 0.1) * s], fill=tongue + (255,))
        if human and level == 2:  # a strip of upper teeth
            teeth = [((x + 0.5) * s, (y + 0.5) * s) for x, y in zip(xs[2:-2], top[2:-2] + 0.6)] + \
                    [((x + 0.5) * s, (y + 0.5) * s) for x, y in zip(xs[2:-2][::-1], (top + bulge * 0.28)[2:-2][::-1])]
            if len(teeth) > 4:
                d.polygon(teeth, fill=(246, 242, 236, 255))
        lw = max(1, round(np.median(bots - tops + 1) * 0.6)) * s
        d.line([((x + 0.5) * s, (y + 0.5) * s) for x, y in zip(xs, top)], fill=line + (255,), width=int(lw), joint="curve")
        d.line([((x + 0.5) * s, (y + 0.5) * s) for x, y in zip(xs, bot)], fill=line + (200,), width=max(1, int(lw * 0.6)), joint="curve")

    return over(img, draw_layer(img.shape, drawing))


def rig(img, spec, stem):
    """The hand as an RGBA patch and the body without it; returns the manifest entry."""
    x0, y0, x1, y1 = spec["box"]
    reg = img[y0:y1, x0:x1].astype(float)
    m = (reg[..., 3] > 0) & (lum(reg) > 125)  # the white hand, not the navy sleeve
    lab, _ = ndi.label(m)
    hand = np.zeros(img.shape[:2], bool)
    hand[y0:y1, x0:x1] = ndi.binary_fill_holes(lab == lab[spec["seed"][1] - y0, spec["seed"][0] - x0])
    hand = ndi.binary_dilation(hand, iterations=1) & (img[..., 3] > 0)  # take the anti-aliased rim along
    ys, xs = np.nonzero(hand)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    cut = hand[by0:by1, bx0:bx1]
    layer = img[by0:by1, bx0:bx1].copy()
    layer[~cut, 3] = 0
    Image.fromarray(layer, "RGBA").save(os.path.join(OUT, f"{stem}.hand.webp"), "WEBP", lossless=True, quality=100, method=6, exact=True)
    sleeve = np.zeros(img.shape[:2], bool)
    sleeve[y0:y1, x0:x1] = (reg[..., 3] > 200) & (lum(reg) < 110)
    sy, sx = np.nonzero(sleeve)
    pts = np.c_[sx, sy]
    hull = Delaunay(pts[ConvexHull(pts).vertices])
    yy, xx = np.mgrid[by0:by1, bx0:bx1]
    inside = cut & (hull.find_simplex(np.c_[xx.ravel(), yy.ravel()]) >= 0).reshape(cut.shape)
    cuff = ndi.binary_dilation(hand, iterations=5) & ~hand & sleeve  # the sleeve right around the hand
    body = img[by0:by1, bx0:bx1].copy()
    body[cut] = (0, 0, 0, 0)
    body[inside] = tuple(int(v) for v in np.median(img[cuff][:, :3], axis=0)) + (255,)
    Image.fromarray(body, "RGBA").save(os.path.join(OUT, f"{stem}.body.webp"), "WEBP", lossless=True, quality=100, method=6, exact=True)
    return {"body": [f"{stem}.body.webp", int(bx0), int(by0)], "hand": [f"{stem}.hand.webp", int(bx0), int(by0)],
            "pivot": list(spec["pivot"])}


def save_patch(arr, img, path, alpha):
    """Save the rectangle where the frame differs from the original; returns [file, x, y]."""
    ys, xs = np.nonzero(np.any(arr != img, axis=-1))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    im = Image.fromarray(arr[y0:y1, x0:x1], "RGBA" if alpha else "RGB")
    im.save(path, "WEBP", lossless=True, quality=100, method=6, exact=True)
    return [os.path.basename(path), int(x0), int(y0)]


def frames_of(name, spec, previews, preview):
    """Blink and talk frames (and the rig) of one image; returns its manifest entry."""
    src = Image.open(os.path.join(ASSETS, name))
    alpha = src.mode in ("RGBA", "LA") or "transparency" in src.info
    img = np.asarray(src.convert("RGBA" if alpha else "RGB"))
    stem = os.path.splitext(name)[0].replace("/", "_")  # personas/x.jpg -> personas_x: alive/ stays flat
    entry = {"size": [img.shape[1], img.shape[0]], "blink": [], "talk": []}
    for i, face in enumerate(spec["faces"]):
        frames = []
        for level in (1, 2):
            arr = blink(img, face, level)
            frames.append(save_patch(arr, img, os.path.join(OUT, f"{stem}.f{i}.blink{level}.webp"), alpha))
            if preview:
                previews.append((name, f"face{i} blink{level}", face, arr, img))
        entry["blink"].append(frames)
        if "mouth" in face and not entry["talk"]:  # one talking face per image
            for level in (1, 2):
                arr = talk(img, face, level)
                entry["talk"].append(save_patch(arr, img, os.path.join(OUT, f"{stem}.talk{level}.webp"), alpha))
                if preview:
                    previews.append((name, f"talk{level}", face, arr, img))
    if name in RIGS:
        entry["rig"] = rig(img, RIGS[name], stem)
    return entry


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    # Build into alive.new and swap it in only when every image worked, so a failing spec never leaves
    # the live site with a manifest whose frames are gone (frames of removed images do not linger either).
    final, OUT = OUT, OUT + ".new"
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    images, previews, digest = {}, [], hashlib.sha256()
    for name, spec in SPECS.items():
        try:
            images[name] = frames_of(name, spec, previews, args.preview)
        except Exception as e:
            shutil.rmtree(OUT, ignore_errors=True)
            raise SystemExit(f"{name}: {type(e).__name__}: {e} (live frames left unchanged)")
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".webp"):
            with open(os.path.join(OUT, f), "rb") as fh:
                digest.update(f.encode() + fh.read())
    # "v" changes whenever a patch does, so browsers fetch the new patches instead of cached ones.
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump({"v": digest.hexdigest()[:10], "images": images}, f, separators=(",", ":"))
    old = final + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.rename(final, old)
    os.rename(OUT, final)
    shutil.rmtree(old, ignore_errors=True)
    OUT = final
    n = sum(len(s) for e in images.values() for s in e["blink"]) + sum(len(e["talk"]) for e in images.values())
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"wrote {n} frames for {len(images)} images, {size / 1024:.0f} KB")
    if args.preview:
        preview_sheet(previews)


def preview_sheet(previews, path="alive_preview.png"):
    """Original vs. frame, cropped around the face and scaled up."""
    tiles = []
    for name, label, face, arr, orig in previews:
        boxes = list(face["boxes"]) + ([face["mouth"]] if "mouth" in face else [])
        x0 = max(0, min(b[0] for b in boxes) - 12); y0 = max(0, min(b[1] for b in boxes) - 12)
        x1 = min(orig.shape[1], max(b[2] for b in boxes) + 12); y1 = min(orig.shape[0], max(b[3] for b in boxes) + 12)
        scale = max(1, int(160 / max(1, y1 - y0)))
        pair = []
        for a in (orig, arr):
            im = Image.fromarray(a[y0:y1, x0:x1]).convert("RGB")
            pair.append(im.resize((im.width * scale, im.height * scale), Image.LANCZOS))
        tile = Image.new("RGB", (pair[0].width * 2 + 6, pair[0].height + 18), (255, 255, 255))
        tile.paste(pair[0], (0, 18)); tile.paste(pair[1], (pair[0].width + 6, 18))
        ImageDraw.Draw(tile).text((2, 2), f"{name} {label}", fill=(0, 0, 0))
        tiles.append(tile)
    cols = 3
    cw = max(t.width for t in tiles); ch = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cw * cols, ch * ((len(tiles) + cols - 1) // cols)), (230, 230, 230))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * cw, (i // cols) * ch))
    sheet.save(path)
    print("preview:", path, sheet.size)


if __name__ == "__main__":
    main()
