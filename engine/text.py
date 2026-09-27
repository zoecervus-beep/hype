"""Kinetic typography."""
from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageDraw

from .core import PAL, font, over, from_pil, ease_out_expo, ease_out_back

LAT2CYR = {
    "LJ": "Љ", "NJ": "Њ", "DŽ": "Џ", "A": "А", "B": "Б", "V": "В", "G": "Г", "D": "Д",
    "Đ": "Ђ", "E": "Е", "Ž": "Ж", "Z": "З", "I": "И", "J": "Ј", "K": "К", "L": "Л",
    "M": "М", "N": "Н", "O": "О", "P": "П", "R": "Р", "S": "С", "T": "Т", "Ć": "Ћ",
    "U": "У", "F": "Ф", "H": "Х", "C": "Ц", "Č": "Ч", "Š": "Ш",
}


def to_cyrillic(s: str) -> str:
    out, i, s = [], 0, s.upper()
    while i < len(s):
        if s[i:i + 2] in LAT2CYR:
            out.append(LAT2CYR[s[i:i + 2]]); i += 2
        else:
            out.append(LAT2CYR.get(s[i], s[i])); i += 1
    return "".join(out)


@functools.lru_cache(maxsize=512)
def _render(text, fname, size, color, tracking, stroke, stroke_color, outline_only, axes):
    f = font(fname, size, axes)
    # measure per-glyph to apply tracking
    widths = [f.getlength(ch) for ch in text]
    track_px = tracking * size
    total_w = sum(widths) + track_px * max(len(text) - 1, 0)
    asc, desc = f.getmetrics()
    pad = stroke + 4
    W = int(math.ceil(total_w)) + 2 * pad
    H = asc + desc + 2 * pad
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    smask = Image.new("L", (W, H), 0) if stroke else None
    ds = ImageDraw.Draw(smask) if stroke else None
    x = pad
    for ch, w in zip(text, widths):
        if stroke:
            ds.text((x, pad), ch, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        d.text((x, pad), ch, font=f, fill=255)
        x += w + track_px
    fill = np.asarray(mask, np.float32) / 255.0
    out = np.zeros((H, W, 4), np.float32)
    if stroke:
        s = np.asarray(smask, np.float32) / 255.0
        if outline_only:
            out[..., :3] = stroke_color
            out[..., 3] = np.clip(s - fill, 0, 1)
        else:
            out[..., :3] = np.asarray(stroke_color, np.float32)
            out[..., :3] = out[..., :3] * (1 - fill[..., None]) + np.asarray(color, np.float32) * fill[..., None]
            out[..., 3] = np.maximum(s, fill)
    else:
        out[..., :3] = color
        out[..., 3] = fill
    # trim empty rows/cols
    ys, xs = np.nonzero(out[..., 3] > 0.01)
    if len(ys):
        out = out[max(ys.min() - 2, 0):ys.max() + 3, max(xs.min() - 2, 0):xs.max() + 3]
    out.setflags(write=False)
    return out


def text_layer(text, fname="anton", size=200, color=PAL["white"], tracking=0.0,
               stroke=0, stroke_color=PAL["black"], outline_only=False, axes=None):
    """RGBA layer tightly cropped around the rendered text."""
    return _render(text, fname, int(size), tuple(color), float(tracking), int(stroke),
                   tuple(stroke_color), bool(outline_only), tuple(axes) if axes else None)


def fit_size(text, fname, max_w, max_h, tracking=0.0, axes=None):
    """Largest pixel size so text fits in max_w x max_h."""
    probe = 200
    lay = text_layer(text, fname, probe, tracking=tracking, axes=axes)
    h, w = lay.shape[:2]
    return max(8, int(probe * min(max_w / max(w, 1), max_h / max(h, 1))))


def transform_layer(layer, scale=1.0, rot=0.0, sx=1.0, sy=1.0):
    if scale == 1 and rot == 0 and sx == 1 and sy == 1:
        return layer
    h, w = layer.shape[:2]
    nw, nh = max(1, int(w * scale * sx)), max(1, int(h * scale * sy))
    im = Image.fromarray((np.clip(layer, 0, 1) * 255).astype(np.uint8), "RGBA")
    im = im.resize((nw, nh), Image.BICUBIC if scale < 3 else Image.BILINEAR)
    if rot:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    return from_pil(im)


def place(frame, layer, cx=0.5, cy=0.5, scale=1.0, rot=0.0, opacity=1.0,
          sx=1.0, sy=1.0, mode="normal", anchor="c"):
    """Composite a layer onto the frame, positioned by its centre (fractions of frame)."""
    H, W = frame.shape[:2]
    L = transform_layer(layer, scale, rot, sx, sy)
    h, w = L.shape[:2]
    x = int(cx * W - (w / 2 if anchor in ("c", "t", "b") else (0 if anchor == "l" else w)))
    y = int(cy * H - (h / 2 if anchor in ("c", "l", "r") else (0 if anchor == "t" else h)))
    over(frame, L, x, y, opacity, mode)
    return (x, y, w, h)


def slam(frame, layer, p, cx=0.5, cy=0.5, s0=3.0, s1=1.0, rot0=0.0, rot1=0.0, opacity=1.0, mode="normal"):
    """Scale-in impact: big + transparent -> settled. p in [0,1] over the slam (~3-5 frames)."""
    e = ease_out_expo(p)
    s = s0 + (s1 - s0) * e
    return place(frame, layer, cx, cy, s, rot0 + (rot1 - rot0) * e, opacity * min(1.0, 0.25 + p * 3), mode=mode)


def echo_stack(frame, layer, cx, cy, n=5, dy=0.11, fade=0.55, scale=1.0, outline=None, mode="normal"):
    """Repeat a word vertically (centre solid, echoes fading)."""
    for k in range(n, 0, -1):
        a = fade ** k
        L = outline if outline is not None else layer
        place(frame, L, cx, cy - dy * k, scale, opacity=a, mode=mode)
        place(frame, L, cx, cy + dy * k, scale, opacity=a, mode=mode)
    place(frame, layer, cx, cy, scale, mode=mode)


def typewriter(text, t, cps=18.0, cursor=True, blink=0.5):
    n = int(t * cps)
    s = text[:n]
    if cursor and (n < len(text) or (t % (2 * blink)) < blink):
        s += "█"
    return s


def scramble(text, p, seed=0, charset="ABCČĆDĐEFGHIJKLMNOPRSŠTUVZŽ0123456789#%&@"):
    """Decode effect: random glyphs that resolve left to right as p goes 0 -> 1."""
    rng = np.random.default_rng(seed + int(p * 40))
    n_fixed = int(len(text) * min(max(p, 0), 1))
    out = []
    for i, ch in enumerate(text):
        if i < n_fixed or ch == " ":
            out.append(ch)
        else:
            out.append(charset[rng.integers(0, len(charset))])
    return "".join(out)


def mask_image_with_text(img, layer, cx=0.5, cy=0.5, scale=1.0):
    """Return RGBA: the image visible only through the text glyphs."""
    H, W = img.shape[:2]
    L = transform_layer(layer, scale)
    h, w = L.shape[:2]
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = img
    x, y = int(cx * W - w / 2), int(cy * H - h / 2)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1, 3] = L[y0 - y:y1 - y, x0 - x:x1 - x, 3]
    return out


def caption(frame, text, cy=0.86, size_frac=0.028, color=PAL["white"], bar=True,
            bar_color=PAL["black"], fname="mono", opacity=1.0, cx=0.5, tracking=0.12):
    """Small English subtitle in mono with a solid bar behind it."""
    H, W = frame.shape[:2]
    L = text_layer(text, fname, int(H * size_frac), color, tracking=tracking)
    h, w = L.shape[:2]
    if bar:
        px, py = int(h * 0.6), int(h * 0.35)
        x0 = int(cx * W - w / 2 - px); y0 = int(cy * H - h / 2 - py)
        x1 = int(cx * W + w / 2 + px); y1 = int(cy * H + h / 2 + py)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        frame[y0:y1, x0:x1] += (np.asarray(bar_color, np.float32) - frame[y0:y1, x0:x1]) * (0.85 * opacity)
    place(frame, L, cx, cy, opacity=opacity)
    return frame


def counter_text(value, fmt="{:.0f}"):
    return fmt.format(value)
