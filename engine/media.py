"""Photo loading + Ken Burns camera moves on stills."""
from __future__ import annotations

import functools
import json
import math
import os

import numpy as np
from PIL import Image, ImageOps

from .core import ASSETS, from_pil

IMG_DIR = os.path.join(ASSETS, "img")


@functools.lru_cache(maxsize=1)
def manifest():
    p = os.path.join(IMG_DIR, "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        data = json.load(f)
    return {os.path.splitext(d["file"])[0]: d for d in data}


def path_for(name):
    if os.path.exists(name):
        return name
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(IMG_DIR, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


@functools.lru_cache(maxsize=48)
def load(name, max_side=2400):
    im = Image.open(path_for(name))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return im


def kenburns(name, W, H, p, zoom=(1.0, 1.12), center=((0.5, 0.5), (0.5, 0.5)),
             rot=(0.0, 0.0), ease=None, resample=Image.BICUBIC, flip=False):
    """Render a cover-fitted crop of an image with animated zoom/pan/rotation.

    p      : progress 0..1 through the move
    zoom   : (start, end) zoom where 1.0 = image just covers the frame
    center : ((x0,y0),(x1,y1)) focus point in image fractions
    rot    : (start, end) degrees
    """
    im = load(name)
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if ease:
        p = ease(p)
    z = zoom[0] + (zoom[1] - zoom[0]) * p
    fx = center[0][0] + (center[1][0] - center[0][0]) * p
    fy = center[0][1] + (center[1][1] - center[0][1]) * p
    r = math.radians(rot[0] + (rot[1] - rot[0]) * p)
    iw, ih = im.size
    cover = max(W / iw, H / ih)
    s = cover * z  # output px per input px
    # keep the view inside the image where possible
    half_w, half_h = W / (2 * s), H / (2 * s)
    cxp = min(max(fx * iw, half_w), iw - half_w) if 2 * half_w < iw else iw / 2
    cyp = min(max(fy * ih, half_h), ih - half_h) if 2 * half_h < ih else ih / 2
    ca, sa = math.cos(r), math.sin(r)
    # inverse map: input = C + R^-1 * (out - O) / s
    a, b = ca / s, sa / s
    d, e = -sa / s, ca / s
    c = cxp - a * W / 2 - b * H / 2
    f = cyp - d * W / 2 - e * H / 2
    out = im.transform((W, H), Image.AFFINE, (a, b, c, d, e, f), resample=resample)
    return from_pil(out)


def contain(name, W, H, scale=1.0, bg=(0, 0, 0)):
    """Whole image fitted inside the frame (letterboxed)."""
    im = load(name)
    s = min(W / im.width, H / im.height) * scale
    r = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (W, H), tuple(int(v * 255) for v in bg))
    canvas.paste(r, ((W - r.width) // 2, (H - r.height) // 2))
    return from_pil(canvas)


def card(name, w, h, zoom=1.0, center=(0.5, 0.5), border=0, border_color=(1, 1, 1)):
    """A cover-cropped photo 'card' as RGBA (for polaroid-style drops / grids)."""
    img = kenburns(name, w, h, 0.0, (zoom, zoom), (center, center))
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = img
    if border:
        out[:border, :, :3] = border_color; out[-border:, :, :3] = border_color
        out[:, :border, :3] = border_color; out[:, -border:, :3] = border_color
    return out
