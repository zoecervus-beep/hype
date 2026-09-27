"""Director-built procedural elements (generator convention: fn(t, dur, W, H, **kw) -> RGBA)."""
from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.core import PAL, font, star_points, ease_out_expo, ease_out_cubic, ease_in_out_cubic, from_pil
from engine import text as tx

WHITE, BLACK, RED, BLUE, GOLD = PAL["white"], PAL["black"], PAL["red"], PAL["blue"], PAL["gold"]


def _rgba(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGBA"), np.float32) / 255.0


def _c8(rgb, a=255):
    return tuple(int(v * 255) for v in rgb) + (a,)


# ------------------------------------------------------------ highway -------
def highway(t, dur, W, H, speed=1.0, horizon=0.42, color=RED, glow=True, km=None):
    """Night motorway rushing towards camera: perspective lane dashes, edge lines, grid.

    Returns RGBA with an opaque dark background (sky gradient + road)."""
    S = 1
    im = Image.new("RGBA", (W, H), _c8(BLACK))
    d = ImageDraw.Draw(im)
    hy = int(H * horizon)
    # sky gradient
    sky = np.zeros((hy, W, 4), np.float32)
    g = np.linspace(0, 1, hy, dtype=np.float32)[:, None]
    sky[..., 0] = 0.02 + 0.45 * g ** 3
    sky[..., 1] = 0.01 + 0.02 * g ** 3
    sky[..., 2] = 0.05 + 0.08 * g
    sky[..., 3] = 1
    im.paste(Image.fromarray((sky * 255).astype(np.uint8), "RGBA"), (0, 0))
    # sun / star on the horizon
    r = H * 0.16
    cx = W / 2
    d.polygon(star_points(cx, hy - r * 0.35, r, r * 0.42), fill=_c8(RED))
    # ground
    d.rectangle([0, hy, W, H], fill=_c8((0.03, 0.02, 0.03)))
    vx = W / 2
    road_w = W * 1.3
    # grid lines (horizontal, moving)
    z_off = (t * speed * 1.8) % 1.0
    for k in range(24):
        z = (k + 1 - z_off)
        yy = hy + (H - hy) * (1.0 / (z * 0.35 + 0.001)) * 0.12
        if yy > H or yy <= hy:
            continue
        a = int(200 * min(1, (yy - hy) / (H - hy) * 2.5))
        d.line([(0, yy), (W, yy)], fill=(int(255 * color[0]), int(255 * color[1] * 0.4), int(255 * color[2] * 0.5), a // 3), width=1)
    # vertical perspective lines
    for k in range(-14, 15):
        xb = vx + k * W * 0.11
        d.line([(vx + k * 4, hy), (xb, H)], fill=(120, 20, 30, 90), width=1)
    # road surface
    d.polygon([(vx - 6, hy), (vx + 6, hy), (vx + road_w / 2, H), (vx - road_w / 2, H)], fill=(16, 14, 18, 255))
    # edge lines
    for side in (-1, 1):
        d.line([(vx + side * 6, hy), (vx + side * road_w / 2, H)], fill=_c8(WHITE), width=max(2, W // 400))
    # centre dashes
    n = 16
    ph = (t * speed * 2.4) % 1.0
    for k in range(n):
        z0 = (k + ph) / n
        z1 = (k + ph + 0.45) / n
        y0 = hy + (H - hy) * z0 ** 2.2
        y1 = hy + (H - hy) * min(z1, 1.0) ** 2.2
        w0 = 2 + 26 * z0 ** 2.2 * W / 1920
        w1 = 2 + 26 * min(z1, 1) ** 2.2 * W / 1920
        d.polygon([(vx - w0, y0), (vx + w0, y0), (vx + w1, y1), (vx - w1, y1)], fill=_c8(GOLD))
    out = _rgba(im)
    if glow:
        small = im.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
        gl = _rgba(small.resize((W, H), Image.BILINEAR))
        out[..., :3] = np.clip(out[..., :3] + gl[..., :3] * 0.8, 0, 1)
    return out


# --------------------------------------------------------- bar chart --------
def bar_chart(t, dur, W, H, values=(20, 45, 70, 100), labels=("1950", "1960", "1970", "1980"),
              color=RED, hits=None, title=None, unit=""):
    """Bars shoot up one by one (at `hits` local times) with value labels."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    n = len(values)
    hits = hits or [dur * 0.15 * (k + 1) for k in range(n)]
    base_y = H * 0.82
    bw = W * 0.12
    gap = W * 0.05
    x0 = W / 2 - (n * bw + (n - 1) * gap) / 2
    vmax = max(values)
    f_lab = font("mono", int(H * 0.032))
    f_val = font("anton", int(H * 0.08))
    for k, (v, lab) in enumerate(zip(values, labels)):
        p = ease_out_expo(max(0, min((t - hits[k]) / 0.25, 1)))
        h = (H * 0.55) * v / vmax * p
        x = x0 + k * (bw + gap)
        if p > 0:
            d.rectangle([x, base_y - h, x + bw, base_y], fill=_c8(color))
            d.rectangle([x, base_y - h, x + bw, base_y - h + max(3, H * 0.006)], fill=_c8(GOLD))
            vs = f"{v * p:.0f}{unit}"
            tw = d.textlength(vs, font=f_val)
            d.text((x + bw / 2 - tw / 2, base_y - h - H * 0.1), vs, font=f_val, fill=_c8(WHITE))
        tw = d.textlength(lab, font=f_lab)
        d.text((x + bw / 2 - tw / 2, base_y + H * 0.02), lab, font=f_lab, fill=_c8(WHITE, 200))
    d.line([(x0 - gap, base_y), (x0 + n * (bw + gap), base_y)], fill=_c8(WHITE, 220), width=max(2, H // 300))
    return _rgba(im)


# ------------------------------------------------------------ religions -----
def faith_symbols(t, dur, W, H, hits=(0.0, 0.15, 0.3), color=WHITE, size=0.26):
    """Orthodox cross, Latin cross, crescent & star — popping on at `hits`."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = H * size
    xs = (0.28, 0.5, 0.72)
    cy = H * 0.5
    lw = max(4, int(s * 0.1))
    for k, cx in enumerate(xs):
        p = (t - hits[k]) / 0.12
        if p < 0:
            continue
        sc = 1 + 1.2 * (1 - ease_out_expo(min(p, 1)))
        ss = s * sc
        X = W * cx
        col = _c8(color)
        if k == 0:  # orthodox: vertical, top bar, main bar, slanted foot
            d.rectangle([X - lw / 2, cy - ss / 2, X + lw / 2, cy + ss / 2], fill=col)
            d.rectangle([X - ss * 0.16, cy - ss * 0.36 - lw / 2, X + ss * 0.16, cy - ss * 0.36 + lw / 2], fill=col)
            d.rectangle([X - ss * 0.3, cy - ss * 0.2 - lw / 2, X + ss * 0.3, cy - ss * 0.2 + lw / 2], fill=col)
            d.line([(X - ss * 0.2, cy + ss * 0.22), (X + ss * 0.2, cy + ss * 0.1)], fill=col, width=lw)
        elif k == 1:  # latin cross
            d.rectangle([X - lw / 2, cy - ss / 2, X + lw / 2, cy + ss / 2], fill=col)
            d.rectangle([X - ss * 0.3, cy - ss * 0.22 - lw / 2, X + ss * 0.3, cy - ss * 0.22 + lw / 2], fill=col)
        else:  # crescent + star
            r = ss * 0.42
            d.ellipse([X - r, cy - r, X + r, cy + r], fill=col)
            r2 = r * 0.82
            ox = r * 0.32
            d.ellipse([X - r2 + ox, cy - r2, X + r2 + ox, cy + r2], fill=(0, 0, 0, 0))
            d.polygon(star_points(X + r * 0.62, cy, r * 0.28, r * 0.11, rot=math.pi / 2 * 0.2), fill=col)
    return _rgba(im)


# ------------------------------------------------------------- numerals -----
def big_number(t, dur, W, H, num="7", word="SUSEDA", sub="7 NEIGHBOURS", color=RED,
               word_color=WHITE, num_font="anton", word_font="anton", layout="left"):
    """Countdown layout: giant numeral + word. Slam in over first 4 frames."""
    im = np.zeros((H, W, 4), np.float32)
    p = ease_out_expo(min(t / 0.12, 1))
    s = 1 + 1.4 * (1 - p)
    Ln = tx.text_layer(num, num_font, int(H * 0.95), color)
    Lw = tx.text_layer(word, word_font, int(H * 0.26), word_color)
    if layout == "left":
        nx, wx = 0.3, 0.64
        Lw = tx.text_layer(word, word_font, min(int(H * 0.26), tx.fit_size(word, word_font, W * 0.52, H * 0.3)), word_color)
    else:
        nx, wx = 0.5, 0.5
    _place_rgba(im, Ln, nx, 0.5, s)
    q = ease_out_expo(max(0, min((t - 0.04) / 0.14, 1)))
    _place_rgba(im, Lw, wx + (1 - q) * 0.2, 0.5 if layout == "left" else 0.82, 1.0, q)
    return im


def _place_rgba(dst, L, cx, cy, scale=1.0, opacity=1.0):
    """Composite RGBA onto an RGBA canvas (straight alpha)."""
    H, W = dst.shape[:2]
    L = tx.transform_layer(L, scale)
    h, w = L.shape[:2]
    x, y = int(cx * W - w / 2), int(cy * H - h / 2)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return dst
    s = L[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    sa = s[..., 3:4] * opacity
    da = d[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    d[..., :3] = rgb
    d[..., 3:4] = oa
    return dst


# ------------------------------------------------------------- tickers ------
def ticker(t, dur, W, H, items=(), speed=0.9, size=0.07, y=0.1, color=WHITE, sep="  ★  ", fname="anton",
           bg=RED):
    """Scrolling news ticker band."""
    s = sep.join(items) + sep
    L = tx.text_layer(s * 3, fname, int(H * size), color)
    h, w = L.shape[:2]
    out = np.zeros((H, W, 4), np.float32)
    bh = int(h * 1.5)
    y0 = int(y * H - bh / 2)
    out[max(y0, 0):y0 + bh, :, :3] = bg
    out[max(y0, 0):y0 + bh, :, 3] = 1
    period = w / 3
    x = -int((t * speed * W) % period)
    _blit(out, L, x, y0 + (bh - h) // 2)
    return out


def _blit(dst, L, x, y):
    H, W = dst.shape[:2]
    h, w = L.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = L[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4]
    dst[y0:y1, x0:x1, :3] = dst[y0:y1, x0:x1, :3] * (1 - a) + s[..., :3] * a
    dst[y0:y1, x0:x1, 3:4] = np.maximum(dst[y0:y1, x0:x1, 3:4], a)


# ---------------------------------------------------------- crt / lines -----
def crt_on(t, dur, W, H, color=WHITE):
    """TV switch-on: a horizontal line that expands, then opens vertically."""
    out = np.zeros((H, W, 4), np.float32)
    p = t / dur
    if p < 0.5:
        w = int(W * ease_out_expo(p * 2))
        h = max(2, H // 300)
    else:
        w = W
        h = max(2, int(H * ease_out_expo((p - 0.5) * 2)))
    y0 = H // 2 - h // 2
    x0 = W // 2 - w // 2
    out[y0:y0 + h, x0:x0 + w, :3] = color
    out[y0:y0 + h, x0:x0 + w, 3] = 1
    return out


def film_burn(t, dur, W, H, seed=1):
    """Warm burn-out blotches growing from the frame edge."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    p = t / dur
    acc = np.zeros_like(xx)
    for _ in range(4):
        cx, cy = rng.uniform(-0.1, 1.1) * W / 4, rng.choice([-0.1, 1.1]) * H / 4
        r = (0.2 + p * 1.2) * W / 4 * rng.uniform(0.6, 1.2)
        acc += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r))
    acc = np.clip(acc, 0, 1)
    im = Image.fromarray((acc * 255).astype(np.uint8), "L").resize((W, H), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    out = np.zeros((H, W, 4), np.float32)
    out[..., 0] = 1.0
    out[..., 1] = 0.35 + 0.6 * a
    out[..., 2] = 0.1 + 0.8 * a ** 3
    out[..., 3] = np.clip(a * 1.3, 0, 1)
    return out
