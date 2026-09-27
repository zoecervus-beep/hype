"""Frame effects. Everything takes / returns float32 RGB (H, W, 3) unless noted."""
from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageFilter

from .core import PAL, from_pil, to_pil, to_u8

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(img):
    return img[..., :3] @ LUMA


# ------------------------------------------------------------- geometry ---
def affine(img, zoom=1.0, rot=0.0, dx=0.0, dy=0.0, cx=0.5, cy=0.5,
           sx=1.0, sy=1.0, resample=Image.BILINEAR, fill=(0, 0, 0)):
    """Zoom/rotate (degrees)/translate (pixels) the frame about (cx, cy) in 0..1."""
    if zoom == 1 and rot == 0 and dx == 0 and dy == 0 and sx == 1 and sy == 1:
        return img
    H, W = img.shape[:2]
    px, py = cx * W, cy * H
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    zx, zy = zoom * sx, zoom * sy
    # output -> input mapping (inverse transform)
    ia, ib = ca / zx, sa / zx
    id_, ie = -sa / zy, ca / zy
    ox, oy = px + dx, py + dy
    c = px - ia * ox - ib * oy
    f = py - id_ * ox - ie * oy
    im = to_pil(img)
    fc = tuple(int(v * 255) for v in fill)
    out = im.transform((W, H), Image.AFFINE, (ia, ib, c, id_, ie, f),
                       resample=resample, fillcolor=fc if im.mode == "RGB" else fc + (0,))
    return from_pil(out)


def shift(img, dx, dy):
    """Integer translate with edge clamp (cheap camera shake)."""
    dx, dy = int(round(dx)), int(round(dy))
    if dx == 0 and dy == 0:
        return img
    H, W = img.shape[:2]
    ys = np.clip(np.arange(H) - dy, 0, H - 1)
    xs = np.clip(np.arange(W) - dx, 0, W - 1)
    return img[ys][:, xs]


def mirror(img, mode="h"):
    out = img.copy()
    H, W = img.shape[:2]
    if mode in ("h", "quad"):
        out[:, W // 2:] = out[:, :W - W // 2][:, ::-1]
    if mode in ("v", "quad"):
        out[H // 2:] = out[:H - H // 2][::-1]
    return out


def grid_repeat(img, n=3, gap=0, bg=(0, 0, 0)):
    """n x n tiled copies of the frame (the 'multi-screen' beat effect)."""
    H, W = img.shape[:2]
    small = from_pil(to_pil(img).resize((W // n, H // n), Image.BILINEAR))
    out = np.empty_like(img)
    out[:] = bg
    h, w = small.shape[:2]
    for j in range(n):
        for i in range(n):
            y, x = j * (H // n), i * (W // n)
            out[y + gap:y + h - gap, x + gap:x + w - gap] = small[gap:h - gap, gap:w - gap]
    return out


def split_screen(frames, axis="v"):
    """Concatenate equal-size frames into vertical (side by side) or horizontal strips."""
    H, W = frames[0].shape[:2]
    n = len(frames)
    out = np.empty_like(frames[0])
    for k, f in enumerate(frames):
        if axis == "v":
            x0, x1 = k * W // n, (k + 1) * W // n
            off = (W - (x1 - x0)) // 2
            out[:, x0:x1] = f[:, off:off + (x1 - x0)]
        else:
            y0, y1 = k * H // n, (k + 1) * H // n
            off = (H - (y1 - y0)) // 2
            out[y0:y1] = f[off:off + (y1 - y0)]
    return out


# ---------------------------------------------------------------- colour ---
def grade_bw(img, contrast=1.3, lift=0.0, gamma=1.0):
    y = luma(img)
    y = np.clip((y - 0.5) * contrast + 0.5 + lift, 0, 1) ** gamma
    return np.repeat(y[..., None], 3, axis=2)


def gradient_map(img, stops):
    """Map luminance through colour stops [(pos, (r,g,b)), ...]."""
    y = np.clip(luma(img), 0, 1)
    lut = _lut(tuple((p, tuple(c)) for p, c in stops))
    idx = (y * 255).astype(np.uint8)
    return lut[idx]


@functools.lru_cache(maxsize=64)
def _lut(stops):
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    g = np.linspace(0, 1, 256, dtype=np.float32)
    return np.stack([np.interp(g, xs, cs[:, k]) for k in range(3)], axis=1).astype(np.float32)


def duotone(img, dark=PAL["black"], light=PAL["red"], contrast=1.4):
    y = np.clip((luma(img) - 0.5) * contrast + 0.5, 0, 1)[..., None]
    d = np.asarray(dark, np.float32)
    l = np.asarray(light, np.float32)
    return d + (l - d) * y


GRADES = {
    "red": [(0, PAL["black"]), (0.45, PAL["deep_red"]), (0.8, PAL["red"]), (1, (1, .85, .8))],
    "blue": [(0, PAL["black"]), (0.5, PAL["navy"]), (0.85, PAL["blue"]), (1, (0.85, 0.9, 1))],
    "gold": [(0, PAL["black"]), (0.5, (0.35, 0.18, 0.02)), (0.85, PAL["gold"]), (1, (1, 1, 0.9))],
    "flag": [(0, PAL["navy"]), (0.35, PAL["blue"]), (0.62, PAL["white"]), (0.8, PAL["red"]), (1, (1, .9, .8))],
    "thermal": [(0, (0, 0, 0.1)), (0.3, (0.3, 0, 0.5)), (0.55, (0.9, 0.1, 0.1)), (0.8, (1, 0.7, 0)), (1, (1, 1, 1))],
    "steel": [(0, (0.02, 0.02, 0.03)), (0.6, (0.45, 0.5, 0.58)), (1, (0.95, 0.97, 1))],
    "neon": [(0, (0.02, 0, 0.05)), (0.4, (0.5, 0, 0.35)), (0.7, PAL["magenta"]), (1, PAL["cyan"])],
    "paper": [(0, (0.08, 0.06, 0.06)), (0.55, (0.55, 0.5, 0.45)), (1, PAL["white"])],
}


def grade(img, name, contrast=1.25, mix=1.0):
    if name in (None, "none"):
        return img
    if name == "bw":
        out = grade_bw(img, contrast)
    elif name == "invert":
        out = 1.0 - img
    else:
        y = np.clip((luma(img) - 0.5) * contrast + 0.5, 0, 1)
        lut = _lut(tuple((p, tuple(c)) for p, c in GRADES[name]))
        out = lut[(y * 255).astype(np.uint8)]
    return out if mix >= 1 else img + (out - img) * mix


def levels(img, black=0.0, white=1.0, gamma=1.0):
    return np.clip((img - black) / max(white - black, 1e-4), 0, 1) ** gamma


def saturate(img, s=1.2):
    y = luma(img)[..., None]
    return np.clip(y + (img - y) * s, 0, 1)


def tint(img, rgb, amt=0.5, mode="multiply"):
    c = np.asarray(rgb, np.float32)
    if mode == "multiply":
        t = img * c
    elif mode == "screen":
        t = 1 - (1 - img) * (1 - c)
    else:
        t = np.broadcast_to(c, img.shape)
    return img + (t - img) * amt


def posterize(img, levels_=4):
    return np.floor(img * levels_) / (levels_ - 1 + 1e-6)


def threshold(img, level=0.5, dark=PAL["black"], light=PAL["white"]):
    m = (luma(img) > level)[..., None]
    return np.where(m, np.asarray(light, np.float32), np.asarray(dark, np.float32)).astype(np.float32)


def flash(img, rgb=(1, 1, 1), amt=1.0):
    if amt <= 0:
        return img
    c = np.asarray(rgb, np.float32)
    return img + (c - img) * min(amt, 1.0)


def invert(img):
    return 1.0 - img


# ------------------------------------------------------------ blur/glow ---
def blur(img, radius):
    if radius <= 0.3:
        return img
    return from_pil(to_pil(img).filter(ImageFilter.GaussianBlur(radius)))


def blur_fast(img, radius, down=4):
    """Blur at reduced resolution (cheap large glows)."""
    H, W = img.shape[:2]
    im = to_pil(img)
    small = im.resize((max(W // down, 1), max(H // down, 1)), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(max(radius / down, 0.5)))
    return from_pil(small.resize((W, H), Image.BILINEAR))


def bloom(img, thresh=0.7, strength=0.8, radius=24):
    bright = np.clip((img - thresh) / (1 - thresh + 1e-4), 0, 1)
    return np.clip(img + blur_fast(bright, radius) * strength, 0, 1)


def zoom_blur(img, strength=0.08, cx=0.5, cy=0.5, n=6):
    """Radial (zoom) blur: average of progressively scaled copies."""
    H, W = img.shape[:2]
    im = to_pil(img)
    acc = np.asarray(im, np.float32)
    for k in range(1, n):
        z = 1 + strength * k / (n - 1)
        w, h = int(W * z), int(H * z)
        big = im.resize((w, h), Image.BILINEAR)
        x0 = int(cx * (w - W)); y0 = int(cy * (h - H))
        acc += np.asarray(big.crop((x0, y0, x0 + W, y0 + H)), np.float32)
    return acc / (n * 255.0)


def motion_blur(img, dx, dy, n=8):
    """Directional blur along (dx, dy) pixels."""
    if abs(dx) < 1 and abs(dy) < 1:
        return img
    acc = np.zeros_like(img)
    for k in range(n):
        f = k / (n - 1) - 0.5
        acc += shift(img, dx * f, dy * f)
    return acc / n


def rgb_split(img, amt=8, angle=0.0):
    """Chromatic aberration: R and B pushed in opposite directions (pixels)."""
    if abs(amt) < 0.5:
        return img
    dx, dy = amt * math.cos(angle), amt * math.sin(angle)
    out = img.copy()
    out[..., 0] = shift(img[..., 0:1], dx, dy)[..., 0]
    out[..., 2] = shift(img[..., 2:3], -dx, -dy)[..., 0]
    return out


def radial_rgb_split(img, amt=0.012):
    """Lens-style CA growing with distance from centre (zoom R/B channels)."""
    H, W = img.shape[:2]
    out = img.copy()
    for ch, z in ((0, 1 + amt), (2, 1 - amt)):
        im = Image.fromarray(to_u8(img[..., ch]), "L")
        w, h = int(W * z), int(H * z)
        r = im.resize((w, h), Image.BILINEAR)
        if z > 1:
            x0, y0 = (w - W) // 2, (h - H) // 2
            r = r.crop((x0, y0, x0 + W, y0 + H))
        else:
            canvas = Image.new("L", (W, H))
            canvas.paste(r, ((W - w) // 2, (H - h) // 2))
            r = canvas
        out[..., ch] = np.asarray(r, np.float32) / 255.0
    return out


def edges(img, color=PAL["red"], gain=3.0, bg=None):
    """Neon edge detection (Sobel on luma)."""
    y = luma(img)
    gx = np.zeros_like(y); gy = np.zeros_like(y)
    gx[:, 1:-1] = y[:, 2:] - y[:, :-2]
    gy[1:-1] = y[2:] - y[:-2]
    m = np.clip(np.sqrt(gx * gx + gy * gy) * gain, 0, 1)[..., None]
    base = np.zeros_like(img) if bg is None else np.broadcast_to(np.asarray(bg, np.float32), img.shape)
    return base + (np.asarray(color, np.float32) - base) * m


# ---------------------------------------------------------------- texture ---
def halftone(img, cell=12, angle=0.26, ink=PAL["black"], paper=PAL["white"], gain=1.0):
    """Rotated-grid dot halftone driven by luminance (dark = big dots)."""
    H, W = img.shape[:2]
    y = luma(img)
    yy, xx = _grid(H, W)
    ca, sa = math.cos(angle), math.sin(angle)
    u = (xx * ca + yy * sa) / cell
    v = (-xx * sa + yy * ca) / cell
    fu, fv = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
    d = np.sqrt(fu * fu + fv * fv)
    # sample luminance at the cell centre (approx: blurred luma)
    small = Image.fromarray(to_u8(y), "L").resize((max(W // cell, 1), max(H // cell, 1)), Image.BILINEAR)
    yb = np.asarray(small.resize((W, H), Image.BILINEAR), np.float32) / 255.0
    r = np.sqrt(np.clip(1 - yb, 0, 1) * gain) * 0.62
    m = np.clip((r - d) * cell * 0.9 + 0.5, 0, 1)[..., None]
    p = np.asarray(paper, np.float32); k = np.asarray(ink, np.float32)
    return p + (k - p) * m


def pixelate(img, block=16):
    H, W = img.shape[:2]
    im = to_pil(img)
    small = im.resize((max(W // block, 1), max(H // block, 1)), Image.BILINEAR)
    return from_pil(small.resize((W, H), Image.NEAREST))


def scanlines(img, strength=0.25, period=4):
    H = img.shape[0]
    m = _scan(H, period, strength)
    return img * m[:, None, None]


@functools.lru_cache(maxsize=8)
def _scan(H, period, strength):
    y = np.arange(H)
    return (1 - strength * (0.5 + 0.5 * np.cos(2 * np.pi * y / period))).astype(np.float32)


@functools.lru_cache(maxsize=4)
def _grid(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return yy, xx


@functools.lru_cache(maxsize=4)
def _vignette(H, W, strength, roundness):
    yy, xx = _grid(H, W)
    nx = (xx / W - 0.5) * 2
    ny = (yy / H - 0.5) * 2 * (roundness)
    d = np.sqrt(nx * nx + ny * ny) / math.sqrt(2)
    return (1 - strength * np.clip(d, 0, 1) ** 2.2).astype(np.float32)


def vignette(img, strength=0.5, roundness=0.9):
    H, W = img.shape[:2]
    return img * _vignette(H, W, strength, roundness)[..., None]


@functools.lru_cache(maxsize=2)
def _grain_bank(H, W, n=6, seed=7):
    rng = np.random.default_rng(seed)
    h, w = H // 2 + 1, W // 2 + 1
    bank = []
    for _ in range(n):
        g = rng.standard_normal((h, w)).astype(np.float32)
        g = np.repeat(np.repeat(g, 2, 0), 2, 1)[:H, :W]
        bank.append(g.astype(np.float16))
    return bank


def grain(img, amount=0.05, frame=0):
    H, W = img.shape[:2]
    g = _grain_bank(H, W)[frame % 6].astype(np.float32)
    # stronger in mid-tones like film
    y = luma(img)
    w = 1 - np.abs(y - 0.5) * 1.2
    return img + (g * amount * w)[..., None]


def glitch_slices(img, amount=1.0, seed=0, n=14, max_shift=0.12, rgb=True):
    """Horizontal slice displacement + per-slice channel offsets."""
    if amount <= 0:
        return img
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(int(n * amount) + 1):
        h = int(rng.uniform(0.005, 0.08) * H)
        y = int(rng.uniform(0, H - h))
        s = int(rng.uniform(-1, 1) * max_shift * W * amount)
        band = np.roll(img[y:y + h], s, axis=1)
        if rgb and rng.random() < 0.5:
            band = band.copy()
            band[..., 0] = np.roll(band[..., 0], int(rng.uniform(5, 30) * amount), axis=1)
            band[..., 2] = np.roll(band[..., 2], -int(rng.uniform(5, 30) * amount), axis=1)
        out[y:y + h] = band
    return out


def block_glitch(img, amount=1.0, seed=0, n=10):
    if amount <= 0:
        return img
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(int(n * amount)):
        w = int(rng.uniform(0.04, 0.3) * W); h = int(rng.uniform(0.02, 0.12) * H)
        x = int(rng.uniform(0, W - w)); y = int(rng.uniform(0, H - h))
        sx = int(np.clip(x + rng.uniform(-0.2, 0.2) * W, 0, W - w))
        sy = int(np.clip(y + rng.uniform(-0.1, 0.1) * H, 0, H - h))
        blk = img[sy:sy + h, sx:sx + w]
        mode = rng.integers(0, 4)
        if mode == 0:
            blk = 1 - blk
        elif mode == 1:
            blk = blk[..., ::-1]
        elif mode == 2:
            blk = blk * np.asarray(PAL["red"], np.float32) * 1.4
        out[y:y + h, x:x + w] = blk
    return out


def vhs(img, t, amount=1.0, seed=0):
    """Tracking wobble + colour bleed + noise bars."""
    H, W = img.shape[:2]
    out = img.copy()
    rng = np.random.default_rng(seed)
    y0 = int(((t * 0.37 + rng.random()) % 1.0) * H)
    h = int(H * 0.06)
    if amount > 0:
        band = slice(y0, min(y0 + h, H))
        out[band] = np.roll(out[band], int(18 * amount), axis=1) * 0.8 + 0.15 * amount
    out[..., 0] = shift(out[..., 0:1], 3 * amount, 0)[..., 0]
    return scanlines(out, 0.18 * amount, 3)


# --------------------------------------------------------------- overlays ---
def speed_lines(W, H, t, n=90, color=(1, 1, 1), inner=0.28, seed=3, alpha=0.9):
    """Manga-style radial speed lines as RGBA layer (re-seeded every 2 frames)."""
    from PIL import ImageDraw
    rng = np.random.default_rng(seed + int(t * 15))
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    cx, cy = W / 2, H / 2
    R = math.hypot(W, H)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        w = rng.uniform(0.003, 0.02)
        r0 = R * rng.uniform(inner, inner + 0.25)
        p = [(cx + math.cos(a - w) * R, cy + math.sin(a - w) * R),
             (cx + math.cos(a) * r0, cy + math.sin(a) * r0),
             (cx + math.cos(a + w) * R, cy + math.sin(a + w) * R)]
        d.polygon(p, fill=int(255 * rng.uniform(0.5, 1)))
    a = np.asarray(im, np.float32) / 255.0 * alpha
    out = np.empty((H, W, 4), np.float32)
    out[..., :3] = color
    out[..., 3] = a
    return out


def light_leak(W, H, t, color=(1.0, 0.35, 0.1), strength=0.6, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = _grid(H, W)
    cx = W * (0.2 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.9 + rng.random() * 6)))
    cy = H * (0.3 + 0.4 * (0.5 + 0.5 * math.cos(t * 0.6 + rng.random() * 6)))
    r = W * 0.45
    d = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
    m = np.exp(-d) * strength
    out = np.empty((H, W, 4), np.float32)
    out[..., :3] = color
    out[..., 3] = m
    return out
