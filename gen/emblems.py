"""gen/emblems.py -- 2-D emblem / prop generators for the SFRJ hype edit.

Every public function follows the engine/core.py generator convention:

    fn(t, dur, W, H, **params) -> float32 straight-alpha RGBA (H, W, 4)

deterministic in t, transparent background unless stated.

    flag          waving SFRY flag (blue / white / red + gold-bordered red star)
    coat_of_arms  stylised SFRY emblem: 6 torches, one flame, wheat wreath, star,
                  blue ribbon "29. XI 1943." (animated build-up via `reveal`)
    star2d        flat red star, gold outline, sweeping shine band, glow / pulse
    passport      red Yugoslav passport, flips open, visa stamps slam in
    fico_drift    top-down Zastava 750 drifting through tyre smoke
    test_card     1970s JRT / TV Beograd test card with running clock + glitch
    vinyl         spinning Jugoton LP
    snowflake     six-fold Sarajevo '84 style snowflake that draws itself on
    basketball    rotating basketball with motion lines

Drawing trick: shapes are rasterised by PIL into a supersampled "CMYK"-mode image
that really holds *premultiplied* RGBA bytes. PIL never applies alpha semantics to
CMYK, so reduce()/transform()/paste(mask) are fast and exactly premultiplied-correct.
"""
from __future__ import annotations

import functools
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.core import (PAL, OUT, font, hexrgb, star_points, ease_out_back,  # noqa: E402
                         ease_out_cubic, ease_in_out_cubic, ease_in_cubic)

GOLD = PAL["gold"]
RED = PAL["red"]
BLUE = PAL["blue"]
WHITE = PAL["white"]
DGOLD = (0.62, 0.40, 0.05)       # gold shadow / outline
DRED = (0.55, 0.03, 0.07)


# ================================================================ helpers ===
def _col(c, default):
    if c is None:
        return tuple(default)
    if isinstance(c, str):
        return PAL[c] if c in PAL else hexrgb(c)
    return tuple(c[:3])


def _mix(a, b, p):
    return tuple(x + (y - x) * p for x, y in zip(a, b))


def _clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def _seg(t, a, b):
    """0..1 progress of t through [a, b]."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return _clamp((t - a) / (b - a))


def _hash01(*xs):
    s = 0.0
    for i, x in enumerate(xs):
        s += float(x) * (12.9898 + 31.7 * i)
    return (math.sin(s) * 43758.5453) % 1.0


def _ink(rgb, a=1.0):
    return (int(rgb[0] * a * 255 + 0.5), int(rgb[1] * a * 255 + 0.5),
            int(rgb[2] * a * 255 + 0.5), int(a * 255 + 0.5))


def _fnt(name, px, weight=None):
    """font() wrapper; weight for variable fonts (noto needs (wght, wdth))."""
    px = max(4, int(px))
    if weight is None:
        return font(name, px)
    if name == "noto":
        return font(name, px, (weight, 100))
    if name == "tektur":
        return font(name, px, (100, weight))
    return font(name, px, (weight,))


class Cv:
    """Supersampled canvas holding premultiplied RGBA in a CMYK-mode image.
    All coordinates passed in are in *output* pixels; ss scaling is internal."""

    def __init__(self, w, h, ss=2):
        self.w, self.h, self.ss = int(math.ceil(w)), int(math.ceil(h)), ss
        self.im = Image.new("CMYK", (self.w * ss, self.h * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def P(self, pts):
        s = self.ss
        return [(float(x) * s, float(y) * s) for x, y in pts]

    def poly(self, pts, rgb, a=1.0, outline=None, width=0.0):
        ink = _ink(rgb, a)
        if outline is not None and width > 0:
            self.d.polygon(self.P(pts), fill=ink)
            self.line(list(pts) + [pts[0]], outline, width)
        else:
            self.d.polygon(self.P(pts), fill=ink)

    def line(self, pts, rgb, width, a=1.0, caps=True):
        s = self.ss
        w = max(1, int(round(width * s)))
        P = self.P(pts)
        ink = _ink(rgb, a)
        self.d.line(P, fill=ink, width=w, joint="curve")
        if caps and w > 2:
            r = w / 2.0
            for (x, y) in (P[0], P[-1]):
                self.d.ellipse((x - r, y - r, x + r, y + r), fill=ink)

    def ellipse(self, cx, cy, rx, ry, rgb, a=1.0, outline=None, width=0.0):
        s = self.ss
        box = ((cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s)
        if outline is not None and width > 0:
            self.d.ellipse(box, fill=_ink(rgb, a) if rgb is not None else None,
                           outline=_ink(outline), width=max(1, int(round(width * s))))
        else:
            self.d.ellipse(box, fill=_ink(rgb, a))

    def ring(self, cx, cy, r, rgb, width, a=1.0):
        s = self.ss
        box = ((cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s)
        self.d.ellipse(box, outline=_ink(rgb, a), width=max(1, int(round(width * s))))

    def text(self, xy, s_, name, px, rgb, anchor="mm", weight=None, a=1.0, spacing=0.0):
        f = _fnt(name, px * self.ss, weight)
        x, y = xy
        if spacing:
            # manual letter spacing (centred on anchor x)
            ws = [f.getlength(ch) for ch in s_]
            tot = sum(ws) + spacing * self.ss * (len(s_) - 1)
            cx = x * self.ss - (tot / 2 if anchor[0] == "m" else (tot if anchor[0] == "r" else 0))
            for ch, wch in zip(s_, ws):
                self.d.text((cx, y * self.ss), ch, font=f, fill=_ink(rgb, a), anchor="l" + anchor[1])
                cx += wch + spacing * self.ss
            return
        self.d.text((x * self.ss, y * self.ss), s_, font=f, fill=_ink(rgb, a), anchor=anchor)

    def blend_poly(self, pts, rgb, a):
        """Alpha-blended polygon (over), for translucent shapes."""
        P = self.P(pts)
        xs = [p[0] for p in P]
        ys = [p[1] for p in P]
        x0, y0 = max(int(min(xs)) - 1, 0), max(int(min(ys)) - 1, 0)
        x1, y1 = min(int(max(xs)) + 2, self.im.width), min(int(max(ys)) + 2, self.im.height)
        if x1 <= x0 or y1 <= y0:
            return
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in P], fill=int(255 * a))
        self.im.paste(_ink(rgb), (x0, y0, x1, y1), m)

    def result(self):
        """-> premultiplied CMYK image at output resolution."""
        return self.im.reduce(self.ss) if self.ss > 1 else self.im


def _straight(img_cmyk):
    """premultiplied CMYK-mode image -> straight RGBA PIL image."""
    return Image.frombuffer("RGBa", img_cmyk.size, img_cmyk.tobytes(), "raw", "RGBa", 0, 1).convert("RGBA")


def _premult(img_rgba):
    """straight RGBA PIL image -> premultiplied CMYK-mode image."""
    im = img_rgba.convert("RGBa")
    return Image.frombuffer("CMYK", im.size, im.tobytes(), "raw", "CMYK", 0, 1)


def _alpha(img_cmyk):
    return img_cmyk.getchannel(3)


def _glow(img, amt, r1, r2, tint=None, k=4):
    """Screen a two-radius bloom of premultiplied CMYK `img` over itself."""
    if amt <= 0:
        return img
    w, h = img.size
    small = img.reduce(k) if k > 1 else img
    b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(max(0.5, r1 / k))), np.float32)
    b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(max(0.5, r2 / k))), np.float32)
    g = (b1[..., :3] * 1.5 + b2[..., :3] * 1.5) * amt
    if tint is not None:
        lum = g.max(axis=2, keepdims=True)
        g = lum * np.asarray(tint, np.float32)[None, None, :]
    g = np.minimum(g, 255.0)
    g4 = np.empty(g.shape[:2] + (4,), np.uint8)
    g4[..., :3] = g
    g4[..., 3] = g.max(axis=2)
    gi = Image.frombuffer("CMYK", (g4.shape[1], g4.shape[0]), g4.tobytes(), "raw", "CMYK", 0, 1)
    return ImageChops.screen(img, gi.resize((w, h), Image.BILINEAR))


def _to_frame(img_cmyk, W, H, x=0, y=0):
    """Place a premultiplied CMYK sprite with top-left (x, y) into a float32
    straight RGBA frame (H, W, 4)."""
    out = np.zeros((H, W, 4), np.float32)
    w, h = img_cmyk.size
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return out
    if (x0, y0, x1, y1) != (x, y, x + w, y + h):
        img_cmyk = img_cmyk.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    a = np.asarray(_straight(img_cmyk), np.float32)
    np.multiply(a, 1.0 / 255.0, out=out[y0:y1, x0:x1])
    return out


def _star_poly(cx, cy, R, rot=0.0, inset=0.0, ratio=0.382):
    """Regular 5-point star (point up), optionally inset by a constant distance."""
    r = R * ratio
    if inset:
        R = R - inset / math.sin(math.radians(18))
        r = r - inset / math.sin(math.radians(126))
    return star_points(cx, cy, R, r, rot)


@functools.lru_cache(maxsize=8)
def _noise(w, h, seed=0, blur=0.0):
    rng = np.random.default_rng(seed)
    n = rng.random((h, w)).astype(np.float32)
    if blur > 0:
        from scipy.ndimage import gaussian_filter
        n = gaussian_filter(n, blur, mode="wrap")
        n = (n - n.min()) / max(n.max() - n.min(), 1e-6)
    return n


def _L(arr):
    """float array 0..1 -> PIL L image."""
    a = np.clip(np.asarray(arr) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.frombuffer("L", (a.shape[1], a.shape[0]), a.tobytes(), "raw", "L", 0, 1)


# ================================================================== flag ===
@functools.lru_cache(maxsize=4)
def _flag_tex(tw, th):
    """Flat SFRY flag texture (premultiplied CMYK, opaque), with a faint weave."""
    cv = Cv(tw, th, 3)
    b = th / 3.0
    cv.poly([(0, 0), (tw, 0), (tw, b), (0, b)], BLUE)
    cv.poly([(0, b), (tw, b), (tw, 2 * b), (0, 2 * b)], WHITE)
    cv.poly([(0, 2 * b), (tw, 2 * b), (tw, th), (0, th)], RED)
    R = 0.36 * th
    cx, cy = tw / 2.0, th / 2.0 + 0.02 * th
    cv.poly(_star_poly(cx, cy, R), GOLD)
    cv.poly(_star_poly(cx, cy, R, inset=0.03 * th), RED)
    img = cv.result()
    # cloth weave: multiply by a subtle thread pattern
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    per = max(2.0, th / 260.0)
    weave = 0.955 + 0.025 * np.sin(xx * (2 * np.pi / per)) * np.sin(yy * (2 * np.pi / per) + 1.0)
    weave += (_noise(tw, th, 11)[:th, :tw] - 0.5) * 0.03
    wl = _L(np.clip(weave, 0, 1))
    return ImageChops.multiply(img, Image.merge("CMYK", (wl, wl, wl, Image.new("L", (tw, th), 255))))


def _flag_disp(u, v, t, wave, speed, Wf, Hf):
    ph = 2 * np.pi * (1.25 * u - 0.80 * speed * t) + 0.9 * v
    ph2 = 2 * np.pi * (2.2 * u - 1.35 * speed * t) + 1.7 * v + 1.3
    amp = wave * np.power(np.clip(u, 0, None), 0.85)
    z = amp * (np.sin(ph) + 0.35 * np.sin(ph2))
    dzdu = amp * (np.cos(ph) * 2 * np.pi * 1.25 + 0.35 * np.cos(ph2) * 2 * np.pi * 2.2)
    dy = Hf * (0.055 * z + 0.035 * wave * u * u)
    dx = -Wf * 0.022 * wave * u * (1 - np.cos(ph)) * 0.5
    return dx, dy, dzdu


def flag(t, dur, W, H, wave=1.0, scale=0.8, cx=0.5, cy=0.5, speed=1.0, shade=1.0, **_):
    """Waving SFRY flag, 1:2, blue/white/red top->bottom, gold-bordered red star
    centred over all three stripes (star ~2/3 flag height).

    wave   displacement amplitude (0 = flat)       speed  wave speed multiplier
    scale  flag width as fraction of min(W, 2H)    cx, cy centre of the flag
    shade  strength of the fold light/dark shading
    """
    Wf = int(round(scale * min(W, 2 * H)))
    Hf = Wf // 2
    tex = _flag_tex(Wf, Hf)
    fx, fy = cx * W - Wf / 2.0, cy * H - Hf / 2.0
    padx, pady = int(0.04 * Wf), int(0.14 * Hf)
    ow, oh = Wf + 2 * padx, Hf + 2 * pady
    cell = max(8, int(round(36 * H / 1080)))
    nx, ny = int(math.ceil(ow / cell)), int(math.ceil(oh / cell))
    gx = np.minimum(np.arange(nx + 1) * cell, ow).astype(np.float64)
    gy = np.minimum(np.arange(ny + 1) * cell, oh).astype(np.float64)
    X, Y = np.meshgrid(gx - padx, gy - pady)
    u, v = X / Wf, Y / Hf
    for _i in range(5):
        dx, dy, _d = _flag_disp(u, v, t, wave, speed, Wf, Hf)
        u = (X - dx) / Wf
        v = (Y - dy) / Hf
    _dx, _dy, dzdu = _flag_disp(u, v, t, wave, speed, Wf, Hf)
    su, sv = u * Wf, v * Hf
    data = []
    for j in range(ny):
        for i in range(nx):
            box = (int(gx[i]), int(gy[j]), int(gx[i + 1]), int(gy[j + 1]))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            quad = (su[j, i], sv[j, i], su[j + 1, i], sv[j + 1, i],
                    su[j + 1, i + 1], sv[j + 1, i + 1], su[j, i + 1], sv[j, i + 1])
            data.append((box, quad))
    img = tex.transform((ow, oh), Image.MESH, data, Image.BILINEAR)
    # fold shading
    s = 0.075 * dzdu * shade
    dark = np.clip(1.0 + np.minimum(s, 0) * 1.1, 0.45, 1.0)
    light = np.clip(np.maximum(s, 0) * 0.30, 0, 0.35)
    dl = _L(dark).resize((ow, oh), Image.BILINEAR)
    img = ImageChops.multiply(img, Image.merge("CMYK", (dl, dl, dl, Image.new("L", (ow, oh), 255))))
    hl = ImageChops.multiply(_L(light).resize((ow, oh), Image.BILINEAR), _alpha(img))
    img = ImageChops.add(img, Image.merge("CMYK", (hl, hl, hl, Image.new("L", (ow, oh), 0))))
    return _to_frame(img, W, H, fx - padx, fy - pady)


# ================================================================== star ===
def star2d(t, dur, W, H, size=0.6, shine=True, glow=0.5, pulse=0.0, pulse_hz=130 / 60.0,
           rot=0.0, cx=0.5, cy=0.5, border=0.075, bevel=False, shine_period=1.8,
           shine_offset=0.0, color=None, **_):
    """Flat red five-pointed star with gold outline.

    size     star outer diameter as fraction of H     rot     rotation (rad)
    shine    sweeping diagonal specular band (every shine_period s)
    glow     soft red/gold bloom amount                pulse   0..1 beat pulse (scale+glow)
    border   gold border width as fraction of R        bevel   10-facet light/dark bevel
    """
    ph = (t * pulse_hz) % 1.0
    beat = math.exp(-ph * 6.0) if pulse > 0 else 0.0
    R = size * H / 2.0 * (1 + 0.06 * pulse * beat)
    pad = int(R * 0.45 + 8) if glow > 0 else 4
    sside = int(2.1 * R) + 4                  # tight supersampled star canvas
    cv = Cv(sside, sside, 2)
    s0 = sside / 2.0
    scy = s0 + R * 0.095                      # optical centring (star bbox centre)
    red = _col(color, RED)
    cv.poly(_star_poly(s0, scy, R, rot), GOLD)
    inner = _star_poly(s0, scy, R, rot, inset=border * R)
    cv.poly(inner, red)
    if bevel:
        for i in range(10):
            p0, p1 = inner[i], inner[(i + 1) % 10]
            lit = 0.18 if i % 2 == 0 else -0.22
            c = tuple(min(1.0, x * (1 + lit)) for x in red)
            cv.poly([(s0, scy), p0, p1], c)
    star = cv.result()
    if shine:
        per = max(0.3, shine_period)
        p = ((t + shine_offset) % per) / per
        pos = -0.15 + 1.3 * ease_in_out_cubic(min(1.0, p / 0.55))
        if -0.14 < pos < 1.14:
            k = 4                                 # mask drawn at 1/4 res, blurred, upsampled
            ms = max(2, sside // k)
            m = Image.new("L", (ms, ms), 0)
            md = ImageDraw.Draw(m)
            bw = 0.16 * R / k
            ang = math.radians(-35)
            dx, dy = math.cos(ang), math.sin(ang)
            cxb = (s0 + (pos - 0.5) * 2 * R * dx) / k
            cyb = (scy + (pos - 0.5) * 2 * R * dy) / k
            L = 3 * R / k
            nx, ny = -dy, dx
            for wmul, val in ((1.0, 110), (0.4, 235)):
                hw = bw * wmul
                md.polygon([(cxb - dx * hw - nx * L, cyb - dy * hw - ny * L),
                            (cxb + dx * hw - nx * L, cyb + dy * hw - ny * L),
                            (cxb + dx * hw + nx * L, cyb + dy * hw + ny * L),
                            (cxb - dx * hw + nx * L, cyb - dy * hw + ny * L)], fill=val)
            m = m.filter(ImageFilter.GaussianBlur(max(0.6, R * 0.02 / k))).resize(star.size, Image.BILINEAR)
            m = ImageChops.multiply(m, _alpha(star))
            star.paste(_ink((1.0, 0.97, 0.85)), (0, 0), m)
    side = sside + 2 * pad
    img = Image.new("CMYK", (side, side), (0, 0, 0, 0))
    img.paste(star, (pad, pad))
    c0, ccy = pad + s0, pad + scy
    # crop to the visible frame before the (full-res) glow pass
    fx0, fy0 = int(round(cx * W - c0)), int(round(cy * H - ccy))
    x0, y0 = max(0, -fx0), max(0, -fy0)
    x1, y1 = min(side, W - fx0), min(side, H - fy0)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((H, W, 4), np.float32)
    if (x0, y0, x1, y1) != (0, 0, side, side):
        img = img.crop((x0, y0, x1, y1))
        c0, ccy = c0 - x0, ccy - y0
    g = glow * (1 + 1.2 * pulse * beat)
    if g > 0:
        img = _glow(img, g * 0.5, R * 0.05, R * 0.30)
    return _to_frame(img, W, H, cx * W - c0, cy * H - ccy)


# ========================================================== coat of arms ===
def _almond(cx, cy, ang, L, w, n=7):
    """Pointed-oval (wheat kernel / leaf) polygon, long axis along ang."""
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for i in range(n + 1):
        s = -1 + 2 * i / n
        x, y = s * L / 2, (1 - s * s) ** 0.8 * w / 2
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    for i in range(n - 1, 0, -1):
        s = -1 + 2 * i / n
        x, y = s * L / 2, -(1 - s * s) ** 0.8 * w / 2
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def _ellipse_pt(phi, a, b, c=(0.0, 0.0)):
    return (c[0] + a * math.sin(phi), c[1] - b * math.cos(phi))


def _ribbon(cv, pl, S, pr, reveal, gold, blue, mono, dgold, ow):
    def X(x, y):
        return pl([(x, y)])[0]
    yb, hb = 0.43, 0.085
    half = 0.34 * pr
    n = 24
    top, bot = [], []
    for i in range(n + 1):
        x = -half + 2 * half * i / n
        yc = yb - 0.10 * (x / 0.34) ** 2 + 0.02
        top.append((x, yc - hb / 2))
        bot.append((x, yc + hb / 2))
    if pr > 0.6:
        tp = _seg(pr, 0.6, 1.0)
        for sd in (-1, 1):
            ex = sd * half
            ey = yb - 0.10 + 0.02
            tail = [(ex, ey - hb / 2), (ex + sd * 0.09 * tp, ey - hb / 2 + 0.05 * tp),
                    (ex + sd * 0.065 * tp, ey + 0.02 * tp), (ex + sd * 0.10 * tp, ey + hb / 2 + 0.06 * tp),
                    (ex, ey + hb / 2)]
            cv.poly(pl(tail), tuple(c * 0.62 for c in blue) if not mono else dgold)
    cv.poly(pl(top + bot[::-1]), blue)
    cv.line(pl(top), gold, ow * 1.2)
    cv.line(pl(bot), gold, ow * 1.2)
    ta = _seg(reveal, 0.10, 0.30)
    if ta > 0:
        txt = "29. XI 1943."
        ncv = int(round(len(txt) * ta))
        f = _fnt("oswald", 0.052 * S * cv.ss, 700)
        ws = [f.getlength(ch) / cv.ss / S for ch in txt]
        tot = sum(ws) + 0.004 * (len(txt) - 1)
        x = -tot / 2
        for ch, wch in zip(txt[:ncv], ws):
            xc = x + wch / 2
            yc = yb - 0.10 * (xc / 0.34) ** 2 + 0.02
            cv.text(X(xc, yc + 0.002), ch, "oswald", 0.052 * S, WHITE if not mono else dgold,
                    anchor="mm", weight=700)
            x += wch + 0.004


def _draw_coat(cv, S, ox, oy, t, reveal, mono=None, flame_amp=1.0):
    """Draw the emblem into canvas cv; unit coords scaled by S around (ox, oy)."""
    def X(x, y):
        return (ox + x * S, oy + y * S)

    def pl(pts):
        return [X(x, y) for x, y in pts]

    gold = mono or GOLD
    dgold = mono and tuple(c * 0.55 for c in mono) or DGOLD
    red = mono or RED
    blue = mono or BLUE
    orange = mono or (1.0, 0.55, 0.08)
    yellow = mono or (1.0, 0.86, 0.35)
    ow = 0.006 * S                       # outline width

    # ---------------------------------------------------------- ribbon (under wheat stems)
    pr = ease_out_cubic(_seg(reveal, 0.0, 0.16))
    # ---------------------------------------------------------- wheat wreath
    pw = _seg(reveal, 0.04, 0.48)
    a_, b_, cen = 0.405, 0.455, (0.0, 0.03)
    for side in (-1, 1):
        phis = np.linspace(math.pi - 0.42, 0.52, 200)
        n_ears = 5
        span = (math.pi - 0.42 - 0.52)
        # stalk
        grow = pw
        if grow <= 0:
            continue
        m = max(2, int(len(phis) * grow))
        stalk = [_ellipse_pt(side * p, a_, b_, cen) for p in phis[:m]]
        cv.line(pl(stalk), dgold, ow * 1.6)
        for e in range(n_ears):
            e0 = e / n_ears
            e1 = (e + 1.0) / n_ears
            if grow < e0 + 0.02:
                break
            eg = _clamp((grow - e0) / (e1 - e0))
            npair = 7
            for k in range(npair):
                f = (k + 0.5) / npair
                if f > eg:
                    break
                q = e0 + (e1 - e0) * (0.08 + 0.92 * f)
                phi = side * (math.pi - 0.42 - span * q)
                x, y = _ellipse_pt(phi, a_, b_, cen)
                # tangent towards the top of the wreath
                x2, y2 = _ellipse_pt(phi - side * 0.01, a_, b_, cen)
                ta = math.atan2(y2 - y, x2 - x)
                sz = 1.0 - 0.45 * f
                Lk, wk = 0.060 * sz, 0.026 * sz
                for sgn in (-1, 1):
                    ang = ta + sgn * 0.55
                    off = 0.019 * sz
                    kx = x + math.cos(ta + sgn * math.pi / 2) * off + math.cos(ta) * 0.012
                    ky = y + math.sin(ta + sgn * math.pi / 2) * off + math.sin(ta) * 0.012
                    pts = pl(_almond(kx, ky, ang, Lk, wk))
                    cv.poly(pts, dgold)
                    pts2 = pl(_almond(kx - math.cos(ang) * 0.003, ky - math.sin(ang) * 0.003, ang,
                                      Lk * 0.8, wk * 0.62))
                    cv.poly(pts2, gold)
            if eg >= 0.98:
                # awns at the ear tip
                q = e0 + (e1 - e0) * 1.0
                phi = side * (math.pi - 0.42 - span * q)
                x, y = _ellipse_pt(phi, a_, b_, cen)
                x2, y2 = _ellipse_pt(phi - side * 0.01, a_, b_, cen)
                ta = math.atan2(y2 - y, x2 - x)
                for da in (-0.35, 0.0, 0.35):
                    ln = 0.05
                    cv.line(pl([(x, y), (x + math.cos(ta + da) * ln, y + math.sin(ta + da) * ln)]),
                            dgold, ow * 0.9)
    # ---------------------------------------------------------- torches
    gam = math.radians(20)
    dxu, dyu = math.sin(gam), -math.cos(gam)        # torch axis (up-right)
    nxu, nyu = -dyu, dxu                             # perpendicular
    Lt = 0.44
    cups = []
    lit = []
    for i in range(6):
        pi = _seg(reveal, 0.30 + i * 0.05, 0.40 + i * 0.05)
        bx, by = -0.29 + i * 0.074, 0.40 - i * 0.004
        if pi <= 0:
            cups.append(None)
            lit.append(0.0)
            continue
        sl = ease_out_back(pi)
        off = (1 - sl) * 0.14
        bx2, by2 = bx - dxu * off, by - dyu * off
        tx2, ty2 = bx2 + dxu * Lt, by2 + dyu * Lt

        def TP(s, w, bx2=bx2, by2=by2, tx2=tx2, ty2=ty2):
            return (bx2 + (tx2 - bx2) * s + nxu * w, by2 + (ty2 - by2) * s + nyu * w)
        # tapered handle with outline + highlight
        cv.poly(pl([TP(0.0, -0.013), TP(0.0, 0.013), TP(0.84, 0.021), TP(0.84, -0.021)]), dgold)
        cv.poly(pl([TP(0.01, -0.008), TP(0.01, 0.009), TP(0.83, 0.015), TP(0.83, -0.015)]), gold)
        cv.poly(pl([TP(0.03, -0.006), TP(0.03, -0.001), TP(0.82, -0.004), TP(0.82, -0.012)]),
                _mix(gold, (1, 1, 1), 0.45) if not mono else gold)
        # grip rings
        for s_ in (0.22, 0.50):
            cv.poly(pl([TP(s_, -0.017), TP(s_, 0.017), TP(s_ + 0.035, 0.018), TP(s_ + 0.035, -0.018)]), dgold)
        # cup
        cv.poly(pl([TP(0.83, -0.025), TP(0.83, 0.025), TP(1.0, 0.036), TP(1.0, -0.036)]), dgold)
        cv.poly(pl([TP(0.85, -0.019), TP(0.85, 0.020), TP(0.985, 0.029), TP(0.985, -0.027)]), gold)
        cups.append(TP(1.0, 0.0))
        lit.append(ease_out_cubic(_seg(reveal, 0.36 + i * 0.05, 0.44 + i * 0.05)))
    # ---------------------------------------------------------- flames
    if any(v > 0 for v in lit):
        big = ease_out_cubic(_seg(reveal, 0.64, 0.80)) * flame_amp
        on = [i for i in range(6) if lit[i] > 0]
        mx = sum(cups[i][0] for i in on) / len(on)
        my = min(cups[i][1] for i in on)
        apex = (mx + 0.02, my - 0.30 * (0.4 + 0.6 * big) * (0.5 + 0.5 * len(on) / 6))

        def tongue(bx, by, w, h, dx, n=14):
            L, R_ = [], []
            for k in range(n + 1):
                u = k / n
                f = (1 - u) ** 0.65 * (1 + 0.9 * u * (1 - u))
                xx = bx + dx * u * u
                yy = by - h * u
                L.append((xx - w / 2 * f, yy))
                R_.append((xx + w / 2 * f, yy))
            return L + R_[::-1]

        layers = ((red, 1.0), (orange, 0.62), (yellow, 0.28)) if not mono else ((gold, 1.0),)
        for li, (colr, sc) in enumerate(layers):
            # central merged flame
            if big > 0:
                fl = 1 + 0.07 * math.sin(t * 11.0 + li) + 0.04 * math.sin(t * 23.0)
                hC = (my - apex[1]) * fl * (0.55 + 0.45 * sc) + 0.03
                wC = (0.20 + 0.10 * big) * sc
                cv.poly(pl(tongue(mx, my + 0.02, wC, hC * (0.35 + 0.65 * big) + 0.02 * sc,
                                  0.05 * math.sin(t * 3.1))), colr)
            for i in on:
                bx, by = cups[i]
                ph = i * 1.7
                fl = 1 + 0.12 * math.sin(t * 13.0 + ph) + 0.06 * math.sin(t * 29.0 + 2 * ph)
                h = (0.10 + 0.05 * (1 - abs(i - 2.5) / 2.5)) * lit[i] * fl
                dx = (apex[0] - bx) * 0.5 + 0.015 * math.sin(t * 7.0 + ph)
                h2 = h + (by - apex[1]) * 0.45 * big
                cv.poly(pl(tongue(bx, by + 0.012, 0.062 * sc, h2 * (0.45 + 0.55 * sc), dx * (0.5 + 0.5 * sc))),
                        colr)
    # ---------------------------------------------------------- ribbon (on top of torch ends)
    if pr > 0:
        _ribbon(cv, pl, S, pr, reveal, gold, blue, mono, dgold, ow)
    # ---------------------------------------------------------- star
    ps = _seg(reveal, 0.80, 0.95)
    if ps > 0:
        s = ease_out_back(ps, 2.2)
        R = 0.095 * s
        sx, sy = 0.0, -0.44
        cv.poly(pl(_star_poly(sx, sy, R)), gold)
        cv.poly(pl(_star_poly(sx, sy, R, inset=0.012 * s)), red if not mono else dgold)


def coat_of_arms(t, dur, W, H, reveal=1.0, scale=0.8, cx=0.5, cy=0.5, glow=0.35, mono=None, **_):
    """Stylised SFRY coat of arms (flat vector, red/gold/blue).

    reveal  0..1 build-up: ribbon unfurls, wheat grows, torches slide in and ignite
            one by one, flames merge, star pops. Pass "auto" to use t/(0.85*dur).
    scale   emblem height as fraction of H        cx, cy  centre
    glow    warm bloom amount                     mono    single rgb colour (for embossing)
    """
    if reveal == "auto":
        reveal = _clamp(t / max(0.85 * dur, 1e-3))
    S = scale * H
    side = int(S * 1.2)
    cv = Cv(side, side, 2)
    _draw_coat(cv, S, side / 2.0, side / 2.0 + 0.01 * S, t, reveal,
               mono=_col(mono, GOLD) if mono is not None else None)
    img = cv.result()
    if glow > 0 and mono is None:
        img = _glow(img, glow, S * 0.01, S * 0.06)
    return _to_frame(img, W, H, cx * W - side / 2.0, cy * H - side / 2.0)


# ============================================================== passport ===
DEFAULT_STAMPS = ("NEW YORK", "MOSKVA", "PARIS", "KAIRO", "NEW DELHI", "LONDON", "TOKYO", "HAVANA", "PEKING")
_INKS = ((0.36, 0.14, 0.58), (0.10, 0.22, 0.62), (0.72, 0.08, 0.12), (0.06, 0.42, 0.26),
         (0.14, 0.14, 0.22), (0.55, 0.12, 0.45), (0.05, 0.35, 0.55))
_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")


def _homography(dst, src):
    """PIL PERSPECTIVE coefficients mapping output quad `dst` -> source quad `src`."""
    A, b = [], []
    for (X, Y), (x, y) in zip(dst, src):
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
        b += [x, y]
    return tuple(np.linalg.solve(np.array(A, np.float64), np.array(b, np.float64)))


def _rounded_mask(w, h, r, ss=3):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=255)
    return m.reduce(ss)


@functools.lru_cache(maxsize=4)
def _cover_tex(wp, hp):
    """Passport front cover (premultiplied CMYK, with 2 px transparent margin)."""
    pad = 2
    w, h = wp + 2 * pad, hp + 2 * pad
    yy, xx = np.mgrid[0:hp, 0:wp].astype(np.float32)
    base = np.array([0.50, 0.045, 0.075], np.float32)
    n1 = _noise(wp, hp, 21)[:hp, :wp]
    per = max(2.0, hp / 330.0)
    weave = 0.5 + 0.5 * np.sin(xx * 2 * np.pi / per) * np.sin(yy * 2 * np.pi / per)
    grain = 0.92 + 0.07 * weave + 0.07 * (n1 - 0.5)
    vig = 1.0 - 0.28 * (np.abs(xx / wp - 0.5) * 2) ** 3 - 0.28 * (np.abs(yy / hp - 0.5) * 2) ** 3
    grad = 1.08 - 0.16 * (xx / wp * 0.5 + yy / hp * 0.5)
    rgb = base[None, None, :] * (grain * vig * grad)[..., None]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[pad:pad + hp, pad:pad + wp, :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA")
    im.putalpha(Image.new("L", (w, h), 0))
    a = Image.new("L", (w, h), 0)
    a.paste(_rounded_mask(wp, hp, int(0.035 * wp)), (pad, pad))
    im.putalpha(a)
    img = _premult(im)
    # gold "foil" elements drawn on a supersampled canvas, then embossed
    cv = Cv(w, h, 2)
    gold = (0.93, 0.74, 0.30)
    cxp = w / 2.0
    y = pad + 0.085 * hp
    cv.text((cxp, y), "SOCIJALISTIČKA FEDERATIVNA", "noto", 0.052 * wp, gold, weight=600, spacing=0.004 * wp)
    cv.text((cxp, y + 0.045 * hp), "REPUBLIKA JUGOSLAVIJA", "noto", 0.052 * wp, gold, weight=600,
            spacing=0.004 * wp)
    cv.text((cxp, y + 0.090 * hp), "СОЦИЈАЛИСТИЧКА ФЕДЕРАТИВНА РЕПУБЛИКА ЈУГОСЛАВИЈА", "noto",
            0.030 * wp, gold, weight=500)
    S = 0.40 * hp
    _draw_coat(cv, S, cxp, pad + 0.47 * hp, 0.0, 1.0, mono=gold, flame_amp=0.9)
    cv.text((cxp, pad + 0.80 * hp), "PASOŠ", "noto", 0.105 * wp, gold, weight=700, spacing=0.02 * wp)
    cv.text((cxp, pad + 0.875 * hp), "PASOŠ · PASSPORT · ПАСОШ", "noto", 0.046 * wp, gold, weight=600,
            spacing=0.004 * wp)
    foil = cv.result()
    fa = _alpha(foil)
    # emboss: dark offset below-right, light offset above-left, then foil with a metallic gradient
    off = max(1, int(round(hp / 420)))
    dark = ImageChops.offset(fa, off, off)
    img.paste(_ink((0.16, 0.0, 0.02)), (0, 0), ImageChops.multiply(dark, Image.new("L", (w, h), 170)))
    lite = ImageChops.offset(fa, -off, -off)
    img.paste(_ink((1.0, 0.92, 0.7)), (0, 0), ImageChops.multiply(lite, Image.new("L", (w, h), 120)))
    grad = np.linspace(1.12, 0.78, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    grad = grad * (0.93 + 0.14 * _noise(w, h, 5, 1.5)[:h, :w])
    gcol = np.stack([gold[0] * grad, gold[1] * grad, gold[2] * grad], -1)
    gim = Image.fromarray(np.clip(gcol * 255, 0, 255).astype(np.uint8), "RGB")
    gim4 = Image.merge("CMYK", (*gim.split(), Image.new("L", (w, h), 255)))
    img.paste(gim4, (0, 0), fa)
    # keep foil inside the rounded cover
    img = ImageChops.multiply(img, Image.merge("CMYK", (a, a, a, a)))
    return img


def _guilloche(d, cx, cy, R, r, dd, rot, col, width, n=720):
    pts = []
    k = (R - r) / r
    for i in range(n + 1):
        u = i / n * 2 * np.pi * 7
        x = (R - r) * math.cos(u) + dd * math.cos(k * u)
        y = (R - r) * math.sin(u) - dd * math.sin(k * u)
        ca, sa = math.cos(rot), math.sin(rot)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    d.line(pts, fill=col, width=width)


@functools.lru_cache(maxsize=4)
def _page_tex(wp, hp, number):
    """Pale guilloche visa page (premultiplied CMYK, opaque, same size as cover)."""
    ss = 2
    base = (0.945, 0.935, 0.875)
    im = Image.new("RGB", (wp * ss, hp * ss), tuple(int(c * 255) for c in base))
    d = ImageDraw.Draw(im)
    # fine wavy background lines
    for j in range(0, hp * ss, max(3, int(hp * ss / 140))):
        pts = [(x, j + 3 * ss * math.sin(x / (wp * ss) * 2 * math.pi * 3 + j * 0.05))
               for x in range(0, wp * ss + 8, 8)]
        d.line(pts, fill=(226, 222, 204), width=1)
    cx, cy = wp * ss / 2, hp * ss * 0.52
    Rr = wp * ss * 0.40
    for k in range(10):
        _guilloche(d, cx, cy, Rr * 0.8, Rr * 0.8 / 7 * 2, Rr * 0.45, k * 0.0628,
                   (232, 196, 200) if k % 2 == 0 else (190, 222, 216), 1 * ss // 2 + 1, n=900)
    for k in range(6):
        _guilloche(d, cx, cy, Rr * 0.35, Rr * 0.35 / 5 * 2, Rr * 0.22, k * 0.21,
                   (205, 212, 232), 1, n=500)
    # faint star watermark
    d.polygon([(x, y) for x, y in _star_poly(cx, cy, Rr * 0.30)], outline=(214, 200, 190), width=2 * ss)
    # header / footer micro text
    f = _fnt("noto", 0.042 * wp * ss, 600)
    d.text((wp * ss / 2, 0.055 * hp * ss), "VIZE · VISAS · ВИЗЕ", font=f, fill=(150, 120, 120), anchor="mm")
    f2 = _fnt("mono", 0.035 * wp * ss)
    d.text((wp * ss / 2, 0.955 * hp * ss), str(number), font=f2, fill=(140, 120, 120), anchor="mm")
    f3 = _fnt("mono", 0.016 * wp * ss)
    micro = "SFRJ " * 60
    d.text((0.05 * wp * ss, 0.085 * hp * ss), micro, font=f3, fill=(200, 185, 185), anchor="lm")
    d.text((0.05 * wp * ss, 0.92 * hp * ss), micro, font=f3, fill=(200, 185, 185), anchor="lm")
    im = im.reduce(ss)
    # paper grain
    arr = np.asarray(im, np.float32) * (0.97 + 0.05 * _noise(wp, hp, 9)[:hp, :wp, None])
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return Image.merge("CMYK", (*im.split(), Image.new("L", (wp, hp), 255)))


@functools.lru_cache(maxsize=64)
def _stamp_sprite(label, idx, size):
    """Ink stamp (premultiplied CMYK) with gaps / uneven ink. size = px diameter/width."""
    rng = np.random.default_rng(1000 + idx * 7 + sum(map(ord, label)))
    ink = _INKS[idx % len(_INKS)]
    rect = (idx % 3 == 1)
    if rect:
        w, h = int(size * 1.25), int(size * 0.72)
    else:
        w = h = int(size)
    pad = 4
    cv = Cv(w + 2 * pad, h + 2 * pad, 2)
    c = (1, 1, 1)
    cx, cy = pad + w / 2, pad + h / 2
    day = 1 + int(rng.integers(0, 28))
    mon = _ROMAN[int(rng.integers(0, 12))]
    yr = 1956 + int(rng.integers(0, 32))
    date = f"{day:02d}. {mon}. {yr}"
    lw = max(1.5, size * 0.03)
    word = ("ULAZ · ENTRY", "IZLAZ · EXIT", "ENTRÉE", "ARRIVAL", "DEPARTURE")[idx % 5]

    def fit(txt, name, maxw, px, weight=None):
        f = _fnt(name, px, weight)
        wtxt = f.getlength(txt)
        return px * min(1.0, maxw / max(wtxt, 1))
    if rect:
        cv.d.rounded_rectangle((pad * 2, pad * 2, (pad + w) * 2, (pad + h) * 2), radius=int(size * 0.06) * 2,
                               outline=_ink(c), width=int(lw * 2))
        i2 = lw * 2.2
        cv.d.rectangle(((pad + i2) * 2, (pad + i2) * 2, (pad + w - i2) * 2, (pad + h - i2) * 2),
                       outline=_ink(c), width=max(1, int(lw)))
        cv.text((cx, pad + h * 0.26), word, "oswald", fit(word, "oswald", w * 0.8, h * 0.16, 500), c, weight=500)
        cv.text((cx, cy + h * 0.02), label, "oswald", fit(label, "oswald", w * 0.86, h * 0.30, 700), c, weight=700)
        cv.text((cx, pad + h * 0.78), date, "mono", fit(date, "mono", w * 0.8, h * 0.15), c)
    else:
        r = w / 2 - lw
        cv.ring(cx, cy, r, c, lw)
        cv.ring(cx, cy, r * 0.80, c, lw * 0.6)
        cv.text((cx, cy - r * 0.47), word, "oswald", fit(word, "oswald", r * 1.05, r * 0.2, 500), c, weight=500)
        cv.text((cx, cy + r * 0.02), label, "oswald", fit(label, "oswald", r * 1.45, r * 0.42, 700), c, weight=700)
        cv.text((cx, cy + r * 0.44), date, "mono", fit(date, "mono", r * 1.2, r * 0.17), c)
        # small stars on the ring
        for sx in (-1, 1):
            cv.poly(_star_poly(cx + sx * r * 0.90, cy, r * 0.07), c)
    m = cv.result().getchannel(0)            # coverage (white ink) as mask
    ww, hh = m.size
    # uneven ink: blotchy density + small gaps
    blot = _noise(ww, hh, 300 + idx, 2.5)[:hh, :ww]
    fine = _noise(ww, hh, 400 + idx, 0.7)[:hh, :ww]
    dens = np.clip(0.55 + 0.55 * blot, 0, 1) * (fine > 0.22)
    dens = np.clip(dens * (0.75 + 0.35 * rng.random()), 0, 1)
    ma = np.asarray(m, np.float32) / 255.0 * dens
    mask = _L(ma)
    out = Image.new("CMYK", (ww, hh), (0, 0, 0, 0))
    out.paste(_ink(ink), (0, 0), mask)
    return out


def _stamp_layout(k, wp, hp):
    """Deterministic stamp slot k -> (page 0=right 1=left, x, y, rot_deg)."""
    right = ((0.30, 0.24), (0.70, 0.30), (0.32, 0.53), (0.70, 0.62), (0.42, 0.81))
    left = ((0.32, 0.28), (0.70, 0.43), (0.34, 0.68), (0.70, 0.82))
    kk = k % 9
    if kk < 5:
        page, (sx, sy) = 0, right[kk]
    else:
        page, (sx, sy) = 1, left[kk - 5]
    if k >= 9:          # further stamps overlap the earlier ones
        sx += (_hash01(k, 7.1) - 0.5) * 0.2
        sy += (_hash01(k, 8.1) - 0.5) * 0.2
    jx = (_hash01(k, 1.1) - 0.5) * 0.10
    jy = (_hash01(k, 2.2) - 0.5) * 0.08
    rot = (_hash01(k, 3.3) - 0.5) * 30
    return page, (sx + jx) * wp, (sy + jy) * hp, rot


@functools.lru_cache(maxsize=32)
def _stamp_final(label, k, ssize, rot):
    return _stamp_sprite(label, k, ssize).rotate(rot, Image.BILINEAR, expand=True)


@functools.lru_cache(maxsize=24)
def _page_settled(number, wp, hp, ssize, settled):
    """Visa page with all already-landed stamps baked in (cached per stamp set)."""
    img = _page_tex(wp, hp, number)
    if not settled:
        return img
    img = img.copy()
    for k, label in settled:
        _pg, x, y, rot = _stamp_layout(k, wp, hp)
        spr = _stamp_final(label, k, ssize, rot)
        _over_premult(img, spr, int(x - spr.size[0] / 2), int(y - spr.size[1] / 2))
    return img


def _page_with_stamps(t, number, stamps, page_id, wp, hp, ssize):
    settled, active = [], []
    for k, (ts, label) in enumerate(stamps):
        page, x, y, rot = _stamp_layout(k, wp, hp)
        if page != page_id:
            continue
        tau = t - ts
        if tau < -0.10:
            continue
        (settled if tau > 0.45 else active).append((k, label, tau, x, y, rot))
    img = _page_settled(number, wp, hp, ssize, tuple((k, lb) for k, lb, *_r in settled))
    if not active:
        return img
    img = img.copy()
    for k, label, tau, x, y, rot in active:
        p = _clamp((tau + 0.10) / 0.10)
        if tau < 0:
            sc = 3.0 - 2.0 * ease_in_cubic(p)
            op = 0.25 + 0.75 * p
        else:
            sc = 1.0 - 0.10 * math.exp(-tau * 18.0) * math.cos(tau * 45.0)
            op = 1.0
        sp = _stamp_sprite(label, k, ssize)
        w0, h0 = sp.size
        ang = rot + (1 - p) * 14
        nw, nh = max(2, int(w0 * sc)), max(2, int(h0 * sc))
        spr = sp.resize((nw, nh), Image.BILINEAR).rotate(ang, Image.BILINEAR, expand=True)
        if op < 1:
            spr = ImageChops.multiply(spr, Image.new("CMYK", spr.size, (int(255 * op),) * 4))
        _over_premult(img, spr, int(x - spr.size[0] / 2), int(y - spr.size[1] / 2))
    return img


@functools.lru_cache(maxsize=4)
def _inner_board(wp, hp):
    inner = _cover_tex(wp, hp).transpose(Image.FLIP_LEFT_RIGHT)
    return ImageChops.multiply(inner, Image.new("CMYK", inner.size, (150, 150, 150, 255)))


@functools.lru_cache(maxsize=4)
def _back_board(wp, hp):
    return _cover_tex(wp, hp).transpose(Image.FLIP_LEFT_RIGHT)


def _over_premult(dst, src, x, y):
    """In-place premultiplied 'over' of CMYK src onto CMYK dst at (x, y)."""
    sw, sh = src.size
    a = src.getchannel(3)
    # dst = src + dst * (1 - a):  paste(straight, mask=a) does exactly that
    st = _straight(src)
    r, g, b, _a = st.split()
    solid = Image.merge("CMYK", (r, g, b, Image.new("L", (sw, sh), 255)))
    dst.paste(solid, (x, y), a)


def passport(t, dur, W, H, open_at=None, stamps=None, scale=0.78, open_dur=0.55, cx=0.5, cy=0.5,
             shake=1.0, tilt=0.0, **_):
    """The red SFRJ passport.

    Closed: deep-red cloth cover, gold-embossed coat of arms, 'SOCIJALISTIČKA FEDERATIVNA
    REPUBLIKA JUGOSLAVIJA' and 'PASOŠ · PASSPORT · ПАСОШ'.
    open_at   time (s) at which the cover flips open (perspective page turn, open_dur s);
              None = stays closed. The book slides so the open spread ends centred.
    stamps    list of (time, label); each stamp slams onto the visa pages at its time
              (3x -> 1x with overshoot, rotation, ink gaps). None = 9 default cities
              every 0.23 s after opening (NEW YORK, MOSKVA, PARIS, KAIRO, ...).
    scale     passport height as fraction of H.   shake: impact jolt amount.
    """
    hp = int(scale * H)
    wp = int(hp * 0.704)
    if stamps is None:
        if open_at is None:
            stamps = []
        else:
            t0 = open_at + open_dur + 0.12
            stamps = [(t0 + i * 60 / 130 / 2, c) for i, c in enumerate(DEFAULT_STAMPS)]
    stamps = [(float(a), str(b)) for a, b in stamps]
    p = 0.0 if open_at is None else ease_in_out_cubic(_clamp((t - open_at) / max(open_dur, 1e-3)))
    theta = math.pi * p
    # impact jolt from the most recent stamp
    jy = 0.0
    if p > 0.99 and stamps:
        last = max((ts for ts, _l in stamps if ts <= t), default=None)
        if last is not None:
            tau = t - last
            jy = shake * 0.012 * hp * math.exp(-tau * 16) * math.sin(tau * 55)
    mgn = int(0.012 * hp)                     # cover visible around pages
    cw, ch = 2 * wp + 4 * mgn + 8, int(hp * 1.32)
    ox = cw // 2                              # canvas x of the spine when fully open
    oy = (ch - hp) // 2
    xs = ox - wp / 2 + (wp / 2) * p           # spine x in canvas
    canvas = Image.new("CMYK", (cw, ch), (0, 0, 0, 0))
    cover = _cover_tex(wp, hp)
    ssize = int(0.33 * wp)
    # drop shadow
    sh = Image.new("L", (cw // 4, ch // 4), 0)
    sd = ImageDraw.Draw(sh)
    lx0 = min(xs, xs + wp * math.cos(theta)) if p > 0 else xs
    sd.rectangle(((lx0 + 0.02 * wp) / 4, (oy + 0.03 * hp) / 4, (xs + wp + 0.02 * wp) / 4, (oy + hp * 1.03) / 4),
                 fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(hp * 0.018)).resize((cw, ch), Image.BILINEAR)
    canvas.paste(_ink((0, 0, 0)), (0, 0), sh)
    # back cover (right) + right page with stamps
    pw_, ph_ = wp - 2 * mgn, hp - 2 * mgn
    ijy = int(round(jy))
    if p > 0.0:
        _over_premult(canvas, _back_board(wp, hp), int(xs) - 2, oy - 2 + ijy)
        pg = _page_with_stamps(t, 7, stamps, 0, pw_, ph_, ssize)
        _over_premult(canvas, pg, int(xs) + mgn // 2, oy + mgn + ijy)
        d = ImageDraw.Draw(canvas)
        for k in range(3):
            xe_ = int(xs + wp - mgn / 2 + k)
            d.line((xe_, oy + mgn + k + ijy, xe_, oy + hp - mgn - k + ijy), fill=_ink((0.8, 0.78, 0.7)), width=1)
    # moving cover
    cth = math.cos(theta)
    persp = 1 + 0.22 * math.sin(theta)
    xe = xs + wp * cth
    ytop_e = oy + hp / 2 - hp / 2 * persp
    ybot_e = oy + hp / 2 + hp / 2 * persp
    if cth >= 0:
        src_img = cover
        shade = 0.55 + 0.45 * cth
        dstq = [(xs, oy), (xe, ytop_e), (xe, ybot_e), (xs, oy + hp)]
    else:
        # inside of the cover: darker red board with the left visa page (page 6)
        pg6 = _page_with_stamps(t, 6, stamps, 1, pw_, ph_, ssize)
        src_img = _inner_board(wp, hp).copy()
        _over_premult(src_img, pg6, 2 + mgn + mgn // 2, 2 + mgn)
        shade = 0.55 + 0.45 * (-cth)
        dstq = [(xe, ytop_e), (xs, oy), (xs, oy + hp), (xe, ybot_e)]
    sw_, sh_ = src_img.size
    srcq = [(2, 2), (sw_ - 2, 2), (sw_ - 2, sh_ - 2), (2, sh_ - 2)]
    if p <= 0.0:
        _over_premult(canvas, src_img, int(round(xs)) - 2, oy - 2)
    elif p >= 0.999:
        _over_premult(canvas, src_img, int(round(xs - wp)) - 2, oy - 2 + ijy)
    elif abs(xe - xs) > 1.0:
        qx0 = int(math.floor(min(q[0] for q in dstq))) - 2
        qy0 = int(math.floor(min(q[1] for q in dstq))) - 2
        qx1 = int(math.ceil(max(q[0] for q in dstq))) + 2
        qy1 = int(math.ceil(max(q[1] for q in dstq))) + 2
        coeffs = _homography([(x_ - qx0, y_ - qy0) for x_, y_ in dstq], srcq)
        warped = src_img.transform((qx1 - qx0, qy1 - qy0), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        if shade < 0.999:
            v = int(255 * shade)
            warped = ImageChops.multiply(warped, Image.new("CMYK", warped.size, (v, v, v, 255)))
        _over_premult(canvas, warped, qx0, qy0)
    # spine shadow on the open spread
    if p > 0.2:
        g = Image.new("L", (cw, ch), 0)
        gd = ImageDraw.Draw(g)
        for k in range(12):
            gd.line((xs + k - 6, oy + 2, xs + k - 6, oy + hp - 2), fill=int(70 * (1 - abs(k - 6) / 6.0) * p))
        canvas.paste(_ink((0.05, 0.02, 0.02)), (0, 0), ImageChops.multiply(g, canvas.getchannel(3)))
    img = canvas
    if tilt:
        img = img.rotate(tilt, Image.BILINEAR)
    return _to_frame(img, W, H, cx * W - ox, cy * H - ch / 2.0)


# ============================================================ fico drift ===
def _superellipse(a, b, n, m=96):
    pts = []
    for i in range(m):
        ph = 2 * math.pi * i / m
        c, s_ = math.cos(ph), math.sin(ph)
        x = a * math.copysign(abs(c) ** (2 / n), c)
        y = b * math.copysign(abs(s_) ** (2 / n), s_)
        pts.append((x, y))
    return pts


@functools.lru_cache(maxsize=8)
def _fico_sprite(Lp, body=(0.93, 0.92, 0.88)):
    """Top-down Zastava 750 sprite, nose pointing right (+x). Premultiplied CMYK.
    Returns (sprite, shadow_mask_L, (cx, cy) centre in sprite px)."""
    Wc = int(Lp * 0.50)
    pad = int(Lp * 0.06)
    w, h = Lp + 2 * pad, Wc + 2 * pad
    cv = Cv(w, h, 3)
    ox, oy = w / 2.0, h / 2.0

    def T(pts):
        return [(ox + x * Lp, oy + y * Lp) for x, y in pts]
    a, b = 0.5, 0.222
    outline = []
    for x, y in _superellipse(a, b, 3.4):
        if x > 0:
            y *= 1 - 0.13 * (x / a) ** 2
        else:
            y *= 1 - 0.06 * (x / a) ** 2
        outline.append((x, y))
    chrome = (0.82, 0.84, 0.87)
    dark = (0.10, 0.10, 0.12)
    glass = (0.12, 0.16, 0.22)
    # bumpers: thin chrome strips following the nose / tail curvature
    for sx in (1, -1):
        ys = np.linspace(-0.165, 0.165, 21)
        pts = [(sx * (0.512 - 0.05 * (yv / 0.165) ** 2), yv) for yv in ys]
        cv.line(T(pts), dark, Lp * 0.026)
        cv.line(T(pts), chrome, Lp * 0.016)
    # mirrors
    for sy in (-1, 1):
        cv.ellipse(ox + 0.10 * Lp, oy + sy * 0.228 * Lp, 0.02 * Lp, 0.012 * Lp, chrome)
    cv.poly(T(outline), _mix(body, (0, 0, 0), 0.35))
    inner = [(x * 0.985, y * 0.965) for x, y in outline]
    cv.poly(T(inner), body)
    # front-trunk crease + badge
    cv.line(T([(0.22, 0.0), (0.46, 0.0)]), _mix(body, (0, 0, 0), 0.16), Lp * 0.004)
    cv.ellipse(ox + 0.47 * Lp, oy, 0.011 * Lp, 0.011 * Lp, (0.75, 0.1, 0.1))
    # headlights in the front wings, small tail lights
    for sy in (-1, 1):
        cv.ellipse(ox + 0.415 * Lp, oy + sy * 0.150 * Lp, 0.028 * Lp, 0.024 * Lp, chrome)
        cv.ellipse(ox + 0.42 * Lp, oy + sy * 0.150 * Lp, 0.019 * Lp, 0.016 * Lp, (1.0, 0.97, 0.82))
        cv.ellipse(ox - 0.475 * Lp, oy + sy * 0.15 * Lp, 0.012 * Lp, 0.02 * Lp, (0.75, 0.05, 0.06))
    # greenhouse (glass) + roof
    gh = [(x * 0.275 - 0.075, y * 0.188) for x, y in _superellipse(1, 1, 3.6, 64)]
    cv.poly(T(gh), glass)
    cv.poly(T([(0.10, -0.13), (0.17, -0.155), (0.175, -0.11), (0.115, -0.07)]), (0.40, 0.48, 0.58))
    cv.poly(T([(-0.30, 0.10), (-0.325, 0.12), (-0.33, 0.07), (-0.305, 0.06)]), (0.30, 0.36, 0.45))
    roof = [(x * 0.168 - 0.092, y * 0.166) for x, y in _superellipse(1, 1, 5, 64)]
    cv.poly(T(roof), _mix(body, (1, 1, 1), 0.3))
    # engine lid with cooling louvres (rear-engined!)
    cv.line(T([(-0.35, -0.155), (-0.35, 0.155)]), _mix(body, (0, 0, 0), 0.3), Lp * 0.004)
    for k in range(6):
        x = -0.375 - k * 0.020
        cv.line(T([(x, -0.08), (x, 0.08)]), _mix(body, (0, 0, 0), 0.55), Lp * 0.0065)
    img = cv.result()
    # rounded-body shading
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ny = (yy - oy) / (b * Lp)
    nx = (xx - ox) / (a * Lp)
    shade = 1.06 - 0.32 * np.clip(np.abs(ny), 0, 1) ** 3.5 - 0.18 * np.clip(np.abs(nx), 0, 1) ** 6
    shade += 0.10 * np.exp(-((ny + 0.35) / 0.25) ** 2) * (np.abs(nx) < 0.9)
    sl = _L(np.clip(shade, 0, 1))
    img = ImageChops.multiply(img, Image.merge("CMYK", (sl, sl, sl, Image.new("L", (w, h), 255))))
    shadow = img.getchannel(3).filter(ImageFilter.GaussianBlur(Lp * 0.025))
    return img, shadow, (ox, oy)


@functools.lru_cache(maxsize=2)
def _asphalt(W, H):
    rng = np.random.default_rng(750)
    grain = rng.random((H, W), dtype=np.float32)
    big = np.asarray(Image.fromarray((rng.random((H // 16 + 1, W // 16 + 1)) * 255).astype(np.uint8))
                     .resize((W, H), Image.BICUBIC), np.float32) / 255.0
    fine = np.asarray(Image.fromarray((grain * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)),
                      np.float32) / 255.0
    v = 0.15 + 0.07 * (fine - 0.5) + 0.018 * (big - 0.5)
    speck = (grain > 0.993) * 0.07
    v = v + speck
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.35 * (((xx / W - 0.5) * 1.6) ** 2 + ((yy / H - 0.5) * 1.6) ** 2)
    v = v * vig
    rgb = np.stack([v * 0.98, v * 0.99, v * 1.03], -1)
    im = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB")
    # old faint skid arcs + a worn painted line
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for k in range(7):
        r = H * (0.5 + 0.4 * _hash01(k, 1))
        cxk, cyk = W * _hash01(k, 2), H * (0.2 + 1.0 * _hash01(k, 3))
        a0 = 360 * _hash01(k, 4)
        d.arc((cxk - r, cyk - r, cxk + r, cyk + r), a0, a0 + 50 + 60 * _hash01(k, 5), fill=40,
              width=max(2, int(H * 0.012)))
    m = m.filter(ImageFilter.GaussianBlur(H * 0.002))
    im.paste((10, 10, 12), (0, 0), m)
    lm = Image.new("L", (W, H), 0)
    ld = ImageDraw.Draw(lm)
    rr = 2.2 * H
    cxl, cyl = 0.5 * W, 0.86 * H + rr
    for k in range(40):
        a0 = 238 + k * 1.6
        if k % 3 == 2:
            continue
        ld.arc((cxl - rr, cyl - rr, cxl + rr, cyl + rr), a0, a0 + 1.2, fill=80, width=max(2, int(H * 0.011)))
    im.paste((225, 222, 205), (0, 0), ImageChops.multiply(lm, _L(_noise(W, H, 77)[:H, :W] > 0.35)))
    return Image.merge("CMYK", (*im.split(), Image.new("L", (W, H), 255)))


def _fico_state(tt, dur, W, H, speed, radius):
    """Car pose at time(s) tt (scalar or array): centre x, y, heading psi (screen rad,
    clockwise), drift beta, velocity heading psi_v, speed px/s, turn centre."""
    tt = np.asarray(tt, np.float64)
    s = tt / max(dur, 1e-3) * speed
    a0, a1 = math.radians(198), math.radians(342)
    al = a0 + (a1 - a0) * s
    Rr = radius * H
    C = (0.5 * W, 0.30 * H + Rr)
    x = C[0] + Rr * np.cos(al)
    y = C[1] + Rr * np.sin(al)
    psi_v = np.arctan2(np.cos(al), -np.sin(al))
    beta = math.radians(38) * (0.55 + 0.45 * np.clip(s * 4, 0, 1)) * (1 + 0.12 * np.sin(2 * np.pi * 1.4 * s + 0.5))
    vel = Rr * (a1 - a0) * speed / max(dur, 1e-3)
    return x, y, psi_v + beta, beta, psi_v, vel, C


def fico_drift(t, dur, W, H, bg=True, smoke=1.0, color="white", scale=1.25, speed=1.0, radius=0.95,
               marks=True, mirror=False, **_):
    """Top-down drone view of a Zastava 750 'Fićo' drifting through a sweeping arc
    (enters bottom-left, arcs over the top, exits bottom-right over `dur`), body yawed
    into the turn with counter-steered front wheels, tyre smoke from the rear wheels
    and black skid marks on asphalt.

    bg      asphalt background (False = transparent, marks/smoke/car only)
    smoke   smoke density multiplier    color  'white' | 'red' | rgb tuple
    scale   car size multiplier (1.25 -> car length ~0.33 H)   speed  path speed multiplier
    radius  arc radius as fraction of H  marks  draw skid marks   mirror  right->left
    """
    body = {"white": (0.93, 0.92, 0.88), "red": (0.80, 0.07, 0.09)}.get(color, None) if isinstance(color, str) \
        else tuple(color)
    if body is None:
        body = _col(color, (0.93, 0.92, 0.88))
    Lp = int(0.26 * H * scale)
    canvas = _asphalt(W, H).copy() if bg else Image.new("CMYK", (W, H), (0, 0, 0, 0))
    x, y, psi, beta, psi_v, vel, C = _fico_state(t, dur, W, H, speed, radius)
    x, y, psi, beta = float(x), float(y), float(psi), float(beta)

    def wheel_pos(tt, wx, wy):
        xx, yy, ps, *_r = _fico_state(tt, dur, W, H, speed, radius)
        c, s_ = np.cos(ps), np.sin(ps)
        return xx + (wx * c - wy * s_) * Lp, yy + (wx * s_ + wy * c) * Lp
    rear = ((-0.30, -0.19), (-0.30, 0.19))
    front = ((0.31, -0.19), (0.31, 0.19))
    # ---- skid marks
    if marks:
        n = max(2, int(t * 60))
        taus = np.linspace(0.0, t, n)
        hk = 2                                   # draw at half res, upsample = soft edges
        m = Image.new("L", (W // hk + 1, H // hk + 1), 0)
        d = ImageDraw.Draw(m)
        tw = max(1, int(0.055 * Lp / hk))
        for (wx, wy), val in [(p, 150) for p in rear] + [(p, 60) for p in front]:
            px_a, py_a = wheel_pos(taus, wx, wy)
            d.line(list(zip((px_a / hk).tolist(), (py_a / hk).tolist())), fill=val, width=tw, joint="curve")
        bb = m.getbbox()
        if bb:
            x0, y0, x1, y1 = bb[0] * hk, bb[1] * hk, min(bb[2] * hk, W), min(bb[3] * hk, H)
            mm = m.crop(bb).resize((x1 - x0, y1 - y0), Image.BILINEAR)
            canvas.paste(_ink((0.02, 0.02, 0.025)), (x0, y0, x1, y1), mm)
    # ---- car shadow, wheels, body
    spr, shmask, (sox, soy) = _fico_sprite(Lp, body)
    sw, sh = spr.size
    car = Image.new("CMYK", (sw, sh), (0, 0, 0, 0))
    cd = ImageDraw.Draw(car)
    steer = -0.55 * beta
    for (wx, wy), ang in [(p, 0.0) for p in rear] + [(p, steer) for p in front]:
        cxw, cyw = sox + wx * Lp, soy + wy * Lp
        hl, hw = 0.062 * Lp, 0.028 * Lp
        c, s_ = math.cos(ang), math.sin(ang)
        q = [(cxw + dx * c - dy * s_, cyw + dx * s_ + dy * c) for dx, dy in
             ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]
        cd.polygon(q, fill=_ink((0.09, 0.09, 0.10)))
    _over_premult(car, spr, 0, 0)
    deg = -math.degrees(psi)
    car_r = car.rotate(deg, Image.BICUBIC, expand=True)
    sh_r = shmask.rotate(deg, Image.BILINEAR, expand=True)
    px_, py_ = int(x - car_r.size[0] / 2), int(y - car_r.size[1] / 2)
    so = int(0.05 * Lp)
    shp = ImageChops.multiply(sh_r, Image.new("L", sh_r.size, 150))
    canvas.paste(_ink((0, 0, 0)), (px_ + so, py_ + int(so * 1.3)), shp)
    _over_premult(canvas, car_r, px_, py_)
    # ---- tyre smoke (quarter-res density splats, vectorised particles)
    if smoke > 0:
        from scipy.ndimage import gaussian_filter
        k = max(1, int(round(6 * H / 1080)))
        sw4, sh4 = W // k + 1, H // k + 1
        life, dt = 1.5, 1.0 / 40.0
        kk = np.arange(max(0, int((t - life) / dt)), int(t / dt) + 1)
        te = kk * dt
        age = t - te
        ok = (age >= 0) & (age <= life)
        kk, te, age = kk[ok], te[ok], age[ok]
        if len(kk):
            xe, ye, pse, be, psv, ve, Ce = _fico_state(te, dur, W, H, speed, radius)
            inten = smoke * np.clip(be / math.radians(22), 0, 1) * (0.6 + 0.4 * np.clip(te * 3, 0, 1))
            ce, se = np.cos(pse), np.sin(pse)
            oxv, oyv = xe - Ce[0], ye - Ce[1]
            on = np.maximum(np.hypot(oxv, oyv), 1e-6)
            oxv, oyv = oxv / on, oyv / on
            classes = np.array([0.05, 0.085, 0.14, 0.22, 0.34]) * Lp / k
            grids = np.zeros((len(classes), sh4 * sw4), np.float64)
            damp = (1 - np.exp(-2.4 * age)) / 2.4
            for wi, (wx, wy) in enumerate(rear):
                h1 = (np.sin(kk * 12.9898 + wi * 78.233 + 1.3) * 43758.5453) % 1.0
                h2 = (np.sin(kk * 39.3468 + wi * 11.135 + 2.7) * 24634.6345) % 1.0
                h3 = (np.sin(kk * 73.1564 + wi * 52.235 + 4.1) * 17563.2231) % 1.0
                bx = xe + (wx * ce - wy * se) * Lp
                by = ye + (wx * se + wy * ce) * Lp
                spd = (0.25 + 0.45 * h1) * Lp
                vx = oxv * spd - np.cos(psv) * 0.2 * Lp + (h2 - 0.5) * 0.5 * Lp
                vy = oyv * spd - np.sin(psv) * 0.2 * Lp + (h3 - 0.5) * 0.5 * Lp - 0.02 * H
                gx = (bx + vx * damp) / k
                gy = (by + vy * damp) / k
                sig = (0.05 + 0.26 * (age / life) ** 0.5 * (0.75 + 0.5 * h2)) * Lp / k
                op = inten * 0.30 * np.clip(age / 0.05, 0, 1) * (1 - age / life) ** 1.9
                ci = np.clip(np.searchsorted(classes, sig), 0, len(classes) - 1)
                wgt = op * 2 * np.pi * classes[ci] ** 2
                ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
                fx, fy = gx - ix, gy - iy
                for ddx, ddy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                                     (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                    xx_, yy_ = ix + ddx, iy + ddy
                    m_ = (xx_ >= 0) & (xx_ < sw4) & (yy_ >= 0) & (yy_ < sh4)
                    np.add.at(grids, (ci[m_], (yy_ * sw4 + xx_)[m_]), (wgt * ww)[m_])
            dens = np.zeros((sh4, sw4), np.float32)
            for ci_, sg in enumerate(classes):
                g = grids[ci_].reshape(sh4, sw4)
                if g.any():
                    dens += gaussian_filter(g.astype(np.float32), sg, mode="constant", truncate=2.6)
            if dens.any():
                nz = _noise(sw4 + 64, sh4 + 64, 99, 3.5)
                ofx, ofy = int(t * 7) % 64, int(t * 4) % 64
                billow = nz[ofy:ofy + sh4, ofx:ofx + sw4]
                alpha = 1 - np.exp(-2.0 * dens * (0.25 + 1.5 * billow ** 1.5))
                aS = _L(np.clip(alpha, 0, 1))
                bb = aS.point(lambda v: 255 if v > 2 else 0).getbbox()
                if bb:
                    x0, y0 = max(0, (bb[0] - 1) * k), max(0, (bb[1] - 1) * k)
                    x1, y1 = min(W, (bb[2] + 1) * k), min(H, (bb[3] + 1) * k)
                    box = (x0 / k, y0 / k, x1 / k, y1 / k)
                    aL = aS.resize((x1 - x0, y1 - y0), Image.BILINEAR, box=box)
                    canvas.paste(_ink((0.60, 0.60, 0.62)), (x0, y0, x1, y1), aL)
                    a2 = _L(np.clip(alpha * (0.25 + 0.75 * billow), 0, 1)).resize((x1 - x0, y1 - y0),
                                                                                   Image.BILINEAR, box=box)
                    canvas.paste(_ink((0.92, 0.92, 0.93)), (x0, y0, x1, y1), a2)
    if mirror:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
    return _to_frame(canvas, W, H, 0, 0)


# ============================================================= test card ===
@functools.lru_cache(maxsize=2)
def _test_card_static(W, H):
    """Static JRT test card (RGB PIL) in the spirit of the Philips PM5544."""
    ss = 2
    im = Image.new("RGB", (W * ss, H * ss), (108, 108, 110))
    d = ImageDraw.Draw(im)
    u = H * ss
    cx, cy = W * ss / 2, H * ss / 2
    # grid
    step = u / 13.5
    lw = max(2, int(u / 540))
    k0 = int(cx / step) + 1
    for i in range(-k0, k0 + 1):
        x = cx + (i + 0.5) * step
        d.line((x, 0, x, u), fill=(235, 235, 235), width=lw)
    for j in range(-8, 9):
        y = cy + (j + 0.5) * step
        d.line((0, y, W * ss, y), fill=(235, 235, 235), width=lw)
    # castellations top / bottom
    for i in range(-k0, k0 + 1):
        x = cx + (i + 0.5) * step
        col = (20, 20, 20) if i % 2 == 0 else (240, 240, 240)
        d.rectangle((x, 0, x + step, step * 0.35), fill=col)
        d.rectangle((x, u - step * 0.35, x + step, u), fill=(240, 240, 240) if i % 2 == 0 else (20, 20, 20))
    # side colour blocks (outside the 4:3 area)
    bars = [(191, 191, 0), (0, 191, 191), (0, 191, 0), (191, 0, 191), (191, 0, 0), (0, 0, 191)]
    for sd in (-1, 1):
        x0 = cx + sd * u * 0.70
        for j, c in enumerate(bars):
            y0 = cy - 3 * step + j * step
            d.rectangle((min(x0, x0 + sd * step), y0, max(x0, x0 + sd * step), y0 + step), fill=c)
    # corner circles
    for sx in (-1, 1):
        for sy in (-1, 1):
            ccx, ccy = cx + sx * u * 0.56, cy + sy * u * 0.36
            r = step * 0.9
            d.ellipse((ccx - r, ccy - r, ccx + r, ccy + r), fill=(20, 20, 20), outline=(240, 240, 240), width=lw)
            d.line((ccx - r, ccy, ccx + r, ccy), fill=(240, 240, 240), width=lw)
            d.line((ccx, ccy - r, ccx, ccy + r), fill=(240, 240, 240), width=lw)
    # big circle content
    R = 0.44 * u
    inner = Image.new("RGB", (int(2 * R), int(2 * R)), (20, 20, 20))
    di = ImageDraw.Draw(inner)
    ic = R

    def band(y0, y1):
        return ic + y0 * u, ic + y1 * u
    # top station band
    a, b = band(-0.44, -0.27)
    di.rectangle((0, a, 2 * R, b), fill=(16, 16, 16))
    f = _fnt("russo", 0.12 * u)
    di.text((ic, (a + b) / 2 + 0.02 * u), "JRT", font=f, fill=(245, 245, 245), anchor="mm")
    # colour bars
    a, b = band(-0.27, -0.07)
    cb = [(192, 192, 192), (192, 192, 0), (0, 192, 192), (0, 192, 0), (192, 0, 192), (192, 0, 0), (0, 0, 192)]
    bw = 2 * R / len(cb)
    for i, c in enumerate(cb):
        di.rectangle((i * bw, a, (i + 1) * bw, b), fill=c)
    # centre band (black with white cross)
    a, b = band(-0.07, 0.07)
    di.rectangle((0, a, 2 * R, b), fill=(12, 12, 12))
    for i in range(1, 12):
        x = i * 2 * R / 12
        di.line((x, a, x, b), fill=(90, 90, 90), width=lw // 2 + 1)
    di.line((0, ic, 2 * R, ic), fill=(250, 250, 250), width=lw * 2)
    di.line((ic, a, ic, b), fill=(250, 250, 250), width=lw * 2)
    rr = 0.035 * u
    di.ellipse((ic - rr, ic - rr, ic + rr, ic + rr), outline=(250, 250, 250), width=lw * 2)
    # greyscale steps
    a, b = band(0.07, 0.17)
    for i in range(6):
        g = int(255 * i / 5)
        di.rectangle((i * 2 * R / 6, a, (i + 1) * 2 * R / 6, b), fill=(g, g, g))
    # frequency gratings
    a, b = band(0.17, 0.27)
    di.rectangle((0, a, 2 * R, b), fill=(128, 128, 128))
    nb = 5
    for i in range(nb):
        x0, x1 = 2 * R * (0.08 + 0.84 * i / nb), 2 * R * (0.08 + 0.84 * (i + 1) / nb) - 0.01 * u
        per = max(2.0, (0.030 - 0.0055 * i) * u)
        x = x0
        while x < x1:
            di.rectangle((x, a + 0.008 * u, min(x + per / 2, x1), b - 0.008 * u), fill=(240, 240, 240))
            x += per
    # bottom station band
    a, b = band(0.27, 0.44)
    di.rectangle((0, a, 2 * R, b), fill=(16, 16, 16))
    f2 = _fnt("russo", 0.058 * u)
    di.text((ic, a + 0.05 * u), "TV BEOGRAD", font=f2, fill=(245, 245, 245), anchor="mm")
    m = Image.new("L", inner.size, 0)
    ImageDraw.Draw(m).ellipse((0, 0, 2 * R - 1, 2 * R - 1), fill=255)
    im.paste(inner, (int(cx - R), int(cy - R)), m)
    d.ellipse((cx - R, cy - R, cx + R, cy + R), outline=(245, 245, 245), width=lw * 2)
    im = im.reduce(ss)
    return im


def test_card(t, dur, W, H, glitch=0.0, clock=True, start="19:59:30", scan=True, **_):
    """1970s JRT / TV Beograd test card (PM5544-style: grid, big circle, colour bars,
    greyscale steps, gratings, 'JRT' + 'TV BEOGRAD') with a running HH:MM:SS clock.

    glitch  0..1 analogue damage: tearing bands, chroma shift, snow, vertical roll, flicker
    start   clock time at t=0      scan  CRT scanlines.   Opaque output.
    """
    im = _test_card_static(W, H).copy()
    if clock:
        hh, mm, ss_ = (int(v) for v in start.split(":"))
        tot = hh * 3600 + mm * 60 + ss_ + int(math.floor(t))
        txt = f"{(tot // 3600) % 24:02d}:{(tot // 60) % 60:02d}:{tot % 60:02d}"
        d = ImageDraw.Draw(im)
        f = _fnt("mono", 0.055 * H)
        cx, cy = W / 2, H / 2 + 0.375 * H
        wtxt = f.getlength("00:00:00")
        d.rectangle((cx - wtxt / 2 - 0.02 * H, cy - 0.038 * H, cx + wtxt / 2 + 0.02 * H, cy + 0.038 * H),
                    fill=(8, 8, 8))
        d.text((cx, cy), txt, font=f, fill=(250, 250, 250), anchor="mm")
    a = np.asarray(im, np.float32) * (1.0 / 255.0)
    g = float(glitch)
    fr = int(t * 25)
    if g > 0:
        a = a.copy()
        # vertical roll
        if g > 0.35 and _hash01(fr // 3, 9.1) > 0.55:
            a = np.roll(a, int(H * _hash01(fr, 9.2) * g * 0.6), axis=0)
        # tearing bands
        for k in range(int(1 + 7 * g)):
            y0 = int(H * _hash01(fr, k, 1.7))
            hh_ = int(H * (0.01 + 0.07 * _hash01(fr, k, 2.9)))
            dx = int((_hash01(fr, k, 3.1) - 0.5) * 0.25 * W * g)
            if dx:
                a[y0:y0 + hh_] = np.roll(a[y0:y0 + hh_], dx, axis=1)
        # chroma shift
        sh = int(round(12 * g * H / 1080 * (0.5 + _hash01(fr, 4.4))))
        if sh:
            a[..., 0] = np.roll(a[..., 0], sh, axis=1)
            a[..., 2] = np.roll(a[..., 2], -sh, axis=1)
        # snow
        nz = _noise(W, H + 64, 55)
        off = int(_hash01(fr, 5.5) * 60)
        a += (nz[off:off + H, :W, None] - 0.5) * (0.5 * g)
        a *= 1.0 + (_hash01(fr, 6.6) - 0.5) * 0.25 * g
    if scan:
        a[1::2] *= 0.86
    out = np.empty((H, W, 4), np.float32)
    np.clip(a, 0, 1, out=out[..., :3])
    out[..., 3] = 1.0
    return out


# ================================================================= vinyl ===
@functools.lru_cache(maxsize=4)
def _vinyl_static(D, label, sub, lab_rgb):
    """(disc+label sprite CMYK premult, highlight sprite CMYK premult)."""
    R = D / 2.0
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    x, y = (xx + 0.5 - R) / R, (yy + 0.5 - R) / R
    r = np.sqrt(x * x + y * y)
    ang = np.arctan2(y, x)
    cov = np.clip((1 - r) * R + 0.5, 0, 1)
    grooves = 0.5 + 0.5 * np.sin(r * R * 1.9)
    gap = np.zeros_like(r)
    for g0 in (0.52, 0.64, 0.76, 0.88):
        gap += np.exp(-((r - g0) / 0.006) ** 2)
    runout = (r < 0.40) & (r > 0.335)
    base = 0.035 + 0.018 * grooves * (1 - np.clip(gap, 0, 1)) * (~runout)
    base += 0.03 * np.clip((r - 0.975) / 0.025, 0, 1)
    # dust specks + faint scratches (these make the rotation visible)
    rng = np.random.default_rng(33)
    sp = np.zeros_like(r)
    ids = rng.integers(0, D * D, 500)
    sp.flat[ids] = rng.random(500) * 0.5
    for k in range(3):
        rr0 = rng.uniform(0.45, 0.95)
        a0 = rng.uniform(-np.pi, np.pi)
        sp += 0.08 * np.exp(-((r - rr0) / 0.0025) ** 2) * (np.abs(np.angle(np.exp(1j * (ang - a0)))) < 0.5)
    v = np.clip(base + sp * (r > 0.34), 0, 1)
    rgb = np.stack([v, v, v * 1.05], -1)
    disc = np.zeros((D, D, 4), np.uint8)
    disc[..., :3] = np.clip(rgb * cov[..., None] * 255, 0, 255)
    disc[..., 3] = np.clip(cov * 255, 0, 255)
    img = Image.frombuffer("CMYK", (D, D), disc.tobytes(), "raw", "CMYK", 0, 1)
    # label
    cv = Cv(D, D, 2)
    lr = 0.335 * R
    c0 = R
    cv.ellipse(c0, c0, lr, lr, lab_rgb)
    cv.ring(c0, c0, lr * 0.96, _mix(lab_rgb, (0, 0, 0), 0.35), lr * 0.015)
    cv.ring(c0, c0, lr * 0.60, _mix(lab_rgb, (1, 1, 1), 0.25), lr * 0.01)
    f_px = lr * 0.30
    fw = _fnt("russo", f_px * 2).getlength(label) / 2
    f_px *= min(1.0, lr * 1.45 / max(fw, 1))
    cv.text((c0, c0 - lr * 0.42), label, "russo", f_px, WHITE)
    cv.text((c0, c0 + lr * 0.40), sub, "noto", lr * 0.09, WHITE, weight=600)
    cv.text((c0, c0 + lr * 0.60), "ZAGREB · YUGOSLAVIA", "noto", lr * 0.07, WHITE, weight=500)
    cv.text((c0 - lr * 0.62, c0), "A", "russo", lr * 0.16, WHITE)
    cv.text((c0 + lr * 0.62, c0), "LP", "russo", lr * 0.12, WHITE)
    cv.poly(_star_poly(c0, c0 - lr * 0.78, lr * 0.07), GOLD)
    lab = cv.result()
    _over_premult(img, lab, 0, 0)
    # spindle hole
    hole = Image.new("L", (D, D), 255)
    hr = 0.024 * R
    ImageDraw.Draw(hole).ellipse((R - hr, R - hr, R + hr, R + hr), fill=0)
    img = ImageChops.multiply(img, Image.merge("CMYK", (hole, hole, hole, hole)))
    # fixed anisotropic highlight (two opposite wedges following the grooves)
    la = math.radians(-50)
    lobe = np.clip(np.cos(ang - la), 0, 1) ** 10 + 0.55 * np.clip(np.cos(ang - la - np.pi), 0, 1) ** 12
    fine = 0.6 + 0.4 * np.sin(r * R * 0.9) ** 2
    hl = lobe * fine * np.clip((r - 0.36) / 0.05, 0, 1) * cov * 0.62
    hl += 0.06 * np.clip(np.cos(ang - la), 0, 1) ** 2 * (r < 0.335) * cov
    h4 = np.zeros((D, D, 4), np.uint8)
    h4[..., 0] = np.clip(hl * 0.92 * 255, 0, 255)
    h4[..., 1] = np.clip(hl * 0.96 * 255, 0, 255)
    h4[..., 2] = np.clip(hl * 1.0 * 255, 0, 255)
    h4[..., 3] = h4[..., :3].max(axis=2)
    himg = Image.frombuffer("CMYK", (D, D), h4.tobytes(), "raw", "CMYK", 0, 1)
    return img, himg


def vinyl(t, dur, W, H, label="JUGOTON", size=0.85, rpm=33.3, cx=0.5, cy=0.5, tilt=0.0,
          sub="STEREO · 33⅓", color=None, **_):
    """Spinning black vinyl LP with grooves, track gaps, dust, a fixed anisotropic
    specular highlight and a rotating red centre label (default 'JUGOTON').

    size  disc diameter as fraction of H   rpm  rotation speed    tilt  0..1.3 rad lay-back
    """
    D = int(size * H) // 2 * 2
    lab_rgb = _col(color, RED)
    disc, hl = _vinyl_static(D, label, sub, tuple(lab_rgb))
    ang = -(rpm * 6.0 * t) % 360.0
    rot = disc.rotate(ang, Image.BILINEAR)
    img = ImageChops.screen(rot, hl)
    if tilt:
        img = img.resize((D, max(2, int(D * math.cos(tilt)))), Image.BILINEAR)
    w, h = img.size
    return _to_frame(img, W, H, cx * W - w / 2, cy * H - h / 2)


# ============================================================= snowflake ===
def _snowflake_strokes():
    """[(kind, geometry, q0, q1)] in unit coords (arm length 1), angle 0 = up."""
    S = []
    for k in range(6):
        a = -math.pi / 2 + k * math.pi / 3

        def P(r, off=0.0, a=a):
            return (r * math.cos(a + off), r * math.sin(a + off))

        def along(r, ang, ln, a=a):
            x0, y0 = r * math.cos(a), r * math.sin(a)
            return [(x0, y0), (x0 + ln * math.cos(a + ang), y0 + ln * math.sin(a + ang))]
        dk = k * 0.012
        S.append(("line", [P(0.16), P(1.0)], 0.08 + dk, 0.42 + dk))
        for j, (rj, lj) in enumerate(((0.40, 0.30), (0.63, 0.23), (0.83, 0.13))):
            for sg in (-1, 1):
                S.append(("line", along(rj, sg * math.radians(52), lj), 0.34 + 0.09 * j + dk, 0.55 + 0.09 * j + dk))
        # tip diamond
        tip = [P(0.97), P(1.05, 0.055), P(1.16), P(1.05, -0.055)]
        S.append(("diamond", tip, 0.72 + dk, 0.88 + dk))
        # bisector diamond
        b = a + math.pi / 6
        dm = [(0.26 * math.cos(b), 0.26 * math.sin(b)), (0.33 * math.cos(b + 0.1), 0.33 * math.sin(b + 0.1)),
              (0.42 * math.cos(b), 0.42 * math.sin(b)), (0.33 * math.cos(b - 0.1), 0.33 * math.sin(b - 0.1))]
        S.append(("diamond", dm, 0.60 + dk, 0.80 + dk))
    hexa = [(0.16 * math.cos(-math.pi / 2 + k * math.pi / 3), 0.16 * math.sin(-math.pi / 2 + k * math.pi / 3))
            for k in range(7)]
    S.append(("line", hexa, 0.0, 0.16))
    S.append(("dot", [(0.0, 0.0)], 0.86, 1.0))
    return S


_SNOW = _snowflake_strokes()


def _partial(pts, f):
    """First fraction f (by length) of a polyline."""
    if f >= 1:
        return pts
    segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    tot = sum(segs) * f
    out = [pts[0]]
    for i, sl in enumerate(segs):
        if tot >= sl:
            out.append(pts[i + 1])
            tot -= sl
        else:
            p = tot / sl if sl else 0
            out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * p, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * p))
            break
    return out


def snowflake(t, dur, W, H, size=0.7, draw_dur=1.2, delay=0.0, spin=0.12, cx=0.5, cy=0.5,
              color=None, glow=0.9, glow_color=(0.45, 0.75, 1.0), width=1.0, **_):
    """Geometric six-fold snowflake emblem (Sarajevo '84 spirit, no rings) that draws
    itself on: centre hexagon -> spines -> branches -> diamonds -> centre dot.

    size       flake diameter as fraction of H   draw_dur  build-up time (s) after `delay`
    spin       rotation (rad/s)                  color     stroke colour (default warm white)
    glow       bloom amount (tinted glow_color)  width     stroke width multiplier
    """
    q = _clamp((t - delay) / max(draw_dur, 1e-3))
    Rf = size * H / 2.0 / 1.16
    pad = int(Rf * 0.25)
    side = int(2 * Rf * 1.16 + 2 * pad)
    cv = Cv(side, side, 2)
    c0 = side / 2.0
    rot = spin * t
    cr, sr = math.cos(rot), math.sin(rot)
    col = _col(color, WHITE)
    lw = 0.030 * Rf * width

    def X(p):
        return (c0 + (p[0] * cr - p[1] * sr) * Rf, c0 + (p[0] * sr + p[1] * cr) * Rf)
    for kind, pts, q0, q1 in _SNOW:
        f = _seg(q, q0, q1)
        if f <= 0:
            continue
        if kind == "line":
            cv.line([X(p) for p in _partial(pts, ease_out_cubic(f))], col, lw)
        elif kind == "diamond":
            loop = pts + [pts[0]]
            cv.line([X(p) for p in _partial(loop, ease_out_cubic(f))], col, lw * 0.8)
            if f >= 1:
                fill = _seg(q, q1, q1 + 0.1)
                if fill > 0:
                    cv.poly([X(p) for p in pts], col, a=1.0 if fill >= 1 else fill)
        elif kind == "dot":
            s = ease_out_back(f, 3.0)
            cv.ellipse(c0, c0, 0.06 * Rf * s, 0.06 * Rf * s, col)
    img = cv.result()
    if glow > 0:
        img = _glow(img, glow, Rf * 0.03, Rf * 0.18, tint=glow_color)
    return _to_frame(img, W, H, cx * W - c0, cy * H - c0)


# ============================================================ basketball ===
@functools.lru_cache(maxsize=4)
def _ball_grid(R):
    """Per-radius cached normals, coverage and view/light dependent shading terms."""
    n = 2 * R + 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (xx + 0.5 - n / 2) / R
    y = (n / 2 - (yy + 0.5)) / R
    d2 = x * x + y * y
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    cov = np.clip((1 - np.sqrt(d2)) * R + 0.5, 0, 1)
    N = np.stack([x, y, z], -1).astype(np.float32)
    L = np.array([-0.45, 0.55, 0.70], np.float32)
    L /= np.linalg.norm(L)
    lam = np.clip(N @ L, 0, 1)
    Hh = L + np.array([0, 0, 1], np.float32)
    Hh /= np.linalg.norm(Hh)
    spec = np.clip(N @ Hh, 0, 1) ** 28
    base = np.array([0.90, 0.40, 0.10], np.float32)
    light = (0.22 + 0.88 * lam) * (0.75 + 0.25 * z ** 0.5)
    rim = np.clip(1 - z, 0, 1) ** 3 * np.clip(x * 0.8 + 0.2, 0, 1)
    add = rim[..., None] * np.array([0.9, 0.45, 0.15], np.float32) * 0.5 + spec[..., None] * 0.28
    inside = cov > 0
    return N, cov, base[None, None, :] * light[..., None], add.astype(np.float32), inside


def _rot_axis(ax, th):
    ax = np.asarray(ax, np.float64)
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def basketball(t, dur, W, H, size=0.5, spin=5.0, cx=0.5, cy=0.5, motion=1.0, direction=0.0,
               axis=(0.35, 1.0, 0.25), **_):
    """Rotating basketball (orange, pebbled, black seams) with speed lines / whoosh
    arcs trailing opposite to `direction` (rad, 0 = flying right).

    size    ball diameter as fraction of H   spin  rad/s about `axis`   motion  speed-line amount
    """
    R = int(size * H / 2)
    N, cov, lit, add, inside = _ball_grid(R)
    M = (_rot_axis(axis, spin * t) @ _rot_axis((1.0, 0.4, 0.2), 0.75)).astype(np.float32)
    P = N @ M                              # world normal -> ball coords (M^T n)
    px, py = P[..., 0], P[..., 1]
    c = 0.62
    apx = np.abs(px)
    dseam = np.minimum(np.minimum(apx, np.abs(py)), np.abs(apx - c))
    w = 0.020
    aa = 1.5 / R
    seam = np.clip((w + aa - dseam) * (1.0 / aa), 0, 1)
    groove = np.clip((w * 3.2 - dseam) * (1.0 / (w * 2.2)), 0, 1)
    q = np.floor((P + 1) * 55)
    peb = np.sin(q[..., 0] * 12.9898 + q[..., 1] * 78.233 + q[..., 2] * 37.719) * 437.5453
    peb -= np.floor(peb)
    shade = (0.93 + 0.08 * peb) * (1 - 0.35 * groove)
    col = lit * shade[..., None] + add
    sc = np.array([0.07, 0.035, 0.02], np.float32)
    col += (sc - col) * seam[..., None]
    n = N.shape[0]
    b4 = np.zeros((n, n, 4), np.uint8)
    b4[..., :3] = np.clip(col * cov[..., None] * 255, 0, 255)
    b4[..., 3] = np.clip(cov * 255, 0, 255)
    ball = Image.frombuffer("CMYK", (n, n), b4.tobytes(), "raw", "CMYK", 0, 1)
    # speed lines: collect geometry first, then draw on a canvas cropped to its bbox
    ext = int(R * 1.9)
    c0 = n / 2.0 + ext
    dx, dy = math.cos(direction), math.sin(direction)
    nx, ny = -dy, dx
    strokes = []
    if motion > 0:
        for i in range(7):
            off = (-0.78 + 1.56 * i / 6) * R
            back = math.sqrt(max(0.0, R * R - off * off))
            ln = R * (0.55 + 0.75 * _hash01(i, 1.7)) * (0.75 + 0.25 * math.sin(t * 17 + i * 1.3)) * motion
            gap = R * 0.12
            x0 = c0 - dx * (back + gap) + nx * off
            y0 = c0 - dy * (back + gap) + ny * off
            x1 = x0 - dx * ln
            y1 = y0 - dy * ln
            wdt = R * (0.035 - 0.02 * abs(off) / R)
            segs = 6
            for sgi in range(segs):
                fa, fb = sgi / segs, (sgi + 1) / segs
                strokes.append(([(x0 + (x1 - x0) * fa, y0 + (y1 - y0) * fa),
                                 (x0 + (x1 - x0) * fb, y0 + (y1 - y0) * fb)],
                                (1.0, 0.93, 0.85), wdt * (1 - 0.6 * fa), (1 - fa) ** 1.5 * 0.85, False))
        for j, rr in enumerate((1.18, 1.36)):
            span = math.radians(40 - 10 * j)
            base_a = math.atan2(-dy, -dx)
            pts = [(c0 + rr * R * math.cos(base_a + s_), c0 + rr * R * math.sin(base_a + s_))
                   for s_ in np.linspace(-span, span, 24)]
            strokes.append((pts, (1.0, 0.80, 0.45), R * 0.018 * (1 - 0.3 * j), 0.65 - 0.25 * j, True))
    xs_ = [c0 - n / 2, c0 + n / 2] + [p_[0] for st in strokes for p_ in st[0]]
    ys_ = [c0 - n / 2, c0 + n / 2] + [p_[1] for st in strokes for p_ in st[0]]
    bx0, by0 = int(min(xs_)) - 4, int(min(ys_)) - 4
    bx1, by1 = int(max(xs_)) + 5, int(max(ys_)) + 5
    cvl = Cv(bx1 - bx0, by1 - by0, 2)
    for pts, colr, wd, al, caps in strokes:
        cvl.line([(x - bx0, y - by0) for x, y in pts], colr, wd, a=al, caps=caps)
    img = cvl.result()
    _over_premult(img, ball, int(round(c0 - n / 2)) - bx0, int(round(c0 - n / 2)) - by0)
    return _to_frame(img, W, H, cx * W - c0 + bx0, cy * H - c0 + by0)


# ============================================================ __main__ ===
def _sheet(fn, samples, W=960, H=540, cols=4, bg=(0.07, 0.07, 0.09), label=None, **kw):
    """Contact sheet of fn at the given (t, dur, params) samples."""
    rows = int(math.ceil(len(samples) / cols))
    tw, th = W // 2, H // 2
    sheet = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    fnt = font("mono", 12)
    for k, (t, dur, params) in enumerate(samples):
        p = dict(kw)
        p.update(params)
        img = fn(t, dur, W, H, **p)
        yy = np.linspace(0, 1, H)[:, None, None]
        b = (np.array(bg) * (1.3 - 0.6 * yy)).astype(np.float32)
        b = np.broadcast_to(b, (H, W, 3))
        a = img[..., 3:4]
        comp = b * (1 - a) + img[..., :3] * a
        tile = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
        d = ImageDraw.Draw(tile)
        lab = f"{label or fn.__name__} t={t:.2f} " + " ".join(f"{k_}={v}" for k_, v in params.items())
        d.text((5, 3), lab[:70], font=fnt, fill=(210, 210, 210))
        sheet.paste(tile, ((k % cols) * tw, (k // cols) * th))
    return sheet


def _time(fn, W=1920, H=1080, reps=3, **kw):
    fn(0.37, 4.0, W, H, **kw)
    ts = []
    for i in range(reps):
        t0 = time.perf_counter()
        fn(0.5 + 0.61 * i, 4.0, W, H, **kw)
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


PREVIEWS = {}


def _register(name, fn, samples, **kw):
    PREVIEWS[name] = (fn, samples, kw)


_register("flag", flag, [(t, 4, {}) for t in (0.0, 0.4, 0.8, 1.2)] +
          [(0.3, 4, {"wave": 0.0}), (0.6, 4, {"wave": 1.6}), (1.0, 4, {"scale": 0.5}), (2.0, 4, {"speed": 2.0})])
_register("coat_of_arms", coat_of_arms, [(0.5 * i, 4, {"reveal": r}) for i, r in
                                         enumerate((0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.0))])
_register("star2d", star2d, [(t, 4, {}) for t in (0.0, 0.5, 0.9, 1.3)] +
          [(0.2, 4, {"bevel": True}), (0.7, 4, {"pulse": 1.0, "glow": 1.0}), (1.1, 4, {"size": 0.3}),
           (0.3, 4, {"rot": 0.3, "shine": False})])


_register("passport", passport, [(0.0, 6, {}), (0.3, 6, {"open_at": 0.2}), (0.45, 6, {"open_at": 0.2}),
                                 (0.62, 6, {"open_at": 0.2}), (0.95, 6, {"open_at": 0.2}),
                                 (1.30, 6, {"open_at": 0.2}), (2.2, 6, {"open_at": 0.2}),
                                 (3.5, 6, {"open_at": 0.2})])


_register("fico_drift", fico_drift, [(t, 4, {}) for t in (0.3, 1.0, 1.6, 2.2, 2.8, 3.5)] +
          [(1.8, 4, {"bg": False, "color": "red"}), (2.0, 4, {"mirror": True, "smoke": 1.5})])


_register("test_card", test_card, [(0.0, 4, {}), (1.0, 4, {}), (0.4, 4, {"glitch": 0.3}), (0.8, 4, {"glitch": 0.6}),
                                   (1.2, 4, {"glitch": 1.0}), (2.37, 4, {"glitch": 0.8}), (3.0, 4, {"scan": False}),
                                   (3.5, 4, {"glitch": 0.15})])
_register("vinyl", vinyl, [(t, 4, {}) for t in (0.0, 0.3, 0.6, 0.9)] +
          [(0.2, 4, {"tilt": 0.9}), (0.5, 4, {"label": "PGP RTB", "color": "blue"}), (1.0, 4, {"size": 0.5}),
           (1.3, 4, {"tilt": 1.1, "size": 1.1})])
_register("snowflake", snowflake, [(t, 4, {}) for t in (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5)] +
          [(2.0, 4, {"color": "cyan", "glow": 1.4})])
_register("basketball", basketball, [(t, 4, {}) for t in (0.0, 0.13, 0.26, 0.39)] +
          [(0.5, 4, {"direction": math.pi}), (0.7, 4, {"direction": -0.6, "size": 0.35}),
           (0.9, 4, {"motion": 0.0}), (1.1, 4, {"spin": 12.0})])


def main(which=None):
    """Write out/preview/emblems_<name>.png contact sheets (960x540 samples) and print
    per-call timings at 1920x1080 (min over 2 runs, worst over the sheet's param sets)."""
    prev = os.path.join(OUT, "preview")
    os.makedirs(prev, exist_ok=True)
    names = which or list(PREVIEWS)
    for n in names:
        fn, samples, kw = PREVIEWS[n]
        sh = _sheet(fn, samples, **kw)
        p = os.path.join(prev, f"emblems_{n}.png")
        sh.save(p)
        times = []
        for (t, dur, params) in samples:
            fn(t, dur, 1920, 1080, **params)            # warm caches
            ts = []
            for _k in range(2):
                t0 = time.perf_counter()
                fn(t + 0.017, dur, 1920, 1080, **params)
                ts.append((time.perf_counter() - t0) * 1e3)
            times.append(min(ts))
        print(f"{n:14s} mean {np.mean(times):6.1f} ms  worst {max(times):6.1f} ms @1920x1080  -> {p}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
