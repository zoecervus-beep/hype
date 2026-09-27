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
                         ease_out_cubic, ease_in_out_cubic, ease_in_cubic, ease_out_expo)

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
    cell = max(8, int(round(22 * H / 1080)))
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
    pad = int(R * 0.45 + 8)
    side = int(2 * R + 2 * pad)
    cv = Cv(side, side, 2)
    c0 = side / 2.0
    ccy = c0 + R * 0.095          # optical centring (star bbox centre)
    red = _col(color, RED)
    cv.poly(_star_poly(c0, ccy, R, rot), GOLD)
    inner = _star_poly(c0, ccy, R, rot, inset=border * R)
    cv.poly(inner, red)
    if bevel:
        for i in range(10):
            p0, p1 = inner[i], inner[(i + 1) % 10]
            lit = 0.18 if i % 2 == 0 else -0.22
            c = tuple(min(1.0, x * (1 + lit)) for x in red)
            cv.poly([(c0, ccy), p0, p1], c)
    img = cv.result()
    if shine:
        per = max(0.3, shine_period)
        p = ((t + shine_offset) % per) / per
        pos = -0.15 + 1.3 * ease_in_out_cubic(min(1.0, p / 0.55))
        m = Image.new("L", (side, side), 0)
        md = ImageDraw.Draw(m)
        bw = 0.16 * R
        ang = math.radians(-35)
        dx, dy = math.cos(ang), math.sin(ang)
        # band perpendicular to (dx, dy), centred at pos along it
        cxb = c0 + (pos - 0.5) * 2 * R * dx
        cyb = ccy + (pos - 0.5) * 2 * R * dy
        L = 3 * R
        nx, ny = -dy, dx
        for wmul, val in ((1.0, 110), (0.4, 235)):
            hw = bw * wmul
            md.polygon([(cxb - dx * hw - nx * L, cyb - dy * hw - ny * L),
                        (cxb + dx * hw - nx * L, cyb + dy * hw - ny * L),
                        (cxb + dx * hw + nx * L, cyb + dy * hw + ny * L),
                        (cxb - dx * hw + nx * L, cyb - dy * hw + ny * L)], fill=val)
        m = m.filter(ImageFilter.GaussianBlur(max(1.0, R * 0.02)))
        m = ImageChops.multiply(m, _alpha(img))
        img.paste(_ink((1.0, 0.97, 0.85)), (0, 0), m)
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
    dred = mono and tuple(c * 0.75 for c in mono) or DRED
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


def main(which=None):
    prev = os.path.join(OUT, "preview")
    os.makedirs(prev, exist_ok=True)
    names = which or list(PREVIEWS)
    for n in names:
        fn, samples, kw = PREVIEWS[n]
        sh = _sheet(fn, samples, **kw)
        p = os.path.join(prev, f"emblems_{n}.png")
        sh.save(p)
        ms = _time(fn, **{k: v for k, v in samples[0][2].items()})
        print(f"{n:14s} {ms:7.1f} ms @1920x1080   -> {p}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
