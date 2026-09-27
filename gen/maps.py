"""Map generators for the SFRJ hype edit (motion-graphics cartography).

Every public generator is ``fn(t, dur, W, H, **params) -> float32 RGBA (H, W, 4)``
(straight alpha, transparent background unless ``bg=`` is given), deterministic
and cached so that a 1920x1080 frame renders in roughly 60-120 ms.

Public API (full parameter docs in each docstring)::

    yugo_map(t, dur, W, H, mode="draw" | "assemble" | "republics" | "neighbors"
                                | "capitals" | "tilt3d", **kw)
    world_nam(t, dur, W, H, **kw)       # 1961 Belgrade Non-Aligned conference
    europe_blocs(t, dur, W, H, **kw)    # NATO / Warsaw Pact / SFRJ in between

Geometry comes from ``assets/geo/*.json`` (built once by ``gen/geo_build.py``
from Natural Earth).  Rendering: PIL ImageDraw into L masks (2x supersampled,
limited to the bbox of what is drawn), glows from blurred quarter-resolution
masks, composited premultiplied in 8 bit and converted to float32 straight
alpha at the end.  Static layers are cached per (W, H, view, style).

    python3 gen/maps.py [name ...]   # contact sheets -> out/preview/maps_*.png + ms/frame
"""
from __future__ import annotations

import functools
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.core import PAL, font, star_points  # noqa: E402
import geoproj  # noqa: E402

GEO = os.path.join(_ROOT, "assets", "geo")

# ================================================================ colours ===
RED = PAL["red"]
DEEP = PAL["deep_red"]
GOLD = PAL["gold"]
WHITE = PAL["white"]
STEEL = PAL["steel"]
NAVY = PAL["navy"]
BLACK = PAL["black"]
BLUE = PAL["blue"]
ICE = (0.45, 0.68, 1.0)      # cold NATO blue (lines / glow)
NATO_FILL = (0.13, 0.30, 0.78)


def _col(c):
    """Colour spec -> float rgb.  PAL names, '#hex', 0-1 floats or 0-255 ints."""
    if c is None:
        return None
    if isinstance(c, str):
        if c in PAL:
            return tuple(PAL[c])
        h = c.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    c = tuple(float(v) for v in c[:3])
    if max(c) > 1.0:
        c = tuple(v / 255.0 for v in c)
    return c


def _mix(a, b, k):
    k = min(max(k, 0.0), 1.0)
    return tuple(x + (y - x) * k for x, y in zip(a, b))


def _u8(c):
    return tuple(int(round(min(max(v, 0.0), 1.0) * 255)) for v in c)


def _hot(c, k=0.45):
    """'Hot' core version of a colour (towards white) for glowing lines."""
    return _mix(c, (1.0, 1.0, 1.0), k)


# ================================================================ easing ===
def _cl(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def _seg(p, a, b):
    """Local 0..1 progress of p inside [a, b]."""
    if b <= a:
        return 1.0 if p >= b else 0.0
    return _cl((p - a) / (b - a))


def _eio(x):
    x = _cl(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def _eo3(x):
    x = _cl(x)
    return 1 - (1 - x) ** 3


def _eo5(x):
    x = _cl(x)
    return 1 - (1 - x) ** 5


def _eback(x, s=1.70158):
    x = _cl(x) - 1
    return x * x * ((s + 1) * x + s) + 1


def _pulse(x, w):
    """1 at x=0 decaying exponentially for x>=0, 0 before."""
    return 0.0 if x < 0 else math.exp(-x / max(w, 1e-6))


# ================================================================== data ====
def _arr(flat):
    return np.asarray(flat, dtype=np.float64).reshape(-1, 2)


def _polys(plist):
    out = []
    for rings in plist:
        rs = [_arr(r) for r in rings]
        e = rs[0]
        out.append((rs, (e[:, 0].min(), e[:, 1].min(), e[:, 0].max(), e[:, 1].max())))
    return out


def _bbox_of(polys):
    b = np.array([bb for _, bb in polys])
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


def _cum(a):
    d = np.hypot(*np.diff(a, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _load_json(name):
    with open(os.path.join(GEO, name + ".json"), encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(None)
def _yu():
    d = _load_json("yugo")
    D = _NS(meta=d["meta"])
    D.reps = [_NS(id=r["id"], name=r["name"], en=r["en"], capital=r["capital"],
                  polys=_polys(r["p"]), label=np.array(r["label"], float)) for r in d["republics"]]
    D.provs = {p["id"]: _NS(id=p["id"], name=p["name"], polys=_polys(p["p"]),
                            label=np.array(p["label"], float)) for p in d["provinces"]}
    D.rep_borders = [_arr(ln) for ln in d["rep_borders"]]
    D.prov_borders = [_arr(ln) for ln in d["prov_borders"]]
    D.yugo = _polys(d["yugo"])
    D.main = _arr(d["outline"]["main"])
    D.main_cum = _cum(D.main)
    D.islands = [_arr(r) for r in d["outline"]["islands"]]
    att = []  # where along the main ring each island "attaches" (draw-on reveal)
    for isl in D.islands:
        c = isl.mean(axis=0)
        k = int(np.argmin(((D.main - c) ** 2).sum(1)))
        att.append(D.main_cum[k] / D.main_cum[-1])
    D.island_at = np.array(att)
    D.coarse = _polys(d["coarse"]["yugo"])
    D.coarse_borders = [_arr(ln) for ln in d["coarse"]["rep_borders"]]
    D.nbrs = [_NS(id=n["id"], name=n["name"], en=n["en"], polys=_polys(n["p"]),
                  label=np.array(n["label"], float)) for n in d["neighbors"]]
    D.context_flat = [p for c in d["context"] for p in _polys(c["p"])]
    D.lakes = _polys(d["lakes"])
    D.rivers = [_arr(r["l"]) for r in d["rivers"]]
    D.cities = {c["name"]: _NS(name=c["name"], today=c["today"], lon=c["lon"], lat=c["lat"],
                               kind=c["kind"], xy=np.array(c["xy"], float)) for c in d["cities"]}
    D.bounds = tuple(float(v) for v in d["meta"]["bounds"])
    return D


def _yu_lonlat(xy):
    """Stored LCC units -> lon, lat (HUD read-outs)."""
    p = dict(_yu().meta["proj"])
    p.pop("type")
    s = _yu().meta["scale"]
    xy = np.asarray(xy, float)
    return geoproj.lcc_inv(xy[..., 0] / s, xy[..., 1] / s, **p)


# ================================================================== view ====
class _View:
    """Stored units -> pixels: p_px = (p - c) * ppu (y flipped) + screen centre + offset."""

    def __init__(self, W, H, cx, cy, ppu, ox=0.0, oy=0.0):
        self.W, self.H = W, H
        self.cx, self.cy, self.ppu = float(cx), float(cy), float(ppu)
        self.ox, self.oy = W * 0.5 + ox, H * 0.5 + oy
        self.vb = (cx - self.ox / ppu, cy - (H - self.oy) / ppu,
                   cx + (W - self.ox) / ppu, cy + self.oy / ppu)

    def key(self):
        return (self.W, self.H, round(self.cx, 2), round(self.cy, 2), round(self.ppu, 7),
                round(self.ox, 2), round(self.oy, 2))

    def px(self, a, ss=1.0):
        a = np.asarray(a, dtype=np.float64)
        out = np.empty(a.shape, dtype=np.float64)
        out[..., 0] = ((a[..., 0] - self.cx) * self.ppu + self.ox) * ss
        out[..., 1] = ((self.cy - a[..., 1]) * self.ppu + self.oy) * ss
        return out

    def visible(self, bb, pad_px=6):
        pad = pad_px / self.ppu
        v = self.vb
        return not (bb[2] < v[0] - pad or bb[0] > v[2] + pad or bb[3] < v[1] - pad or bb[1] > v[3] + pad)

    def bbpx(self, bb):
        a = self.px(np.array([[bb[0], bb[1]], [bb[2], bb[3]]]))
        return (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())


def _make_view(W, H, bounds, fit=(0.86, 0.84), scale=1.0, center=None, offset=(0.0, 0.0), meta=None):
    x0, y0, x1, y1 = bounds
    ppu = min(W * fit[0] / (x1 - x0), H * fit[1] / (y1 - y0)) * scale
    if center is None:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        cx, cy = geoproj.project(meta, center[0], center[1])
        cx, cy = float(cx), float(cy)
    return _View(W, H, cx, cy, ppu, offset[0] * W, offset[1] * H)


# ============================================================== raster =====
def _lw(w, ss):
    return max(1, int(round(w * ss)))


def _bbp(arrs, pad=0.0):
    """Union bbox (px) of a list of (N,2) arrays."""
    arrs = [a for a in arrs if len(a)]
    if not arrs:
        return (0, 0, 0, 0)
    x0 = min(float(a[:, 0].min()) for a in arrs) - pad
    y0 = min(float(a[:, 1].min()) for a in arrs) - pad
    x1 = max(float(a[:, 0].max()) for a in arrs) + pad
    y1 = max(float(a[:, 1].max()) for a in arrs) + pad
    return (x0, y0, x1, y1)


def _partial(pts, cum, s):
    """Pixel polyline cut at arc length s."""
    if s <= 0:
        return pts[:1]
    if s >= cum[-1]:
        return pts
    k = int(np.searchsorted(cum, s))
    f = (s - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-9)
    tip = pts[k - 1] + (pts[k] - pts[k - 1]) * f
    return np.vstack([pts[:k], tip[None]])


def _subpath(pts, cum, a, b):
    """Pixel polyline between arc lengths a < b."""
    a, b = max(a, 0.0), min(b, cum[-1])
    if b <= a:
        return pts[:0]
    i0 = int(np.searchsorted(cum, a))
    i1 = int(np.searchsorted(cum, b))
    pa = (np.interp(a, cum, pts[:, 0]), np.interp(a, cum, pts[:, 1]))
    pb = (np.interp(b, cum, pts[:, 0]), np.interp(b, cum, pts[:, 1]))
    return np.vstack([[pa], pts[i0:i1], [pb]])


def _dashes(pts, on, off):
    """Split a pixel polyline into dash sub-polylines."""
    cum = _cum(pts)
    out = []
    s = 0.0
    while s < cum[-1]:
        seg = _subpath(pts, cum, s, s + on)
        if len(seg) >= 2:
            out.append(seg)
        s += on + off
    return out


class _SS:
    """Supersampled L mask limited to a pixel bbox (whole frame if bbox is None)."""

    def __init__(self, W, H, ss=2, bbox=None, pad=8):
        if bbox is None:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0 = max(0, int(math.floor(bbox[0] - pad)))
            y0 = max(0, int(math.floor(bbox[1] - pad)))
            x1 = min(W, int(math.ceil(bbox[2] + pad)))
            y1 = min(H, int(math.ceil(bbox[3] + pad)))
            if x1 <= x0 or y1 <= y0:
                x0, y0, x1, y1 = 0, 0, 1, 1
        self.W, self.H, self.ss, self.full = W, H, ss, bbox is None
        self.x0, self.y0 = x0, y0
        self.im = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
        self.dr = ImageDraw.Draw(self.im)

    def P(self, pts):
        a = np.asarray(pts, dtype=np.float64)
        if self.x0 or self.y0:
            a = a - (self.x0, self.y0)
        return (a * self.ss).ravel().tolist()

    def line(self, pts, width, fill=255):
        if len(pts) >= 2:
            w = _lw(width, self.ss)
            self.dr.line(self.P(pts), fill=int(fill), width=w, joint="curve" if w > 2 else None)

    def poly(self, pts, fill=255):
        if len(pts) >= 3:
            self.dr.polygon(self.P(pts), fill=int(fill))

    def circle(self, x, y, r, fill=None, outline=None, width=1.0):
        s = self.ss
        X, Y, R = (x - self.x0) * s, (y - self.y0) * s, r * s
        if outline is None:
            self.dr.ellipse((X - R, Y - R, X + R, Y + R), fill=fill)
        else:
            self.dr.ellipse((X - R, Y - R, X + R, Y + R), fill=fill, outline=int(outline), width=_lw(width, s))

    def mask(self):
        return self.im.reduce(self.ss) if self.ss > 1 else self.im

    def paint(self, cv, col, op=1.0):
        cv.paint(self.mask(), col, op, (self.x0, self.y0))


def _ss_polys(m, polys, view, fill=255, holes=0):
    for rings, bb in polys:
        if view.visible(bb):
            m.poly(view.px(rings[0]), fill)
            for h in rings[1:]:
                m.poly(view.px(h), holes)


def _ss_outlines(m, polys, view, width, fill=255, holes=True):
    for rings, bb in polys:
        if view.visible(bb):
            for r in (rings if holes else rings[:1]):
                m.line(view.px(r), width, fill)


def _ss_lines(m, lines, view, width, fill=255):
    for a in lines:
        m.line(view.px(a), width, fill)


@functools.lru_cache(maxsize=512)
def _lut_scale(k255):
    k = k255 / 255.0
    return [int(round(v * k)) for v in range(256)]


def _scale_mask(m, k):
    if k >= 0.998:
        return m
    return m.point(_lut_scale(int(round(_cl(k) * 255))))


class _Canvas:
    """Premultiplied 8-bit compositor: C (RGB, premultiplied colour) + A (L alpha)."""

    def __init__(self, W, H, bg=None, base=None, op=1.0):
        self.W, self.H = W, H
        if base is not None:
            if op >= 0.998:
                self.C, self.A = base[0].copy(), base[1].copy()
            else:
                lut = _lut_scale(int(round(_cl(op) * 255)))
                self.C, self.A = base[0].point(lut * 3), base[1].point(lut)
            if bg is not None:  # base is premultiplied -> composite over the bg colour
                inv = ImageChops.invert(self.A)
                bgc = ImageChops.multiply(Image.new("RGB", (W, H), _u8(bg)), Image.merge("RGB", (inv, inv, inv)))
                self.C = ImageChops.add(self.C, bgc)
                self.A = Image.new("L", (W, H), 255)
        elif bg is None:
            self.C = Image.new("RGB", (W, H), (0, 0, 0))
            self.A = Image.new("L", (W, H), 0)
        else:
            self.C = Image.new("RGB", (W, H), _u8(bg))
            self.A = Image.new("L", (W, H), 255)

    def snapshot(self):
        return (self.C.copy(), self.A.copy())

    def paint(self, mask, col, op=1.0, xy=(0, 0)):
        """Solid colour through an L mask (mask may be a sprite placed at xy)."""
        if mask is None or op <= 0.003:
            return
        m = _scale_mask(mask, op)
        bb = m.getbbox()
        if bb is None:
            return
        x, y = int(round(xy[0])), int(round(xy[1]))
        cx0, cy0 = max(bb[0] + x, 0), max(bb[1] + y, 0)
        cx1, cy1 = min(bb[2] + x, self.W), min(bb[3] + y, self.H)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        m = m.crop((cx0 - x, cy0 - y, cx1 - x, cy1 - y))
        c8 = _u8(col)
        w, h = m.size
        if w * h < 160000:
            box = (cx0, cy0, cx1, cy1)
            self.C.paste(c8, box, m)
            self.A.paste(255, box, m)
            return
        T = 256  # large masks: paste tile by tile, skipping empty tiles (sparse lines/rings)
        for ty in range(0, h, T):
            for tx in range(0, w, T):
                sub = m.crop((tx, ty, min(tx + T, w), min(ty + T, h)))
                sb = sub.getbbox()
                if sb is None:
                    continue
                if sb != (0, 0) + sub.size:
                    sub = sub.crop(sb)
                box = (cx0 + tx + sb[0], cy0 + ty + sb[1], cx0 + tx + sb[2], cy0 + ty + sb[3])
                self.C.paste(c8, box, sub)
                self.A.paste(255, box, sub)

    def paste_rgb(self, rgb, mask, xy=(0, 0)):
        """An RGB sprite (straight colour) through an L mask."""
        x, y = int(round(xy[0])), int(round(xy[1]))
        self.C.paste(rgb, (x, y), mask)
        self.A.paste(255, (x, y, x + mask.size[0], y + mask.size[1]), mask)

    def add(self, rgb, a, box=None):
        """Additive (premultiplied) light such as glows; rgb / a sized to box."""
        if box is None:
            self.C = ImageChops.add(self.C, rgb)
            self.A = ImageChops.screen(self.A, a)
        else:
            self.C.paste(ImageChops.add(self.C.crop(box), rgb), box[:2])
            self.A.paste(ImageChops.screen(self.A.crop(box), a), box[:2])

    def array(self, scan=False):
        im = Image.merge("RGBa", (*self.C.split(), self.A)).convert("RGBA")
        arr = np.asarray(im)
        out = np.empty(arr.shape, np.float32)
        np.multiply(arr, np.float32(1.0 / 255.0), out=out)
        if scan:
            out[1::3, :, 3] *= np.float32(0.55)
            out[2::3, :, 3] *= np.float32(0.85)
        return out


@functools.lru_cache(maxsize=2048)
def _lut_gain_q(kq):
    k = kq / 64.0
    return [min(255, int(v * k + 0.5)) for v in range(256)]


def _lut_gain(k):
    return _lut_gain_q(int(round(max(k, 0.0) * 64)))


class _Glow:
    """Quarter-resolution glow accumulator.  Draw into ``new()`` masks with
    coordinates / ds, ``put`` them (blur + colour), then ``apply`` additively."""

    def __init__(self, W, H, ds=4):
        self.W, self.H, self.ds = W, H, ds
        self.w, self.h = -(-W // ds), -(-H // ds)
        self.rgb = None
        self.a = None

    def new(self):
        im = Image.new("L", (self.w, self.h), 0)
        return im, ImageDraw.Draw(im)

    def put(self, m, col, radius_px, strength=1.0):
        if m is None or strength <= 0.003 or m.getbbox() is None:
            return
        b = m.filter(ImageFilter.GaussianBlur(max(radius_px / self.ds, 0.6)))
        rgb = Image.merge("RGB", [b.point(_lut_gain(c * strength)) for c in col])
        a = b.point(_lut_gain(strength))
        if self.rgb is None:
            self.rgb, self.a = rgb, a
        else:
            self.rgb = ImageChops.add(self.rgb, rgb)
            self.a = ImageChops.screen(self.a, a)

    def sprite(self, m_full, x, y):
        """Downsample a full-res sprite mask into a low-res mask at (x, y)."""
        im, _ = self.new()
        w, h = m_full.size
        sw, sh = max(1, int(round(w / self.ds))), max(1, int(round(h / self.ds)))
        im.paste(m_full.resize((sw, sh), Image.BILINEAR), (int(round(x / self.ds)), int(round(y / self.ds))))
        return im

    def lines(self, arrs, width_px=4.0, fill=255):
        im, dr = self.new()
        w = max(1, int(round(width_px / self.ds)))
        for a in arrs:
            if len(a) >= 2:
                dr.line((np.asarray(a) / self.ds).ravel().tolist(), fill=int(fill), width=w)
        return im

    def polys(self, arrs, fill=255):
        im, dr = self.new()
        for a in arrs:
            if len(a) >= 3:
                dr.polygon((np.asarray(a) / self.ds).ravel().tolist(), fill=int(fill))
        return im

    def dot(self, x, y, r_px, fill=255, im=None):
        if im is None:
            im, _ = self.new()
        dr = ImageDraw.Draw(im)
        r = max(r_px / self.ds, 0.8)
        dr.ellipse((x / self.ds - r, y / self.ds - r, x / self.ds + r, y / self.ds + r), fill=int(fill))
        return im

    def apply(self, cv: _Canvas):
        if self.rgb is None:
            return
        bb = self.a.getbbox()
        if bb is None:
            return
        ds = self.ds
        bx0, by0 = max(bb[0] - 1, 0), max(bb[1] - 1, 0)
        bx1, by1 = min(bb[2] + 1, self.w), min(bb[3] + 1, self.h)
        X0, Y0 = bx0 * ds, by0 * ds
        X1, Y1 = min(bx1 * ds, self.W), min(by1 * ds, self.H)
        if X1 <= X0 or Y1 <= Y0:
            return
        src = (bx0, by0, bx0 + (X1 - X0) / ds, by0 + (Y1 - Y0) / ds)
        size = (X1 - X0, Y1 - Y0)
        cv.add(self.rgb.resize(size, Image.BILINEAR, box=src),
               self.a.resize(size, Image.BILINEAR, box=src), (X0, Y0, X1, Y1))


# ================================================================ text =====
@functools.lru_cache(maxsize=4096)
def _tmask(txt, fname, size, track=0.0):
    """Render text -> (L mask, ox, oy, advance, cap_height); (ox, oy) = mask top-left
    relative to the left end of the baseline."""
    f = font(fname, max(6, int(round(size))))
    cap = -f.getbbox("H", anchor="ls")[1]
    if not txt:
        return Image.new("L", (1, 1), 0), 0, 0, 0.0, cap
    if track == 0:
        l, t, r, b = f.getbbox(txt, anchor="ls")
        adv = f.getlength(txt)
        im = Image.new("L", (max(1, r - l + 2), max(1, b - t + 2)), 0)
        ImageDraw.Draw(im).text((-l + 1, -t + 1), txt, font=f, fill=255, anchor="ls")
        return im, l - 1, t - 1, adv, cap
    tr = track * size
    xs, x = [], 0.0
    for ch in txt:
        xs.append(x)
        x += f.getlength(ch) + tr
    adv = x - tr
    l, t, r, b = f.getbbox(txt, anchor="ls")
    pad = int(size * 0.3) + 2
    im = Image.new("L", (int(adv + 2 * pad + size), int(b - t + 2 * pad)), 0)
    dr = ImageDraw.Draw(im)
    for ch, xx in zip(txt, xs):
        dr.text((pad + xx, pad - t), ch, font=f, fill=255, anchor="ls")
    bb = im.getbbox() or (0, 0, 1, 1)
    im = im.crop(bb)
    return im, bb[0] - pad, bb[1] - pad + t, adv, cap


def _tadv(txt, fname, size, track=0.0):
    return _tmask(txt, fname, int(round(size)), float(track))[3]


@functools.lru_cache(maxsize=512)
def _shadow_mask(txt, fname, size, track, r):
    m = _tmask(txt, fname, size, track)[0]
    pad = int(r * 3) + 2
    big = Image.new("L", (m.size[0] + 2 * pad, m.size[1] + 2 * pad), 0)
    big.paste(m, (pad, pad))
    big = big.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(r))
    return big.point(_lut_gain(2.2)), pad


def _text(cv, txt, fname, size, x, y, col, op=1.0, align="l", valign="base", track=0.0,
          reveal=1.0, glow=None, glow_col=None, glow_r=10.0, glow_k=0.8, shadow=0.0):
    """Draw text anchored at (x, y); align l/m/r; valign base/cap/mid.  reveal<1 wipes
    it in from the left; shadow>0 lays a soft dark halo underneath (legibility over
    bright fills).  Returns the pixel bbox."""
    if not txt or op <= 0.003 or reveal <= 0:
        return (x, y, x, y)
    m, ox, oy, adv, cap = _tmask(txt, fname, int(round(size)), float(track))
    if align == "m":
        x -= adv / 2
    elif align == "r":
        x -= adv
    if valign == "cap":
        y += cap
    elif valign == "mid":
        y += cap / 2
    px, py = x + ox, y + oy
    if shadow > 0:
        sm, pad = _shadow_mask(txt, fname, int(round(size)), float(track), max(1.5, size * 0.12))
        if reveal < 1.0:
            sm = sm.crop((0, 0, max(1, int(pad + m.size[0] * _cl(reveal) + pad * 0.5)), sm.size[1]))
        cv.paint(sm, BLACK, shadow * op, (px - pad, py - pad))
    if reveal < 1.0:
        m = m.crop((0, 0, max(1, int(m.size[0] * _cl(reveal))), m.size[1]))
    cv.paint(m, col, op, (px, py))
    if glow is not None:
        glow.put(glow.sprite(m, px, py), glow_col or col, glow_r, glow_k * op)
    return (px, py, px + m.size[0], py + m.size[1])


@functools.lru_cache(maxsize=64)
def _rect_mask(w, h):
    return Image.new("L", (max(1, w), max(1, h)), 255)


@functools.lru_cache(maxsize=64)
def _dot_mask(r10):
    r = r10 / 10.0
    s = int(math.ceil(2 * r + 3))
    m = _SS(s, s, 4)
    m.circle(s / 2, s / 2, r, fill=255)
    return m.mask()


def _dot(cv, x, y, r, col, op=1.0):
    m = _dot_mask(int(round(r * 10)))
    cv.paint(m, col, op, (x - m.size[0] / 2, y - m.size[1] / 2))


@functools.lru_cache(maxsize=64)
def _star_mask(r10):
    r = r10 / 10.0
    s = int(math.ceil(2.2 * r + 4))
    m = _SS(s, s, 4)
    m.poly(np.array(star_points(s / 2, s / 2 + r * 0.06, r)), 255)
    return m.mask()


def _star(cv, x, y, r, col, op=1.0):
    if r < 0.5:
        return
    m = _star_mask(int(round(r * 10)))
    cv.paint(m, col, op, (x - m.size[0] / 2, y - m.size[1] / 2))


# ================================================================= HUD =====
_CACHE: dict = {}


def _cache_get(key, fn, limit=40):
    v = _CACHE.get(key)
    if v is None:
        while len(_CACHE) >= limit:
            _CACHE.pop(next(iter(_CACHE)))
        v = fn()
        _CACHE[key] = v
    return v


def _grat_lines(meta, lons, lats, lon_rng, lat_rng, step=0.25):
    """Graticule polylines in stored units: [(kind, value, array)]."""
    out = []
    la = np.arange(lat_rng[0], lat_rng[1] + 1e-9, step)
    for lo in lons:
        x, y = geoproj.project(meta, np.full_like(la, lo), la)
        out.append(("lon", lo, np.column_stack([x, y])))
    lo_ = np.arange(lon_rng[0], lon_rng[1] + 1e-9, step)
    for lt in lats:
        x, y = geoproj.project(meta, lo_, np.full_like(lo_, lt))
        out.append(("lat", lt, np.column_stack([x, y])))
    return out


def _deg(v, kind):
    v = round(v, 6)
    if v == 0 or abs(v) == 180:
        return f"{abs(v):g}°"
    if kind == "lat":
        return f"{abs(v):g}°{'N' if v >= 0 else 'S'}"
    return f"{abs(v):g}°{'E' if v >= 0 else 'W'}"


def _edge_hits(pts, W, H, m):
    """Crossings of a pixel polyline with the bottom / left inset frame edges."""
    hits = []
    x, y = pts[:, 0], pts[:, 1]
    s = np.sign(y - (H - m))
    for i in np.nonzero(np.diff(s) != 0)[0]:
        f = ((H - m) - y[i]) / (y[i + 1] - y[i])
        xx = x[i] + (x[i + 1] - x[i]) * f
        if m + 60 < xx < W - m - 160:
            hits.append(("b", xx))
    s = np.sign(x - m)
    for i in np.nonzero(np.diff(s) != 0)[0]:
        f = (m - x[i]) / (x[i + 1] - x[i])
        yy = y[i] + (y[i + 1] - y[i]) * f
        if m + 80 < yy < H - m - 40:
            hits.append(("l", yy))
    return hits


def _hud_masks(view, grat, title, sub, km_unit=None, grid=True, hud=True, globe=None, gkey=""):
    """(grid_mask, hud_mask) for a view, cached."""
    key = ("hud", view.key(), title, sub, km_unit, grid, hud, gkey)

    def build():
        W, H = view.W, view.H
        u = H / 1080.0
        m = int(round(28 * u))
        g = _SS(W, H, 2) if grid else None
        hm = Image.new("L", (W, H), 0)
        hd = ImageDraw.Draw(hm)
        hits = []
        for kind, val, a in grat:
            p = view.px(a)
            if g is not None:
                g.line(p, 1.0 * u)
            for e, pos in _edge_hits(p, W, H, m):
                hits.append((kind, val, e, pos))
        if globe is not None and g is not None:
            g.line(view.px(globe), 1.3 * u)
        gm = g.mask() if g is not None else None
        if not hud:
            return gm, None
        L = int(round(26 * u))
        lw = max(1, int(round(1.5 * u)))
        for (cx, cy, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
            hd.line([(cx, cy + sy * L), (cx, cy), (cx + sx * L, cy)], fill=255, width=lw)
        f = font("mono", max(9, int(round(13 * u))))
        for kind, val, e, pos in hits:
            if (kind == "lon" and e == "b") or (kind == "lat" and e == "l"):
                t = _deg(val, kind)
                if e == "b":
                    hd.line([(pos, H - m), (pos, H - m - 8 * u)], fill=255, width=lw)
                    hd.text((pos + 4 * u, H - m - 2 * u), t, font=f, fill=190, anchor="ls")
                else:
                    hd.line([(m, pos), (m + 8 * u, pos)], fill=255, width=lw)
                    hd.text((m + 11 * u, pos - 3 * u), t, font=f, fill=190, anchor="ls")
        if title:
            hd.text((m + 36 * u, m + 2 * u), title, font=font("mono", max(9, int(round(15 * u)))), fill=235, anchor="lt")
        if sub:
            hd.text((m + 36 * u, m + 23 * u), sub, font=font("mono", max(8, int(round(12 * u)))), fill=140, anchor="lt")
        if km_unit:
            km, units_per_km = km_unit
            bl = km * units_per_km * view.ppu
            x1, y1 = W - m - 36 * u, H - m - 4 * u
            x0 = x1 - bl
            fs = font("mono", max(8, int(round(12 * u))))
            hd.line([(x0, y1 - 6 * u), (x0, y1), (x1, y1), (x1, y1 - 6 * u)], fill=235, width=lw)
            hd.line([((x0 + x1) / 2, y1), ((x0 + x1) / 2, y1 - 4 * u)], fill=235, width=lw)
            hd.text((x1, y1 - 10 * u), f"{km} KM", font=fs, fill=200, anchor="rs")
            hd.text((x0, y1 - 10 * u), "0", font=fs, fill=200, anchor="ls")
        return gm, hm

    return _cache_get(key, build)


def _hud(cv, view, grat, title, sub, km_unit=None, grid=True, hud=True, op=1.0, globe=None, gkey="",
         grid_col=STEEL, grid_op=0.16, hud_col=WHITE, hud_op=0.55):
    if not (grid or hud) or op <= 0:
        return
    gm, hm = _hud_masks(view, grat, title, sub, km_unit, grid, hud, globe, gkey)
    if gm is not None:
        cv.paint(gm, grid_col, grid_op * op)
    if hm is not None:
        cv.paint(hm, hud_col, hud_op * op)


# ========================================================= yugo: common ====
@functools.lru_cache(None)
def _yu_grat():
    return _grat_lines(_yu().meta, list(range(6, 33)), list(range(34, 54)), (4, 34), (33, 54))


def _cmin(lon, lat):
    return f"{abs(lat):05.2f}°{'N' if lat >= 0 else 'S'}  {abs(lon):05.2f}°{'E' if lon >= 0 else 'W'}"


def _yu_hud(cv, view, grid, hud, op=1.0, title="SFRJ // 1945–1992", sub=None):
    if sub is None:
        lon, lat = _yu_lonlat(np.array([view.cx, view.cy]))
        sub = f"LAMBERT CONFORMAL CONIC  ·  {_cmin(float(lon), float(lat))}"
    _hud(cv, view, _yu_grat(), title, sub, (100, 10.0), grid, hud, op, gkey="yu")


def _yu_mask(view, which, width=1.0, ss=2):
    """Cached full-frame masks for Yugoslav geometry.
    which: yugo_fill | yugo_line | borders | prov | repN_fill | repN_line |
           nbI,J,.._fill | nbI,J,.._line | ctx_fill | ctx_line | rivers | lakes"""
    key = ("yum", view.key(), which, width, ss)

    def build():
        D = _yu()
        W, H = view.W, view.H
        u = H / 1080.0
        m = _SS(W, H, ss)
        if which == "yugo_fill":
            _ss_polys(m, D.yugo, view)
            _ss_polys(m, D.lakes, view, 0)
        elif which == "yugo_line":
            _ss_outlines(m, D.yugo, view, width * u)
        elif which == "borders":
            _ss_lines(m, D.rep_borders, view, width * u)
        elif which == "prov":
            for ln in D.prov_borders:
                for d in _dashes(view.px(ln), 7 * u, 5 * u):
                    m.line(d, width * u)
        elif which.startswith("rep"):
            i = int(which[3:].split("_")[0])
            if which.endswith("fill"):
                _ss_polys(m, D.reps[i].polys, view)
                _ss_polys(m, D.lakes, view, 0)
            else:
                _ss_outlines(m, D.reps[i].polys, view, width * u)
        elif which.startswith("nb") and "," in which:  # union of cached single masks
            ids, suf = which[2:].split("_")
            ids = ids.split(",")
            acc = _yu_mask(view, f"nb{ids[0]}_{suf}", width, ss)
            for i in ids[1:]:
                acc = ImageChops.lighter(acc, _yu_mask(view, f"nb{i}_{suf}", width, ss))
            return acc
        elif which.startswith("nb"):
            ids = [int(v) for v in which[2:].split("_")[0].split(",") if v != ""]
            for i in ids:
                if which.endswith("fill"):
                    _ss_polys(m, D.nbrs[i].polys, view)
                else:
                    _ss_outlines(m, D.nbrs[i].polys, view, width * u)
        elif which == "ctx_fill":
            _ss_polys(m, D.context_flat + [p for n in D.nbrs for p in n.polys], view)
        elif which == "ctx_line":
            _ss_outlines(m, D.context_flat + [p for n in D.nbrs for p in n.polys], view, width * u)
        elif which == "rivers":
            _ss_lines(m, D.rivers, view, width * u)
        elif which == "lakes":
            _ss_outlines(m, D.lakes, view, width * u)
        else:
            raise KeyError(which)
        return m.mask()

    return _cache_get(key, build, limit=120)


def _yu_view(W, H, scale=1.0, center=None, offset=(0.0, 0.0), fit=(0.86, 0.84)):
    D = _yu()
    return _make_view(W, H, D.bounds, fit, scale, center, offset, D.meta)


def _yu_underlay(W, H, view, kw, op=1.0, ctx_op=1.0):
    """Fresh canvas with the cached context land + graticule + HUD (faded by op)."""
    key = ("under", view.key(), kw.get("grid", True), kw.get("hud", True), kw.get("rivers", False), ctx_op)

    def build():
        cv = _Canvas(W, H)
        cv.paint(_yu_mask(view, "ctx_fill", ss=1), STEEL, 0.045 * ctx_op)
        cv.paint(_yu_mask(view, "ctx_line", 1.0), STEEL, 0.30 * ctx_op)
        if kw.get("rivers", False):
            cv.paint(_yu_mask(view, "rivers", 0.9), ICE, 0.22 * ctx_op)
        _yu_hud(cv, view, kw.get("grid", True), kw.get("hud", True))
        return cv.snapshot()

    return _Canvas(W, H, _col(kw.get("bg")), base=_cache_get(key, build), op=op)


def _yu_mapbb(view):
    return view.bbpx(_yu().bounds)


# ============================================================ yugo modes ====
def _mode_draw(t, dur, W, H, p, view, kw):
    """Glowing outline draws itself on; spark at the head; internal borders follow."""
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    end = float(kw.get("reveal_end", 0.62))
    lw = float(kw.get("width", 1.8)) * u
    cv = _yu_underlay(W, H, view, kw, _eo3(_seg(p, 0.0, 0.25)))
    glow = _Glow(W, H)
    mbb = _yu_mapbb(view)

    q = _eio(_seg(p, 0.02, end))
    done = p >= end
    fa = float(kw.get("fill_alpha", 0.2)) * _eo3(_seg(p, end, end + 0.25))
    if fa > 0:
        cv.paint(_yu_mask(view, "yugo_fill"), _mix(fillc, DEEP, 0.4), fa)

    pts = view.px(D.main)
    cum = D.main_cum * view.ppu
    s = q * cum[-1]
    part = _partial(pts, cum, s)
    sm = _SS(W, H, 2, mbb)
    sm.line(part, lw)
    isl_on = []
    for isl, at in zip(D.islands, D.island_at):
        a = _seg(q, at, at + 0.03)
        if a > 0:
            ip = view.px(isl)
            sm.line(ip, lw * 0.8, 255 * a)
            isl_on.append((ip, a))
    sm.paint(cv, _hot(stroke, 0.35), 1.0)

    boost = 1.0 + 1.6 * _pulse(p - end, 0.05) if done else 1.0
    lg = glow.lines([part], lw * 1.3)
    ld = ImageDraw.Draw(lg)
    for ip, a in isl_on:
        ld.line((ip / glow.ds).ravel().tolist(), fill=int(255 * a), width=1)
    glow.put(lg, stroke, 5 * u, 1.3 * gk * boost)
    glow.put(lg, stroke, 22 * u, 0.9 * gk * boost)

    qb = _eo3(_seg(p, end - 0.02, end + 0.22))
    if qb > 0:
        bm = _SS(W, H, 2, mbb)
        for ln in D.rep_borders:
            lp = view.px(ln)
            c = _cum(lp)
            bm.line(_partial(lp, c, qb * c[-1]), 1.1 * u)
        bm.paint(cv, stroke, 0.85)
    qp = _seg(p, end + 0.1, end + 0.3)
    if qp > 0:
        cv.paint(_yu_mask(view, "prov", 1.0), stroke, 0.7 * qp)

    if 0.0 < q < 1.0 or (done and p < end + 0.06):
        hx, hy = part[-1]
        sp_op = 1.0 if not done else 1.0 - _seg(p, end, end + 0.06)
        sg = glow.dot(hx, hy, 5 * u)
        glow.put(sg, _hot(stroke, 0.6), 8 * u, 2.6 * sp_op)
        glow.put(sg, stroke, 36 * u, 1.8 * sp_op)
        fl = 90 * u
        fm = _SS(W, H, 2, (hx - fl, hy - 20 * u, hx + fl, hy + 20 * u))
        for j in range(12):  # anamorphic streak, fading outwards
            a0_, a1_ = j / 12, (j + 1) / 12
            v = 255 * (1 - a0_) ** 2.2
            for sgn in (-1, 1):
                fm.line(np.array([[hx + sgn * a0_ * fl, hy], [hx + sgn * a1_ * fl, hy]]), 1.1 * u, v)
        fm.paint(cv, _hot(stroke, 0.55), 0.8 * sp_op)
        cr = _SS(W, H, 2, (hx - 20 * u, hy - 20 * u, hx + 20 * u, hy + 20 * u))
        cr.circle(hx, hy, 2.6 * u, fill=255)
        cr.line(np.array([[hx, hy - 14 * u], [hx, hy + 14 * u]]), 0.8 * u, 150)
        cr.paint(cv, (1, 1, 1), sp_op)
        if kw.get("labels", True) and kw.get("hud", True):
            lon, lat = _yu_lonlat(D.main[min(int(np.searchsorted(cum, s)), len(D.main) - 1)])
            _text(cv, _cmin(float(lon), float(lat)), "mono", 13 * u, hx + 16 * u, hy - 12 * u,
                  WHITE, 0.85 * sp_op, track=0.06)
    glow.apply(cv)
    return cv


def _slot(p, n, a=0.0, b=1.0):
    x = _seg(p, a, b) * n
    i = min(int(x), n - 1)
    return i, x - i


def _big_name(cv, glow, W, H, name, en, idx_txt, local, kw, col=WHITE, accent=RED):
    """Big condensed name block, lower-left, wiped in; returns its top-right anchor."""
    u = H / 1080.0
    m = 72 * u
    maxw = W * float(kw.get("name_width", 0.33))
    size = float(kw.get("name_size", 150)) * u
    fn = kw.get("name_font", "anton")
    lines = [name]
    if _tadv(name, fn, size) > maxw and " I " in name:
        a, b = name.split(" I ", 1)
        lines = [a + " I", b]
    adv = max(_tadv(ln, fn, size) for ln in lines)
    if adv > maxw:
        size *= maxw / adv
        adv = maxw
    rv = _eo5(_seg(local, 0.0, 0.18))
    lh = size * 1.0
    yb = H - m - 58 * u
    ys = [yb - lh * (len(lines) - 1 - i) for i in range(len(lines))]
    top = ys[0] - _tmask(lines[0], fn, int(round(size)))[4]
    slide = (1 - rv) * 40 * u
    for ln, y in zip(lines, ys):
        _text(cv, ln, fn, size, m - slide, y, col, 1.0, reveal=rv, glow=glow, glow_col=accent,
              glow_r=18 * u, glow_k=0.35)
    rc = _seg(local, 0.08, 0.3)
    _text(cv, en, "mono", 21 * u, m, yb + 40 * u, WHITE, 0.8 * rc, track=0.18, reveal=_eo3(rc * 1.4))
    if idx_txt:
        _text(cv, idx_txt, "mono", 19 * u, m, top - 22 * u, accent, rc, track=0.2)
        cv.paint(_rect_mask(int(max(2, 50 * u * _eo3(rc))), max(1, int(2 * u))), accent, 1.0,
                 (m + 105 * u, top - 29 * u))
    return (m + adv * rv, top)


def _rep_base(W, H, view, kw, stroke, fillc, gk):
    key = ("rep_base", view.key(), kw.get("grid", True), kw.get("hud", True), kw.get("rivers", False),
           stroke, fillc, gk)

    def build():
        u = H / 1080.0
        cv = _yu_underlay(W, H, view, dict(kw, bg=None))
        cv.paint(_yu_mask(view, "yugo_fill"), _mix(fillc, DEEP, 0.6), 0.30)
        cv.paint(_yu_mask(view, "borders", 1.0), stroke, 0.5)
        cv.paint(_yu_mask(view, "prov", 0.9), stroke, 0.3)
        cv.paint(_yu_mask(view, "yugo_line", 1.0), stroke, 0.75)
        g = _Glow(W, H)
        g.put(g.lines([view.px(r) for rs, bb in _yu().yugo for r in rs[:1]], 2 * u), stroke, 18 * u, 0.35 * gk)
        g.apply(cv)
        return cv.snapshot()

    return _Canvas(W, H, _col(kw.get("bg")), base=_cache_get(key, build))


def _mode_republics(t, dur, W, H, p, view, kw):
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    hi = kw.get("hi", None)
    seq = kw.get("sequence", hi is None)
    local = p
    if hi is None and seq:
        hi, local = _slot(p, 6)
    elif hi is not None and kw.get("local") is not None:
        local = float(kw["local"])
    labels = kw.get("labels", True)
    cv = _rep_base(W, H, view, kw, stroke, fillc, gk)
    if hi is None:
        return cv
    hi = int(hi) % 6
    rep = D.reps[hi]
    glow = _Glow(W, H)
    fl = _pulse(local, 0.05)
    cv.paint(_yu_mask(view, f"rep{hi}_fill"), _mix(fillc, (1, 1, 1), 0.85 * fl), 0.93)
    if rep.id == "SRB":
        cv.paint(_yu_mask(view, "prov", 1.0), DEEP, 0.8)
    cv.paint(_yu_mask(view, f"rep{hi}_line", 2.0), _hot(stroke, 0.5), 1.0)
    rings = [view.px(r) for rs, bb in rep.polys for r in rs[:1]]
    glow.put(glow.lines(rings, 3 * u), stroke, 6 * u, 1.2 * gk * (1 + fl))
    glow.put(glow.polys(rings), stroke, 30 * u, 0.55 * gk * (1 + 1.5 * fl))
    if labels:
        cap = D.cities[rep.capital]
        cx, cy = view.px(cap.xy)
        rc = _seg(local, 0.1, 0.3)
        _dot(cv, cx, cy, 4.5 * u, WHITE, rc)
        _text(cv, cap.name.upper(), "mono", 16 * u, cx + 12 * u, cy - 8 * u, WHITE, rc, track=0.12)
        ax, ay = _big_name(cv, glow, W, H, rep.name, rep.en, f"{hi + 1:02d} / 06", local, kw)
        lx, ly = view.px(rep.label)
        k = _eo3(_seg(local, 0.12, 0.35))
        if k > 0 and kw.get("leader", True):
            x0, y0 = ax + 26 * u, ay + 12 * u
            path = np.array([[x0, y0], [x0 + (lx - x0) * 0.45, y0], [lx, ly]])
            c = _cum(path)
            lm = _SS(W, H, 2, _bbp([path], 8 * u))
            lm.line(_partial(path, c, c[-1] * k), 1.1 * u)
            if k >= 1:
                lm.circle(lx, ly, 3.5 * u, outline=255, width=1.2 * u)
            lm.paint(cv, WHITE, 0.6)
    glow.apply(cv)
    return cv


_NB_LABEL_AT = {"ALB": (18.55, 39.35)}  # label placed in the sea with a leader to the anchor


def _mode_neighbors(t, dur, W, H, p, view, kw):
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    labels = kw.get("labels", True)
    ncol = _col(kw.get("nb_color")) or WHITE
    a0, a1 = float(kw.get("start", 0.06)), float(kw.get("end", 0.86))
    key = ("nb_base", view.key(), kw.get("grid", True), kw.get("hud", True), kw.get("rivers", False),
           stroke, fillc, gk)

    def build():
        cv = _yu_underlay(W, H, view, dict(kw, bg=None), ctx_op=0.8)
        cv.paint(_yu_mask(view, "yugo_fill"), fillc, 0.9)
        cv.paint(_yu_mask(view, "borders", 1.0), DEEP, 0.7)
        cv.paint(_yu_mask(view, "yugo_line", 1.3), _hot(stroke, 0.45), 1.0)
        g = _Glow(W, H)
        rings = [view.px(r) for rs, bb in D.yugo for r in rs[:1]]
        g.put(g.lines(rings, 3 * u), stroke, 6 * u, 1.1 * gk)
        g.put(g.polys(rings), stroke, 34 * u, 0.5 * gk)
        g.apply(cv)
        return cv.snapshot()

    cv = _Canvas(W, H, _col(kw.get("bg")), base=_cache_get(key, build))
    glow = _Glow(W, H)
    n = len(D.nbrs)
    times = [a0 + (a1 - a0) * i / n for i in range(n)]
    settle = 0.18  # after this (fraction) a lit neighbour is drawn in its calm, merged state
    calm = [i for i in range(n) if p >= times[i] + settle]
    if calm:
        ids = ",".join(str(i) for i in calm)
        cv.paint(_yu_mask(view, f"nb{ids}_fill", ss=1), ncol, 0.06)
        cv.paint(_yu_mask(view, f"nb{ids}_line", 1.1), ncol, 0.55)
    lit = len([1 for ti in times if p >= ti])
    for i, nb in enumerate(D.nbrs):
        ti = times[i]
        if p < ti:
            continue
        k = _seg(p, ti, ti + 0.03)
        fl = _pulse(p - ti, 0.045)
        if i not in calm:
            cv.paint(_yu_mask(view, f"nb{i}_fill", ss=1), ncol, (0.06 + 0.3 * fl) * k)
            cv.paint(_yu_mask(view, f"nb{i}_line", 1.1), ncol, (0.55 + 0.45 * _cl(fl * 1.5)) * k)
            rings = [view.px(r) for rs, bb in nb.polys if view.visible(bb) for r in rs[:1]]
            glow.put(glow.lines(rings, 2.5 * u), ncol, 8 * u, 1.2 * fl * k * gk)
        if labels:
            ax, ay = view.px(nb.label)
            if nb.id in _NB_LABEL_AT:
                lx, ly = view.px(np.array(geoproj.project(D.meta, *_NB_LABEL_AT[nb.id])))
            else:
                lx, ly = ax, ay
            lx = min(max(lx, 130 * u), W - 130 * u)
            ly = min(max(ly, 110 * u), H - 80 * u)
            rv = _eo5(_seg(p, ti, ti + 0.05))
            if abs(lx - ax) + abs(ly - ay) > 20 * u:
                path = np.array([[lx + 24 * u, ly - 46 * u], [ax, ay]])
                lm = _SS(W, H, 2, _bbp([path], 6 * u))
                lm.line(_partial(path, _cum(path), _cum(path)[-1] * rv), 1.1 * u)
                lm.circle(ax, ay, 3 * u, fill=255)
                lm.paint(cv, ncol, 0.7 * k)
            _text(cv, f"{i + 1:02d}", "mono", 15 * u, lx, ly - 52 * u, stroke, k, align="m", track=0.2)
            _text(cv, nb.name, "oswald", 40 * u, lx, ly, ncol, k, align="m", reveal=rv, shadow=0.5,
                  glow=glow if fl > 0.05 else None, glow_col=ncol, glow_r=10 * u, glow_k=0.8 * fl)
            _text(cv, nb.en, "mono", 12 * u, lx, ly + 21 * u, ncol, 0.55 * k, align="m", track=0.25)
    if labels and kw.get("counter", True):
        x, y = W - 84 * u, 250 * u
        fl = max([_pulse(p - ti, 0.04) for ti in times if p >= ti] or [0.0])
        _text(cv, str(lit), "anton", 150 * u, x - 66 * u, y, stroke if lit else STEEL, 1.0 if lit else 0.35,
              align="r", glow=glow, glow_col=stroke, glow_r=16 * u, glow_k=0.4 + 0.8 * fl)
        _text(cv, "/7", "anton", 58 * u, x, y, WHITE, 0.8, align="r")
        _text(cv, "SUSEDI", "mono", 18 * u, x, y + 36 * u, WHITE, 0.85, align="r", track=0.3)
        _text(cv, "NEIGHBOURS", "mono", 12 * u, x, y + 58 * u, STEEL, 0.8, align="r", track=0.3)
    glow.apply(cv)
    return cv


CAP_ORDER = ["Ljubljana", "Zagreb", "Sarajevo", "Titograd", "Skopje", "Novi Sad", "Priština", "Beograd"]
_CAP_LABEL = {"Ljubljana": (-1, -1), "Zagreb": (1, -1), "Sarajevo": (-1, 1), "Beograd": (1, -1),
              "Titograd": (-1, 1), "Skopje": (1, 1), "Novi Sad": (-1, -1), "Priština": (1, 1)}
_CAP_SUB = {"Beograd": "GLAVNI GRAD SFRJ · FEDERAL CAPITAL", "Titograd": "DANAS PODGORICA",
            "Novi Sad": "SAP VOJVODINA", "Priština": "SAP KOSOVO"}


def _mode_capitals(t, dur, W, H, p, view, kw):
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    labels = kw.get("labels", True)
    order = kw.get("order", CAP_ORDER)
    a0, a1 = float(kw.get("start", 0.04)), float(kw.get("end", 0.8))
    cv = _rep_base(W, H, view, kw, stroke, fillc, gk)
    glow = _Glow(W, H)
    gdots, _ = glow.new()
    n = len(order)
    rings = []
    for i, name in enumerate(order):
        c = D.cities[name]
        ti = a0 + (a1 - a0) * i / max(n - 1, 1)
        if p < ti:
            continue
        age = (p - ti) * dur  # seconds, so pings read the same at any dur
        big = c.kind == "federal"
        x, y = view.px(c.xy)
        for a in age - np.array([0.0, 0.12, 0.24])[: 3 if big else 2]:
            if 0 <= a < 0.9:
                rr = (14 + 150 * _eo3(a / 0.9)) * u * (1.5 if big else 1.0)
                rings.append((x, y, rr, (1 - a / 0.9) ** 1.5))
        k = _eo3(_seg(age, 0, 0.12))
        if big:
            _star(cv, x, y, 22 * u * _eback(_seg(age, 0, 0.3)), GOLD)
            glow.dot(x, y, 12 * u, im=gdots)
        else:
            _dot(cv, x, y, 5 * u, WHITE, k)
            glow.dot(x, y, 5 * u, im=gdots)
        if labels:
            dx, dy = _CAP_LABEL.get(name, (1, -1))
            L1, L2 = (34 if big else 26) * u, (46 if big else 34) * u
            ex, ey = x + dx * L1, y + dy * L1
            fx = ex + dx * L2
            path = np.array([[x + dx * 8 * u, y + dy * 8 * u], [ex, ey], [fx, ey]])
            cc = _cum(path)
            lm = _SS(W, H, 2, _bbp([path], 4))
            lm.line(_partial(path, cc, cc[-1] * _eo3(_seg(age, 0.02, 0.2))), 1.2 * u)
            lm.paint(cv, WHITE, 0.8)
            rv = _eo3(_seg(age, 0.1, 0.35))
            al = "l" if dx > 0 else "r"
            tx = fx + dx * 6 * u
            fsz = 34 * u if big else 21 * u
            _text(cv, c.name.upper(), "oswald" if big else "mono", fsz, tx, ey + fsz * 0.36,
                  GOLD if big else WHITE, 1.0, align=al, reveal=rv, track=0 if big else 0.1,
                  glow=glow if big else None, glow_col=GOLD, glow_r=10 * u, glow_k=0.4)
            sub = _CAP_SUB.get(name) or f"{c.lat:.2f}°N {c.lon:.2f}°E"
            _text(cv, sub, "mono", 12 * u, tx, ey + fsz * 0.36 + 18 * u, STEEL, 0.9 * rv, align=al, track=0.15)
    if rings:
        rm = _SS(W, H, 2, (min(r[0] - r[2] for r in rings), min(r[1] - r[2] for r in rings),
                           max(r[0] + r[2] for r in rings), max(r[1] + r[2] for r in rings)))
        for x, y, rr, op in rings:
            rm.circle(x, y, rr, outline=255 * op, width=1.6 * u)
        rm.paint(cv, _hot(stroke, 0.3), 1.0)
    glow.put(gdots, _hot(stroke, 0.3), 8 * u, 1.2 * gk)
    glow.apply(cv)
    return cv


# ----------------------------------------------------------- assemble ------
_ASM = {  # launch direction (screen x, y), start rotation (deg), delay (fraction of impact)
    "SVN": ((-0.9, -0.6), -40, 0.00),
    "HRV": ((-1.0, 0.2), 55, 0.06),
    "BIH": ((-0.1, -1.0), -65, 0.12),
    "SRB": ((1.0, -0.25), 30, 0.03),
    "MNE": ((-0.3, 1.0), 75, 0.15),
    "MKD": ((0.7, 0.85), -50, 0.09),
}


def _xf(pts, c, ang, dx, dy, S, gc):
    """Rotate pts (px) about c by ang, translate, then scale by S about gc."""
    ca, sa = math.cos(ang), math.sin(ang)
    x = pts[:, 0] - c[0]
    y = pts[:, 1] - c[1]
    X = x * ca - y * sa + c[0] + dx
    Y = x * sa + y * ca + c[1] + dy
    return np.column_stack([(X - gc[0]) * S + gc[0], (Y - gc[1]) * S + gc[1]])


def _asm_geom(view):
    def build():
        D = _yu()
        W, H = view.W, view.H
        out = []
        for rep in D.reps:
            rings = [view.px(r) for rs, bb in rep.polys for r in rs[:1]]
            c = view.px(rep.label)
            bx0, by0, bx1, by1 = _bbp(rings)
            (ddx, ddy), rot0, dl = _ASM[rep.id]
            n = math.hypot(ddx, ddy)
            ddx, ddy = ddx / n, ddy / n
            ext = max(bx1 - bx0, by1 - by0)
            need = []
            if ddx > 1e-3:
                need.append((W - bx0) / ddx)
            if ddx < -1e-3:
                need.append(bx1 / -ddx)
            if ddy > 1e-3:
                need.append((H - by0) / ddy)
            if ddy < -1e-3:
                need.append(by1 / -ddy)
            dist = min(need) + 0.12 * ext + 20
            prov = [view.px(ln) for ln in D.prov_borders] if rep.id == "SRB" else []
            out.append((rings, c, (ddx, ddy), dist, math.radians(rot0), dl, prov))
        yug = [view.px(r) for rs, bb in D.yugo for r in rs[:1]]
        return out, yug

    return _cache_get(("asm", view.key()), build)


def _mode_assemble(t, dur, W, H, p, view, kw):
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fills = kw.get("fill")
    if fills is None:
        fills = [RED] * 6
    elif isinstance(fills, (list, tuple)) and len(fills) == 6 and not isinstance(fills[0], (int, float)):
        fills = [_col(f) for f in fills]
    else:
        fills = [_col(fills)] * 6
    uniform = all(f == fills[0] for f in fills)
    gk = float(kw.get("glow", 1.0))
    imp = float(kw.get("impact", 0.55))
    trails = kw.get("trails", True)
    geo, yug = _asm_geom(view)
    ts = (p - imp) * dur  # seconds after impact
    S, shx, shy = 1.0, 0.0, 0.0
    if ts >= 0:  # camera punch + shake
        S = 1.0 + 0.055 * math.exp(-ts * 7.0) * math.cos(ts * 26.0)
        amp = 9 * u * math.exp(-ts * 12.0)
        shx, shy = amp * math.sin(ts * 91.0), amp * math.cos(ts * 73.0)
    gc = (W / 2 + shx, H / 2 + shy)
    cv = _yu_underlay(W, H, view, kw, _seg(p, 0, 0.1) if ts < 0 else 1.0)
    glow = _Glow(W, H)

    def pose(i, pp):
        rings, c, (ddx, ddy), dist, rot0, dl, prov = geo[i]
        q = _seg(pp, dl * imp, imp)
        k = 1.0 - q ** 0.95
        ang = rot0 * k ** 1.2
        dx, dy = ddx * dist * k + shx, ddy * dist * k + shy
        if pp >= imp:  # recoil: tiny radial bounce
            tt = (pp - imp) * dur
            rx, ry = c[0] - W / 2, c[1] - H / 2
            nn = math.hypot(rx, ry) + 1e-6
            b = 7 * u * math.sin(min(tt / 0.16, 1.0) * math.pi) * math.exp(-tt * 6)
            dx += rx / nn * b
            dy += ry / nn * b
        return ang, dx, dy

    flash = _pulse(ts, 0.09) if ts >= 0 else 0.0
    seam = _seg(ts, 0.08, 0.5) if ts >= 0 else 0.0
    placed = []
    for i in range(6):
        rings, c, _, _, _, _, prov = geo[i]
        ang, dx, dy = pose(i, p)
        placed.append(([_xf(r, c, ang, dx, dy, S, gc) for r in rings],
                       [_xf(ln, c, ang, dx, dy, S, gc) for ln in prov]))
    allr = [r for pr, _ in placed for r in pr]
    bb = _bbp(allr, 6 * u)
    if trails and p < imp:
        ghosts = []
        for i in range(6):
            rings, c = geo[i][0], geo[i][1]
            a0, dx0, dy0 = pose(i, p)
            for j, dtp in enumerate((0.014, 0.03, 0.05)):
                a2, dx2, dy2 = pose(i, max(p - dtp, 0))
                if abs(dx2 - dx0) + abs(dy2 - dy0) >= 3:
                    ghosts.append((_xf(rings[0], c, a2, dx2, dy2, S, gc), int(80 / (j + 1))))
        if ghosts:
            gh = _SS(W, H, 1, _bbp([g for g, _ in ghosts], 2))
            for g, v in ghosts:
                gh.poly(g, v)
            gh.paint(cv, stroke, 0.9)
    fa = float(kw.get("fill_alpha", 0.92))
    if uniform:
        fm = _SS(W, H, 2, bb)
        for r in allr:
            fm.poly(r)
        fm.paint(cv, _mix(fills[0], (1, 1, 1), flash), fa)
    else:
        for i, (pr, _) in enumerate(placed):
            fm = _SS(W, H, 2, _bbp(pr, 4))
            for r in pr:
                fm.poly(r)
            fm.paint(cv, _mix(fills[i], (1, 1, 1), flash), fa)
    pm = _SS(W, H, 2, bb)
    for _, pl in placed:
        for ln in pl:
            for d in _dashes(ln, 6 * u, 5 * u):
                pm.line(d, 1.0 * u)
    pm.paint(cv, DEEP, 0.9)
    em = _SS(W, H, 2, bb)
    for r in allr:
        em.line(r, 1.5 * u)
    em.paint(cv, _mix(_hot(stroke, 0.6), DEEP, seam), 1.0)
    glow.put(glow.lines(allr, 4), stroke, 6 * u, (1.0 - 0.6 * seam) * gk)
    if ts >= 0:  # unified outline + flash + shock ring after the slam
        yr = [_xf(r, (0, 0), 0.0, shx, shy, S, gc) for r in yug]
        om = _SS(W, H, 2, _bbp(yr, 4))
        for r in yr:
            om.line(r, 2.0 * u)
        om.paint(cv, _hot(stroke, 0.35 + 0.5 * flash), _seg(ts, 0.0, 0.15))
        gl = glow.lines(yr, 4)
        glow.put(gl, stroke, 6 * u, 1.3 * gk * (1 + 2 * flash))
        glow.put(glow.polys(yr) if flash > 0.05 else gl, stroke, 30 * u, (0.8 + 2.4 * flash) * gk)
        if ts < 0.6:
            rr = (0.15 + 1.1 * _eo3(ts / 0.6)) * H
            op = (1 - ts / 0.6) ** 1.5
            sw = _SS(W, H, 2, (gc[0] - rr - 12 * u, gc[1] - rr - 12 * u, gc[0] + rr + 12 * u, gc[1] + rr + 12 * u))
            for dr_, a_ in ((10, 50), (5, 110), (0, 255)):
                sw.circle(gc[0], gc[1], rr - dr_ * u * 0.5, outline=a_, width=(2.5 + dr_) * u)
            sw.paint(cv, _hot(stroke, 0.75), op * 0.85)
    glow.apply(cv)
    return cv


# ------------------------------------------------------------- tilt 3D -----
@functools.lru_cache(maxsize=8)
def _vignette(W, H):
    y, x = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    r = np.hypot((x - W / 8) / (W / 8), (y - H * 0.55 / 4) / (H / 8))
    v = np.clip(1.2 - r, 0, 1) ** 1.6
    return Image.fromarray((v * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)


def _sheen(w, h, p):
    ww, hh = max(2, w // 6), max(2, h // 6)
    y, x = np.mgrid[0:hh, 0:ww].astype(np.float32)
    c = (-0.35 + 1.7 * p) * (ww + hh * 0.7)
    d = (x + y * 0.7 - c) / (0.14 * (ww + hh))
    v = np.exp(-d * d)
    return Image.fromarray((v * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)


def _mode_tilt3d(t, dur, W, H, p, view, kw):
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or GOLD
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    ext = float(kw.get("extrude", 0.09))
    yaw = math.radians(float(kw.get("yaw", 0.0)) + float(kw.get("spin", 22.0)) * (p - 0.5))
    pitch = math.radians(float(kw.get("pitch", 50.0)) + float(kw.get("nod", 5.0)) * math.sin(p * math.pi))
    scale = float(kw.get("scale", 1.0))
    dist = float(kw.get("distance", 2.2))
    def build():
        c = _Canvas(W, H)
        if kw.get("hud", True):
            _hud(c, view, [], "SFRJ // 1945–1992", None, None, False, True, 1.0)
        return c.snapshot()

    cv = _Canvas(W, H, _col(kw.get("bg")), base=_cache_get(("t3d_base", W, H, kw.get("hud", True)), build))
    glow = _Glow(W, H)

    x0, y0, x1, y1 = D.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = 1.0 / max(x1 - x0, y1 - y0)
    hgt = ext * (y1 - y0) * k
    ca, sa = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    f = H * 2.05 * scale * dist / 2.2
    offx, offy = kw.get("offset", (0.0, 0.0))
    scx, scy = W / 2 + offx * W, H * (0.5 + 0.04 - float(kw.get("lift", 0.0))) + offy * H

    def P3(xy, z):
        X = (xy[:, 0] - cx) * k
        Y = (xy[:, 1] - cy) * k
        Xr = X * ca - Y * sa
        Yr = X * sa + Y * ca
        y2 = Yr * cp + z * sp
        z2 = -Yr * sp + z * cp
        dd = dist - z2
        return np.column_stack([scx + f * Xr / dd, scy - f * y2 / dd]), dd

    rings = [r for rs, bb in D.coarse for r in rs[:1]]
    tops = [P3(r, 0.0)[0] for r in rings]
    bots = [P3(r, -hgt)[0] for r in rings]
    bb = _bbp(tops + bots, 6 * u)

    if kw.get("floor", True):
        fz = -hgt - 0.03
        # floor grid drawn straight into the (still empty) canvas, faded radially per segment
        dC, dA = ImageDraw.Draw(cv.C), ImageDraw.Draw(cv.A)
        bgc = _col(kw.get("bg"))
        n, R = 12, 1.5
        ss_ = np.linspace(0, 1, 21)[:, None]
        for i in range(-n, n + 1):
            v = i / n * R / k
            for a, b in (((v + cx, -R / k + cy), (v + cx, R / k + cy)), ((-R / k + cx, v + cy), (R / k + cx, v + cy))):
                pts = np.array(a) + (np.array(b) - np.array(a)) * ss_
                sc, dd = P3(pts, fz)
                mid = (sc[1:] + sc[:-1]) / 2
                rr = np.hypot((mid[:, 0] - W / 2) / (W / 2), (mid[:, 1] - H * 0.55) / (H / 2))
                al = 0.28 * np.clip(1.2 - rr, 0, 1) ** 1.6
                for j in np.nonzero((al > 0.01) & (dd[1:] > 0.4) & (dd[:-1] > 0.4))[0]:
                    seg = sc[j:j + 2].ravel().tolist()
                    if bgc is None:
                        dC.line(seg, fill=_u8(tuple(c * al[j] for c in STEEL)), width=max(1, int(u + 0.5)))
                        dA.line(seg, fill=int(255 * al[j]), width=max(1, int(u + 0.5)))
                    else:
                        dC.line(seg, fill=_u8(_mix(bgc, STEEL, al[j])), width=max(1, int(u + 0.5)))
        sh = glow.polys([P3(r, fz)[0] for r in rings])
        sh = sh.filter(ImageFilter.GaussianBlur(16 * u / glow.ds))
        shb = sh.getbbox()
        if shb:
            ds = glow.ds
            crop = sh.crop(shb).resize(((shb[2] - shb[0]) * ds, (shb[3] - shb[1]) * ds), Image.BILINEAR)
            cv.paint(crop, BLACK, 0.75, (shb[0] * ds, shb[1] * ds))

    # side walls: per-edge quads, back-face culled, painter-sorted, Lambert shaded
    light = np.array([-0.45, 0.35, 0.82])
    light /= np.linalg.norm(light)
    quads = []
    for r, top, bot in zip(rings, tops, bots):
        if abs(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1])) / 2 < 15000:  # walls only for > 150 km^2
            continue
        X = (r[:, 0] - cx) * k
        Y = (r[:, 1] - cy) * k
        nx, ny = -np.diff(Y), np.diff(X)  # outward normals of a clockwise ring
        nn = np.hypot(nx, ny) + 1e-12
        nx, ny = nx / nn, ny / nn
        nxr = nx * ca - ny * sa
        nyr = nx * sa + ny * ca
        ny2, nz2 = nyr * cp, -nyr * sp
        mx, my = (X[:-1] + X[1:]) / 2, (Y[:-1] + Y[1:]) / 2
        mxr = mx * ca - my * sa
        myr = mx * sa + my * ca
        mz = -hgt / 2
        py2 = myr * cp + mz * sp
        pz2 = -myr * sp + mz * cp
        vis = (nxr * -mxr + ny2 * -py2 + nz2 * (dist - pz2)) > 0
        lam = np.clip(nxr * light[0] + ny2 * light[1] + nz2 * light[2], 0, 1)
        for _ in range(2):  # smooth the facet shading along the ring (jagged coasts)
            lam = 0.25 * np.roll(lam, 1) + 0.5 * lam + 0.25 * np.roll(lam, -1)
        depth = dist - pz2
        for j in np.nonzero(vis)[0]:
            quads.append((depth[j], lam[j], top[j], top[j + 1], bot[j + 1], bot[j]))
    quads.sort(key=lambda q: -q[0])
    wm = _SS(W, H, 2, bb)
    wrgb = Image.new("RGB", wm.im.size, (0, 0, 0))
    wd = ImageDraw.Draw(wrgb)
    wc = np.array(_mix(fillc, DEEP, 0.55))
    edge = _u8(_mix(wc * 0.4, stroke, 0.55))
    ew = _lw(1.0 * u, 2)
    for dep, lam, a, b, c, d in quads:
        shade = 0.25 + 1.0 * lam
        colr = tuple(int(min(255, v * 255 * shade)) for v in wc)
        poly = wm.P(np.array([a, b, c, d]))
        wd.polygon(poly, fill=colr, outline=colr)
        wd.line(poly[4:8], fill=edge, width=ew)  # gold-ish bottom edge, occluded by nearer walls
        wm.dr.polygon(poly, fill=255, outline=255)
    wa = wm.mask()
    cv.paste_rgb(wrgb.reduce(2), wa, (wm.x0, wm.y0))

    tm = _SS(W, H, 2, bb)
    for tp in tops:
        tm.poly(tp)
    topm = tm.mask()
    cv.paint(topm, fillc, 1.0, (tm.x0, tm.y0))
    cv.paint(ImageChops.multiply(topm, _sheen(topm.size[0], topm.size[1], p)), (1.0, 0.8, 0.72), 0.35,
             (tm.x0, tm.y0))
    bm = _SS(W, H, 2, bb)
    for ln in D.coarse_borders:
        bm.line(P3(ln, 0.0)[0], 1.3 * u)
    bm.paint(cv, DEEP, 0.85)
    om = _SS(W, H, 2, bb)
    for tp in tops:
        om.line(tp, 2.4 * u)
    om.paint(cv, _hot(stroke, 0.2), 1.0)
    gl = glow.lines(tops, 4)
    glow.put(gl, stroke, 6 * u, 0.9 * gk)
    glow.put(gl, RED, 28 * u, 0.7 * gk)
    if kw.get("star", True):
        sx, sy = P3(D.cities["Beograd"].xy[None], 0.0)[0][0]
        _star(cv, sx, sy, 13 * u, GOLD)
        glow.put(glow.dot(sx, sy, 8 * u), GOLD, 10 * u, 1.2 * gk)
    if kw.get("hud", True):
        m = int(round(28 * u))
        _text(cv, f"YAW {math.degrees(yaw):+06.1f}°  PITCH {math.degrees(pitch):04.1f}°", "mono", 12 * u,
              m + 36 * u, m + 34 * u, WHITE, 0.4)
    glow.apply(cv)
    return cv


# ============================================================ yugo_map =====
_MODE_DEFAULTS = {
    "neighbors": dict(scale=0.64, center=(19.4, 43.3)),
    "republics": dict(scale=0.93, offset=(0.12, 0.0)),
}


def yugo_map(t, dur, W, H, mode="draw", **kw):
    """Yugoslavia map generator -> float32 RGBA (H, W, 4), straight alpha.

    mode
      "draw"       glowing red outline draws itself on (bright spark + coordinate
                   read-out at the head), islands pop in as the head passes, then
                   republic borders and dashed province borders; faint fill wash.
                   kw: reveal_end=0.62 (fraction of dur when the outline closes),
                       width=1.8 (px @1080), fill_alpha=0.2
      "assemble"   the 6 republics (Serbia incl. its provinces, dashed) fly in from
                   off-screen with rotation + motion trails and slam together at
                   p=impact: white flash, camera punch + shake, shock ring, seams
                   cool down, the unified outline glows.
                   kw: impact=0.55, fill=<colour | list of 6 colours>, trails=True,
                       fill_alpha=0.92
      "republics"  all dim; hi=0..5 lights one republic (SLOVENIJA, HRVATSKA, BOSNA I
                   HERCEGOVINA, SRBIJA, CRNA GORA, MAKEDONIJA) + big name + English
                   caption + capital + leader line.  hi=None -> sequence through all
                   six over dur.  Default view is shifted right (offset=(0.12, 0),
                   scale 0.93) to leave the lower-left for the name.
                   kw: hi, sequence, local=<0..1 entry progress when hi is fixed
                       (default t/dur)>, name_font="anton", name_size=150,
                       name_width=0.33 (max fraction of W), leader=True
      "neighbors"  SFRJ filled red; ITALIJA, AUSTRIJA, MAĐARSKA, RUMUNIJA, BUGARSKA,
                   GRČKA, ALBANIJA light up in turn (flash, then calm) with names +
                   counter N/7 (SUSEDI).  Default view zoomed out (scale 0.64).
                   kw: start=0.06, end=0.86, nb_color=white, counter=True
      "capitals"   ping rings from the capitals in sequence, callout labels
                   (Ljubljana, Zagreb, Sarajevo, Titograd, Skopje, Novi Sad, Priština,
                   then BEOGRAD with a gold star).
                   kw: order=[names], start=0.04, end=0.8
      "tilt3d"     extruded slab in perspective, slowly yawing; red top face, gold rim,
                   Lambert-lit side walls, floor grid + soft shadow, sheen sweep,
                   Beograd star.  (scale / offset act on the 3D camera; center ignored)
                   kw: extrude=0.09 (thickness / map height), yaw=0 (deg, centre of the
                       motion), spin=22 (deg over dur), pitch=50 (deg; 0 = top-down),
                       nod=5, distance=2.2 (smaller = stronger perspective),
                       floor=True, star=True, lift=0 (raise on screen, fraction of H),
                       stroke=gold rim colour

    Common kw: scale=1 (zoom multiplier on the mode's default), center=(lon, lat),
    offset=(dx, dy) (fraction of W/H to shift the map on screen), stroke=<colour>,
    fill=<colour>, glow=1.0 (strength), labels=True, hud=True (corner brackets, edge
    ticks, read-outs, scale bar), grid=True (graticule), rivers=False, scan=False
    (scanline alpha), bg=None (transparent) or a colour, phase=None (override t/dur).
    Colours: PAL names ('red', 'gold', 'navy', ...), '#hex' or rgb floats.
    """
    p = kw.pop("phase", None)
    p = _cl(t / dur if dur > 0 else 1.0) if p is None else _cl(float(p))
    d = _MODE_DEFAULTS.get(mode, {})
    scale = float(kw.get("scale", 1.0)) * d.get("scale", 1.0)
    center = kw.get("center", d.get("center"))
    offset = kw.get("offset", d.get("offset", (0.0, 0.0)))
    view = _yu_view(W, H, scale, center, offset)
    fn = {"draw": _mode_draw, "assemble": _mode_assemble, "republics": _mode_republics,
          "neighbors": _mode_neighbors, "capitals": _mode_capitals, "tilt3d": _mode_tilt3d}[mode]
    return fn(t, dur, W, H, p, view, kw).array(scan=kw.get("scan", False))


# ================================================================= WORLD ====
@functools.lru_cache(None)
def _wo():
    d = _load_json("world")
    Wd = _NS(meta=d["meta"])
    Wd.feats = {f["id"]: _NS(id=f["id"], name=f["name"], polys=_polys(f["p"]),
                             label=np.array(f["label"], float)) for f in d["features"]}
    Wd.nam = d["nam1961"]
    Wd.obs = d["observers1961"]
    Wd.cities = {c["name"]: _NS(name=c["name"], en=c["en"], lon=c["lon"], lat=c["lat"],
                                xy=np.array(c["xy"], float)) for c in d["cities"]}
    Wd.allp = [p for f in Wd.feats.values() for p in f.polys]
    Wd.bounds = _bbox_of(Wd.allp)
    lon0 = d["meta"]["proj"]["lon0"]
    la = np.linspace(-90, 90, 181)
    e = 1e-6
    left = np.column_stack(geoproj.project(Wd.meta, np.full_like(la, lon0 - 180 + e), la))
    right = np.column_stack(geoproj.project(Wd.meta, np.full_like(la, lon0 + 180 - e), la))
    Wd.globe = np.vstack([left, right[::-1], left[:1]])
    Wd.grat = _grat_lines(Wd.meta, [v for v in range(-150, 181, 30) if abs(((v - lon0 + 180) % 360) - 180) < 179],
                          list(range(-60, 76, 15)), (lon0 - 180 + 1e-6, lon0 + 180 - 1e-6), (-90, 90), step=1.0)
    return Wd


def _wo_groups(Wd, uar_syria):
    groups = []
    for g in Wd.nam:
        ids = [i for i in g["ids"] if not (i == "SYR" and not uar_syria)]
        groups.append(_NS(name=g["name"], en=g["en"], ids=ids))
    return groups


WORLD_ARCS = [("Kairo", "NASER", (-1, 1)), ("Nju Delhi", "NEHRU", (1, -1)),
              ("Džakarta", "SUKARNO", (-1, 1)), ("Akra", "NKRUMA", (-1, 1))]
_FOUNDER_GROUP = {"Kairo": "UAR", "Nju Delhi": "INDIJA", "Džakarta": "INDONEZIJA", "Akra": "GANA"}


def world_nam(t, dur, W, H, **kw):
    """World map for the 1961 Belgrade Non-Aligned conference -> float32 RGBA.

    Dark world (thin grey outlines, Equal Earth, seam in the Bering Strait, no
    Antarctica); Yugoslavia red + glowing.  Great-circle arcs draw from Beograd to
    Kairo (Naser), Nju Delhi (Nehru), Džakarta (Sukarno), Akra (Nkruma) with
    travelling light pulses; each founder's country lights gold on arrival; then the
    remaining full participants of 1-6 Sep 1961 light up gold in a wave ordered by
    distance from Belgrade, with a counter NN/25 and a ticker of the latest name.
    1961 borders where it matters: UAR = Egypt + Syria (Syria seceded 28 Sep 1961),
    Yemen = North Yemen only, Indonesia without Dutch New Guinea, Ethiopia incl.
    Eritrea, Sudan incl. South Sudan, Somalia incl. Somaliland, Cyprus whole,
    Morocco without Western Sahara, USSR / Czechoslovakia / FRG-GDR as in 1961.

    kw: scale=1, center=(lon, lat) (default: whole map), offset=(dx, dy),
        arc_start=0.06, arc_len=0.2, stagger=0.055, wave=(0.45, 0.85),
        pulse_period=1.1 (s), lift=0.0 (bow the arcs, fraction of their chord),
        uar_syria=True, observers=False (Bolivia, Brazil, Ecuador in dim gold),
        counter=True, labels=True, hud=True, grid=True, glow=1.0, scan=False,
        bg=None | colour ('navy' works well), phase=None (override t/dur).
    """
    p = kw.pop("phase", None)
    p = _cl(t / dur if dur > 0 else 1.0) if p is None else _cl(float(p))
    Wd = _wo()
    u = H / 1080.0
    gk = float(kw.get("glow", 1.0))
    labels = kw.get("labels", True)
    uar_syria = kw.get("uar_syria", True)
    observers = kw.get("observers", False)
    view = _make_view(W, H, Wd.bounds, (0.97, 0.80), float(kw.get("scale", 1.0)), kw.get("center"),
                      kw.get("offset", (0.0, 0.02)), Wd.meta)
    groups = _wo_groups(Wd, uar_syria)
    gidx = {g.name: i for i, g in enumerate(groups)}
    yi = gidx["JUGOSLAVIJA"]

    def build_base():
        cv = _Canvas(W, H)
        fm = _SS(W, H, 1)
        _ss_polys(fm, Wd.allp, view)
        cv.paint(fm.mask(), STEEL, 0.07)
        lm = _SS(W, H, 2)
        _ss_outlines(lm, Wd.allp, view, 0.8 * u)
        cv.paint(lm.mask(), STEEL, 0.38)
        _hud(cv, view, Wd.grat, "BEOGRAD // 1.–6. IX 1961",
             "PRVA KONFERENCIJA NESVRSTANIH  ·  EQUAL EARTH", None, kw.get("grid", True), kw.get("hud", True),
             globe=Wd.globe, gkey="world", grid_op=0.13)
        y = Wd.feats["YUG"].polys
        ym = _SS(W, H, 2)
        _ss_polys(ym, y, view)
        cv.paint(ym.mask(), RED, 1.0)
        yo = _SS(W, H, 2)
        _ss_outlines(yo, y, view, 1.2 * u)
        cv.paint(yo.mask(), _hot(RED, 0.4), 1.0)
        g = _Glow(W, H)
        rings = [view.px(r) for rs, bb in y for r in rs[:1]]
        g.put(g.polys(rings), RED, 8 * u, 1.6 * gk)
        g.put(g.polys(rings), RED, 30 * u, 1.2 * gk)
        g.apply(cv)
        return cv.snapshot()

    def build_ids():
        idm = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(idm)
        out = _SS(W, H, 2)
        lab = {}
        wd = max(1, int(round(2 * u)))
        for i, g in enumerate(groups):
            if i == yi:
                continue
            for fid in g.ids:
                for rs, bb in Wd.feats[fid].polys:
                    if view.visible(bb):
                        e = view.px(rs[0])
                        dr.polygon(e.ravel().tolist(), fill=i + 1, outline=i + 1, width=wd)
                        out.line(e, 1.0 * u)
            lab[i] = view.px(Wd.feats[g.ids[0]].label)
        for j, g in enumerate(Wd.obs):
            for fid in g["ids"]:
                for rs, bb in Wd.feats[fid].polys:
                    e = view.px(rs[0])
                    dr.polygon(e.ravel().tolist(), fill=40 + j, outline=40 + j, width=wd)
                    out.line(e, 1.0 * u)
        low = idm.resize((-(-W // 4), -(-H // 4)), Image.NEAREST)
        return idm, out.mask(), low, lab

    base = _cache_get(("wo_base", view.key(), kw.get("grid", True), kw.get("hud", True), gk), build_base)
    idm, outl, idlow, lab = _cache_get(("wo_ids", view.key(), uar_syria), build_ids)
    cv = _Canvas(W, H, _col(kw.get("bg")), base=base, op=_eo3(_seg(p, 0.0, 0.08)))
    glow = _Glow(W, H)

    # ---- timing
    a0 = float(kw.get("arc_start", 0.06))
    alen = float(kw.get("arc_len", 0.2))
    stag = float(kw.get("stagger", 0.055))
    w0, w1 = kw.get("wave", (0.45, 0.85))
    bel = Wd.cities["Beograd"]
    bx, by = view.px(bel.xy)
    lit_t = {yi: 0.0}
    arcs = []
    for j, (city, leader, ldir) in enumerate(WORLD_ARCS):
        s0 = a0 + j * stag
        arcs.append((Wd.cities[city], leader, ldir, s0))
        lit_t[gidx[_FOUNDER_GROUP[city]]] = s0 + alen
    rest = [i for i in range(len(groups)) if i not in lit_t]
    rest.sort(key=lambda i: math.hypot(lab[i][0] - bx, lab[i][1] - by))
    for r, i in enumerate(rest):
        lit_t[i] = w0 + (w1 - w0) * r / max(len(rest) - 1, 1)

    # ---- participants fill (id-map LUTs)
    fill_lut, flash_lut, edge_lut, glow_lut = [0] * 256, [0] * 256, [0] * 256, [0] * 256
    count, latest = 0, None
    for i, tt in lit_t.items():
        if p < tt:
            continue
        count += 1
        if latest is None or tt >= lit_t[latest]:
            latest = i
        if i == yi:
            continue
        k = _seg(p, tt, tt + 0.025)
        fl = _pulse((p - tt) * dur, 0.12)
        fill_lut[i + 1] = int(255 * 0.78 * k)
        flash_lut[i + 1] = int(255 * 0.9 * fl)
        edge_lut[i + 1] = int(255 * k)
        glow_lut[i + 1] = int(255 * min(1.0, 0.25 + 1.2 * fl) * k)
    if observers:
        ob = _seg(p, w1, w1 + 0.06)
        for j in range(len(Wd.obs)):
            fill_lut[40 + j] = int(255 * 0.18 * ob)
            edge_lut[40 + j] = int(255 * 0.5 * ob)
    cv.paint(idm.point(fill_lut), GOLD, 1.0)
    cv.paint(idm.point(flash_lut), WHITE, 1.0)
    cv.paint(ImageChops.multiply(outl, idm.point(edge_lut)), _hot(GOLD, 0.45), 1.0)
    glow.put(idlow.point(glow_lut), GOLD, 14 * u, 0.9 * gk)

    # ---- arcs + pulses + endpoints
    ts = p * dur
    lift = float(kw.get("lift", 0.0))
    arc_px = []
    for c, leader, ldir, s0 in arcs:
        lon, lat = geoproj.great_circle(bel.lon, bel.lat, c.lon, c.lat, 160)
        xy = view.px(np.column_stack(geoproj.project(Wd.meta, lon, lat)))
        if lift:
            d = xy[-1] - xy[0]
            nrm = np.array([d[1], -d[0]]) / (np.hypot(*d) + 1e-9)
            if nrm[1] > 0:
                nrm = -nrm
            s = np.linspace(0, 1, len(xy))
            xy = xy + nrm[None] * (np.sin(np.pi * s) * lift * np.hypot(*d))[:, None]
        arc_px.append(xy)
    am = _SS(W, H, 2, _bbp(arc_px, 30 * u))
    pl, pld = glow.new()
    rings = []
    for (c, leader, ldir, s0), xy in zip(arcs, arc_px):
        q = _eio(_seg(p, s0, s0 + alen))
        if q <= 0:
            continue
        cum = _cum(xy)
        part = _partial(xy, cum, q * cum[-1])
        am.line(part, 1.6 * u)
        glow.put(glow.lines([part], 4), GOLD, 5 * u, 0.9 * gk)
        if q < 1:
            hx, hy = part[-1]
            glow.dot(hx, hy, 4 * u, im=pl)
            _dot(cv, hx, hy, 3 * u, WHITE)
            continue
        arr_s = (s0 + alen) * dur
        per = float(kw.get("pulse_period", 1.1))
        for off in (0.0, 0.5):
            ph = ((ts - arr_s) / per + off) % 1.0
            sp = ph * cum[-1]
            seg = _subpath(xy, cum, sp - 0.12 * cum[-1], sp)
            if len(seg) >= 2:
                pld.line((seg / glow.ds).ravel().tolist(), fill=255, width=1)
                am.line(seg, 2.2 * u)
                _dot(cv, seg[-1][0], seg[-1][1], 2.6 * u, WHITE)
        age = ts - arr_s
        ex, ey = xy[-1]
        for a in (age, age - 0.15):
            if 0 <= a < 0.8:
                rings.append((ex, ey, (6 + 60 * _eo3(a / 0.8)) * u, (1 - a / 0.8) ** 1.5))
        _dot(cv, ex, ey, 4 * u, WHITE)
        glow.dot(ex, ey, 5 * u, im=pl)
        if labels:
            rv = _eo3(_seg(age, 0.0, 0.3))
            dx, dy = ldir
            lx, ly = ex + dx * 16 * u, ey + dy * 30 * u
            al = "l" if dx > 0 else "r"
            _text(cv, c.name.upper(), "oswald", 26 * u, lx, ly, WHITE, rv, align=al, reveal=rv, shadow=0.7)
            _text(cv, f"{leader}  ·  {c.en.upper()}", "mono", 12 * u, lx, ly + 18 * u, _hot(GOLD, 0.35),
                  rv, align=al, track=0.15, shadow=0.85)
    am.paint(cv, _hot(GOLD, 0.35), 1.0)
    glow.put(pl, _hot(GOLD, 0.4), 6 * u, 1.6 * gk)
    if rings:
        rm = _SS(W, H, 2, (min(r[0] - r[2] for r in rings), min(r[1] - r[2] for r in rings),
                           max(r[0] + r[2] for r in rings), max(r[1] + r[2] for r in rings)))
        for x, y, rr, op in rings:
            rm.circle(x, y, rr, outline=255 * op, width=1.4 * u)
        rm.paint(cv, GOLD, 1.0)
    _star(cv, bx, by, 9 * u, GOLD)
    glow.put(glow.dot(bx, by, 6 * u), _hot(RED, 0.3), 10 * u, 1.2 * gk)
    if labels:
        rv = _eo3(_seg(p, 0.02, 0.1))
        _text(cv, "BEOGRAD", "oswald", 28 * u, bx - 14 * u, by - 22 * u, WHITE, rv, align="r", reveal=rv,
              shadow=0.6)
        _text(cv, "TITO  ·  BELGRADE", "mono", 12 * u, bx - 14 * u, by - 6 * u, _hot(RED, 0.3), 0.9 * rv,
              align="r", track=0.15, shadow=0.8)
    if labels and kw.get("counter", True):
        x, y = 64 * u, H - 120 * u
        fl = _pulse((p - lit_t[latest]) * dur, 0.1) if latest is not None and latest != yi else 0.0
        _text(cv, f"{count:02d}", "anton", 120 * u, x, y, GOLD, 1.0, glow=glow, glow_col=GOLD, glow_r=14 * u,
              glow_k=0.35 + 0.8 * fl)
        adv = _tadv(f"{count:02d}", "anton", 120 * u)
        _text(cv, "/25", "anton", 46 * u, x + adv + 8 * u, y, WHITE, 0.85)
        adv += 8 * u + _tadv("/25", "anton", 46 * u) - 40 * u
        _text(cv, "ZEMALJA UČESNICA", "mono", 17 * u, x, y + 34 * u, WHITE, 0.9, track=0.25)
        _text(cv, "FULL MEMBERS · 1ST NON-ALIGNED SUMMIT", "mono", 12 * u, x, y + 54 * u, STEEL, 0.9, track=0.2)
        if latest is not None and count > 1:
            _text(cv, "+ " + groups[latest].name, "mono", 15 * u, x + adv + 70 * u, y - 4 * u, GOLD,
                  0.6 + 0.4 * fl, track=0.15)
    glow.apply(cv)
    return cv.array(scan=kw.get("scan", False))


# ================================================================ EUROPE ====
@functools.lru_cache(None)
def _eu():
    d = _load_json("europe")
    E = _NS(meta=d["meta"])
    E.feats = {f["id"]: _NS(id=f["id"], name=f["name"], polys=_polys(f["p"]),
                            label=np.array(f["label"], float)) for f in d["features"]}
    E.allp = [p for f in E.feats.values() for p in f.polys]
    E.grat = _grat_lines(E.meta, list(range(-30, 81, 10)), list(range(30, 81, 10)), (-40, 90), (20, 85), step=0.5)
    return E


NATO_1949 = ["BEL", "CAN", "DNK", "FRA", "ISL", "ITA", "LUX", "NLD", "NOR", "PRT", "GBR", "USA", "GRL", "FRO"]


def _blocs(year):
    nato = list(NATO_1949)
    if year >= 1952:
        nato += ["GRC", "TUR"]
    if year >= 1955:
        nato += ["FRG"]
    if year >= 1982:
        nato += ["ESP"]
    pact = []
    if year >= 1955:
        pact = ["SUN", "POL", "CSK", "GDR", "HUN", "ROU", "BGR"]
        if year < 1968:
            pact.append("ALB")
    return nato, pact


def europe_blocs(t, dur, W, H, **kw):
    """Cold-War Europe: NATO cold blue floods in from the west, the Warsaw Pact
    red-black (hatched) from the east, then Yugoslavia punches through in gold/white,
    standing alone between them, with 'IZMEĐU ISTOKA I ZAPADA'.  -> float32 RGBA.

    Entities: USSR, Czechoslovakia, FRG / GDR (split along the inner-German border,
    Berlin drawn wholly in the GDR), SFRJ.  Membership follows ``year`` (default 1975):
    NATO 1949 + Greece/Turkey 1952 + FRG 1955 + Spain 1982; Warsaw Pact 1955 with
    Albania until 1968.

    kw: year=1975, flood=(0.05, 0.42) (NATO front sweep; the Pact front follows
        0.04 later), punch=0.5 (Yugoslavia impact), label="IZMEĐU ISTOKA I ZAPADA"
        (None to hide), caption="BETWEEN EAST AND WEST", bloc_labels=True,
        scale=1, center=(lon, lat) (default 16.0E 47.3N), offset=(dx, dy),
        labels=True, hud=True, grid=True, glow=1.0, scan=False, bg=None | colour,
        phase=None (override t/dur).
    """
    p = kw.pop("phase", None)
    p = _cl(t / dur if dur > 0 else 1.0) if p is None else _cl(float(p))
    E = _eu()
    u = H / 1080.0
    gk = float(kw.get("glow", 1.0))
    year = int(kw.get("year", 1975))
    labels = kw.get("labels", True)
    lon_c, lat_c = kw.get("center") or (16.0, 47.3)
    cx, cy = geoproj.project(E.meta, lon_c, lat_c)
    ppu = W / 4600.0 * float(kw.get("scale", 1.0))
    ox, oy = kw.get("offset", (0.0, 0.0))
    view = _View(W, H, float(cx), float(cy), ppu, ox * W, oy * H)
    nato, pact = _blocs(year)
    nato = [c for c in nato if c in E.feats]
    pact = [c for c in pact if c in E.feats]

    def fpolys(ids):
        return [p for i in ids for p in E.feats[i].polys]

    def build_base():
        cv = _Canvas(W, H)
        fm = _SS(W, H, 1)
        _ss_polys(fm, E.allp, view)
        cv.paint(fm.mask(), STEEL, 0.06)
        lm = _SS(W, H, 2)
        _ss_outlines(lm, E.allp, view, 0.9 * u)
        cv.paint(lm.mask(), STEEL, 0.33)
        _hud(cv, view, E.grat, "HLADNI RAT // 1945–1991", f"EVROPA {year}  ·  LAMBERT AZIMUTHAL EQUAL-AREA",
             (500, 1.0), kw.get("grid", True), kw.get("hud", True), gkey="eu", grid_op=0.12)
        if kw.get("hud", True):
            _text(cv, "← ZAPAD / WEST", "mono", 14 * u, 64 * u, H * 0.5, WHITE, 0.55, track=0.2)
            _text(cv, "ISTOK / EAST →", "mono", 14 * u, W - 64 * u, H * 0.5, WHITE, 0.55, align="r", track=0.2)
        return cv.snapshot()

    def build_masks():
        out = {}
        for nm, ids in (("nato", nato), ("pact", pact)):
            f = _SS(W, H, 1)
            _ss_polys(f, fpolys(ids), view)
            o = _SS(W, H, 2)
            _ss_outlines(o, fpolys(ids), view, 1.0 * u)
            fmk = f.mask()
            out[nm] = (fmk, o.mask(), fmk.resize((-(-W // 4), -(-H // 4)), Image.BILINEAR))
        hatch = Image.new("L", (W, H), 0)
        hd = ImageDraw.Draw(hatch)
        step = max(4, int(round(9 * u)))
        for x in range(-H, W, step):
            hd.line((x, H, x + H, 0), fill=255, width=max(1, int(round(1.6 * u))))
        out["hatch"] = ImageChops.multiply(hatch, out["pact"][0])
        rings = [view.px(r) for rs, bb in E.feats["YUG"].polys for r in rs[:1]]
        out["yug"] = (rings, view.px(E.feats["YUG"].label))
        return out

    bkey = ("eu_base", view.key(), year, kw.get("grid", True), kw.get("hud", True))
    base = _cache_get(bkey, build_base)
    M = _cache_get(("eu_masks", view.key(), year), build_masks)
    f0, f1 = kw.get("flood", (0.05, 0.42))
    qn = _seg(p, f0, f1)
    qp = _seg(p, f0 + 0.04, f1 + 0.04)
    BLOCS = {"nato": (True, NATO_FILL, 0.82, ICE), "pact": (False, DEEP, 0.92, RED)}

    def paint_bloc(cv, nm, box=None, grad=None):
        left, fcol, fop, lcol = BLOCS[nm]
        fm, om, _ = M[nm]
        layers = [(fm, fcol, fop)]
        if nm == "pact":
            layers.append((M["hatch"], RED, 0.55))
        layers.append((om, _hot(lcol, 0.25), 0.9))
        for m, col, op in layers:
            if box is None:
                cv.paint(m, col, op)
            else:
                cv.paint(ImageChops.multiply(m.crop(box), grad), col, op, box[:2])

    def snap(names):
        def build():
            c = _Canvas(W, H, base=base)
            for nm in names:
                paint_bloc(c, nm)
            return c.snapshot()
        return _cache_get(bkey + tuple(names), build)

    in_snap = ["nato", "pact"] if (qn >= 1 and qp >= 1) else (["nato"] if qn >= 1 else [])
    if in_snap:
        cv = _Canvas(W, H, _col(kw.get("bg")), base=snap(in_snap))
    else:
        cv = _Canvas(W, H, _col(kw.get("bg")), base=base, op=_eo3(_seg(p, 0.0, 0.08)))
    glow = _Glow(W, H)
    soft = 0.1 * W
    xsl = np.arange(glow.w, dtype=np.float32) * glow.ds

    for nm, q in (("nato", qn), ("pact", qp)):
        if q <= 0:
            continue
        left, fcol, fop, lcol = BLOCS[nm]
        if left:
            xf = -soft + (W + 2 * soft) * _eo3(q)
        else:
            xf = W + soft - (W + 2 * soft) * _eo3(q)
        if nm in in_snap:
            pass
        elif q >= 1:
            paint_bloc(cv, nm)
        else:
            if left:
                x0, x1 = 0, int(min(W, max(1, math.ceil(xf))))
                g = np.clip((xf - np.arange(x0, x1, dtype=np.float32)) / soft, 0, 1)
            else:
                x0, x1 = int(max(0, min(W - 1, math.floor(xf)))), W
                g = np.clip((np.arange(x0, x1, dtype=np.float32) - xf) / soft, 0, 1)
            if x1 > x0:
                grad = Image.fromarray((g * 255).astype(np.uint8)[None]).resize((x1 - x0, H), Image.NEAREST)
                paint_bloc(cv, nm, (x0, 0, x1, H), grad)
            band = np.exp(-((xsl - xf) / (0.012 * W)) ** 2)
            bi = Image.fromarray((band * 255).astype(np.uint8)[None]).resize((glow.w, glow.h), Image.NEAREST)
            glow.put(ImageChops.multiply(M[nm][2], bi), lcol, 6 * u, 0.45 * gk)
        if labels and kw.get("bloc_labels", True):
            if nm == "nato":
                lx, ly = view.px(np.array(geoproj.project(E.meta, 1.5, 47.5)))
                name, cap, en = "NATO", "SEVERNOATLANTSKI SAVEZ · 1949", "NORTH ATLANTIC TREATY"
            else:
                lx, ly = view.px(np.array(geoproj.project(E.meta, 31.0, 52.0)))
                name, cap, en = "VARŠAVSKI PAKT", "VARŠAVSKI UGOVOR · 1955", "WARSAW PACT"
            k = _seg(xf, lx - 0.05 * W, lx + 0.05 * W) if left else _seg(-xf, -lx - 0.05 * W, -lx + 0.05 * W)
            if k > 0:
                _text(cv, name, "oswald", 54 * u, lx, ly, WHITE, k, align="m", reveal=_eo3(k),
                      glow=glow, glow_col=lcol, glow_r=12 * u, glow_k=0.5)
                _text(cv, cap, "mono", 13 * u, lx, ly + 24 * u, _hot(lcol, 0.4), 0.9 * k, align="m", track=0.2)
                _text(cv, en, "mono", 11 * u, lx, ly + 42 * u, STEEL, 0.8 * k, align="m", track=0.25)

    # Yugoslavia punches through
    rings, c = M["yug"]
    pu = float(kw.get("punch", 0.5))
    ts = (p - pu) * dur
    if ts < 0:
        om = _SS(W, H, 2, _bbp(rings, 4))
        for r in rings:
            om.line(r, 1.0 * u)
        om.paint(cv, WHITE, 0.25 + 0.2 * _seg(p, pu - 0.1, pu))
    else:
        S = 0.35 + 0.65 * _eback(_seg(ts, 0, 0.35), 2.2)
        rs = [(r - c) * S + c for r in rings]
        fl = _pulse(ts, 0.1)
        m = _SS(W, H, 2, _bbp(rs, 4))
        for r in rs:
            m.poly(r)
        m.paint(cv, _mix(GOLD, (1, 1, 1), fl), 1.0)
        o = _SS(W, H, 2, _bbp(rs, 4))
        for r in rs:
            o.line(r, 2.0 * u)
        o.paint(cv, WHITE, 1.0)
        gpoly = glow.polys(rs)
        glow.put(glow.lines(rs, 6), GOLD, 8 * u, (1.0 + 1.5 * fl) * gk)
        glow.put(gpoly, GOLD, 40 * u, (0.55 + 2.2 * fl) * gk)
        if ts < 0.7:  # shock ring + speed streaks
            rr = (20 + 520 * _eo3(ts / 0.7)) * u
            op = (1 - ts / 0.7) ** 1.4
            sw = _SS(W, H, 2, (c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr))
            sw.circle(c[0], c[1], rr, outline=255, width=2.2 * u)
            for j in range(16):
                a = j / 16 * 2 * math.pi + 0.2
                r0, r1 = rr * 0.55, rr * 0.9
                sw.line(np.array([[c[0] + math.cos(a) * r0, c[1] + math.sin(a) * r0],
                                  [c[0] + math.cos(a) * r1, c[1] + math.sin(a) * r1]]), 1.4 * u)
            sw.paint(cv, _hot(GOLD, 0.5), op)
            sg, sd = glow.new()
            sd.ellipse(((c[0] - rr) / 4, (c[1] - rr) / 4, (c[0] + rr) / 4, (c[1] + rr) / 4), outline=255, width=2)
            glow.put(sg, GOLD, 12 * u, op * gk)
        if labels:
            lab = kw.get("label", "IZMEĐU ISTOKA I ZAPADA")
            if lab:
                k = _seg(ts, 0.12, 0.5)
                y = H - 118 * u
                _text(cv, lab, "anton", 84 * u, W / 2, y, WHITE, 1.0, align="m", reveal=_eo5(k),
                      glow=glow, glow_col=GOLD, glow_r=16 * u, glow_k=0.35)
                cap = kw.get("caption", "BETWEEN EAST AND WEST")
                if cap:
                    _text(cv, cap, "mono", 19 * u, W / 2, y + 36 * u, GOLD, _seg(ts, 0.3, 0.6), align="m", track=0.3)
    glow.apply(cv)
    return cv.array(scan=kw.get("scan", False))


# ================================================================ preview ===
def _composite(rgba, bg=(0.02, 0.02, 0.03)):
    rgb = np.empty(rgba.shape[:2] + (3,), np.float32)
    rgb[:] = bg
    a = rgba[..., 3:4]
    rgb = rgb * (1 - a) + rgba[..., :3] * a
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def _grid_sheet(ims, name, cols=3):
    W, H = ims[0].size
    rows = -(-len(ims) // cols)
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 4, rows * H + (rows - 1) * 4), (60, 60, 60))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * (W + 4), (i // cols) * (H + 4)))
    out = os.path.join(_ROOT, "out", "preview")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"maps_{name}.png")
    sheet.save(path)
    return path


def _sheet(fn, name, samples, W=960, H=540, **kw):
    """Render samples [(t, dur, extra_kw)] into a contact sheet on a dark bg."""
    ims = []
    for (t, dur, ex) in samples:
        im = _composite(fn(t, dur, W, H, **dict(kw, **ex)))
        lab = f"t={t:.2f}/{dur:.2f}" + (f"  {ex}" if ex else "")
        ImageDraw.Draw(im).text((8, H - 8), lab, font=font("mono", 12), fill=(255, 255, 0), anchor="ls")
        ims.append(im)
    return _grid_sheet(ims, name)


def _time(fn, n=10, W=1920, H=1080, dur=2.0, **kw):
    """(first-call ms, mean ms, max ms) over n frames spread across the shot."""
    t0 = time.perf_counter()
    fn(0.37 * dur, dur, W, H, **kw)
    cold = (time.perf_counter() - t0) * 1000
    ts = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        t0 = time.perf_counter()
        fn(t, dur, W, H, **kw)
        ts.append((time.perf_counter() - t0) * 1000)
    return cold, float(np.mean(ts)), float(np.max(ts))


PREVIEWS = {  # name: (fn, kw, sample phases, dur)
    "draw": (yugo_map, dict(mode="draw"), [0.1, 0.3, 0.5, 0.64, 0.8, 1.0], 3.0),
    "assemble": (yugo_map, dict(mode="assemble"), [0.12, 0.3, 0.48, 0.56, 0.62, 0.95], 2.0),
    "republics": (yugo_map, dict(mode="republics"), [0.03, 0.22, 0.4, 0.55, 0.72, 0.9], 6.0),
    "neighbors": (yugo_map, dict(mode="neighbors"), [0.1, 0.25, 0.4, 0.6, 0.8, 1.0], 2.0),
    "capitals": (yugo_map, dict(mode="capitals"), [0.1, 0.3, 0.5, 0.7, 0.85, 1.0], 4.0),
    "tilt3d": (yugo_map, dict(mode="tilt3d"), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 4.0),
    "world": (world_nam, dict(), [0.05, 0.2, 0.34, 0.5, 0.7, 0.95], 6.0),
    "europe": (europe_blocs, dict(), [0.1, 0.22, 0.36, 0.52, 0.62, 0.9], 4.0),
}


def _variants():
    """Parameter smoke-test sheet."""
    tiles = [
        (yugo_map, 2.4, 3.0, dict(mode="draw", scale=2.2, center=(20.46, 44.82), bg="navy")),
        (yugo_map, 1.7, 2.0, dict(mode="assemble", fill=["red", "gold", "white", "red", "gold", "white"],
                                  hud=False)),
        (yugo_map, 0.5, 1.0, dict(mode="republics", hi=2, local=0.6, offset=(0.0, 0.0), scale=1.0)),
        (yugo_map, 1.0, 2.0, dict(mode="tilt3d", yaw=25, pitch=35, extrude=0.16, bg="black")),
        (world_nam, 5.0, 6.0, dict(bg="navy", observers=True, lift=0.12)),
        (europe_blocs, 3.0, 4.0, dict(year=1960, scan=True)),
    ]
    ims = []
    for fn, t, dur, kw in tiles:
        im = _composite(fn(t, dur, 960, 540, **kw))
        ImageDraw.Draw(im).text((8, 532), f"{fn.__name__} {kw}"[:120], font=font("mono", 11),
                                fill=(255, 255, 0), anchor="ls")
        ims.append(im)
    return _grid_sheet(ims, "variants")


def main(argv):
    which = argv[1:] or list(PREVIEWS) + ["variants"]
    for name in which:
        if name == "variants":
            print(f"{'variants':10s} -> {os.path.relpath(_variants(), _ROOT)}")
            continue
        fn, kw, ps, dur = PREVIEWS[name]
        path = _sheet(fn, name, [(pp * dur, dur, {}) for pp in ps], **kw)
        cold, mean, mx = _time(fn, dur=dur, **kw)
        print(f"{name:10s} mean {mean:6.1f} ms  max {mx:6.1f} ms  (first call {cold:6.0f} ms)  @1920x1080"
              f"  -> {os.path.relpath(path, _ROOT)}")


if __name__ == "__main__":
    main(sys.argv)
