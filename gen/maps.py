"""Map generators for the SFRJ hype edit (motion-graphics cartography).

Every public generator is ``fn(t, dur, W, H, **params) -> float32 RGBA (H, W, 4)``
(straight alpha, transparent background unless ``bg=`` is given), deterministic
and cached so that a 1920x1080 frame renders in well under ~120 ms.

Public API (see the docstrings for all parameters)::

    yugo_map(t, dur, W, H, mode="draw" | "assemble" | "republics" | "neighbors"
                                | "capitals" | "tilt3d", **kw)
    world_nam(t, dur, W, H, **kw)       # 1961 Belgrade Non-Aligned conference
    europe_blocs(t, dur, W, H, **kw)    # NATO / Warsaw Pact / SFRJ in between

Geometry comes from ``assets/geo/*.json`` (built once by ``gen/geo_build.py``
from Natural Earth).  Rendering: PIL ImageDraw into L masks (2x supersampled
where it matters), glows from blurred quarter-resolution masks, composited
premultiplied in 8 bit and converted to float32 straight alpha at the end.

    python3 gen/maps.py          # contact sheets -> out/preview/maps_*.png + timings
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
ICE = (0.42, 0.66, 1.0)  # cold NATO blue (lines / glow)


def _col(c):
    """Colour spec -> float rgb tuple.  Accepts PAL names, '#hex', 0-1 floats or 0-255 ints."""
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
    """A 'hot' core version of a colour (towards white) for glowing lines."""
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
    """1 at x=0 decaying exponentially (x>=0), 0 before."""
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
    D.rep_borders = [_arr(l) for l in d["rep_borders"]]
    D.prov_borders = [_arr(l) for l in d["prov_borders"]]
    D.yugo = _polys(d["yugo"])
    D.main = _arr(d["outline"]["main"])
    D.main_cum = _cum(D.main)
    D.islands = [_arr(r) for r in d["outline"]["islands"]]
    # where along the main ring each island "attaches" (for the draw-on reveal)
    att = []
    for isl in D.islands:
        c = isl.mean(axis=0)
        k = int(np.argmin(((D.main - c) ** 2).sum(1)))
        att.append(D.main_cum[k] / D.main_cum[-1])
    D.island_at = np.array(att)
    D.coarse = _polys(d["coarse"]["yugo"])
    D.coarse_borders = [_arr(l) for l in d["coarse"]["rep_borders"]]
    D.nbrs = [_NS(id=n["id"], name=n["name"], en=n["en"], polys=_polys(n["p"]),
                  label=np.array(n["label"], float)) for n in d["neighbors"]]
    D.context = [_polys(c["p"]) for c in d["context"]]
    D.context_flat = [p for c in D.context for p in c]
    D.lakes = _polys(d["lakes"])
    D.rivers = [_arr(r["l"]) for r in d["rivers"]]
    D.cities = {c["name"]: _NS(name=c["name"], today=c["today"], lon=c["lon"], lat=c["lat"],
                               kind=c["kind"], xy=np.array(c["xy"], float)) for c in d["cities"]}
    D.bounds = tuple(float(v) for v in d["meta"]["bounds"])
    return D


def _yu_lonlat(xy):
    """Stored LCC units -> lon, lat (for HUD read-outs)."""
    p = dict(_yu().meta["proj"])
    p.pop("type")
    s = _yu().meta["scale"]
    return geoproj.lcc_inv(np.asarray(xy[..., 0]) / s, np.asarray(xy[..., 1]) / s, **p)


# ================================================================== view ====
class _View:
    """Stored-units -> pixel transform: p_px = (p - c) * ppu (y flipped) + (W/2, H/2) + off."""

    def __init__(self, W, H, cx, cy, ppu, ox=0.0, oy=0.0):
        self.W, self.H = W, H
        self.cx, self.cy, self.ppu = float(cx), float(cy), float(ppu)
        self.ox, self.oy = W * 0.5 + ox, H * 0.5 + oy
        hw, hh = self.ox / ppu, self.oy / ppu
        hw2, hh2 = (W - self.ox) / ppu, (H - self.oy) / ppu
        self.vb = (cx - hw, cy - hh2, cx + hw2, cy + hh)

    def key(self):
        return (self.W, self.H, round(self.cx, 2), round(self.cy, 2), round(self.ppu, 7),
                round(self.ox, 2), round(self.oy, 2))

    def px(self, a, ss=1):
        a = np.asarray(a, dtype=np.float64)
        out = np.empty(a.shape, dtype=np.float64)
        out[..., 0] = ((a[..., 0] - self.cx) * self.ppu + self.ox) * ss
        out[..., 1] = ((self.cy - a[..., 1]) * self.ppu + self.oy) * ss
        return out

    def unpx(self, x, y):
        return (x - self.ox) / self.ppu + self.cx, self.cy - (y - self.oy) / self.ppu

    def visible(self, bb, pad_px=4):
        pad = pad_px / self.ppu
        v = self.vb
        return not (bb[2] < v[0] - pad or bb[0] > v[2] + pad or bb[3] < v[1] - pad or bb[1] > v[3] + pad)


def _make_view(W, H, bounds, fit=(0.86, 0.84), scale=1.0, center=None, offset=(0.0, 0.0),
               meta=None):
    x0, y0, x1, y1 = bounds
    ppu = min(W * fit[0] / (x1 - x0), H * fit[1] / (y1 - y0)) * scale
    if center is None:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    elif meta is not None and abs(center[0]) <= 360 and abs(center[1]) <= 90:
        cx, cy = geoproj.project(meta, center[0], center[1])
        cx, cy = float(cx), float(cy)
    else:
        cx, cy = center
    return _View(W, H, cx, cy, ppu, offset[0] * W, offset[1] * H)


# ============================================================== raster =====
def _lw(w, ss):
    return max(1, int(round(w * ss)))


def _draw_polys(dr, polys, view, ss, fill, hole=0):
    for rings, bb in polys:
        if not view.visible(bb):
            continue
        dr.polygon(view.px(rings[0], ss).ravel().tolist(), fill=fill)
        for h in rings[1:]:
            dr.polygon(view.px(h, ss).ravel().tolist(), fill=hole)


def _draw_poly_outlines(dr, polys, view, ss, fill, width, holes=True):
    w = _lw(width, ss)
    for rings, bb in polys:
        if not view.visible(bb):
            continue
        for r in (rings if holes else rings[:1]):
            dr.line(view.px(r, ss).ravel().tolist(), fill=fill, width=w, joint="curve" if w > 2 else None)


def _draw_lines(dr, lines, view, ss, fill, width, px=False):
    w = _lw(width, ss)
    for a in lines:
        p = a * ss if px else view.px(a, ss)
        if len(p) >= 2:
            dr.line(p.ravel().tolist(), fill=fill, width=w, joint="curve" if w > 2 else None)


def _partial(pts, cum, s):
    """Polyline pts (px) cut at arc length s."""
    if s <= 0:
        return pts[:1]
    if s >= cum[-1]:
        return pts
    k = int(np.searchsorted(cum, s))
    f = (s - cum[k - 1]) / max(cum[k] - cum[k - 1], 1e-9)
    tip = pts[k - 1] + (pts[k] - pts[k - 1]) * f
    return np.vstack([pts[:k], tip[None]])


def _dashes(pts, on, off, phase=0.0):
    """Split a pixel polyline into dash sub-polylines."""
    cum = _cum(pts)
    L = cum[-1]
    out = []
    s = -phase % (on + off)
    s -= (on + off)
    while s < L:
        a, b = max(s, 0.0), min(s + on, L)
        if b > a:
            i0 = int(np.searchsorted(cum, a))
            i1 = int(np.searchsorted(cum, b))
            pa = np.interp(a, cum, pts[:, 0]), np.interp(a, cum, pts[:, 1])
            pb = np.interp(b, cum, pts[:, 0]), np.interp(b, cum, pts[:, 1])
            mid = pts[i0:i1]
            out.append(np.vstack([[pa], mid, [pb]]))
        s += on + off
    return out


class _SS:
    """A supersampled L mask with a draw handle."""

    def __init__(self, W, H, ss=2):
        self.W, self.H, self.ss = W, H, ss
        self.im = Image.new("L", (W * ss, H * ss), 0)
        self.dr = ImageDraw.Draw(self.im)

    def get(self):
        return self.im.reduce(self.ss) if self.ss > 1 else self.im


@functools.lru_cache(maxsize=512)
def _lut_scale(k255):
    k = k255 / 255.0
    return [int(round(v * k)) for v in range(256)]


def _scale_mask(m, k):
    if k >= 0.998:
        return m
    return m.point(_lut_scale(int(round(_cl(k) * 255))))


class _Canvas:
    """Premultiplied 8-bit compositor: C (RGB, premultiplied) + A (L)."""

    def __init__(self, W, H, bg=None, base=None, op=1.0):
        self.W, self.H = W, H
        if base is not None:
            if op >= 0.998:
                self.C, self.A = base[0].copy(), base[1].copy()
            else:
                lut = _lut_scale(int(round(_cl(op) * 255)))
                self.C, self.A = base[0].point(lut * 3), base[1].point(lut)
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
        x0, y0, x1, y1 = bb[0] + x, bb[1] + y, bb[2] + x, bb[3] + y
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.W), min(y1, self.H)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        m = m.crop((cx0 - x, cy0 - y, cx1 - x, cy1 - y))
        box = (cx0, cy0, cx1, cy1)
        self.C.paste(_u8(col), box, m)
        self.A.paste(255, box, m)

    def add(self, rgb, a, box=None):
        """Additive (premultiplied) light, e.g. glows. rgb/a sized to box (or full)."""
        if box is None:
            self.C = ImageChops.add(self.C, rgb)
            self.A = ImageChops.screen(self.A, a)
        else:
            c = ImageChops.add(self.C.crop(box), rgb)
            al = ImageChops.screen(self.A.crop(box), a)
            self.C.paste(c, box[:2])
            self.A.paste(al, box[:2])

    def array(self, scan=False):
        im = Image.merge("RGBa", (*self.C.split(), self.A)).convert("RGBA")
        arr = np.asarray(im)
        out = np.empty(arr.shape, np.float32)
        np.multiply(arr, np.float32(1.0 / 255.0), out=out)
        if scan:
            out[1::3, :, 3] *= np.float32(0.55)
            out[2::3, :, 3] *= np.float32(0.85)
        return out


class _Glow:
    """Quarter-resolution glow accumulator.  Draw into ``mask()`` handles (coords in
    full-res px, scaled here), blur, colourise, then ``apply`` additively."""

    def __init__(self, W, H, ds=4):
        self.W, self.H, self.ds = W, H, ds
        self.w, self.h = -(-W // ds), -(-H // ds)
        self.rgb = None
        self.a = None

    def new(self):
        im = Image.new("L", (self.w, self.h), 0)
        return im, ImageDraw.Draw(im)

    def put(self, m, col, radius_px, strength=1.0):
        """Add a low-res mask m blurred by radius_px (full-res px) in colour col."""
        if m is None or strength <= 0.003:
            return
        if m.getbbox() is None:
            return
        r = max(radius_px / self.ds, 0.6)
        b = m.filter(ImageFilter.GaussianBlur(r))
        k = strength
        rgb = Image.merge("RGB", [b.point(_lut_gain(c * k)) for c in col])
        a = b.point(_lut_gain(k))
        if self.rgb is None:
            self.rgb, self.a = rgb, a
        else:
            self.rgb = ImageChops.add(self.rgb, rgb)
            self.a = ImageChops.screen(self.a, a)

    def sprite(self, m_full, x, y):
        """Downsample a full-res sprite mask into a low-res mask positioned at (x, y)."""
        im, _ = self.new()
        w, h = m_full.size
        sw, sh = max(1, int(round(w / self.ds))), max(1, int(round(h / self.ds)))
        im.paste(m_full.resize((sw, sh), Image.BILINEAR), (int(round(x / self.ds)), int(round(y / self.ds))))
        return im

    def apply(self, cv: _Canvas):
        if self.rgb is None:
            return
        bb = self.a.getbbox()
        if bb is None:
            return
        ds = self.ds
        pad = 1
        bx0, by0 = max(bb[0] - pad, 0), max(bb[1] - pad, 0)
        bx1, by1 = min(bb[2] + pad, self.w), min(bb[3] + pad, self.h)
        X0, Y0 = bx0 * ds, by0 * ds
        X1, Y1 = min(bx1 * ds, self.W), min(by1 * ds, self.H)
        if X1 <= X0 or Y1 <= Y0:
            return
        src = (bx0, by0, bx0 + (X1 - X0) / ds, by0 + (Y1 - Y0) / ds)
        size = (X1 - X0, Y1 - Y0)
        rgb = self.rgb.resize(size, Image.BILINEAR, box=src)
        a = self.a.resize(size, Image.BILINEAR, box=src)
        cv.add(rgb, a, (X0, Y0, X1, Y1))


@functools.lru_cache(maxsize=1024)
def _lut_gain_q(kq):
    k = kq / 64.0
    return [min(255, int(v * k + 0.5)) for v in range(256)]


def _lut_gain(k):
    return _lut_gain_q(int(round(max(k, 0.0) * 64)))


# ================================================================ text =====
@functools.lru_cache(maxsize=2048)
def _tmask(txt, fname, size, track=0.0):
    """Render text -> (L mask, ox, oy, advance, cap_height).  (ox, oy) is the mask's
    top-left relative to the left end of the baseline."""
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


def _text(cv, txt, fname, size, x, y, col, op=1.0, align="l", valign="base", track=0.0,
          reveal=1.0, glow=None, glow_col=None, glow_r=10.0, glow_k=0.8):
    """Draw text; (x, y) is the anchor.  align l/m/r, valign base/cap/mid/top.
    reveal<1 wipes the text in from the left.  Returns the pixel bbox."""
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
    elif valign == "top":
        y -= oy
    px, py = x + ox, y + oy
    if reveal < 1.0:
        w = max(1, int(m.size[0] * _cl(reveal)))
        m = m.crop((0, 0, w, m.size[1]))
    cv.paint(m, col, op, (px, py))
    if glow is not None:
        glow.put(glow.sprite(m, px, py), glow_col or col, glow_r, glow_k * op)
    return (px, py, px + m.size[0], py + m.size[1])


# ================================================================= HUD =====
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
    if kind == "lat":
        return f"{abs(v):g}°{'N' if v >= 0 else 'S'}"
    return f"{abs(v):g}°{'E' if v >= 0 else 'W'}"


def _edge_hits(pts, W, H, m):
    """Where a pixel polyline crosses the inset frame (left edge for parallels,
    bottom edge for meridians).  Returns list of (edge, pos)."""
    hits = []
    x, y = pts[:, 0], pts[:, 1]
    # bottom edge
    s = np.sign(y - (H - m))
    idx = np.nonzero(np.diff(s) != 0)[0]
    for i in idx:
        f = ((H - m) - y[i]) / (y[i + 1] - y[i])
        xx = x[i] + (x[i + 1] - x[i]) * f
        if m < xx < W - m:
            hits.append(("b", xx))
    s = np.sign(x - m)
    idx = np.nonzero(np.diff(s) != 0)[0]
    for i in idx:
        f = (m - x[i]) / (x[i + 1] - x[i])
        yy = y[i] + (y[i + 1] - y[i]) * f
        if m < yy < H - m:
            hits.append(("l", yy))
    return hits


_BASE_CACHE: dict = {}


def _cache_get(key, fn):
    v = _BASE_CACHE.get(key)
    if v is None:
        if len(_BASE_CACHE) > 24:
            _BASE_CACHE.pop(next(iter(_BASE_CACHE)))
        v = fn()
        _BASE_CACHE[key] = v
    return v


def _hud_masks(view, grat, title, sub, km_unit=None, grid=True, hud=True, globe=None):
    """(grid_mask, hud_mask) for a view, cached."""
    key = ("hud", view.key(), title, sub, km_unit, grid, hud, id(grat))

    def build():
        W, H = view.W, view.H
        u = H / 1080.0
        g = _SS(W, H, 2)
        hm = Image.new("L", (W, H), 0)
        hd = ImageDraw.Draw(hm)
        m = int(round(28 * u))
        hits = []
        for kind, val, a in grat:
            p = view.px(a)
            if grid:
                g.dr.line((p * 2).ravel().tolist(), fill=255, width=_lw(1.0 * u, 2))
            for e, pos in _edge_hits(p, W, H, m):
                hits.append((kind, val, e, pos))
        if globe is not None and grid:
            g.dr.line((view.px(globe) * 2).ravel().tolist(), fill=255, width=_lw(1.2 * u, 2))
        gm = g.get() if grid else None
        if hud:
            L = int(round(26 * u))
            lw = max(1, int(round(1.5 * u)))
            for (cx, cy, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
                hd.line([(cx, cy + sy * L), (cx, cy), (cx + sx * L, cy)], fill=255, width=lw)
            fs = max(9, int(round(14 * u)))
            f = font("mono", fs)
            for kind, val, e, pos in hits:
                if (kind == "lon" and e == "b") or (kind == "lat" and e == "l"):
                    t = _deg(val, kind)
                    if e == "b":
                        hd.line([(pos, H - m), (pos, H - m - 8 * u)], fill=255, width=lw)
                        hd.text((pos + 4 * u, H - m - 3 * u), t, font=f, fill=200, anchor="ls")
                    else:
                        hd.line([(m, pos), (m + 8 * u, pos)], fill=255, width=lw)
                        hd.text((m + 11 * u, pos - 3 * u), t, font=f, fill=200, anchor="ls")
            ft = max(9, int(round(15 * u)))
            if title:
                hd.text((m + 36 * u, m + 4 * u), title, font=font("mono", ft), fill=235, anchor="lt")
            if sub:
                hd.text((m + 36 * u, m + 24 * u), sub, font=font("mono", max(8, int(round(12 * u)))), fill=150, anchor="lt")
            if km_unit:
                km, units_per_km = km_unit
                bl = km * units_per_km * view.ppu
                x1, y1 = W - m - 36 * u, H - m - 4 * u
                x0 = x1 - bl
                hd.line([(x0, y1 - 6 * u), (x0, y1), (x1, y1), (x1, y1 - 6 * u)], fill=235, width=lw)
                hd.line([((x0 + x1) / 2, y1), ((x0 + x1) / 2, y1 - 4 * u)], fill=235, width=lw)
                hd.text((x1, y1 - 10 * u), f"{km} KM", font=font("mono", max(8, int(round(12 * u)))), fill=200, anchor="rs")
                hd.text((x0, y1 - 10 * u), "0", font=font("mono", max(8, int(round(12 * u)))), fill=200, anchor="ls")
        return gm, (hm if hud else None)

    return _cache_get(key, build)


def _hud(cv, view, grat, title, sub, km_unit=None, grid=True, hud=True, op=1.0, globe=None,
         grid_col=STEEL, grid_op=0.16, hud_col=WHITE, hud_op=0.55):
    if not (grid or hud) or op <= 0:
        return
    gm, hm = _hud_masks(view, grat, title, sub, km_unit, grid, hud, globe)
    if gm is not None:
        cv.paint(gm, grid_col, grid_op * op)
    if hm is not None:
        cv.paint(hm, hud_col, hud_op * op)


# ========================================================= yugo: common ====
@functools.lru_cache(None)
def _yu_grat():
    return _grat_lines(_yu().meta, list(range(6, 33)), list(range(34, 54)), (4, 34), (33, 54))


def _yu_view(W, H, scale=1.0, center=None, offset=(0.0, 0.0), fit=(0.86, 0.84)):
    D = _yu()
    return _make_view(W, H, D.bounds, fit, scale, center, offset, D.meta)


def _cmin(lonlat):
    lon, lat = lonlat
    return f"{abs(lat):05.2f}°{'N' if lat >= 0 else 'S'}  {abs(lon):05.2f}°{'E' if lon >= 0 else 'W'}"


def _yu_hud(cv, view, grid, hud, op=1.0, title="SFRJ // 1945–1992", sub=None):
    D = _yu()
    if sub is None:
        lon, lat = _yu_lonlat(np.array([view.cx, view.cy]))
        sub = f"LAMBERT CONFORMAL CONIC  ·  {_cmin((float(lon), float(lat)))}"
    _hud(cv, view, _yu_grat(), title, sub, (100, 10.0), grid, hud, op)


def _context_layer(W, H, view, ss=2, rivers=False, lakes=True):
    """Cached (fill_mask, line_mask, lake_mask) for the non-Yugoslav land."""
    key = ("ctx", view.key(), rivers, lakes)

    def build():
        D = _yu()
        fm = _SS(W, H, 1)
        lm = _SS(W, H, ss)
        u = H / 1080.0
        allp = D.context_flat + [p for n in D.nbrs for p in n.polys]
        _draw_polys(fm.dr, allp, view, 1, 255)
        _draw_poly_outlines(lm.dr, allp, view, ss, 255, 1.0 * u)
        rm = None
        if rivers:
            r = _SS(W, H, ss)
            _draw_lines(r.dr, D.rivers, view, ss, 255, 0.9 * u)
            rm = r.get()
        km = None
        if lakes:
            k = _SS(W, H, ss)
            _draw_polys(k.dr, D.lakes, view, ss, 255)
            km = k.get()
        return fm.get(), lm.get(), km, rm

    return _cache_get(key, build)


def _yu_masks(view, which="yugo", ss=2, width=1.0):
    """Cached fill/outline masks for Yugoslavia-level geometry."""
    key = ("yum", view.key(), which, ss, width)

    def build():
        D = _yu()
        W, H = view.W, view.H
        u = H / 1080.0
        if which == "yugo":
            f = _SS(W, H, ss)
            _draw_polys(f.dr, D.yugo, view, ss, 255)
            _draw_polys(f.dr, D.lakes, view, ss, 0)
            o = _SS(W, H, ss)
            _draw_poly_outlines(o.dr, D.yugo, view, ss, 255, width * u)
            return f.get(), o.get()
        if which == "borders":
            o = _SS(W, H, ss)
            _draw_lines(o.dr, D.rep_borders, view, ss, 255, width * u)
            return o.get()
        if which == "prov":
            o = _SS(W, H, ss)
            for ln in D.prov_borders:
                for d in _dashes(view.px(ln), 7 * u, 5 * u):
                    o.dr.line((d * ss).ravel().tolist(), fill=255, width=_lw(width * u, ss))
            return o.get()
        if which.startswith("rep"):
            i = int(which[3:])
            f = _SS(W, H, ss)
            _draw_polys(f.dr, D.reps[i].polys, view, ss, 255)
            _draw_polys(f.dr, D.lakes, view, ss, 0)
            o = _SS(W, H, ss)
            _draw_poly_outlines(o.dr, D.reps[i].polys, view, ss, 255, width * u)
            return f.get(), o.get()
        if which.startswith("nb"):
            i = int(which[2:])
            f = _SS(W, H, 1)
            _draw_polys(f.dr, D.nbrs[i].polys, view, 1, 255)
            o = _SS(W, H, ss)
            _draw_poly_outlines(o.dr, D.nbrs[i].polys, view, ss, 255, width * u)
            return f.get(), o.get()
        raise KeyError(which)

    return _cache_get(key, build)


def _low_lines(glow, view, lines, width_px, closed_polys=None):
    im, dr = glow.new()
    s = 1.0 / glow.ds
    w = max(1, int(round(width_px * s)))
    for a in lines:
        dr.line(view.px(a, s).ravel().tolist(), fill=255, width=w)
    if closed_polys:
        for rings, bb in closed_polys:
            if view.visible(bb):
                for r in rings:
                    dr.line(view.px(r, s).ravel().tolist(), fill=255, width=w)
    return im


def _low_fill(glow, view, polys):
    im, dr = glow.new()
    _draw_polys(dr, polys, view, 1.0 / glow.ds, 255)
    return im


def _ctx_paint(cv, W, H, view, op, rivers=False, col=STEEL):
    fm, lm, km, rm = _context_layer(W, H, view, rivers=rivers)
    cv.paint(fm, col, 0.045 * op)
    cv.paint(lm, col, 0.30 * op)
    if rm is not None:
        cv.paint(rm, ICE, 0.18 * op)


def _yu_underlay(W, H, view, kw, op=1.0, ctx_op=1.0):
    """Cached context + graticule + HUD snapshot as a fresh canvas (faded by op)."""
    bg = _col(kw.get("bg"))
    key = ("under", view.key(), bg, kw.get("grid", True), kw.get("hud", True), kw.get("rivers", False), ctx_op)

    def build():
        cv = _Canvas(W, H)
        _ctx_paint(cv, W, H, view, ctx_op, kw.get("rivers", False))
        _yu_hud(cv, view, kw.get("grid", True), kw.get("hud", True))
        return cv.snapshot()

    snap = _cache_get(key, build)
    if bg is None:
        return _Canvas(W, H, base=snap, op=op)
    cv = _Canvas(W, H, bg)
    m = Image.merge("RGBa", (*snap[0].split(), snap[1])).convert("RGBA")
    lut = _lut_scale(int(round(_cl(op) * 255)))
    a = snap[1].point(lut)
    cv.C.paste(m.convert("RGB"), (0, 0), a)
    return cv


def _flash_star(dr, x, y, r, rot=0.0):
    dr.polygon(star_points(x, y, r, rot=rot), fill=255)


# ============================================================ yugo modes ====
def _mode_draw(t, dur, W, H, p, view, kw):
    """Glowing outline draws itself on; spark at the head; internal borders follow."""
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    end = float(kw.get("reveal_end", 0.62))
    cv = _yu_underlay(W, H, view, kw, _eo3(_seg(p, 0.0, 0.25)))
    glow = _Glow(W, H)

    q = _eio(_seg(p, 0.02, end))
    done = p >= end
    # fill wash after completion
    fa = float(kw.get("fill_alpha", 0.2)) * _eo3(_seg(p, end, end + 0.25))
    if fa > 0:
        fm, _ = _yu_masks(view, "yugo")
        cv.paint(fm, _mix(fillc, DEEP, 0.4), fa)

    # the outline
    pts = view.px(D.main)
    cum = D.main_cum * view.ppu
    s = q * cum[-1]
    part = _partial(pts, cum, s)
    lw = float(kw.get("width", 1.8)) * u
    ssm = _SS(W, H, 2)
    ssm.dr.line((part * 2).ravel().tolist(), fill=255, width=_lw(lw, 2), joint="curve")
    isl_op = []
    for isl, at in zip(D.islands, D.island_at):
        a = _seg(q, at, at + 0.03)
        if a > 0:
            ssm.dr.line(view.px(isl, 2).ravel().tolist(), fill=int(255 * a), width=_lw(lw * 0.8, 2))
            isl_op.append((isl, a))
    line_m = ssm.get()
    core = _hot(stroke, 0.35)
    cv.paint(line_m, core, 1.0)

    # glow of the drawn part
    lg, ld = glow.new()
    ld.line((part / glow.ds).ravel().tolist(), fill=255, width=max(1, int(round(lw * 1.2 / glow.ds + 0.5))))
    for isl, a in isl_op:
        ld.line(view.px(isl, 1 / glow.ds).ravel().tolist(), fill=int(255 * a), width=1)
    boost = 1.0 + 1.6 * _pulse(p - end, 0.05) if done else 1.0
    glow.put(lg, stroke, 5 * u, 1.3 * gk * boost)
    glow.put(lg, stroke, 22 * u, 0.9 * gk * boost)

    # internal borders + provinces
    qb = _eo3(_seg(p, end - 0.02, end + 0.22))
    if qb > 0:
        bm = _SS(W, H, 2)
        for ln in D.rep_borders:
            lp = view.px(ln)
            c = _cum(lp)
            bm.dr.line((_partial(lp, c, qb * c[-1]) * 2).ravel().tolist(), fill=255, width=_lw(1.1 * u, 2))
        cv.paint(bm.get(), stroke, 0.85)
    qp = _seg(p, end + 0.1, end + 0.3)
    if qp > 0:
        cv.paint(_yu_masks(view, "prov", width=1.0), stroke, 0.7 * qp)

    # the spark
    if 0.0 < q < 1.0 or (done and p < end + 0.06):
        hx, hy = part[-1]
        sp_op = 1.0 if not done else 1.0 - _seg(p, end, end + 0.06)
        sg, sd = glow.new()
        r = 3.2 * u / glow.ds
        sd.ellipse((hx / glow.ds - r, hy / glow.ds - r, hx / glow.ds + r, hy / glow.ds + r), fill=255)
        fl = 70 * u / glow.ds
        sd.line((hx / glow.ds - fl, hy / glow.ds, hx / glow.ds + fl, hy / glow.ds), fill=120, width=1)
        glow.put(sg, _hot(stroke, 0.6), 6 * u, 2.2 * sp_op)
        glow.put(sg, stroke, 26 * u, 1.4 * sp_op)
        spm = Image.new("L", (int(40 * u) + 1, int(40 * u) + 1), 0)
        sdd = ImageDraw.Draw(spm)
        c0 = spm.size[0] / 2
        rr = 2.6 * u
        sdd.ellipse((c0 - rr, c0 - rr, c0 + rr, c0 + rr), fill=255)
        sdd.line((c0 - 18 * u, c0, c0 + 18 * u, c0), fill=160, width=1)
        sdd.line((c0, c0 - 18 * u, c0, c0 + 18 * u), fill=160, width=1)
        cv.paint(spm, (1, 1, 1), sp_op, (hx - c0, hy - c0))
        if kw.get("labels", True) and kw.get("hud", True):
            lon, lat = _yu_lonlat(D.main[min(int(np.searchsorted(cum, s)), len(D.main) - 1)])
            _text(cv, _cmin((float(lon), float(lat))), "mono", 13 * u, hx + 16 * u, hy - 12 * u,
                  WHITE, 0.85 * sp_op, track=0.08)
    glow.apply(cv)
    return cv


def _slot(p, n, a=0.0, b=1.0):
    """Which of n equal slots p falls in, plus local progress."""
    x = _seg(p, a, b) * n
    i = min(int(x), n - 1)
    return i, x - i


def _big_name(cv, glow, W, H, name, en, idx_txt, local, kw, col=WHITE, accent=RED, anchor=None):
    """Big condensed name block, lower-left, with wipe-in; returns top-right anchor."""
    u = H / 1080.0
    m = 70 * u
    maxw = W * 0.42
    size = 150 * u
    fn = kw.get("name_font", "anton")
    lines = [name]
    _, _, _, adv, _ = _tmask(name, fn, int(round(size)))
    if adv > maxw and " I " in name:
        a, b = name.split(" I ", 1)
        lines = [a + " I", b]
    adv = max(_tmask(l, fn, int(round(size)))[3] for l in lines)
    if adv > maxw:
        size *= maxw / adv
    rv = _eo5(_seg(local, 0.0, 0.18))
    lh = size * 1.02
    yb = H - m - 58 * u
    ys = [yb - lh * (len(lines) - 1 - i) for i in range(len(lines))]
    top = ys[0] - _tmask(lines[0], fn, int(round(size)))[4]
    slide = (1 - rv) * 40 * u
    for l, y in zip(lines, ys):
        _text(cv, l, fn, size, m - slide, y, col, 1.0, reveal=rv, glow=glow, glow_col=accent,
              glow_r=18 * u, glow_k=0.35)
    rc = _seg(local, 0.08, 0.3)
    _text(cv, en, "mono", 22 * u, m, yb + 42 * u, WHITE, 0.8 * rc, track=0.18, reveal=_eo3(rc * 1.4))
    if idx_txt:
        _text(cv, idx_txt, "mono", 20 * u, m, top - 20 * u, accent, rc, track=0.2)
        cv.paint(_rect_mask(int(max(2, 44 * u * _eo3(rc))), max(1, int(2 * u))), accent, 1.0,
                 (m + 100 * u, top - 28 * u))
    return (m + adv * rv, top)


@functools.lru_cache(maxsize=64)
def _rect_mask(w, h):
    return Image.new("L", (max(1, w), max(1, h)), 255)


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
    elif hi is not None:
        local = p if kw.get("local") is None else kw["local"]
    labels = kw.get("labels", True)

    bkey = ("rep_base", view.key(), kw.get("bg"), kw.get("grid", True), kw.get("hud", True), stroke, fillc)

    def base():
        cv = _Canvas(W, H, _col(kw.get("bg")))
        _ctx_paint(cv, W, H, view, 1.0, kw.get("rivers", False))
        _yu_hud(cv, view, kw.get("grid", True), kw.get("hud", True))
        fm, om = _yu_masks(view, "yugo")
        cv.paint(fm, _mix(fillc, DEEP, 0.55), 0.35)
        cv.paint(_yu_masks(view, "borders", width=1.0), stroke, 0.55)
        cv.paint(_yu_masks(view, "prov", width=0.9), stroke, 0.35)
        cv.paint(om, stroke, 0.75)
        g = _Glow(W, H)
        g.put(_low_lines(g, view, [], 2 * u, D.yugo), stroke, 18 * u, 0.35 * gk)
        g.apply(cv)
        return cv.snapshot()

    cv = _Canvas(W, H, base=_cache_get(bkey, base))
    if hi is None:
        return cv
    hi = int(hi) % 6
    glow = _Glow(W, H)
    fl = _pulse(local, 0.05)  # entry flash
    fm, om = _yu_masks(view, f"rep{hi}", width=2.0)
    cv.paint(fm, _mix(fillc, (1, 1, 1), 0.85 * fl), 0.92)
    if hi == 3:
        cv.paint(_yu_masks(view, "prov", width=1.0), DEEP, 0.8)
    cv.paint(om, _hot(stroke, 0.5), 1.0)
    rep = D.reps[hi]
    g1 = _low_lines(glow, view, [], 3 * u, rep.polys)
    glow.put(g1, stroke, 6 * u, 1.2 * gk * (1 + fl))
    glow.put(_low_fill(glow, view, rep.polys), stroke, 30 * u, 0.55 * gk * (1 + 1.5 * fl))

    if labels:
        cap = D.cities[rep.capital]
        cx, cy = view.px(cap.xy)
        r = 4.5 * u
        dm = Image.new("L", (int(4 * r) + 2, int(4 * r) + 2), 0)
        dd = ImageDraw.Draw(dm)
        c0 = dm.size[0] / 2
        dd.ellipse((c0 - r, c0 - r, c0 + r, c0 + r), fill=255)
        rc = _seg(local, 0.1, 0.3)
        cv.paint(dm, WHITE, rc, (cx - c0, cy - c0))
        _text(cv, cap.name.upper(), "mono", 17 * u, cx + 12 * u, cy - 8 * u, WHITE, rc, track=0.12)
        ax, ay = _big_name(cv, glow, W, H, rep.name, rep.en, f"{hi + 1:02d} / 06", local, kw)
        # leader line from the name block to the republic
        lx, ly = view.px(rep.label)
        k = _eo3(_seg(local, 0.12, 0.35))
        if k > 0:
            lm = _SS(W, H, 2)
            x0, y0 = ax + 24 * u, ay + 10 * u
            xm, ym = x0 + (lx - x0) * 0.5, y0
            path = np.array([[x0, y0], [xm, ym], [lx, ly]])
            c = _cum(path)
            lm.dr.line((_partial(path, c, c[-1] * k) * 2).ravel().tolist(), fill=255, width=_lw(1.2 * u, 2))
            if k >= 1:
                lm.dr.ellipse(((lx - 3.5 * u) * 2, (ly - 3.5 * u) * 2, (lx + 3.5 * u) * 2, (ly + 3.5 * u) * 2),
                              outline=255, width=_lw(1.2 * u, 2))
            cv.paint(lm.get(), WHITE, 0.7)
    glow.apply(cv)
    return cv


def _mode_neighbors(t, dur, W, H, p, view, kw):
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    labels = kw.get("labels", True)
    ncol = _col(kw.get("nb_color")) or WHITE
    a0, a1 = float(kw.get("start", 0.06)), float(kw.get("end", 0.86))

    bkey = ("nb_base", view.key(), kw.get("bg"), kw.get("grid", True), kw.get("hud", True), stroke, fillc)

    def base():
        cv = _Canvas(W, H, _col(kw.get("bg")))
        _ctx_paint(cv, W, H, view, 0.8, kw.get("rivers", False))
        _yu_hud(cv, view, kw.get("grid", True), kw.get("hud", True))
        fm, om = _yu_masks(view, "yugo")
        cv.paint(fm, fillc, 0.9)
        cv.paint(_yu_masks(view, "borders", width=1.0), DEEP, 0.7)
        cv.paint(om, _hot(stroke, 0.45), 1.0)
        g = _Glow(W, H)
        g.put(_low_lines(g, view, [], 3 * u, D.yugo), stroke, 6 * u, 1.1 * gk)
        g.put(_low_fill(g, view, D.yugo), stroke, 34 * u, 0.5 * gk)
        g.apply(cv)
        return cv.snapshot()

    cv = _Canvas(W, H, base=_cache_get(bkey, base))
    glow = _Glow(W, H)
    n = len(D.nbrs)
    lit = 0
    for i, nb in enumerate(D.nbrs):
        ti = a0 + (a1 - a0) * i / n
        k = _seg(p, ti, ti + 0.04)
        if k <= 0:
            continue
        lit = i + 1
        fl = _pulse(p - ti, 0.05)
        fm, om = _yu_masks(view, f"nb{i}", width=1.6)
        cv.paint(fm, ncol, (0.06 + 0.25 * fl) * k)
        cv.paint(om, ncol, (0.75 + 0.25 * fl) * k)
        glow.put(_low_lines(glow, view, [], 2.5 * u, nb.polys), ncol, 8 * u, (0.35 + 0.9 * fl) * k * gk)
        if labels:
            lx, ly = view.px(nb.label)
            lx = min(max(lx, 150 * u), W - 150 * u)
            ly = min(max(ly, 90 * u), H - 90 * u)
            rv = _eo5(_seg(p, ti, ti + 0.06))
            _text(cv, f"{i + 1:02d}", "mono", 15 * u, lx, ly - 44 * u, stroke, k, align="m", track=0.2)
            _text(cv, nb.name, "oswald", 44 * u, lx, ly, ncol, k, align="m", reveal=rv,
                  glow=glow, glow_col=ncol, glow_r=10 * u, glow_k=0.25 + 0.6 * fl)
            _text(cv, nb.en, "mono", 13 * u, lx, ly + 22 * u, ncol, 0.6 * k, align="m", track=0.25)
    if labels and kw.get("counter", True):
        x, y = W - 80 * u, 150 * u
        cnum = str(lit)
        _text(cv, cnum, "anton", 150 * u, x - 64 * u, y + 90 * u, stroke if lit else STEEL, 1.0 if lit else 0.4,
              align="r", glow=glow, glow_col=stroke, glow_r=16 * u, glow_k=0.5)
        _text(cv, "/7", "anton", 60 * u, x, y + 90 * u, WHITE, 0.8, align="r")
        _text(cv, "SUSEDI", "mono", 18 * u, x, y + 124 * u, WHITE, 0.85, align="r", track=0.3)
        _text(cv, "NEIGHBOURS", "mono", 13 * u, x, y + 146 * u, STEEL, 0.8, align="r", track=0.3)
    glow.apply(cv)
    return cv


CAP_ORDER = ["Ljubljana", "Zagreb", "Sarajevo", "Titograd", "Skopje", "Novi Sad", "Priština", "Beograd"]
_CAP_LABEL = {  # callout direction (dx, dy) in units of ~px@1080
    "Ljubljana": (-1, -1), "Zagreb": (1, -1), "Sarajevo": (-1, 1), "Beograd": (1, -1),
    "Titograd": (-1, 1), "Skopje": (1, 1), "Novi Sad": (-1, -1), "Priština": (1, 1),
}
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

    bkey = ("rep_base", view.key(), kw.get("bg"), kw.get("grid", True), kw.get("hud", True), stroke, fillc)

    def base():
        return _mode_republics(0, 1, W, H, 0, view, dict(kw, hi=None, sequence=False)).snapshot()

    cv = _Canvas(W, H, base=_cache_get(bkey, base))
    glow = _Glow(W, H)
    rings = _SS(W, H, 2)
    n = len(order)
    gdots, gd = glow.new()
    for i, name in enumerate(order):
        c = D.cities[name]
        ti = a0 + (a1 - a0) * i / max(n - 1, 1)
        age = p - ti
        if age < 0:
            continue
        big = c.kind == "federal"
        x, y = view.px(c.xy)
        # ping rings (in seconds so they read the same at any dur)
        ages = (age * dur - np.array([0.0, 0.12, 0.24])[: 3 if big else 2])
        for a in ages:
            if 0 <= a < 0.9:
                rr = (14 + 150 * (1 - (1 - a / 0.9) ** 3)) * u * (1.5 if big else 1.0)
                op = (1 - a / 0.9) ** 1.5
                rings.dr.ellipse(((x - rr) * 2, (y - rr) * 2, (x + rr) * 2, (y + rr) * 2),
                                 outline=int(255 * op), width=_lw(1.6 * u, 2))
        k = _eo3(_seg(age * dur, 0, 0.12))
        if big:
            sm = Image.new("L", (int(60 * u), int(60 * u)), 0)
            _flash_star(ImageDraw.Draw(sm), 30 * u, 31 * u, 22 * u * _eback(_seg(age * dur, 0, 0.3)))
            cv.paint(sm, GOLD, 1.0, (x - 30 * u, y - 31 * u))
            gd.polygon([(v[0] / glow.ds, v[1] / glow.ds) for v in
                        star_points(x, y, 24 * u)], fill=255)
        else:
            r = 5 * u
            dm = Image.new("L", (int(4 * r) + 2, int(4 * r) + 2), 0)
            c0 = dm.size[0] / 2
            ImageDraw.Draw(dm).ellipse((c0 - r, c0 - r, c0 + r, c0 + r), fill=255)
            cv.paint(dm, WHITE, k, (x - c0, y - c0))
            gd.ellipse((x / glow.ds - 1.5, y / glow.ds - 1.5, x / glow.ds + 1.5, y / glow.ds + 1.5), fill=255)
        if labels:
            dx, dy = _CAP_LABEL.get(name, (1, -1))
            L1 = (34 if big else 26) * u
            L2 = (46 if big else 34) * u
            ex, ey = x + dx * L1, y + dy * L1
            fx = ex + dx * L2
            lm = _SS(W, H, 2)
            path = np.array([[x + dx * 8 * u, y + dy * 8 * u], [ex, ey], [fx, ey]])
            cc = _cum(path)
            lm.dr.line((_partial(path, cc, cc[-1] * _eo3(_seg(age * dur, 0.02, 0.2))) * 2).ravel().tolist(),
                       fill=255, width=_lw(1.2 * u, 2))
            cv.paint(lm.get(), WHITE, 0.8)
            rv = _seg(age * dur, 0.1, 0.35)
            al = "l" if dx > 0 else "r"
            tx = fx + dx * 6 * u
            fsz = 34 * u if big else 22 * u
            fnm = "oswald" if big else "mono"
            _text(cv, c.name.upper(), fnm, fsz, tx, ey + fsz * 0.36, GOLD if big else WHITE, 1.0, align=al,
                  reveal=_eo3(rv), track=0 if big else 0.1, glow=glow if big else None, glow_col=GOLD,
                  glow_r=10 * u, glow_k=0.4)
            sub = _CAP_SUB.get(name) or f"{c.lat:.2f}°N {c.lon:.2f}°E"
            _text(cv, sub, "mono", 12 * u, tx, ey + fsz * 0.36 + 18 * u, STEEL, 0.9 * _eo3(rv), align=al,
                  track=0.15)
    cv.paint(rings.get(), _hot(stroke, 0.3), 1.0)
    glow.put(gdots, _hot(stroke, 0.3), 8 * u, 1.2 * gk)
    glow.apply(cv)
    return cv


# ----------------------------------------------------------- assemble ------
_ASM = {  # launch direction (x, y screen), start rotation (deg), delay
    "SVN": ((-0.95, -0.55), -40, 0.00),
    "HRV": ((-1.0, 0.25), 55, 0.04),
    "BIH": ((-0.15, -1.0), -65, 0.08),
    "SRB": ((1.0, -0.2), 30, 0.02),
    "MNE": ((-0.35, 1.0), 75, 0.10),
    "MKD": ((0.75, 0.85), -50, 0.06),
}


def _piece_xf(pts, c, ang, dx, dy, S, gc):
    """rotate pts (px) around c by ang, translate, then global scale S about gc."""
    ca, sa = math.cos(ang), math.sin(ang)
    x = pts[:, 0] - c[0]
    y = pts[:, 1] - c[1]
    X = x * ca - y * sa + c[0] + dx
    Y = x * sa + y * ca + c[1] + dy
    return np.column_stack([(X - gc[0]) * S + gc[0], (Y - gc[1]) * S + gc[1]])


def _mode_assemble(t, dur, W, H, p, view, kw):
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or RED
    fills = kw.get("fill")
    if fills is None:
        fills = [RED] * 6
    elif isinstance(fills, (list, tuple)) and len(fills) == 6 and not isinstance(fills[0], (int, float)):
        fills = [_col(f) for f in fills]
    else:
        fills = [_col(fills)] * 6
    gk = float(kw.get("glow", 1.0))
    imp = float(kw.get("impact", 0.55))
    trails = kw.get("trails", True)
    cv = _Canvas(W, H, _col(kw.get("bg")))
    glow = _Glow(W, H)
    ts = (p - imp) * dur  # seconds after impact

    # camera punch + shake after impact
    S = 1.0
    shx = shy = 0.0
    if ts >= 0:
        S = 1.0 + 0.055 * math.exp(-ts * 7.0) * math.cos(ts * 26.0)
        amp = 9 * u * math.exp(-ts * 12.0)
        shx = amp * math.sin(ts * 91.0)
        shy = amp * math.cos(ts * 73.0)
    gc = (W / 2 + shx, H / 2 + shy)

    _ctx_paint(cv, W, H, view, 0.35 + 0.65 * _seg(ts, 0, 0.3) if ts >= 0 else 0.35 * _seg(p, 0, 0.2),
               kw.get("rivers", False))
    _yu_hud(cv, view, kw.get("grid", True), kw.get("hud", True), _seg(p, 0, 0.1))

    diag = math.hypot(W, H)

    def pose(i, pp):
        rep = D.reps[i]
        (ddx, ddy), rot0, dl = _ASM[rep.id]
        q = _seg(pp, dl * imp / 0.55, imp)
        k = 1.0 - q ** 1.6
        dist = 0.85 * diag
        ang = math.radians(rot0) * k ** 1.3
        dx, dy = ddx * dist * k + shx, ddy * dist * k + shy
        if pp >= imp:  # recoil: tiny radial bounce
            tt = (pp - imp) * dur
            c = view.px(rep.label)
            rx, ry = c[0] - W / 2, c[1] - H / 2
            n = math.hypot(rx, ry) + 1e-6
            b = 7 * u * math.sin(min(tt / 0.16, 1.0) * math.pi) * math.exp(-tt * 6)
            dx += rx / n * b
            dy += ry / n * b
        return ang, dx, dy

    flash = _pulse(ts, 0.09) if ts >= 0 else 0.0
    seam = _seg(ts, 0.08, 0.5) if ts >= 0 else 0.0
    fill_m = _SS(W, H, 2)
    edge_m = _SS(W, H, 2)
    prov_m = _SS(W, H, 2)
    ghost = Image.new("L", (W, H), 0)
    gdr = ImageDraw.Draw(ghost)
    lowm, lowd = glow.new()
    per_piece = []
    for i, rep in enumerate(D.reps):
        c = view.px(rep.label)
        ang, dx, dy = pose(i, p)
        m1 = _SS(W, H, 2)
        for rings, bb in rep.polys:
            ext = _piece_xf(view.px(rings[0]), c, ang, dx, dy, S, gc)
            m1.dr.polygon((ext * 2).ravel().tolist(), fill=255)
            edge_m.dr.line((ext * 2).ravel().tolist(), fill=255, width=_lw(1.5 * u, 2), joint="curve")
            lowd.line((ext / glow.ds).ravel().tolist(), fill=255, width=1)
        if rep.id == "SRB":
            for ln in D.prov_borders:
                lp = _piece_xf(view.px(ln), c, ang, dx, dy, S, gc)
                for dsh in _dashes(lp, 6 * u, 5 * u):
                    prov_m.dr.line((dsh * 2).ravel().tolist(), fill=255, width=_lw(1.0 * u, 2))
        per_piece.append(m1.get())
        if trails and p < imp:
            for j, dtp in enumerate((0.012, 0.026, 0.045)):
                a2, dx2, dy2 = pose(i, max(p - dtp, 0))
                if abs(dx2 - dx) + abs(dy2 - dy) < 2:
                    continue
                for rings, bb in rep.polys[:1]:
                    ext = _piece_xf(view.px(rings[0]), c, a2, dx2, dy2, S, gc)
                    gdr.polygon(ext.ravel().tolist(), fill=int(70 / (j + 1)))
    if trails and p < imp:
        cv.paint(ghost, stroke, 0.9)
    fa = float(kw.get("fill_alpha", 0.92))
    for i, m in enumerate(per_piece):
        cv.paint(m, _mix(fills[i], (1, 1, 1), flash), fa)
    cv.paint(prov_m.get(), DEEP, 0.9)
    edge_col = _mix(_hot(stroke, 0.6), DEEP, seam)
    cv.paint(edge_m.get(), edge_col, 1.0)
    glow.put(lowm, stroke, 6 * u, (1.0 - 0.6 * seam) * gk)
    # unified outline after the slam
    if ts >= 0:
        om = _SS(W, H, 2)
        gm, gdd = glow.new()
        for rings, bb in D.yugo:
            ext = _piece_xf(view.px(rings[0]), (0, 0), 0, shx, shy, S, gc)
            om.dr.line((ext * 2).ravel().tolist(), fill=255, width=_lw(2.0 * u, 2), joint="curve")
            gdd.line((ext / glow.ds).ravel().tolist(), fill=255, width=1)
            gdd.polygon((ext / glow.ds).ravel().tolist(), fill=int(255 * flash))
        cv.paint(om.get(), _hot(stroke, 0.35 + 0.5 * flash), _seg(ts, 0.0, 0.15))
        glow.put(gm, stroke, 6 * u, 1.3 * gk * (1 + 2 * flash))
        glow.put(gm, stroke, 30 * u, 0.8 * gk * (1 + 3 * flash))
        # shockwave
        if ts < 0.6:
            sw = _SS(W, H, 2)
            rr = (0.15 + 1.1 * _eo3(ts / 0.6)) * H
            op = (1 - ts / 0.6) ** 1.5
            sw.dr.ellipse(((gc[0] - rr) * 2, (gc[1] - rr) * 2, (gc[0] + rr) * 2, (gc[1] + rr) * 2),
                          outline=255, width=_lw(2.5 * u, 2))
            cv.paint(sw.get(), WHITE, op * 0.8)
            swg, swd = glow.new()
            swd.ellipse(((gc[0] - rr) / glow.ds, (gc[1] - rr) / glow.ds, (gc[0] + rr) / glow.ds,
                         (gc[1] + rr) / glow.ds), outline=255, width=2)
            glow.put(swg, _hot(stroke, 0.3), 14 * u, op * gk)
    glow.apply(cv)
    return cv


# ------------------------------------------------------------- tilt 3D -----
def _mode_tilt3d(t, dur, W, H, p, view, kw):
    D = _yu()
    u = H / 1080.0
    stroke = _col(kw.get("stroke")) or GOLD
    fillc = _col(kw.get("fill")) or RED
    gk = float(kw.get("glow", 1.0))
    ext = float(kw.get("extrude", 0.09))  # slab thickness as a fraction of map height
    yaw = math.radians(float(kw.get("yaw", -18.0)) + float(kw.get("spin", 26.0)) * (p - 0.5))
    pitch = math.radians(float(kw.get("pitch", 52.0)) + float(kw.get("nod", 5.0)) * math.sin(p * math.pi))
    scale = float(kw.get("scale", 1.0))
    cv = _Canvas(W, H, _col(kw.get("bg")))
    glow = _Glow(W, H)

    x0, y0, x1, y1 = D.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = max(x1 - x0, y1 - y0)
    k = 1.0 / size  # normalised model units: map ~1 wide
    hgt = ext * (y1 - y0) * k

    ca, sa = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    dist = float(kw.get("distance", 2.2))
    f = H * 1.55 * scale * dist / 2.2
    lift = float(kw.get("lift", 0.0))

    def P3(xy, z):
        """model (N,2) stored units + z -> screen (N,2), depth (N,)"""
        X = (xy[:, 0] - cx) * k
        Y = (xy[:, 1] - cy) * k
        Xr = X * ca - Y * sa
        Yr = X * sa + Y * ca
        Z = np.full_like(Xr, z)
        y2 = Yr * cp + Z * sp
        z2 = -Yr * sp + Z * cp
        dd = dist - z2
        sx = W / 2 + f * Xr / dd
        sy = H / 2 + H * (0.06 - lift) - f * y2 / dd
        return np.column_stack([sx, sy]), dd

    cam = np.array([0.0, 0.0, dist])
    rings = [r for rs, bb in D.coarse for r in rs[:1]]

    # floor grid + shadow
    if kw.get("floor", True):
        fz = -hgt - 0.04
        gm = _SS(W, H, 2)
        n = 14
        for i in range(-n, n + 1):
            a = np.array([[i / n * 1.4 / k + cx, -1.4 / k + cy], [i / n * 1.4 / k + cx, 1.4 / k + cy]])
            b = np.array([[-1.4 / k + cx, i / n * 1.4 / k + cy], [1.4 / k + cx, i / n * 1.4 / k + cy]])
            for seg in (a, b):
                ss_ = np.linspace(0, 1, 24)[:, None]
                pts = seg[0] + (seg[1] - seg[0]) * ss_
                sc, dd = P3(pts, fz)
                ok = dd > 0.3
                if ok.sum() > 1:
                    gm.dr.line((sc[ok] * 2).ravel().tolist(), fill=255, width=_lw(1.0 * u, 2))
        # fade the grid with a radial vignette
        g = gm.get()
        vig = _vignette(W, H)
        cv.paint(ImageChops.multiply(g, vig), STEEL, 0.22)
        sg, sdr = glow.new()
        for r in rings:
            sc, _ = P3(r, fz)
            sdr.polygon((sc / glow.ds).ravel().tolist(), fill=255)
        sh = sg.filter(ImageFilter.GaussianBlur(14 * u / glow.ds)).resize((W, H), Image.BILINEAR)
        cv.paint(sh, BLACK, 0.7)

    # side walls
    light = np.array([-0.45, 0.35, 0.82])
    light /= np.linalg.norm(light)
    quads = []
    for r in rings:
        top, dt = P3(r, 0.0)
        bot, db = P3(r, -hgt)
        X = (r[:, 0] - cx) * k
        Y = (r[:, 1] - cy) * k
        dx = np.diff(X)
        dy = np.diff(Y)
        nx, ny = -dy, dx  # outward for clockwise rings
        nn = np.hypot(nx, ny) + 1e-12
        nx, ny = nx / nn, ny / nn
        # rotate normals (yaw, then pitch)
        nxr = nx * ca - ny * sa
        nyr = nx * sa + ny * ca
        ny2 = nyr * cp
        nz2 = -nyr * sp
        # midpoints in camera space for back-face test
        mx = (X[:-1] + X[1:]) / 2
        my = (Y[:-1] + Y[1:]) / 2
        mxr = mx * ca - my * sa
        myr = mx * sa + my * ca
        mz = -hgt / 2
        py2 = myr * cp + mz * sp
        pz2 = -myr * sp + mz * cp
        vis = (nxr * (0 - mxr) + ny2 * (0 - py2) + nz2 * (dist - pz2)) > 0
        lam = np.clip(nxr * light[0] + ny2 * light[1] + nz2 * light[2], 0, 1)
        depth = dist - pz2
        for j in np.nonzero(vis)[0]:
            quads.append((depth[j], lam[j], top[j], top[j + 1], bot[j + 1], bot[j]))
    quads.sort(key=lambda q: -q[0])
    wall = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
    wa = Image.new("L", (W * 2, H * 2), 0)
    wd = ImageDraw.Draw(wall)
    wad = ImageDraw.Draw(wa)
    wc = np.array(_mix(fillc, DEEP, 0.55))
    for dep, lam, a, b, c, d in quads:
        shade = 0.28 + 0.95 * lam
        colr = tuple(int(min(255, v * 255 * shade)) for v in wc)
        poly = [a[0] * 2, a[1] * 2, b[0] * 2, b[1] * 2, c[0] * 2, c[1] * 2, d[0] * 2, d[1] * 2]
        wd.polygon(poly, fill=colr, outline=colr)
        wad.polygon(poly, fill=255, outline=255)
    wall = wall.reduce(2)
    wa = wa.reduce(2)
    cv.C.paste(wall, (0, 0), wa)
    cv.A.paste(255, (0, 0), wa)
    # bottom gold edge
    be = _SS(W, H, 2)
    for r in rings:
        sc, _ = P3(r, -hgt)
        be.dr.line((sc * 2).ravel().tolist(), fill=255, width=_lw(1.0 * u, 2))
    cv.paint(ImageChops.multiply(be.get(), wa), stroke, 0.35)

    # top face
    tm = _SS(W, H, 2)
    for r in rings:
        sc, _ = P3(r, 0.0)
        tm.dr.polygon((sc * 2).ravel().tolist(), fill=255)
    topm = tm.get()
    cv.paint(topm, fillc, 1.0)
    # sheen sweep
    sw = _sheen(W, H, p)
    cv.paint(ImageChops.multiply(topm, sw), (1.0, 0.75, 0.7), 0.35)
    bm = _SS(W, H, 2)
    for ln in D.coarse_borders:
        sc, _ = P3(ln, 0.0)
        bm.dr.line((sc * 2).ravel().tolist(), fill=255, width=_lw(1.3 * u, 2))
    cv.paint(bm.get(), DEEP, 0.85)
    om = _SS(W, H, 2)
    gl, gld = glow.new()
    for r in rings:
        sc, _ = P3(r, 0.0)
        om.dr.line((sc * 2).ravel().tolist(), fill=255, width=_lw(2.4 * u, 2), joint="curve")
        gld.line((sc / glow.ds).ravel().tolist(), fill=255, width=1)
    cv.paint(om.get(), _hot(stroke, 0.2), 1.0)
    glow.put(gl, stroke, 6 * u, 0.9 * gk)
    glow.put(gl, RED, 28 * u, 0.7 * gk)
    if kw.get("star", True):
        c = D.cities["Beograd"]
        sc, _ = P3(c.xy[None], 0.0)
        sx, sy = sc[0]
        sm = Image.new("L", (int(40 * u), int(40 * u)), 0)
        _flash_star(ImageDraw.Draw(sm), 20 * u, 21 * u, 13 * u)
        cv.paint(sm, GOLD, 1.0, (sx - 20 * u, sy - 21 * u))
        sg, sd = glow.new()
        sd.ellipse((sx / glow.ds - 2, sy / glow.ds - 2, sx / glow.ds + 2, sy / glow.ds + 2), fill=255)
        glow.put(sg, GOLD, 10 * u, 1.2 * gk)
    _hud(cv, view, [], "SFRJ // 1945–1992", f"YAW {math.degrees(yaw):+06.1f}°  PITCH {math.degrees(pitch):04.1f}°",
         None, False, kw.get("hud", True), 1.0)
    glow.apply(cv)
    return cv


@functools.lru_cache(maxsize=8)
def _vignette(W, H):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((x - W / 2) / (W / 2), (y - H * 0.55) / (H / 2))
    v = np.clip(1.25 - r, 0, 1) ** 1.5
    return Image.fromarray((v * 255).astype(np.uint8))


def _sheen(W, H, p):
    w, h = W // 8, H // 8
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    c = (-0.3 + 1.6 * p) * (w + h)
    d = (x + y * 0.7 - c) / (0.12 * (w + h))
    v = np.exp(-d * d)
    return Image.fromarray((v * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)


# ============================================================ yugo_map =====
_MODE_DEFAULTS = {
    "neighbors": dict(scale=0.64, center=(19.4, 43.3)),
    "tilt3d": dict(),
}


def yugo_map(t, dur, W, H, mode="draw", **kw):
    """Yugoslavia map generator -> float32 RGBA (H, W, 4), straight alpha.

    mode
      "draw"       glowing red outline draws itself on (spark at the head, coordinate
                   read-out), islands pop in as the head passes, then republic borders
                   and dashed province borders; faint fill wash.
                   kw: reveal_end=0.62 (fraction of dur when the outline closes), width=1.8
      "assemble"   the 6 republics fly in from off-screen with rotation + motion trails,
                   slam together at p=impact (0.55): white flash, camera punch/shake,
                   shock ring, seams cool down, unified outline glows.
                   kw: impact=0.55, fill=<colour or list of 6>, trails=True, fill_alpha=0.92
      "republics"  all dim; hi=0..5 lights one republic + big name + English caption +
                   capital + leader line.  hi=None -> sequence through all 6 over dur.
                   kw: hi, sequence, local=<0..1 progress inside the highlight for the
                   entry animation when hi is fixed; default = t/dur>, name_font="anton"
      "neighbors"  SFRJ filled red; ITALIJA, AUSTRIJA, MAĐARSKA, RUMUNIJA, BUGARSKA,
                   GRČKA, ALBANIJA light up in turn with names + counter N/7.
                   kw: start=0.06, end=0.86, nb_color, counter=True
      "capitals"   ping rings from the capitals in sequence (Beograd last, gold star).
                   kw: order=[...names...], start=0.04, end=0.8
      "tilt3d"     extruded slab in perspective, slowly yawing; red top, gold rim,
                   lit side walls, floor grid + shadow, sheen sweep, Beograd star.
                   kw: extrude=0.09, yaw=-18 (deg, centre of motion), spin=26 (deg over dur),
                   pitch=52 (deg), nod=5, distance=2.2, floor=True, star=True, lift=0

    Common kw: scale=1 (zoom multiplier), center=(lon, lat), offset=(dx, dy) (fraction of
    W/H, shifts the map on screen), stroke=<colour>, fill=<colour>, glow=1.0 (strength),
    labels=True, hud=True (corner brackets, ticks, read-outs), grid=True (graticule),
    rivers=False, scan=False (scanline alpha), bg=None (transparent) or colour,
    phase=None (override t/dur progress 0..1).
    Colours: PAL names ('red', 'gold', ...), '#hex' or rgb floats.
    """
    p = kw.pop("phase", None)
    p = _cl(t / dur if dur > 0 else 1.0) if p is None else _cl(float(p))
    d = dict(_MODE_DEFAULTS.get(mode, {}))
    scale = float(kw.get("scale", 1.0)) * d.get("scale", 1.0)
    center = kw.get("center", d.get("center"))
    view = _yu_view(W, H, scale, center, kw.get("offset", (0.0, 0.0)))
    fn = {"draw": _mode_draw, "assemble": _mode_assemble, "republics": _mode_republics,
          "neighbors": _mode_neighbors, "capitals": _mode_capitals, "tilt3d": _mode_tilt3d}[mode]
    cv = fn(t, dur, W, H, p, view, kw)
    return cv.array(scan=kw.get("scan", False))


# ================================================================ preview ===
def _sheet(fn, name, samples, W=960, H=540, cols=3, bg=(0.02, 0.02, 0.03), **kw):
    """Render samples [(t, dur, extra_kw)] -> contact sheet PNG on a dark bg."""
    tiles = []
    for (t, dur, ex) in samples:
        rgba = fn(t, dur, W, H, **dict(kw, **ex))
        rgb = np.empty((H, W, 3), np.float32)
        rgb[:] = bg
        a = rgba[..., 3:4]
        rgb = rgb * (1 - a) + rgba[..., :3] * a
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((8, H - 8), f"t={t:.2f}/{dur:.2f}  {ex if ex else ''}", font=font("mono", 12),
               fill=(255, 255, 0), anchor="ls")
        tiles.append(im)
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * W + (cols - 1) * 4, rows * H + (rows - 1) * 4), (60, 60, 60))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * (W + 4), (i // cols) * (H + 4)))
    out = os.path.join(_ROOT, "out", "preview")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"maps_{name}.png")
    sheet.save(path)
    return path


def _time(fn, n=8, W=1920, H=1080, dur=2.0, **kw):
    fn(0.37 * dur, dur, W, H, **kw)  # warm caches
    ts = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        t0 = time.perf_counter()
        fn(t, dur, W, H, **kw)
        ts.append((time.perf_counter() - t0) * 1000)
    return np.mean(ts), np.max(ts)


PREVIEWS = {
    "draw": (yugo_map, dict(mode="draw"), [0.1, 0.3, 0.5, 0.64, 0.8, 1.0], 3.0),
    "assemble": (yugo_map, dict(mode="assemble"), [0.15, 0.35, 0.5, 0.56, 0.62, 0.95], 2.0),
    "republics": (yugo_map, dict(mode="republics"), [0.05, 0.22, 0.4, 0.55, 0.72, 0.9], 6.0),
    "neighbors": (yugo_map, dict(mode="neighbors"), [0.05, 0.2, 0.4, 0.6, 0.8, 1.0], 2.0),
    "capitals": (yugo_map, dict(mode="capitals"), [0.1, 0.3, 0.5, 0.7, 0.85, 1.0], 4.0),
    "tilt3d": (yugo_map, dict(mode="tilt3d"), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 4.0),
}


def main(argv):
    which = argv[1:] or list(PREVIEWS)
    for name in which:
        fn, kw, ps, dur = PREVIEWS[name]
        path = _sheet(fn, name, [(pp * dur, dur, {}) for pp in ps], **kw)
        mean, mx = _time(fn, dur=dur, **kw)
        print(f"{name:10s} {mean:6.1f} ms/frame (max {mx:5.1f})  @1920x1080  -> {os.path.relpath(path, _ROOT)}")


if __name__ == "__main__":
    main(sys.argv)
