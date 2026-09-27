"""Editing kit: shot builders (S_*), frame ops (O_*), transitions (X_*), text (T_*).

Shots:  fn(c) -> RGB frame             (c = engine.timeline.Ctx)
Ops:    fn(c, img) -> RGB frame        (applied in z order after the shot)
All sizes are fractions of frame height unless noted; positions fractions of W/H.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw

from engine import fx, media, text as tx
from engine.core import (PAL, blank, over, ease_out_expo, ease_out_cubic, ease_in_cubic,
                         ease_in_out_cubic, ease_out_back, font, star_points, from_pil, to_pil, lerp)
from engine.timeline import b, BEAT

WHITE, BLACK, RED, BLUE, GOLD = PAL["white"], PAL["black"], PAL["red"], PAL["blue"], PAL["gold"]


# ------------------------------------------------------------------ photos ---
def _photo_or_placeholder(name, W, H, p, **kw):
    try:
        return media.kenburns(name, W, H, p, **kw)
    except FileNotFoundError:
        img = np.zeros((H, W, 3), np.float32)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        img[..., 0] = 0.25 + 0.2 * yy
        img[..., 1] = 0.1 * yy
        img[..., 2] = 0.12
        tx.place(img, tx.text_layer(f"[{name}]", "mono", H * 0.05, WHITE), 0.5, 0.5)
        return img


PHOTO_EASE = None  # optional fn(c) -> ease fn, set by the edit (velocity-style moves in drops)


def velocity(p):
    """Snap-then-drift: most of the move happens in the first frames (velocity edit feel)."""
    return 0.65 * ease_out_expo(min(p * 1.6, 1.0)) + 0.35 * p


def S_photo(name, zoom=(1.0, 1.12), center=((0.5, 0.5), (0.5, 0.5)), rot=(0.0, 0.0),
            grade=None, contrast=1.3, fxs=(), flip=False, ease=None, pdur=None):
    """Still with Ken Burns move + grade + fx chain. pdur: seconds the move spans (default shot dur)."""
    def f(c):
        p = c.lt / (pdur or c.dur)
        if ease is None and PHOTO_EASE is not None:
            e = PHOTO_EASE(c)
            if e is not None:
                p = e(min(max(p, 0), 1))
        img = _photo_or_placeholder(name, c.W, c.H, min(max(p, 0), 1.5), zoom=zoom, center=center,
                                    rot=rot, flip=flip, ease=ease)
        if grade:
            img = fx.grade(img, grade, contrast)
        for g in fxs:
            img = g(c, img)
        return img
    return f


def S_solid(rgb=BLACK, fxs=()):
    def f(c):
        img = blank(c.W, c.H, rgb)
        for g in fxs:
            img = g(c, img)
        return img
    return f


def S_gen(gen, bg=BLACK, dur=None, t_off=0.0, fxs=(), under=None, **params):
    """Wrap an agent generator fn(t, dur, W, H, **params) -> RGBA over a background."""
    def f(c):
        img = under(c) if under else blank(c.W, c.H, bg)
        lay = gen(c.lt + t_off, dur or c.dur, c.W, c.H, **params)
        over(img, lay)
        for g in fxs:
            img = g(c, img)
        return img
    return f


def S_layers(base, *ops):
    """Base shot + ops applied in order (ops are fn(c, img))."""
    def f(c):
        img = base(c)
        for g in ops:
            out = g(c, img)
            if out is not None:
                img = out
        return img
    return f


def S_seq(parts):
    """Sub-cuts inside one shot: parts = [(t_local_start, shot_fn), ...]."""
    parts = sorted(parts, key=lambda x: x[0])

    def f(c):
        cur = parts[0]
        nxt_t = c.dur
        for k, pr in enumerate(parts):
            if c.lt >= pr[0]:
                cur = pr
                nxt_t = parts[k + 1][0] if k + 1 < len(parts) else c.dur
        return cur[1](c.at(cur[0], nxt_t - cur[0]))
    return f


# ------------------------------------------------------------------- ops ----
def O_fx(fn, *args, **kw):
    """Wrap a plain fx.* function as an op."""
    return lambda c, img: fn(img, *args, **kw)


def O_grade(name, contrast=1.3, mix=1.0):
    return lambda c, img: fx.grade(img, name, contrast, mix)


def O_flash(rgb=WHITE, decay=0.08, peak=1.0, at=0.0):
    def f(c, img):
        d = c.lt - at
        if d < 0:
            return img
        return fx.flash(img, rgb, peak * math.exp(-d / decay))
    return f


def O_strobe(colors=(WHITE, BLACK), rate=BEAT / 4, mode="flash", amt=1.0):
    """Cycle through colours every `rate` seconds. mode: flash | multiply | invert | grade names."""
    def f(c, img):
        k = int(c.lt / rate)
        col = colors[k % len(colors)]
        if col is None:
            return img
        if mode == "flash":
            return fx.flash(img, col, amt)
        if mode == "multiply":
            return fx.tint(img, col, amt, "multiply")
        if mode == "gradient":
            return fx.grade(img, col)
        if mode == "invert":
            return fx.invert(img) if col else img
        return img
    return f


def O_grade_cycle(names, rate=BEAT / 2, contrast=1.35):
    def f(c, img):
        n = names[int(c.lt / rate) % len(names)]
        return fx.grade(img, n, contrast) if n else img
    return f


def O_glitch(amount=1.0, every=2, blocks=True):
    def f(c, img):
        s = c.frame // every + c.seed
        img = fx.glitch_slices(img, amount, s)
        if blocks:
            img = fx.block_glitch(img, amount * 0.7, s + 1)
        return img
    return f


def O_glitch_burst(at=0.0, dur=0.12, amount=1.2):
    def f(c, img):
        if at <= c.lt < at + dur:
            return fx.block_glitch(fx.glitch_slices(img, amount, c.frame), amount, c.frame + 7)
        return img
    return f


def O_rgb(amt=10, angle=0.0):
    return lambda c, img: fx.rgb_split(img, amt * c.W / 1920, angle)


def O_halftone(cell=10, ink=BLACK, paper=WHITE, angle=0.3):
    return lambda c, img: fx.halftone(img, max(4, int(cell * c.W / 1920)), angle, ink, paper)


def O_zoomblur(strength=0.1, at=0.0, decay=0.15):
    def f(c, img):
        s = strength * math.exp(-max(c.lt - at, 0) / decay) if c.lt >= at else 0
        return fx.zoom_blur(img, s) if s > 0.004 else img
    return f


def O_speedlines(alpha=0.8, color=WHITE, inner=0.3, n=110, decay=None):
    def f(c, img):
        a = alpha if decay is None else alpha * math.exp(-c.lt / decay)
        if a < 0.02:
            return img
        lay = fx.speed_lines(c.W, c.H, c.t, n=n, color=color, inner=inner, alpha=a)
        return over(img, lay)
    return f


def O_bloom(thresh=0.65, strength=0.9, radius=30):
    return lambda c, img: fx.bloom(img, thresh, strength, radius * c.W / 1920)


def O_vhs(amount=1.0):
    return lambda c, img: fx.vhs(img, c.t, amount, c.seed)


def O_grid(n=3, every=None):
    """Tile the frame n x n (optionally only on alternating beats)."""
    def f(c, img):
        if every and int(c.lt / every) % 2 == 0:
            return img
        return fx.grid_repeat(img, n)
    return f


def O_mirror(mode="h"):
    return lambda c, img: fx.mirror(img, mode)


def O_invert_on(beats, width=0.07):
    """Invert frame for `width` seconds after each local time in beats list."""
    def f(c, img):
        for t0 in beats:
            if t0 <= c.lt < t0 + width:
                return fx.invert(img)
        return img
    return f


def O_edges(color=RED, gain=4.0, mix=1.0):
    def f(c, img):
        e = fx.edges(img, color, gain)
        return img + (e - img) * mix
    return f


def O_scan(strength=0.25, period=4):
    return lambda c, img: fx.scanlines(img, strength, period)


def O_colorbars(p_fn=None):
    """Horizontal Yugoslav-flag colour bars wipe (overlay)."""
    def f(c, img):
        p = p_fn(c) if p_fn else c.p
        H, W = img.shape[:2]
        cols = (BLUE, WHITE, RED)
        for k, col in enumerate(cols):
            y0, y1 = int(k * H / 3), int((k + 1) * H / 3)
            x1 = int(W * ease_out_expo(min(max(p * 3 - k * 0.3, 0), 1)))
            img[y0:y1, :x1] = col
        return img
    return f


def O_letterbox(amount_fn):
    """Cinema bars; amount_fn(c) -> fraction of height covered by each bar."""
    def f(c, img):
        a = amount_fn(c)
        if a <= 0:
            return img
        h = int(img.shape[0] * a)
        img[:h] = 0
        img[img.shape[0] - h:] = 0
        return img
    return f


def O_star_iris(at=0.0, dur=0.3, opening=True, rgb=BLACK, rot=0.4):
    """Star-shaped iris wipe (opening reveals the image from a growing red-star hole)."""
    def f(c, img):
        p = (c.lt - at) / dur
        if p < 0:
            return img if not opening else fx.flash(img, rgb, 1.0)
        if p >= 1:
            return img if opening else fx.flash(img, rgb, 1.0)
        e = ease_in_out_cubic(p)
        H, W = img.shape[:2]
        r = (e if opening else 1 - e) * math.hypot(W, H) * 0.95
        m = Image.new("L", (W, H), 0)
        if r > 1:
            ImageDraw.Draw(m).polygon(star_points(W / 2, H / 2, r, r * 0.45, rot * e), fill=255)
        a = (np.asarray(m, np.float32) / 255.0)[..., None]
        return img * a + np.asarray(rgb, np.float32) * (1 - a)
    return f


# ------------------------------------------------------------ transitions ---
def X_whip(tl, t_cut, dx=1, dy=0, half=0.1, amp=0.45):
    """Whip pan across a hard cut: outgoing slides out, incoming slides in, motion blurred."""
    def f(c, img):
        u = (c.t - t_cut) / half          # -1..1
        k = (1 - abs(u))                   # 0 at edges, 1 at cut
        off = (u - math.copysign(1, u)) if u != 0 else 0
        sx = off * amp * c.W * dx * (1 if u >= 0 else -1) * -1
        sy = off * amp * c.H * dy * (1 if u >= 0 else -1) * -1
        img = fx.shift(img, sx, sy)
        return fx.motion_blur(img, dx * k * 0.25 * c.W, dy * k * 0.25 * c.H, n=10)
    tl.op(t_cut - half, t_cut + half, f, z=45, name="whip")


def X_zoom(tl, t_cut, half=0.12, strength=0.35):
    """Zoom-through: outgoing punches in, incoming lands from wide, both zoom-blurred."""
    def f(c, img):
        u = (c.t - t_cut) / half
        if u < 0:
            z = 1 + ease_in_cubic(1 + u) * 1.2
        else:
            z = 1 + (1 - ease_out_cubic(u)) * 0.6
        img = fx.affine(img, zoom=z)
        return fx.zoom_blur(img, strength * (1 - abs(u)))
    tl.op(t_cut - half, t_cut + half, f, z=45, name="zoomthru")


def X_spin(tl, t_cut, half=0.12, deg=35):
    def f(c, img):
        u = (c.t - t_cut) / half
        r = deg * (1 - abs(u)) * (1 if u < 0 else -1)
        z = 1 + 0.4 * (1 - abs(u))
        return fx.zoom_blur(fx.affine(img, zoom=z, rot=r), 0.12 * (1 - abs(u)))
    tl.op(t_cut - half, t_cut + half, f, z=45, name="spin")


def X_flash(tl, t_cut, rgb=WHITE, decay=0.07, pre=0.0):
    tl.op(t_cut - pre, t_cut + decay * 5, lambda c, img: fx.flash(img, rgb, math.exp(-max(c.t - t_cut, 0) / decay)),
          z=70, name="flash")


def X_glitch(tl, t_cut, half=0.07, amount=1.3):
    tl.op(t_cut - half, t_cut + half,
          lambda c, img: fx.block_glitch(fx.glitch_slices(img, amount, c.frame), amount, c.frame + 3),
          z=46, name="glitchcut")


# ------------------------------------------------------------------ text ----
def _L(text, fname, size_px, color, tracking=0.0, stroke=0, stroke_color=BLACK, outline_only=False):
    return tx.text_layer(text, fname, size_px, color, tracking, stroke, stroke_color, outline_only)


def T(text, fname="anton", size=0.3, color=WHITE, cx=0.5, cy=0.5, at=0.0, until=None,
      slam=0.13, s0=2.2, rot=0.0, tracking=0.0, stroke=0.0, stroke_color=BLACK,
      outline=False, shadow=None, mode="normal", fit_w=None, sx=1.0, sy=1.0, drift=0.0,
      opacity=1.0, fade_out=0.0, glow=None):
    """Text op: slams in at local time `at` (s0 -> 1 over `slam` s), optional slow drift scale."""
    def f(c, img):
        lt = c.lt - at
        if lt < 0 or (until is not None and c.lt >= until):
            return img
        H = c.H
        sz = size * H
        if fit_w:
            sz = min(sz, tx.fit_size(text, fname, fit_w * c.W, 10 * H, tracking))
        st = int(stroke * H)
        L = _L(text, fname, int(sz), color, tracking, st, stroke_color, outline)
        p = lt / slam if slam else 1.0
        e = ease_out_expo(min(p, 1.0))
        s = (s0 + (1 - s0) * e) * (1 + drift * lt)
        a = opacity * (min(1.0, 0.2 + p * 2) if slam else 1.0)
        if fade_out and until is not None:
            a *= min(1.0, (until - c.lt) / fade_out)
        if shadow:
            Ls = _L(text, fname, int(sz), shadow, tracking, st, shadow, outline)
            tx.place(img, Ls, cx + 0.006, cy + 0.01, s, rot, a * 0.9, sx, sy)
        if glow is not None:
            Lg = _L(text, fname, int(sz), glow, tracking, st, glow, outline)
            Lg = fx.blur_fast(Lg[..., :3] * Lg[..., 3:4], 8, 2)
            g = np.concatenate([np.clip(Lg * 3, 0, 1), np.clip(fx.luma(Lg) * 3, 0, 1)[..., None]], axis=2)
            tx.place(img, g, cx, cy, s * 1.0, rot, a * 0.8, sx, sy, mode="add")
        tx.place(img, L, cx, cy, s, rot, a, sx, sy, mode=mode)
        return img
    return f


def T_cap(text, cy=0.9, at=0.0, until=None, size=0.028, color=WHITE, bar_color=BLACK, cx=0.5):
    """English caption (typed on quickly)."""
    def f(c, img):
        lt = c.lt - at
        if lt < 0 or (until is not None and c.lt >= until):
            return img
        n = max(1, int(lt * 70))
        s = text[:n]
        return tx.caption(img, s, cy=cy, size_frac=size, color=color, bar_color=bar_color, cx=cx)
    return f


def T_type(text, fname="mono", size=0.06, color=WHITE, cx=0.5, cy=0.5, cps=16, at=0.0, cursor=True,
           tracking=0.05):
    def f(c, img):
        lt = c.lt - at
        if lt < 0:
            return img
        s = tx.typewriter(text, lt, cps, cursor)
        if fname != "mono":
            s = s.replace("█", "_")
        if not s:
            return img
        L = _L(s, fname, int(size * c.H), color, tracking)
        full = _L(text + ("█" if fname == "mono" else "_"), fname, int(size * c.H), color, tracking)
        x0 = cx - full.shape[1] / c.W / 2
        tx.place(img, L, x0, cy, anchor="l")
        return img
    return f


def T_scramble(text, fname="anton", size=0.2, color=WHITE, cx=0.5, cy=0.5, at=0.0, dur=0.5, tracking=0.02):
    def f(c, img):
        lt = c.lt - at
        if lt < 0:
            return img
        s = tx.scramble(text, lt / dur, seed=len(text))
        L = _L(s, fname, int(size * c.H), color, tracking)
        tx.place(img, L, cx, cy)
        return img
    return f


def T_echo(text, fname="anton", size=0.25, color=WHITE, cx=0.5, cy=0.5, n=4, dy=0.2, at=0.0,
           outline_color=RED, spread=0.25):
    """Solid word with outlined echoes spreading out vertically."""
    def f(c, img):
        lt = c.lt - at
        if lt < 0:
            return img
        sz = int(size * c.H)
        L = _L(text, fname, sz, color)
        O = _L(text, fname, sz, outline_color, 0, max(2, sz // 40), outline_color, outline_only=True)
        e = ease_out_expo(min(lt / spread, 1))
        for k in range(n, 0, -1):
            a = 0.85 ** k
            tx.place(img, O, cx, cy - dy * k * e, opacity=a)
            tx.place(img, O, cx, cy + dy * k * e, opacity=a)
        tx.place(img, L, cx, cy)
        return img
    return f


def S_img_text(text, photo, fname="anton", size=0.6, cx=0.5, cy=0.5, zoom=(1.0, 1.2), grade=None,
               bg=RED, open_at=None, open_dur=0.35, tracking=0.0, fit_w=0.92):
    """Shot: photo seen through giant letters; optionally zooms through a letter to full frame."""
    def f(c):
        H, W = c.H, c.W
        ph = _photo_or_placeholder(photo, W, H, c.p, zoom=zoom)
        if grade:
            ph = fx.grade(ph, grade)
        sz = min(size * H, tx.fit_size(text, fname, fit_w * W, H, tracking))
        L = _L(text, fname, int(sz), WHITE, tracking)
        lay = tx.mask_image_with_text(ph, L, cx, cy, 1.0)
        if open_at is not None and c.lt > open_at:
            # zoom through the letters by transforming the frame-sized mask (never a giant layer)
            q = min((c.lt - open_at) / open_dur, 1)
            if q >= 1:
                return ph
            s = 1 + ease_in_cubic(q) * 30
            m = Image.fromarray((lay[..., 3] * 255).astype(np.uint8), "L")
            m = m.transform((W, H), Image.AFFINE, (1 / s, 0, cx * W * (1 - 1 / s), 0, 1 / s, cy * H * (1 - 1 / s)),
                            resample=Image.BILINEAR)
            lay[..., 3] = np.asarray(m, np.float32) / 255.0
        out = blank(W, H, bg)
        over(out, lay)
        return out
    return f


# --------------------------------------------------------------- camera -----
def O_camera(punch=0.045, shake=16.0, rgb=9.0, drift=3.0, kick_decay=0.1, snare_decay=0.08,
             extra_zoom=None, extra_rot=None):
    """Beat-reactive camera: kick zoom punch, snare shake + chroma, slow handheld drift."""
    def f(c, img):
        bm = c.bm
        pk = bm.pulse(bm.kick, c.t, kick_decay)
        ps = bm.pulse(bm.snare, c.t, snare_decay)
        z = 1 + punch * pk + (extra_zoom(c) if extra_zoom else 0)
        rng = np.random.default_rng(c.frame)
        sh = shake * ps * c.W / 1920
        dx = drift * math.sin(c.t * 1.7) * c.W / 1920 + rng.uniform(-1, 1) * sh
        dy = drift * math.cos(c.t * 1.3) * c.W / 1920 + rng.uniform(-1, 1) * sh
        r = (extra_rot(c) if extra_rot else 0) + rng.uniform(-1, 1) * 0.6 * ps
        img = fx.affine(img, zoom=z, rot=r, dx=dx, dy=dy)
        a = rgb * ps + rgb * 0.4 * pk
        if a > 0.6:
            img = fx.rgb_split(img, a * c.W / 1920, 0.3)
        return img
    return f


def O_finish(grain=0.045, vig=0.4):
    def f(c, img):
        img = fx.vignette(img, vig)
        img = fx.grain(img, grain, c.frame)
        return img
    return f


# ------------------------------------------------------------ backgrounds ---
def S_bg(kind="radial", c1=PAL["deep_red"], c2=PAL["black"], speed=1.0, fxs=()):
    """Animated backgrounds: radial | stripes | grid | sunburst."""
    def f(c):
        H, W = c.H, c.W
        yy, xx = fx._grid(H, W)
        if kind == "radial":
            d = np.sqrt(((xx - W / 2) / W) ** 2 + ((yy - H / 2) / W) ** 2)
            m = np.clip(1 - d * 1.9, 0, 1)[..., None]
            img = np.asarray(c2, np.float32) + (np.asarray(c1, np.float32) - np.asarray(c2, np.float32)) * m
        elif kind == "stripes":
            u = ((xx + yy) / (H * 0.12) - c.lt * 2 * speed) % 1.0
            m = (u < 0.5)[..., None]
            img = np.where(m, np.asarray(c1, np.float32), np.asarray(c2, np.float32)).astype(np.float32)
        elif kind == "sunburst":
            a = np.arctan2(yy - H / 2, xx - W / 2) + c.lt * 0.6 * speed
            m = ((a / (2 * np.pi) * 24) % 1.0 < 0.5)[..., None]
            img = np.where(m, np.asarray(c1, np.float32), np.asarray(c2, np.float32)).astype(np.float32)
        else:  # grid
            img = blank(W, H, c2)
            step = int(H / 12)
            off = int(c.lt * 40 * speed) % step
            img[(np.arange(H) + off) % step == 0] = c1
            img[:, (np.arange(W) + off) % step == 0] = c1
        for g in fxs:
            img = g(c, img)
        return img
    return f


def S_triptych(names, grades=("red", "bw", "blue"), hits=None, zoom=(1.1, 1.25), centers=None, gap=0.006):
    """Three vertical photo strips; strip k slams in (from alternating sides) at hits[k]."""
    def f(c):
        H, W = c.H, c.W
        img = blank(W, H, BLACK)
        n = len(names)
        hs = hits or [k * c.dur / (n + 1) for k in range(n)]
        sw = W // n
        for k, nm in enumerate(names):
            lt = c.lt - hs[k]
            if lt < 0:
                continue
            ctr = centers[k] if centers else (0.5, 0.4)
            ph = _photo_or_placeholder(nm, sw, H, c.p, zoom=zoom, center=(ctr, ctr))
            if grades[k % len(grades)]:
                ph = fx.grade(ph, grades[k % len(grades)], 1.4)
            e = ease_out_expo(min(lt / 0.18, 1))
            off = int((1 - e) * H * (1 if k % 2 else -1))
            y0, y1 = max(off, 0), min(H + off, H)
            g = int(gap * W)
            img[y0:y1, k * sw + g:(k + 1) * sw - g] = ph[y0 - off:y1 - off, g:sw - g]
        return img
    return f


def S_cards(names, hits, bg=None, grades=None, size=0.62, seed=5):
    """Photo cards (white border) dropping onto a background with random tilt, one per hit."""
    def f(c):
        H, W = c.H, c.W
        img = bg(c) if bg else blank(W, H, BLACK)
        rng = np.random.default_rng(seed)
        for k, nm in enumerate(names):
            r = rng.uniform(-9, 9); ox = rng.uniform(-0.18, 0.18); oy = rng.uniform(-0.1, 0.1)
            lt = c.lt - hits[k]
            if lt < 0:
                continue
            h = int(H * size); w = int(h * 1.3)
            try:
                card = media.card(nm, w, h, zoom=1.05, border=max(4, h // 40), border_color=WHITE)
            except FileNotFoundError:
                card = np.ones((h, w, 4), np.float32); card[..., :3] = (0.3, 0.1, 0.1)
            if grades:
                card[..., :3] = fx.grade(card[..., :3], grades[k % len(grades)], 1.3)
            e = ease_out_expo(min(lt / 0.16, 1))
            s = 1 + 1.6 * (1 - e)
            tx.place(img, card, 0.5 + ox, 0.5 + oy, s, r * (2 - e))
        return img
    return f


def O_idtag(box=(0.35, 0.15, 0.3, 0.5), lines=("JOSIP BROZ TITO", "1892 — 1980"), at=0.0, color=RED, side="r"):
    """Freeze-frame style ID tag: corner brackets snap onto a box, label slides out beside it."""
    def f(c, img):
        lt = c.lt - at
        if lt < 0:
            return img
        H, W = c.H, c.W
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        e = ease_out_expo(min(lt / 0.2, 1))
        grow = (1 - e) * 0.25
        x0, y0 = x - w * grow, y - h * grow
        x1, y1 = x + w * (1 + grow), y + h * (1 + grow)
        im = to_pil(img)
        d = ImageDraw.Draw(im)
        L = int(min(w, h) * 0.18)
        lw = max(2, int(H / 180))
        col = tuple(int(v * 255) for v in color)
        for (cx_, cy_, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            d.line([(cx_, cy_), (cx_ + sx * L, cy_)], fill=col, width=lw)
            d.line([(cx_, cy_), (cx_, cy_ + sy * L)], fill=col, width=lw)
        img = from_pil(im)
        q = ease_out_expo(min(max(lt - 0.12, 0) / 0.25, 1))
        if q > 0:
            tx_x = (x1 + W * 0.02) if side == "r" else (x0 - W * 0.02)
            for k, s in enumerate(lines):
                Lt = tx.text_layer(s, "anton" if k == 0 else "mono", int(H * (0.075 if k == 0 else 0.032)),
                                   WHITE if k == 0 else color, tracking=0.02 if k == 0 else 0.15)
                ww = Lt.shape[1] / W
                cx_ = (tx_x / W + ww / 2) if side == "r" else (tx_x / W - ww / 2)
                cx_ += (0.05 * (1 - q)) * (1 if side == "r" else -1)
                tx.place(img, Lt, cx_, y1 / H - 0.12 + k * 0.08, opacity=q)
        return img
    return f


def S_mosaic(names, cols=4, rows=3, fill_dur=None, grades=("bw", "red", "blue", "gold"), seed=11):
    """Wall of photos filling cell by cell (random order) over fill_dur."""
    def f(c):
        H, W = c.H, c.W
        img = blank(W, H, BLACK)
        n = cols * rows
        order = np.random.default_rng(seed).permutation(n)
        fd = fill_dur or c.dur * 0.8
        shown = int(min(c.lt / fd, 1) * n) + 1
        cw, ch = W // cols, H // rows
        for k in range(min(shown, n)):
            cell = order[k]
            i, j = cell % cols, cell // cols
            nm = names[(cell * 7 + seed) % len(names)]
            ph = _photo_or_placeholder(nm, cw, ch, c.p, zoom=(1.1, 1.3))
            ph = fx.grade(ph, grades[cell % len(grades)], 1.4)
            if k == shown - 1:
                ph = fx.flash(ph, WHITE, 0.6)
            img[j * ch:(j + 1) * ch, i * cw:(i + 1) * cw] = ph
        return img
    return f


def S_band(name, grade=None, contrast=1.3, zoom=(1.0, 1.08), bg_grade="blue", blur_r=30):
    """Wide panorama shown as a full-width band over a blurred, darkened cover of itself."""
    def f(c):
        H, W = c.H, c.W
        bg = _photo_or_placeholder(name, W, H, c.p, zoom=(1.0, 1.0))
        bg = fx.blur_fast(bg, blur_r * W / 1920, 4)
        bg = fx.grade(bg, bg_grade, 1.0) * 0.45
        im = media.load(name)
        z = zoom[0] + (zoom[1] - zoom[0]) * c.p
        bw = int(W * z)
        bh = int(im.height * bw / im.width)
        band = from_pil(im.resize((bw, bh), Image.BICUBIC))
        if grade:
            band = fx.grade(band, grade, contrast)
        x0 = (W - bw) // 2
        y0 = (H - bh) // 2
        xs0, xs1 = max(0, -x0), min(bw, W - x0)
        bg[max(y0, 0):y0 + bh, max(x0, 0):max(x0, 0) + (xs1 - xs0)] = band[max(0, -y0):max(0, -y0) + min(bh, H), xs0:xs1]
        return bg
    return f


def O_impact_frames(times, dur=2 / 30, colors=((PAL["red"], PAL["black"]), (PAL["white"], PAL["black"]))):
    """Manga impact frames: for `dur` after each time (global s), the frame becomes a
    2-tone threshold (alternating palettes), with a slight zoom."""
    ts = sorted(times)

    def f(c, img):
        import bisect
        i = bisect.bisect_right(ts, c.t + 1e-6) - 1
        if i < 0 or c.t - ts[i] >= dur:
            return img
        light, dark = colors[i % len(colors)]
        out = fx.threshold(img, float(np.median(fx.luma(img))), dark, light)
        return fx.affine(out, zoom=1.06)
    return f


def O_cowbell_tick(amt=5.0, decay=0.05):
    """Tiny chroma kick on every cowbell note (keeps the frame alive between drums)."""
    def f(c, img):
        v = c.bm.pulse(c.bm.cowbell, c.t, decay) if c.bm.cowbell else 0
        return fx.rgb_split(img, amt * v * c.W / 1920, 1.2) if v > 0.15 else img
    return f
