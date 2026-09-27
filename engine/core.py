"""Shared conventions for every module in the edit.

Image convention
----------------
* RGB frames:  float32 numpy array, shape (H, W, 3), values in [0, 1] (sRGB-ish).
* RGBA layers: float32 numpy array, shape (H, W, 4), STRAIGHT (non-premultiplied)
  alpha, values in [0, 1].
* Generators are pure functions of time: ``fn(t, dur, W, H, **params) -> RGBA``
  where ``t`` is seconds since the shot started and ``dur`` the shot length.
  They must be deterministic for a given (t, params) because frames are rendered
  out of order in several processes. Cache static geometry at module level.
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
FONTS = os.path.join(ASSETS, "fonts")
OUT = os.path.join(ROOT, "out")


def hexrgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------- palette ---
PAL = {
    "red": hexrgb("#E3121E"),       # hot flag red
    "deep_red": hexrgb("#6E0610"),  # shadows / backgrounds
    "blue": hexrgb("#1B3FAE"),      # flag blue
    "navy": hexrgb("#08112E"),
    "gold": hexrgb("#F5C22B"),      # star border / emblem gold
    "white": hexrgb("#F4F0E6"),     # warm paper white
    "black": hexrgb("#060608"),
    "cyan": hexrgb("#2FF3FF"),      # sparing neon accent (glitch)
    "magenta": hexrgb("#FF2E7E"),   # sparing neon accent (glitch)
    "steel": hexrgb("#8C97A8"),     # concrete / spomenik grey
}

# ------------------------------------------------------------------ fonts ---
# All of these cover Latin Extended (ČĆĐŠŽ). Cyrillic-capable ones are marked *.
FONT_FILES = {
    "anton": "Anton-Regular.ttf",            # ultra condensed impact display
    "bebas": "BebasNeue-Regular.ttf",        # condensed caps
    "archivo": "ArchivoBlack-Regular.ttf",   # heavy grotesk
    "russo": "RussoOne-Regular.ttf",         # * soviet-ish geometric
    "oswald": "Oswald[wght].ttf",            # * variable 200-700
    "unbounded": "Unbounded[wght].ttf",      # * wide variable 200-900
    "rubikmono": "RubikMonoOne-Regular.ttf", # * wide mono heavy
    "mono": "IBMPlexMono-Bold.ttf",          # * typewriter / data
    "pixel": "PressStart2P-Regular.ttf",     # * 8-bit
    "serif": "PlayfairDisplay[wght].ttf",    # * elegant serif
    "tektur": "Tektur[wdth,wght].ttf",       # * techno
    "glitch": "RubikGlitch-Regular.ttf",     # * glitch display
    "noto": "NotoSans[wdth,wght].ttf",       # * fallback
}

_VAR_DEFAULTS = {"oswald": [700], "unbounded": [900], "serif": [900],
                 "tektur": [100, 900], "noto": [100, 900]}


@functools.lru_cache(maxsize=256)
def font(name: str, size: int, axes: tuple | None = None) -> ImageFont.FreeTypeFont:
    """Load a font by short name (see FONT_FILES) at pixel size."""
    f = ImageFont.truetype(os.path.join(FONTS, FONT_FILES[name]), int(size))
    ax = list(axes) if axes is not None else _VAR_DEFAULTS.get(name)
    if ax:
        try:
            f.set_variation_by_axes(ax)
        except Exception:
            pass
    return f


# ---------------------------------------------------------------- helpers ---
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def blank(W: int, H: int, rgb=(0, 0, 0)) -> np.ndarray:
    out = np.empty((H, W, 3), np.float32)
    out[:] = rgb
    return out


def blank_rgba(W: int, H: int) -> np.ndarray:
    return np.zeros((H, W, 4), np.float32)


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def from_pil(im: Image.Image) -> np.ndarray:
    """PIL image -> float32 RGB or RGBA array."""
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.mode else "RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(to_u8(arr), "RGBA" if arr.shape[2] == 4 else "RGB")


def over(dst: np.ndarray, src: np.ndarray, x: int = 0, y: int = 0,
         opacity: float = 1.0, mode: str = "normal") -> np.ndarray:
    """Composite RGBA ``src`` onto RGB ``dst`` (in place) with top-left at (x, y).

    mode: normal | add | screen | multiply | difference
    """
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return dst
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    a = s[..., 3:4] * opacity if s.shape[2] == 4 else np.float32(opacity)
    c = s[..., :3]
    if mode == "normal":
        blended = c
    elif mode == "add":
        blended = np.minimum(d + c, 1.0)
    elif mode == "screen":
        blended = 1.0 - (1.0 - d) * (1.0 - c)
    elif mode == "multiply":
        blended = d * c
    elif mode == "difference":
        blended = np.abs(d - c)
    else:
        raise ValueError(mode)
    d += (blended - d) * a
    return dst


def ease_out_cubic(p):
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** 3


def ease_in_cubic(p):
    p = min(max(p, 0.0), 1.0)
    return p ** 3


def ease_in_out_cubic(p):
    p = min(max(p, 0.0), 1.0)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out_expo(p):
    p = min(max(p, 0.0), 1.0)
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def ease_out_back(p, s=1.70158):
    p = min(max(p, 0.0), 1.0) - 1
    return p * p * ((s + 1) * p + s) + 1


def lerp(a, b, p):
    return a + (b - a) * p


def star_points(cx, cy, r_outer, r_inner=None, rot=0.0, n=5):
    """Vertices of an n-pointed star (point up at rot=0)."""
    if r_inner is None:
        r_inner = r_outer * 0.382
    pts = []
    for i in range(2 * n):
        r = r_outer if i % 2 == 0 else r_inner
        a = rot - math.pi / 2 + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts
