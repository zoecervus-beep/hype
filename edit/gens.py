"""Lazy access to generator modules written by the agents (placeholder if missing)."""
from __future__ import annotations

import importlib
import traceback

import numpy as np

from engine import text as tx
from engine.core import PAL

_warned = set()


def _placeholder(label, W, H):
    out = np.zeros((H, W, 4), np.float32)
    L = tx.text_layer(f"<{label}>", "mono", int(H * 0.04), PAL["cyan"])
    h, w = L.shape[:2]
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w] = L
    return out


def G(mod, fn):
    """Return a generator fn(t, dur, W, H, **kw) resolved lazily from gen.<mod>."""
    def call(t, dur, W, H, **kw):
        try:
            f = getattr(importlib.import_module(f"gen.{mod}"), fn)
        except Exception:
            if (mod, fn) not in _warned:
                _warned.add((mod, fn))
                traceback.print_exc(limit=1)
            return _placeholder(f"{mod}.{fn}", W, H)
        return f(t, dur, W, H, **kw)
    call.__name__ = f"{mod}.{fn}"
    return call


yugo_map = G("maps", "yugo_map")
world_nam = G("maps", "world_nam")
europe_blocs = G("maps", "europe_blocs")
render_model = G("spomenik3d", "render_model")
flag = G("emblems", "flag")
coat_of_arms = G("emblems", "coat_of_arms")
star2d = G("emblems", "star2d")
passport = G("emblems", "passport")
fico_drift = G("emblems", "fico_drift")
test_card = G("emblems", "test_card")
vinyl = G("emblems", "vinyl")
snowflake = G("emblems", "snowflake")
basketball = G("emblems", "basketball")
