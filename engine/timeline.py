"""Timeline: a single cut track of shots + time-ranged ops (overlays / post FX)."""
from __future__ import annotations

import bisect
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .core import OUT

BPM = 130.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT


def b(n: float) -> float:
    """Time of absolute beat n."""
    return n * BEAT


def bar(n: float) -> float:
    return n * BAR


class Beatmap:
    def __init__(self, path=os.path.join(OUT, "audio", "beatmap.json")):
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)
        ev = self.data.get("events", {})
        grid_k = [b(n) for n in range(32, 176) if n % 4 in (0,)]
        self.kick = sorted(ev.get("kick", grid_k))
        self.snare = sorted(ev.get("snare", [b(n) for n in range(32, 176) if n % 4 in (1, 3)]))
        self.hat = sorted(ev.get("hat", []))
        self.cowbell = sorted(e["t"] if isinstance(e, dict) else e for e in ev.get("cowbell", []))
        self.impact = sorted(ev.get("impact", [b(32), b(112), b(176)]))
        self.crash = sorted(ev.get("crash", []))
        self.countdown = ev.get("countdown", [b(n) for n in (16, 18, 20, 22, 24, 26, 28)])
        en = self.data.get("energy", {})
        self.energy = {k: np.asarray(v, np.float32) for k, v in en.items() if isinstance(v, list)}
        self.energy_fps = en.get("fps", 30)
        self.duration = self.data.get("duration", 86.0)

    @staticmethod
    def since(events, t):
        """Seconds since the latest event <= t (inf if none)."""
        i = bisect.bisect_right(events, t + 1e-6) - 1
        return t - events[i] if i >= 0 else math.inf

    @staticmethod
    def last_index(events, t):
        return bisect.bisect_right(events, t + 1e-6) - 1

    def pulse(self, events, t, decay=0.12):
        """1.0 at each event, decaying exponentially."""
        d = self.since(events, t)
        return math.exp(-d / decay) if d != math.inf else 0.0

    def en(self, t, key="total"):
        arr = self.energy.get(key)
        if arr is None or len(arr) == 0:
            return 0.5
        i = min(max(int(t * self.energy_fps), 0), len(arr) - 1)
        return float(arr[i])


@dataclass
class Ctx:
    t: float          # global time
    lt: float         # local time inside the shot/op
    dur: float
    W: int
    H: int
    frame: int
    bm: Beatmap
    seed: int = 0

    @property
    def p(self):
        return min(max(self.lt / self.dur, 0.0), 1.0) if self.dur > 0 else 1.0

    def at(self, t0, dur):
        """Derived context for a sub-clip starting at local time t0."""
        return Ctx(self.t, self.lt - t0, dur, self.W, self.H, self.frame, self.bm, self.seed)


@dataclass(order=True)
class Op:
    z: float
    t0: float
    t1: float
    fn: object = field(compare=False)
    name: str = field(default="", compare=False)


class Timeline:
    def __init__(self, W=1920, H=1080, fps=30, duration=86.0):
        self.W, self.H, self.fps, self.duration = W, H, fps, duration
        self.shots = []   # (t0, t1, fn, name)
        self.ops = []     # Op
        self.bm = Beatmap()

    # --- authoring
    def shot(self, t0, t1, fn, name=""):
        self.shots.append((t0, t1, fn, name))
        return fn

    def op(self, t0, t1, fn, z=10, name=""):
        self.ops.append(Op(z, t0, t1, fn, name))
        return fn

    def finalize(self):
        self.shots.sort(key=lambda s: s[0])
        self.ops.sort()
        self._starts = [s[0] for s in self.shots]

    # --- rendering
    def render(self, frame_idx, scale=1.0):
        W, H = int(self.W * scale) // 2 * 2, int(self.H * scale) // 2 * 2
        t = frame_idx / self.fps
        i = bisect.bisect_right(self._starts, t + 1e-9) - 1
        img = None
        if i >= 0:
            t0, t1, fn, _ = self.shots[i]
            if t < t1:
                c = Ctx(t, t - t0, t1 - t0, W, H, frame_idx, self.bm, seed=i * 131)
                img = fn(c)
        if img is None:
            img = np.zeros((H, W, 3), np.float32)
        for k, o in enumerate(self.ops):
            if o.t0 <= t < o.t1:
                c = Ctx(t, t - o.t0, o.t1 - o.t0, W, H, frame_idx, self.bm, seed=k * 977)
                out = o.fn(c, img)
                if out is not None:
                    img = out
        return img
