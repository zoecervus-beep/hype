"""gen/spomenik3d.py -- tiny numpy software 3D renderer + procedural spomenik models.

No GPU / OpenGL: vertices are transformed with numpy, faces are back-face culled,
depth sorted (painter's algorithm) and rasterised with PIL ImageDraw polygons into
a supersampled raw 4-channel canvas (premultiplied RGBA stored in a "CMYK"-mode
image so PIL never touches the alpha while drawing or reducing).  Glow is computed
on a quarter-resolution copy and upsampled.

Generator (engine/core.py convention, float32 straight-alpha RGBA (H, W, 4)):

    render_model(t, dur, W, H, model="tjentiste", style="solid", yaw0=0.0, spin=0.6,
                 pitch=-0.15, zoom=1.0, cx=0.5, cy=0.55, color=None, edge=None,
                 glow=1.0, **extra)

Models (all ~1 unit tall, standing on y=0, centred on the vertical axis):
    tjentiste, kosmaj, jasenovac, makedonium, petrova_gora, star
Styles:
    solid, wire, neon, hologram, xray
"""
from __future__ import annotations

import functools
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import ConvexHull

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.core import PAL, OUT, font, hexrgb  # noqa: E402

MODELS = ("tjentiste", "kosmaj", "jasenovac", "makedonium", "petrova_gora", "star")
STYLES = ("solid", "wire", "neon", "hologram", "xray")

# materials
M_CONCRETE, M_GLASS, M_RED, M_GOLD, M_STEEL, M_INNER = range(6)


# =============================================================== mesh build ===
class Mesh:
    """Polygon mesh: vertices (N,3), faces padded to (F,K) + counts, per-face data."""

    def __init__(self, V, faces, mats, two, name=""):
        self.name = name
        self.V = np.asarray(V, np.float64)
        F = len(faces)
        K = max(len(f) for f in faces)
        idx = np.zeros((F, K), np.int64)
        cnt = np.zeros(F, np.int64)
        for i, f in enumerate(faces):
            idx[i, :len(f)] = f
            idx[i, len(f):] = f[-1]
            cnt[i] = len(f)
        self.fidx, self.fcnt = idx, cnt
        self.faces = [list(f) for f in faces]
        self.mat = np.asarray(mats, np.int64)
        self.two = np.asarray(two, bool)
        P = self.V[idx]                                   # (F,K,3)
        Q = np.roll(P, -1, axis=1)
        # Newell normal (padding repeats the last vertex -> zero contribution)
        n = np.stack([
            ((P[..., 1] - Q[..., 1]) * (P[..., 2] + Q[..., 2])).sum(1),
            ((P[..., 2] - Q[..., 2]) * (P[..., 0] + Q[..., 0])).sum(1),
            ((P[..., 0] - Q[..., 0]) * (P[..., 1] + Q[..., 1])).sum(1)], -1)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        self.area = ln[:, 0] * 0.5
        self.N = n / np.maximum(ln, 1e-12)
        w = (np.arange(K)[None, :] < cnt[:, None]).astype(np.float64)
        self.C = (P * w[..., None]).sum(1) / cnt[:, None]
        # unique undirected edges
        es = set()
        for f in faces:
            for a, b in zip(f, f[1:] + f[:1]):
                if a != b:
                    es.add((min(a, b), max(a, b)))
        self.edges = np.array(sorted(es), np.int64)
        # deterministic per-face tone jitter
        self.tone = (np.sin(np.arange(F) * 12.9898 + 78.233) * 43758.5453) % 1.0
        # polygon index lists for the draw loop
        self.polys = [np.asarray(f, np.int64) for f in faces]
        self.ymin, self.ymax = self.V[:, 1].min(), self.V[:, 1].max()


class MB:
    """Mesh builder."""

    def __init__(self):
        self.V, self.F, self.M, self.T = [], [], [], []

    def verts(self, pts):
        base = len(self.V)
        self.V.extend([tuple(map(float, p)) for p in pts])
        return list(range(base, base + len(pts)))

    def face(self, idx, mat=M_CONCRETE, two=False):
        # drop consecutive duplicates
        out = []
        for i in idx:
            if not out or out[-1] != i:
                out.append(i)
        if len(out) > 1 and out[0] == out[-1]:
            out.pop()
        if len(out) >= 3:
            self.F.append(out)
            self.M.append(mat)
            self.T.append(two)

    def hull(self, pts, mat=M_CONCRETE, mats_fn=None):
        """Add the convex hull of pts, merging coplanar triangles into polygons."""
        pts = np.asarray(pts, np.float64)
        h = ConvexHull(pts)
        groups = []                      # [normal, offset, set(vertex ids)]
        for simp, eq in zip(h.simplices, h.equations):
            n, d = eq[:3], eq[3]
            for g in groups:
                if np.dot(g[0], n) > 1 - 1e-7 and abs(g[1] - d) < 1e-6:
                    g[2].update(simp.tolist())
                    break
            else:
                groups.append([n, d, set(simp.tolist())])
        vmap = {}
        for n, d, ids in groups:
            ids = list(ids)
            P = pts[ids]
            c = P.mean(0)
            # in-plane basis with u x v = n  ->  increasing angle is CCW about n
            a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(n, a)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            ang = np.arctan2((P - c) @ v, (P - c) @ u)
            order = [ids[i] for i in np.argsort(ang)]
            gi = []
            for i in order:
                if i not in vmap:
                    vmap[i] = self.verts([pts[i]])[0]
                gi.append(vmap[i])
            m = mats_fn(n, c) if mats_fn else mat
            self.face(gi, m)

    def grid(self, P, mat=M_CONCRETE, wrap=True, two=False, flip=False):
        """Quad-strip surface. P: (nv, nu, 3), rows bottom->top, u = angle (CCW
        seen from above for surfaces of revolution). Returns index grid."""
        nv, nu = P.shape[:2]
        ids = np.array(self.verts(P.reshape(-1, 3))).reshape(nv, nu)
        for i in range(nv - 1):
            for j in range(nu if wrap else nu - 1):
                j1 = (j + 1) % nu
                q = [ids[i, j], ids[i + 1, j], ids[i + 1, j1], ids[i, j1]]
                if flip:
                    q = q[::-1]
                self.face(q, mat, two)
        return ids

    def fan(self, ring_ids, center, mat=M_CONCRETE, up=True, two=False):
        c = self.verts([center])[0]
        n = len(ring_ids)
        for j in range(n):
            a, b = ring_ids[j], ring_ids[(j + 1) % n]
            self.face([c, b, a] if up else [c, a, b], mat, two)

    def build(self, name=""):
        return Mesh(self.V, self.F, self.M, self.T, name)


def _normalize(mb: MB, height=1.0, max_radius=None):
    """Scale so the model spans y in [0, height] (optionally cap horizontal radius)."""
    V = np.asarray(mb.V)
    y0, y1 = V[:, 1].min(), V[:, 1].max()
    s = height / (y1 - y0)
    if max_radius is not None:
        r = np.sqrt(V[:, 0] ** 2 + V[:, 2] ** 2).max() * s
        if r > max_radius:
            s *= max_radius / r
    V = (V - [0, y0, 0]) * s
    mb.V = [tuple(p) for p in V]
    return mb


# ------------------------------------------------------------------ models ---
def _shard(rng, cx, cz, rx, rz, h, lx, lz, n=6, twist=0.25, taper=(0.9, 0.62),
           apex=(0.0, 0.0), chisel=0.0, xmin=None, xsign=1):
    """Crystalline shard: lofted irregular polygon rings + pointed / chiselled top."""
    base_ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi)
    jit = rng.uniform(-0.28, 0.28, n) * (2 * np.pi / n)
    rj = rng.uniform(0.82, 1.12, n)
    pts = []
    for fy, sc in ((0.0, 1.0), (0.48, taper[0]), (0.8, taper[1])):
        y = fy * h
        a = base_ang + jit + twist * fy
        rj2 = rj * rng.uniform(0.92, 1.08, n)
        x = cx + lx * fy + rx * sc * rj2 * np.cos(a)
        z = cz + lz * fy + rz * sc * rj2 * np.sin(a)
        if xmin is not None:
            x = np.maximum(x, xmin) if xsign > 0 else np.minimum(x, -xmin)
        pts += list(zip(x, np.full(n, y), z))
    ax, az = cx + lx + apex[0], cz + lz + apex[1]
    if chisel > 0:   # two top points -> chisel ridge
        pts += [(ax - chisel * 0.5, h, az + chisel * 0.3), (ax + chisel * 0.5, h * 0.93, az - chisel * 0.3)]
    else:
        pts.append((ax, h, az))
    return np.array(pts)


def _model_tjentiste():
    """Sutjeska memorial: two mirrored massive faceted rock-like wings with a narrow
    passage between them; each wing is a heavy base mass whose top splits into
    jagged slab-like fingers, tallest next to the passage."""
    mb = MB()
    gap = 0.05
    parts = []
    # heavy lower masses (sloping down away from the passage)
    parts.append(np.array([
        (gap, 0, -0.36), (gap, 0, 0.36), (0.20, 0, 0.44), (0.40, 0, 0.18), (0.42, 0, -0.14), (0.22, 0, -0.44),
        (gap, 0.60, -0.20), (gap, 0.66, 0.16), (0.13, 0.52, 0.30), (0.27, 0.30, 0.12), (0.26, 0.34, -0.22),
        (0.12, 0.48, -0.32)]))
    parts.append(np.array([
        (0.20, 0, -0.10), (0.20, 0, 0.30), (0.50, 0, 0.14), (0.48, 0, -0.08),
        (0.26, 0.40, 0.04), (0.25, 0.36, 0.22), (0.40, 0.16, 0.10), (0.39, 0.18, -0.02)]))
    # slab fingers: (cx, cz, rx, rz, y0, h, lean_x, lean_z, apex_dx, apex_dz, chisel)
    spec = [
        (0.10, 0.02, 0.060, 0.17, 0.30, 1.00, 0.07, 0.03, 0.02, 0.06, 0.07),
        (0.10, -0.20, 0.055, 0.12, 0.25, 0.86, 0.07, -0.06, 0.02, -0.03, 0.05),
        (0.10, 0.25, 0.055, 0.10, 0.25, 0.92, 0.06, 0.07, 0.00, 0.04, 0.0),
        (0.19, 0.10, 0.060, 0.12, 0.25, 0.80, 0.10, 0.04, 0.03, 0.02, 0.06),
        (0.20, -0.10, 0.065, 0.13, 0.20, 0.70, 0.11, -0.03, 0.02, -0.02, 0.0),
        (0.28, 0.18, 0.060, 0.10, 0.10, 0.56, 0.12, 0.05, 0.03, 0.0, 0.05),
        (0.29, -0.22, 0.060, 0.10, 0.10, 0.48, 0.12, -0.05, 0.0, -0.02, 0.0),
    ]
    rng = np.random.default_rng(1971)
    for (cx, cz, rx, rz, y0, h, lx, lz, ax, az, ch) in spec:
        P = _shard(rng, cx, cz, rx, rz, h - y0, lx, lz, n=5, twist=0.15, taper=(0.97, 0.85),
                   apex=(ax, az), chisel=ch, xmin=gap)
        P[:, 1] += y0
        parts.append(P)
    for P in parts:
        mb.hull(P)
        Q = P.copy()
        Q[:, 0] *= -1
        mb.hull(Q)
    return _normalize(mb, 1.0).build("tjentiste")


def _model_kosmaj():
    """Kosmaj: five tall sharp blades leaning outwards, star-shaped plan, joined at a core."""
    mb = MB()
    for i in range(5):
        phi = math.pi / 2 + i * 2 * math.pi / 5
        er = np.array([math.cos(phi), 0, math.sin(phi)])
        et = np.array([-math.sin(phi), 0, math.cos(phi)])

        def E(r, y, t):
            return r * er + np.array([0, y, 0]) + t * et
        # lower blade: rises steeply from the core foot
        low = [E(0.02, 0, 0.06), E(0.02, 0, -0.06), E(0.26, 0, 0.035), E(0.26, 0, -0.035), E(0.33, 0, 0.0),
               E(0.03, 0.50, 0.05), E(0.03, 0.50, -0.05), E(0.40, 0.46, 0.0),
               E(0.30, 0.47, 0.03), E(0.30, 0.47, -0.03)]
        mb.hull(low)
        # upper blade: leans outwards to a needle tip
        up = [E(0.03, 0.44, 0.05), E(0.03, 0.44, -0.05), E(0.30, 0.42, 0.03), E(0.30, 0.42, -0.03),
              E(0.40, 0.42, 0.0), E(0.64, 1.0, 0.0), E(0.10, 0.60, 0.02), E(0.10, 0.60, -0.02)]
        mb.hull(up)
    core = []
    for k in range(5):
        a = math.pi / 2 + (k + 0.5) * 2 * math.pi / 5
        for y, r in ((0.0, 0.08), (0.55, 0.06)):
            core.append((r * math.cos(a), y, r * math.sin(a)))
    core.append((0, 0.62, 0))
    mb.hull(core)
    return _normalize(mb, 1.0).build("kosmaj")


def _lobe(theta, n, sharp=0.55):
    """0..1 lobe profile with n rounded petals and pointed notches between."""
    return np.abs(np.cos(theta * n / 2.0)) ** sharp


def _model_jasenovac():
    """Stone Flower: narrow waist flaring into an open bloom of scalloped petals,
    with an inner ring of petals. Open shells rendered two-sided."""
    mb = MB()
    nu = 72
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    # outer profile control points (r, y)
    prof = np.array([(0.30, 0.00), (0.24, 0.05), (0.17, 0.13), (0.13, 0.22), (0.12, 0.30),
                     (0.14, 0.40), (0.18, 0.50), (0.24, 0.60), (0.32, 0.70), (0.41, 0.79),
                     (0.50, 0.87), (0.58, 0.93), (0.64, 0.97), (0.68, 1.0)])
    nv = len(prof)
    v = np.linspace(0, 1, nv)
    lob = _lobe(th + np.pi / 6, 6)
    P = np.zeros((nv, nu, 3))
    for i, (r, y) in enumerate(prof):
        w = np.clip((v[i] - 0.3) / 0.7, 0, 1) ** 1.5
        rr = r * (1 + 0.30 * w * (lob - 0.6))
        yy = y + 0.15 * w * (lob - 0.6)
        P[i, :, 0] = rr * np.cos(th)
        P[i, :, 1] = yy
        P[i, :, 2] = rr * np.sin(th)
    mb.grid(P, M_CONCRETE, wrap=True, two=True)
    # inner petals
    prof2 = np.array([(0.10, 0.28), (0.12, 0.40), (0.16, 0.52), (0.22, 0.63), (0.29, 0.72),
                      (0.35, 0.79), (0.39, 0.83)])
    nv2 = len(prof2)
    lob2 = _lobe(th, 6)
    P2 = np.zeros((nv2, nu, 3))
    for i, (r, y) in enumerate(prof2):
        w = (i / (nv2 - 1)) ** 1.3
        rr = r * (1 + 0.35 * w * (lob2 - 0.6))
        yy = y + 0.11 * w * (lob2 - 0.6)
        P2[i, :, 0] = rr * np.cos(th)
        P2[i, :, 1] = yy
        P2[i, :, 2] = rr * np.sin(th)
    mb.grid(P2, M_INNER, wrap=True, two=True)
    # base plinth (low octagon)
    base = []
    for k in range(8):
        a = k * np.pi / 4 + np.pi / 8
        base += [(0.42 * math.cos(a), -0.03, 0.42 * math.sin(a)), (0.40 * math.cos(a), 0.0, 0.40 * math.sin(a))]
    mb.hull(base)
    return _normalize(mb, 1.0).build("jasenovac")


def _fib_dirs(n, zmin=-0.25, zmax=0.95, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    ga = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y = zmax - (zmax - zmin) * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1 - y * y))
        a = i * ga + rng.uniform(-0.2, 0.2)
        out.append((r * math.cos(a), y, r * math.sin(a)))
    return np.array(out)


def _model_makedonium():
    """Makedonium: a dome with many short tubular spikes ending in round windows."""
    mb = MB()
    R, cy = 0.46, 0.36
    pts = []
    nlon = 20
    for lat in np.linspace(0, math.acos(-cy / R), 10)[1:]:
        for k in range(nlon):
            a = k * 2 * math.pi / nlon + (0.5 * (int(lat * 100) % 2)) * 0
            pts.append((R * math.sin(lat) * math.cos(a), cy + R * math.cos(lat), R * math.sin(lat) * math.sin(a)))
    pts.append((0, cy + R, 0))
    pts = np.array(pts)
    pts[:, 1] = np.maximum(pts[:, 1], 0.0)
    mb.hull(pts)
    dirs = _fib_dirs(17, zmin=-0.05, zmax=0.97)
    rng = np.random.default_rng(1974)
    nseg = 10
    for d in dirs:
        d = d / np.linalg.norm(d)
        a = np.array([0, 1.0, 0]) if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(d, a); u /= np.linalg.norm(u)
        w = np.cross(d, u)
        L = rng.uniform(0.07, 0.17)
        rb, re = rng.uniform(0.075, 0.095), rng.uniform(0.055, 0.07)
        c0 = np.array([0, cy, 0]) + d * R * 0.9
        c1 = np.array([0, cy, 0]) + d * (R + L)
        ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
        ring0 = [c0 + rb * (math.cos(t) * u + math.sin(t) * w) for t in ang]
        ring1 = [c1 + re * (math.cos(t) * u + math.sin(t) * w) for t in ang]
        tube = np.array(ring0 + ring1)
        tube[:, 1] = np.maximum(tube[:, 1], 0.0)
        mb.hull(tube)
        # window: dark disc just proud of the end cap (CCW about d)
        win = [c1 + d * 0.004 + 0.72 * re * (math.cos(t) * u + math.sin(t) * w) for t in ang]
        ids = mb.verts(win)
        mb.face(ids, M_GLASS)
    return _normalize(mb, 1.0).build("makedonium")


def _tri(x):
    """Triangle wave, period 1, range [-1, 1]."""
    f = x - np.floor(x)
    return 4 * np.abs(f - 0.5) - 1


def _model_petrova_gora():
    """Petrova Gora: tall organic tower of folded vertical ribs over a few large
    bulging lobes, wider at the bottom, with a curved domed top."""
    mb = MB()
    nu, nv = 96, 22
    th = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    lobe = 1 + 0.13 * np.cos(5 * th + 0.4) + 0.05 * np.cos(3 * th + 1.1)
    Ht = 0.78 + 0.13 * np.cos(th - 0.5) + 0.06 * np.cos(5 * th + 0.4)
    ribs = 26
    P = np.zeros((nv, nu, 3))
    for i in range(nv):
        v = i / (nv - 1)
        if v <= 0.78:
            q = v / 0.78
            rf = 1.0 - 0.36 * q ** 1.25 + 0.05 * math.sin(math.pi * q)
            yf = 0.82 * q
        else:
            ph = (v - 0.78) / 0.22 * (math.pi / 2) * 0.93
            rf0 = 1.0 - 0.36 + 0.0
            rf = rf0 * math.cos(ph)
            yf = 0.82 + 0.18 * math.sin(ph)
        wave = 0.08 * math.sin(v * 6.0)
        fold = _tri(ribs * th / (2 * np.pi) + wave * ribs / (2 * np.pi))
        r = 0.36 * rf * lobe * (1 + 0.045 * fold)
        P[i, :, 0] = r * 1.1 * np.cos(th)
        P[i, :, 2] = r * 0.9 * np.sin(th)
        P[i, :, 1] = yf * Ht
    ids = mb.grid(P, M_STEEL, wrap=True)
    top = P[-1].mean(0)
    top[1] = P[-1, :, 1].max() + 0.005
    mb.fan(list(ids[-1]), top, M_STEEL, up=True)
    mb.fan(list(ids[0]), (0, 0, 0), M_STEEL, up=False)
    return _normalize(mb, 1.0).build("petrova_gora")


def offset_polygon(pts, d):
    """Offset a closed CCW/CW 2-D polygon by d (negative = inwards for CCW in y-up)."""
    P = np.asarray(pts, np.float64)
    n = len(P)
    out = []
    # signed area to know orientation
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    s = 1.0 if area > 0 else -1.0
    for i in range(n):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % n]
        e0 = p1 - p0
        e1 = p2 - p1
        n0 = np.array([e0[1], -e0[0]]) * s / np.linalg.norm(e0)
        n1 = np.array([e1[1], -e1[0]]) * s / np.linalg.norm(e1)
        # lines p0+n0*d + a*e0 and p1+n1*d + b*e1 intersect
        A = np.array([e0, -e1]).T
        rhs = (p1 + n1 * d) - (p0 + n0 * d)
        try:
            a = np.linalg.solve(A, rhs)[0]
            out.append(p0 + n0 * d + a * e0)
        except np.linalg.LinAlgError:
            out.append(p1 + n0 * d)
    return np.array(out)


def star2d_points(R=1.0, r_ratio=0.382, n=5):
    """Star outline in a y-up plane, point up, CCW."""
    pts = []
    for i in range(2 * n):
        r = R if i % 2 == 0 else R * r_ratio
        a = math.pi / 2 + i * math.pi / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return np.array(pts)


def _model_star():
    """Extruded, bevelled five-pointed red star with a gold rim + gold side band."""
    mb = MB()
    R = 0.55
    outer = star2d_points(R)                       # CCW in (x, y)
    inner = offset_polygon(outer, -0.045)          # inset for the gold border
    t, apex = 0.035, 0.13
    for side in (1, -1):
        z = side * t
        O = mb.verts([(x, y, z) for x, y in outer])
        I = mb.verts([(x, y, z + side * 0.004) for x, y in inner])
        A = mb.verts([(0, 0, side * apex)])[0]
        n = len(outer)
        for i in range(n):
            j = (i + 1) % n
            if side > 0:   # front faces +z, CCW seen from +z
                mb.face([A, I[i], I[j]], M_RED)
                mb.face([O[i], O[j], I[j], I[i]], M_GOLD)
            else:
                mb.face([A, I[j], I[i]], M_RED)
                mb.face([O[j], O[i], I[i], I[j]], M_GOLD)
        if side > 0:
            Of = O
        else:
            Ob = O
    n = len(outer)
    for i in range(n):
        j = (i + 1) % n
        mb.face([Of[i], Ob[i], Ob[j], Of[j]], M_GOLD)
    mb = _normalize(mb, 1.0)
    return mb.build("star")


_BUILDERS = {
    "tjentiste": _model_tjentiste,
    "kosmaj": _model_kosmaj,
    "jasenovac": _model_jasenovac,
    "makedonium": _model_makedonium,
    "petrova_gora": _model_petrova_gora,
    "star": _model_star,
}


@functools.lru_cache(maxsize=None)
def get_mesh(model: str) -> Mesh:
    if model not in _BUILDERS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    return _BUILDERS[model]()


# ================================================================ renderer ===
_CAM_D = 4.2          # camera distance (model units)
_FIT = 0.70           # model height as fraction of frame height at zoom=1


def _col(c, default):
    if c is None:
        return np.array(default, np.float64)
    if isinstance(c, str):
        return np.array(PAL[c] if c in PAL else hexrgb(c), np.float64)
    return np.array(c[:3], np.float64)


def _norm(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


_L_KEY = _norm((-0.55, 0.75, -0.55))     # towards key light (camera space: up-left-front)
_L_RIM = _norm((0.85, 0.30, 0.75))       # rim light from behind-right
_L_FILL = _norm((0.6, -0.1, -0.5))


def _hash01(*xs):
    s = 0.0
    for i, x in enumerate(xs):
        s += x * (12.9898 + 31.7 * i)
    return (math.sin(s) * 43758.5453) % 1.0


def _u8(c):
    return tuple(int(v) for v in np.clip(np.asarray(c) * 255.0 + 0.5, 0, 255))


def _prep(mesh, yaw, pitch, W, H, zoom, cx, cy):
    """Transform + project. Returns dict of camera-space data (full resolution px)."""
    cyw, syw = math.cos(yaw), math.sin(yaw)
    Ry = np.array([[cyw, 0, syw], [0, 1, 0], [-syw, 0, cyw]])
    e = -pitch                    # camera elevation: pitch<0 -> camera above, looking down
    ce, se = math.cos(e), math.sin(e)
    Rx = np.array([[1, 0, 0], [0, ce, se], [0, -se, ce]])
    R = Rx @ Ry
    tgt = np.array([0.0, 0.5 * (mesh.ymin + mesh.ymax), 0.0])
    Vc = (mesh.V - tgt) @ R.T
    Vc[:, 2] += _CAM_D
    Nc = mesh.N @ R.T
    Cc = (mesh.C - tgt) @ R.T
    Cc[:, 2] += _CAM_D
    f = zoom * _FIT * H * _CAM_D
    z = np.maximum(Vc[:, 2], 0.05)
    sx = cx * W + f * Vc[:, 0] / z
    sy = cy * H - f * Vc[:, 1] / z
    return Vc, Nc, Cc, np.stack([sx, sy], -1)


def _composite(canvas, ss, W, H, bx0, by0, glow_amt, glow_r, post=None):
    """canvas (CMYK raw premult RGBA at ss) -> full frame float32 straight RGBA."""
    img = canvas.reduce(ss) if ss > 1 else canvas
    a = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)
    h, w = a.shape[:2]
    P = a[..., :3]
    A = a[..., 3]
    if glow_amt > 0:
        k = 4
        small = img.reduce(k)
        sm = Image.fromarray(np.asarray(small)[..., :3].copy(), "RGB")
        r1, r2 = glow_r
        b1 = np.asarray(sm.filter(ImageFilter.GaussianBlur(max(0.5, r1 / k))), np.float32)
        b2 = np.asarray(sm.filter(ImageFilter.GaussianBlur(max(0.5, r2 / k))), np.float32)
        g = (b1 * 1.6 + b2 * 1.4) * (glow_amt / 255.0)
        g = np.clip(g, 0, 1)
        gi = Image.fromarray((g * 255 + 0.5).astype(np.uint8), "RGB").resize((w, h), Image.BILINEAR)
        G = np.asarray(gi, np.float32) * (1.0 / 255.0)
        P = P + G
        A = A + (1.0 - A) * G.max(axis=2)
    if post is not None:
        P, A = post(P, A, bx0, by0)
    out = np.zeros((H, W, 4), np.float32)
    x0, y0 = max(bx0, 0), max(by0, 0)
    x1, y1 = min(bx0 + w, W), min(by0 + h, H)
    if x1 <= x0 or y1 <= y0:
        return out
    Ac = A[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    Pc = P[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    out[y0:y1, x0:x1, :3] = np.minimum(Pc / np.maximum(Ac, 1e-4)[..., None], 1.0)
    out[y0:y1, x0:x1, 3] = np.clip(Ac, 0, 1)
    return out


def render_model(t, dur, W, H, model="tjentiste", style="solid", yaw0=0.0, spin=0.6,
                 pitch=-0.15, zoom=1.0, cx=0.5, cy=0.55, color=None, edge=None, glow=1.0,
                 line=1.0, fill_alpha=None, ss=2, fog=1.0, **_):
    """Render a spomenik model as a float32 straight-alpha RGBA layer (H, W, 4).

    t, dur     shot-local time / shot length (s). Deterministic in t.
    model      tjentiste | kosmaj | jasenovac | makedonium | petrova_gora | star
    style      solid | wire | neon | hologram | xray
    yaw0, spin initial yaw (rad) and rotation speed (rad/s) about the vertical axis
    pitch      camera tilt (rad); negative = camera above looking down, positive = low hero angle
    zoom       1.0 -> model is ~80% of frame height; cx, cy = screen position of model centre
    color      base/face colour (PAL key, '#hex' or rgb tuple); edge = edge colour
    glow       bloom multiplier (0 disables); line = edge width multiplier
    fill_alpha face opacity for wire/neon (wire default 0 = hidden-line wire only)
    ss         supersampling factor (2 = anti-aliased; 1 = fastest)
    """
    mesh = get_mesh(model)
    yaw = yaw0 + spin * t
    Vc, Nc, Cc, S = _prep(mesh, yaw, pitch, W, H, zoom, cx, cy)
    px = H / 1080.0
    lw_px = 2.2 * px * line
    # visibility
    facing = np.einsum("ij,ij->i", Nc, Cc) < 0
    two = mesh.two
    vis = facing | two
    flipn = (~facing) & two
    Nv = np.where(flipn[:, None], -Nc, Nc)

    # bounding box (full res) with padding for lines + glow
    glow_r = (0.007 * H, 0.03 * H)
    pad = int(lw_px * 4 + (glow_r[1] * 2.2 if (glow > 0 and style != "solid") else 0) + 4)
    bx0 = int(math.floor(S[:, 0].min())) - pad
    by0 = int(math.floor(S[:, 1].min())) - pad
    bx1 = int(math.ceil(S[:, 0].max())) + pad
    by1 = int(math.ceil(S[:, 1].max())) + pad
    bx0, by0 = max(bx0, -pad), max(by0, -pad)
    bx1, by1 = min(bx1, W + pad), min(by1, H + pad)
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return np.zeros((H, W, 4), np.float32)
    cw, ch = (bx1 - bx0) * ss, (by1 - by0) * ss
    canvas = Image.new("CMYK", (cw, ch), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    S2 = (S - [bx0, by0]) * ss
    Sl = S2.tolist()
    lw = max(1, int(round(lw_px * ss)))

    order = np.nonzero(vis)[0]
    order = order[np.argsort(-Cc[order, 2], kind="stable")]

    # normalised depth 0 (near) .. 1 (far) for fog / fades
    zc = Cc[:, 2]
    zlo, zhi = Vc[:, 2].min(), Vc[:, 2].max()
    dn = np.clip((zc - zlo) / max(zhi - zlo, 1e-6), 0, 1)
    ycen = mesh.C[:, 1]
    hgt = (ycen - mesh.ymin) / max(mesh.ymax - mesh.ymin, 1e-6)

    def polypts(fi):
        ids = mesh.polys[fi]
        out = []
        for i in ids:
            out.extend(Sl[i])
        return out

    ndl = np.clip(Nv @ _L_KEY, 0, 1)
    ndr = np.clip(Nv @ _L_RIM, 0, 1)
    ndf = np.clip(Nv @ _L_FILL, 0, 1)
    vdir = -Cc / np.linalg.norm(Cc, axis=1, keepdims=True)
    ndv = np.clip(np.einsum("ij,ij->i", Nv, vdir), 0, 1)
    hemi = 0.5 + 0.5 * Nv[:, 1]
    hv = _norm(_L_KEY + np.array([0, 0, -1.0]))
    spec = np.clip(Nv @ hv, 0, 1)

    post = None
    glow_amt = 0.0

    if style == "solid":
        concrete = _col(color, (0.70, 0.69, 0.66))
        base = np.zeros((len(mesh.mat), 3))
        base[:] = concrete
        is_star = model == "star"
        red = _col(color, PAL["red"]) if is_star else np.array(PAL["red"])
        base[mesh.mat == M_RED] = red
        base[mesh.mat == M_GOLD] = PAL["gold"]
        base[mesh.mat == M_STEEL] = _col(color, (0.74, 0.77, 0.82))
        base[mesh.mat == M_INNER] = concrete * 0.93
        glass = np.array([0.05, 0.07, 0.12])
        keyc = np.array([1.0, 0.95, 0.86])
        fillc = np.array([0.62, 0.70, 0.85])
        light = (0.16 + 0.24 * hemi + 0.14 * ndf)[:, None] * fillc + (1.0 * ndl)[:, None] * keyc
        ao = 0.55 + 0.45 * np.clip(hgt / 0.45, 0, 1) ** 0.7 if not is_star else np.ones_like(hgt)
        inside = flipn.astype(np.float64)
        ao = ao * (1 - 0.35 * inside)
        c = base * light * ao[:, None]
        c *= (0.95 + 0.1 * mesh.tone)[:, None]
        shiny = np.isin(mesh.mat, (M_RED, M_GOLD, M_STEEL)).astype(np.float64)
        sp = spec ** 24 * (0.25 + 0.75 * shiny)
        c += sp[:, None] * np.array([1.0, 0.97, 0.9]) * 0.55
        rimc = np.array([1.0, 0.86, 0.72]) if not is_star else np.array([1.0, 0.8, 0.5])
        c += (ndr ** 2 * (1 - ndv) ** 0.5)[:, None] * rimc * 0.55
        gm = mesh.mat == M_GLASS
        tint = np.array([[0.10, 0.18, 0.55], [0.55, 0.08, 0.08], [0.60, 0.42, 0.06], [0.08, 0.32, 0.30]])
        c[gm] = (glass + tint[(mesh.tone[gm] * 4).astype(int) % 4] * (0.35 + 0.5 * ndl[gm, None])
                 + spec[gm, None] ** 8 * 0.5 * np.array([0.6, 0.7, 0.9]))
        fogc = np.array([0.10, 0.11, 0.16])
        fg = (dn ** 1.5 * 0.40 * fog)[:, None]
        c = c * (1 - fg) + fogc * fg
        cols = [_u8(list(ci) + [1.0]) for ci in np.clip(c, 0, 1)]
        for fi in order:
            p = polypts(fi)
            draw.polygon(p, fill=cols[fi], outline=cols[fi])
    elif style in ("wire", "neon"):
        ec = _col(edge, PAL["cyan"] if (style == "wire" and color == "cyan") else PAL["red"])
        if style == "neon":
            ec = _col(edge, (1.0, 0.16, 0.14))
        fa = fill_alpha if fill_alpha is not None else (0.0 if style == "wire" else 1.0)
        fc = _col(color if style != "wire" or color not in ("cyan", "red") else None, (0.015, 0.012, 0.02))
        if style == "neon" and color is None:
            fc = np.array([0.03, 0.01, 0.015])
        if style == "neon":
            # pass 1: gold silhouette rim = fat outlines of all front faces
            gold = _u8(list(PAL["gold"]) + [1.0])
            rimw = max(2, int(round(3.2 * px * ss * line)) * 2)
            for fi in order:
                p = polypts(fi)
                draw.polygon(p, fill=gold)
                draw.line(p + p[:2], fill=gold, width=rimw)
        shade = (0.35 + 0.65 * ndl) if style == "neon" else np.ones(len(ndl))
        fade = 1.0 - 0.55 * dn
        for fi in order:
            p = polypts(fi)
            f = fc * shade[fi] * (0.6 + 0.4 * fade[fi])
            draw.polygon(p, fill=_u8(list(f * fa) + [fa]))
            e = ec * fade[fi]
            if style == "neon":
                e = ec * (0.75 + 0.25 * fade[fi])
            draw.line(p + p[:2], fill=_u8(list(e) + [1.0]), width=lw)
        glow_amt = glow * (1.0 if style == "wire" else 1.5)
    elif style == "hologram":
        ec = _col(edge, PAL["cyan"])
        fc = _col(color, PAL["cyan"])
        # back faces first: faint lines
        back = np.nonzero(~vis)[0]
        back = back[np.argsort(-Cc[back, 2])]
        bc = _u8(list(ec * 0.22) + [0.22])
        for fi in back:
            p = polypts(fi)
            draw.line(p + p[:2], fill=bc, width=max(1, lw // 2))
        fres = 1 - ndv
        for fi in order:
            p = polypts(fi)
            a = 0.10 + 0.34 * fres[fi] ** 1.5 + 0.12 * ndl[fi]
            f = fc * (0.55 + 0.45 * ndl[fi])
            draw.polygon(p, fill=_u8(list(f * a) + [a]))
            ea = 0.55 + 0.45 * (1 - dn[fi])
            draw.line(p + p[:2], fill=_u8(list(ec * ea) + [ea]), width=max(1, int(lw * 0.75)))
        glow_amt = glow * 0.9
        # flicker / scanlines / jitter (deterministic in t)
        fr = int(t * 30)
        flick = 0.82 + 0.18 * _hash01(fr, 1.3)
        if _hash01(fr, 7.7) > 0.93:
            flick *= 0.55
        jit_rows = []
        for k in range(3):
            if _hash01(fr, k, 2.1) > 0.55:
                y0 = _hash01(fr, k, 3.3)
                hh = 0.01 + 0.05 * _hash01(fr, k, 4.4)
                dx = (_hash01(fr, k, 5.5) - 0.5) * 0.03 * W
                jit_rows.append((y0, hh, int(dx)))
        sweep = (t * 0.35) % 1.3 - 0.15

        def post(P, A, bx, by):
            h = P.shape[0]
            rows = np.arange(by, by + h)
            sl = np.where((rows // max(1, int(round(3 * px)))) % 2 == 0, 1.0, 0.55).astype(np.float32)
            band = np.exp(-(((rows / H) - sweep) / 0.03) ** 2).astype(np.float32) * 0.6
            m = (sl + band) * flick
            P = P * m[:, None, None]
            A = np.clip(A * m[:, None], 0, 1)
            for y0, hh, dx in jit_rows:
                r0 = int(y0 * H) - by
                r1 = r0 + int(hh * H)
                r0, r1 = max(r0, 0), min(r1, h)
                if r1 > r0:
                    P[r0:r1] = np.roll(P[r0:r1], dx, axis=1)
                    A[r0:r1] = np.roll(A[r0:r1], dx, axis=1)
            return P, A
    elif style == "xray":
        ec = _col(edge, PAL["cyan"] if color is None else color)
        # faint body
        fa = 0.06
        fb = _u8(list(ec * 0.25 * fa) + [fa])
        for fi in order:
            if facing[fi]:
                draw.polygon(polypts(fi), fill=fb)
        E = mesh.edges
        ez = 0.5 * (Vc[E[:, 0], 2] + Vc[E[:, 1], 2])
        eo = np.argsort(-ez)
        edn = np.clip((ez - (_CAM_D - 0.8)) / 1.6, 0, 1)
        for k in eo:
            a = 0.95 - 0.8 * edn[k]
            i, j = E[k]
            draw.line(Sl[i] + Sl[j], fill=_u8(list(ec * a) + [a]), width=max(1, int(lw * 0.8)))
        glow_amt = glow * 0.8
    else:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")

    return _composite(canvas, ss, W, H, bx0, by0, glow_amt, glow_r, post)


# ============================================================ contact sheets ===
def turntable_sheet(models=MODELS, styles=("solid",), n=4, W=960, H=540, t_step=None,
                    spin=0.6, label=True, **kw):
    """Grid (rows = model x style, cols = n yaw samples) of render_model frames
    composited on a dark background. Returns a PIL RGB image."""
    if isinstance(models, str):
        models = (models,)
    if isinstance(styles, str):
        styles = (styles,)
    rows = [(m, s) for m in models for s in styles]
    tw, th = W // 2, H // 2
    sheet = Image.new("RGB", (tw * n, th * len(rows)), (0, 0, 0))
    step = t_step if t_step is not None else 1.15 / spin
    fnt = font("mono", max(10, th // 18))
    for r, (m, s) in enumerate(rows):
        for c in range(n):
            t = c * step
            img = render_model(t, 10.0, W, H, model=m, style=s, spin=spin, **kw)
            yy = np.linspace(0, 1, H)[:, None, None]
            bg = (np.array([0.10, 0.10, 0.13]) * (1 - yy) + np.array([0.03, 0.03, 0.05]) * yy).astype(np.float32)
            bg = np.broadcast_to(bg, (H, W, 3)).copy()
            a = img[..., 3:4]
            comp = bg * (1 - a) + img[..., :3] * a
            tile = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
            if label:
                d = ImageDraw.Draw(tile)
                d.text((6, 4), f"{m} / {s} / t={t:.2f}", font=fnt, fill=(200, 200, 200))
            sheet.paste(tile, (c * tw, r * th))
    return sheet


def _bench(W=1920, H=1080, reps=3):
    res = {}
    for m in MODELS:
        for s in STYLES:
            ts = []
            for k in range(reps):
                t0 = time.perf_counter()
                render_model(0.37 * k + 0.1, 5.0, W, H, model=m, style=s)
                ts.append((time.perf_counter() - t0) * 1e3)
            res[(m, s)] = min(ts)
    return res


if __name__ == "__main__":
    prev = os.path.join(OUT, "preview")
    os.makedirs(prev, exist_ok=True)
    which = sys.argv[1:] or ["sheets", "bench"]
    for m in MODELS:
        print(m, "faces", len(get_mesh(m).faces), "verts", len(get_mesh(m).V))
    if "sheets" in which:
        for m in MODELS:
            sh = turntable_sheet(m, STYLES, n=4)
            p = os.path.join(prev, f"spomenik_{m}.png")
            sh.save(p)
            print("wrote", p)
        sh = turntable_sheet(MODELS, ("solid",), n=4, spin=0.6, pitch=0.12)
        sh.save(os.path.join(prev, "spomenik_all_lowangle.png"))
    if "bench" in which:
        res = _bench()
        for (m, s), ms in res.items():
            print(f"{m:13s} {s:9s} {ms:7.1f} ms @1920x1080")
        print("max %.1f ms" % max(res.values()))
