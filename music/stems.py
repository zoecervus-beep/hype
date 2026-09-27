"""Render the arrangement into four stereo stems: drums, bass, cowbell, fx."""
import numpy as np
from dsp import (SR, lp, hp, bp, drive, stereo, widen, delay, reverb, sidechain_curve, place,
                 tv_filter, fade)
import instruments as ins
import fxsounds as fx
from arrange import BEAT, build_events, GAPS

DUR = 86.0
N = int(round(DUR * SR))


def T(b):
    return b * BEAT


def _buf():
    return np.zeros((N, 2))


class Cache(dict):
    def get_or(self, key, fn):
        if key not in self:
            self[key] = fn()
        return self[key]


def render_drums(ev, C):
    d = _buf()
    k = C.get_or("kick", lambda: stereo(ins.kick()))
    for b, v in ev["kick"]:
        place(d, k, T(b), v * 1.0)
    for b, v in ev["clap"]:
        c = C.get_or("clap", lambda: ins.clap())
        place(d, c, T(b), v * 0.85)
    for b, v in ev["snare"]:
        s = C.get_or("snare", lambda: stereo(ins.snare(length=0.14)))
        place(d, s, T(b), v * 0.7)
    rng = np.random.default_rng(130)
    for i, (b, v) in enumerate(ev["hat"]):
        h = C.get_or(("hat", i % 3), lambda: ins.hat(length=0.045 + 0.01 * (i % 3)))
        pan = 0.25 * np.sin(b * np.pi)
        place(d, stereo(h, pan), T(b), v * 0.75 * (0.9 + 0.2 * rng.uniform()))
    for b, v in ev["open_hat"]:
        o = C.get_or("ohat", lambda: stereo(ins.hat(open_=True), -0.2))
        place(d, o, T(b), v * 0.65)
    for b in ev["crash"]:
        c = C.get_or("crash", lambda: ins.crash())
        place(d, c, T(b), 0.8)
    return d


def render_bass(ev):
    """808 line; dropB gets more drive. Mono, returned as (dist, sub) stereo arrays."""
    evA = [{"t": T(b), "midi": m, "dur": ln * BEAT, "glide": g, "vel": v}
           for b, m, ln, g, v in ev["bass"] if b < 112 or b >= 176]
    evB = [{"t": T(b), "midi": m, "dur": ln * BEAT, "glide": g, "vel": v}
           for b, m, ln, g, v in ev["bass"] if 112 <= b < 176]
    dA, sA = ins.bass_line(evA, N, drive_amt=3.0, clip=0.8)
    dB, sB = ins.bass_line(evB, N, drive_amt=5.0, clip=0.6, dist_lp=3400)
    dist = dA + dB * 1.1
    dist = dist + 0.8 * drive(bp(dist, 120, 900, 2) * 2.0, 2.0)   # growl layer for small speakers
    sub = sA + sB
    for b, m, ln, v in ev["boom808"]:
        place(sub, ins.boom808(m, ln * BEAT, 1.0) * 0.7, T(b), v)
    return stereo(dist) * 0.7071, stereo(sub) * 0.7071


def beat_repeat(x, regions):
    """regions: (b0, b1, div). Capture 1/div note at each beat start and repeat it across the beat."""
    y = x.copy()
    for b0, b1, div in regions:
        L = int(round(4 / div * BEAT * SR))
        fl = min(int(0.002 * SR), L // 4)
        for bb in np.arange(b0, b1, 1.0):
            s, e = int(round(T(bb) * SR)), int(round(T(bb + 1) * SR))
            seg = x[s:s + L].copy()
            seg[:fl] *= np.linspace(0, 1, fl)[:, None]
            seg[-fl:] *= np.linspace(1, 0, fl)[:, None]
            reps = int(np.ceil((e - s) / L))
            tile = np.concatenate([seg * (1 - 0.03 * r) for r in range(reps)])[: e - s]
            y[s:e] = tile
        e = int(round(T(b1) * SR))                # fade the dry signal back in after the region
        y[e:e + fl] *= np.linspace(0, 1, fl)[:, None]
    return y


def render_cowbell(ev, C):
    bufs = {k: _buf() for k in ("main", "hard", "tele", "verb", "echo")}
    for b, m, v, var in ev["cowbell"]:
        if var == "hard":
            cb = C.get_or(("cbh", m), lambda: stereo(ins.cowbell(m, decay=0.125, drive_amt=4.2)))
        else:
            cb = C.get_or(("cb", m), lambda: stereo(ins.cowbell(m)))
        place(bufs[var], cb, T(b), v)
    bell = _buf()
    for b, m, ln, v in ev["bell"]:
        bl = C.get_or(("bell", m, ln), lambda: ins.bell(m, decay=0.25 + ln * BEAT * 0.4))
        place(bell, stereo(bl, 0.15), T(b), v)
    stut = [r for r in ev["stutter"]]
    hard = beat_repeat(bufs["hard"], stut)
    bell = beat_repeat(bell, stut)
    out = _buf()
    slap = 0.105
    out += widen(delay(bufs["main"], slap, fb=0.25, mix=0.22, n_taps=2), 11, 0.6)
    out += widen(delay(hard, slap, fb=0.25, mix=0.2, n_taps=2), 11, 0.7) * 1.05
    tele = fx.telephone(bufs["tele"].mean(axis=1), 600, 2200)
    tele = reverb(stereo(tele), decay=1.6, mix=0.35, tail=False)
    out += lp(tele, 3000, 2) * 0.28
    vb = lp(bufs["verb"], 1400, 2)
    out += reverb(vb, decay=4.0, mix=0.6, pre=0.03, tail=False) * 0.4
    echo = delay(bufs["echo"], BEAT * 0.75, fb=0.55, mix=0.5, n_taps=10, lp_f=3500)
    out += reverb(echo, decay=3.0, mix=0.3, tail=False)
    bell = reverb(bell, decay=1.8, mix=0.25, tail=False)
    out += drive(bell * 0.9, 1.4) * 0.55
    return out


def render_fx(ev, C):
    f = _buf()
    for b0, b1, midis, c0, c1, v in ev["pad"]:
        dur = T(b1) - T(b0)
        if c0 == c1:                              # short drop pad: fast attack, fixed filter
            p = ins.pad(midis, dur, cutoff=c0, vel=v, attack=0.03, release=0.25)
            place(f, p, T(b0), 0.8)
            continue
        p = ins.pad(midis, dur, cutoff=8000, vel=v)
        n = len(p)
        cut = np.concatenate([np.geomspace(c0, c1, int(dur * SR)), np.full(n - int(dur * SR), c1)])
        p = tv_filter(p, cut[:n], "lowpass", 2, block=512)
        p = tv_filter(p, cut[:n] * 1.2, "lowpass", 2, block=512)
        place(f, p, T(b0), 0.8)
    f = reverb(f, decay=3.0, mix=0.35, tail=False)
    stab_bus = _buf()
    for i, (b, midis, br, ln) in enumerate(ev["stab"]):
        s = ins.chord_stab(midis, 1.0, ln * BEAT, bright=0.15 + 0.85 * br, dist=2.5 + 2 * br, seed=i)
        place(stab_bus, s, T(b), 0.95 + 0.35 * br)
    f += reverb(stab_bus, decay=1.8, mix=0.3, tail=False)
    for b0, b1 in ev["riser"]:
        place(f, fx.riser(T(b1) - T(b0)), T(b0), 0.9)
    for b0, b1 in ev["downlifter"]:
        place(f, fx.downlifter(T(b1) - T(b0)), T(b0), 0.9)
    for b0, b1 in ev["revcym"]:
        place(f, fx.reverse_cymbal(T(b1) - T(b0)), T(b0), 0.8)
    for b, kind in ev["impact"]:
        imp = fx.sub_boom(3.5, 1.0) if kind == "boom" else fx.big_impact()
        place(f, imp, T(b), 0.7 if kind == "boom" else 0.9)
    # textures: crackle throughout, radio static in the intro
    cr = fx.vinyl_crackle(DUR)
    env = np.ones(N)
    env[: int(T(16) * SR)] = 1.6
    f += cr[:N] * env[:, None] * 0.9
    st = fx.radio_static(T(16))
    ramp = np.clip(np.linspace(0, 16, len(st)) / 2, 0, 1) * np.clip((16 - np.linspace(0, 16, len(st))) / 4, 0, 1)
    place(f, st * ramp[:, None], 0.0, 0.35)
    return f


def tape_stop(x, t0, length=1.6):
    s = int(t0 * SR)
    n = int(length * SR)
    y = x.copy()
    u = np.arange(n) / n
    speed = (1 - u) ** 1.7
    pos = s + np.cumsum(speed)
    for c in range(x.shape[1]):
        y[s:s + n, c] = np.interp(pos, np.arange(len(x)), x[:, c]) * (1 - u ** 3)
    y[s + n:] = 0
    return y


def render_all():
    ev = build_events()
    C = Cache()
    drums = render_drums(ev, C)
    dist, sub = render_bass(ev)
    cow = render_cowbell(ev, C)
    fxb = render_fx(ev, C)
    # finale 1/32 stutters also chop the drums
    drums = beat_repeat(drums, [r for r in ev["stutter"] if r[2] == 32])
    kicks = [T(b) for b, _ in ev["kick"]]
    dist *= sidechain_curve(N, kicks, depth=0.55, release=0.06)[:, None]
    sub *= sidechain_curve(N, kicks, depth=0.75, release=0.05)[:, None]
    cow *= sidechain_curve(N, kicks, depth=0.35, release=0.12)[:, None]
    fxb *= sidechain_curve(N, kicks, depth=0.65, release=0.2)[:, None]
    bass = dist + sub
    ts = T(ev["tape_stop"][0])
    bass, cow, drums = tape_stop(bass, ts), tape_stop(cow, ts), tape_stop(drums, ts)
    return {"drums": drums, "bass": bass, "cowbell": cow, "fx": fxb}, ev


if __name__ == "__main__":
    import time
    t0 = time.time()
    stems, ev = render_all()
    for k, v in stems.items():
        print(k, v.shape, np.isfinite(v).all(), round(float(np.abs(v).max()), 3),
              round(float(np.sqrt(np.mean(v ** 2))), 4))
    print("time", round(time.time() - t0, 1))
