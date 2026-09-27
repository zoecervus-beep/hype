"""Transition FX + texture: riser, downlifter, impacts, reverse cymbal, crackle, static."""
import numpy as np
from dsp import (SR, mtof, saw, sine, noise, exp_env, fade, lp, hp, bp, drive, stereo,
                 reverb, tv_filter)
from instruments import crash, bass_line


def _n(d):
    return int(round(d * SR))


def riser(dur, f0=300.0, f1=11000.0, vel=1.0, pitched=True, seed=5):
    """White-noise riser with opening band + optional rising saw. Stereo."""
    n = _n(dur)
    t = np.arange(n) / SR
    p = t / dur
    rng = np.random.default_rng(seed)
    x = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)], 1)
    cut = f0 * (f1 / f0) ** (p ** 1.6)
    y = tv_filter(x, cut, "lowpass", 2, block=256)
    y = hp(y, 150, 2)
    amp = p ** 2.2
    if pitched:
        f = 110 * 2 ** (3 * p ** 1.5)
        sw = saw(f) + saw(f * 1.01)
        sw = lp(sw, 3000, 2)
        y = y + 0.18 * stereo(sw)
    # tremolo accelerating toward the end
    trem = 1 - 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(2 + 14 * p ** 2) / SR))
    y = y * (amp * trem)[:, None]
    return fade(y * vel * 0.6, 0.01, 0.004)


def downlifter(dur, vel=1.0):
    n = _n(dur)
    t = np.arange(n) / SR
    p = t / dur
    x = stereo(noise(n))
    cut = 9000 * (200 / 9000) ** (p ** 0.7)
    y = tv_filter(x, cut, "lowpass", 2, block=256)
    f = 900 * 2 ** (-4 * p)
    y = y * 0.7 + 0.35 * stereo(sine(f))
    env = (1 - p) ** 1.5 * np.clip(t / 0.01, 0, 1)
    return fade(y * env[:, None] * vel * 0.6, 0.002, 0.02)


def sub_boom(dur=3.0, vel=1.0, midi=29, verb=2.5):
    """Sub-drop impact: pitch-dropping sine + distorted thump + noise burst + long reverb."""
    n = _n(dur)
    t = np.arange(n) / SR
    f = mtof(midi) * 2 ** (-1.2 * t / dur) + 90 * np.exp(-t / 0.03)
    body = sine(f) * np.exp(-t / (dur * 0.35))
    thump = drive(body * 2.0, 3.0) * exp_env(n, 0.25)
    nb = lp(noise(n), 1800, 2) * exp_env(n, 0.12, 0.0005)
    mono = body * 0.8 + 0.35 * lp(thump, 600, 2) + 0.35 * nb
    y = stereo(mono)
    wet = reverb(stereo(lp(thump * 0.5 + nb, 3000, 2)), decay=verb, mix=0.7, hp_f=120)
    out = np.zeros((max(len(wet), n), 2))
    out[:n] += y
    out[:len(wet)] += wet * 0.5
    return fade(out * vel, 0.0005, 0.2)


def big_impact(vel=1.0):
    """Outro impact: boom + crash + chord + very long reverb."""
    b = sub_boom(3.5, vel=1.0, midi=29, verb=4.0)
    c = crash(1.0, 3.0)
    out = np.zeros((len(b), 2))
    out += b
    out[:len(c)] += c * 0.9
    out = reverb(out * 0.8, decay=4.5, mix=0.35, hp_f=250, tail=False)
    return out * vel


def reverse_cymbal(dur):
    c = crash(1.0, dur + 0.3)[: _n(dur)][::-1]
    t = np.arange(len(c)) / SR
    env = (t / dur) ** 1.5
    return fade(c * env[:, None] * 1.6, 0.01, 0.003)


def vinyl_crackle(dur, density=18.0, vel=1.0, seed=42):
    n = _n(dur)
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2))
    k = int(density * dur)
    pos = rng.integers(0, n - 50, k)
    amp = rng.exponential(0.3, k) * rng.choice([-1, 1], k)
    ch = rng.integers(0, 2, k)
    x[pos, ch] += amp
    big = rng.integers(0, n - 50, int(dur * 2))
    x[big, :] += rng.uniform(0.6, 1.0, (len(big), 1))
    x = bp(x, 900, 7000, 1)
    hiss = lp(hp(np.stack([noise(n), noise(n)], 1), 3000, 1), 9000, 1) * 0.02
    rumble = lp(stereo(noise(n)), 60, 2) * 0.08
    return (x * 0.6 + hiss + rumble) * vel


def radio_static(dur, vel=1.0, seed=9):
    n = _n(dur)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    x = bp(rng.uniform(-1, 1, n), 1200, 3800, 2)
    mod = 0.5 + 0.5 * lp(rng.uniform(-1, 1, n), 6, 2) * 12
    mod = np.clip(mod, 0, 1)
    tone = 0.15 * sine(np.full(n, 1800.0)) * (np.sin(2 * np.pi * 0.3 * t) > 0.6)
    y = (x * mod + tone) * vel
    return stereo(y, 0.3) * 0.5


def telephone(x, lo=500, hi=2500):
    """Band-limited 'through the radio' filter."""
    return drive(bp(x, lo, hi, 2) * 1.5, 1.5)


if __name__ == "__main__":
    tests = {
        "riser": riser(3.0), "downlifter": downlifter(0.92), "sub_boom": sub_boom(),
        "big_impact": big_impact(), "rev_cym": reverse_cymbal(2.0), "crackle": vinyl_crackle(4.0),
        "static": radio_static(4.0),
    }
    for k, v in tests.items():
        print(f"{k:10s} len={len(v)/SR:5.2f}s ch={v.ndim} peak={np.abs(v).max():.3f} "
              f"rms={np.sqrt(np.mean(v**2)):.3f} finite={np.isfinite(v).all()}")
