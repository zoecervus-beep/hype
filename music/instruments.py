"""Drum, bass, cowbell, stab and pad voices. Each returns a numpy array (mono unless noted)."""
import numpy as np
from dsp import (SR, RNG, mtof, saw, pulse, sine, noise, exp_env, fade, lp, hp, bp, drive,
                 hardclip, stereo, reverb, tv_filter)


def _n(d):
    return int(round(d * SR))


# ----------------------------------------------------------------- drums
def kick(vel=1.0, length=0.32, punch=1.0):
    n = _n(length)
    t = np.arange(n) / SR
    f = 46 + 150 * np.exp(-t / 0.028) + 60 * np.exp(-t / 0.004)
    body = sine(f) * np.exp(-t / (0.11 * punch)) * np.clip(1 - (t / length) ** 4, 0, 1)
    click = hp(noise(n) * exp_env(n, 0.0025), 2500, 2) * 0.5
    click += sine(np.full(n, 3200.0)) * exp_env(n, 0.0012) * 0.4
    y = drive(body * 1.5 + click, 2.2)
    return fade(y * vel * 0.95, 0.0002, 0.004)


def clap(vel=1.0, tail=0.16, body_f=205.0, room=True):
    n = _n(0.45)
    t = np.arange(n) / SR
    env = np.zeros(n)
    for i, off in enumerate([0.0, 0.009, 0.019]):
        s = _n(off)
        env[s:] += np.exp(-(t[: n - s]) / 0.004) * (0.8 + 0.1 * i)
    env += np.exp(-np.maximum(t - 0.024, 0) / tail) * (t >= 0.024) * 0.8
    nz = bp(noise(n), 900, 5200, 2) * env
    body = sine(body_f * (1 + 0.5 * np.exp(-t / 0.01))) * exp_env(n, 0.045) * 0.7
    y = drive((nz * 1.3 + body) * 1.2, 1.8)
    y = stereo(y)
    if room:
        y = reverb(y, decay=0.7, mix=0.25, pre=0.004, damp=7000, hp_f=400, tail=False)
    return fade(y * vel * 0.8, 0.0003, 0.02)


def snare(vel=1.0, length=0.18, tone=190.0):
    n = _n(length)
    t = np.arange(n) / SR
    nz = bp(noise(n), 1500, 9000, 2) * exp_env(n, length * 0.3)
    body = sine(tone * (1 + 0.6 * np.exp(-t / 0.008))) * exp_env(n, 0.03)
    return fade(drive(nz + 0.8 * body, 1.6) * vel * 0.8, 0.0002, 0.01)


_HAT_F = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * 1.7


def _metal(n):
    m = sum(pulse(np.full(n, f)) for f in _HAT_F) / 6
    return m


def hat(vel=1.0, length=0.05, open_=False):
    n = _n(0.45 if open_ else length * 3)
    tau = 0.13 if open_ else length * 0.45
    src = 0.55 * _metal(n) + 0.6 * noise(n)
    y = hp(bp(src, 6000, 16000, 2), 7000, 2) * exp_env(n, tau, 0.0003)
    return fade(y * vel * 0.7, 0.0001, 0.01)


def crash(vel=1.0, length=2.2):
    n = _n(length)
    rng = np.random.default_rng(3)
    src = np.stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)], 1) * 0.8
    src += 0.4 * _metal(n)[:, None]
    y = hp(src, 4000, 2) * exp_env(n, length * 0.3, 0.001)[:, None]
    y = lp(y, 14000, 2)
    return fade(y * vel * 0.55, 0.0005, 0.2)


# ----------------------------------------------------------------- cowbell (the hook)
def cowbell(midi, vel=1.0, decay=0.11, bright=1.0, drive_amt=2.6):
    """808 cowbell pitched to `midi`: two pulses at f and 1.48 f, band-passed and driven."""
    f = float(mtof(midi))
    n = _n(decay * 5 + 0.02)
    a = pulse(np.full(n, f), width=0.5)
    b = pulse(np.full(n, f * 1.48), width=0.5)
    x = a * 0.62 + b * 0.5
    x = bp(x, 700, 4000 * bright, 2)
    x = x + 0.25 * bp(x, 1500, 3200, 1)
    t = np.arange(n) / SR
    env = np.exp(-t / (decay * 0.18)) * 0.45 + np.exp(-t / decay) * 0.55
    env[: _n(0.0008)] *= np.linspace(0, 1, _n(0.0008))
    y = drive(x * env * 1.3, drive_amt)
    return fade(y * vel * 0.9, 0.0, 0.01)


def bell(midi, vel=1.0, decay=0.35):
    """Glassy FM bell for the dropB counter-melody."""
    f = float(mtof(midi))
    n = _n(decay * 4)
    t = np.arange(n) / SR
    mod = sine(np.full(n, f * 3.5)) * 2.2 * np.exp(-t / (decay * 0.4))
    y = np.sin(2 * np.pi * f * t + mod) * exp_env(n, decay, 0.001)
    y += 0.3 * sine(np.full(n, f * 2)) * exp_env(n, decay * 0.3)
    return fade(drive(y, 1.5) * vel * 0.6, 0.0, 0.02)


# ----------------------------------------------------------------- 808 bass line
def bass_line(events, n, glide_time=0.07, drive_amt=3.0, clip=0.8, dist_lp=2600):
    """events: list of dicts {t, midi, dur, glide(bool), vel}. Returns (dist, sub) mono arrays of length n."""
    freq = np.full(n, float(mtof(29)))
    amp = np.zeros(n)
    prev, last_lvl = None, 0.8
    evs = sorted(events, key=lambda e: e["t"])
    for i, ev in enumerate(evs):
        s = _n(ev["t"])
        e = min(n, _n(ev["t"] + ev["dur"]))
        if s >= n:
            continue
        L = e - s
        t = np.arange(L) / SR
        f1 = float(mtof(ev["midi"]))
        legato_in = ev.get("glide") and prev is not None
        nxt = evs[i + 1] if i + 1 < len(evs) else None
        legato_out = nxt is not None and nxt.get("glide") and nxt["t"] - (ev["t"] + ev["dur"]) < 0.002
        if legato_in:
            f0 = float(mtof(prev))
            fr = f1 * (f0 / f1) ** np.exp(-t / (glide_time / 3))
            env = last_lvl * np.exp(-t / 3.0)
        else:
            fr = f1 * 2 ** (5 / 12 * np.exp(-t / 0.012))
            env = (0.78 + 0.22 * np.exp(-t / 0.12)) * np.exp(-t / 3.0)
            at = min(L, _n(0.003))
            env[:at] *= np.linspace(0, 1, at)
        last_lvl = float(env[-1])
        if not legato_out:
            rl = min(L, _n(0.025))
            env[L - rl:] *= np.linspace(1, 0, rl)
        freq[s:e] = fr
        amp[s:e] = np.maximum(amp[s:e], env * ev.get("vel", 1.0))
        prev = ev["midi"]
        stop = min(n, _n(nxt["t"])) if nxt is not None else n
        if e < stop:
            freq[e:stop] = fr[-1]
    ph = np.cumsum(freq) / SR
    base = np.sin(2 * np.pi * ph)
    sub = lp(base * amp, 140, 2)
    dist = drive(base * 1.2, drive_amt)
    dist = hardclip(dist * 1.3 + 0.12 * np.sin(4 * np.pi * ph), clip) / clip
    dist = lp(dist, dist_lp, 2) * amp
    return dist, sub


def boom808(midi=29, dur=1.2, vel=1.0):
    d, s = bass_line([{"t": 0.0, "midi": midi, "dur": dur, "vel": vel}], _n(dur + 0.05), drive_amt=4.0)
    return d * 0.55 + s * 0.9


# ----------------------------------------------------------------- tonal
def chord_stab(midis, vel=1.0, dur=0.35, bright=0.5, dist=3.0, seed=0):
    """Distorted detuned saw chord stab (stereo). bright 0..1 opens filter."""
    n = _n(dur + 0.3)
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    rng = np.random.default_rng(seed)
    for m in midis:
        f = float(mtof(m))
        for k, det in enumerate([-0.14, -0.05, 0.06, 0.15]):
            v = saw(np.full(n, f * 2 ** (det / 12)), phase0=rng.uniform())
            (L if k % 2 == 0 else R)[:] += v
            (R if k % 2 == 0 else L)[:] += 0.35 * v
        sq = pulse(np.full(n, f / 2), width=0.4)
        L += 0.5 * sq
        R += 0.5 * sq
    x = np.stack([L, R], 1) / (len(midis) * 2.2)
    cut = 500 + 7000 * bright ** 1.3
    fenv = cut * (0.35 + 1.4 * np.exp(-t / 0.06))
    x = tv_filter(x, fenv, "lowpass", 2, block=128)
    env = np.where(t < dur, np.exp(-t / (dur * 0.9)), np.exp(-dur / (dur * 0.9)) * np.exp(-(t - dur) / 0.06))
    env[: _n(0.002)] *= np.linspace(0, 1, _n(0.002))
    y = drive(x * env[:, None] * 1.4, dist)
    return fade(y * vel * 0.7, 0.0, 0.03)


def pad(midis, dur, cutoff=900.0, vel=1.0, seed=11, attack=1.2, release=1.5):
    """Dark detuned-saw pad, stereo, low-passed."""
    n = _n(dur + release)
    rng = np.random.default_rng(seed)
    L = np.zeros(n)
    R = np.zeros(n)
    t = np.arange(n) / SR
    for m in midis:
        f = float(mtof(m))
        for k in range(5):
            det = (k - 2) * 0.09 + rng.normal(0, 0.02)
            lfo = 1 + 0.0015 * np.sin(2 * np.pi * (0.15 + 0.05 * k) * t + rng.uniform(0, 6))
            v = saw(f * 2 ** (det / 12) * lfo, phase0=rng.uniform())
            pan = (k - 2) / 2.5
            L += v * (1 - pan) * 0.5
            R += v * (1 + pan) * 0.5
    x = np.stack([L, R], 1) / (len(midis) * 3.0)
    x = lp(x, cutoff, 4)
    env = np.clip(t / attack, 0, 1) ** 1.5
    env *= np.where(t > dur, np.exp(-(t - dur) / (release / 4)), 1.0)
    return fade(x * env[:, None] * vel, 0.0, 0.05)


if __name__ == "__main__":
    tests = {
        "kick": kick(), "clap": clap(), "snare": snare(), "hat": hat(), "ohat": hat(open_=True),
        "crash": crash(), "cowbell": cowbell(65), "cowbell_hi": cowbell(77), "bell": bell(84),
        "stab": chord_stab([53, 56, 60], bright=0.8), "pad": pad([41, 48, 56], 4.0),
        "boom808": boom808(),
    }
    d, s = bass_line([{"t": 0, "midi": 29, "dur": 0.5}, {"t": 0.5, "midi": 36, "dur": 0.5, "glide": True}], SR)
    tests["bass_dist"], tests["bass_sub"] = d, s
    for k, v in tests.items():
        print(f"{k:10s} len={len(v)/SR:5.2f}s ch={v.ndim} peak={np.abs(v).max():.3f} "
              f"rms={np.sqrt(np.mean(v**2)):.3f} finite={np.isfinite(v).all()}")
