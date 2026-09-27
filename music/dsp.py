"""Small DSP toolkit (numpy/scipy) for the phonk track."""
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 44100
RNG = np.random.default_rng(1945)


def secs(n):
    return np.arange(int(round(n * SR))) / SR


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


# ----------------------------------------------------------------- oscillators
def phase_of(freq, n=None, phase0=0.0):
    f = np.broadcast_to(np.asarray(freq, float), (n,)) if n else np.asarray(freq, float)
    return (phase0 + np.cumsum(f) / SR) % 1.0, f / SR


def _polyblep(t, dt):
    out = np.zeros_like(t)
    dt = np.maximum(np.broadcast_to(dt, t.shape), 1e-9)
    a = t < dt
    x = t[a] / dt[a]
    out[a] = x + x - x * x - 1.0
    b = t > 1.0 - dt
    x = (t[b] - 1.0) / dt[b]
    out[b] = x * x + x + x + 1.0
    return out


def saw(freq, n=None, phase0=0.0):
    ph, dt = phase_of(freq, n, phase0)
    return 2.0 * ph - 1.0 - _polyblep(ph, dt)


def pulse(freq, n=None, width=0.5, phase0=0.0):
    ph, dt = phase_of(freq, n, phase0)
    ph2 = (ph + (1.0 - width)) % 1.0
    sq = np.where(ph < width, 1.0, -1.0)
    return sq + _polyblep(ph, dt) - _polyblep(ph2, dt)


def sine(freq, n=None, phase0=0.0):
    ph, _ = phase_of(freq, n, phase0)
    return np.sin(2 * np.pi * ph)


def noise(n, rng=RNG):
    return rng.uniform(-1, 1, n)


# ----------------------------------------------------------------- envelopes
def exp_env(n, tau, attack=0.0005):
    t = np.arange(n) / SR
    e = np.exp(-t / max(tau, 1e-5))
    na = int(attack * SR)
    if na > 1:
        e[:na] *= np.linspace(0, 1, na)
    return e


def adsr(n, a=0.005, d=0.1, s=0.7, r=0.1, gate=None):
    gate = n / SR - r if gate is None else gate
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = t > gate
    if rel.any():
        g0 = e[np.argmax(rel) - 1] if np.argmax(rel) > 0 else s
        e[rel] = g0 * np.exp(-(t[rel] - gate) / max(r / 4, 1e-6))
    return e


def fade(x, fin=0.002, fout=0.005):
    x = x.copy()
    a, b = int(fin * SR), int(fout * SR)
    if a > 1:
        x[:a] *= np.linspace(0, 1, a)[:, None] if x.ndim == 2 else np.linspace(0, 1, a)
    if b > 1:
        x[-b:] *= np.linspace(1, 0, b)[:, None] if x.ndim == 2 else np.linspace(1, 0, b)
    return x


# ----------------------------------------------------------------- filters
from functools import lru_cache


def _sos(kind, f, order):
    if kind != "bandpass":
        f = float(np.round(np.log2(max(f, 10.0)) * 96) / 96)
        return _sos_c(kind, 2 ** f, order)
    return _sos_c(kind, tuple(float(v) for v in f), order)


@lru_cache(maxsize=4096)
def _sos_c(kind, f, order):
    nyq = SR / 2
    if kind == "bandpass":
        wn = [max(f[0], 10) / nyq, min(f[1], nyq * 0.95) / nyq]
    else:
        wn = min(max(f, 10), nyq * 0.95) / nyq
    return signal.butter(order, wn, btype=kind, output="sos")


def filt(x, kind, f, order=2):
    return signal.sosfilt(_sos(kind, f, order), x, axis=0)


def lp(x, f, order=2):
    return filt(x, "lowpass", f, order)


def hp(x, f, order=2):
    return filt(x, "highpass", f, order)


def bp(x, lo, hi, order=2):
    return filt(x, "bandpass", (lo, hi), order)


def tv_filter(x, cutoff, kind="lowpass", order=2, block=256):
    """Time-varying Butterworth filter; cutoff is an array (per-sample) or callable."""
    n = len(x)
    cut = np.broadcast_to(np.asarray(cutoff, float), (n,))
    y = np.zeros_like(x)
    zi = None
    for s in range(0, n, block):
        e = min(n, s + block)
        sos = _sos(kind, float(cut[(s + e) // 2]), order)
        if zi is None:
            shp = (sos.shape[0], 2) + x.shape[1:]
            zi = np.zeros(shp)
        y[s:e], zi = signal.sosfilt(sos, x[s:e], axis=0, zi=zi)
    return y


def onepole_lp(x, f):
    a = np.exp(-2 * np.pi * f / SR)
    return signal.lfilter([1 - a], [1, -a], x, axis=0)


def peak_eq(x, f, gain_db, q=1.0):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f / SR
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


# ----------------------------------------------------------------- saturation
def drive(x, amount=2.0):
    return np.tanh(amount * x) / np.tanh(amount)


def hardclip(x, c=1.0):
    return np.clip(x, -c, c)


def fold(x, amount=1.5):
    return np.sin(np.pi / 2 * amount * x)


def bitcrush(x, bits=8, down=1):
    q = 2 ** (bits - 1)
    y = np.round(x * q) / q
    if down > 1:
        y = np.repeat(y[::down], down, axis=0)[: len(x)]
    return y


# ----------------------------------------------------------------- space
def stereo(x, pan=0.0):
    if x.ndim == 2:
        return x
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    return np.stack([x * l * 1.4142, x * r * 1.4142], axis=1)


def widen(x, ms_delay=12.0, amount=0.5):
    """Haas-style widening of a mono/stereo signal (keeps lows mono)."""
    x = stereo(x)
    d = int(ms_delay * SR / 1000)
    side_src = hp(x.mean(axis=1), 300)
    side = np.concatenate([np.zeros(d), side_src[:-d]]) - side_src
    out = x.copy()
    out[:, 0] += amount * 0.5 * side
    out[:, 1] -= amount * 0.5 * side
    return out


def delay(x, time, fb=0.35, mix=0.3, n_taps=6, pingpong=True, lp_f=5000):
    """Feedback delay built from shifted copies. x stereo or mono."""
    x = stereo(x)
    d = int(time * SR)
    out = x.copy()
    tap = x
    for i in range(1, n_taps + 1):
        tap = lp(tap, lp_f, 1) * fb
        if pingpong:
            tap = tap[:, ::-1]
        sh = np.zeros_like(x)
        if i * d < len(x):
            sh[i * d:] = tap[: len(x) - i * d]
        out += sh * (mix / fb)
    return out


_IR_CACHE = {}


def reverb_ir(decay=2.0, pre=0.01, damp=6000, seed=7):
    key = (decay, pre, damp, seed)
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    rng = np.random.default_rng(seed)
    n = int((decay * 1.2 + pre) * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2))
    env = np.exp(-6.9 * t / decay)[:, None]
    bright = ir * env
    dark = lp(ir, damp * 0.25, 1) * np.exp(-6.9 * t / (decay * 1.4))[:, None]
    mixw = np.clip(t / decay, 0, 1)[:, None]
    ir = bright * (1 - mixw) * 0.6 + dark * (0.4 + mixw)
    ir = lp(ir, damp, 2)
    ir[: int(pre * SR)] = 0
    ir[: int(0.004 * SR) + int(pre * SR)] *= np.linspace(0, 1, int(0.004 * SR) + int(pre * SR))[:, None]
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True)) + 1e-9
    _IR_CACHE[key] = ir
    return ir


def _extent(x):
    nz = np.flatnonzero(np.abs(x).max(axis=1) > 1e-7)
    return (0, 0) if len(nz) == 0 else (nz[0], nz[-1] + 1)


def reverb(x, decay=2.0, mix=0.3, pre=0.01, damp=6000, hp_f=200, tail=True):
    x = stereo(x)
    ir = reverb_ir(decay, pre, damp)
    s, e = _extent(x)
    n_out = len(x) + (len(ir) - 1 if tail else 0)
    out = np.zeros((n_out, 2))
    out[: len(x)] = x * (1 - mix * 0.5)
    if e <= s:
        return out
    src = hp(x[s:e], hp_f)
    wet = signal.fftconvolve(src, ir, axes=0)
    e2 = min(n_out, s + len(wet))
    out[s:e2] += wet[: e2 - s] * mix
    return out


# ----------------------------------------------------------------- dynamics
def sidechain_curve(n, times, depth=0.7, attack=0.004, release=0.18, shape=2.0):
    """Gain curve ducked at each trigger time (s)."""
    g = np.ones(n)
    ln = int((attack + release * 2.5) * SR)
    t = np.arange(ln) / SR
    duck = np.where(t < attack, t / attack, 1.0) * np.exp(-((t / release) ** shape) * 2.2)
    for tt in times:
        s = int(round(tt * SR))
        if s >= n:
            continue
        e = min(n, s + ln)
        g[s:e] = np.minimum(g[s:e], 1 - depth * duck[: e - s])
    return g


def compress(x, thresh_db=-12, ratio=4, attack=0.005, release=0.1, makeup_db=0):
    mono = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    env = onepole_lp(mono, 1 / (2 * np.pi * release))
    env = np.maximum(env, onepole_lp(mono, 1 / (2 * np.pi * attack)))
    lvl = 20 * np.log10(env + 1e-9)
    over = np.maximum(lvl - thresh_db, 0)
    gr = -over * (1 - 1 / ratio) + makeup_db
    g = 10 ** (gr / 20)
    return x * (g[:, None] if x.ndim == 2 else g)


def limiter(x, ceiling_db=-1.0, lookahead=0.003, release=0.06, oversample=4):
    """Lookahead brickwall; with oversample>1 the detector sees inter-sample (true) peaks."""
    c = 10 ** (ceiling_db / 20)
    if oversample > 1:
        up = np.abs(signal.resample_poly(x, oversample, 1, axis=0))
        up = up.max(axis=1) if up.ndim == 2 else up
        pk = up[: len(x) * oversample].reshape(len(x), oversample).max(axis=1)
    else:
        pk = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    graw = np.minimum(1.0, c / (pk + 1e-12))
    L = max(3, int(lookahead * SR))
    g1 = minimum_filter1d(graw, size=2 * L + 1, mode="nearest")
    g2 = uniform_filter1d(g1, size=L, mode="nearest")
    # slow release: blend with a longer-window min (smoothed) to avoid pumping distortion
    R = int(release * SR)
    g3 = uniform_filter1d(minimum_filter1d(graw, size=R, mode="nearest"), size=R, mode="nearest")
    g = np.minimum(g2, g3)
    y = x * (g[:, None] if x.ndim == 2 else g)
    return np.clip(y, -c, c)


def place(buf, clip, t, gain=1.0):
    """Mix `clip` into `buf` starting at time t (s). Shapes must be compatible."""
    s = int(round(t * SR))
    if s >= len(buf) or s + len(clip) <= 0:
        return buf
    a = max(0, -s)
    e = min(len(buf), s + len(clip))
    buf[s + a:e] += clip[a:e - s] * gain
    return buf


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


if __name__ == "__main__":
    n = SR
    for name, y in [("saw", saw(110.0, n)), ("pulse", pulse(mtof(65), n)), ("sine", sine(55.0, n))]:
        print(name, y.shape, np.isfinite(y).all(), round(float(np.abs(y).max()), 3))
    y = reverb(stereo(saw(220.0, n) * exp_env(n, 0.1)), decay=1.5)
    print("reverb", y.shape, np.isfinite(y).all(), round(float(np.abs(y).max()), 3))
    y = tv_filter(noise(n), np.linspace(200, 8000, n))
    print("tvf", np.isfinite(y).all(), round(rms_db(y), 1))
    y = limiter(stereo(noise(n) * 3), -1.0)
    print("lim", round(float(np.abs(y).max()), 4), 10 ** (-1 / 20))
    d = delay(pulse(880.0, n) * exp_env(n, 0.05), 0.1)
    print("delay", d.shape, np.isfinite(d).all())
