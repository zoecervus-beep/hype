"""Mix bus + mastering: balance, saturation, glue comp, clipper, limiter, loudness targeting."""
import numpy as np
from scipy import signal
from dsp import SR, drive, compress, limiter, hp, lp, peak_eq

# stem balance targets (RMS dBFS measured over the region given, before the bus)
STEM_TARGET = {"drums": -15.0, "bass": -14.0, "cowbell": -15.5, "fx": -17.0}


def k_weight(x):
    fc, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    A = 10 ** (G / 40)
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / (2 * Q)
    cs = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cs + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * cs),
         A * ((A + 1) + (A - 1) * cs - 2 * np.sqrt(A) * al)]
    a = [(A + 1) - (A - 1) * cs + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * cs),
         (A + 1) - (A - 1) * cs - 2 * np.sqrt(A) * al]
    y = signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)
    fc, Q = 38.13547087602444, 0.5003270373238773
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / (2 * Q)
    a = [1 + al, -2 * np.cos(w0), 1 - al]
    return signal.lfilter(np.array([1.0, -2.0, 1.0]) / a[0], np.array(a) / a[0], y, axis=0)


def lufs(x):
    y = k_weight(x)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    p = np.cumsum(np.concatenate([np.zeros((1, 2)), y ** 2]), axis=0)
    starts = np.arange(0, len(y) - blk, hop)
    z = (p[starts + blk] - p[starts]) / blk
    l = -0.691 + 10 * np.log10(z.sum(axis=1) + 1e-12)
    g = z[l > -70]
    rel = -0.691 + 10 * np.log10(g.mean(axis=0).sum()) - 10
    g2 = z[(l > -70) & (l > rel)]
    return -0.691 + 10 * np.log10(g2.mean(axis=0).sum())


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


def softclip(x, c=1.0, knee=0.8):
    k = c * knee
    a = np.abs(x)
    over = a > k
    y = x.copy()
    y[over] = np.sign(x[over]) * (k + (c - k) * np.tanh((a[over] - k) / (c - k)))
    return y


def balance(stems, regions):
    """Scale each stem so its RMS over its reference region hits STEM_TARGET."""
    out, gains = {}, {}
    for k, x in stems.items():
        s, e = regions[k]
        seg = x[int(s * SR):int(e * SR)]
        r = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        g = 10 ** ((STEM_TARGET[k] - r) / 20)
        out[k], gains[k] = x * g, g
    return out, gains


def bus(stems):
    mix = sum(stems.values())
    mix = hp(mix, 25, 2)
    # keep everything below 120 Hz mono
    low = lp(mix, 120, 2)
    mono_low = low.mean(axis=1, keepdims=True)
    mix = mix - low + mono_low
    mix = peak_eq(mix, 3200, 1.5, 0.8)       # a bit of bite for the cowbell
    mix = drive(mix * 1.1, 1.3) / 1.1          # bus saturation
    mix = compress(mix, thresh_db=-14, ratio=2.2, attack=0.012, release=0.15)
    return mix


def master(mix, gaps_s, target=-9.0, tp_max=-1.0, fade_out=(83.0, 85.8)):
    """Find pre-gain so the clipped+limited result hits `target` LUFS with true peak <= tp_max."""
    def chain(g, ceil_db):
        y = softclip(mix * g, 10 ** ((ceil_db + 1.2) / 20), 0.75)
        y = limiter(y, ceil_db, lookahead=0.0025, release=0.05)
        y = apply_gates(y, gaps_s, fade_out)
        return y

    ceil_db = -1.2
    g = 10 ** ((target - lufs(mix)) / 20)
    for it in range(8):
        y = chain(g, ceil_db)
        L = lufs(y)
        g *= 10 ** ((target - L) / 20 * 1.15)
        tp = true_peak_db(y)
        print(f"  master iter {it}: gain={20*np.log10(g):+.2f} dB lufs={L:.2f} tp={tp:.2f} ceil={ceil_db:.2f}")
        if tp > tp_max - 0.05:
            ceil_db -= (tp - tp_max) + 0.1
        if abs(L - target) < 0.1 and tp <= tp_max - 0.05:
            break
    y = chain(g, ceil_db)
    return y, {"gain_db": 20 * np.log10(g), "lufs": lufs(y), "tp": true_peak_db(y), "ceil": ceil_db}


def apply_gates(y, gaps_s, fade_out):
    y = y.copy()
    n = len(y)
    f_out, f_in = int(0.006 * SR), int(0.0008 * SR)
    for s, e in gaps_s:
        a, b = int(round(s * SR)), int(round(e * SR))
        y[a - f_out:a] *= np.linspace(1, 0, f_out)[:, None]
        y[a:b] = 0.0
        y[b:b + f_in] *= np.linspace(0, 1, f_in)[:, None]
    a, b = int(fade_out[0] * SR), int(fade_out[1] * SR)
    w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b - a))
    y[a:b] *= w[:, None] ** 1.5
    y[b:] = 0.0
    return y
