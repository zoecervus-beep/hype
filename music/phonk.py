"""SFRJ hype edit - drift phonk track. Entry point: python3 music/phonk.py

Writes out/audio/track.wav, out/audio/stems/*.wav and out/audio/beatmap.json.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp import SR  # noqa: E402
from arrange import BPM, BEAT, BAR, GAPS, COUNTDOWN, section_list  # noqa: E402
from stems import render_all, DUR, N, T  # noqa: E402
from master import balance, bus, master, apply_gates  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "audio")
FPS = 30
FADE_OUT = (83.0, 85.8)


def r4(t):
    return round(float(t), 4)


def write_wav(path, x):
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, SR, (x * 32767.0).round().astype(np.int16))


def frame_rms(x, n_frames):
    hop = SR / FPS
    mono = x.mean(axis=1) if x.ndim == 2 else x
    p = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    out = np.zeros(n_frames)
    for i in range(n_frames):
        a, b = int(i * hop), min(len(mono), int((i + 1) * hop))
        out[i] = np.sqrt((p[b] - p[a]) / max(1, b - a)) if b > a else 0.0
    m = out.max()
    return [round(float(v), 4) for v in (out / m if m > 0 else out)]


def beatmap(ev, y, stems):
    snare = sorted([b for b, _ in ev["clap"]] + [b for b, _ in ev["snare"]])
    n_frames = int(np.ceil(DUR * FPS))
    return {
        "bpm": BPM, "sr": SR, "duration": r4(len(y) / SR), "beat": round(BEAT, 6), "bar": round(BAR, 6),
        "sections": section_list(),
        "events": {
            "kick": [r4(T(b)) for b, _ in sorted(ev["kick"])],
            "snare": [r4(T(b)) for b in snare],
            "hat": [r4(T(b)) for b, _ in sorted(ev["hat"])],
            "open_hat": [r4(T(b)) for b, _ in sorted(ev["open_hat"])],
            "cowbell": [{"t": r4(T(b)), "midi": int(m), "vel": round(float(v), 3), "var": var}
                        for b, m, v, var in sorted(ev["cowbell"], key=lambda e: e[0])],
            "bell": [{"t": r4(T(b)), "midi": int(m), "dur": r4(ln * BEAT)} for b, m, ln, _ in ev["bell"]],
            "bass": [{"t": r4(T(b)), "midi": int(m), "dur": r4(ln * BEAT), "glide": bool(g)}
                     for b, m, ln, g, _ in sorted(ev["bass"], key=lambda e: e[0])],
            "countdown": [r4(T(b)) for b in COUNTDOWN],
            "impact": [r4(T(b)) for b, _ in ev["impact"]],
            "gap": [{"start": r4(T(g)), "end": r4(T(g + 1))} for g in GAPS],
            "riser": [{"start": r4(T(a)), "end": r4(T(b))} for a, b in ev["riser"]],
            "downlifter": [{"start": r4(T(a)), "end": r4(T(b))} for a, b in ev["downlifter"]],
            "stutter": [{"start": r4(T(a)), "end": r4(T(b)), "div": d} for a, b, d in ev["stutter"]],
            "tape_stop": [r4(T(b)) for b in ev["tape_stop"]],
            "crash": [r4(T(b)) for b in ev["crash"]],
            "drop": [r4(T(32)), r4(T(112))],
        },
        "energy": {
            "fps": FPS,
            "total": frame_rms(y, n_frames),
            "drums": frame_rms(stems["drums"], n_frames),
            "bass": frame_rms(stems["bass"], n_frames),
            "cowbell": frame_rms(stems["cowbell"], n_frames),
        },
    }


def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "stems"), exist_ok=True)
    print("rendering stems ...")
    stems, ev = render_all()
    drop = (T(32), T(95))
    stems, gains = balance(stems, {"drums": drop, "bass": drop, "cowbell": drop, "fx": (T(16), T(31))})
    print("  stem gains (dB):", {k: round(20 * np.log10(g), 1) for k, g in gains.items()})
    gaps_s = [(T(g), T(g + 1)) for g in GAPS]
    print("mix bus + master ...")
    mix = bus(stems)
    y, info = master(mix, gaps_s, target=-9.15, tp_max=-1.0, fade_out=FADE_OUT)
    y = y[:N]
    print(f"  final: lufs={info['lufs']:.2f} tp={info['tp']:.2f} dBTP gain={info['gain_db']:+.2f} dB")
    write_wav(os.path.join(OUT, "track.wav"), y)
    g = 10 ** (info["gain_db"] / 20)
    for k, x in stems.items():
        write_wav(os.path.join(OUT, "stems", f"{k}.wav"), apply_gates(x * g * 0.5, gaps_s, FADE_OUT))
    bm = beatmap(ev, y, stems)
    with open(os.path.join(OUT, "beatmap.json"), "w") as f:
        json.dump(bm, f, separators=(",", ":"))
    print(f"done in {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
