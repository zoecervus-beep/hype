"""Numerical checks + spectrogram for out/audio/track.wav. Run: python3 music/verify.py"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import wavfile
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV = os.path.join(ROOT, "out", "audio", "track.wav")
BM = os.path.join(ROOT, "out", "audio", "beatmap.json")
PNG = os.path.join(ROOT, "out", "preview", "track_spectrogram.png")
BEAT = 60 / 130


def ffmpeg_loudness():
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ff, "-hide_banner", "-nostats", "-i", WAV, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr[r.stderr.rfind("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", txt).group(1))
    lra = float(re.search(r"LRA:\s+(-?[\d.]+) LU", txt).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", txt).group(1))
    return I, lra, tp


def spectrogram(x, sr, bm):
    mono = x.mean(axis=1)
    f, t, S = signal.stft(mono, sr, nperseg=2048, noverlap=2048 - 512)
    S = 20 * np.log10(np.abs(S) + 1e-9)
    W, H = 1720, 600
    # log-frequency rows 30 Hz .. 16 kHz
    fr = np.geomspace(30, 16000, H)
    rows = np.searchsorted(f, fr)
    img = S[rows][::-1]
    cols = np.linspace(0, img.shape[1] - 1, W).astype(int)
    img = img[:, cols]
    img = np.clip((img - img.max() + 85) / 85, 0, 1)
    # inferno-ish ramp: black -> purple -> red -> orange -> yellow
    stops = np.array([[0, 0, 4], [60, 15, 110], [180, 40, 70], [240, 120, 20], [252, 250, 160]]) / 255
    pos = img * (len(stops) - 1)
    i0 = np.clip(pos.astype(int), 0, len(stops) - 2)
    fr_ = (pos - i0)[..., None]
    rgb = stops[i0] * (1 - fr_) + stops[i0 + 1] * fr_
    im = Image.fromarray((rgb * 255).astype(np.uint8)).convert("RGB")
    canvas = Image.new("RGB", (W, H + 120), (10, 10, 12))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    dur = len(mono) / sr
    for s in bm["sections"]:
        xs = int(s["start"] / dur * W)
        d.line([(xs, 0), (xs, H + 20)], fill=(255, 255, 255), width=1)
        d.text((xs + 3, H + 4), s["name"], fill=(255, 255, 255))
    for g in bm["events"]["gap"]:
        xs = int(g["start"] / dur * W)
        d.text((xs - 3, H + 22), "gap", fill=(255, 80, 80))
    # RMS lane
    e = np.array(bm["energy"]["total"])
    pts = [(int(i / len(e) * W), H + 118 - int(v * 70)) for i, v in enumerate(e)]
    d.line(pts, fill=(230, 50, 50), width=1)
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    canvas.save(PNG)


def main():
    sr, x = wavfile.read(WAV)
    x = x.astype(np.float64) / 32768.0
    bm = json.load(open(BM))
    ok = True
    print(f"sr={sr} shape={x.shape} dur={len(x)/sr:.3f}s dtype=int16 finite={np.isfinite(x).all()}")
    print(f"sample peak={20*np.log10(np.abs(x).max()):.2f} dBFS")
    I, lra, tp = ffmpeg_loudness()
    print(f"ffmpeg ebur128: I={I} LUFS  LRA={lra} LU  true peak={tp} dBFS")
    ok &= abs(I + 9) < 0.5 and tp <= -1.0
    for g in bm["events"]["gap"]:
        a, b = int(round(g["start"] * sr)) + 1, int(round(g["end"] * sr)) - 1
        r = np.sqrt(np.mean(x[a:b] ** 2))
        print(f"gap {g['start']:.3f}-{g['end']:.3f}: rms={20*np.log10(r+1e-12):.1f} dBFS max={np.abs(x[a:b]).max():.6f}")
        ok &= r < 1e-4
    print("tail (last 0.2 s) max:", np.abs(x[-int(0.2 * sr):]).max())
    # section loudness (RMS dB)
    for s in bm["sections"]:
        seg = x[int(s["start"] * sr):int(s["end"] * sr)]
        print(f"  {s['name']:6s} {s['start']:7.3f}-{s['end']:7.3f} rms={20*np.log10(np.sqrt(np.mean(seg**2))+1e-12):6.1f} dB")
    # grid check
    bad = 0
    for k, v in bm["events"].items():
        items = v if isinstance(v, list) else []
        for it in items:
            ts = [it] if isinstance(it, (int, float)) else [it.get("t", it.get("start")), it.get("end", 0)]
            for t in ts:
                if t is None:
                    continue
                q = t / (BEAT / 8)
                if abs(q - round(q)) * (BEAT / 8) > 0.00051:
                    bad += 1
    print("off-grid event times (1/32 grid):", bad)
    ok &= bad == 0
    n_fr = int(np.ceil(bm["duration"] * 30))
    for k in ("total", "drums", "bass", "cowbell"):
        e = bm["energy"][k]
        print(f"energy {k}: len={len(e)} (expect {n_fr}) max={max(e)} min={min(e)}")
        ok &= len(e) == n_fr
    print("countdown:", bm["events"]["countdown"])
    spectrogram(x, sr, bm)
    print("spectrogram ->", PNG)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
