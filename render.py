#!/usr/bin/env python3
"""Render the edit.

  python3 render.py                       # full 1080p30 render -> out/sfrj_hype.mp4
  python3 render.py --scale 0.5 --out out/preview.mp4
  python3 render.py --start 14 --end 20   # a range
  python3 render.py --stills 15.0,15.2    # PNG stills -> out/stills/
  python3 render.py --sheet 14.7 44.3 48  # contact sheet of N frames in a range
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.core import to_u8, font  # noqa: E402

_TL = None
_SCALE = 1.0


def _init(scale):
    global _TL, _SCALE
    from edit.sfrj import build
    _TL = build()
    _SCALE = scale


def _frame(i):
    img = _TL.render(i, _SCALE)
    return i, to_u8(img).tobytes()


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "sfrj_hype.mp4"))
    ap.add_argument("--audio", default=os.path.join(ROOT, "out", "audio", "track.wav"))
    ap.add_argument("--stills", default=None)
    ap.add_argument("--sheet", nargs=3, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--crf", type=int, default=17)
    args = ap.parse_args()

    from edit.sfrj import build
    tl = build()
    fps = tl.fps
    W, H = int(tl.W * args.scale) // 2 * 2, int(tl.H * args.scale) // 2 * 2

    if args.stills or args.sheet:
        if args.stills:
            times = [float(x) for x in args.stills.split(",")]
        else:
            a, bb, n = float(args.sheet[0]), float(args.sheet[1]), int(args.sheet[2])
            times = [a + (bb - a) * k / max(n - 1, 1) for k in range(n)]
        frames = sorted({int(round(t * fps)) for t in times})
        with mp.Pool(args.workers, _init, (args.scale,)) as pool:
            res = dict(pool.imap_unordered(_frame, frames))
        os.makedirs(os.path.join(ROOT, "out", "stills"), exist_ok=True)
        imgs = []
        for fi in frames:
            im = Image.frombytes("RGB", (W, H), res[fi])
            if args.stills:
                p = os.path.join(ROOT, "out", "stills", f"f{fi:05d}.png")
                im.save(p)
                print(p)
            imgs.append((fi, im))
        if args.sheet:
            cols = 6
            tw = 320
            th = int(tw * H / W)
            rows = math.ceil(len(imgs) / cols)
            sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), (20, 20, 20))
            d = ImageDraw.Draw(sheet)
            f = font("mono", 13)
            for k, (fi, im) in enumerate(imgs):
                x, y = (k % cols) * tw, (k // cols) * (th + 18)
                sheet.paste(im.resize((tw, th), Image.BILINEAR), (x, y))
                d.text((x + 4, y + th + 2), f"{fi / fps:6.2f}s  f{fi}", font=f, fill=(230, 230, 230))
            p = os.path.join(ROOT, "out", "stills", f"sheet_{args.sheet[0]}_{args.sheet[1]}.png")
            sheet.save(p)
            print(p)
        return

    end = args.end if args.end is not None else tl.duration
    f0, f1 = int(round(args.start * fps)), int(round(end * fps))
    frames = list(range(f0, f1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    has_audio = os.path.exists(args.audio)
    if has_audio:
        cmd += ["-ss", f"{args.start:.4f}", "-t", f"{(f1 - f0) / fps:.4f}", "-i", args.audio]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
            "-tune", "grain", "-x264-params", "aq-mode=3",
            "-r", str(fps)]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "320k", "-shortest"]
    cmd += ["-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t_start = time.time()
    with mp.Pool(args.workers, _init, (args.scale,)) as pool:
        for n, (i, buf) in enumerate(pool.imap(_frame, frames, chunksize=2)):
            proc.stdin.write(buf)
            if n % 60 == 0:
                el = time.time() - t_start
                print(f"frame {i} ({n + 1}/{len(frames)})  {el:.0f}s elapsed  "
                      f"{el / (n + 1) * 1000:.0f} ms/frame", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", args.out, f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
