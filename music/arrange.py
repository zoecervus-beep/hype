"""Arrangement: every note/hit of the track on the 130 BPM grid. All times in beats."""
BPM = 130
BEAT = 60.0 / BPM
BAR = 4 * BEAT
S = 0.25  # one 16th step in beats

SECTIONS = [("intro", 0, 4), ("build", 4, 8), ("dropA", 8, 24), ("break", 24, 28),
            ("dropB", 28, 44), ("outro", 44, 46)]
GAPS = [31, 63, 111, 175]          # beats that are total silence
COUNTDOWN = [16, 18, 20, 22, 24, 26, 28]
CRASHES = [32, 48, 64, 80, 112, 128, 144, 160]

# ---- the hook (F minor). (step, midi) per bar, 16 steps/bar; 3-3-2 syncopation
F4, G4, Ab4, Bb4, C5, Db5, Eb5, E5, F5, E4 = 65, 67, 68, 70, 72, 73, 75, 76, 77, 64
CALL = [(0, F5), (2, F5), (3, C5), (5, Ab4), (7, C5), (8, Db5), (10, C5), (11, Ab4), (13, F4), (15, Ab4)]
RESP = [(0, C5), (2, C5), (3, Ab4), (5, F4), (7, Ab4), (8, Eb5), (10, Db5), (11, C5), (13, Ab4), (15, F4)]
VCALL = [(0, Eb5), (2, Eb5), (3, Bb4), (5, G4), (7, Bb4), (8, Eb5), (10, F5), (11, Eb5), (13, Bb4), (15, G4)]
VRESP = [(0, C5), (2, C5), (3, G4), (5, E4), (7, G4), (8, Bb4), (9, C5), (10, Db5), (11, C5),
         (12, Bb4), (13, G4), (14, Bb4), (15, E5)]
HOOK = [CALL, RESP, VCALL, VRESP]
ACCENT = {0: 1.0, 3: 0.95, 6: 0.9, 8: 0.97, 11: 0.93, 14: 0.9}

# counter-melody (dropB, bell, octave up) - (step, midi, len_steps) per bar
COUNTER = [[(0, 80, 3), (3, 84, 3), (6, 82, 2), (8, 80, 4), (12, 77, 4)],
           [(0, 77, 3), (3, 80, 3), (6, 85, 2), (8, 84, 6), (14, 82, 2)],
           [(0, 82, 3), (3, 79, 3), (6, 87, 2), (8, 85, 4), (12, 84, 4)],
           [(0, 84, 3), (3, 79, 3), (6, 82, 2), (8, 76, 3), (11, 79, 3), (14, 88, 2)]]

ROOTS = [29, 25, 27, 24]      # F1 Db1 Eb1 C1 (one per bar of a 4-bar phrase)
PADS = [[41, 48, 56, 60], [37, 44, 53, 60], [39, 46, 55, 58], [36, 43, 52, 55]]

# 808 patterns: (step, len_steps, octave_offset, glide)
BASS_A = [(0, 3, 0, False), (3, 3, 0, False), (6, 2, 12, True), (8, 3, 0, False), (11, 3, 0, False),
          (14, 2, 12, True)]
BASS_B = [(0, 3, 0, False), (3, 3, 0, False), (6, 4, 12, True), (10, 3, 0, False), (13, 3, "next", True)]
BASS_HALF = [(0, 6, 0, False), (6, 4, 12, True), (10, 6, 0, True)]
KICK_A, KICK_B, KICK_HALF = [0, 3, 8, 11], [0, 3, 10], [0, 10]
KICK_A2, KICK_B2 = [0, 3, 8, 11, 14], [0, 3, 7, 10, 13]

STABS = [([53, 56, 60], 29), ([56, 60, 63], 32), ([58, 61, 65], 34), ([60, 63, 67], 36),
         ([61, 65, 68], 37), ([63, 67, 70], 39), ([64, 67, 72, 76], 40)]


def in_gap(b):
    return any(g <= b < g + 1 for g in GAPS)


def hook_bars(bar0, nbars, ev, variant="main", vel=1.0, transpose=0):
    for i in range(nbars):
        bar = bar0 + i
        for step, m in HOOK[i % 4]:
            v = ACCENT.get(step, 0.72) * vel
            ev["cowbell"].append((bar * 4 + step * S, m + transpose, v, variant))


def drums_bar(bar, ev, kind, hats="16", clap=True):
    b0 = bar * 4
    kicks = {"A": KICK_A, "B": KICK_B, "A2": KICK_A2, "B2": KICK_B2, "H": KICK_HALF}[kind]
    for s in kicks:
        ev["kick"].append((b0 + s * S, 1.0 if s == 0 else 0.92))
    if clap:
        for s in ([8] if kind == "H" else [4, 12]):
            ev["clap"].append((b0 + s * S, 1.0))
    phrase_end = bar % 4 == 3
    n = 16 if hats in ("16", "8") else 32
    step = 4 / n
    for k in range(n):
        b = b0 + k * step
        if hats == "8" and k % 2:
            continue
        if phrase_end and b >= b0 + 3:
            continue
        pos = k * step
        acc = 1.0 if pos % 1 == 0.5 else 0.8 if pos % 0.5 == 0 else 0.6 if pos % 0.25 == 0 else 0.38
        ev["hat"].append((b, acc))
    if phrase_end:  # 32nd roll with crescendo on the last beat of each phrase
        for k in range(8):
            ev["hat"].append((b0 + 3 + k / 8, 0.5 + 0.5 * k / 7))
    elif bar % 2 == 1:
        for k in range(4):
            ev["hat"].append((b0 + 3.5 + k / 8, 0.5 + 0.15 * k))
    for s in ([6, 14] if kind != "H" else [14]):
        ev["open_hat"].append((b0 + s * S, 0.8))


def bass_bar(bar, ev, pattern, vel=1.0):
    root = ROOTS[bar % 4]
    nxt = ROOTS[(bar + 1) % 4]
    for step, ln, off, gl in pattern:
        m = nxt if off == "next" else root + off
        ev["bass"].append((bar * 4 + step * S, m, ln * S, gl, vel))


def counter_bars(bar0, nbars, ev, vel=0.8):
    for i in range(nbars):
        bar = bar0 + i
        for step, m, ln in COUNTER[bar % 4]:
            ev["bell"].append((bar * 4 + step * S, m, ln * S, vel))


def build_events():
    keys = ["kick", "clap", "snare", "hat", "open_hat", "cowbell", "bell", "bass", "stab", "crash",
            "impact", "riser", "downlifter", "revcym", "pad", "stutter", "tape_stop", "boom808"]
    ev = {k: [] for k in keys}
    # ---- intro: pad (low-passed), crackle/static (rendered as beds), hook teaser bars 2-3
    ev["pad"].append((0, 12, PADS[0], 350, 800, 0.55))
    ev["pad"].append((12, 16, PADS[1], 800, 1100, 0.55))
    hook_bars(2, 2, ev, "tele", vel=0.8)
    ev["revcym"].append((12, 16))
    # ---- build: pad opens, snare roll accelerates, riser, 7 countdown stabs
    ev["pad"].append((16, 31, [41, 48, 53, 60], 600, 5000, 0.7))
    for i, (bar, div) in enumerate([(4, 1), (5, 2), (6, 4), (7, 8)]):
        for k in range(4 * div):
            b = bar * 4 + k / div
            if b >= 31:
                break
            ev["snare"].append((b, 0.35 + 0.65 * ((b - 16) / 15) ** 1.2))
    ev["riser"].append((16, 31))
    for i, b in enumerate(COUNTDOWN):
        midis, bm = STABS[i]
        ln = 2.7 if i == 6 else 1.0
        ev["stab"].append((b, midis, i / 6, ln))
        ev["boom808"].append((b, bm, ln, 0.75 + 0.25 * i / 6))
        ev["clap"].append((b, 0.8 + 0.2 * i / 6))
    # ---- dropA
    for bar in range(8, 24):
        drums_bar(bar, ev, "A" if bar % 2 == 0 else "B", hats="16")
        bass_bar(bar, ev, BASS_A if bar % 2 == 0 else BASS_B)
    hook_bars(8, 16, ev, "main")
    for bar in list(range(8, 24)) + list(range(28, 44)):   # quiet pumping pad bed in the drops
        ev["pad"].append((bar * 4, bar * 4 + 4, PADS[bar % 4], 1400, 1400, 0.3 if bar < 24 else 0.36))
    ev["downlifter"].append((94, 96))
    # ---- break
    ev["impact"].append((96, "boom"))
    ev["pad"].append((96, 104, PADS[1], 500, 900, 0.6))
    ev["pad"].append((104, 111, PADS[2], 600, 2500, 0.6))
    hook_bars(24, 4, ev, "verb", vel=0.85)
    ev["riser"].append((104, 111))
    for k in range(16):                         # snare roll in the second half of the riser
        ev["snare"].append((108 + k / 4 if k < 8 else 110 + (k - 8) / 8, 0.3 + 0.5 * k / 15))
    # ---- dropB (harder)
    for bar in range(28, 44):
        if 36 <= bar < 40:                      # switch-up: half-time
            drums_bar(bar, ev, "H", hats="16")
            bass_bar(bar, ev, BASS_HALF, vel=1.0)
        else:
            drums_bar(bar, ev, "A2" if bar % 2 == 0 else "B2", hats="32")
            bass_bar(bar, ev, BASS_A if bar % 2 == 0 else BASS_B)
    hook_bars(28, 16, ev, "hard")
    counter_bars(28, 8, ev)
    counter_bars(40, 4, ev, vel=0.9)
    ev["stutter"] += [(146, 148, 8), (150, 151, 8), (151, 152, 16), (154, 156, 16), (158, 159, 16),
                      (159, 160, 32), (163, 164, 32), (167, 168, 32), (171, 172, 32), (173, 175, 32)]
    # ---- outro
    ev["impact"].append((176, "big"))
    hook_bars(44, 1, ev, "echo", vel=0.9)
    ev["kick"].append((176, 1.0))
    ev["bass"].append((176, 29, 6.0, False, 1.0))
    ev["tape_stop"].append(178)
    for b in CRASHES:
        ev["crash"].append(b)
    # silence the gaps
    for k in keys:
        if k in ("pad", "riser", "downlifter", "revcym", "stutter"):
            continue
        ev[k] = [e for e in ev[k] if not in_gap(e[0] if isinstance(e, tuple) else e)]
    return ev


def section_list():
    return [{"name": n, "start": round(b0 * BAR, 4), "end": round(b1 * BAR, 4), "bar0": b0, "bar1": b1}
            for n, b0, b1 in SECTIONS]


if __name__ == "__main__":
    ev = build_events()
    for k, v in ev.items():
        print(f"{k:10s} {len(v):4d}  first={v[0] if v else None}")
