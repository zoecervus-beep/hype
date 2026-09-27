"""SFRJ — the edit. Every cut is placed on the 130 BPM beat grid (see SPEC.md)."""
from __future__ import annotations

import math

import numpy as np

from engine import fx, text as tx
from engine.core import PAL, blank, over, ease_out_expo, ease_in_cubic, ease_out_cubic, ease_in_out_cubic, star_points
from engine.timeline import Timeline, b, BEAT, BAR
from edit.kit import *  # noqa: F401,F403
from edit import elements as el
from edit import gens as g
from edit.photos import PH

W_, H_ = 1920, 1080
BLACK, WHITE, RED, BLUE, GOLD, NAVY, DRED = (PAL[k] for k in ("black", "white", "red", "blue", "gold", "navy", "deep_red"))


def cut(tl, b0, b1, shot, name=""):
    tl.shot(b(b0), b(b1), shot, name)


def cap(tl, b0, b1, text, cy=0.9, **kw):
    tl.op(b(b0), b(b1), T_cap(text, cy=cy, **kw), z=60, name="cap")


def fl(tl, bt, rgb=WHITE, decay=0.08):
    X_flash(tl, b(bt), rgb, decay)


# ======================================================================= INTRO
def intro(tl):
    flicker = lambda c, img: img * (0.86 + 0.14 * np.random.default_rng(c.frame).random())
    cut(tl, 0, 4, S_layers(S_solid(BLACK),
                           T_type("6. APRIL 1941.", "mono", 0.06, WHITE, 0.5, 0.47, cps=11, at=0.35)))
    cap(tl, 2, 4, "THE AXIS INVADES YUGOSLAVIA", cy=0.6)
    cut(tl, 4, 8, S_layers(S_photo(PH["partisans_a"], zoom=(1.05, 1.22), center=((0.5, 0.45), (0.52, 0.42)),
                                   grade="bw", contrast=1.45, fxs=(flicker,)),
                           T("USTANAK", "anton", 0.2, WHITE, 0.5, 0.5, at=b(2), s0=1.4, slam=0.3, drift=0.02,
                             tracking=0.18)))
    cap(tl, 6, 8, "1941 · THE PARTISAN UPRISING")
    cut(tl, 8, 10, S_layers(S_photo(PH["partisans_b"], zoom=(1.1, 1.25), grade="red", contrast=1.5, fxs=(flicker,)),
                            T("SMRT FAŠIZMU", "anton", 0.22, WHITE, 0.5, 0.5, s0=1.6, slam=0.2, drift=0.03)))
    cap(tl, 8, 10, "DEATH TO FASCISM")
    cut(tl, 10, 12, S_layers(S_photo(PH["partisans_c"], zoom=(1.25, 1.1), grade="bw", contrast=1.5, fxs=(flicker,)),
                             T("SLOBODA NARODU", "anton", 0.22, RED, 0.5, 0.5, s0=1.6, slam=0.2, drift=0.03,
                               stroke=0.004, stroke_color=BLACK)))
    cap(tl, 10, 12, "FREEDOM TO THE PEOPLE")
    cut(tl, 12, 14, S_layers(S_photo(PH["partisans_d"], zoom=(1.0, 1.35), grade="bw", contrast=1.6, fxs=(flicker,)),
                             O_edges(RED, 5.0, 0.35)))
    cap(tl, 12, 14, "1941 — 1945")
    # the star rises out of the dark and punches through the lens into the build
    cut(tl, 14, 16, S_layers(S_solid(BLACK),
                             lambda c, img: over(img, g.star2d(c.lt, c.dur, c.W, c.H, size=0.25 + 1.6 * ease_in_cubic(c.p) ** 1.5)),
                             O_bloom(0.5, 1.2, 40)))
    tl.op(0, b(16), lambda c, img: img * (0.9 + 0.1 * math.sin(c.t * 37) * np.random.default_rng(c.frame + 3).random()),
          z=41, name="projector")


# ======================================================================= BUILD
COUNT_BG = (0.035, 0.03, 0.04)


def count_item(num, word, visual, bg=COUNT_BG, num_color=RED, grade=None):
    """Countdown card: giant numeral left, word under it, visual on the right 60%."""
    def f(c):
        img = blank(c.W, c.H, bg)
        vis = visual(c)
        if vis is not None:
            if vis.shape[2] == 4:
                over(img, vis)
            else:
                img = vis
        if grade:
            img = fx.grade(img, grade)
        # numeral with hard slam
        p = min(c.lt / 0.1, 1.0)
        s = 1 + 1.5 * (1 - ease_out_expo(p))
        Ln = tx.text_layer(num, "anton", int(c.H * 0.78), num_color, stroke=int(c.H * 0.004), stroke_color=BLACK)
        tx.place(img, Ln, 0.19, 0.44, s)
        sz = min(int(c.H * 0.12), tx.fit_size(word, "anton", c.W * 0.34, c.H))
        Lw = tx.text_layer(word, "anton", sz, WHITE, tracking=0.04)
        q = ease_out_expo(min(max(c.lt - 0.03, 0) / 0.12, 1))
        tx.place(img, Lw, 0.19 - 0.08 * (1 - q), 0.87, opacity=q)
        return img
    return f


def buildup(tl):
    B0 = 16
    # 7 neighbours
    cut(tl, B0, B0 + 2, count_item("7", "SUSEDA",
        lambda c: g.yugo_map(c.lt, c.dur, c.W, c.H, mode="neighbors", offset=(0.19, 0.0), scale=0.95,
                             counter=False, hud=False, start=0.02, end=0.8)))
    cap(tl, B0, B0 + 2, "7 NEIGHBOURS", cy=0.95, cx=0.19)
    # 6 republics
    reps = ["SLOVENIJA", "HRVATSKA", "BOSNA I HERCEGOVINA", "SRBIJA", "CRNA GORA", "MAKEDONIJA"]

    def rep_vis(c):
        k = min(int(c.p * 6), 5)
        lay = g.yugo_map(c.lt, c.dur, c.W, c.H, mode="republics", hi=k, local=(c.p * 6) % 1.0,
                         labels=False, hud=False, offset=(0.2, -0.04), scale=0.82)
        from edit.elements import _place_rgba
        L = tx.text_layer(reps[k], "anton", min(int(c.H * 0.1), tx.fit_size(reps[k], "anton", c.W * 0.5, c.H)), WHITE)
        _place_rgba(lay, L, 0.69, 0.9, 1 + 0.25 * (1 - ease_out_expo(((c.p * 6) % 1.0) * 4)))
        return lay
    cut(tl, B0 + 2, B0 + 4, count_item("6", "REPUBLIKA", rep_vis))
    cap(tl, B0 + 2, B0 + 4, "6 REPUBLICS", cy=0.95, cx=0.19)
    # 5 nations
    nations = ["SRBI", "HRVATI", "SLOVENCI", "MAKEDONCI", "CRNOGORCI"]

    def nations_vis(c):
        ph = S_photo(PH["crowd"], zoom=(1.15, 1.3), grade="blue", contrast=1.4)(c)
        k = min(int(c.p * 5), 4)
        tx.place(ph, tx.text_layer(nations[k], "anton", int(c.H * 0.2), WHITE), 0.68, 0.45,
                 1 + 0.3 * (1 - ease_out_expo((c.p * 5) % 1 * 3)))
        return ph
    cut(tl, B0 + 4, B0 + 6, count_item("5", "NARODA", nations_vis))
    cap(tl, B0 + 4, B0 + 6, "5 NATIONS", cy=0.95, cx=0.19)
    # 4 languages
    langs = [("ЈЕЗИК", "SRPSKI"), ("JEZIK", "HRVATSKI"), ("JEZIK", "SLOVENSKI"), ("ЈАЗИК", "MAKEDONSKI")]

    def lang_vis(c):
        img = np.zeros((c.H, c.W, 4), np.float32)
        k = min(int(c.p * 4), 3)
        w, sub = langs[k]
        col = (RED, WHITE, BLUE, GOLD)[k]
        from edit.elements import _place_rgba
        _place_rgba(img, tx.text_layer(w, "russo", int(c.H * 0.24), col), 0.68, 0.44)
        _place_rgba(img, tx.text_layer(sub, "mono", int(c.H * 0.04), WHITE, tracking=0.2), 0.68, 0.64)
        return img
    cut(tl, B0 + 6, B0 + 8, count_item("4", "JEZIKA", lang_vis))
    cap(tl, B0 + 6, B0 + 8, "4 LANGUAGES", cy=0.95, cx=0.19)
    # 3 faiths
    cut(tl, B0 + 8, B0 + 10, count_item("3", "VERE",
        lambda c: _shift_rgba(el.faith_symbols(c.lt, c.dur, c.W, c.H, hits=(0.0, BEAT * 0.5, BEAT), size=0.28), 0.17)))
    cap(tl, B0 + 8, B0 + 10, "3 FAITHS", cy=0.95, cx=0.19)
    # 2 scripts — flips every 16th
    def scripts_vis(c):
        img = np.zeros((c.H, c.W, 4), np.float32)
        k = int(c.lt / (BEAT / 4))
        s = "JUGOSLAVIJA" if k % 2 == 0 else "ЈУГОСЛАВИЈА"
        from edit.elements import _place_rgba
        sz = tx.fit_size(s, "russo", c.W * 0.58, c.H * 0.3)
        _place_rgba(img, tx.text_layer(s, "russo", sz, WHITE if k % 2 == 0 else RED), 0.68, 0.46)
        _place_rgba(img, tx.text_layer("LATINICA" if k % 2 == 0 else "ЋИРИЛИЦА", "mono", int(c.H * 0.04), GOLD,
                                       tracking=0.3), 0.68, 0.64)
        return img
    cut(tl, B0 + 10, B0 + 12, count_item("2", "PISMA", scripts_vis))
    cap(tl, B0 + 10, B0 + 12, "2 ALPHABETS", cy=0.95, cx=0.19)
    # 1 TITO — the portrait, then the gap
    def tito_card(c):
        img = S_photo(PH["tito_hero"], zoom=(1.08, 1.2), center=((0.5, 0.4), (0.5, 0.38)), grade="bw", contrast=1.55)(c)
        img = img * 0.85
        Ln = tx.text_layer("1", "anton", int(c.H * 0.9), RED)
        tx.place(img, Ln, 0.16, 0.47, 1 + 1.5 * (1 - ease_out_expo(min(c.lt / 0.1, 1))), mode="normal")
        Lt = tx.text_layer("TITO", "anton", int(c.H * 0.34), WHITE, tracking=0.08)
        q = ease_out_expo(min(max(c.lt - BEAT, 0) / 0.12, 1))
        if c.lt >= BEAT:
            tx.place(img, Lt, 0.7, 0.78, 1 + 1.2 * (1 - q))
        return img
    cut(tl, B0 + 12, B0 + 15, tito_card)
    fl(tl, B0 + 13, WHITE, 0.06)
    cap(tl, B0 + 12, B0 + 15, "1 TITO", cy=0.95, cx=0.16)
    # the gap: dead black, then a TV line switching on
    cut(tl, B0 + 15, B0 + 16, S_layers(S_solid(BLACK),
        lambda c, img: over(img, el.crt_on(max(c.lt - BEAT * 0.45, 0), BEAT * 0.55, c.W, c.H)) if c.lt > BEAT * 0.45 else img))
    # build camera: zoom creeping in, shake growing with the snare roll
    tl.op(b(B0), b(B0 + 15), O_camera(punch=0.0, shake=10, rgb=6, drift=4,
          extra_zoom=lambda c: 0.04 * (c.t - b(16)) / (b(31) - b(16))), z=40, name="cam-build")
    for k in range(1, 7):
        X_glitch(tl, b(B0 + 2 * k), 0.05, 1.0)


def _shift_rgba(lay, dx):
    """Shift an RGBA layer horizontally by fraction of width."""
    W = lay.shape[1]
    s = int(dx * W)
    out = np.zeros_like(lay)
    out[:, s:] = lay[:, :W - s]
    return out


# ====================================================================== DROP A
def dropA(tl):
    # --- b32 impact: star + SFRJ
    def impact(c):
        img = blank(c.W, c.H, RED)
        star = g.render_model(c.lt, c.dur, c.W, c.H, model="star", style="solid", spin=3.0, zoom=1.6 - 0.5 * ease_out_expo(c.p * 2))
        Ls = tx.text_layer("SFRJ", "unbounded", int(c.H * 0.46), BLACK)
        tx.place(img, Ls, 0.5, 0.5, 1.25 - 0.15 * c.p)
        over(img, star)
        return img
    cut(tl, 32, 34, S_layers(impact, O_speedlines(0.7, WHITE, 0.35, decay=0.4), O_zoomblur(0.25, 0, 0.18)))
    fl(tl, 32, WHITE, 0.09)
    cap(tl, 32, 34, "SOCIALIST FEDERAL REPUBLIC OF YUGOSLAVIA")
    # --- b34 republics slam together
    cut(tl, 34, 36, S_gen(g.yugo_map, bg=NAVY, mode="assemble", dur=BEAT * 2))
    cap(tl, 34, 36, "SIX REPUBLICS · ONE FEDERATION")
    # --- b36 JUGOSLAVIJA through letters
    cut(tl, 36, 38, S_img_text("JUGOSLAVIJA", PH["partisans_b"], "anton", 0.62, grade="bw", bg=RED,
                               open_at=BEAT * 1.4, open_dur=BEAT * 0.6))
    # --- b38 tilt3d map + capitals
    cut(tl, 38, 40, S_gen(g.yugo_map, bg=BLACK, mode="tilt3d"))
    # --- b40-48 liberation 1945 — 4 photos, 1 beat each, then OSLOBOĐENJE
    lib = [(PH["partisans_e"], "bw"), (PH["partisans_f"], "red"), (PH["partisans_g"], "bw"), (PH["liberation"], "red")]
    for k, (ph, gr) in enumerate(lib):
        cut(tl, 40 + k, 41 + k, S_photo(ph, zoom=(1.25, 1.1) if k % 2 else (1.1, 1.3), grade=gr, contrast=1.5,
                                        rot=(-2, 2) if k % 2 else (1.5, -1)))
    tl.op(b(40), b(44), T("1945", "anton", 0.55, WHITE, 0.5, 0.5, s0=1.3, slam=0.15, drift=0.04,
                         stroke=0.004, stroke_color=BLACK), z=20)
    cut(tl, 44, 48, S_layers(S_photo(PH["liberation_b"], zoom=(1.0, 1.25), grade="bw", contrast=1.5),
                             T_echo("OSLOBOĐENJE", "anton", 0.2, WHITE, 0.5, 0.5, n=4, dy=0.19, outline_color=RED),
                             O_invert_on([BEAT * 2], 0.07)))
    cap(tl, 40, 48, "1945 · THE PARTISANS LIBERATE THE COUNTRY")
    for k in range(40, 48):
        X_glitch(tl, b(k), 0.04, 0.8)

    # --- b48 1948: NE.
    cut(tl, 48, 50, S_layers(S_solid(BLACK), T_type("28. JUN 1948.", "mono", 0.07, WHITE, 0.5, 0.5, cps=30)))
    cut(tl, 50, 52, S_layers(S_photo(PH["tito_stern"], zoom=(1.2, 1.3), grade="bw", contrast=1.6),
                             T("NE.", "anton", 0.9, RED, 0.5, 0.5, s0=3.0, slam=0.1, stroke=0.006, stroke_color=BLACK)))
    fl(tl, 50, RED, 0.1)
    cap(tl, 48, 52, "1948 · YUGOSLAVIA SAYS NO TO STALIN")
    cut(tl, 52, 56, S_layers(S_gen(g.europe_blocs, bg=BLACK),
                             T("IZMEĐU ISTOKA I ZAPADA", "anton", 0.1, WHITE, 0.5, 0.88, at=b(1), s0=1.5)))
    cap(tl, 52, 56, "NEITHER EAST NOR WEST", cy=0.96)
    # --- b56 self-management
    cut(tl, 56, 58, S_layers(S_photo(PH["industry_a"], zoom=(1.1, 1.3), grade=None),
                             O_halftone(9, BLACK, RED),
                             T_scramble("SAMOUPRAVLJANJE", "anton", 0.2, WHITE, 0.5, 0.5, dur=BEAT * 1.2)))
    cut(tl, 58, 60, S_layers(S_photo(PH["industry_b"], zoom=(1.25, 1.05), grade="bw", contrast=1.4),
                             T("FABRIKE", "anton", 0.3, WHITE, 0.5, 0.36, s0=2), T("RADNICIMA", "anton", 0.3, RED, 0.5, 0.66,
                                                                                    at=BEAT, s0=2, stroke=0.004)))
    cap(tl, 56, 60, "1950 · FACTORIES TO THE WORKERS")
    # 8th-note strobe of work photos
    work = [PH["industry_a"], PH["industry_b"], PH["industry_c"], PH["construction"], PH["sava"], PH["ceremony"]]
    parts = [(k * BEAT / 2, S_photo(work[k % len(work)], zoom=(1.3, 1.4), grade=("red", "bw", "blue")[k % 3]))
             for k in range(6)]
    cut(tl, 60, 63, S_layers(S_seq(parts), T("RADNIČKO SAMOUPRAVLJANJE", "anton", 0.11, WHITE, 0.5, 0.5, slam=0,
                                                   fit_w=0.9)))
    cap(tl, 60, 63, "WORKERS' SELF-MANAGEMENT")
    # gap beat 63: freeze negative
    cut(tl, 63, 64, S_layers(S_photo(PH["industry_b"], zoom=(1.4, 1.4), grade="bw", contrast=2.0),
                             O_fx(fx.invert), O_fx(fx.threshold, 0.5, BLACK, WHITE)))

    # --- b64 youth work brigades
    cut(tl, 64, 66, S_layers(S_gen(el.railway, bg=BLACK, speed=1.4),
                             T("OMLADINSKE", "anton", 0.2, WHITE, 0.5, 0.2, s0=1.8, stroke=0.004),
                             T("RADNE AKCIJE", "anton", 0.2, RED, 0.5, 0.4, at=BEAT, s0=1.8, stroke=0.004)))
    cut(tl, 66, 68, S_layers(S_gen(el.railway, bg=BLACK, speed=2.4, t_off=b(2)),
                             T("BRČKO–BANOVIĆI", "anton", 0.14, WHITE, 0.5, 0.16, fit_w=0.9, stroke=0.004),
                             lambda c, img: T(f"{int(92 * ease_out_cubic(min(c.p * 1.25, 1)))} KM", "anton", 0.2, GOLD,
                                              0.5, 0.34, slam=0, stroke=0.004)(c, img)))
    cap(tl, 64, 66, "YOUTH BRIGADES BUILT THE RAILWAYS")
    cap(tl, 66, 68, "1946 · 62,000 YOUNG VOLUNTEERS")
    cut(tl, 68, 72, S_layers(S_gen(el.highway, bg=BLACK, speed=1.6),
                             T("AUTOPUT", "anton", 0.22, WHITE, 0.5, 0.2, s0=2.0),
                             T("BRATSTVO I JEDINSTVO", "anton", 0.12, GOLD, 0.5, 0.36, at=BEAT, s0=1.6)))
    cap(tl, 68, 72, "ZAGREB–BELGRADE · 382 KM · 1948–50")
    # --- growth + literacy
    cut(tl, 72, 76, S_layers(S_solid(NAVY),
        lambda c, img: over(img, el.bar_chart(c.lt, c.dur, c.W, c.H, values=(9.5, 6.6, 6.1),
                                              labels=("1953–59", "1960–69", "1970–79"),
                                              hits=[BEAT, BEAT * 2, BEAT * 3], unit="%", decimals=1)),
        T("RAST ~6% GODIŠNJE", "anton", 0.12, GOLD, 0.5, 0.1, s0=1.6)))
    cap(tl, 72, 76, "~6% GROWTH A YEAR, 1953–79")
    cut(tl, 76, 78, S_layers(S_gen(el.alphabet_wall, bg=NAVY, speed=1.5, alpha=0.22),
                             T("NEPISMENOST", "anton", 0.2, WHITE, 0.5, 0.36, s0=1.6),
                             lambda c, img: T(f"{25 - 15.5 * ease_out_cubic(min(c.p * 1.3, 1)):.1f}%".replace(".", ","),
                                              "anton", 0.3, GOLD, 0.5, 0.66, slam=0)(c, img)))
    cap(tl, 76, 78, "ILLITERACY · 1948: 25% → 1981: 9.5%")
    cut(tl, 78, 80, S_layers(S_photo(PH["crowd_b"], zoom=(1.2, 1.05), grade="gold", contrast=1.3),
                             T("ŽIVOTNI VEK", "anton", 0.2, WHITE, 0.5, 0.36, s0=1.6),
                             lambda c, img: T(f"{50 + 20 * ease_out_cubic(min(c.p * 1.3, 1)):.0f}", "anton", 0.3, GOLD,
                                              0.5, 0.66, slam=0)(c, img)))
    cap(tl, 78, 80, "LIFE EXPECTANCY · ~50 → ~70 YEARS")

    # --- passport
    stamps = [(b(3) + k * BEAT / 2, lab) for k, lab in
              enumerate(["PARIS", "MOSKVA", "ROMA", "PRAHA", "BEČ", "VARŠAVA"])]
    cut(tl, 80, 86, S_layers(S_gen(g.passport, bg=DRED, open_at=b(1.5), stamps=stamps, scale=0.68, cy=0.54),
                             T("CRVENI PASOŠ", "anton", 0.1, WHITE, 0.5, 0.075, s0=1.5)))
    cap(tl, 80, 86, "VISA-FREE TO MOST OF EAST AND WEST")
    # --- NAM
    cut(tl, 86, 88, S_seq([(0, S_photo(PH["nam_a"], zoom=(1.2, 1.3), grade="gold", contrast=1.4)),
                           (BEAT / 2, S_photo(PH["nam_b"], zoom=(1.3, 1.2), grade="bw", contrast=1.4)),
                           (BEAT, S_photo(PH["nam_c"], zoom=(1.2, 1.35), grade="gold", contrast=1.4)),
                           (BEAT * 1.5, S_photo(PH["nam_d"], zoom=(1.1, 1.3), center=((0.55, 0.45), (0.55, 0.4)),
                                                grade="bw", contrast=1.4))]))
    cut(tl, 88, 94, S_layers(S_gen(g.world_nam, bg=BLACK),
                             T("NESVRSTANI", "anton", 0.14, GOLD, 0.5, 0.12, at=0, s0=1.6),
                             T("BEOGRAD 1961 · 25 ZEMALJA", "mono", 0.05, WHITE, 0.5, 0.22, at=BEAT, s0=1.2)))
    cap(tl, 86, 94, "1961 · THE NON-ALIGNED MOVEMENT IS BORN IN BELGRADE")
    # --- 16th blitz into the break
    blitz = [PH[k] for k in ("tito_hero", "partisans_b", "spomenik_b", "nam_a", "industry_b", "spomenik_d",
                             "partisans_a", "arch_a")]
    parts = [(k * BEAT / 4, S_photo(ph, zoom=(1.4, 1.4), grade=("red", "bw", "blue", "gold")[k % 4], contrast=1.6))
             for k, ph in enumerate(blitz)]
    cut(tl, 94, 96, S_layers(S_seq(parts), O_strobe((None, None, WHITE, None), BEAT / 4, amt=0.4)))

    # camera for the whole drop
    tl.op(b(32), b(96), O_camera(punch=0.05, shake=18, rgb=10), z=40, name="cam-dropA")
    for t_ in (36, 38, 48, 52, 56, 64, 68, 72, 76, 80, 86, 88):
        X_whip(tl, b(t_), dx=1 if t_ % 8 else -1, half=0.08)


# ====================================================================== BREAK
def brk(tl):
    cut(tl, 96, 100, S_layers(S_gen(g.render_model, bg=(0.03, 0.03, 0.04), model="tjentiste", style="solid", spin=0.35),
                              T("SPOMENICI", "serif", 0.11, WHITE, 0.5, 0.2, s0=1.0, slam=0.6, tracking=0.3)))
    fl(tl, 96, WHITE, 0.25)
    cap(tl, 96, 100, "MONUMENTS TO THE FALLEN")
    cut(tl, 100, 102, S_photo(PH["spomenik_a"], zoom=(1.0, 1.15), center=((0.5, 0.6), (0.5, 0.55)), grade="steel",
                              contrast=1.3))
    cap(tl, 100, 102, "TJENTIŠTE · SUTJESKA · 1971")
    cut(tl, 102, 104, S_photo(PH["spomenik_fog"], zoom=(1.0, 1.12), grade="steel", contrast=1.2))
    cap(tl, 102, 104, "KOZARA · 1972")
    cut(tl, 104, 108, S_layers(S_seq([(0, S_photo(PH["funeral"], zoom=(1.0, 1.12), grade="bw", contrast=1.3)),
                                      (BEAT * 2, S_photo(PH["condolences"], zoom=(1.05, 1.2), grade="bw", contrast=1.3))]),
                               O_fx(lambda im: im * 0.55),
                               T("4. MAJ 1980.", "mono", 0.06, WHITE, 0.5, 0.2, slam=0),
                               lambda c, img: T(f"{int(120 * ease_out_cubic(min(c.p * 1.4, 1)))}" + ("+" if c.p > 0.72 else ""),
                                                "anton", 0.4, WHITE, 0.5, 0.52, slam=0)(c, img),
                               T("ZEMALJA", "anton", 0.1, RED, 0.5, 0.78, slam=0)))
    cap(tl, 104, 108, "TITO'S FUNERAL · 120+ COUNTRIES", cy=0.93)
    cut(tl, 108, 111, S_layers(S_photo(PH["batons"], zoom=(1.0, 1.25), grade="red", contrast=1.2),
                               O_fx(lambda im: im * 0.45), T_type("DRUŽE TITO,", "anton", 0.13, WHITE, 0.5, 0.4, cps=12),
                               T_type("MI TI SE KUNEMO", "anton", 0.13, WHITE, 0.5, 0.6, cps=16, at=BEAT * 1.2)))
    cap(tl, 108, 111, "COMRADE TITO, WE SWEAR TO YOU")
    cut(tl, 111, 112, S_solid(BLACK))
    tl.op(b(96), b(111), O_camera(punch=0.0, shake=3, rgb=0, drift=5), z=40)
    # flicker increases towards the drop
    tl.op(b(108), b(111), lambda c, img: img * (1 - 0.6 * c.p * (c.frame % 2)), z=42)


# ====================================================================== DROP B
def dropB(tl):
    cut(tl, 112, 114, S_layers(S_gen(g.flag, bg=BLACK, wave=1.4, scale=1.1),
                               O_strobe((None, RED, None, BLUE, None, WHITE, None, None), BEAT / 4, "multiply", 0.9),
                               O_speedlines(0.5, WHITE, decay=0.3)))
    fl(tl, 112, WHITE, 0.1)
    cut(tl, 114, 116, S_layers(S_gen(g.coat_of_arms, bg=NAVY), O_bloom(0.7, 0.6)))
    cap(tl, 114, 116, "29 NOVEMBER 1943 · AVNOJ")
    # sarajevo 84
    cut(tl, 116, 118, S_layers(S_gen(g.snowflake, bg=BLUE),
                               T("SARAJEVO '84", "anton", 0.2, WHITE, 0.5, 0.76, s0=1.8)))
    cut(tl, 118, 120, S_layers(S_seq([(0, S_band(PH["sport_a"], zoom=(1.0, 1.1))),
                                      (BEAT, S_photo(PH["sport_bob"], zoom=(1.1, 1.4), rot=(0, -4), grade="blue")),
                                      (BEAT * 1.5, S_photo(PH["sport_jumps"], zoom=(1.2, 1.4), grade="blue"))]),
                               T("XIV ZIMSKE OLIMPIJSKE IGRE", "anton", 0.09, WHITE, 0.5, 0.14, fit_w=0.9, stroke=0.003)))
    cap(tl, 116, 118, "1984 · XIV WINTER OLYMPICS")
    cap(tl, 118, 120, "FIRST WINTER GAMES IN A SOCIALIST COUNTRY")
    # basketball
    def bball(c):
        img = blank(c.W, c.H, (0.9, 0.35, 0.05))
        over(img, g.basketball(c.lt, c.dur, c.W, c.H))
        for k, yr in enumerate(("1970", "1978", "1990")):
            if c.lt >= BEAT * (k + 1):
                T(yr, "anton", 0.22, WHITE, 0.2 + 0.3 * k, 0.56, at=BEAT * (k + 1), s0=2.2, stroke=0.005)(c, img)
        return img
    cut(tl, 120, 124, S_layers(bball, T("PRVACI SVETA", "anton", 0.15, BLACK, 0.5, 0.15, s0=1.5)))
    cap(tl, 120, 124, "BASKETBALL WORLD CHAMPIONS")
    cut(tl, 124, 126, S_layers(S_bg("stripes", RED, BLACK, speed=2.0),
                               T("ČILE '87", "anton", 0.3, WHITE, 0.5, 0.45, s0=2), T("PRVACI SVETA", "anton", 0.11, GOLD, 0.5, 0.7, at=BEAT, s0=1.6)))
    cap(tl, 124, 126, "1987 · U-20 WORLD CHAMPIONS")
    cut(tl, 126, 128, S_layers(S_gen(g.vinyl, bg=BLACK, label="JUGOTON"),
                               T("ROCK ME", "anton", 0.3, WHITE, 0.5, 0.5, s0=2.5, glow=RED),
                               T("EUROVIZIJA 1989", "mono", 0.05, GOLD, 0.5, 0.75)))
    cap(tl, 126, 128, "1989 · RIVA WINS EUROVISION")
    # new wave — names at 8ths with colour cycling
    bands = ["BIJELO DUGME", "AZRA", "EKV", "IDOLI", "RIBLJA ČORBA", "ŠARLO AKROBATA", "LAIBACH", "HAUSTOR",
             "PARNI VALJAK", "FILM", "ZABRANJENO PUŠENJE", "PRLJAVO KAZALIŠTE", "DISCIPLINA KIČME", "ELEKTRIČNI ORGAZAM",
             "PANKRTI", "BIJELO DUGME"]
    bgs = [RED, BLACK, WHITE, BLUE, GOLD, BLACK, RED, NAVY]
    fgs = [WHITE, RED, BLACK, WHITE, BLACK, WHITE, BLACK, GOLD]
    fonts = ["anton", "rubikmono", "unbounded", "russo", "anton", "glitch", "archivo", "bebas"]

    def newwave(c):
        k = int(c.lt / (BEAT / 2))
        img = blank(c.W, c.H, bgs[k % 8])
        vin = g.vinyl(c.lt, c.dur, c.W, c.H, label="JUGOTON")
        if k % 2:
            over(img, vin, opacity=0.35)
        name = bands[k % len(bands)]
        fn = fonts[k % len(fonts)]
        sz = tx.fit_size(name, fn, c.W * 0.86, c.H * 0.34)
        L = tx.text_layer(name, fn, sz, fgs[k % 8])
        lt = c.lt - k * BEAT / 2
        tx.place(img, L, 0.5, 0.5, 1 + 0.5 * (1 - ease_out_expo(min(lt / 0.08, 1))) + 0.05 * lt)
        return img
    cut(tl, 128, 136, newwave)
    cap(tl, 128, 136, "NOVI TALAS · THE YUGOSLAV NEW WAVE")
    # architecture — one beat each, alternating spins
    arch = [("arch_a", "GENEX · 1977", (0, 25)), ("arch_b", "ZAPADNA KAPIJA", (-3, 0)),
            ("arch_c", "PALATA FEDERACIJE", (0, 0)), ("arch_d", "AVALA", (2, -2)),
            ("arch_e", "UŠĆE", (0, 0)), ("arch_f", "BEOGRAĐANKA", (-2, 2))]
    for k, (ph, lab, rot) in enumerate(arch):
        cut(tl, 136 + k, 137 + k, S_layers(
            S_photo(PH[ph], zoom=(1.05, 1.3) if k % 2 == 0 else (1.3, 1.05), rot=rot,
                    grade=("steel", "bw", "gold")[k % 3], contrast=1.5),
            T(lab, "anton", 0.16, WHITE, 0.5, 0.14, s0=1.8, stroke=0.004, fit_w=0.9)))
    cap(tl, 136, 142, "BUILDING THE FUTURE")
    cut(tl, 142, 144, S_layers(S_gen(g.test_card, bg=BLACK), O_glitch(1.0, 1)))

    # --- switch-up: spomenik cycle per beat
    cycle = [("3d", "tjentiste", "neon", "TJENTIŠTE · 1971"), ("ph", "spomenik_b", "red", "BUBANJ · 1963"),
             ("3d", "kosmaj", "wire", "KOSMAJ · 1971"), ("ph", "spomenik_c", "gold", "PODGARIĆ · 1967"),
             ("3d", "makedonium", "neon", "KRUŠEVO · 1974"), ("ph", "spomenik_e", "steel", "KADINJAČA · 1979"),
             ("3d", "petrova_gora", "hologram", "PETROVA GORA · 1981"), ("ph", "spomenik_d", "red", "MAKEDONIUM")]
    for k, (kind, m, st, lab) in enumerate(cycle):
        if kind == "3d":
            shot = S_layers(S_gen(g.render_model, bg=BLACK, model=m, style=st, spin=1.8, yaw0=k),
                            T(lab, "mono", 0.05, WHITE, 0.5, 0.08, slam=0, tracking=0.15))
        else:
            shot = S_layers(S_photo(PH[m], zoom=(1.15, 1.4), grade=st, contrast=1.5),
                            O_halftone(7, BLACK, RED) if st == "red" else O_scan(0.3, 4),
                            T(lab, "mono", 0.05, WHITE, 0.5, 0.08, slam=0, tracking=0.15))
        cut(tl, 144 + k, 145 + k, shot)
    cap(tl, 144, 152, "SPOMENICI · CONCRETE MEMORY")
    # --- Fićo drift
    drift = S_gen(g.fico_drift, bg=(0.1, 0.1, 0.11), dur=b(8))
    cut(tl, 152, 160, S_layers(S_seq([(0, drift),
                                      (b(4), S_photo(PH["fico_a"], zoom=(1.3, 1.4), grade=None)),
                                      (b(4.5), lambda c: drift(c.at(-b(4.5), b(3.5)))),
                                      (b(6), S_photo(PH["fico_b"], zoom=(1.2, 1.35), grade=None)),
                                      (b(6.5), lambda c: drift(c.at(-b(6.5), b(1.5))))]),
                               T("FIĆO", "anton", 0.3, WHITE, 0.18, 0.18, s0=2.4, stroke=0.005),
                               T("ZASTAVA 750", "mono", 0.045, GOLD, 0.18, 0.33, at=BEAT),
                               T("923.487", "anton", 0.12, GOLD, 0.82, 0.18, at=BEAT * 4, s0=2.0)))
    cap(tl, 152, 156, "ZASTAVA 750 · 1955–85")
    cap(tl, 156, 160, "923,487 PEOPLE'S CARS BUILT")

    # --- finale: recap at 8ths
    rec = ["tito_hero", "partisans_b", "land_a", "nam_a", "spomenik_b", "industry_b", "land_c", "arch_a",
           "partisans_a", "spomenik_d", "land_b", "nam_b", "spomenik_c", "industry_a", "land_d", "arch_b"]
    grades = ["red", "bw", "blue", "gold", "red", "neon", "bw", "flag"]
    parts = [(k * BEAT / 2, S_photo(PH[r], zoom=(1.3, 1.5), grade=grades[k % 8], contrast=1.6,
                                   fxs=((O_grid(3),) if k % 4 == 3 else ()) + ((O_mirror("h"),) if k % 4 == 1 else ())))
             for k, r in enumerate(rec)]
    cut(tl, 160, 168, S_layers(S_seq(parts), O_rgb(6)))
    # BRATSTVO I JEDINSTVO
    def bij(c):
        img = blank(c.W, c.H, RED if int(c.lt / (BEAT / 2)) % 2 == 0 else BLACK)
        words = [("BRATSTVO", 0), ("I", BEAT), ("JEDINSTVO", BEAT * 2)]
        k = sum(1 for _, t0 in words if c.lt >= t0) - 1
        w, t0 = words[max(k, 0)]
        col = WHITE if img[0, 0, 0] < 0.5 else BLACK
        T_echo(w, "anton", 0.33 if w != "I" else 0.6, col, 0.5, 0.5, n=3, dy=0.28, at=0,
               outline_color=WHITE if col == BLACK else RED)(c.at(t0, BEAT), img)
        return img
    cut(tl, 168, 172, bij)
    cap(tl, 168, 172, "BROTHERHOOD AND UNITY")
    # final 16th strobe
    strobe_ph = ["tito_hero", "partisans_b", "spomenik_b", "arch_a"]

    def final_strobe(c):
        k = int(c.lt / (BEAT / 4))
        if k % 4 == 0:
            f = S_gen(g.flag, bg=BLACK, wave=2.0, scale=1.2)(c)
        elif k % 4 == 2:
            f = blank(c.W, c.H, RED)
            over(f, g.star2d(c.lt, c.dur, c.W, c.H, size=0.8))
        else:
            f = S_photo(PH[strobe_ph[k % 4]], zoom=(1.4, 1.6), grade="bw", contrast=1.8)(c)
        return f
    cut(tl, 172, 175, final_strobe)
    cut(tl, 175, 176, S_solid(BLACK))
    tl.op(b(112), b(175), O_camera(punch=0.06, shake=22, rgb=12), z=40, name="cam-dropB")
    for t_ in (114, 116, 118, 120, 124, 126, 136, 138, 140):
        X_zoom(tl, b(t_), 0.1) if t_ % 4 == 0 else X_spin(tl, b(t_), 0.1)


def S_gen_overlay(gen, **params):
    return lambda c, img: over(img, gen(c.lt, c.dur, c.W, c.H, **params))


# ====================================================================== OUTRO
def outro(tl):
    def final(c):
        img = blank(c.W, c.H, BLACK)
        star = g.render_model(c.lt, c.dur, c.W, c.H, model="star", style="solid", spin=0.8,
                              zoom=1.2 - 0.3 * ease_out_cubic(c.p), cy=0.42)
        over(img, star)
        img = fx.bloom(img, 0.6, 0.8, 40)
        T("SFRJ", "unbounded", 0.12, WHITE, 0.5, 0.78, at=0.25, s0=1.3, slam=0.4, tracking=0.4)(c, img)
        T("1945 — 1992", "mono", 0.045, GOLD, 0.5, 0.88, at=0.9, s0=1.0, slam=0.4, tracking=0.3)(c, img)
        fade = min(max((c.lt - (c.dur - 1.6)) / 1.6, 0), 1)
        return img * (1 - fade)
    cut(tl, 176, 176 + 11, final)
    fl(tl, 176, WHITE, 0.2)


# ====================================================================== GLOBAL
def build_timeline():
    tl = Timeline(W_, H_, 30, duration=86.0)
    intro(tl)
    buildup(tl)
    dropA(tl)
    brk(tl)
    dropB(tl)
    outro(tl)
    # letterbox: 2.39:1 in intro, closing tighter through the build, gone on the drop
    def lb(c):
        if c.t < b(16):
            return 0.12
        if b(96) <= c.t < b(112):
            return 0.1
        return 0.0
    tl.op(0, tl.duration, O_letterbox(lb), z=95, name="letterbox")
    tl.op(0, tl.duration, O_finish(0.05, 0.4), z=90, name="finish")
    tl.finalize()
    return tl


def build():
    return build_timeline()
