# SFRJ — hype edit spec

Shared contract between the music, assets, generator and edit modules.
Read `engine/core.py` for the image / generator conventions, palette and fonts.

## Format

* 1920x1080, 30 fps, H.264 + AAC, output `out/sfrj_hype.mp4`
* Python 3.11 + numpy / scipy / Pillow only. ffmpeg binary: `python3 -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())"`
* Everything is regenerated from code in this repo (`make`-style scripts), no manual steps.

## Beat grid (the single source of truth for sync)

`BPM = 130`, 4/4. `beat = 60/130 = 0.461538 s`, `bar = 1.846154 s`.
Absolute beat `b` starts at `t = b * 60/130`. Bar `n` starts at beat `4n`.

| section | bars | start (s) | end (s) | music | picture |
|---|---|---|---|---|---|
| `intro` | 0–3 | 0.000 | 7.385 | dark pad, vinyl crackle, radio static, **low-passed teaser of the cowbell hook**, reverse cymbal into bar 4 | 1941, Partisans, *SMRT FAŠIZMU – SLOBODA NARODU*, film grain, flicker |
| `build` | 4–7 | 7.385 | 14.769 | no kick; snare roll accelerating (4ths → 8ths → 16ths → 32nds), white-noise riser + filter sweep up; **7 countdown stabs** (distorted 808 hit + clap + orchestral-ish chord stab) on beats 16, 18, 20, 22, 24, 26, 28; beat 28 stab rings out; **beat 31 = total silence (drop gap)** | countdown 7 → 1 (see below) |
| `dropA` | 8–23 | 14.769 | 44.308 | full drift phonk: pitched 808-cowbell hook (16ths), distorted gliding 808, kick, clap/snare on 2 & 4, hats with 32nd rolls, sidechain pump. Mini drop gap on **last beat of bar 15** (beat 63). Downlifter + impact into the break | achievements montage |
| `break` | 24–27 | 44.308 | 51.692 | drums out; pad + reverb-drenched filtered cowbell hook; sub boom on beat 96; riser bars 26–27; **beat 111 = silence (drop gap)** | slow, emotional: spomenici, Tito funeral, oath |
| `dropB` | 28–43 | 51.692 | 81.231 | harder variant: extra counter-melody an octave up, more distortion, double-time hats. **Bars 36–39 = switch-up** (half-time 808 pattern + stuttered/chopped cowbell). Bars 40–43 finale with 1/32 stutters; **stop on beat 175** | culture / sport / architecture blitz, finale |
| `outro` | 44–45 | 81.231 | 84.923 | huge final impact on beat 176 (boom + long reverb), cowbell echo, tape-stop, fades to silence by ~86 s | red star, *SFRJ 1945–1992* |

Total length ≈ 86 s.

## Storyboard (picture)

Big words in Serbo-Croatian (Latin, with Cyrillic moments), small English caption under them in mono.

* **Intro** – `1941.` typewriter; B&W Partisan photos with slow push-ins; *SMRT FAŠIZMU / SLOBODA NARODU* (Death to fascism / Freedom to the people); red star glow.
* **Build** – the classic saying as a countdown, one item per stab:
  `7 SUSEDA` (map: Yugoslavia + 7 neighbours: Italy, Austria, Hungary, Romania, Bulgaria, Greece, Albania) ·
  `6 REPUBLIKA` (Slovenia, Croatia, Bosnia and Herzegovina, Serbia, Montenegro, Macedonia flash in turn) ·
  `5 NARODA` · `4 JEZIKA` · `3 VERE` · `2 PISMA` (JUGOSLAVIJA ⇄ ЈУГОСЛАВИЈА) · `1 TITO` (portrait) → gap → **DROP**.
* **Drop A** (2 bars each): impact + republics slam together into one map / giant `SFRJ` · 1945 liberation · 1948 `NE.` to Stalin · 1950 workers' self-management `FABRIKE RADNICIMA` · youth work brigades + `AUTOPUT BRATSTVO I JEDINSTVO` · growth / literacy counters · the red passport (visa-free East **and** West, stamps slamming) · 1961 Belgrade, Non-Aligned Movement (world map, arcs Belgrade → Cairo, New Delhi, Jakarta, Accra).
* **Break** – rotating spomenik (3D) + real spomenik photos, *1980: 128 zemalja* at Tito's funeral, `DRUŽE TITO, MI TI SE KUNEMO`.
* **Drop B** – flag + coat of arms strobe · Sarajevo '84 · basketball world champions 1970/78/90 · Eurovision 1989 `ROCK ME` · rock band names at 16ths · **switch-up**: brutalism + spomenik wireframes cycling per beat + a Zastava 750 (Fićo) drifting in tyre smoke · finale: everything, `BRATSTVO I JEDINSTVO`.
* **Outro** – red star, `SFRJ 1945–1992`.

## Deliverables per module

| path | owner | what |
|---|---|---|
| `music/phonk.py` → `out/audio/track.wav`, `out/audio/beatmap.json` | music agent | 44.1 kHz 16-bit stereo; beatmap = bpm, sections, every event time (see music prompt) |
| `assets/img/*.jpg`, `assets/img/manifest.json` | assets agent | real photographs with source URL + licence note |
| `gen/maps.py`, `assets/geo/*.json` | maps agent | Yugoslavia / neighbours / world (NAM) map generators |
| `gen/spomenik3d.py` | 3D agent | software-rendered spomenik models + extruded red star |
| `gen/emblems.py` | 3D agent | flag, coat of arms, passport + stamps, Fićo drift, test card |
| `engine/*.py`, `edit/*.py`, `render.py` | director (main session) | effects, typography, timeline, render + mux |

Every generator: `fn(t, dur, W, H, **params) -> float32 RGBA (H, W, 4)` straight alpha, deterministic,
< ~150 ms per call at 1920x1080 (cache static geometry), with a `__main__` that writes a contact sheet
PNG to `out/preview/` so it can be checked by eye.
