# SFRJ — a drift phonk hype edit about socialist Yugoslavia

`out/sfrj_hype.mp4` — 1920x1080, 30 fps, ~89 s. Everything except the archive photographs is
generated in code: the music, the maps, the 3D spomenici, the emblems, the typography and every
cut, grade and transition.

## Structure of the edit (130 BPM)

| time | section | what happens |
|---|---|---|
| 0:00 | intro | 6 April 1941, Partisan columns, *SMRT FAŠIZMU — SLOBODA NARODU*, the star rises |
| 0:07 | build | the old saying as a countdown on the stabs: 7 neighbours · 6 republics · 5 nations · 4 languages · 3 faiths · 2 alphabets · 1 Tito → silence |
| 0:15 | drop A | SFRJ · republics slam together · 1945 · *NE.* to Stalin in 1948 · between the blocs · workers' self-management · youth railways · the Brotherhood and Unity highway · growth, literacy, life expectancy · the red passport · Non-Aligned Movement, Belgrade 1961 |
| 0:44 | break | spomenici, Tito's funeral (120+ countries), *Druže Tito, mi ti se kunemo* |
| 0:52 | drop B | flag + coat of arms · Sarajevo '84 · basketball world champions · Čile '87 · *Rock Me* · the New Wave at 8th notes · Belgrade modernism · spomenik switch-up · a Fićo drifting · recap · *BRATSTVO I JEDINSTVO* |
| 1:21 | outro | the star, SFRJ 1945–1992, photo credits |

On-screen claims were fact-checked; verdicts and sources are in `research/facts.md`.

## Rebuild

```bash
pip install numpy scipy pillow imageio-ffmpeg     # shapely only for gen/geo_build.py
python3 music/phonk.py                            # -> out/audio/track.wav + beatmap.json (~40 s)
python3 render.py                                 # -> out/sfrj_hype.mp4 (1080p, ~10-15 min on 4 cores)
python3 render.py --scale 0.5 --out out/preview.mp4   # quick preview
python3 render.py --sheet 14.7 44.3 48            # contact sheet of a time range
```

## Layout

| path | what |
|---|---|
| `SPEC.md` | the contract every module was built against: beat grid, sections, interfaces |
| `engine/` | compositing core, effects (grades, halftone, glitch, blur, chroma, grain), kinetic type, Ken Burns, timeline |
| `edit/sfrj.py` | the edit itself — every cut placed on the beat grid |
| `edit/kit.py`, `edit/elements.py` | shot builders, ops, transitions; highway, railway, charts, alphabet wall |
| `gen/maps.py` | Yugoslavia (draw / assemble / republics / neighbours / capitals / 3D slab), 1961 NAM world map, Cold War blocs |
| `gen/spomenik3d.py` | software 3D renderer: Tjentište, Kosmaj, Jasenovac, Makedonium, Petrova Gora, the star |
| `gen/emblems.py` | flag, coat of arms, passport + stamps, Fićo drift, JRT test card, vinyl, snowflake, basketball |
| `music/` | the drift phonk track, synthesised from scratch (no samples) |
| `assets/img/` | 66 archive photographs, `manifest.json` + `SOURCES.md` with licences |
| `CREDITS.md` | photo attribution, fonts (OFL), Natural Earth |

Built by a director session plus parallel agents for music, archive research, maps, 3D/emblems and fact-checking.
