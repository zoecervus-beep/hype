"""Build the compact map caches used by gen/maps.py.

    python3 gen/geo_build.py            # downloads Natural Earth once, writes assets/geo/*.json

Source: Natural Earth (public domain) - https://github.com/nvkelso/natural-earth-vector
The raw GeoJSON (~70 MB) is cached in out/cache/naturalearth/ and never read at
render time.  Build-time dependency: shapely (``pip install shapely``); the
render side (maps.py) needs only numpy / Pillow.

Outputs (all coordinates are *projected* and stored as integers):

* ``assets/geo/yugo.json``   Lambert conformal conic (lon0 18.5, lat0 44, std 41.5/46),
                             unit 0.1 km.  Republics, provinces, internal borders,
                             union outline (ordered for the draw-on reveal), the 7
                             neighbours, the rest of Europe for context, lakes,
                             rivers, cities.
* ``assets/geo/europe.json`` Lambert azimuthal equal-area (lon0 15, lat0 52), unit 1 km.
                             Cold-War entities (USSR, CSK, FRG/GDR split, SFRJ).
* ``assets/geo/world.json``  Equal Earth (lon0 11 -> seam in the Bering Strait), unit 1e-4 R.
                             1961 entities for the Non-Aligned map (USSR, UAR, North
                             Yemen, Indonesia without West New Guinea, ...).

Polygon encoding: a feature's ``p`` is a list of polygons, each a list of rings
(exterior first, then holes), each ring a flat ``[x0, y0, x1, y1, ...]`` int list.
Line encoding: a list of flat int lists.
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

import numpy as np

try:
    import shapely
    from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                                  MultiPolygon, Point, Polygon, box, shape)
    from shapely.ops import linemerge, polylabel, unary_union
except ImportError:  # pragma: no cover
    sys.exit("geo_build.py needs shapely at build time:  pip install shapely")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geoproj  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "out", "cache", "naturalearth")
DST = os.path.join(ROOT, "assets", "geo")
BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"

FILES = ["ne_10m_admin_0_countries.geojson", "ne_50m_admin_0_countries.geojson",
         "ne_10m_admin_1_states_provinces.geojson", "ne_10m_lakes.geojson",
         "ne_10m_rivers_lake_centerlines.geojson"]


# ------------------------------------------------------------------ io ----
def fetch(name: str) -> str:
    os.makedirs(SRC, exist_ok=True)
    path = os.path.join(SRC, name)
    legacy = os.path.join(ROOT, "out", "cache", "ne", name)
    if not os.path.exists(path) and os.path.exists(legacy):
        os.replace(legacy, path)
    if not os.path.exists(path):
        print("  downloading", name)
        tmp = path + ".part"
        urllib.request.urlretrieve(BASE + name, tmp)
        os.replace(tmp, path)
    return path


_LOADED: dict = {}


def load(name: str):
    if name not in _LOADED:
        with open(fetch(name), encoding="utf-8") as fh:
            d = json.load(fh)
        feats = []
        for f in d["features"]:
            if f.get("geometry") is None:
                continue
            g = shape(f["geometry"])
            if not g.is_valid:
                g = g.buffer(0)
            feats.append((f["properties"], g))
        _LOADED[name] = feats
    return _LOADED[name]


def adm0(name: str) -> dict:
    out = {}
    for p, g in load(name):
        out[p["ADM0_A3"]] = g
    return out


# ------------------------------------------------------------- geometry ---
def polys_of(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if isinstance(g, GeometryCollection):
        return [q for x in g.geoms for q in polys_of(x)]
    return []


def lines_of(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    if isinstance(g, (MultiLineString, GeometryCollection)):
        return [q for x in g.geoms for q in lines_of(x)]
    return []


def proj_geom(g, fn, scale):
    def tr(xy):
        x, y = fn(xy[:, 0], xy[:, 1])
        return np.column_stack([x * scale, y * scale])
    return shapely.transform(g, tr)


def clean(g, eps):
    """Close hairline gaps between adjacent source polygons."""
    return g.buffer(eps, join_style="mitre").buffer(-eps, join_style="mitre")


def enc_ring(coords) -> list:
    a = np.rint(np.asarray(coords)[:, :2]).astype(np.int64)
    # drop consecutive duplicates created by rounding
    keep = np.ones(len(a), bool)
    keep[1:] = np.any(a[1:] != a[:-1], axis=1)
    a = a[keep]
    return a.ravel().tolist()


def enc_poly(g, tol, min_area, hole_min=None) -> list:
    """Simplify + encode (Multi)Polygon -> [[ring, hole, ...], ...]."""
    hole_min = min_area if hole_min is None else hole_min
    out = []
    for p in polys_of(g):
        if p.area < min_area:
            continue
        s = p.simplify(tol, preserve_topology=True)
        for q in polys_of(s):
            if q.area < min_area or len(q.exterior.coords) < 4:
                continue
            q = shapely.geometry.polygon.orient(q, sign=-1.0)  # exterior clockwise (north-up)
            rings = [enc_ring(q.exterior.coords)]
            for h in q.interiors:
                if Polygon(h).area >= hole_min:
                    rings.append(enc_ring(h.coords))
            if len(rings[0]) >= 8:
                out.append(rings)
    return out


def enc_lines(g, tol, min_len=0.0) -> list:
    ls = lines_of(g)
    if len(ls) > 1:
        g = linemerge(unary_union(ls))
    out = []
    for ln in lines_of(g):
        if ln.length < min_len:
            continue
        s = ln.simplify(tol, preserve_topology=False)
        r = enc_ring(s.coords)
        if len(r) >= 4:
            out.append(r)
    return out


def label_pt(g, tol):
    big = max(polys_of(g), key=lambda p: p.area)
    pt = polylabel(big, tolerance=tol)
    return [int(round(pt.x)), int(round(pt.y))]


def split_seam(g, lon0):
    """Cut lon/lat geometry at the anti-meridian of lon0 so no polygon wraps."""
    seam = lon0 + 180.0 if lon0 <= 0 else lon0 - 180.0
    if seam > 180:
        seam -= 360
    eps = 1e-7
    parts = []
    for p in polys_of(g):
        minx, _, maxx, _ = p.bounds
        if minx < seam < maxx:
            parts += polys_of(p.intersection(box(-181, -91, seam - eps, 91)))
            parts += polys_of(p.intersection(box(seam + eps, -91, 181, 91)))
        else:
            parts.append(p)
    return MultiPolygon(parts) if parts else g


# ============================================================ YUGOSLAVIA ===
YU_PROJ = {"type": "lcc", "lon0": 18.5, "lat0": 44.0, "lat1": 41.5, "lat2": 46.0}
YU_SCALE = 10.0  # 0.1 km units

REPUBLICS = [  # order is the public index used by maps.py (hi=0..5)
    ("SVN", "SLOVENIJA", "SLOVENIA", "Ljubljana"),
    ("HRV", "HRVATSKA", "CROATIA", "Zagreb"),
    ("BIH", "BOSNA I HERCEGOVINA", "BOSNIA AND HERZEGOVINA", "Sarajevo"),
    ("SRB", "SRBIJA", "SERBIA", "Beograd"),
    ("MNE", "CRNA GORA", "MONTENEGRO", "Titograd"),
    ("MKD", "MAKEDONIJA", "MACEDONIA", "Skopje"),
]
NEIGHBORS = [  # clockwise from the west, label anchor lon/lat (visible part)
    ("ITA", "ITALIJA", "ITALY", (13.2, 42.6)),
    ("AUT", "AUSTRIJA", "AUSTRIA", (14.6, 47.45)),
    ("HUN", "MAĐARSKA", "HUNGARY", (19.3, 47.15)),
    ("ROU", "RUMUNIJA", "ROMANIA", (24.6, 45.9)),
    ("BGR", "BUGARSKA", "BULGARIA", (25.2, 42.75)),
    ("GRC", "GRČKA", "GREECE", (22.0, 39.7)),
    ("ALB", "ALBANIJA", "ALBANIA", (20.05, 40.95)),
]
CITIES = [  # name (1963-1991 form), today's name, lon, lat, kind
    ("Beograd", "Belgrade", 20.4569, 44.8176, "federal"),
    ("Ljubljana", "Ljubljana", 14.5058, 46.0569, "republic"),
    ("Zagreb", "Zagreb", 15.9819, 45.8150, "republic"),
    ("Sarajevo", "Sarajevo", 18.4131, 43.8563, "republic"),
    ("Titograd", "Podgorica", 19.2594, 42.4304, "republic"),
    ("Skopje", "Skopje", 21.4254, 41.9981, "republic"),
    ("Novi Sad", "Novi Sad", 19.8335, 45.2671, "province"),
    ("Priština", "Pristina", 21.1655, 42.6629, "province"),
]
VOJVODINA_DISTRICTS = {"Severno-Backi", "Zapadno-Backi", "Južno-Backi", "Severno-Banatski",
                       "Srednje-Banatski", "Južno-Banatski", "Sremski"}
YU_BBOX = (-8.0, 29.0, 46.0, 59.0)  # context clip (lon/lat); edges stay off-screen for scale >= ~0.4


def build_yugo():
    print("yugo.json")
    a0 = adm0("ne_10m_admin_0_countries.geojson")
    fn = lambda x, y: geoproj.lcc(x, y, **{k: v for k, v in YU_PROJ.items() if k != "type"})  # noqa: E731
    P = lambda g: proj_geom(g, fn, YU_SCALE)  # noqa: E731
    eps = 0.8  # 80 m gap closing

    # --- republics & provinces (projected, 0.1 km units)
    srb = P(a0["SRB"])
    kos = P(a0["KOS"])
    voj_parts = [g for p, g in load("ne_10m_admin_1_states_provinces.geojson")
                 if p["adm0_a3"] == "SRB" and p["name"] in VOJVODINA_DISTRICTS]
    assert len(voj_parts) == 7, len(voj_parts)
    voj = clean(unary_union([P(g) for g in voj_parts]), eps).intersection(srb)
    uza = srb.difference(voj.buffer(eps))  # Serbia proper ("uža Srbija")
    uza = MultiPolygon([p for p in polys_of(uza) if p.area > 1e4]) if isinstance(uza, MultiPolygon) else uza
    srbija = clean(unary_union([srb, kos]), eps)
    reps = {}
    for code, *_ in REPUBLICS:
        reps[code] = srbija if code == "SRB" else P(a0[code])
    yugo = clean(unary_union(list(reps.values())), eps)
    # fill pinholes left by the union
    yugo = unary_union([Polygon(p.exterior) for p in polys_of(yugo)])

    TOL, MINA = 2.0, 100.0  # 200 m simplification, drop islets < 1 km^2
    border = yugo.boundary.buffer(3.0)
    rep_lines = unary_union([g.boundary for g in reps.values()]).difference(border)
    srbija_border = srbija.boundary.buffer(3.0)
    prov_lines = unary_union([voj.boundary, kos.boundary]).difference(srbija_border)

    # --- ordered union outline for the stroke reveal: main ring clockwise from the
    #     Italy/Austria/Slovenia tripoint (Peč, 13.71E 46.52N), then islands.
    polys = sorted(polys_of(yugo), key=lambda p: -p.area)
    main = polys[0].simplify(TOL, preserve_topology=True)
    main = shapely.geometry.polygon.orient(main, sign=-1.0)  # clockwise
    ring = np.asarray(main.exterior.coords)[:-1]
    sx, sy = fn(13.71, 46.52)
    k = int(np.argmin((ring[:, 0] - sx * YU_SCALE) ** 2 + (ring[:, 1] - sy * YU_SCALE) ** 2))
    ring = np.vstack([ring[k:], ring[:k], ring[k:k + 1]])
    islands = []
    for p in polys[1:]:
        if p.area < MINA:
            continue
        s = p.simplify(TOL, preserve_topology=True)
        for q in polys_of(s):
            islands.append(enc_ring(q.exterior.coords))

    # coarse versions for the 3D slab (walls are built per edge)
    coarse = []
    for p in polys:
        if p.area < 3000:  # > 30 km^2 islands only
            continue
        coarse += enc_poly(p, 12.0, 3000)
    coarse_lines = enc_lines(rep_lines, 12.0, 20.0)

    rep_out = []
    for (code, name, en, cap), in zip(REPUBLICS):
        g = reps[code]
        rep_out.append({"id": code, "name": name, "en": en, "capital": cap,
                        "p": enc_poly(g, TOL, MINA), "label": label_pt(g, 10.0),
                        "area_km2": round(g.area / 100.0)})
    prov_out = [
        {"id": "VOJ", "name": "VOJVODINA", "en": "VOJVODINA", "capital": "Novi Sad",
         "p": enc_poly(voj, TOL, MINA), "label": label_pt(voj, 10.0)},
        {"id": "KOS", "name": "KOSOVO", "en": "KOSOVO", "capital": "Priština",
         "p": enc_poly(kos, TOL, MINA), "label": label_pt(kos, 10.0)},
        {"id": "UZA", "name": "UŽA SRBIJA", "en": "SERBIA PROPER", "capital": "Beograd",
         "p": enc_poly(uza, TOL, MINA), "label": label_pt(uza, 10.0)},
    ]

    # --- neighbours + context (clipped to a generous box)
    clip = box(*YU_BBOX)
    yu_codes = {c for c, *_ in REPUBLICS} | {"KOS"}
    merge = {"ITA": ["ITA", "SMR", "VAT"], "CYP": ["CYP", "CYN", "CNM", "ESB", "WSB"]}
    skip = set(yu_codes) | {"SMR", "VAT", "CYN", "CNM", "ESB", "WSB"}
    nb_out = []
    for code, name, en, (lon, lat) in NEIGHBORS:
        g = unary_union([a0[c] for c in merge.get(code, [code])]).intersection(clip)
        g = clean(P(g), eps)
        lx, ly = fn(lon, lat)
        nb_out.append({"id": code, "name": name, "en": en, "p": enc_poly(g, 3.0, 200.0),
                       "label": [int(lx * YU_SCALE), int(ly * YU_SCALE)]})
    nb_codes = {c for c, *_ in NEIGHBORS}
    ctx_out = []
    for code, g in sorted(a0.items()):
        if code in skip or code in nb_codes or not g.intersects(clip):
            continue
        if code in merge:
            g = unary_union([a0[c] for c in merge[code]])
        g = P(g.intersection(clip))
        p = enc_poly(g, 5.0, 500.0)
        if p:
            ctx_out.append({"id": code, "p": p})

    # --- lakes & rivers
    lakes = []
    for p, g in load("ne_10m_lakes.geojson"):
        if g.intersects(clip) and g.area > 0.004:
            lakes += enc_poly(P(g.intersection(clip)), 2.0, 300.0)
    rivers = []
    for p, g in load("ne_10m_rivers_lake_centerlines.geojson"):
        if p.get("scalerank", 99) <= 8 and g.intersects(box(9, 38, 30, 50)):
            for ln in enc_lines(P(g.intersection(box(9, 38, 30, 50))), 3.0, 50.0):
                rivers.append({"name": p.get("name"), "l": ln})

    cities = []
    for name, today, lon, lat, kind in CITIES:
        x, y = fn(lon, lat)
        cities.append({"name": name, "today": today, "lon": lon, "lat": lat, "kind": kind,
                       "xy": [int(round(x * YU_SCALE)), int(round(y * YU_SCALE))]})

    return {
        "meta": {"source": "Natural Earth 10m (public domain), built by gen/geo_build.py",
                 "proj": YU_PROJ, "scale": YU_SCALE, "unit": "0.1 km",
                 "bounds": [int(v) for v in yugo.bounds]},
        "republics": rep_out,
        "provinces": prov_out,
        "rep_borders": enc_lines(rep_lines, TOL, 5.0),
        "prov_borders": enc_lines(prov_lines, TOL, 5.0),
        "yugo": enc_poly(yugo, TOL, MINA),
        "outline": {"main": enc_ring(ring), "islands": islands},
        "coarse": {"yugo": coarse, "rep_borders": coarse_lines},
        "neighbors": nb_out,
        "context": ctx_out,
        "lakes": lakes,
        "rivers": rivers,
        "cities": cities,
    }


# ================================================================ EUROPE ===
EU_PROJ = {"type": "laea", "lon0": 15.0, "lat0": 52.0}
EU_SCALE = 1.0  # km
EU_BBOX = (-32.0, 24.0, 72.0, 76.0)
USSR = ["RUS", "UKR", "BLR", "MDA", "EST", "LVA", "LTU", "GEO", "ARM", "AZE", "KAZ", "UZB",
        "TKM", "KGZ", "TJK", "KAB"]
GDR_STATES = {"Mecklenburg-Vorpommern", "Brandenburg", "Berlin", "Sachsen", "Sachsen-Anhalt",
              "Thüringen"}


def build_europe():
    print("europe.json")
    a0 = adm0("ne_10m_admin_0_countries.geojson")
    fn = lambda x, y: geoproj.laea(x, y, lon0=EU_PROJ["lon0"], lat0=EU_PROJ["lat0"])  # noqa: E731
    clip = box(*EU_BBOX)
    P = lambda g: proj_geom(g.intersection(clip), fn, EU_SCALE)  # noqa: E731
    gdr_parts = [g for p, g in load("ne_10m_admin_1_states_provinces.geojson")
                 if p["adm0_a3"] == "DEU" and p["name"] in GDR_STATES]
    assert len(gdr_parts) == 6
    deu = a0["DEU"]
    gdr = clean(unary_union(gdr_parts), 0.002).intersection(deu)
    frg = deu.difference(gdr.buffer(0.0005))
    groups = {
        "SUN": USSR, "CSK": ["CZE", "SVK"],
        "YUG": ["SVN", "HRV", "BIH", "SRB", "KOS", "MNE", "MKD"],
        "CYP": ["CYP", "CYN", "CNM", "ESB", "WSB"], "ITA": ["ITA", "SMR", "VAT"],
        "FIN": ["FIN", "ALD"],
    }
    names = {"SUN": "SSSR", "CSK": "ČEHOSLOVAČKA", "YUG": "SFRJ", "FRG": "SR NEMAČKA",
             "GDR": "DR NEMAČKA"}
    used = {c for v in groups.values() for c in v} | {"DEU", "ATA"}
    ents = {k: unary_union([a0[c] for c in v if c in a0]) for k, v in groups.items()}
    ents["FRG"] = frg
    ents["GDR"] = gdr
    for c, g in a0.items():
        if c not in used and g.intersects(clip):
            ents[c] = g
    feats = []
    for code, g in sorted(ents.items()):
        if not g.intersects(clip):
            continue
        pg = clean(P(g), 0.2)
        p = enc_poly(pg, 1.2, 60.0)
        if not p:
            continue
        feats.append({"id": code, "name": names.get(code, code), "p": p,
                      "label": label_pt(pg, 5.0)})
    return {"meta": {"source": "Natural Earth 10m (public domain), built by gen/geo_build.py",
                     "proj": EU_PROJ, "scale": EU_SCALE, "unit": "1 km",
                     "note": "Cold-War entities: SUN=USSR, CSK, FRG/GDR split along the 1949-90 "
                             "inner-German border (Berlin wholly in GDR), YUG=SFRJ"},
            "features": feats}


# ================================================================= WORLD ===
WO_PROJ = {"type": "equal_earth", "lon0": 11.0}
WO_SCALE = 10000.0
PAPUA = {"Papua", "Papua Barat"}
SOUTH_YEMEN = {"Hadramawt", "Al Mahrah", "Lahij", "`Adan", "Abyan", "Shabwah", "Al Dali'"}
NAM1961 = [  # (label, [entity ids]) - the 25 full members of the Belgrade conference, 1-6 Sep 1961
    ("AFGANISTAN", "Afghanistan", ["AFG"]),
    ("ALŽIR", "Algeria (GPRA)", ["DZA"]),
    ("BURMA", "Burma", ["MMR"]),
    ("KAMBODŽA", "Cambodia", ["KHM"]),
    ("CEJLON", "Ceylon", ["LKA"]),
    ("KONGO", "Congo-Léopoldville", ["COD"]),
    ("KUBA", "Cuba", ["CUB"]),
    ("KIPAR", "Cyprus", ["CYP"]),
    ("ETIOPIJA", "Ethiopia", ["ETH"]),
    ("GANA", "Ghana", ["GHA"]),
    ("GVINEJA", "Guinea", ["GIN"]),
    ("INDIJA", "India", ["IND"]),
    ("INDONEZIJA", "Indonesia", ["IDN"]),
    ("IRAK", "Iraq", ["IRQ"]),
    ("LIBAN", "Lebanon", ["LBN"]),
    ("MALI", "Mali", ["MLI"]),
    ("MAROKO", "Morocco", ["MAR"]),
    ("NEPAL", "Nepal", ["NPL"]),
    ("SAUDIJSKA ARABIJA", "Saudi Arabia", ["SAU"]),
    ("SOMALIJA", "Somalia", ["SOM"]),
    ("SUDAN", "Sudan", ["SDN"]),
    ("TUNIS", "Tunisia", ["TUN"]),
    ("UAR", "United Arab Republic (Egypt + Syria)", ["EGY", "SYR"]),
    ("JEMEN", "Yemen (North)", ["YEN"]),
    ("JUGOSLAVIJA", "Yugoslavia", ["YUG"]),
]
OBSERVERS1961 = [("BOLIVIJA", "Bolivia", ["BOL"]), ("BRAZIL", "Brazil", ["BRA"]),
                 ("EKVADOR", "Ecuador", ["ECU"])]
WORLD_CITIES = [("Beograd", "Belgrade", 20.4569, 44.8176), ("Kairo", "Cairo", 31.2357, 30.0444),
                ("Nju Delhi", "New Delhi", 77.2090, 28.6139), ("Džakarta", "Jakarta", 106.8456, -6.2088),
                ("Akra", "Accra", -0.1870, 5.6037)]


def build_world():
    print("world.json")
    a0 = adm0("ne_50m_admin_0_countries.geojson")
    a1 = load("ne_10m_admin_1_states_provinces.geojson")
    lon0 = WO_PROJ["lon0"]
    fn = lambda x, y: geoproj.equal_earth(x, y, lon0=lon0)  # noqa: E731
    P = lambda g: proj_geom(split_seam(g, lon0), fn, WO_SCALE)  # noqa: E731
    papua = unary_union([g for p, g in a1 if p["adm0_a3"] == "IDN" and p["name"] in PAPUA])
    syem = unary_union([g for p, g in a1 if p["adm0_a3"] == "YEM" and p["name"] in SOUTH_YEMEN])
    a1_de = [g for p, g in a1 if p["adm0_a3"] == "DEU" and p["name"] in GDR_STATES]
    gdr = clean(unary_union(a1_de), 0.01).intersection(a0["DEU"])
    groups = {
        "SUN": [c for c in USSR if c in a0], "CSK": ["CZE", "SVK"],
        "YUG": ["SVN", "HRV", "BIH", "SRB", "KOS", "MNE", "MKD"],
        "ETH": ["ETH", "ERI"], "SDN": ["SDN", "SDS"], "SOM": ["SOM", "SOL"],
        "CYP": ["CYP", "CYN"], "IND": ["IND", "KAS"], "MAR": ["MAR"],
    }
    used = {c for v in groups.values() for c in v} | {"ATA", "IDN", "YEM", "DEU"}
    ents = {k: clean(unary_union([a0[c] for c in v]), 0.02) for k, v in groups.items()}
    idn = a0["IDN"]
    wng = idn.intersection(papua.buffer(0.08))
    ents["IDN"] = idn.difference(wng)
    ents["IWP"] = wng  # Netherlands New Guinea in 1961
    yem = a0["YEM"]
    ents["YES"] = yem.intersection(syem.buffer(0.02))  # Aden / South Arabia (British) in 1961
    ents["YEN"] = yem.difference(ents["YES"])
    ents["GDR"] = gdr
    ents["FRG"] = a0["DEU"].difference(gdr.buffer(0.005))
    for c, g in a0.items():
        if c not in used:
            ents[c] = g
    names = {p["ADM0_A3"]: p["NAME"] for p, g in load("ne_50m_admin_0_countries.geojson")}
    names.update({"SUN": "USSR", "CSK": "Czechoslovakia", "YUG": "Yugoslavia", "IWP": "Neth. New Guinea",
                  "YES": "Aden", "YEN": "Yemen", "GDR": "GDR", "FRG": "FRG"})
    feats = []
    for code, g in sorted(ents.items()):
        pg = P(g)
        p = enc_poly(pg, 14.0, 60.0)
        if not p:
            continue
        feats.append({"id": code, "name": names.get(code, code), "p": p, "label": label_pt(pg, 20.0)})
    cities = []
    for name, en, lon, lat in WORLD_CITIES:
        x, y = fn(lon, lat)
        cities.append({"name": name, "en": en, "lon": lon, "lat": lat,
                       "xy": [int(round(x * WO_SCALE)), int(round(y * WO_SCALE))]})
    return {"meta": {"source": "Natural Earth 50m (+10m admin-1 for 1961 splits), public domain",
                     "proj": WO_PROJ, "scale": WO_SCALE, "unit": "1e-4 Earth radii (Equal Earth)"},
            "features": feats,
            "nam1961": [{"name": n, "en": e, "ids": ids} for n, e, ids in NAM1961],
            "observers1961": [{"name": n, "en": e, "ids": ids} for n, e, ids in OBSERVERS1961],
            "cities": cities}


def write(name, obj):
    os.makedirs(DST, exist_ok=True)
    path = os.path.join(DST, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"  -> {os.path.relpath(path, ROOT)}  {os.path.getsize(path) / 1024:.0f} KB")


def main(argv):
    for f in FILES:
        fetch(f)
    todo = argv[1:] or ["yugo", "europe", "world"]
    if "yugo" in todo:
        write("yugo.json", build_yugo())
    if "europe" in todo:
        write("europe.json", build_europe())
    if "world" in todo:
        write("world.json", build_world())


if __name__ == "__main__":
    main(sys.argv)
