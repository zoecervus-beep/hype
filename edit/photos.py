"""Photo slots used by the edit -> files in assets/img.

Slots are named "<category>_<letter>" (partisans_a, partisans_b, ...). Explicit picks in
PICKS win; otherwise the n-th image of that category in the archive manifest is used,
cycling through the category (or a fallback category) if it runs short.
"""
from engine import media

# explicit curation (slot -> file stem in assets/img)
PICKS = {}

FALLBACK = {
    "tito": ["nam", "partisans"],
    "partisans": ["tito", "brigades"],
    "liberation": ["partisans"],
    "brigade": ["brigades", "industry", "partisans"],
    "crowd": ["brigades", "funeral", "partisans"],
    "industry": ["brigades", "arch"],
    "school": ["brigades", "industry", "misc"],
    "nam": ["tito"],
    "funeral": ["tito", "nam"],
    "spomenik": ["arch"],
    "arch": ["spomenik", "landscape"],
    "sport": ["landscape", "misc"],
}

ALIAS = {"brigade": "brigades"}


class _Slots(dict):
    def __missing__(self, key):
        if key in PICKS:
            return PICKS[key]
        man = media.manifest()
        cat, _, idx = key.rpartition("_")
        if not cat:
            cat, idx = key, "a"
        n = ord(idx[0]) - ord("a") if idx and idx[0].isalpha() and len(idx) == 1 else 0
        if idx in ("hero", "stern"):
            n = {"hero": 0, "stern": 1}[idx]
        cats = [ALIAS.get(cat, cat)] + FALLBACK.get(cat, [])
        pool = []
        for ct in cats:
            pool += sorted(k for k, v in man.items() if v.get("category") == ct)
        if not pool:
            return key  # renders as labelled placeholder
        return pool[n % len(pool)]


PH = _Slots()
