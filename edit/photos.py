"""Photo slots used by the edit -> files in assets/img.

PICKS is the curated mapping. Unknown slots fall back to the n-th image of a category
("<category>_<letter>") from the archive manifest, or render as a labelled placeholder.
"""
from engine import media

PICKS = {
    # intro
    "partisans_a": "partisans_column_mountain_road_1941",
    "partisans_b": "partisans_first_proletarian_brigade_foca_1942",
    "partisans_c": "partisans_valjevo_group_rifles_1941",
    "partisans_d": "partisans_suvobor_snow_forest_1942",
    # countdown
    "tito_hero": "tito_marshal_uniform_portrait",
    "tito_stern": "tito_marshal_uniform_portrait",
    "crowd": "tito_knin_crowd_visit",
    # liberation
    "partisans_e": "partisans_cacak_detachment_group_1941",
    "partisans_f": "partisans_takovo_battalion_march_1941",
    "partisans_g": "partisans_airforce_osvetnik_1944",
    "liberation": "partisans_young_fighters_rogatica_1941",
    "liberation_b": "partisans_cacak_rally_speech_1941",
    # work / self-management
    "industry_a": "industry_tito_visits_shipyard",
    "industry_b": "arch_novi_beograd_1978_tito_billboard",
    "industry_c": "industry_zagreb_fair_opening",
    "construction": "spomenik_petrova_gora_construction",
    "sava": "arch_sava_centar_1978",
    "ceremony": "tito_1941_1976_ceremony",
    "crowd_b": "landscape_mostar_old_bridge_1979",
    # non-aligned
    "nam_a": "nam_nehru_visit_yugoslavia",
    "nam_b": "nam_tito_arrives_algiers",
    "nam_c": "nam_yugoslav_ghanaian_talks",
    "nam_d": "nam_tito_tanzania_trip",
    # break
    "spomenik_a": "spomenik_tjentiste_sutjeska",
    "spomenik_fog": "spomenik_kozara_fog",
    "funeral": "funeral_tito_grave_1892_1980",
    "condolences": "funeral_book_of_condolences",
    "batons": "funeral_relay_of_youth_batons",
    # drop B
    "sport_a": "sport_sarajevo84_opening_ceremony",
    "sport_bob": "sport_sarajevo84_bobsled_track",
    "sport_jumps": "sport_igman_olympic_ski_jumps",
    "arch_a": "arch_genex_tower_looking_up",
    "arch_b": "arch_genex_tower_west_gate",
    "arch_c": "arch_palace_of_federation_siv",
    "arch_d": "arch_avala_tv_tower",
    "arch_e": "arch_usce_tower_sunset",
    "arch_f": "arch_beogradjanka",
    "spomenik_b": "spomenik_bubanj_fists",
    "spomenik_c": "spomenik_podgaric",
    "spomenik_d": "spomenik_makedonium_krusevo",
    "spomenik_e": "spomenik_kadinjaca",
    "spomenik_f": "spomenik_petrova_gora",
    "spomenik_g": "spomenik_slobodiste_krusevac",
    "spomenik_h": "spomenik_kozara_visitor",
    "fico_a": "industry_zastava750_fico_red_rear",
    "fico_b": "industry_zastava750_fico_orange",
    "fico_c": "industry_zastava750_fico_white",
    "land_a": "landscape_mostar_old_bridge",
    "land_b": "landscape_bled_foggy_sunrise",
    "land_c": "landscape_dubrovnik_walls",
    "land_d": "landscape_ohrid_lake_sunset",
    "bw_fists": "spomenik_bubanj_1966",
}

FALLBACK = {
    "tito": ["nam", "partisans"],
    "partisans": ["tito"],
    "industry": ["arch"],
    "nam": ["tito"],
    "funeral": ["tito", "nam"],
    "spomenik": ["arch"],
    "arch": ["spomenik", "landscape"],
    "sport": ["landscape"],
}


class _Slots(dict):
    def __missing__(self, key):
        if key in PICKS:
            return PICKS[key]
        man = media.manifest()
        if key in man:
            return key
        cat, _, idx = key.rpartition("_")
        if not cat:
            cat, idx = key, "a"
        n = ord(idx[0]) - ord("a") if len(idx) == 1 and idx.isalpha() else 0
        pool = []
        for ct in [cat] + FALLBACK.get(cat, []):
            pool += sorted(k for k, v in man.items() if v.get("category") == ct)
        return pool[n % len(pool)] if pool else key


PH = _Slots()
