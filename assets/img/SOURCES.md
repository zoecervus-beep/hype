# SFRJ hype edit - archive photo sources

66 real photographs, all opened and checked by eye (see `contact_sheet.jpg`). Every file is an RGB JPEG (q92, long side <= 2400 px).
Full per-image metadata (source URL, page URL, licence, author, size, framing notes) is in `manifest.json`.

**Per category:** arch 8, funeral 4, industry 6, landscape 5, nam 6, partisans 11, spomenik 15, sport 4, tito 7, brigades 0.

## Where the images came from

The container's proxy blocks Wikimedia, Flickr, archive.org, museum and library sites, so every file was pulled from an allowed host (S3 or GitHub) that holds a copy of a Commons or Flickr original:

| source | count | how | licence |
|---|---|---|---|
| **Google Landmarks Dataset v2** (`s3.amazonaws.com/google-landmark`) | 49 | Wikimedia Commons images, downscaled to 800 px, stored in 500 x 1 GB tars. We looked up IDs in `metadata/train_attribution.csv` (Commons title, author, licence), then pulled single files out of the tars with HTTP range requests (binary search on the tar headers) | per image, all CC BY / CC BY-SA / GFDL. Includes about 30 official photos from the **Museum of Yugoslavia** (Photo service of the President's Cabinet), CC BY-SA 3.0 RS |
| **Open Images v4** (`s3.amazonaws.com/open-images-dataset`) | 6 | Flickr CC BY 2.0 photos, 1024 px copies. We grepped the titles in `image_ids_and_rotation.csv`. Only the ~1.9M "boxable" subset is actually on S3 | CC BY 2.0 (credit the Flickr author) |
| **GitHub `mudroljub/avantura-1941`** (raw.githubusercontent.com) | 10 | WWII 1941-44 Partisan photos (the Znaci.org / Commons photo corpus) kept in a game repo | **Presumed public domain** (WWII Yugoslav photos, usually tagged PD-Serbia/PD-Yugoslavia on Commons). The repo itself has no licence and gives no provenance, so check before commercial use |
| **GitHub `polarorb/untitled-grand-strategy`** | 1 | `assets/leaders/YUG.jpg` = Commons *File:Josip Broz Tito uniform portrait.jpg* (960x1335) | Public domain according to Commons and the repo's CREDITS.md |

**Attribution / share-alike:** because most of the set is CC BY or CC BY-SA, the finished video needs a credits card, for example "Photos: Wikimedia Commons contributors (CC BY-SA), Museum of Yugoslavia (CC BY-SA 3.0 RS), Flickr: Brian Eager, Charlie, Nick, tomislav medak, Luigi Torreggiani (CC BY 2.0)". Take the exact names from `manifest.json` `author`. The CC BY-SA images make stylised derivatives share-alike.

**Sensitive items:** `partisans_stjepan_filipovic_1942.jpg` shows the moment before an execution (it is where the SMRT FAŠIZMU - SLOBODA NARODU slogan comes from). `spomenik_jasenovac_stone_flower.jpg` is a memorial at a death-camp site. Use both with care.

## Hero picks
* Tito: `tito_marshal_uniform_portrait.jpg` (frontal, 960x1335, clean B&W)
* Spomenik: `spomenik_tjentiste_sutjeska.jpg`, `spomenik_podgaric.jpg`, `spomenik_kozara_fog.jpg`, `spomenik_makedonium_krusevo.jpg`, `spomenik_bubanj_fists.jpg`
* Partisans: `partisans_first_proletarian_brigade_foca_1942.jpg` (red-star flag), `partisans_column_mountain_road_1941.jpg`
* Everyday socialism: `arch_novi_beograd_1978_tito_billboard.jpg` (workers putting up a Tito billboard, 1978)
* Sport: `sport_sarajevo84_opening_ceremony.jpg` (actual 1984 ceremony, letterbox 800x254)

## What we could NOT find (not reachable through the proxy, or not in the reachable datasets)
* **Youth work brigades** (Brčko-Banovići 1946, Šamac-Sarajevo 1947, Autoput): none found.
* **1961 Belgrade NAM summit**, and the **Brioni 1956 Tito-Nehru-Nasser** photo: the only copy we found (`ZoeLeBlanc/zoeleblanc.github.io` Slide06.jpg) carries an AFP credit, so it was excluded. NAM is covered by Museum of Yugoslavia state-visit photos (Nehru visit, Sékou Touré, Ghana, Algiers, Tanzania, Brezhnev).
* **US presidential photos** (JFK 1963, Nixon 1970/71, Carter 1978): the NARA catalog metadata is public (`s3://nara-national-archives-catalog`), but every digital object in `s3://NARAprodstorage` returns S3 AccessDenied.
* **Tito's funeral 1980 (the event itself)**, **liberation of Belgrade 1944**, a **WWII-era Tito portrait**, **women Partisans** (only tiny game sprites), **Yugoslav basketball** 1970/78/90, **Galeb**, **red passport**, **Hotel Jugoslavija**, **Skopje / Kenzo Tange**, **Mamutica**, **Split 3**, **Ilirska Bistrica**.

## Leads for later
* GLDv2 also holds more Museum of Yugoslavia photos (~180 in category `Museum_of_Yugoslavia`) and more spomenik views: Šumarice, Vraca (1985 photo), Kosmaj, Garavice, Iriški venac, the Partisan cemetery in Mostar, Tito's Cave on Vis. The IDs are in the GLDv2 attribution CSV, so more can be pulled the same way.
* Open Images has more Yugoslav-era Flickr material by tomislav medak (Museum of Yugoslavia exhibits, "Sremski front" map) and Zastava 101.

## Blocked-host findings (probed with curl, Sept 2026)
* **Reachable:** raw.githubusercontent.com, github.com (git clone/ls-remote of public repos works; api.github.com only for session repos), media.githubusercontent.com, s3.amazonaws.com (public buckets: google-landmark, open-images-dataset, nara-national-archives-catalog, multimedia-commons, smithsonian-open-access), storage.googleapis.com (openimages metadata), gitlab.com, bitbucket.org, npm, pypi, maven, rubygems, archive.ubuntu.com, repo.anaconda.com.
* **Reachable but no data:** `s3://NARAprodstorage` objects give S3 AccessDenied (403). `multimedia-commons` (YFCC100M) images are sparse and ~500 px, and the metadata is a 65 GB SQLite file.
* **Blocked (CONNECT 403):** all Wikimedia / Wikipedia, archive.org, loc.gov, Flickr (live/farm/c*.staticflickr), catalog.archives.gov, archives.gov, JFK/Nixon/Carter/Truman/Eisenhower/Ford/LBJ/Reagan libraries, Smithsonian (si.edu, ids.si.edu), Europeana, Nationaal Archief, Bundesarchiv, Deutsche Fotothek, ETH e-pics, Polish NAC, IWM, USHMM, NYPL, Gallica, Rijksmuseum, V&A, Met images, NGA, Cleveland, Art Institute, Fortepan, Muzej Jugoslavije, znaci.org, spomenikdatabase.org, digitalna.nb.rs, Kiwix mirrors, dumps.wikimedia.org, HuggingFace, Kaggle, Zenodo, figshare, imgur, pinterest/twitter/reddit CDNs, Google Drive/Dropbox, codeberg, gitee, sourcehut, *.github.io, jsdelivr/unpkg/cdnjs, user-images/private-user-images.githubusercontent.com.

## File list
| file | category | year | licence | author |
|---|---|---|---|---|
| `tito_marshal_uniform_portrait.jpg` | tito | ca. 1961 | Public domain (as tagged on Wikimedia Commons; credited PD i | unknown (official portrait) |
| `tito_57th_birthday_1949.jpg` | tito | 1949 | CC BY-SA 3.0 | Photo service of the Cabinet of the Pres |
| `tito_83rd_birthday_1975.jpg` | tito | 1975 | CC BY-SA 3.0 rs | Photo service of the Cabinet of the Pres |
| `tito_beli_dvor_park_walk.jpg` | tito | 1950s-60s | CC BY-SA 3.0 rs | Photo service of the Cabinet of the Pres |
| `tito_1941_1976_ceremony.jpg` | tito | 1976 | CC BY-SA 3.0 rs | Photo service of the Cabinet of the Pres |
| `tito_knin_crowd_visit.jpg` | tito | 1940s-50s | CC BY-SA 3.0 | Photo service of the Cabinet of the Pres |
| `tito_eleanor_roosevelt_brioni.jpg` | tito | 1953 | CC BY-SA 3.0 rs | Photo service of the Cabinet of the Pres |
| `partisans_first_proletarian_brigade_foca_1942.jpg` | partisans | 1942 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_column_mountain_road_1941.jpg` | partisans | 1941 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_takovo_battalion_march_1941.jpg` | partisans | 1941 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_valjevo_group_rifles_1941.jpg` | partisans | 1941 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_young_fighters_rogatica_1941.jpg` | partisans | 1941 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_suvobor_snow_forest_1942.jpg` | partisans | 1942 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_cacak_detachment_group_1941.jpg` | partisans | 1941 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_cacak_rally_speech_1941.jpg` | partisans | 1941 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_stjepan_filipovic_1942.jpg` | partisans | 1942 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_airforce_osvetnik_1944.jpg` | partisans | 1944 | Public domain (presumed): WWII-era Yugoslav photograph, auth | unknown (Partisan / period photographer) |
| `partisans_supreme_hq_table.jpg` | partisans | 1944 (print); photo of exhibit | CC BY 2.0 | Yugoslav Armies Museum |
| `nam_nehru_visit_yugoslavia.jpg` | nam | 1950s | CC BY-SA 3.0 rs | Фото-служба Кабинета председника Републи |
| `nam_tito_arrives_algiers.jpg` | nam | 1960s-70s | CC BY-SA 3.0 | Photo service of the Cabinet of the Pres |
| `nam_tito_tanzania_trip.jpg` | nam | 1970 | CC BY-SA 3.0 | Photo service of the Cabinet of the Pres |
| `nam_sekou_toure_visit.jpg` | nam | 1960s | CC BY-SA 3.0 rs | Photo service of the Cabinet of the Pres |
| `nam_yugoslav_ghanaian_talks.jpg` | nam | 1960s | CC BY-SA 3.0 | Photo service of the Cabinet of the Pres |
| `nam_brezhnev_visit.jpg` | nam | 1970s | CC BY-SA 3.0 rs | Фото-служба Кабинета председника Републи |
| `industry_tito_visits_shipyard.jpg` | industry | 1950s-60s | CC BY-SA 3.0 | Photo service of the Cabinet of the Pres |
| `industry_zagreb_fair_opening.jpg` | industry | 1950s | CC BY-SA 3.0 | Photo service of the Cabinet of the Pres |
| `industry_zastava750_fico_white.jpg` | industry | photo 2011 (car 1960s-80s) | CC BY 2.0 | Brian Eager |
| `industry_zastava750_fico_orange.jpg` | industry | photo 2011 (car 1960s-80s) | CC BY 2.0 | Brian Eager |
| `industry_zastava750_fico_red_rear.jpg` | industry | photo 2012 (car 1960s-80s) | CC BY 2.0 | Luigi Torreggiani |
| `industry_yugo45.jpg` | industry | photo 2013 (car 1980s) | CC BY 2.0 | Charlie |
| `spomenik_tjentiste_sutjeska.jpg` | spomenik | 1971 (photo 2010s) | CC BY-SA 4.0 | Kathinka. |
| `spomenik_tjentiste_valley.jpg` | spomenik | 1971 (photo 2010s) | CC BY-SA 4.0 | Kathinka. |
| `spomenik_podgaric.jpg` | spomenik | 1967 (photo 2012) | CC BY 2.0 | tomislav medak |
| `spomenik_kozara_fog.jpg` | spomenik | 1972 (photo 2010s) | CC BY-SA 4.0 | Indir Photography |
| `spomenik_kozara_visitor.jpg` | spomenik | 1972 (photo 2010s) | CC BY-SA 4.0 | Tamara Tica |
| `spomenik_jasenovac_stone_flower.jpg` | spomenik | 1966 (photo 2010s) | CC BY-SA 4.0 | Bobonajbolji |
| `spomenik_kadinjaca.jpg` | spomenik | 1979 (photo 2010s) | CC BY-SA 4.0 | Bokačo |
| `spomenik_makedonium_krusevo.jpg` | spomenik | 1974 (photo 2010s) | CC BY-SA 3.0 | Kristijan006Kris |
| `spomenik_petrova_gora.jpg` | spomenik | 1981 (photo 2010) | CC BY-SA 3.0 | Sandor Bordas |
| `spomenik_petrova_gora_construction.jpg` | spomenik | ca. 1970s-1981 | CC BY-SA 3.0 | DobarSkroz |
| `spomenik_bubanj_fists.jpg` | spomenik | 1963 (photo 2000s) | CC BY-SA 3.0 | This image was made by White Writer Plea |
| `spomenik_bubanj_1966.jpg` | spomenik | 1966 | CC BY-SA 3.0 | M. Sokolović |
| `spomenik_slobodiste_krusevac.jpg` | spomenik | 1965 (photo 2000s) | CC BY-SA 3.0 | Dragan |
| `spomenik_tito_visits_kumanovo_nob_monument.jpg` | spomenik | 1960s-70s | CC BY-SA 3.0 | UnknownUnknown author |
| `spomenik_tito_visits_partisan_cemetery_mostar.jpg` | spomenik | 1965-70s | CC BY-SA 3.0 | UnknownUnknown author |
| `arch_genex_tower_west_gate.jpg` | arch | 1977 (photo 2010) | CC BY-SA 3.0 | User:Михајло Анђелковић |
| `arch_genex_tower_looking_up.jpg` | arch | 1977 (photo 2010s) | CC BY-SA 3.0 | Lawrence Jesterton |
| `arch_avala_tv_tower.jpg` | arch | 1965 (rebuilt 2010; photo 2010s) | CC BY-SA 2.0 | {{{1}}} |
| `arch_usce_tower_sunset.jpg` | arch | 1964 (photo 2000s) | GFDL | Amazonite at English Wikipedia |
| `arch_palace_of_federation_siv.jpg` | arch | 1961 (photo 2010s) | CC BY-SA 3.0 | Bjoertvedt |
| `arch_sava_centar_1978.jpg` | arch | 1978 | CC BY 2.0 | nicksarebi |
| `arch_beogradjanka.jpg` | arch | 1974 (photo 2000s) | CC BY 3.0 | Mister No |
| `arch_novi_beograd_1978_tito_billboard.jpg` | arch | 1978 | CC BY 2.0 | Nick |
| `funeral_tito_grave_1892_1980.jpg` | funeral | 1980 (photo 2000s) | CC BY-SA 3.0 | Ferran Cornellà |
| `funeral_book_of_condolences.jpg` | funeral | 1980 (photo 2000s) | CC BY-SA 4.0 | Josip Broz Tito_knjige zalosti |
| `funeral_relay_of_youth_batons.jpg` | funeral | 1945-1987 (photo 2000s) | CC BY-SA 4.0 | Ahill34 |
| `funeral_relay_batons_gallery.jpg` | funeral | photo 2010s | CC BY-SA 4.0 | Марко Станојевић |
| `sport_sarajevo84_opening_ceremony.jpg` | sport | 1984 | CC BY-SA 4.0 | BiHVolim |
| `sport_sarajevo84_bobsled_track.jpg` | sport | 1984 (photo 2010s) | CC BY-SA 3.0 | spacebirdy (also known as geimfyglið (:> |
| `sport_igman_olympic_ski_jumps.jpg` | sport | 1984 (photo 2010s) | CC BY-SA 3.0 | Julian Nyča |
| `sport_sarajevo84_olympic_symbol.jpg` | sport | 1984 (photo 2010s) | CC BY-SA 3.0 | Hedwig KlawuttkeHedwig Klawuttke (german |
| `landscape_mostar_old_bridge_1979.jpg` | landscape | 1979 | CC BY-SA 3.0 | Zoran Kurelić Rabko |
| `landscape_mostar_old_bridge.jpg` | landscape | photo 2010s | CC BY 2.0 | Jocelyn Erskine-Kellie from London, UK |
| `landscape_bled_foggy_sunrise.jpg` | landscape | photo 2010s | CC BY 4.0 | Dreamy Pixel |
| `landscape_ohrid_lake_sunset.jpg` | landscape | photo 2010s | CC BY-SA 3.0 | Karadimce |
| `landscape_dubrovnik_walls.jpg` | landscape | photo 2010s | CC BY-SA 4.0 | Akampfer |
