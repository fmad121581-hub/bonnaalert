# Event files

`2023-09-21_rain_raw.json`: hourly rain for 19 Sep 06:00 – 22 Sep 06:00 2023 (Asia/Dhaka) from the
Open-Meteo historical archive (ERA5-based best match), one row per forecast grid cell; `idx` maps each
ward in `data/wards_static.geojson` (same order) to its row. Values are tenths of a millimetre.
ERA5 at ~25 km smooths out convective downpours, so local station totals for this storm were likely higher.

`live_2026-10-19_raw.json`: the Open-Meteo 72 h forecast fetched on 19 Oct 2026 08:00 (dry: 0 mm everywhere).

## What the Sep 2023 Sentinel-1 labels showed (19 Oct 2026 check)

Ward flooded share from `delta_s1_sep22_minus_sep10.tif`, 188 wards with ≥ 50 pixels:

- Brightening rule (delta ≥ +3 dB): almost no relation to any ward trait (|rho| ≤ 0.12). v0 AUC 0.33.
- Darkening rule (delta ≤ −3 dB, open water): strong relation to low-lying land (rho +0.60) and
  negative with built-up share (−0.47) — it sees fringe inundation, not street waterlogging in the core.

Decision: do not train the trigger model on radar labels from this one event. Core-city labels need
DNCC/DSCC waterlogging hotspot lists or news reports; radar labels are kept for fringe wards.
