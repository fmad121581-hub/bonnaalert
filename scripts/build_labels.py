"""Turn one Sentinel-1 change raster into ward-level training labels.

For each ward: share of valid pixels flagged as flooded, plus the rainfall
that fell before the image. Rows are appended to data/training/events.csv.
Run once per storm (and at least once for a dry-season pair, which gives
the model its "no flood" examples).

    python scripts/build_labels.py --event 2023-09-21 \
        --delta path/to/delta_s1_sep22_minus_sep10.tif --rule ge:3

--rule says which pixels count as flooded: "ge:3" = delta >= +3 dB (what
the flood-risk validation used), "le:-3" = delta <= -3 dB (open-water
darkening). Check the sign of your own difference raster first.
--p24 sets the event's max 24 h rain (mm) for every ward; leave it out to
pull it per ward from the Open-Meteo ERA5 archive.
"""
import argparse
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterstats import zonal_stats

import config


def flooded_share(wards, raster, rule):
    op, val = rule.split(":")
    val = float(val)
    test = (lambda a: a >= val) if op == "ge" else (lambda a: a <= val)

    def share(a):
        v = a.compressed()
        v = v[np.isfinite(v)]
        return float(test(v).mean()) if v.size else np.nan

    import rasterio
    with rasterio.open(raster) as src:
        w = wards.to_crs(src.crs)
    z = zonal_stats(w, raster, stats=["count"], add_stats={"share": share})
    return np.array([s["share"] for s in z]), np.array([s["count"] for s in z])


def archive_p24(wards, event):
    d = date.fromisoformat(event)
    out = []
    for start in range(0, len(wards), 50):
        part = wards.iloc[start:start + 50]
        r = requests.get(config.OPEN_METEO_ARCHIVE_URL, params={
            "latitude": ",".join(map(str, part["lat"])),
            "longitude": ",".join(map(str, part["lon"])),
            "start_date": (d - timedelta(days=2)).isoformat(),
            "end_date": d.isoformat(),
            "daily": "precipitation_sum", "timezone": config.TIMEZONE},
            timeout=60)
        r.raise_for_status()
        data = r.json()
        data = data if isinstance(data, list) else [data]
        out += [max(x or 0 for x in loc["daily"]["precipitation_sum"]) for loc in data]
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="date of the post-storm image, YYYY-MM-DD")
    ap.add_argument("--delta", required=True, help="Sentinel-1 difference raster (dB)")
    ap.add_argument("--rule", default="ge:3")
    ap.add_argument("--p24", type=float, help="max 24 h rain for the event, mm")
    ap.add_argument("--rain-json", help="event rain file (per-ward hourly rain before the image)")
    ap.add_argument("--label-frac", type=float, default=0.10,
                    help="ward counts as flooded when this share of pixels is flagged")
    ap.add_argument("--min-pixels", type=int, default=50)
    args = ap.parse_args()

    wards = gpd.read_file(config.WARDS_STATIC)
    frac, count = flooded_share(wards, args.delta, args.rule)
    if args.rain_json:
        from fetch_forecast import load_event_rain
        _, per_ward, _ = load_event_rain(args.rain_json)
        p24 = np.array([pd.Series(r).rolling(24, min_periods=1).sum().max() for r in per_ward])
    elif args.p24 is not None:
        p24 = np.full(len(wards), args.p24)
    else:
        p24 = archive_p24(wards, args.event)

    rows = pd.DataFrame({
        "event_id": args.event, "ward_id": wards["ward_id"], "p24": p24,
        "flooded_frac": np.round(frac, 4), "n_pixels": count,
        "label": (frac >= args.label_frac).astype(int)})
    rows = rows[rows["n_pixels"] >= args.min_pixels]

    config.TRAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    if config.TRAINING_CSV.exists():
        old = pd.read_csv(config.TRAINING_CSV)
        old = old[old["event_id"] != args.event]   # re-running an event replaces it
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(config.TRAINING_CSV, index=False)
    ev = rows[rows["event_id"] == args.event]
    print(f"Event {args.event}: {len(ev)} wards, {int(ev['label'].sum())} labelled flooded, "
          f"median flagged share {ev['flooded_frac'].median():.3f}")
    print(f"Training table now has {rows['event_id'].nunique()} events, {len(rows)} rows")


if __name__ == "__main__":
    main()
