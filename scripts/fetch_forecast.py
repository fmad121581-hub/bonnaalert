"""Step 2: pull the 72-hour rainfall forecast for every ward.

Uses the free Open-Meteo forecast API (best-match blend of GFS, ICON and
others) at each ward's representative point. Writes:
  data/forecast.csv          ward_id, p24_max, p72_total, peak_time
  docs/data/hourly.json      hourly rain per ward, for the dashboard chart

    python scripts/fetch_forecast.py
    python scripts/fetch_forecast.py --demo     # synthetic storm, no internet
"""
import argparse
import json
import pathlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import config

CHUNK = 50  # locations per request


def fetch_real(wards):
    rows, hourly = [], {}
    for start in range(0, len(wards), CHUNK):
        part = wards.iloc[start:start + CHUNK]
        params = {
            "latitude": ",".join(map(str, part["lat"])),
            "longitude": ",".join(map(str, part["lon"])),
            "hourly": "precipitation",
            "forecast_hours": config.FORECAST_HOURS,
            "timezone": config.TIMEZONE,
        }
        r = requests.get(config.OPEN_METEO_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):  # a single location comes back as an object
            data = [data]
        for wid, loc in zip(part["ward_id"], data):
            times = loc["hourly"]["time"]
            rain = [x or 0.0 for x in loc["hourly"]["precipitation"]]
            hourly[wid] = rain
            rows.append(summarise(wid, times, rain))
        times_ref = times
    return pd.DataFrame(rows), times_ref, hourly


def fetch_demo(wards, seed=3):
    """A synthetic storm crossing the city, peaking about 30 h from now."""
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone(timedelta(hours=6))).replace(
        minute=0, second=0, microsecond=0)
    times = [(now + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")
             for h in range(config.FORECAST_HOURS)]
    h = np.arange(config.FORECAST_HOURS)
    rows, hourly = [], {}
    for _, w in wards.iterrows():
        # heavier toward the south-east of the demo grid
        scale = 1.0 + 1.2 * (w["lon"] - 90.34) / 0.14 * (23.88 - w["lat"]) / 0.18
        storm = 5 * scale * np.exp(-0.5 * ((h - 30) / 4.0) ** 2)
        rain = np.clip(storm + rng.gamma(0.3, 1.0, len(h)) * 0.6, 0, None).round(1)
        hourly[w["ward_id"]] = rain.tolist()
        rows.append(summarise(w["ward_id"], times, rain.tolist()))
    return pd.DataFrame(rows), times, hourly


def load_event_rain(path):
    """Read a compact event rain file: one row per forecast grid cell, wards point to a row."""
    d = json.loads(pathlib.Path(path).read_text())
    rows = [[int(x) / 10 for x in r.split()] for r in d["rows"].split("|")]
    idx = [int(i) for i in d["idx"].split()]
    start = datetime.fromisoformat(d["start"])
    times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(d["n"])]
    return times, [rows[i] for i in idx], d.get("source", "")


def fetch_replay(wards, path, scale_to=None):
    """Replay a past storm as if it were the forecast (wards in wards_static order).

    scale_to: multiply every ward's series by one factor so the city-median
    wettest-24 h total equals this many mm (e.g. a station-observed total
    when the gridded archive smooths a downpour away).
    """
    times, per_ward, _ = load_event_rain(path)
    if scale_to:
        p24 = [pd.Series(r).rolling(24, min_periods=1).sum().max() for r in per_ward]
        k = scale_to / float(np.median(p24))
        per_ward = [[round(v * k, 1) for v in r] for r in per_ward]
        print(f"Scaled rain by {k:.2f} so the city-median 24 h total is {scale_to} mm")
    if len(per_ward) != len(wards):
        raise SystemExit(f"Event file has {len(per_ward)} wards, ward table has {len(wards)}")
    rows, hourly = [], {}
    for wid, rain in zip(wards["ward_id"], per_ward):
        hourly[wid] = rain
        rows.append(summarise(wid, times, rain))
    return pd.DataFrame(rows), times, hourly


def summarise(wid, times, rain):
    s = pd.Series(rain, dtype=float)
    roll = s.rolling(24, min_periods=1).sum()
    i = int(s.idxmax())          # hour of heaviest rain
    return {
        "ward_id": wid,
        "p24_max": round(float(roll.max()), 1),
        "p72_total": round(float(s.sum()), 1),
        "peak_time": times[min(i, len(times) - 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--replay", help="event rain file, e.g. data/events/2023-09-21_rain_raw.json")
    ap.add_argument("--scale-to", type=float, help="with --replay: scale rain so the city-median 24 h total is this many mm")
    args = ap.parse_args()

    import geopandas as gpd
    wards = gpd.read_file(config.WARDS_STATIC)
    if args.replay:
        df, times, hourly = fetch_replay(wards, args.replay, args.scale_to)
        kind = json.loads(pathlib.Path(args.replay).read_text()).get("kind", "replay")
        source = "open-meteo" if kind == "forecast" else "replay:" + pathlib.Path(args.replay).name.split("_")[0]
        if args.scale_to:
            source += f":scaled-{args.scale_to:g}mm"
    elif args.demo:
        df, times, hourly = fetch_demo(wards)
        source = "demo"
    else:
        df, times, hourly = fetch_real(wards)
        source = "open-meteo"

    df["source"] = source
    df.to_csv(config.FORECAST_CSV, index=False)
    config.WEB_DATA.mkdir(parents=True, exist_ok=True)
    (config.WEB_DATA / "hourly.json").write_text(json.dumps(
        {"times": times, "rain_mm": hourly}, separators=(",", ":")))
    print(f"Forecast for {len(df)} wards -> {config.FORECAST_CSV}")
    print(df[["p24_max", "p72_total"]].describe().round(1).T)


if __name__ == "__main__":
    main()
