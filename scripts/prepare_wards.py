"""Step 1: build the static ward feature table.

Reads the flood-risk-mapping-dhaka outputs (ward_risk.shp plus the CN, TWI,
DEM and land-cover rasters), computes one row of features per ward and writes
data/wards_static.geojson in WGS84.

    python scripts/prepare_wards.py            # real data from the laptop
    python scripts/prepare_wards.py --demo     # synthetic wards for testing
"""
import argparse
import json
import sys

import numpy as np

import config


def _ward_name(gdf):
    for c in ["NAME_4", "NAME_3", "name", "NAME"]:
        if c in gdf.columns:
            return gdf[c].astype(str)
    return gdf.index.astype(str)


def build_real():
    import geopandas as gpd
    import rasterio
    from rasterstats import zonal_stats

    missing = [k for k, p in config.INPUTS.items() if not p.exists()]
    if missing:
        sys.exit("Missing inputs: " + ", ".join(
            f"{k} ({config.INPUTS[k]})" for k in missing)
            + "\nSet FLOOD_RISK_DIR or edit scripts/config.py.")

    wards = gpd.read_file(config.INPUTS["wards"])
    with rasterio.open(config.INPUTS["dem"]) as src:
        crs = src.crs
    wards = wards.to_crs(crs)
    wards = wards[wards.geometry.notna() & ~wards.geometry.is_empty].copy()
    wards["geometry"] = wards.geometry.buffer(0)
    wards = wards.reset_index(drop=True)

    def zonal(key, func):
        """func gets the ward's valid pixel values (finite, not nodata)."""
        path = str(config.INPUTS[key])
        with rasterio.open(path) as src:
            nod = src.nodata

        def stat(a):
            v = a.compressed().astype("float64")
            v = v[np.isfinite(v)]
            if nod is not None:
                v = v[v != nod]
            return float(func(v)) if v.size else np.nan

        z = zonal_stats(wards, path, stats=["count"], add_stats={"v": stat},
                        nodata=nod, all_touched=False)
        return np.array([s["v"] for s in z])

    out = gpd.GeoDataFrame({
        "ward_id": [f"W{i:03d}" for i in range(len(wards))],
        "name": _ward_name(wards).values,
        "pop_total": wards.get("pop_total", 0),
        "risk_static": wards.get("risk_mean", np.nan),
        "vuln_mean": wards.get("vuln_mean", np.nan),
        "exp_mean": wards.get("exp_mean", np.nan),
        "haz_mean": wards.get("haz_mean", np.nan),
    }, geometry=wards.geometry, crs=crs)

    out["cn_mean"] = zonal("cn", np.mean)
    out["twi_mean"] = zonal("twi", np.mean)
    out["elev_mean"] = zonal("dem", np.median)
    out["low_elev_share"] = zonal("dem", lambda v: np.mean(v <= config.LOW_ELEV_M))
    out["built_share"] = zonal("lulc", lambda v: np.mean(v == config.BUILT_CLASS))
    if "grid_cov" in wards.columns:
        out["grid_cov"] = wards["grid_cov"].values

    # same ward name in two thanas: add the thana/upazila to tell them apart
    if "NAME_3" in wards.columns:
        dup = out["name"].duplicated(keep=False)
        out.loc[dup, "name"] = out.loc[dup, "name"] + ", " + wards.loc[dup, "NAME_3"].astype(str)

    # wards outside the analysis grid have no features: leave them out
    keep = out["cn_mean"].notna() & out["vuln_mean"].notna()
    print(f"Dropped {int((~keep).sum())} wards with no raster coverage")
    out = out[keep].reset_index(drop=True)
    out["ward_id"] = [f"W{i:03d}" for i in range(len(out))]
    return out


def build_demo(n_side=8, seed=7):
    """Synthetic grid of wards over central Dhaka, for testing only."""
    import geopandas as gpd
    from shapely.geometry import box

    rng = np.random.default_rng(seed)
    lon0, lon1, lat0, lat1 = 90.34, 90.48, 23.70, 23.88
    dx, dy = (lon1 - lon0) / n_side, (lat1 - lat0) / n_side
    rows = []
    for i in range(n_side):
        for j in range(n_side):
            # lower and wetter toward the south-east, like the Buriganga side
            low = (i + (n_side - j)) / (2 * n_side)
            rows.append({
                "ward_id": f"W{len(rows):03d}",
                "name": f"Demo ward {len(rows) + 1}",
                "pop_total": int(rng.integers(20_000, 120_000)),
                "risk_static": float(np.clip(0.3 + 0.6 * low + rng.normal(0, .08), 0, 1)),
                "vuln_mean": float(np.clip(0.3 + 0.5 * low + rng.normal(0, .1), 0, 1)),
                "exp_mean": float(rng.uniform(0.2, 1.0)),
                "haz_mean": float(np.clip(2 + 2 * low + rng.normal(0, .3), 1, 4)),
                "cn_mean": float(np.clip(80 + 15 * rng.random(), 60, 98)),
                "twi_mean": float(6 + 6 * low + rng.normal(0, .8)),
                "elev_mean": float(9 - 6 * low + rng.normal(0, .5)),
                "low_elev_share": float(np.clip(low * 1.2 + rng.normal(0, .1), 0, 1)),
                "built_share": float(np.clip(0.55 + 0.35 * rng.random(), 0, 1)),
                "geometry": box(lon0 + j * dx, lat0 + i * dy,
                                lon0 + (j + 1) * dx, lat0 + (i + 1) * dy),
            })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic wards")
    args = ap.parse_args()

    gdf = build_demo() if args.demo else build_real()
    gdf = gdf.to_crs("EPSG:4326")
    cent = gdf.geometry.representative_point()
    gdf["lat"], gdf["lon"] = cent.y.round(5), cent.x.round(5)
    gdf["geometry"] = gdf.geometry.simplify(0.0002, preserve_topology=True)
    gdf = gdf.round(4)

    config.DATA.mkdir(parents=True, exist_ok=True)
    gdf.to_file(config.WARDS_STATIC, driver="GeoJSON")
    meta = {"source": "demo" if args.demo else str(config.FLOOD_RISK_DIR),
            "n_wards": len(gdf)}
    (config.DATA / "wards_static_meta.json").write_text(json.dumps(meta))
    print(f"Wrote {len(gdf)} wards -> {config.WARDS_STATIC}")
    print(gdf.drop(columns="geometry").describe().round(2).T[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
