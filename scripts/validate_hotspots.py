"""Check ward scores against reported waterlogging hotspots.

A ward is a 'hotspot ward' if any reported waterlogging location
(data/validation/hotspots_geocoded.json) falls inside it. AUC tells how well a
score ranks hotspot wards above the others (0.5 = no better than chance).
Run on city-corporation wards only ("Ward No-..."), because the hotspot lists
cover the city corporations and comparing against rural unions would flatter
every score.

    python scripts/validate_hotspots.py
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score

import config
from features import build_features
from score import predict_v0


def main():
    st = gpd.read_file(config.WARDS_STATIC)
    hs = json.loads((config.DATA / "validation" / "hotspots_geocoded.json").read_text())
    pts = gpd.GeoDataFrame({"name": [p[0] for p in hs["points"]]},
                           geometry=[Point(p[2], p[1]) for p in hs["points"]], crs="EPSG:4326")
    hit = gpd.sjoin(pts, st[["ward_id", "geometry"]], predicate="within", how="left")
    hot_ids = set(hit["ward_id"].dropna())
    st["hotspot"] = st["ward_id"].isin(hot_ids).astype(int)
    city = st[st["name"].str.startswith("Ward No")].copy()

    rows = []
    for label, p24 in [("v0 at 113 mm / 24 h (Sep 2023 BMD total)", 113.0),
                       ("v0 at 60 mm / 24 h", 60.0)]:
        df = city.copy()
        df["p24"] = p24
        prob = predict_v0(build_features(df, st), st)[0]
        v = st["vuln_mean"]
        vn = ((city["vuln_mean"] - v.min()) / (v.max() - v.min())).fillna(.5)
        rows.append((label + ", probability", prob))
        rows.append((label + ", risk = prob x vulnerability", prob * (config.VULN_FLOOR + (1 - config.VULN_FLOOR) * vn)))
    rows += [("static hazard x exposure x vulnerability index", city["risk_static"]),
             ("curve number (imperviousness)", city["cn_mean"]),
             ("TWI (wetness)", city["twi_mean"]),
             ("lower elevation", -city["elev_mean"]),
             ("built-up share", city["built_share"]),
             ("vulnerability", city["vuln_mean"])]

    y = city["hotspot"].values
    out = pd.DataFrame([{"score": k, "auc": round(roc_auc_score(y, s), 3)} for k, s in rows])
    rng = np.random.default_rng(0)
    # bootstrap 95% interval for the headline score
    s0 = np.asarray(rows[1][1]); boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            boots.append(roc_auc_score(y[i], s0[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    unmatched = hit[hit["ward_id"].isna()]["name"].tolist()
    print(f"{len(pts)} hotspot points -> {len(hot_ids)} wards; city wards: {len(city)} "
          f"({int(y.sum())} hotspot); points outside the ward set: {unmatched}")
    print(out.to_string(index=False))
    print(f"Headline (v0 risk at 113 mm): AUC {out.auc[1]:.3f}, bootstrap 95% CI {lo:.2f}-{hi:.2f}")
    res = {"n_points": len(pts), "n_hot_wards": len(hot_ids), "n_city_wards": int(len(city)),
           "n_city_hot": int(y.sum()), "unmatched": unmatched,
           "auc": out.to_dict("records"), "headline_ci": [round(lo, 2), round(hi, 2)]}
    (config.DATA / "validation" / "hotspot_validation.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
