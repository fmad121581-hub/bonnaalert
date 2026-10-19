"""Step 3: score every ward and assign an alert level.

probability  = event model (trained XGBoost if models/trigger_xgb.json exists,
               otherwise the v0 expert-prior logistic model in config.py)
risk         = probability x vulnerability weight
drivers      = per-ward feature contributions (SHAP values for XGBoost,
               logit contributions for v0), so every alert can say why

Writes docs/data/wards_scored.geojson and docs/data/summary.json.
"""
import json
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd

import config
from features import FEATURES, build_features


def predict_v0(X, ref):
    c = config.V0_COEF
    baseline = {"log_runoff": 0.0, "twi_z": 0.0,
                "low_elev_share": float(ref["low_elev_share"].median()),
                "built_share": float(ref["built_share"].median())}
    logit = c["intercept"] + sum(c[f] * X[f] for f in FEATURES)
    prob = 1 / (1 + np.exp(-logit))
    contrib = pd.DataFrame({f: c[f] * (X[f] - baseline[f]) for f in FEATURES})
    return prob, contrib, "v0 expert-prior logistic (not yet calibrated)"


def predict_xgb(X):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(config.MODEL_PATH))
    d = xgb.DMatrix(X[FEATURES], feature_names=FEATURES)
    prob = booster.predict(d)
    shap = booster.predict(d, pred_contribs=True)[:, :len(FEATURES)]
    return prob, pd.DataFrame(shap, columns=FEATURES, index=X.index), \
        "XGBoost trained on Sentinel-1 flood events"


def level_for(risk):
    for lo, key, label, colour in config.LEVELS:
        if risk >= lo:
            return key, label, colour
    return config.LEVELS[-1][1:]


def main():
    static = gpd.read_file(config.WARDS_STATIC)
    fc = pd.read_csv(config.FORECAST_CSV)
    g = static.merge(fc, on="ward_id", how="left")
    g["p24"] = g["p24_max"]

    X = build_features(g, static)
    if config.MODEL_PATH.exists():
        prob, contrib, model_name = predict_xgb(X)
    else:
        prob, contrib, model_name = predict_v0(X, static)

    v = g["vuln_mean"].astype(float)
    vuln_norm = ((v - v.min()) / (v.max() - v.min())).fillna(0.5)
    weight = config.VULN_FLOOR + (1 - config.VULN_FLOOR) * vuln_norm

    g["prob"] = np.round(prob, 3)
    g["vuln_norm"] = vuln_norm.round(3)
    g["risk"] = np.round(prob * weight, 3)
    g["runoff_mm"] = X["runoff_mm"].round(1)
    lv = g["risk"].apply(level_for)
    g["level"] = [x[0] for x in lv]
    g["level_bn"] = [x[1] for x in lv]
    g["color"] = [x[2] for x in lv]

    # top drivers: positive contributions, largest first; vulnerability added
    # when the ward sits in the top third
    drivers = []
    for i in g.index:
        row = contrib.loc[i].sort_values(ascending=False)
        keys = [k for k, val in row.items() if val > 0.15][:3]
        if vuln_norm[i] >= 0.66:
            keys.append("vulnerability")
        drivers.append(keys)
    g["drivers"] = [json.dumps(d) for d in drivers]
    g["contrib"] = [json.dumps({k: round(float(v), 3) for k, v in contrib.loc[i].items()})
                    for i in g.index]

    g = g.sort_values("risk", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    config.WEB_DATA.mkdir(parents=True, exist_ok=True)
    keep = ["ward_id", "name", "rank", "pop_total", "p24_max", "p72_total",
            "peak_time", "runoff_mm", "prob", "vuln_norm", "risk", "level",
            "level_bn", "color", "drivers", "contrib", "risk_static", "geometry"]
    g[keep].to_file(config.WARDS_SCORED, driver="GeoJSON")

    counts = g["level"].value_counts().to_dict()
    at_risk = g[g["level"].isin(["orange", "red"])]
    summary = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "model": model_name,
        "forecast_source": str(fc["source"].iloc[0]) if "source" in fc else "",
        "n_wards": int(len(g)),
        "counts": {k: int(counts.get(k, 0)) for k in ["red", "orange", "yellow", "green"]},
        "people_orange_red": int(at_risk["pop_total"].sum()),
        "city_p24_max": float(g["p24_max"].max()),
        "top": g.head(10)[["ward_id", "name", "risk", "level"]].to_dict("records"),
    }
    config.SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"Model: {model_name}")
    print("Levels:", summary["counts"], "| people in orange/red:", summary["people_orange_red"])
    print(g.head(8)[["name", "p24_max", "runoff_mm", "prob", "risk", "level"]].to_string(index=False))


if __name__ == "__main__":
    main()
