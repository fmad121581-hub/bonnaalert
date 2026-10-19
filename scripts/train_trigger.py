"""Train the XGBoost trigger model on radar-labelled events.

Validation is leave-one-event-out: train on all storms but one, score the
held-out storm, repeat. The model is saved (and then used by score.py) only
if it beats the v0 expert-prior model on that test, unless --force.

    python scripts/train_trigger.py
"""
import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import config
from features import FEATURES, build_features
from score import predict_v0

PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 3, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 1.0,
    "min_child_weight": 5,
    # more rain, wetter, lower or more paved can never lower the probability
    "monotone_constraints": "(1,1,1,1)",
}
ROUNDS = 200


def auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="save even if it loses to v0")
    args = ap.parse_args()

    static = gpd.read_file(config.WARDS_STATIC)
    ev = pd.read_csv(config.TRAINING_CSV)
    df = ev.merge(static.drop(columns="geometry"), on="ward_id", how="inner")
    X = build_features(df, static)
    y = df["label"].values
    events = df["event_id"].unique()
    print(f"{len(events)} events, {len(df)} rows, {y.mean():.1%} flooded")
    if len(events) < 2:
        raise SystemExit("Need at least 2 events (one can be a dry-season pair) to validate.")

    folds = []
    for e in events:
        tr, te = df["event_id"] != e, df["event_id"] == e
        dtr = xgb.DMatrix(X.loc[tr, FEATURES], label=y[tr], feature_names=FEATURES)
        dte = xgb.DMatrix(X.loc[te, FEATURES], feature_names=FEATURES)
        bst = xgb.train(PARAMS, dtr, ROUNDS)
        p_x = bst.predict(dte)
        p_v0 = predict_v0(X.loc[te], static)[0]
        folds.append({"held_out": str(e), "n": int(te.sum()), "flooded": int(y[te].sum()),
                      "auc_xgb": auc(y[te], p_x), "auc_v0": auc(y[te], p_v0)})

    rep = pd.DataFrame(folds)
    print(rep.round(3).to_string(index=False))
    # pooled AUC across held-out predictions is fairer when single events are one-class
    pooled_x, pooled_v0, pooled_y = [], [], []
    for e in events:
        tr, te = df["event_id"] != e, df["event_id"] == e
        bst = xgb.train(PARAMS, xgb.DMatrix(X.loc[tr, FEATURES], label=y[tr],
                                            feature_names=FEATURES), ROUNDS)
        pooled_x += list(bst.predict(xgb.DMatrix(X.loc[te, FEATURES], feature_names=FEATURES)))
        pooled_v0 += list(predict_v0(X.loc[te], static)[0])
        pooled_y += list(y[te])
    a_x, a_v0 = auc(pooled_y, pooled_x), auc(pooled_y, pooled_v0)
    print(f"\nPooled held-out AUC  XGBoost {a_x:.3f}  |  v0 {a_v0:.3f}")

    final = xgb.train(PARAMS, xgb.DMatrix(X[FEATURES], label=y, feature_names=FEATURES), ROUNDS)
    # must clearly beat v0 and be clearly better than chance
    base = a_v0 if np.isfinite(a_v0) else 0.5
    keep = args.force or (np.isfinite(a_x) and a_x >= max(base + 0.02, 0.6))
    report = {"events": list(map(str, events)), "rows": int(len(df)),
              "pooled_auc_xgb": a_x, "pooled_auc_v0": a_v0,
              "saved": bool(keep), "folds": folds}
    (config.TRAINING_CSV.parent / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if keep:
        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        final.save_model(str(config.MODEL_PATH))
        print(f"Saved -> {config.MODEL_PATH} (score.py will use it from now on)")
    else:
        print("XGBoost did not beat v0; model not saved. Add more events or use --force.")


if __name__ == "__main__":
    main()
