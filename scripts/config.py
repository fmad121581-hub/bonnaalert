"""Shared settings for the BonnaAlert pipeline.

Edit FLOOD_RISK_DIR to point at the flood-risk-mapping-dhaka project folder on
your laptop. Everything else is relative to this repository.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

# Laptop folder that holds the flood-risk-mapping-dhaka outputs (phase 2 and 3).
FLOOD_RISK_DIR = Path(os.environ.get(
    "FLOOD_RISK_DIR", "C:/Users/user/OneDrive/Projects/Flood_risk"))

PHASE2 = FLOOD_RISK_DIR / "data/output/phase2"
PHASE3 = FLOOD_RISK_DIR / "data/output/phase3"

INPUTS = {
    "wards": PHASE3 / "ward_risk.shp",          # GADM L4 wards with risk attributes
    "cn": PHASE2 / "cn_raster.tif",             # SCS curve number
    "twi": PHASE2 / "twi.tif",                  # topographic wetness index
    "dem": PHASE2 / "dem_clipped.tif",          # SRTM 30 m, metres
    "lulc": PHASE3 / "lulc_aligned.tif",        # ESA WorldCover on the DEM grid
}

# Outputs
DATA = ROOT / "data"
# served by GitHub Pages; BONNA_RUN=replay-2023-09-21 writes to docs/data/replay-2023-09-21
WEB_DATA = ROOT / "docs" / "data" / os.environ.get("BONNA_RUN", "")
WARDS_STATIC = DATA / "wards_static.geojson"
FORECAST_CSV = DATA / "forecast.csv"
WARDS_SCORED = WEB_DATA / "wards_scored.geojson"
SUMMARY_JSON = WEB_DATA / "summary.json"
TRAINING_CSV = DATA / "training" / "events.csv"
MODEL_PATH = ROOT / "models" / "trigger_xgb.json"
GUIDANCE = ROOT / "guidance" / "actions_bn.json"

# Forecast
FORECAST_HOURS = 72
TIMEZONE = "Asia/Dhaka"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Elevation below which a pixel counts as low-lying (same cut-off as phase 3)
LOW_ELEV_M = 4.0
# ESA WorldCover class for built-up
BUILT_CLASS = 50

# v0 event model: expert-prior logistic coefficients, NOT calibrated yet.
# Replace with the trained XGBoost model once radar-labelled events exist.
V0_COEF = {
    "intercept": -6.0,
    "log_runoff": 1.3,      # ln(1 + SCS-CN runoff depth in mm, from max 24 h rain)
    "twi_z": 0.6,           # standardised TWI (city-wide z-score)
    "low_elev_share": 1.5,  # share of ward area <= LOW_ELEV_M
    "built_share": 0.8,     # share of ward area that is built-up
}

# Risk = probability x vulnerability weight; alert levels on that risk score
VULN_FLOOR = 0.5            # weight = VULN_FLOOR + (1 - VULN_FLOOR) * vuln_norm
LEVELS = [                  # (lower bound, key, Bangla label, colour)
    (0.60, "red", "লাল: এখনই ব্যবস্থা নিন", "#c62828"),
    (0.35, "orange", "কমলা: প্রস্তুতি নিন", "#ef6c00"),
    (0.15, "yellow", "হলুদ: সতর্ক থাকুন", "#f9a825"),
    (0.00, "green", "সবুজ: ঝুঁকি কম", "#2e7d32"),
]

# Optional LLM wording for alerts (leave unset to use templates only)
LLM_MODEL = os.environ.get("BONNA_LLM_MODEL", "claude-sonnet-5-5")
