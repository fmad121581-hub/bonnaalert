# BonnaAlert

Ward-level flood and waterlogging early warning for Dhaka, in Bangla.
Entry for the International AI Builders Congress 2026, ClimateSphere AI → Climate Shield.

Twice a day the pipeline pulls a 72-hour rainfall forecast for every ward, turns it into
a flood probability, weights it by how vulnerable the ward's people are, and writes a
Bangla alert that says why and what to do. A static web dashboard (`docs/`) shows the
result; GitHub Pages serves it and GitHub Actions keeps it fresh.

## How a score is made

| Step | What happens | Script |
| --- | --- | --- |
| Ward features | Mean curve number, TWI, elevation, low-lying share, built-up share and vulnerability per ward, from the flood-risk-mapping-dhaka outputs | `prepare_wards.py` |
| Forecast | Hourly rain for 72 h at each ward's representative point (Open-Meteo) | `fetch_forecast.py` |
| Event model | SCS-CN runoff from the wettest 24 h, plus ponding terms → probability. Uses the trained XGBoost model if present, otherwise the v0 expert-prior logistic model | `score.py` |
| Risk and level | risk = probability × (0.5 + 0.5 × vulnerability); levels green / yellow / orange / red at 0.15, 0.35, 0.60 | `score.py` |
| Explanation | SHAP values (XGBoost) or logit contributions (v0) give each ward's top drivers | `score.py` |
| Alerts | Bangla template from model outputs + guidance file; optional LLM rewording with the same facts only; CAP 1.2 XML for orange/red | `alerts.py` |
| Training | Sentinel-1 change rasters → flooded share per ward per event; XGBoost with monotone constraints, validated leave-one-event-out against v0 | `build_labels.py`, `train_trigger.py` |

The v0 coefficients in `scripts/config.py` are expert priors, not fitted values. Treat its
probabilities as relative until the trained model replaces it.

## Run it

```bash
pip install -r requirements.txt

# 1. ward features (once, on the machine that has the flood-risk outputs)
set FLOOD_RISK_DIR=C:/Users/user/OneDrive/Projects/Flood_risk     # Windows
python scripts/prepare_wards.py

# 2. live run: forecast -> score -> alerts
python scripts/run_all.py              # add --llm to reword alerts (needs ANTHROPIC_API_KEY)

# 3. view the dashboard
cd docs && python -m http.server 8000  # open http://localhost:8000
```

Demo without any data: `python scripts/prepare_wards.py --demo` then `python scripts/run_all.py --demo`.

### Train the trigger model

```bash
# one row per ward for each storm (and one dry-season pair for "no flood" examples)
python scripts/build_labels.py --event 2023-09-21 --delta path/to/delta_s1_sep22_minus_sep10.tif --rule ge:3
python scripts/build_labels.py --event 2024-02-05 --delta path/to/dry_pair_delta.tif --rule ge:3
python scripts/train_trigger.py
```

The model is saved only if it beats v0 on held-out events.

## Publish

1. Push this folder to a new GitHub repository.
2. Settings → Pages → deploy from branch `main`, folder `/docs`.
3. Optional: add `ANTHROPIC_API_KEY` under Settings → Secrets → Actions.
4. Actions → "Ward forecast" → Run workflow once; it then runs at about 06:00 and 18:00 Dhaka time.

## Data

Open-Meteo forecast and ERA5 archive; Copernicus Sentinel-1 via ASF HyP3; ESA WorldCover;
WorldPop; SRTM; GADM ward boundaries. No personal data is collected; everything is
aggregated to wards.

## Limitations

- Forecast models run at roughly 10–25 km, so neighbouring wards often share rain values;
  ward differences come mostly from terrain, land cover and vulnerability.
- Radar change detection in dense urban areas is noisy; labels need checking against
  reported waterlogging hotspots.
- `guidance/actions_bn.json` is a draft and must be replaced with cited official guidance
  before any public use.
