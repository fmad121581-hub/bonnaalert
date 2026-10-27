# BonnaAlert

Ward-level flood and waterlogging early warning for Dhaka, in Bangla.

**Live dashboard:** https://fmad121581-hub.github.io/bonnaalert/ · storm replay: [`#replay-2023-09-21`](https://fmad121581-hub.github.io/bonnaalert/#replay-2023-09-21)

> **Status: personal prototype.** This is not an official warning service. The model is
> uncalibrated (v0 expert priors), and the Bangla guidance has not been reviewed by DDM or the
> city corporations. For real warnings, follow BMD and DDM.

Twice a day the pipeline pulls a 72-hour rainfall forecast for every ward, turns it into
a flood probability, weights it by how vulnerable the ward's people are, and writes a
Bangla alert that says why and what to do. A static web dashboard (`docs/`) shows the
result; GitHub Pages serves it and GitHub Actions keeps it fresh.

Alerts are written in Bangla because that is what residents read. The dashboard opens in English
(`?lang=bn` or the switch for Bangla) and shows an English translation under each Bangla alert.

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

### Replay a past storm

```bash
# 19-22 Sep 2023, ERA5 rain pattern scaled so the city-median 24 h total matches BMD's 113 mm
BONNA_RUN=replay-2023-09-21 python scripts/run_all.py --replay data/events/2023-09-21_rain_raw.json --scale-to 113
```

The dashboard shows it under "২১ সেপ্টেম্বর ২০২৩ রিপ্লে" or at `#replay-2023-09-21`.

## Validation so far

| Check | Result |
| --- | --- |
| Sentinel-1 flooded share per ward, 21 Sep 2023 (delta ≥ +3 dB) vs v0 | AUC 0.33; radar brightening does not track street waterlogging |
| Same, darkening rule (delta ≤ −3 dB) vs ward traits | rho +0.60 with low-lying land: it sees fringe inundation, not the core |
| 49 reported waterlogging spots (DSCC/WASA 2018, DNCC 2021, Sep 2023 news, 2026) → 28 of 115 city wards; v0 risk at 113 mm | AUC 0.52 (95% CI 0.40–0.65) |
| Static H × E × V index, terrain features, OSM canal and drain features on the same wards | AUC 0.41–0.54 |

Terrain and land cover do not separate waterlogging hotspots inside the city corporations; drain capacity and
blockage do, and those are not in public data. Next step: city corporation complaint and hotspot records as
labels and features. Scripts: `scripts/build_labels.py`, `scripts/validate_hotspots.py`; data in `data/validation/`.

`train_trigger.py` has not been run, on purpose. Labels from one radar event were rejected (see above), and
a model fitted to them would look better on paper without being more useful. `models/` stays empty until
better labels exist.

## Where it goes next

The pipeline is built so the model can improve with every storm:

1. After each heavy-rain event, collect DNCC/DSCC complaint and hotspot records as ward-level labels.
2. Add the event to `data/training/` and rerun `train_trigger.py` (XGBoost, monotone constraints,
   leave-one-event-out).
3. The trained model replaces v0 only if it beats v0 on held-out storms.

Before any of this reaches the public, the guidance in `guidance/actions_bn.json` needs DDM or city-corporation
review, and a small pilot (a handful of wards over one monsoon) would show whether people act on the alerts.

Possible sustainability model: free alerts for residents, with ward-level risk data offered to MFIs, NGOs,
insurers and the city corporations.

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
- `guidance/actions_bn.json` paraphrases the sources it lists; DDM or city corporation review is still
  needed before any public alert.
- ERA5 (~25 km) recorded about 25 mm for the 21 Sep 2023 storm while BMD measured 113 mm in six hours;
  gridded forecasts and archives smooth out convective downpours.
