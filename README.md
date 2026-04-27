# arian-wildfire-prediction

## Problem Statement
Azerbaijan faces rising wildfire risk — temperatures have increased +0.4 °C/decade since 1980 while summer precipitation declines. No regional 30-day fire-risk forecast exists. ARIAN integrates satellite, meteorological, and forest inventory data into a unified ML pipeline that delivers daily wildfire probability, expected fire counts, and 30-day weather forecasts for 16 Azerbaijani cities.

## Why It Matters
Wildfire incidents increased measurably since 2010, with the 2021–2022 seasons causing substantial forest loss across the Greater and Lesser Caucasus (~1.2 M hectares). Early-warning forecasts give emergency services 7–14 days of lead time to pre-position resources and issue evacuations, reducing damage (~$70–80 M/year from extreme weather in Azerbaijan).

## Target
**`fire_occurred`** — binary (1 = ≥1 NASA FIRMS VIIRS hotspot within 50 km of city centroid on forecast date). Label window: next-day 00:00–23:59 local time.  
**`fire_count`** — expected number of FIRMS hotspots in the same 50 km buffer (non-negative integer; Poisson regression).

## Features

| Source | Name | Units | Aggregation |
|--------|------|-------|-------------|
| Open-Meteo | temperature_2m | °C | daily mean; lags 1/3/7/14 d |
| Open-Meteo | wind_speed_10m | m/s | daily mean; lags 1/3/7/14 d |
| Open-Meteo | relative_humidity_2m | % | daily mean; lags 1/3/7/14 d |
| Open-Meteo | precipitation_7d_sum | mm | rolling 7-day sum |
| Open-Meteo | rain_30d_sum | mm | rolling 30-day sum |
| Open-Meteo | heatwave_flag | binary | 3+ consecutive days above local p90 |
| FWI (computed) | fwi, ffmc, dc, dsr | — | daily value from Open-Meteo inputs |
| SPEI (computed) | spei_3, spi_1 | σ | 3-month / 1-month standardized |
| ESA WorldCover | land_cover_class | class | dominant class in 50 km buffer |
| AZ Forest Boundary | forest_fraction | fraction | forest area / buffer area |
| MESE QURULUSU | dominant_species | class | modal species by stand area |
| MESE QURULUSU | mean_crown_density | 0.1–1.0 | area-weighted canopy closure |
| MESE QURULUSU | pct_old_growth | fraction | stands age ≥ 80 yr / total area |
| WorldPop + OSM | human_activity_score | — | population_density × road_density |
| NASA FIRMS | days_since_last_fire | days | days since last hotspot in buffer |
| NASA FIRMS | fire_count_30d | count | hotspot count in prior 30 days |

## Horizon
**t+1 to t+30 days** — daily wildfire risk scores and weather forecasts. Intermediate horizons (2, 4–6, 8–13, 15–29) are piecewise-linearly interpolated between direct-forecast checkpoints.

## Dataset
| Source | Period | Granularity | Region |
|--------|--------|-------------|--------|
| Open-Meteo weather | 2020–present | Hourly + daily | 16 Azerbaijani cities |
| NASA FIRMS VIIRS-SNPP | 2020–2025 | ~375 m point events | 50 km city buffers |
| ESA WorldCover | 2020 | 10 m raster | Azerbaijan |
| WorldPop | 2020–2026 | 100 m annual raster | Azerbaijan |
| MESE QURULUSU Forest Inventory | Static | Stand-level polygons | National forest boundary |

## Key Definitions
- **FWI** — Canadian Fire Weather Index (FFMC, DMC, DC, ISI, BUI, FWI, DSR); computed from temperature, humidity, wind, rain.
- **SPEI** — Standardized Precipitation-Evapotranspiration Index; drought measure accounting for evaporative demand.
- **FIRMS** — NASA Fire Information for Resource Management System; satellite fire hotspot detections.
- **MESE QURULUSU** — Azerbaijani State Forest Management Inventory; stand-level species, crown density, and age class.
- **HGBC / HGBR** — HistGradientBoostingClassifier / Regressor; primary wildfire models with isotonic calibration.
- **Mann-Kendall** — Non-parametric test for monotonic trend in climate time series.
- **AUC-ROC** — Area Under the ROC Curve; primary wildfire classifier metric (target ≥ 0.85).

## Team & Roles

| Member | Phase | Tasks |
|--------|-------|-------|
| **Raul** | **Phase 1 — Data Ingestion** | Fetch Open-Meteo hourly + daily (16 cities); load NASA FIRMS hotspots, ESA WorldCover, WorldPop, OSM roads, Azerbaycan.kmz forest mask, MESE QURULUSU forest inventory; compute FWI (`compute_fire_weather_index.py`) and SPEI/SPI (`compute_drought_indices.py`); apply quality controls (dedup, clip, ≥80% coverage check); output `data/interim/` |
| **Raul** | **Phase 4 — Wildfire Modeling** | Assemble wildfire feature matrix (weather lags, FWI, drought indices, land cover, forest inventory, human activity, fire history); train `CalibratedClassifierCV(HGBC)` for `fire_occurred`; train `HGBR(Poisson)` for `fire_count`; evaluate AUC-ROC, Average Precision, Brier Score on temporal hold-out |
| **Aysu** | **Phase 2 — Weather Forecasting** | Weather feature engineering (lag/rolling features, sin/cos wind direction, city dummies, no-leakage temporal split); train HGBR models per target per horizon; select best model by MAE; save `.joblib` files |
| **Aysu** | **Phase 5 — Web / API** | FastAPI app (`app/main.py`); `/weather` endpoint; Pydantic response schemas (`schemas.py`); static single-page dashboard (`app/static/index.html`) |
| **Ilaha** | **Phase 2 — Weather Forecasting** | Weather data cleaning (notebook 02); EDA and distribution checks; model experiments and evaluation on held-out 6-month period; reliability and residual diagnostics |
| **Ilaha** | **Phase 4 — Wildfire Modeling** | Wildfire feature validation and EDA (notebook 07); model evaluation — calibration curve, PR curve, threshold optimization for F1; per-city performance breakdown |
| **Asif** | **Phase 3 — Climate Analysis** | Run Mann-Kendall + Theil-Sen annual trend analysis; compute monthly and daily climatology (DOY mean ± 1σ); z-score anomaly labelling; forecast-vs-climatology comparison; seasonal peaks report; output `reports/climate/` |
| **Asif** | **Phase 4 — Feature Engineering** | Compute FWI system outputs; spatial join of MESE QURULUSU polygons to city buffers (dominant_species, mean_crown_density, pct_old_growth); prepare extended feature set slots (NDWI, terrain, soil moisture — pending data) |
| **Asif** | **Phase 5 — Web / API** | `/wildfire` and `/insights` routes (`routes/wildfire.py`, `routes/insights.py`); integrate forest inventory attributes into API response |
| **Nurana** | **Phase 2 — Weather Forecasting** | Weather feature engineering (notebooks 03–04); cross-city comparison EDA; horizon interpolation validation |
| **Nurana** | **Phase 5 — Web / API** | `/health` endpoint; front-end dashboard UX; end-to-end integration testing |

## Daily Activities
2026-04-20 — All Team — Project kick-off; repo structure set up, Open-Meteo API explored, 16 cities and target variables selected.
2026-04-21 — Raul — Data ingestion pipeline built; full Open-Meteo historical fetch (hourly + daily, 2020–present) for all 16 cities completed.
2026-04-22 — Raul — Database design finalized; DuckDB schema implemented, raw data loaded and validated.
2026-04-23 — Aysu / Ilaha / Nurana — Data cleaning pipeline applied; feature engineering (lags, rolling windows, heatwave flag) completed for weather data.
2026-04-24 — Raul — Pipeline automation complete; orchestrator (`generate_data.py`) with incremental loading, quality gates, and logging operational.
2026-04-25 — Asif — ARIAN v3.1 blueprint finalized; all Tier-0 datasets confirmed collected (Open-Meteo, FIRMS, ESA WorldCover, WorldPop, OSM, Azerbaycan.kmz, MESE QURULUSU).
2026-04-27 — All Team — README updated with full task breakdown per member; EDA phase (Day 6) begins.
