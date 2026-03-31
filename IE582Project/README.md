# IE582Project: Nationwide Food Insecurity Index

This folder has been refactored into a production-style structure so model building and index generation are reproducible, testable, and separated from exploratory notebooks.

## Project structure

```text
IE582Project/
├── build_food_insecurity_index.py        # Backward-compatible wrapper entrypoint
├── pyproject.toml                        # Package + CLI metadata
├── requirements.txt                      # Python dependencies
├── Makefile                              # Common setup/run commands
├── src/
│   └── ie582_food_insecurity/
│       ├── __init__.py
│       ├── cli.py                        # CLI argument parsing + orchestration
│       ├── pipeline.py                   # Core index pipeline logic
│       ├── map_cli.py                    # County map CLI
│       └── visualization.py              # Map rendering + missing handling
├── data/
│   ├── raw/                              # Source datasets
│   └── interim/                          # Engineered/intermediate files
├── notebooks/
│   ├── 01_pipeline_run_and_review.ipynb  # Professional run/review notebook
│   └── legacy/                           # Original exploratory notebooks (archived)
├── docs/
│   └── report/                           # Report assets/screenshots/presentation
└── outputs/                              # Generated artifacts (created by pipeline)
```

## Setup

From `IE582Project`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (recommended for module/CLI workflows):

```bash
pip install -e .
```

If you are on macOS and `xgboost` fails to load `libxgboost.dylib`, install OpenMP once:

```bash
brew install libomp
```

## Run the pipeline

### Option A (existing command, still supported)

```bash
python build_food_insecurity_index.py
```

### Option B (module command)

```bash
PYTHONPATH=src python -m ie582_food_insecurity.cli
```

### Option C (installed CLI)

```bash
ie582-build-index
```

## Build county map visualization

Generate an interactive county-level map from pipeline outputs:

```bash
PYTHONPATH=src python -m ie582_food_insecurity.map_cli
```

Backward-compatible wrapper:

```bash
python build_food_insecurity_map.py
```

Installed CLI version:

```bash
ie582-build-map
```

Map CLI supports missing index handling with:

- `--missing-strategy keep`
- `--missing-strategy state_median` (default)
- `--missing-strategy national_median`
- `--missing-strategy zero`

Example:

```bash
PYTHONPATH=src python -m ie582_food_insecurity.map_cli \
  --index-file outputs/nationwide_food_insecurity_index.csv \
  --fips-col fips \
  --index-col food_insecurity_index \
  --engine geo \
  --missing-strategy state_median \
  --output-html outputs/maps/county_index_map.html
```

If your IDE preview still shows an empty map, start a local server and open in browser:

```bash
python -m http.server 8000
```

Then open:

`http://localhost:8000/outputs/maps/county_index_map_geo.html`

## Useful options

```bash
python build_food_insecurity_index.py \
  --feature-data data/interim/pcaAfter_data.csv \
  --fips-data data/raw/FI.csv \
  --target-col "% Food Insecure" \
  --cv-folds 5 \
  --n-estimators 400 \
  --max-depth 4 \
  --learning-rate 0.05 \
  --output-dir outputs
```

## Outputs

Generated under `outputs/`:

- `nationwide_food_insecurity_index.csv`
- `nationwide_food_insecurity_index_unsorted.csv`
- `feature_weights_shap.csv`
- `map_fivi.csv`
- `top10_high_risk_counties.csv`
- `model_metrics.json`
- `data_quality_report.json`
- `run_config.json`
- `maps/county_index_map.html`
- `maps/county_index_map_data.csv`
- `maps/missing_counties.csv`
- `maps/map_summary.json`

## Method summary

1. Train/test split + XGBoost regression for `% Food Insecure`
2. Refit model on full dataset
3. SHAP mean absolute value for feature weighting
4. Correlation-aware feature orientation (flip negatively correlated features)
5. Weighted composite score normalized to 0–100 (`higher = higher risk`)
6. Join with county `FIPS` and export analytics + map-ready file

## Notes

- `notebooks/legacy/` preserves original class project notebooks without mixing them into production logic.
- Core pipeline code is now centralized in `src/ie582_food_insecurity/pipeline.py`.
