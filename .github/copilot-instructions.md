# Copilot Instructions for tennis10

This repository is centered on an ATP tennis data pipeline, predictive modeling, and live betting/trading support.

## Big picture
- `tennis_data.db` is the central SQLite data store. Most scripts read from it.
- `data_pipeline.py` builds the database from tennis-data.co.uk and creates the schema for `players`, `matches`, `statistics`, and `odds`.
- `features.py` exposes `TennisFeatureExtractor`, which is the core feature-engineering contract used by training and live prediction.
- `train_and_evaluate.py` is the main training pipeline that combines logistic regression, neural networks, and the hierarchical Markov model.
- `live_prediction.py` is the live prediction entrypoint; it loads artifacts from `ml_models/` and uses `TennisFeatureExtractor` + `HierarchicalTennisModel`.

## What to edit carefully
- Do not assume an external database or data store; the codebase uses a local SQLite file (`tennis_data.db`) and explicit SQL queries.
- Model persistence is in `ml_models/` as pickle files, especially `ml_models/logistic_regression_trained.pkl` and `ml_models/neural_network_ensemble.pkl`.
- Feature engineering is not simple raw stats: it uses time decay, surface correlations, and symmetric feature differences.
- The logistic model is intentionally symmetric with `fit_intercept=False`; avoid adding a bias term unless you understand the design.

## Key workflows
- Install dependencies: `pip install -r requirements.txt`.
- Build or refresh the dataset: `python data_pipeline.py`.
- Run the unified model pipeline: `python train_and_evaluate.py`.
- Predict a live match: `python live_prediction.py "Player A" "Player B" "Hard" --odds1 1.85 --odds2 2.10`.
- Launch dashboard: `streamlit run dashboard/streamlit_app.py`.
- Run tests: `pytest tests/ -v`.

## Project-specific patterns
- Top-level scripts often use `if __name__ == "__main__":` and are intended to be run directly.
- Database queries use raw SQL inside Python, often via `pandas.read_sql_query` and `sqlite3`.
- The repository splits responsibilities by domain: data ingestion (`data_pipeline.py`, `validate_pipeline.py`), feature extraction (`features.py`), model training (`train_and_evaluate.py`, `ml_models/`), and live/trading support (`live_prediction.py`, `dashboard/streamlit_app.py`, `betfair/live_loop.py`).
- `tests/conftest.py` sets an autouse randomness seed for reproducible behavior.

## CI and quality checks
- GitHub Actions are defined in `.github/workflows/tests.yml`.
- CI installs dependencies and runs `pytest` with coverage, Black formatting, isort, flake8, and mypy.
- Keep code compatible with Python 3.10–3.12 as the workflow tests those versions.

## Useful files
- `data_pipeline.py` — data ingestion and SQLite schema
- `features.py` — all feature extraction logic and surface/time decay rules
- `hierarchical_model.py` — Markov-style prediction engine
- `ml_models/logistic_regression.py` — symmetric logistic regression implementation
- `train_and_evaluate.py` — consolidated training/evaluation runner
- `live_prediction.py` — live prediction CLI using saved model artifacts
- `dashboard/streamlit_app.py` — Streamlit analytics dashboard
- `tests/` — pytest tests for features, models, betting logic, and integration

## Avoid
- Do not invent new stateful database layers; keep the data flow centered on the existing SQLite files.
- Do not assume the app is a web service unless editing the dashboard or trading components.
- Do not change naming of core model artifacts without updating `live_prediction.py` and training scripts accordingly.
