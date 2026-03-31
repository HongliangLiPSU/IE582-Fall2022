from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - runtime env specific
    raise SystemExit(
        "Failed to import xgboost. On macOS, install OpenMP runtime first:\n"
        "  brew install libomp\n"
        "Then re-run inside your virtual environment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PipelineConfig:
    feature_data: Path = PROJECT_ROOT / "data" / "interim" / "pcaAfter_data.csv"
    fips_data: Path = PROJECT_ROOT / "data" / "raw" / "FI.csv"
    target_col: str = "% Food Insecure"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    output_dir: Path = PROJECT_ROOT / "outputs"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return cleaned


def load_feature_data(path: Path, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Feature data file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df = _clean_columns(df)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if numeric_df.isna().any().any():
        numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    y = numeric_df[target_col]
    X = numeric_df.drop(columns=[target_col])
    return X, y


def make_model(
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=random_state,
    )


def evaluate_model(
    model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_test": int(len(y_test)),
    }


def evaluate_with_cv(
    random_state: int,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> Dict[str, float]:
    model = make_model(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=1,
        return_train_score=False,
    )
    rmse = -scores["test_rmse"]
    mae = -scores["test_mae"]
    r2 = scores["test_r2"]
    return {
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "mae_mean": float(np.mean(mae)),
        "mae_std": float(np.std(mae)),
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "n_folds": int(cv_folds),
    }


def compute_shap_weights(model: XGBRegressor, X: pd.DataFrame) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_array.shape}")

    importance = np.abs(shap_array).mean(axis=0)
    total = float(importance.sum())
    if total == 0.0:
        weights = np.repeat(1.0 / len(importance), len(importance))
    else:
        weights = importance / total

    return pd.Series(weights, index=X.columns, name="shap_weight")


def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    min_vals = df.min(axis=0)
    ranges = df.max(axis=0) - min_vals
    ranges = ranges.replace(0, np.nan)
    normalized = (df - min_vals) / ranges
    return normalized.fillna(0.0)


def build_index(
    X: pd.DataFrame, y: pd.Series, weights: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X_norm = min_max_normalize(X)

    corr = X.corrwith(y).fillna(0.0)
    direction = np.where(corr < 0, -1, 1)
    direction_series = pd.Series(direction, index=X.columns, name="direction")

    oriented = X_norm.copy()
    negative_cols = direction_series[direction_series < 0].index
    for col in negative_cols:
        oriented[col] = 1.0 - oriented[col]

    raw_index = oriented.mul(weights, axis=1).sum(axis=1)

    raw_min = float(raw_index.min())
    raw_max = float(raw_index.max())
    if raw_max > raw_min:
        score_0_100 = ((raw_index - raw_min) / (raw_max - raw_min)) * 100.0
    else:
        score_0_100 = pd.Series(np.repeat(50.0, len(raw_index)), index=raw_index.index)

    rank_pct = score_0_100.rank(method="average", pct=True) * 100.0

    index_df = pd.DataFrame(
        {
            "food_insecurity_index": score_0_100,
            "food_insecurity_index_rank_pct": rank_pct,
            "food_insecurity_index_raw": raw_index,
        }
    )

    return index_df, corr.rename("corr_with_target"), direction_series


def load_fips_data(path: Path, expected_rows: int) -> Tuple[pd.Series, pd.Series | None]:
    if not path.exists():
        raise FileNotFoundError(f"FIPS data file not found: {path}")

    fips_df = pd.read_csv(path, dtype=str)
    fips_df = _clean_columns(fips_df)

    fips_col = None
    for col in fips_df.columns:
        if "fips" in col.lower():
            fips_col = col
            break
    if fips_col is None:
        raise ValueError(f"No FIPS column found in {path}. Columns: {list(fips_df.columns)}")

    if len(fips_df) != expected_rows:
        raise ValueError(
            f"Row mismatch between FIPS data ({len(fips_df)}) and feature data ({expected_rows}). "
            "Please ensure both files are aligned by county row order."
        )

    fips = fips_df[fips_col].astype(str).str.strip().str.zfill(5)

    fi_ref = None
    for col in fips_df.columns:
        if col.lower() == "fi":
            fi_ref = pd.to_numeric(fips_df[col], errors="coerce")
            fi_ref.name = "observed_fi_reference"
            break

    return fips, fi_ref


def build_data_quality_report(
    X: pd.DataFrame,
    y: pd.Series,
    fips: pd.Series,
) -> Dict[str, Any]:
    missing_feature_cells = int(X.isna().sum().sum())
    missing_target_cells = int(y.isna().sum())
    const_features = [col for col in X.columns if float(X[col].nunique()) <= 1]

    fips_clean = fips.astype(str).str.strip()
    duplicate_fips = int(fips_clean.duplicated().sum())
    missing_fips = int((fips_clean == "").sum() + fips_clean.isna().sum())

    return {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "missing_feature_cells": missing_feature_cells,
        "missing_target_cells": missing_target_cells,
        "constant_feature_count": int(len(const_features)),
        "constant_features": const_features,
        "duplicate_fips_count": duplicate_fips,
        "missing_fips_count": missing_fips,
    }


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_feature_data(config.feature_data, config.target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    eval_model = make_model(
        random_state=config.random_state,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
    )
    eval_model.fit(X_train, y_train)
    split_metrics = evaluate_model(eval_model, X_test, y_test)

    cv_metrics = evaluate_with_cv(
        random_state=config.random_state,
        X=X,
        y=y,
        cv_folds=config.cv_folds,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
    )

    full_model = make_model(
        random_state=config.random_state,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
    )
    full_model.fit(X, y)

    pred_full = pd.Series(full_model.predict(X), index=X.index, name="predicted_fi")
    weights = compute_shap_weights(full_model, X)
    index_df, corr, direction = build_index(X, y, weights)

    fips, fi_ref = load_fips_data(config.fips_data, expected_rows=len(X))

    county_df = pd.DataFrame(
        {
            "fips": fips.reset_index(drop=True),
            "food_insecurity_index": index_df["food_insecurity_index"].reset_index(drop=True),
            "food_insecurity_index_rank_pct": index_df[
                "food_insecurity_index_rank_pct"
            ].reset_index(drop=True),
            "predicted_fi": pred_full.reset_index(drop=True),
            "observed_fi_model_target": y.reset_index(drop=True),
        }
    )

    if fi_ref is not None:
        county_df["observed_fi_reference"] = fi_ref.reset_index(drop=True)

    county_df_unsorted = county_df.copy()
    county_df = county_df.sort_values("food_insecurity_index", ascending=False).reset_index(drop=True)
    county_df["rank_desc"] = np.arange(1, len(county_df) + 1)

    weight_df = pd.concat([weights, corr, direction], axis=1).reset_index()
    weight_df = weight_df.rename(columns={"index": "feature"})
    weight_df = weight_df.sort_values("shap_weight", ascending=False).reset_index(drop=True)

    metrics = {
        "test_split": split_metrics,
        "cross_validation": cv_metrics,
    }

    outputs = {
        "index": config.output_dir / "nationwide_food_insecurity_index.csv",
        "index_unsorted": config.output_dir / "nationwide_food_insecurity_index_unsorted.csv",
        "weights": config.output_dir / "feature_weights_shap.csv",
        "metrics": config.output_dir / "model_metrics.json",
        "quality": config.output_dir / "data_quality_report.json",
        "map": config.output_dir / "map_fivi.csv",
        "top10": config.output_dir / "top10_high_risk_counties.csv",
        "run_config": config.output_dir / "run_config.json",
    }

    county_df.to_csv(outputs["index"], index=False)
    county_df_unsorted.to_csv(outputs["index_unsorted"], index=False)
    weight_df.to_csv(outputs["weights"], index=False)

    map_df = county_df[["fips", "food_insecurity_index"]].rename(
        columns={"fips": "FIPS", "food_insecurity_index": "FIVI"}
    )
    map_df.to_csv(outputs["map"], index=False)
    county_df.head(10).to_csv(outputs["top10"], index=False)

    with outputs["metrics"].open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with outputs["quality"].open("w", encoding="utf-8") as f:
        json.dump(build_data_quality_report(X, y, fips), f, indent=2)
    with outputs["run_config"].open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}, f, indent=2)

    return {
        "n_rows": len(X),
        "metrics": metrics,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
