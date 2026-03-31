from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nationwide food insecurity index from ML model outputs."
    )
    parser.add_argument(
        "--feature-data",
        type=Path,
        default=PipelineConfig.feature_data,
        help="CSV with engineered features and target column.",
    )
    parser.add_argument(
        "--fips-data",
        type=Path,
        default=PipelineConfig.fips_data,
        help="CSV containing county FIPS codes (and optional FI reference column).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=PipelineConfig.target_col,
        help="Target column name in feature dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=PipelineConfig.test_size,
        help="Test split ratio for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=PipelineConfig.random_state,
        help="Random seed for deterministic train/test split and model training.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=PipelineConfig.cv_folds,
        help="K-fold cross-validation folds for stable model quality estimates.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=PipelineConfig.n_estimators,
        help="Number of boosting trees in XGBoost model.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=PipelineConfig.max_depth,
        help="Maximum tree depth in XGBoost model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=PipelineConfig.learning_rate,
        help="Learning rate in XGBoost model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PipelineConfig.output_dir,
        help="Directory to write pipeline outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        feature_data=args.feature_data,
        fips_data=args.fips_data,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    result = run_pipeline(config)
    split = result["metrics"]["test_split"]
    cv = result["metrics"]["cross_validation"]

    print("Pipeline complete.")
    print(f"Feature rows: {result['n_rows']}")
    print(
        "Test metrics: "
        f"RMSE={split['rmse']:.4f}, "
        f"MAE={split['mae']:.4f}, "
        f"R2={split['r2']:.4f}"
    )
    print(
        "CV metrics: "
        f"RMSE={cv['rmse_mean']:.4f}±{cv['rmse_std']:.4f}, "
        f"MAE={cv['mae_mean']:.4f}±{cv['mae_std']:.4f}, "
        f"R2={cv['r2_mean']:.4f}±{cv['r2_std']:.4f}"
    )

    for name, path in result["outputs"].items():
        print(f"Saved ({name}): {path}")


if __name__ == "__main__":
    main()
