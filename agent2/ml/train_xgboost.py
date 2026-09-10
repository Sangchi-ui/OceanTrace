"""
Training Pipeline for OceanTrace Agent 2 XGBoost Physics Residual Model.
Trains gradient-boosted regressors on observed drift trajectory datasets to
calibrate Lagrangian transport residuals and predict dynamic ensemble uncertainty.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from agent2.ml.generate_drift_dataset import generate_drift_dataset
from agent2.ml.xgboost_residual import XGBoostPhysicsResidualModel


def train_xgboost_residual_model(
    data_path: str,
    output_dir: str = "models/agent2",
    n_estimators: int = 150,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains XGBoost regressors on physical residual drift datasets and exports models.
    """
    if not os.path.exists(data_path):
        print(f"Dataset '{data_path}' not found. Generating default calibrated dataset...")
        generate_drift_dataset(n_samples=2500, output_path=data_path, seed=random_state)

    print(f"\nLoading drift trajectory dataset from: {data_path}")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    feature_cols = XGBoostPhysicsResidualModel.FEATURE_NAMES
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols}")

    target_cols = ["dx_residual_meters", "dy_residual_meters", "uncertainty_scale"]
    for t in target_cols:
        if t not in df.columns:
            raise ValueError(f"Dataset is missing target column: '{t}'")

    X = df[feature_cols].values
    y_dx = df["dx_residual_meters"].values
    y_dy = df["dy_residual_meters"].values
    y_unc = df["uncertainty_scale"].values

    # Train / Val Split
    X_train, X_val, y_dx_tr, y_dx_val, y_dy_tr, y_dy_val, y_u_tr, y_u_val = train_test_split(
        X, y_dx, y_dy, y_unc, test_size=test_size, random_state=random_state
    )

    print(f"Dataset size: {len(df)} records | Train: {len(X_train)} | Validation: {len(X_val)}")
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")

    # 1. Train model_dx
    print("\nTraining XGBoost model for X-displacement residual (dx)...")
    model_dx = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=random_state
    )
    model_dx.fit(X_train, y_dx_tr)
    dx_preds = model_dx.predict(X_val)
    dx_mae = float(mean_absolute_error(y_dx_val, dx_preds))
    dx_rmse = float(np.sqrt(mean_squared_error(y_dx_val, dx_preds)))
    dx_r2 = float(r2_score(y_dx_val, dx_preds))
    print(f"  Validation dx MAE:  {dx_mae:.2f} m | RMSE: {dx_rmse:.2f} m | R^2: {dx_r2:.4f}")

    # 2. Train model_dy
    print("\nTraining XGBoost model for Y-displacement residual (dy)...")
    model_dy = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=random_state
    )
    model_dy.fit(X_train, y_dy_tr)
    dy_preds = model_dy.predict(X_val)
    dy_mae = float(mean_absolute_error(y_dy_val, dy_preds))
    dy_rmse = float(np.sqrt(mean_squared_error(y_dy_val, dy_preds)))
    dy_r2 = float(r2_score(y_dy_val, dy_preds))
    print(f"  Validation dy MAE:  {dy_mae:.2f} m | RMSE: {dy_rmse:.2f} m | R^2: {dy_r2:.4f}")

    # 3. Train model_uncertainty
    print("\nTraining XGBoost model for uncertainty scale factor...")
    model_unc = xgb.XGBRegressor(
        n_estimators=max(80, n_estimators // 2),
        max_depth=max(3, max_depth - 1),
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=random_state
    )
    model_unc.fit(X_train, y_u_tr)
    unc_preds = model_unc.predict(X_val)
    unc_mae = float(mean_absolute_error(y_u_val, unc_preds))
    unc_r2 = float(r2_score(y_u_val, unc_preds))
    print(f"  Validation uncertainty MAE: {unc_mae:.4f} | R^2: {unc_r2:.4f}")

    # Save models and metadata
    os.makedirs(output_dir, exist_ok=True)
    model_dx.save_model(os.path.join(output_dir, "xgb_dx_residual.json"))
    model_dy.save_model(os.path.join(output_dir, "xgb_dy_residual.json"))
    model_unc.save_model(os.path.join(output_dir, "xgb_uncertainty.json"))

    # Feature Importances
    fi_dx = {f: round(float(imp), 4) for f, imp in zip(feature_cols, model_dx.feature_importances_)}
    fi_dy = {f: round(float(imp), 4) for f, imp in zip(feature_cols, model_dy.feature_importances_)}

    metrics = {
        "dataset_path": data_path,
        "total_records": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "metrics": {
            "dx_residual": {
                "mae_meters": round(dx_mae, 2),
                "rmse_meters": round(dx_rmse, 2),
                "r2_score": round(dx_r2, 4)
            },
            "dy_residual": {
                "mae_meters": round(dy_mae, 2),
                "rmse_meters": round(dy_rmse, 2),
                "r2_score": round(dy_r2, 4)
            },
            "uncertainty_scale": {
                "mae": round(unc_mae, 4),
                "r2_score": round(unc_r2, 4)
            }
        },
        "feature_importance_dx": fi_dx,
        "feature_importance_dy": fi_dy
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTrained models and metrics successfully saved to: '{output_dir}/'")
    print(f"  - {os.path.join(output_dir, 'xgb_dx_residual.json')}")
    print(f"  - {os.path.join(output_dir, 'xgb_dy_residual.json')}")
    print(f"  - {os.path.join(output_dir, 'xgb_uncertainty.json')}")
    print(f"  - {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train OceanTrace Agent 2 XGBoost Physics Residual Model")
    parser.add_argument("--data", "-d", type=str, default="data/drift_trajectories_sample.csv", help="Path to input CSV dataset")
    parser.add_argument("--output", "-o", type=str, default="models/agent2", help="Directory to save trained models")
    parser.add_argument("--estimators", "-e", type=int, default=150, help="Number of boosting trees")
    parser.add_argument("--depth", type=int, default=5, help="Maximum tree depth")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    train_xgboost_residual_model(
        data_path=args.data,
        output_dir=args.output,
        n_estimators=args.estimators,
        max_depth=args.depth,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
