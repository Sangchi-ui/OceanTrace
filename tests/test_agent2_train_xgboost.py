"""
Tests for Agent 2 XGBoost Physics Residual Training Pipeline and Dataset Generation.
"""

import os
import tempfile
import pytest

from agent2.ml.generate_drift_dataset import generate_drift_dataset
from agent2.ml.train_xgboost import train_xgboost_residual_model
from agent2.ml.xgboost_residual import XGBoostPhysicsResidualModel


def test_generate_drift_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_drift.csv")
        df = generate_drift_dataset(n_samples=100, output_path=csv_path, seed=123)

        assert os.path.exists(csv_path)
        assert len(df) == 100
        for col in XGBoostPhysicsResidualModel.FEATURE_NAMES:
            assert col in df.columns
        assert "dx_residual_meters" in df.columns
        assert "dy_residual_meters" in df.columns
        assert "uncertainty_scale" in df.columns


def test_train_xgboost_residual_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "train_drift.csv")
        generate_drift_dataset(n_samples=200, output_path=csv_path, seed=42)

        out_model_dir = os.path.join(tmpdir, "models")
        metrics = train_xgboost_residual_model(
            data_path=csv_path,
            output_dir=out_model_dir,
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1
        )

        assert "metrics" in metrics
        assert "dx_residual" in metrics["metrics"]
        assert "dy_residual" in metrics["metrics"]
        assert "r2_score" in metrics["metrics"]["dx_residual"]

        assert os.path.exists(os.path.join(out_model_dir, "xgb_dx_residual.json"))
        assert os.path.exists(os.path.join(out_model_dir, "xgb_dy_residual.json"))
        assert os.path.exists(os.path.join(out_model_dir, "xgb_uncertainty.json"))
        assert os.path.exists(os.path.join(out_model_dir, "metrics.json"))

        # Verify model loading
        loaded_model = XGBoostPhysicsResidualModel(model_path=out_model_dir)
        pred = loaded_model.predict_residual(
            u_current=0.2,
            v_current=-0.1,
            u_wind=5.0,
            v_wind=2.0,
            windage=0.03,
            sst=26.0,
            duration_hours=12.0,
            observed_area_km2=5.0
        )
        assert "dx_residual_meters" in pred
        assert "dy_residual_meters" in pred
        assert "uncertainty_scale" in pred
