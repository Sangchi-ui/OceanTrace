"""
Tests for OceanTrace Agent 2 XGBoost Physics Residual and Uncertainty Model.
"""

import os
import tempfile
import numpy as np
import pytest

from agent2.ml.xgboost_residual import XGBoostPhysicsResidualModel
from agent2.physics.oil_properties import get_oil_properties


def test_xgboost_residual_init_and_predict():
    model = XGBoostPhysicsResidualModel()
    assert model.is_trained

    props = get_oil_properties("medium_crude")
    pred = model.predict_residual(
        u_current=0.25,
        v_current=-0.15,
        u_wind=6.5,
        v_wind=-4.0,
        windage=0.035,
        sst=22.0,
        duration_hours=12.0,
        observed_area_km2=5.2,
        aspect_ratio=2.1,
        oil_props=props
    )

    assert "dx_residual_meters" in pred
    assert "dy_residual_meters" in pred
    assert "uncertainty_scale" in pred
    assert isinstance(pred["dx_residual_meters"], float)
    assert isinstance(pred["dy_residual_meters"], float)
    assert pred["uncertainty_scale"] > 0.0


def test_xgboost_feature_shape():
    model = XGBoostPhysicsResidualModel()
    props = get_oil_properties("heavy_fuel_oil")
    feats = model.build_features(
        u_current=0.1,
        v_current=0.2,
        u_wind=5.0,
        v_wind=5.0,
        windage=0.03,
        sst=20.0,
        duration_hours=24.0,
        observed_area_km2=10.0,
        aspect_ratio=1.5,
        oil_props=props
    )
    assert feats.shape == (1, 11)


def test_xgboost_save_and_load():
    model = XGBoostPhysicsResidualModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "xgb_dx_residual.json"))
        assert os.path.exists(os.path.join(tmpdir, "xgb_dy_residual.json"))
        assert os.path.exists(os.path.join(tmpdir, "xgb_uncertainty.json"))

        loaded_model = XGBoostPhysicsResidualModel(model_path=tmpdir)
        assert loaded_model.is_trained
        pred = loaded_model.predict_residual(
            u_current=0.0,
            v_current=0.0,
            u_wind=0.0,
            v_wind=0.0,
            windage=0.03,
            sst=25.0,
            duration_hours=6.0,
            observed_area_km2=1.0
        )
        assert "uncertainty_scale" in pred
