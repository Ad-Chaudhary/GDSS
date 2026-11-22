import os
import joblib
import numpy as np
import pandas as pd
import pytest

from app import evaluate_cgwb_status

MODEL_PATH = "groundwater_model.pkl"


@pytest.fixture(scope="module")
def artifacts():
    assert os.path.exists(MODEL_PATH), f"Model file '{MODEL_PATH}' not found. Run train_model.py first."
    return joblib.load(MODEL_PATH)


class TestGroundwaterModel:

    def test_artifacts_loading(self, artifacts):
        assert "model" in artifacts
        assert "feature_scaler" in artifacts
        assert "target_scaler" in artifacts

    def test_soe_calculation(self):
        extraction = 480.0
        recharge = 620.0
        expected_soe = (480.0 / 620.0) * 100.0
        
        assert pytest.approx(expected_soe, 0.01) == 77.42

    @pytest.mark.parametrize(
        "soe, expected_status",
        [
            (30.0, "SAFE"),
            (70.0, "SAFE"),
            (77.42, "SEMI_CRITICAL"),
            (90.0, "SEMI_CRITICAL"),
            (95.0, "CRITICAL"),
            (100.0, "CRITICAL"),
            (145.16, "OVER_EXPLOITED"),
            (520.0, "OVER_EXPLOITED"),
        ],
    )
    def test_cgwb_status_classification(self, soe, expected_status):
        status, _, _, _ = evaluate_cgwb_status(soe)
        assert status == expected_status

    def test_model_prediction_physical_bounds(self, artifacts):
        model = artifacts["model"]
        feature_scaler = artifacts["feature_scaler"]
        target_scaler = artifacts["target_scaler"]

        input_data = pd.DataFrame(
            [[185.50, 50.00, 620.00]],
            columns=["precipitation", "extraction", "recharge"],
        )

        input_scaled = feature_scaler.transform(input_data)
        pred_scaled = model.predict(input_scaled).reshape(-1, 1)
        predicted_depth = target_scaler.inverse_transform(pred_scaled)[0][0]

        assert 2.0 <= predicted_depth <= 50.0, f"Predicted depth {predicted_depth} m is physically unrealistic!"

    def test_prediction_trend_responsiveness(self, artifacts):
        model = artifacts["model"]
        feature_scaler = artifacts["feature_scaler"]
        target_scaler = artifacts["target_scaler"]

        safe_input = pd.DataFrame([[200.0, 50.0, 600.0]], columns=["precipitation", "extraction", "recharge"])
        stress_input = pd.DataFrame([[0.0, 1000.0, 200.0]], columns=["precipitation", "extraction", "recharge"])

        safe_scaled = feature_scaler.transform(safe_input)
        stress_scaled = feature_scaler.transform(stress_input)

        depth_safe = target_scaler.inverse_transform(model.predict(safe_scaled).reshape(-1, 1))[0][0]
        depth_stress = target_scaler.inverse_transform(model.predict(stress_scaled).reshape(-1, 1))[0][0]

        assert depth_stress > depth_safe, "High extraction stress scenario should result in a deeper water table!"

    def test_classifier_proba_feature_mapping(self, artifacts):
        if "clf" in artifacts:
            from ml_models import get_prediction_features, predict_risk_proba
            clf = artifacts["clf"]
            feature_df, _ = get_prediction_features("DWLR_TEST", 25.0, 300.0, 350.0)
            proba = predict_risk_proba(clf, feature_df)
            assert proba.shape[1] == 4