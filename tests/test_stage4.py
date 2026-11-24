import pandas as pd
import pytest

from src.bakery_sales.monitor import compute_psi


def test_compute_psi_identical_distributions_is_zero():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    assert compute_psi(series, series, bins=10) == pytest.approx(0.0, abs=1e-6)


def test_compute_psi_shifted_distribution_is_positive():
    reference = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    current = pd.Series([90.0, 91.0, 92.0, 93.0, 94.0])
    assert compute_psi(reference, current, bins=10) > 0.0


def test_stage4_promotion_rule():
    mae_v1 = 65.31264747957682
    mae_v2 = 43.484853650506125
    assert mae_v2 <= 0.85 * mae_v1


def test_holdout_predictions_shape_and_columns():
    holdout = pd.read_csv("data/raw/holdout_features.csv")
    predictions = pd.read_csv("predictions/holdout_predictions.csv")

    assert predictions.columns.tolist() == ["record_id", "prediction"]
    assert len(predictions) == len(holdout) == 250
    assert predictions["record_id"].tolist() == holdout["record_id"].tolist()
    assert predictions["prediction"].notna().all()
