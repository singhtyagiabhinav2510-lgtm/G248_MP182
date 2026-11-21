import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from src.bakery_sales.preprocess import clean_data, get_preprocessor

def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "day_type": ["weekday", "weekday", "weekend"],
        "loaves_sold": [100, 100, 200]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2

def test_clean_data_normalizes_day_type():
    df = pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R4"],
        "day_type": [" fes. ", "wee.", "WEEKDAY", "Festival"],
        "loaves_sold": [100, 200, 300, 400]
    })
    cleaned = clean_data(df)
    expected = ["festival", "weekend", "weekday", "festival"]
    assert list(cleaned["day_type"]) == expected

def test_clean_data_fixes_target_typo():
    df = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "day_type": ["weekday", "weekday"],
        "loaves_sold": [3400.0, 340.0]
    })
    cleaned = clean_data(df, is_training=True)
    assert cleaned.loc[0, "loaves_sold"] == 340.0
    assert cleaned.loc[1, "loaves_sold"] == 340.0

def test_clean_data_drops_closing_register():
    df = pd.DataFrame({
        "record_id": ["R1"],
        "day_type": ["weekday"],
        "loaves_in_closing_register": [320],
        "loaves_sold": [320]
    })
    cleaned = clean_data(df)
    assert "loaves_in_closing_register" not in cleaned.columns

def test_pipeline_transforms_and_predicts():
    num_features = ["price_rs", "temp_c", "footfall_index", "competitor_shops", "shelf_hours", "orders_previous_day"]
    cat_features = ["day_type", "outlet"]

    df = pd.DataFrame({
        "price_rs": [40, 50, 60],
        "temp_c": [20.0, 25.0, 30.0],
        "footfall_index": [50, 60, 70],
        "competitor_shops": [2, 3, 4],
        "shelf_hours": [8.0, 10.0, 12.0],
        "orders_previous_day": [200, 300, 400],
        "day_type": ["weekday", "weekend", "festival"],
        "outlet": ["Station Road", "Mall Road", "College Gate"],
        "loaves_sold": [300, 350, 400]
    })

    pipe = Pipeline([
        ("preprocessor", get_preprocessor(num_features, cat_features)),
        ("regressor", Ridge())
    ])

    X = df[num_features + cat_features]
    y = df["loaves_sold"]

    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 3
    assert not np.isnan(preds).any()