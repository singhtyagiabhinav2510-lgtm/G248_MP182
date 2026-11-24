
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction():
    payload = {
        "price_rs": 40,
        "temp_c": 30,
        "footfall_index": 0.8,
        "competitor_shops": 3,
        "shelf_hours": 5,
        "orders_previous_day": 200,
        "day_type": "weekday",
        "outlet": "outlet_1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_loaves_sold" in response.json()


def test_invalid_shelf_hours():
    payload = {
        "price_rs": 40,
        "temp_c": 30,
        "footfall_index": 0.8,
        "competitor_shops": 3,
        "shelf_hours": 1,
        "orders_previous_day": 200,
        "day_type": "weekday",
        "outlet": "outlet_1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_previous_day_orders():
    payload = {
        "price_rs": 40,
        "temp_c": 30,
        "footfall_index": 0.8,
        "competitor_shops": 3,
        "shelf_hours": 5,
        "orders_previous_day": 851,
        "day_type": "weekday",
        "outlet": "outlet_1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
