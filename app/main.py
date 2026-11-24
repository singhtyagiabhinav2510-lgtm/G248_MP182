
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="MP182 Bakery Sales Prediction API")


class PredictionInput(BaseModel):
    price_rs: float
    temp_c: float
    footfall_index: float
    competitor_shops: int
    shelf_hours: float = Field(ge=2)
    orders_previous_day: int = Field(le=850)
    day_type: str
    outlet: str



class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "event": record.getMessage()
        }

        if hasattr(record, "prediction"):
            log_data["predicted_loaves_sold"] = record.prediction

        return json.dumps(log_data)



logger = logging.getLogger("prediction_api")
logger.setLevel(logging.INFO)
logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.propagate = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = float(model.predict(input_data)[0])

    logger.info("prediction", extra={"prediction": prediction})

    return {"predicted_loaves_sold": prediction}
