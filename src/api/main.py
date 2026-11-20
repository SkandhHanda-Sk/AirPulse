"""AirPulse FastAPI prediction service.

Serves the trained Persistence-Anchored Residual HistGradientBoosting model
for next-hour PM2.5 forecasting.

Endpoints:
    GET  /health   - Health check and model readiness
    POST /predict  - Single next-hour PM2.5 prediction
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.model_loader import get_model_source, load_model, predict
from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse

app = FastAPI(
    title="AirPulse PM2.5 Forecasting API",
    description="Next-hour PM2.5 concentration predictions for Delhi monitoring stations.",
    version="1.0.0",
)

_model = None


def _get_model():
    """Lazy-load the @champion model from MLflow Model Registry on first request."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Check API health and model readiness."""
    model_loaded = False
    model_source = None
    try:
        _get_model()
        model_loaded = True
        model_source = get_model_source()
    except Exception:  # noqa: BLE001 - model may not be trained yet
        model_loaded = False
    return HealthResponse(model_loaded=model_loaded, model_source=model_source)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_next_hour(request: PredictionRequest) -> PredictionResponse:
    """Predict next-hour PM2.5 concentration for a monitoring station.

    Accepts all 18 model features and returns the model's next-hour PM2.5
    estimate along with the anchor (current) value and residual correction.
    """
    try:
        model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    features_df = request.to_dataframe()
    predictions = predict(model, features_df)
    predicted_value = predictions[0]
    anchor_value = float(features_df["pm25_at_origin"].iloc[0])

    return PredictionResponse(
        station_id=request.station_id,
        predicted_pm25_next_hour=round(predicted_value, 4),
        anchor_pm25_at_origin=round(anchor_value, 4),
        residual=round(predicted_value - anchor_value, 4),
    )