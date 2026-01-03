from fastapi import FastAPI
from src.api.routes import prediction, metrics, dataset_info, models
from src.common.logger import logger

app = FastAPI(
    title="Fraud Detection Model API",
    description="API for fraud detection models: prediction, metrics, and dataset info",
    version="1.0.0",
)

try:
    from src.api.utils.logger import setup_logging
    setup_logging()
    logger.info("setup_logging() executed")
except Exception:
    logger.warning("setup_logging not available or failed; continuing without it.")

app.include_router(prediction.router, prefix="/predict", tags=["Prediction"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])

try:
    app.include_router(dataset_info.router, prefix="/dataset", tags=["Dataset"])
except Exception:
    logger.warning("dataset_info router not included (missing).")

app.include_router(models.router, prefix="/models", tags=["Models"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}