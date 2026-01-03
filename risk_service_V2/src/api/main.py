from fastapi import FastAPI
from src.api.routes import prediction, metrics, dataset_info
from src.api.utils.logger import setup_logging

app = FastAPI(
    title="Fraud Detection Model API",
    description="API for fraud detection models: prediction, metrics, and dataset info",
    version="1.0.0",
)

# Setup logging
setup_logging()

# Include routers
app.include_router(prediction.router, prefix="/predict", tags=["Prediction"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
app.include_router(dataset_info.router, prefix="/dataset", tags=["Dataset"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}