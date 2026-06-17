from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from src.api.routes import prediction, metrics, dataset_info, models
from src.common.logger import logger

app = FastAPI(
    title="Fraud Detection Model API",
    description="""
## 🕵️ Fraud Detection Model API

API para servir modelos de detecção de fraude, incluindo:

- **Prediction**: endpoints de predição/inferência
- **Metrics**: métricas do modelo e/ou do serviço
- **Dataset**: informações do dataset (se disponível)
- **Models**: informações/versões de modelos

### Documentação
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
""",
    version="1.0.0",
    contact={
        "name": "Fraud Detection Team",
    },
    openapi_tags=[
        {"name": "Prediction", "description": "Inferência / predições do modelo"},
        {"name": "Metrics", "description": "Métricas do modelo/serviço"},
        {"name": "Dataset", "description": "Informações do dataset (quando disponível)"},
        {"name": "Models", "description": "Informações e versionamento dos modelos"},
        {"name": "Health", "description": "Health checks e endpoints operacionais"},
    ],
)

# Instrumentação para o Prometheus
Instrumentator().instrument(app).expose(app)

try:
    from src.api.utils.logger import setup_logging
    setup_logging()
    logger.info("setup_logging() executed")
except Exception:
    logger.warning("setup_logging not available or failed; continuing without it.")


@app.get(
    "/",
    tags=["Health"],
    summary="Links rápidos para documentação",
    description="Endpoint raiz com links para Swagger/ReDoc/OpenAPI e lista de rotas principais.",
)
async def root():
    return {
        "service": "Fraud Detection Model API",
        "version": "1.0.0",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "routes": {
            "prediction": "/predict",
            "metrics": "/metrics",
            "dataset": "/dataset",
            "models": "/models",
            "health": "/health",
        },
    }


app.include_router(prediction.router, prefix="/predict", tags=["Prediction"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])

try:
    app.include_router(dataset_info.router, prefix="/dataset", tags=["Dataset"])
except Exception:
    logger.warning("dataset_info router not included (missing).")

app.include_router(models.router, prefix="/models", tags=["Models"])


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Verifica se o serviço está respondendo.",
    responses={
        200: {
            "description": "Serviço OK",
            "content": {"application/json": {"example": {"status": "ok"}}},
        }
    },
)
async def health_check():
    return {"status": "ok"}