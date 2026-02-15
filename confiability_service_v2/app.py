"""
Confiability Service V2 - Trust Engine Only
"""
from flask import Flask, jsonify
from flask_smorest import Api
import os
import sys

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar Confiability Sevice V2 Controller
from api.v2.trust_controller import trust_v2_bp

# Importar bootstrap
from bootstrap_v2 import initialize_trust_engine


def ensure_dirs():
    """Cria diretórios necessários"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def create_app():
    """Factory para criar a aplicação Flask"""
    ensure_dirs()

    # Inicializar Confiability Sevice V2
    try:
        initialize_trust_engine()
        print("Confiability Sevice V2 initialized")
    except Exception as e:
        print(f"Confiability Sevice V2 initialization failed: {e}")
        import traceback
        traceback.print_exc()

    app = Flask(__name__)

    # ========== CONFIGURAÇÃO DO SWAGGER/OPENAPI ==========
    app.config["API_TITLE"] = "Confiability Sevice V2 API"
    app.config["API_VERSION"] = "2.0.0"
    app.config["OPENAPI_VERSION"] = "3.0.3"
    app.config["OPENAPI_URL_PREFIX"] = "/"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
    app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    app.config["OPENAPI_REDOC_PATH"] = "/redoc"
    app.config["OPENAPI_REDOC_URL"] = "https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"
    app.config["API_SPEC_OPTIONS"] = {
        "info": {
            "description": """
# Confiability Sevice V2 - Information Trust Evaluation API

API completa para avaliação contextual de confiança em informações.

## Recursos Principais
- **Avaliação de Confiança**: Análise multi-dimensional de trustworthiness
- **Sinais Contextuais**: Temporal, semântico, anomalia, consistência
- **Histórico**: Rastreamento de entidades e fontes
- **Estatísticas**: Métricas agregadas do sistema

## Fluxo de Uso
1. Envie dados para `/api/v2/trust/evaluate`
2. Receba score de confiança (0-1) e nível de risco
3. Consulte histórico e estatísticas conforme necessário
            """,
            "contact": {
                "name": "Trust Engine Team",
                "email": "support@trustengine.com"
            },
            "license": {
                "name": "Apache 2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
            }
        },
        "servers": [
            {"url": "http://localhost:8083", "description": "Desenvolvimento Local"},
            {"url": "http://0.0.0.0:8083", "description": "Servidor Local"},
        ],
        "tags": [
            {"name": "Trust Evaluation", "description": "Endpoints de avaliação de confiança"},
            {"name": "Health & Config", "description": "Status e configuração do sistema"},
            {"name": "History", "description": "Histórico de avaliações"},
            {"name": "Statistics", "description": "Estatísticas e métricas"}
        ]
    }

    # Inicializa a API com Swagger
    api = Api(app)

    # Registrar blueprint do Confiability Sevice V2
    api.register_blueprint(trust_v2_bp)       # /api/v2/trust/*

    # Health check geral
    @app.get("/health")
    def health():
        """Health check geral da aplicação"""
        return jsonify({
            "status": "ok",
            "version": "2.0.0",
            "services": {
                "trust_engine_v2": "/api/v2/trust/*"
            }
        }), 200

    @app.get("/")
    def root():
        """Endpoint raiz com informações da API"""
        return jsonify({
            "service": "Confiability Sevice V2",
            "version": "2.0.0",
            "description": "Information Trust Engine - Contextual Trustworthiness Evaluation",
            "documentation": {
                "swagger_ui": "/swagger-ui",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json"
            },
            "endpoints": {
                "health": "GET /health",
                "trust_v2": {
                    "evaluate": "POST /api/v2/trust/evaluate",
                    "health": "GET /api/v2/trust/health",
                    "config": "GET /api/v2/trust/config",
                    "signals": "GET /api/v2/trust/signals",
                    "stats": "GET /api/v2/trust/stats",
                    "history_entity": "GET /api/v2/trust/history/<entity_id>",
                    "history_source": "GET /api/v2/trust/history/source/<source_id>"
                }
            }
        }), 200

    return app


# Criar aplicação
app = create_app()


if __name__ == "__main__":
    print("=" * 70)
    print("Confiability Sevice V2 - Starting...")
    print("=" * 70)
    print("Service:")
    print("    Confiability Sevice V2: /api/v2/trust/*")
    print("=" * 70)
    print("Server: http://0.0.0.0:8083")
    print("Swagger UI: http://0.0.0.0:8083/swagger-ui")
    print("ReDoc: http://0.0.0.0:8083/redoc")
    print("OpenAPI Spec: http://0.0.0.0:8083/openapi.json")
    print("=" * 70)
    print("Quick Health Checks:")
    print("   curl http://localhost:8083/health")
    print("   curl http://localhost:8083/api/v2/trust/health")
    print("=" * 70)
    print("\nTest Evaluation:")
    print('   curl -X POST http://localhost:8083/api/v2/trust/evaluate \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d "{\\"payload\\": {\\"test\\": \\"data\\"}, \\"metadata\\": {\\"source_id\\": \\"test\\"}}"')
    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=8083,
        debug=False,
        threaded=True
    )
