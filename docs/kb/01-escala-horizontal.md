# Escala horizontal — o que existe hoje

Registro do que foi implementado em 2026-11-18 para tirar o Q-OPSEC do modelo de réplica única.
Serve como referência antes de mexer em qualquer um desses pontos.

## Estado por camada

| Camada | Antes | Agora | Arquivo |
| --- | --- | --- | --- |
| Sessões do KMS | dict no processo | Redis com TTL e índice por `expires_at` | `kms_service_v2/storage.py` |
| Pipeline | 11 chamadas síncronas | `POST /run-pipeline/async` (202) + fila RabbitMQ + idempotência | `orchestrator_linux.py` |
| Resiliência Python | sem breaker | breaker por host + retry com backoff e jitter | `common/resilience.py` |
| Rate limit | dict local | Redis com fallback para memória | `common/rate_limit.py` |
| Tracing | ausente | OTel opcional com no-op | `common/tracing.py` |
| Postgres (v1) | URL fixa no código | env + pool + réplica de leitura | `kms_service/database.py` |
| Deploy | só Dockerfile do KMS | Dockerfiles, compose, k8s com HPA/PDB/probes | `deploy/` |
| Java | `Thread.sleep`, `runAsync` | scheduler com backoff + breaker | `*/resilience/`, `InterceptService` |

## Contratos que não podem quebrar

- `POST /run-pipeline` continua síncrono e com o mesmo corpo; o assíncrono é aditivo.
- `Idempotency-Key` é opcional: sem header, a chave vem de `request_id` ou do hash do payload.
- `GET /pipeline/jobs/{id}` devolve 404 para job inexistente e 503 quando a fila não está disponível.
- `/health` do KMS nunca derruba o processo por Redis fora: reporta `degraded`.
- Resultado de job passa por `_redact_pipeline_result`; `key_material`, `api_key`, `secret`, `token`
  e `password` nunca são persistidos na fila.

## Variáveis de ambiente

Tabela completa em `deploy/README.md`. As que mudam comportamento em runtime:

| Variável | Default | Efeito |
| --- | --- | --- |
| `QOPSEC_KMS_SESSION_BACKEND` | `auto` | `redis` falha rápido, `auto` cai para memória |
| `QOPSEC_RATE_LIMIT_REQUESTS` | `600` | requisições por janela |
| `QOPSEC_RATE_LIMIT_PATHS` | `/run-pipeline` | prefixos limitados |
| `QOPSEC_BREAKER_FAILURE_THRESHOLD` | `5` | falhas antes de abrir |
| `QOPSEC_RETRY_ATTEMPTS` | `3` | tentativas por chamada |
| `QOPSEC_TRACING_ENABLED` | `auto` | `true` exige o SDK instalado |

## Como validar rápido

```bash
# Python
./qopsec_env/bin/python -m pytest -q tests/test_background_jobs.py
(cd kms_service_v2 && ../qopsec_env/bin/python -m pytest -q test_storage_regression.py test_key_manager_service.py)

# Java (maven em container; context_api precisa de --network host para o Postgres)
docker run --rm -v $PWD:/src -v qopsec-m2:/root/.m2 -w /src/interceptor_api \
  maven:3.9-eclipse-temurin-17 mvn -B test

# Deploy
(cd deploy && QOPSEC_REDIS_PASSWORD=x QOPSEC_RABBITMQ_USER=u QOPSEC_RABBITMQ_PASSWORD=y docker compose config -q)

# Runtime
curl -s http://127.0.0.1:8090/architecture/status
curl -s http://127.0.0.1:8090/architecture-metrics | grep -E "qopsec_(pipeline|circuit|rate_limit)"
```

## Limites conhecidos

- O HPA por fila depende do Prometheus Adapter e de um Prometheus no namespace (ver KB-003).
- O breaker é por processo: com N réplicas, cada uma abre o seu.
- A fila é RabbitMQ de um nó nos manifests; para HA é preciso quorum queue com 3 réplicas.
- O rate limit usa janela fixa: rajadas na virada da janela passam pelo dobro do limite.
