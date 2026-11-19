# Q-OPSEC — estado atual da arquitetura

Consolidado em 2026-11-18, após as rodadas de escala horizontal, resiliência, observabilidade e
deploy. Complementa `00-pendencias.md` (backlog com evidência) e `01-escala-horizontal.md`
(contratos e env vars).

## 1. Visão geral

O Q-OPSEC é um pipeline de 11 serviços que enriquece um pedido com contexto, avalia risco,
classifica, decide política por reinforcement learning, negocia algoritmos, emite chave, entrega,
cifra e valida o envio. O `orchestrator_linux.py` (FastAPI, porta 8090) é o plano de controle: sobe
os serviços, executa o pipeline de forma síncrona ou assíncrona, expõe métricas e serve o dashboard.

```
                        ┌──────────────────────────────────────────┐
   cliente ──HTTP──▶    │  orchestrator  :8090                     │
                        │  /run-pipeline        (síncrono)         │
                        │  /run-pipeline/async  (202 + job_id)     │
                        │  /pipeline/jobs/{id}  /jobs/stats        │
                        │  /architecture/status  /metrics  /       │
                        └───────┬──────────────────────┬───────────┘
                                │ publica job          │ executa steps
                                ▼                      ▼
                    ┌───────────────────┐   ┌──────────────────────────┐
                    │ RabbitMQ :15673   │   │ pipeline de 11 serviços  │
                    │ fila + retry      │   │ (ver §3)                 │
                    └───────────────────┘   └──────────────────────────┘
                                │                      │
                    ┌───────────▼──────────┐  ┌────────▼─────────────┐
                    │ Redis :6380          │  │ OTLP :4318           │
                    │ cache, idempotência, │  │ otel-collector       │
                    │ rate limit, sessões  │  └──────────────────────┘
                    └──────────────────────┘
```

## 2. Serviços e portas

| Serviço | Tipo | Porta | Papel |
| --- | --- | --- | --- |
| `orchestrator` | FastAPI | 8090 | plano de controle, pipeline, dashboard, métricas |
| `interceptor_api` | Spring | 8080 | entrada, encaminha para classificação |
| `context_api` | Spring | 65534 | enriquece com contexto e confiabilidade |
| `risk_service` | FastAPI | 8000 | predição de risco |
| `confiability_service` | FastAPI | 8083 | classificação de confiabilidade |
| `classification_agent` | FastAPI | 8088 | agente de classificação |
| `rl_engine` | FastAPI | 9009 | política por reinforcement learning |
| `handshake_negotiator` | FastAPI | 8001 | negociação de algoritmos |
| `kms_service` | FastAPI | 8002 | emissão e guarda de chaves |
| `key_destination_engine` | FastAPI | 8003 | entrega de chave ao destino |
| `crypto_module` | FastAPI | 8004 | cifra do payload |
| `validation_send_api` | Spring | 8005 | validação e envio à origem |

Infraestrutura: Redis 6380, RabbitMQ 15673, Prometheus 9091→9090, Grafana, otel-collector 4317/4318,
Postgres 5432 (primary), 5433 (réplica), 6432 (pgBouncer).

## 3. Pipeline

Ordem executada pelo orchestrator, com o endpoint de cada step:

1. `interceptor_api` — `/intercept` (8080)
2. `context_api` — `/context/enrich` (65534)
3. `risk_service` — `/predict/` (8000)
4. `confiability_service` — `/classify` (8083)
5. `classification_agent` — `/api/v1/predict` (8088)
6. `rl_engine` — `/act` (9009)
7. `handshake_negotiator` — `/handshake` (8001)
8. `kms_service` — `/keys` (8002)
9. `key_destination_engine` — `/deliver` (8003)
10. `crypto_module` — `/encrypt` (8004)
11. `validation_send_api` — `/validation/send` (8005)

Cada step tem breaker próprio (`PIPELINE_BREAKERS`) e retry com backoff (`PIPELINE_RETRY`). Um step
que falha vira `error` e interrompe o pipeline; `failed_steps` lista o que quebrou.

## 4. Caminho assíncrono

`POST /run-pipeline/async` responde `202` com `job_id` e `poll_url`. A idempotência vem do header
`Idempotency-Key`, com fallback para `request:<id>` ou hash do payload, TTL de 24 h. O job vai para o
RabbitMQ e é consumido por `QOPSEC_RABBITMQ_WORKERS` (default 4) workers.

Falha de job não volta para a fila: `requeue=False`, contador `attempts` persistido, `max_attempts`
(default 3) e republicação para retry; ao esgotar, marca `failed` e dá ack. `recover_stale` no
startup marca jobs `running` órfãos como `failed` após `QOPSEC_JOB_STALE_SECONDS` (300).

## 5. Resiliência

- `common/resilience.py`: `CircuitBreaker` (threshold + half-open), `BreakerRegistry`, `RetryPolicy`
  com backoff exponencial e jitter, `ResilientHTTPClient` com breaker por host e retry em
  502/503/504.
- Java: breaker sem dependência nova, copiado em `interceptor_api`, `context_api` e
  `validation_send_api`, com `tests/test_java_breaker_parity.py` impedindo divergência.
- `interceptor_api`: `forwardScheduler` (pool 4) com backoff configurável.
- `context_api`: `Retry.backoff` do Reactor nos clientes; Tomcat dimensionado para o `block()`
  intencional (`threads.max=200`, `min-spare=20`, `accept-count=200`).
- `validation_send_api`: HTTP fora da transação (`NOT_SUPPORTED` no forward, `REQUIRES_NEW` nas
  atualizações de status), `open-in-view: false`.

## 6. Rate limit e cache

`common/rate_limit.py` aplica limite distribuído em `/run-pipeline*`: `RedisRateLimiter`
(INCR+EXPIRE por janela), `InMemoryRateLimiter` e `FailoverRateLimiter` (Redis quando disponível,
memória como fallback). Resposta `429` com headers `X-RateLimit-*`; identidade é hash de API key ou
IP, nunca credencial em claro. Limite padrão 600/min.

Cache: `common/ttl_cache.py` (`TTLCache`, `RedisTTLCache`, `get_or_set`, `file_cache_key`) usado em
hardware summary e benchmark do KMS, key_context do crypto_module, tracker de entregas do KDE,
políticas do handshake e registry de modelos.

## 7. Observabilidade

- `/architecture/status` devolve `pipeline`, `resilience`, `rate_limit` e `tracing`.
- Métricas Prometheus: `qopsec_pipeline_jobs_*`, `qopsec_pipeline_requests_total`,
  `qopsec_circuit_breakers_open`, `qopsec_rate_limit_decisions_total`.
- Tracing OTLP para o `qopsec_otel_collector` (4318), sampler por `QOPSEC_TRACE_SAMPLE_RATIO`;
  spans do orchestrator e das chamadas httpx aos serviços.
- Dashboard em `/` com abas Services, Pipeline, AI Predictions, Trace, Flow, Timeline, Active
  Requests, Logs, Metrics, Datasets, Monitoring e Redis/Queue.

## 8. Dados

`kms_service/database.py` separa escrita e leitura: `QOPSEC_DATABASE_URL` (via pgBouncer 6432) e
`QOPSEC_DATABASE_READ_URL` (réplica 5433), com pool (`pool_size`, `max_overflow`, `pool_pre_ping`,
`pool_recycle`) e `pool_status()` reportando `replica_configured`.

`deploy/postgres-replica/setup.sh` provisiona o conjunto (modos `apply`, `--check`, `--teardown`) e
valida `pg_is_in_recovery()`, replicação de escrita e `select 1` pelo bouncer.

## 9. Deploy

- Imagens: `deploy/Dockerfile.orchestrator` (base `python:3.13-slim`, instala `requirements.txt` +
  `requirements-architecture.txt`) e `deploy/Dockerfile.kms`, com `.dockerignore`.
- Compose: `deploy/docker-compose.yml` (redis, rabbitmq, kms, orchestrator, prometheus, grafana).
- Kubernetes: `deploy/k8s/00..30` — namespace, ConfigMap+Secret, StatefulSets de Redis e RabbitMQ,
  Deployments de KMS e orchestrator com probes, HPA v2, PDB, prometheus-adapter com métrica
  `External` de profundidade de fila, Prometheus com RBAC e scrape dos pods, e Ingress.
- `tests/test_k8s_manifests.py` valida os 29 documentos estaticamente.

## 10. Verificação

| Verificação | Resultado |
| --- | --- |
| `pytest tests/ -q` | 150 passed |
| `mvn -B test` (validation) | 12 passed |
| `mvn -B test` (context) | 17 passed |
| `mvn -B test` (interceptor) | 11 passed |
| Pipeline assíncrono | 11 steps, `failed_steps: []` |
| Carga JMeter (100 req) | 100% 202, p95 25 ms, fila drenada |
| Réplica Postgres | `pg_is_in_recovery=t`, escrita replicada |
| Rate limit | `allowed, allowed, denied` + failover `memory` |
| Tracing | spans no coletor, `tracing.enabled: true` |

## 11. O que falta

Apenas o apply real dos manifests num cluster (KB-005). Não há `kubectl`, `kind` nem `k3d` nesta
máquina, e baixar binário da internet exige autorização explícita. Todo o resto do backlog
KB-001..KB-014 está concluído.
