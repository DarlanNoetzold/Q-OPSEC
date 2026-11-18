# Q-OPSEC deployment

## Local (docker compose)

```bash
cd deploy
export QOPSEC_REDIS_PASSWORD='<REDIS_PASSWORD>'
export QOPSEC_RABBITMQ_USER='<RABBITMQ_USER>'
export QOPSEC_RABBITMQ_PASSWORD='<RABBITMQ_PASSWORD>'
docker compose up -d --build
```

Endpoints: orchestrator `http://127.0.0.1:8081`, Prometheus `http://127.0.0.1:9091`, Grafana `http://127.0.0.1:3001`.

## Kubernetes

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-config.yaml
kubectl apply -f k8s/10-redis.yaml -f k8s/11-rabbitmq.yaml
kubectl apply -f k8s/20-kms.yaml -f k8s/21-orchestrator.yaml
kubectl apply -f k8s/22-prometheus-adapter.yaml
kubectl apply -f k8s/30-ingress.yaml
```

The orchestrator HPA scales on CPU and on `qopsec_pipeline_queue_depth` (external metric served by
`22-prometheus-adapter.yaml` from `qopsec_pipeline_jobs_queued`). The adapter points at
`http://prometheus.qopsec.svc:9090/`, so a Prometheus Service named `prometheus` must exist in the
`qopsec` namespace.
```

Fill `k8s/01-config.yaml` (`qopsec-secrets`) with the real values before applying, or create the secret from the CLI:

```bash
kubectl -n qopsec create secret generic qopsec-secrets \
  --from-literal=QOPSEC_REDIS_PASSWORD='<REDIS_PASSWORD>' \
  --from-literal=QOPSEC_RABBITMQ_USER='<RABBITMQ_USER>' \
  --from-literal=QOPSEC_RABBITMQ_PASSWORD='<RABBITMQ_PASSWORD>' \
  --from-literal=QOPSEC_REDIS_URL='redis://:<REDIS_PASSWORD>@qopsec-redis:6379/0' \
  --from-literal=QOPSEC_RABBITMQ_URL='amqp://<RABBITMQ_USER>:<RABBITMQ_PASSWORD>@qopsec-rabbitmq:5672/'
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QOPSEC_REDIS_URL` | — | Shared cache, KMS sessions, idempotency, rate limit |
| `QOPSEC_RABBITMQ_URL` | — | Background job queue |
| `QOPSEC_KMS_SESSION_BACKEND` | `auto` | `redis`, `memory` or `auto` |
| `QOPSEC_KMS_SESSION_PREFIX` | `qopsec:kms:session:` | Session key prefix |
| `QOPSEC_RATE_LIMIT_REQUESTS` | `600` | Requests per window |
| `QOPSEC_RATE_LIMIT_WINDOW_SECONDS` | `60` | Window size |
| `QOPSEC_RATE_LIMIT_PATHS` | `/run-pipeline` | Limited path prefixes |
| `QOPSEC_PIPELINE_IDEMPOTENCY_TTL_SECONDS` | `86400` | Idempotency key TTL |
| `QOPSEC_BREAKER_FAILURE_THRESHOLD` | `5` | Failures before a breaker opens |
| `QOPSEC_BREAKER_RESET_SECONDS` | `30` | Half-open probe delay |
| `QOPSEC_RETRY_ATTEMPTS` | `3` | Retry attempts per upstream call |
| `QOPSEC_RETRY_BASE_DELAY` | `0.2` | Backoff base (seconds) |
| `QOPSEC_RETRY_MAX_DELAY` | `5` | Backoff ceiling (seconds) |
| `QOPSEC_TRACING_ENABLED` | `auto` | `auto`, `true` or `false` |
| `QOPSEC_OTLP_ENDPOINT` | — | OTLP HTTP collector endpoint |
| `QOPSEC_TRACE_SAMPLE_RATIO` | `0.1` | Trace sampling ratio |
| `QOPSEC_DATABASE_URL` | local postgres | Write database (KMS v1) |
| `QOPSEC_DATABASE_READ_URL` | same as write | Read replica |
| `QOPSEC_DB_POOL_SIZE` | `10` | SQLAlchemy pool size |
| `QOPSEC_DB_MAX_OVERFLOW` | `20` | Pool overflow |
| `QOPSEC_DB_POOL_RECYCLE_SECONDS` | `1800` | Connection recycle |

## Health and metrics

- Orchestrator: `GET /architecture/status`, `GET /architecture-metrics`, `GET /metrics`
- KMS: `GET /health` (includes `storage_backend` and `storage_healthy`)
- Async pipeline: `POST /run-pipeline/async` with `Idempotency-Key`, poll `GET /pipeline/jobs/{job_id}`
