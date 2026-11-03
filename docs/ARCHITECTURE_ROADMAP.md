# Q-OPSEC Architecture Roadmap

## Executive summary

Q-OPSEC already has useful building blocks: independently deployable services, a YAML service catalog, Prometheus/Grafana, structured module tests, and some domain-oriented code (notably `confiability_service_v2`). It is not yet a consistent Clean Architecture/DDD design. The 2,170-line `orchestrator_linux.py` currently owns process supervision, HTTP API, proxying, pipeline business logic, job launching, filesystem log parsing, monitoring container startup, and mutable runtime state. Several services have duplicate/versioned directories and the cross-service request flow is tightly coupled to a fixed list of ports and bespoke payload transformations.

Recommendation: evolve incrementally as a modular monolith for shared orchestration/control-plane concerns while keeping security/domain services independently deployable. Treat `orchestrator_linux.py` as the composition root and HTTP adapter, not as the home of business logic. Keep every runtime service, worker, cache/broker, Prometheus and Grafana under its service catalog and managed by the orchestrator; PM2 should supervise only the orchestrator entrypoint.

## Current-state observations

- PM2 reports `q-opsec` online for 17 days, entrypoint `start_qopsec.sh`, cwd `/home/umbrel/projetos/daily`. The running process is currently `orchestrator_linux.py` from `/home/umbrel/projetos/Q-OPSEC`. Correct the PM2 cwd/config so restart and future deploys are reproducible.
- The orchestrator is a 2,170-line FastAPI script with global `CONFIG`/`STATE`, broad responsibilities, many hard-coded host addresses, and legacy branches. It contains endpoint-specific business transformations and an 11-step synchronous pipeline.
- `POST /run-pipeline` executes serial service calls and mutates the inbound dictionary throughout; a downstream error ends the flow. It also performs synchronous metrics HTTP calls in the request flow.
- `/monitoring/start` creates a detached thread, shells out to Docker with interpolated shell commands, removes existing monitoring containers, and returns success without observing readiness or exit status. Monitoring containers currently run independently, and Prometheus/Grafana are already present on ports 9091/3001.
- There are 13 configured services; training endpoints also start scripts directly using hard-coded paths and `Popen`, so jobs are not persisted or queryable after orchestrator restart.
- Cache/queue packages are absent in the project virtualenv. Docker already hosts Redis instances for other applications; do not reuse an unrelated Redis instance without an isolated database/ACL/namespace and explicit capacity/ownership. No Q-OPSEC broker is configured today.
- The whole-repository pytest collection command currently fails during collection due to import namespace collisions (`main`, `models`, `tests`), test imports that assume a module working directory, and an existing `MockLog.BoundLogger` failure. The focused orchestrator test suites pass: 10 passed.
- `services.yaml` contains a placeholder `CLASSIFY_API_KEY`; secrets must move to environment/secret storage, never be logged or cached.

## Proposed target architecture

### Bounded contexts

1. Security decisioning: context enrichment, risk assessment, trust/confidentiality, classification, RL policy. Define explicit request/result contracts and preserve explainability/version metadata.
2. Key lifecycle: handshake negotiation, KMS/key sessions, cryptographic operations, key delivery, validation. Keep key material out of job messages, cache, logs, and metrics; retain strict lifecycle/TTL semantics.
3. Orchestration/control plane: service catalog, health/status, pipeline coordination, run/job status, audit/trace correlation, monitoring lifecycle. This is the orchestrator's domain, not the security decision itself.
4. Data/model operations: dataset generation, model training, metrics and artifact registry. Training belongs on a durable worker queue; prediction stays synchronous unless product/API contracts explicitly change.

### Clean Architecture boundaries per Python service

- `domain/`: entities, value objects, domain services, domain exceptions; no FastAPI, HTTP clients, Redis, filesystem, SQL, Pydantic, or infrastructure imports.
- `application/`: use cases/commands/queries, ports (Protocols), orchestration policy, transaction boundaries.
- `adapters/inbound/`: FastAPI routers, CLI, worker handlers, request/response schemas.
- `adapters/outbound/`: HTTP clients, persistence, cache, queue, model/artifact access, clocks.
- `bootstrap.py` or `composition.py`: instantiate adapters and inject them into use cases; app entrypoint only wires dependencies and lifecycle.

Use DDD where the business rules justify it: explicit bounded contexts, ubiquitous language, value objects (request/session IDs, security levels, algorithm names, TTLs), aggregates around key/session lifecycle and pipeline/job lifecycle, and domain events for completed decisions/training. Avoid adding repositories/entities mechanically to stateless ML inference code.

### Orchestrator structure

Keep compatibility routes initially, delegating to small application services such as `ServiceLifecycle`, `PipelineCoordinator`, `TrainingJobs`, `MonitoringLifecycle`, and `ServiceCatalog`. Move direct Docker/process/HTTP/YAML/filesystem operations behind interfaces. Replace global state with an injected runtime container created in FastAPI lifespan and closed on shutdown. Replace hard-coded base URLs with validated catalog values. Run independent pipeline stages concurrently only after dependency edges and semantics are explicit; preserve ordering for policy decisions, handshake, key creation, encryption, delivery and validation.

### Cache policy

Start with a bounded TTL cache behind a `Cache` port. Good first candidates: service catalog/config snapshots, health/status probes (very short TTL), dataset lists and dataset previews (short TTL), immutable model metadata and static capability registries. Prefer HTTP cache headers/ETags when supported. Do not cache predictions until deterministic inputs, model/version identity, tenant/auth scope, and privacy retention are specified. Never cache secrets, plaintext, key material, ciphertext lifecycle state, user/identity payloads, or security decisions that could become stale and weaken a policy. Add explicit invalidation/versioned keys and cache hit/miss/eviction metrics. Current implementation uses a bounded, process-local TTL cache; Redis is a later adapter if multi-worker/shared invalidation needs arise.

### Asynchronous jobs

Use the included initial SQLite-backed job adapter for one-host durability and a single worker. Its interface is replaceable by Redis Streams/RQ/Celery or RabbitMQ when multiple workers, independent scaling, or delivery guarantees justify external infrastructure. Start with CPU-expensive training/dataset generation; keep the security-critical `/run-pipeline` synchronous until callers agree to an accepted `202 + job_id` contract and idempotency/recovery behavior is defined. Jobs must contain references/identifiers, not secrets or key material. Add bounded retries, max attempts, deduplication/idempotency key, cancellation, retention, and dead-letter behavior before broad production use. Current starter supports durable enqueue/status, one worker, startup recovery of interrupted jobs, and success/failure state; it does not yet include retries, cancellation, or a multi-worker lease.

### Monitoring and lifecycle ownership

Represent cache and worker as named managed components in `services.yaml`, just like application services. Orchestrator lifespan should start configured dependencies in topological startup order, perform readiness checks, and stop them in reverse order. Prometheus/Grafana remain orchestrator-managed. Start containers using argument arrays (no shell), validate config paths/images/ports, never remove an existing container implicitly, capture process handles/logs, and expose status/health/errors. Require explicit opt-in for container recreation. Add Prometheus metrics for job queue depth/age/status, cache hits/misses/evictions, pipeline duration/failures, dependency health and worker activity; avoid high-cardinality IDs and sensitive labels.

## Incremental implementation plan

1. Establish a reliable test baseline: test each Python service from its own cwd or use explicit package/import configuration; add per-service CI/test commands. Fix broad collection before a large refactor.
2. Correct PM2 ecosystem cwd/script and ensure `start_qopsec.sh` is the reproducible launch path. Confirm `pm2 restart q-opsec` launches only the orchestrator and it owns child lifecycle.
3. Extract orchestrator config, service catalog, and process/Docker lifecycle behind tested modules; keep endpoints and response contracts stable.
4. Add a lifespan-owned application container and request-scoped use-case dependencies; remove global mutable state carefully.
5. Split `run_pipeline` into explicit application stages with typed input/output contracts and dependency graph; add contract tests and correlation/event tracing.
6. Apply TTL cache to low-risk read endpoints first. Verify cold/hot behavior, expiration, invalidation, and safe bypass on errors.
7. Route training/dataset generation through durable job use case and worker managed by orchestrator; add job polling, idempotency, retention, and queue metrics. Add broker only when measured need warrants it.
8. Apply the same layer boundaries to services with clear domain complexity, starting with trust evaluation and key/session lifecycle. Retire duplicate `*_V2`/legacy directories only after traffic/config/tests prove migration complete.
9. Add architecture fitness tests: forbidden dependency directions, OpenAPI/contract compatibility, service catalog validation, and lifecycle integration tests.

## Acceptance criteria

- Orchestrator start/restart/stop is reproducible via PM2 with the correct project cwd; its health/status reports all supervised components.
- All runtime dependencies (including monitoring, cache, and workers) are declared in service config and can be started/stopped/status-checked through orchestrator APIs; no detached unmanaged threads or containers.
- No business rules in FastAPI routes or infrastructure adapters; domain/application unit tests run without network/Docker.
- Background work survives orchestrator restart, is observable by job ID, and does not put secrets/key material into persistence.
- Cache scope, TTL, invalidation and bypass behavior are tested; sensitive or stale security decisions are not cached.
- Per-service suites and contract tests pass, and monitoring exposes operational signals without high-cardinality/sensitive dimensions.

## Implemented initial building blocks

- `common/ttl_cache.py`: bounded in-process TTL cache with defensive copies and hit/miss statistics.
- `common/background_jobs.py`: SQLite durable queue/repository with one async worker loop, job status and restart recovery.
- `tests/test_background_jobs.py`: tests for expiry, bounded cache, defensive copies, job persistence/recovery, worker success and failure.

The job worker/cache are started and stopped by orchestrator startup/shutdown hooks. Prometheus/Grafana are declared in `services.yaml`, managed by `/services/{name}/start|stop|status` and `/start/all|stop/all`, and the compatibility `/monitoring/start` endpoint delegates to those lifecycle methods. `/architecture-metrics` exposes cache/job counters and gauges, and Prometheus scrapes it as a separate job. The currently deployed Prometheus/Grafana containers predate the new orchestrator ownership labels and are deliberately not taken over or recreated automatically; they remain running until an explicit, safe migration/recreation is approved. This avoids destructive changes to monitoring data or unowned containers. The declared definitions will be used when the names are absent.
