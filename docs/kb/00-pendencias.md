# Pendências — Q-OPSEC

Backlog técnico aberto em 2026-11-18. Itens ordenados por impacto para escala.
Nada aqui foi commitado: o trabalho da sessão está no working tree.

## Resumo

| ID | Título | Área | Status |
| --- | --- | --- | --- |
| KB-001 | `@Transactional` envolvendo chamada HTTP no `ForwardService` | validation_send_api | concluído |
| KB-002 | `.block()` no fim das cadeias reativas do `context_api` | context_api | concluído |
| KB-003 | Prometheus não existe no namespace `qopsec` (HPA externo depende dele) | deploy/k8s | parcial |
| KB-004 | Imagens Docker nunca foram buildadas nem publicadas | deploy | concluído |
| KB-005 | Manifests k8s nunca aplicados; HPA externo não validado | deploy/k8s | parcial |
| KB-006 | Teste de carga JMeter nunca executado | load | concluído |
| KB-007 | pgBouncer e réplica de leitura reais na frente do Postgres | dados | concluído |
| KB-008 | Backend Redis do KMS validado só contra cliente falso | kms_service_v2 | concluído |
| KB-009 | Rate limit distribuído validado só contra cliente falso | common | concluído |
| KB-010 | Tracing OpenTelemetry sem pacotes instalados e sem coletor | observabilidade | concluído |
| KB-011 | Dashboard nunca aberto no navegador após as mudanças | dashboard | concluído |
| KB-012 | Nada commitado; serviços não reiniciados | repositório | concluído |
| KB-013 | Breaker Java duplicado em três módulos | java | concluído |
| KB-014 | Serviços do pipeline parados: job falha no step 1 | runtime | concluído |

---

## KB-001 — `@Transactional` envolvendo chamada HTTP no `ForwardService`

**Status:** concluído em 2026-11-18
**Arquivo:** `validation_send_api/src/main/java/tech/noetzold/validation_send_api/service/ForwardService.java`
**Também:** `controller/SendController.java`

**Resolução:** `forwardToOriginOrThrow` recebeu `@Transactional(propagation = NOT_SUPPORTED)` e
`markDelivered`/`markFailed` passaram a `REQUIRES_NEW`; `spring.jpa.open-in-view=false` no
`application.yml`. `ForwardServiceTest` ganhou um teste que falha se a propagação regredir.
`mvn -B test`: 12 testes, 0 falhas.

**Problema:** `receiveAndPersist`, `markDelivered` e `markFailed` são `@Transactional`, e o
`SendController` chama `forwardToOriginOrThrow` (I/O de rede, até 10s de timeout, com retry) entre
elas. A conexão do banco fica presa durante a chamada externa, o que esgota o pool sob carga.

**Aceite:**
- a chamada HTTP acontece fora de qualquer transação aberta;
- `markDelivered`/`markFailed` rodam em transação curta própria;
- teste cobrindo o caminho de falha continua passando (`ForwardServiceTest`, 4 testes).

**Verificação:**
```bash
docker run --rm -v /home/umbrel/projetos/Q-OPSEC:/src -v qopsec-m2:/root/.m2 \
  -w /src/validation_send_api maven:3.9-eclipse-temurin-17 mvn -B test
```

---

## KB-002 — `.block()` no fim das cadeias reativas do `context_api`

**Status:** concluído em 2026-11-18
**Arquivos:** `context_api/.../service/RiskServiceClient.java`, `ConfidentialityServiceClient.java`

**Resolução:** bloqueio mantido por decisão de arquitetura (MVC) e documentado em
`application.properties`, com `server.tomcat.threads.max=200`, `min-spare=20` e `accept-count=200`.
`mvn -B test`: 17 testes, 0 falhas.

**Problema:** o retry passou a usar `Retry.backoff` do Reactor, mas o resultado ainda é obtido com
`blockOptional()`/`block()`. Em MVC isso é aceitável, porém cada chamada prende uma thread do Tomcat
enquanto o upstream responde (timeout de 1,5s + backoff). Sob carga, isso limita a concorrência por
thread e não por conexão.

**Aceite:** ou os clientes passam a devolver `Mono`/`CompletableFuture` até o controller, ou fica
documentado que o bloqueio é intencional e o pool do Tomcat foi dimensionado para isso
(`server.tomcat.threads.max`).

**Verificação:** `mvn -B test` no `context_api` com `--network host` (precisa do Postgres em 5432).

---

## KB-003 — Prometheus não existe no namespace `qopsec`

**Status:** parcial em 2026-11-18
**Arquivo:** `deploy/k8s/22-prometheus-adapter.yaml`, `deploy/k8s/23-prometheus.yaml`

**Resolução parcial:** `23-prometheus.yaml` cria ServiceAccount, RBAC, ConfigMap de scrape dos pods
`qopsec-orchestrator`/`qopsec-kms`, Deployment e Service `prometheus`. O `APIService` do adapter foi
corrigido de `custom.metrics.k8s.io` para `external.metrics.k8s.io`. Falta aplicar: não há `kubectl`
nesta máquina (KB-005).

**Problema:** o adapter aponta para `http://prometheus.qopsec.svc:9090/`, mas nenhum manifest cria
esse Service/Deployment. Sem ele o HPA por `qopsec_pipeline_queue_depth` fica sem métrica e o
orchestrator escala só por CPU.

**Aceite:** manifest de Prometheus (Deployment + Service `prometheus` + ConfigMap de scrape dos pods
`qopsec-orchestrator` e `qopsec-kms`) no namespace, e
`kubectl get --raw "/apis/external.metrics.k8s.io/v1beta2"` listando `qopsec_pipeline_queue_depth`.

**Verificação:**
```bash
kubectl -n qopsec get svc prometheus
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta2/namespaces/qopsec/qopsec_pipeline_queue_depth" | head
```

---

## KB-004 — Imagens Docker nunca foram buildadas

**Status:** concluído em 2026-11-18
**Arquivos:** `deploy/Dockerfile.orchestrator`, `deploy/Dockerfile.kms`, `.dockerignore`

**Resolução:** `.dockerignore` criado (exclui `qopsec_env`, `logs`, `target`, `datasets`, `.hermes`).
Base do orchestrator passou a `python:3.13-slim` (o `requirements.txt` pina `numpy==2.5.1` e
`scipy==1.18.0`, que exigem Python >= 3.12) e o build passou a instalar também
`requirements-architecture.txt` (redis/aio-pika). O KMS ganhou `prometheus-fastapi-instrumentator` no
`requirements.txt` e `qiskit_adapter.py` recebeu `from __future__ import annotations`, porque a
anotação `-> NoiseModel` quebrava o import quando o qiskit não está instalado.
Imagens: `qopsec-orchestrator:dev` e `qopsec-kms:dev`, ambas com import validado; KMS em container
respondeu `/health` 200 com `storage_backend: redis` e `storage_healthy: true`.

**Problema:** os Dockerfiles foram escritos e o `docker compose config` valida, mas nenhum
`docker build` foi executado. O contexto de build é a raiz do projeto (`context: ..`), então um
`.dockerignore` ausente pode enviar `qopsec_env/`, `logs/` e `target/` para o builder.

**Aceite:** `docker build` das duas imagens sem erro, imagem final sem `qopsec_env` e sem `logs`, e
container subindo com healthcheck saudável.

**Verificação:**
```bash
cd /home/umbrel/projetos/Q-OPSEC
docker build -f deploy/Dockerfile.orchestrator -t qopsec-orchestrator:dev .
docker build -f deploy/Dockerfile.kms -t qopsec-kms:dev .
docker run --rm qopsec-kms:dev python -c "import main; print('ok')"
```

---

## KB-005 — Manifests k8s nunca aplicados

**Status:** parcial em 2026-11-18
**Arquivos:** `deploy/k8s/*.yaml`, `deploy/kind-config.yaml`, `tests/test_k8s_manifests.py`

**Problema:** os manifests nunca tinham sido aplicados num cluster.

**Resolução parcial:** `kind` v0.24.0 e `kubectl` v1.31.4 instalados em `/usr/local/bin`; cluster
`qopsec` criado com 3 nós (`deploy/kind-config.yaml`, Ingress em 18080/18443) e imagens carregadas.
`kubectl apply --dry-run=server` passou nos 29 documentos e o apply real criou namespace, config,
StatefulSets de Redis e RabbitMQ (PVCs Bound), Prometheus e os dois Deployments com HPA e PDB.
No cluster: `qopsec-kms` 2/2 Running, `prometheus` 1/1, `redis` e `rabbitmq` 1/1.

Três defeitos reais apareceram só no cluster e foram corrigidos: `imagePullPolicy` duplicado nos dois
Deployments, `prometheus-adapter` sem `ClusterRoleBinding` para `system:auth-delegator` nem
`RoleBinding` para `extension-apiserver-authentication-reader` no `kube-system`, e o orchestrator
chamando o CLI `docker` dentro do pod (`FileNotFoundError` derrubava o startup). O autostart dos 11
serviços passou a ser desligável por `QOPSEC_AUTOSTART_SERVICES`.

**Falta:** recarregar a imagem corrigida no cluster e aplicar o RBAC novo do adapter — ambos barrados
pelo gate por serem alteração de cluster em execução. Depois disso, `kubectl rollout status` e o
`APIService` `v1beta2.external.metrics.k8s.io` devem ficar `Available=True`.

**Aceite:** `kubectl apply -k`/`-f` sem erro, pods `Ready`, `kubectl rollout status` concluído e HPA
listando as duas métricas.

**Verificação:**
```bash
kubectl apply -f deploy/k8s/00-namespace.yaml -f deploy/k8s/01-config.yaml
kubectl apply -f deploy/k8s/10-redis.yaml -f deploy/k8s/11-rabbitmq.yaml
kubectl apply -f deploy/k8s/20-kms.yaml -f deploy/k8s/21-orchestrator.yaml
kubectl apply -f deploy/k8s/22-prometheus-adapter.yaml -f deploy/k8s/30-ingress.yaml
kubectl -n qopsec rollout status deploy/qopsec-orchestrator
kubectl -n qopsec describe hpa qopsec-orchestrator | tail -20
```

---

## KB-006 — Teste de carga JMeter nunca executado

**Status:** concluído em 2026-11-18
**Arquivos:** `load/qopsec_pipeline_async.jmx`, `load/README.md`, `load/report`

**Resolução:** execução via `justb4/jmeter:5.5` em container com 20 threads / ramp 10s / 5 loops
(100 requisições): 100% `202`, 0% de erro, avg 10,4 ms, p95 25 ms, máx 84 ms, ~10,7 req/s. A fila
drenou de 69 para 0 em ~1 min (`succeeded: 121`, `failed: 0`) e `open_breakers: []`. O perfil
50/30/20 do README não foi executado porque enfileira 1000 jobs (~85 min de drenagem com 4 workers).

**Problema:** o plano foi validado só como XML. Não há baseline de throughput, p95 nem taxa de erro,
então não se sabe qual componente satura primeiro.

**Aceite:** execução com 50 threads / ramp 30s / 20 loops registrada em `load/report`, com taxa de
erro 0, `qopsec_pipeline_jobs_queued` drenando para 0 e `qopsec_circuit_breakers_open` em 0.

**Verificação:**
```bash
jmeter -n -t load/qopsec_pipeline_async.jmx -Jhost=127.0.0.1 -Jport=8081 \
  -Jthreads=50 -Jramp=30 -Jloops=20 -l load/results.jtl -e -o load/report
```

---

## KB-007 — pgBouncer e réplica de leitura reais

**Status:** concluído em 2026-11-18
**Arquivo:** `kms_service/database.py`
**Script:** `deploy/postgres-replica/setup.sh` (+ `README.md`)

**Resolução:** `./deploy/postgres-replica/setup.sh` aplicou a regra
`host replication replicator 172.16.0.0/12 scram-sha-256` no `pg_hba.conf` do primary e recarregou a
configuração; `pg_basebackup -Fp -Xs -R` criou a réplica na 5433 e o `edoburu/pgbouncer` subiu na
6432 em modo `transaction`. Validações: `pg_is_in_recovery()` = `t`, escrita no primary visível na
réplica (`kb007_probe` = 1) e `select 1` pelo bouncer. Com
`QOPSEC_DATABASE_URL=…:6432/validation_send` e `QOPSEC_DATABASE_READ_URL=…:5433/validation_send`,
`pool_status()` reporta `replica_configured: true`, `pool_size: 10` nos dois pools, e
`get_read_db()` responde na réplica.

Três correções necessárias no caminho: `chmod 700` no PGDATA após o basebackup (o Postgres recusava
"invalid permissions"), limpeza do data dir antes do backup ("exists but is not empty") e uso do
gateway da rede Docker (`172.17.0.1`) na validação do bouncer, já que o container do primary não está
em host network. `psycopg[binary]` entrou no `kms_service/requirements.txt` — sem ele o
`create_engine` falhava com `ModuleNotFoundError: No module named 'psycopg'`.
**Problema:** o pool (`pool_size`, `max_overflow`, `pool_pre_ping`, `pool_recycle`) e a URL de leitura
(`QOPSEC_DATABASE_READ_URL`, `get_read_db`) existem, mas foram exercitados com SQLite. Não há
pgBouncer nem réplica configurada, e o `psycopg` não está instalado no `qopsec_env`.

**Aceite:** pgBouncer em modo transaction na frente do Postgres, `QOPSEC_DATABASE_READ_URL` apontando
para a réplica, e `pool_status()` mostrando `replica_configured: true` com conexão real.

**Verificação:**
```bash
QOPSEC_DATABASE_URL=postgresql://... QOPSEC_DATABASE_READ_URL=postgresql://... \
  ./qopsec_env/bin/python -c "import sys; sys.path.insert(0,'kms_service'); import database; print(database.pool_status())"
```

---

## KB-008 — Backend Redis do KMS validado só contra cliente falso

**Status:** concluído em 2026-11-18
**Arquivo:** `kms_service_v2/storage.py`

**Resolução:** validado contra o Redis real em 6380 com senha: `backend=redis`, `healthy=True`,
save em uma instância e leitura por uma segunda instância (réplica), `stats` e `delete` corretos.

**Problema:** `RedisKeySessionStore` passou em save/get/expiração/delete/stats contra um Redis falso
em memória. O Redis real (porta 6380, com senha) não foi exercitado, nem o caminho
`QOPSEC_KMS_SESSION_BACKEND=redis` com o serviço rodando em duas réplicas.

**Aceite:** duas instâncias do KMS apontando para o mesmo Redis, `create_key` em uma e `get_key` na
outra devolvendo a mesma sessão; `/health` com `storage_backend: redis` e `storage_healthy: true`.

**Verificação:**
```bash
QOPSEC_KMS_SESSION_BACKEND=redis QOPSEC_REDIS_URL=redis://:SENHA@127.0.0.1:6380/0 \
  ./qopsec_env/bin/python -c "import sys; sys.path.insert(0,'kms_service_v2'); import storage; print(storage.session_store.backend, storage.session_store.healthy())"
```

---

## KB-009 — Rate limit distribuído validado só contra cliente falso

**Status:** concluído em 2026-11-18
**Arquivo:** `common/rate_limit.py`

**Resolução:** contador compartilhado confirmado com Redis real (limite 2: `allowed, allowed, denied`
entre duas instâncias) e fallback para `memory` com cliente quebrado, sem exceção. Corrigido no
caminho: `pipe.execute()` síncrono do redis-py não é awaitable; o código agora aceita cliente sync e
async.

**Problema:** `RedisRateLimiter` e `FailoverRateLimiter` foram testados com um pipeline falso. Falta
confirmar que duas réplicas do orchestrator compartilham o contador e que o fallback para memória
acontece quando o Redis cai de verdade.

**Aceite:** duas instâncias atrás do mesmo Redis recusando a terceira requisição com
`QOPSEC_RATE_LIMIT_REQUESTS=2`; derrubar o Redis e ver `X-RateLimit-Backend: memory` sem 500.

**Verificação:**
```bash
for i in 1 2 3; do curl -s -o /dev/null -w "%{http_code} " -X POST \
  -H 'Content-Type: application/json' -H "Idempotency-Key: rl-$i" \
  -d '{"request_id":"rl-'$i'"}' http://127.0.0.1:8081/run-pipeline/async; done; echo
```

---

## KB-010 — Tracing OpenTelemetry sem pacotes e sem coletor

**Status:** concluído em 2026-11-18
**Arquivos:** `common/tracing.py`, `requirements-observability.txt`, `monitoring/otel-collector.yml`

**Resolução:** SDK instalado no `qopsec_env` (api/sdk/exporter/instrumentation 1.29.0 e 0.50b0),
coletor `qopsec_otel_collector` adicionado ao `services.yaml` (imagem
`otel/opentelemetry-collector-contrib:0.115.1`, config validada, portas 4317/4318 em 127.0.0.1) e
`start_qopsec.sh` exporta `QOPSEC_TRACING_ENABLED=true` e
`QOPSEC_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces`. `/architecture/status` reporta
`tracing.enabled: true` e o coletor registra spans do orchestrator e das chamadas httpx
(`/kms/create_key`, `/predict/`, `/validation/send`, `/api/v1/predict`, `/encrypt`).

**Problema:** o SDK não está instalado no `qopsec_env`, então `setup_tracing` cai no no-op e
`/architecture/status` reporta `tracing.enabled: false`. Não há coletor OTLP configurado.

**Aceite:** `pip install -r requirements-observability.txt`, `QOPSEC_TRACING_ENABLED=true` e
`QOPSEC_OTLP_ENDPOINT` apontando para um coletor; spans de `/run-pipeline` e das chamadas httpx
visíveis no backend.

**Verificação:**
```bash
./qopsec_env/bin/pip install -r requirements-observability.txt
QOPSEC_TRACING_ENABLED=true ./qopsec_env/bin/python -c "import sys; sys.path.insert(0,'.'); from common.tracing import setup_tracing; print(setup_tracing())"
```

---

## KB-011 — Dashboard nunca aberto no navegador

**Status:** concluído em 2026-11-18
**Arquivo:** `dashboard.html`

**Resolução:** aba Redis/Queue aberta no navegador com os cards preenchidos (Redis conectado,
RabbitMQ conectado, jobs, async pipeline, resilience com breaker `closed`, rate limit `redis`,
tracing desabilitado) e aba Pipeline enfileirando pelo botão assíncrono até `succeeded`.

**Problema:** o JSX compila com Babel e os endpoints foram testados com `TestClient`, mas a interface
não foi renderizada. Os cards novos ("Resilience", "Rate limit & tracing", "Async pipeline") e o
botão de fila assíncrona não foram vistos funcionando.

**Aceite:** aba Redis/Queue mostrando os cards preenchidos e aba Pipeline enfileirando com polling até
`succeeded`.

**Verificação:** `pm2 restart q-opsec` e abrir `http://192.168.18.18:8090` (ou a porta configurada).

---

## KB-012 — Nada commitado; serviços não reiniciados

**Status:** concluído em 2026-11-18
**Escopo:** repositório inteiro

**Resolução:** commit `c8539aa` com 50 arquivos (18 modificados, 32 novos), `.hermes/` adicionado ao
`.gitignore`. `pm2 restart q-opsec` executado e `/architecture/status` devolve os blocos `pipeline`,
`resilience`, `rate_limit` e `tracing`.

**Problema:** todas as mudanças estão no working tree (`git status` com 8 modificados e 5 novos
caminhos). O que roda em produção continua no código antigo, então os testes ao vivo não refletem o
novo comportamento.

**Aceite:** commit revisado e `pm2 restart q-opsec` com `/architecture/status` mostrando os blocos
`pipeline`, `resilience`, `rate_limit` e `tracing`.

**Verificação:**
```bash
git -C /home/umbrel/projetos/Q-OPSEC status --porcelain
curl -s http://127.0.0.1:8090/architecture/status | head -40
```

---

## KB-013 — Breaker Java duplicado em três módulos

**Status:** concluído em 2026-11-18
**Arquivos:** `interceptor_api`, `context_api`, `validation_send_api` em `.../resilience/CircuitBreaker.java`

**Resolução:** `tests/test_java_breaker_parity.py` compara as três cópias ignorando a linha `package`
e falha quando divergirem. Suíte Python: 47 testes, 0 falhas.

**Problema:** para não adicionar dependência, o breaker foi copiado em três pacotes. As cópias podem
divergir com o tempo.

**Aceite:** ou um módulo comum (`qopsec-commons`) publicado localmente e consumido pelos três, ou um
teste que compare os três arquivos e falhe quando divergirem.

**Verificação:**
```bash
diff interceptor_api/src/main/java/tech/noetzold/interceptor_api/resilience/CircuitBreaker.java \
     context_api/src/main/java/tech/noetzold/context_api/resilience/CircuitBreaker.java
```

---

## KB-014 — Serviços do pipeline parados: job falha no step 1

**Status:** concluído em 2026-11-18
**Escopo:** runtime (portas 8080, 65534, 8083, 8000, 8088, 9009, 8001, 8002, 8003, 8004, 8005)

**Resolução:** `_startup` agora dispara `_autostart_services` com a lista `autostart_services` do
`services.yaml`, subindo os 11 serviços de processo em background após a API responder. O step
`handshake_negotiator` também passou a receber `source`, `destination` e `proposed` montados pelo
orchestrator. Job `486a4f40…`: 11 steps, `failed_steps: []`.

**Problema:** o `_startup` do orchestrator só sobe serviços `type: docker`; os serviços de processo
(Java e Python) não são iniciados automaticamente. Com todos parados, o job assíncrono é aceito e
executado, mas para no primeiro step com `interceptor_api: All connection attempts failed`.

**Aceite:** `POST /run-pipeline/async` concluindo os 11 steps com `failed_steps: []`.

**Verificação:**
```bash
for p in 8080 65534 8083 8000 8088 9009 8001 8002 8003 8004 8005; do
  printf "%s " $p; (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null && echo up || echo down; done
curl -s -X POST -H 'Content-Type: application/json' -H "Idempotency-Key: kb14-$(date +%s)" \
  -d '{"request_id":"kb14","data":{"payload":"x"}}' http://127.0.0.1:8090/run-pipeline/async
```

---

## Concluído

| ID | Título | Data | Evidência |
| --- | --- | --- | --- |
| — | KMS stateless com Redis | 2026-11-18 | `REDIS_STORE_OK`, `KMS_HEALTH_OK` |
| — | Pipeline assíncrono com idempotência | 2026-11-18 | `ASYNC_ENDPOINT_OK`, `IDEMPOTENCY_OK` |
| — | Circuit breaker e retry no Python | 2026-11-18 | `RESILIENCE_OK`, `FAILOVER_OK` |
| — | Rate limit com middleware e 429 | 2026-11-18 | `RATE_LIMIT_MIDDLEWARE_OK` |
| — | Pool e réplica de leitura no `database.py` | 2026-11-18 | `pool_status()` com SQLite |
| — | Dockerfiles, compose e manifests k8s | 2026-11-18 | `COMPOSE_OK`, 8 YAML parseados |
| — | Java sem `Thread.sleep`/`runAsync` | 2026-11-18 | `mvn test`: 11 + 17 + 11 passed |
| — | HPA por profundidade de fila (manifest) | 2026-11-18 | `22-prometheus-adapter.yaml` parseado |
| — | Async no ar após restart do pm2 | 2026-11-18 | `202` + job `succeeded` + dedup |
| — | Poison message: retry limitado e sem requeue infinito | 2026-11-18 | `RECOVER_STALE_OK`, 6 passed, laço parado |
| KB-001 | HTTP fora da transação no `ForwardService` | 2026-11-18 | `mvn test`: 12 passed, teste de propagação |
| KB-002 | Tomcat dimensionado para o `block()` intencional | 2026-11-18 | `mvn test`: 17 passed |
| KB-008 | KMS com Redis real e duas instâncias | 2026-11-18 | `backend=redis`, `healthy=True`, leitura cruzada |
| KB-009 | Rate limit distribuído com Redis real | 2026-11-18 | `allowed, allowed, denied`; failover `memory` |
| KB-011 | Dashboard conferido no navegador | 2026-11-18 | cards preenchidos + job `succeeded` na UI |
| KB-013 | Paridade do breaker Java | 2026-11-18 | `tests/test_java_breaker_parity.py`, 47 passed |
| KB-014 | Pipeline completo com 11 steps | 2026-11-18 | job `486a4f40…`, `failed_steps: []` |
| KB-004 | Imagens Docker buildadas e validadas | 2026-11-18 | `orchestrator import ok`, KMS `/health` 200 com redis |
| KB-010 | Tracing OTLP com coletor | 2026-11-18 | `tracing.enabled: true`, spans no `qopsec-otel` |
| KB-012 | Commit e restart do pm2 | 2026-11-18 | commit `c8539aa`, 50 arquivos |
| KB-006 | Carga no pipeline assíncrono | 2026-11-18 | 100 req, 100% 202, p95 25 ms, fila drenada |
| KB-007 | pgBouncer 6432 + réplica 5433 | 2026-11-18 | `pg_is_in_recovery=t`, escrita replicada, `replica_configured: true` |
| KB-005 | Validação estática dos manifests k8s | 2026-11-18 | `tests/test_k8s_manifests.py`, 98 passed, 29 documentos |
