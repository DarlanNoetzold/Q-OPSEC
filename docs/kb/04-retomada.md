# Retomada — onde parar e como continuar

Documento de handoff. Escrito em 2026-11-18 para retomar o trabalho sem reconstruir contexto.
Complementa `02-estado-atual.md` (arquitetura) e `03-deploy.md` (deploy).

## 1. Estado no momento da pausa

| Item | Valor |
| --- | --- |
| Repositório | `/home/umbrel/projetos/Q-OPSEC`, branch `main`, árvore limpa |
| Último commit | `03de971` — KB-005 com evidência do cluster kind |
| Suíte Python | 180 passed (`./qopsec_env/bin/python -m pytest tests/ -q`) |
| Testes Maven | 40 passed (validation 12, context 17, interceptor 11) |
| pm2 | `q-opsec` online, 10 restarts |
| Pipeline | `{queued: 0, running: 0, succeeded: 124, failed: 0}` |
| Breakers | `[]` (nenhum aberto) |
| Rate limit | backend `redis` |
| Tracing | `enabled: true` |
| Cluster | kind `qopsec`, 3 nós, contexto `kind-qopsec` |
| Ferramentas | `kind` v0.24.0 e `kubectl` v1.31.4 em `/usr/local/bin` |

Containers no ar:

| Container | Porta publicada |
| --- | --- |
| `qopsec-redis` | 127.0.0.1:6380 |
| `qopsec-rabbitmq` | 127.0.0.1:5673 (AMQP), 127.0.0.1:15673 (UI) |
| `qopsec-otel` | 127.0.0.1:4317, 127.0.0.1:4318 |
| `qopsec-prom` | 0.0.0.0:9091 → 9090 |
| `qopsec-graf` | 0.0.0.0:3001 → 3000 |
| `qopsec-pg-replica` | 5433 |
| `qopsec-pgbouncer` | 6432 |

Serviços de processo nas portas 8080, 65534, 8083, 8000, 8088, 9009, 8001, 8002, 8003, 8004, 8005,
todos lançados pelo orchestrator via `autostart_services`.

Cluster kind `qopsec` (3 nós), namespace `qopsec`:

| Workload | Estado na pausa |
| --- | --- |
| `qopsec-kms` | 2/2 Running |
| `prometheus` | 1/1 Running |
| `qopsec-redis-0` | 1/1 Running, PVC `Bound` |
| `qopsec-rabbitmq-0` | 1/1 Running, PVC `Bound` |
| `qopsec-orchestrator` | 0/2 CrashLoopBackOff (imagem antiga no nó) |
| `prometheus-adapter` | 0/1 CrashLoopBackOff (RBAC novo ainda não aplicado) |

O cluster sobrevive a reboot do host? Não: containers do kind param junto com o Docker. Para
recriar, ver §3.

## 2. O que está concluído

Backlog `KB-001` a `KB-014` em `00-pendencias.md`, com exceção do apply real do KB-005:

- **KB-001** HTTP fora da transação no `ForwardService`.
- **KB-002** Tomcat dimensionado para o `block()` intencional no `context_api`.
- **KB-003** Prometheus e adapter no namespace `qopsec`, `APIService` em `external.metrics.k8s.io`.
- **KB-004** Imagens Docker buildáveis (`.dockerignore`, base 3.13, `psycopg`, instrumentator).
- **KB-006** Carga JMeter: 100 req, 100% 202, p95 25 ms, fila drenada.
- **KB-007** pgBouncer 6432 + réplica 5433 validados.
- **KB-008** KMS com Redis real e leitura por segunda instância.
- **KB-009** Rate limit distribuído com Redis real e failover.
- **KB-010** Tracing OTLP com coletor e spans visíveis.
- **KB-011** Dashboard conferido no navegador.
- **KB-012** Commit e restart do pm2.
- **KB-013** Paridade do breaker Java coberta por teste.
- **KB-014** Pipeline completo com 11 steps e `failed_steps: []`.

## 3. O que falta

### KB-005 — fechar o cluster (dois comandos bloqueados)

O cluster existe e a maior parte está no ar. Faltam exatamente dois comandos, barrados pelo gate por
serem alteração de cluster em execução:

```bash
# 1. recarregar a imagem corrigida do orchestrator nos 3 nós
kind load docker-image ghcr.io/darlannoetzold/qopsec-orchestrator:1.0.0 --name qopsec

# 2. aplicar o RBAC novo do prometheus-adapter
kubectl apply -f deploy/k8s/22-prometheus-adapter.yaml
```

Para liberar, peça autorização nomeando as duas ações:

> sim, rode o kind load docker-image do orchestrator e o kubectl apply do 22-prometheus-adapter.yaml

Depois, validar:

```bash
kubectl -n qopsec rollout status deploy/qopsec-orchestrator --timeout=180s
kubectl -n qopsec rollout status deploy/prometheus-adapter --timeout=180s
kubectl get apiservice v1beta2.external.metrics.k8s.io \
  -o jsonpath='{.status.conditions[0].type}={.status.conditions[0].status}{"\n"}'
kubectl -n qopsec get hpa
```

Esperado: `Available=True` no APIService e as duas métricas no HPA do orchestrator.

Se o cluster tiver sido perdido (reboot do host), recriar antes:

```bash
kind create cluster --config deploy/kind-config.yaml --wait 180s
docker build -f deploy/Dockerfile.orchestrator -t ghcr.io/darlannoetzold/qopsec-orchestrator:1.0.0 .
docker build -f deploy/Dockerfile.kms -t ghcr.io/darlannoetzold/qopsec-kms:1.0.0 .
kind load docker-image ghcr.io/darlannoetzold/qopsec-orchestrator:1.0.0 \
  ghcr.io/darlannoetzold/qopsec-kms:1.0.0 --name qopsec
./deploy/deploy.sh k8s
```

### Dívidas registradas, fora do backlog

1. **Nove serviços sem Deployment**: só KMS e orchestrator têm manifest; os outros nove são lançados
   pelo orchestrator dentro do pod, o que impede escala independente. Próximo passo natural é um
   Deployment por serviço com `services.yaml` como fonte. No cluster o autostart já está desligado
   por `QOPSEC_AUTOSTART_SERVICES=""`.
2. **`~/.pm2/logs/q-opsec-error.log` com ~974 MB**: rotacionar (`pm2 install pm2-logrotate`) antes que
   ocupe o disco.
3. **Perfil de carga 50/30/20 do `load/README.md`** não executado: enfileira 1000 jobs (~85 min com 4
   workers). Rodar em janela dedicada.
4. **Réplica e pgBouncer são containers locais**, sem failover automático nem slot de replicação
   persistente.
5. **Credenciais de exemplo** em `deploy/postgres-replica` (`replpass`, `postgres:postgres`) são de
   desenvolvimento; produção exige secret store.

## 4. Como retomar em cinco comandos

```bash
cd /home/umbrel/projetos/Q-OPSEC
git log --oneline -1                      # confirma 03de971
./qopsec_env/bin/python -m pytest tests/ -q   # espera 180 passed
./deploy/deploy.sh status                 # pm2, containers e /architecture/status
./deploy/deploy.sh verify                 # espera "status succeeded steps 11 failed []"
kubectl -n qopsec get pods                # cluster kind, se ainda existir
```

Se o pm2 estiver parado após reboot:

```bash
pm2 restart q-opsec
./deploy/deploy.sh local
```

Se os containers de infraestrutura não subirem sozinhos:

```bash
curl -X POST localhost:8090/services/qopsec_redis/start
curl -X POST localhost:8090/services/qopsec_rabbitmq/start
curl -X POST localhost:8090/services/qopsec_otel_collector/start
```

## 5. Mapa dos documentos

| Arquivo | Para quê |
| --- | --- |
| `00-pendencias.md` | backlog com ID, evidência, aceite e comando de verificação |
| `01-escala-horizontal.md` | contratos, env vars e limites da escala |
| `02-estado-atual.md` | fotografia da arquitetura |
| `03-deploy.md` | deploy nos três alvos, verificação e rollback |
| `04-retomada.md` | este documento |

## 6. Regras que economizam tempo na próxima sessão

- Antes de mexer em `pg_hba.conf`, subir container de serviço, baixar binário ou aplicar manifest num
  cluster em execução, peça autorização **nomeando a ação**; o gate recusa "sim" isolado.
- `context_api` só roda os testes Maven com `--network host` (precisa do Postgres em 5432).
- `pipe.execute()` do redis-py síncrono não é awaitable; `common/rate_limit.py` aceita sync e async.
- Após `pg_basebackup`, o PGDATA precisa de `chmod 700` e o diretório deve estar vazio antes.
- O bouncer é validado pelo gateway `172.17.0.1`, não por `127.0.0.1`, a partir do container do primary.
- No pod do cluster não existe CLI `docker`; os helpers `docker_*` toleram `FileNotFoundError` e o
  autostart é desligado por `QOPSEC_AUTOSTART_SERVICES=""`.
- `kubectl apply --dry-run=server` precisa do namespace criado antes; o `deploy.sh k8s --dry-run` já
  aplica o `00-namespace.yaml` por isso.
- Imagens no kind precisam da tag completa do registry (`ghcr.io/...`), não só `qopsec-*:1.0.0`.
