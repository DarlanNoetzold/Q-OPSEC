# Deploy do Q-OPSEC

Três alvos suportados, do mais simples ao mais próximo de produção. Todos usam o mesmo artefato:
o código deste repositório, com `services.yaml` como fonte da verdade dos serviços.

| Alvo | Quando usar | Comando |
| --- | --- | --- |
| local (pm2) | máquina de desenvolvimento, estado atual do Umbrel | `./deploy/deploy.sh local` |
| compose | stack isolada com Redis, RabbitMQ, KMS e orchestrator em containers | `./deploy/deploy.sh compose` |
| kubernetes | cluster real, com HPA, PDB, probes e Ingress | `./deploy/deploy.sh k8s` |

Verificação e diagnóstico, em qualquer alvo:

```bash
./deploy/deploy.sh status   # pm2, containers e /architecture/status
./deploy/deploy.sh verify   # enfileira um job e espera os 11 steps
```

## 0. Pré-requisitos comuns

Segredos ficam fora do repositório, em `/home/umbrel/.config/qopsec/middleware.env` (ou no caminho
apontado por `SECRETS_FILE`):

```bash
QOPSEC_REDIS_PASSWORD=...
QOPSEC_RABBITMQ_USER=...
QOPSEC_RABBITMQ_PASSWORD=...
```

O script carrega esse arquivo e falha cedo se alguma variável faltar. Nada de credencial em
`services.yaml`, compose ou manifest — os manifests usam `Secret` com placeholders.

## 1. Local (pm2)

É como a máquina roda hoje: o pm2 mantém o `orchestrator_linux.py` vivo e o próprio orchestrator
sobe os 11 serviços de processo via `autostart_services`.

```bash
./deploy/deploy.sh local
```

O que acontece:

1. `pm2 restart q-opsec`;
2. espera `/architecture/status` responder em `:8090`;
3. espera as 11 portas de serviço abrirem (8080, 65534, 8083, 8000, 8088, 9009, 8001, 8002, 8003,
   8004, 8005).

Para subir só um serviço: `curl -X POST localhost:8090/services/<nome>/start`. Para parar tudo:
`curl -X POST localhost:8090/stop/all`.

## 2. Docker Compose

Sobe Redis, RabbitMQ, KMS, orchestrator, Prometheus e Grafana numa rede própria.

```bash
./deploy/deploy.sh compose
```

O script valida o arquivo (`docker compose config`) antes de subir, e usa `--build` para reconstruir
as imagens do KMS e do orchestrator a partir dos Dockerfiles.

Portas publicadas, todas em loopback:

| Serviço | Porta no host |
| --- | --- |
| orchestrator | 8081 |
| Prometheus | 9091 |
| Grafana | 3001 |

Dentro da rede do compose o orchestrator escuta em 8081 e o KMS em 8002. Para acompanhar:

```bash
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs -f orchestrator
```

Para derrubar sem perder volumes: `docker compose -f deploy/docker-compose.yml down`. Com
`-v` os volumes de Redis, RabbitMQ, Prometheus e Grafana são removidos.

## 3. Kubernetes

### 3.1 Cluster local com kind

```bash
kind create cluster --name qopsec
kubectl cluster-info --context kind-qopsec
```

As imagens precisam estar disponíveis no nó. Duas opções:

```bash
# A) carregar as imagens já buildadas na máquina
docker build -f deploy/Dockerfile.orchestrator -t qopsec-orchestrator:1.0.0 .
docker build -f deploy/Dockerfile.kms -t qopsec-kms:1.0.0 .
kind load docker-image qopsec-orchestrator:1.0.0 qopsec-kms:1.0.0 --name qopsec

# B) publicar num registry e ajustar a imagem nos manifests
docker tag qopsec-orchestrator:1.0.0 ghcr.io/darlannoetzold/qopsec-orchestrator:1.0.0
docker push ghcr.io/darlannoetzold/qopsec-orchestrator:1.0.0
```

Os manifests referenciam `ghcr.io/darlannoetzold/qopsec-{kms,orchestrator}:1.0.0` com
`imagePullPolicy: IfNotPresent`, então a opção A funciona sem registry.

### 3.2 Segredos

`deploy/k8s/01-config.yaml` traz placeholders. Substitua antes de aplicar, sem commitar o valor real:

```bash
kubectl create namespace qopsec --dry-run=client -o yaml | kubectl apply -f -
kubectl -n qopsec create secret generic qopsec-secrets \
  --from-literal=QOPSEC_REDIS_PASSWORD="$QOPSEC_REDIS_PASSWORD" \
  --from-literal=QOPSEC_RABBITMQ_USER="$QOPSEC_RABBITMQ_USER" \
  --from-literal=QOPSEC_RABBITMQ_PASSWORD="$QOPSEC_RABBITMQ_PASSWORD" \
  --from-literal=QOPSEC_REDIS_URL="redis://:$QOPSEC_REDIS_PASSWORD@qopsec-redis:6379/0" \
  --from-literal=QOPSEC_RABBITMQ_URL="amqp://$QOPSEC_RABBITMQ_USER:$QOPSEC_RABBITMQ_PASSWORD@qopsec-rabbitmq:5672/" \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3.3 Aplicar

```bash
./deploy/deploy.sh k8s --dry-run   # valida no servidor sem alterar nada
./deploy/deploy.sh k8s             # aplica e espera o rollout
```

A ordem aplicada é: namespace e config, depois Redis/RabbitMQ/Prometheus/adapter, depois KMS,
orchestrator e Ingress. O script termina com `kubectl rollout status` nos dois Deployments e imprime
`pods,svc,hpa,pdb`.

### 3.4 O que o cluster entrega

- **KMS e orchestrator**: 2 réplicas mínimas, `readinessProbe` e `livenessProbe`, `resources`
  requests/limits, `topologySpreadConstraints` entre nós.
- **HPA v2**: KMS escala por CPU (65%); orchestrator escala por CPU e pela métrica `External`
  `qopsec_pipeline_queue_depth` (target 20), servida pelo `prometheus-adapter`.
- **PDB**: `minAvailable: 1` nos dois, para não derrubar tudo num drain.
- **Prometheus**: ServiceAccount com RBAC, scrape por `kubernetes_sd_configs` restrito aos pods
  `qopsec-orchestrator` e `qopsec-kms`.
- **Ingress**: host `qopsec.local`, encaminhando para o Service do orchestrator.

Para o Ingress responder localmente:

```bash
# com ingress-nginx instalado no kind
echo "127.0.0.1 qopsec.local" | sudo tee -a /etc/hosts
curl -H 'Host: qopsec.local' http://127.0.0.1/architecture/status
```

## 4. Verificação

```bash
./deploy/deploy.sh verify
```

Enfileira um job com `Idempotency-Key` novo, faz polling em `/pipeline/jobs/{id}` e exige
`status: succeeded` com `failed_steps: []`. Saída esperada:

```
status succeeded steps 11 failed []
```

No cluster, o equivalente:

```bash
kubectl -n qopsec get pods,hpa
kubectl -n qopsec get --raw "/apis/external.metrics.k8s.io/v1beta2" | head -c 400
kubectl -n qopsec port-forward svc/qopsec-orchestrator 8090:8081
curl -s localhost:8090/architecture/status | python3 -m json.tool
```

## 5. Rollback

| Alvo | Comando |
| --- | --- |
| local | `git checkout <commit> && ./deploy/deploy.sh local` |
| compose | `docker compose -f deploy/docker-compose.yml down && git checkout <commit> && ./deploy/deploy.sh compose` |
| k8s | `kubectl -n qopsec rollout undo deploy/qopsec-orchestrator` (idem para `qopsec-kms`) |

No k8s o rollback usa o histórico de ReplicaSets, então funciona mesmo sem o commit antigo em mãos.

## 6. Limites conhecidos

- O compose não inclui os 11 serviços de processo: eles são lançados pelo orchestrator, que precisa
  do código e do `services.yaml` montados. Para um cluster completo, cada serviço precisa do seu
  próprio Deployment — hoje só KMS e orchestrator têm manifest.
- O `prometheus-adapter` exige que o `APIService` fique `Available`; em cluster sem metrics-server o
  HPA por CPU não funciona, mas o HPA por métrica `External` continua.
- O Ingress usa `qopsec.local`; troque o host e o `ingressClassName` conforme o cluster.
- As credenciais de exemplo do Postgres em `deploy/postgres-replica` são de desenvolvimento.
