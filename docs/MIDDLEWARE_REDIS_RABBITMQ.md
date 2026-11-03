# Q-OPSEC middleware provisioning (Redis cache + RabbitMQ jobs)

O orchestrator continua sendo o único processo de aplicação supervisionado pelo PM2. O `start_qopsec.sh` instala os adapters e carrega credenciais do arquivo protegido `/home/umbrel/.config/qopsec/middleware.env`; o orchestrator inicia/valida os containers declarados em `services.yaml` antes de aceitar tráfego. Não reutilize o Redis de outros aplicativos.

## Segredos

O arquivo já existe nesta máquina com modo `0600`. Para credenciais novas/rotacionadas, gere secrets localmente e atualize, em sincronia, Redis config/start args e RabbitMQ env. Não comite nem cole os valores em tickets/chat.

Variáveis esperadas:

- `QOPSEC_REDIS_PASSWORD`
- `QOPSEC_REDIS_URL` (opcional; default aponta para localhost:6380)
- `QOPSEC_RABBITMQ_USER`
- `QOPSEC_RABBITMQ_PASSWORD`
- `QOPSEC_RABBITMQ_URL` (opcional; default aponta para localhost:5673)
- `QOPSEC_RABBITMQ_QUEUE` (opcional; default `qopsec.jobs`)

## Componentes

- `qopsec_redis`: container dedicado, Redis 7.4.3, AOF, senha, volume persistente local.
- `qopsec_rabbitmq`: RabbitMQ 3.13 management, volume persistente local, portas publicadas somente em localhost.
- Ambos têm label `com.qopsec.managed-by=orchestrator`; operações start/stop/remove recusam tomar posse de containers antigos sem essa label. Dados existentes nunca são removidos implicitamente.
- Cache Redis: listagem de datasets (30s) e preview (15s), com scope de autorização em hash na chave. Não cachear predições, decisões de segurança, payloads privados, credenciais ou material criptográfico.
- Jobs RabbitMQ: treino permitido continua exposto como `POST /train/{service_name}` com HTTP 202; consultar status com `GET /jobs/{job_id}`. Mensagens persistentes em fila durável; metadados de estado mantidos em Redis.
- Observabilidade: `GET /cache/stats`, `GET /jobs`, `GET /architecture-metrics`, health/status do catálogo `/services`, dashboards Prometheus/Grafana.

## Operação

`pm2 restart q-opsec` invoca `start_qopsec.sh`; o script aplica as dependências Python e o orquestrador garante Redis/RabbitMQ antes de ligar seus clients e workers. Infra ausente/inacessível deve falhar o startup de forma visível nos logs PM2, sem fallback silencioso para cache em memória/SQLite.

Uma vez que RabbitMQ/Redis estejam em execução, use `/services/qopsec_redis/status` e `/services/qopsec_rabbitmq/status` (ou `/services`) para conferir. Evite remover/recriar containers em operação normal: use migração explícita, backup e janela de manutenção.
