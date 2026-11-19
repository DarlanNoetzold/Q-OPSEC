#!/usr/bin/env bash
# Deploy do Q-OPSEC em três alvos: local (pm2), compose e kubernetes.
#
#   ./deploy/deploy.sh local            # pm2 + serviços de processo (estado atual da máquina)
#   ./deploy/deploy.sh compose          # docker compose (redis, rabbitmq, kms, orchestrator, prom, grafana)
#   ./deploy/deploy.sh k8s              # aplica deploy/k8s no cluster do kubeconfig atual
#   ./deploy/deploy.sh k8s --dry-run    # só valida no servidor (server-side dry-run)
#   ./deploy/deploy.sh status           # resumo do que está no ar
#   ./deploy/deploy.sh verify           # smoke test do pipeline assíncrono
#
# Variáveis: NAMESPACE=qopsec K8S_DIR=deploy/k8s COMPOSE_FILE=deploy/docker-compose.yml
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NAMESPACE="${NAMESPACE:-qopsec}"
K8S_DIR="${K8S_DIR:-deploy/k8s}"
COMPOSE_FILE="${COMPOSE_FILE:-deploy/docker-compose.yml}"
SECRETS_FILE="${SECRETS_FILE:-/home/umbrel/.config/qopsec/middleware.env}"
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8090}"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'erro: %s\n' "$*" >&2; exit 1; }

require() {
  command -v "$1" >/dev/null 2>&1 || die "$1 não encontrado no PATH"
}

load_secrets() {
  [ -f "$SECRETS_FILE" ] || die "arquivo de segredos ausente: $SECRETS_FILE"
  set -a
  # shellcheck disable=SC1090
  . "$SECRETS_FILE"
  set +a
}

deploy_local() {
  require pm2
  load_secrets
  say "reiniciando o orchestrator no pm2"
  pm2 restart q-opsec >/dev/null
  say "aguardando /architecture/status"
  for _ in $(seq 1 40); do
    if curl -fsS -m 3 "$ORCHESTRATOR_URL/architecture/status" >/dev/null 2>&1; then
      echo "orchestrator pronto"
      break
    fi
    sleep 2
  done
  say "aguardando os 11 serviços de processo"
  local ports=(8080 65534 8083 8000 8088 9009 8001 8002 8003 8004 8005)
  for _ in $(seq 1 40); do
    local up=0
    for port in "${ports[@]}"; do
      (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null && up=$((up + 1))
    done
    [ "$up" -eq "${#ports[@]}" ] && { echo "todos os ${#ports[@]} serviços no ar"; return 0; }
    sleep 3
  done
  die "nem todos os serviços subiram"
}

deploy_compose() {
  require docker
  load_secrets
  say "validando $COMPOSE_FILE"
  docker compose -f "$COMPOSE_FILE" config >/dev/null
  say "subindo stack"
  docker compose -f "$COMPOSE_FILE" up -d --build
  say "aguardando healthchecks"
  docker compose -f "$COMPOSE_FILE" ps
}

deploy_k8s() {
  require kubectl
  local dry="${1:-}"
  say "validando manifests em $K8S_DIR"
  if [ "$dry" = "--dry-run" ]; then
    kubectl apply --dry-run=server -f "$K8S_DIR"/
    return 0
  fi
  say "aplicando namespace e configuração"
  kubectl apply -f "$K8S_DIR/00-namespace.yaml" -f "$K8S_DIR/01-config.yaml"
  say "aplicando infraestrutura (redis, rabbitmq, prometheus, adapter)"
  kubectl apply -f "$K8S_DIR/10-redis.yaml" -f "$K8S_DIR/11-rabbitmq.yaml" \
    -f "$K8S_DIR/23-prometheus.yaml" -f "$K8S_DIR/22-prometheus-adapter.yaml"
  say "aplicando workloads"
  kubectl apply -f "$K8S_DIR/20-kms.yaml" -f "$K8S_DIR/21-orchestrator.yaml" -f "$K8S_DIR/30-ingress.yaml"
  say "aguardando rollout"
  kubectl -n "$NAMESPACE" rollout status deploy/qopsec-kms --timeout=180s
  kubectl -n "$NAMESPACE" rollout status deploy/qopsec-orchestrator --timeout=180s
  say "estado final"
  kubectl -n "$NAMESPACE" get pods,svc,hpa,pdb
}

show_status() {
  say "pm2"
  pm2 list 2>/dev/null | sed -n '1,12p' || echo "pm2 indisponível"
  say "containers qopsec"
  docker ps --filter name=qopsec --format '{{.Names}}\t{{.Status}}' || true
  say "orchestrator"
  curl -fsS -m 5 "$ORCHESTRATOR_URL/architecture/status" 2>/dev/null \
    | python3 -c 'import json,sys;d=json.load(sys.stdin);print("jobs",d["pipeline"]["jobs"]);print("breakers",d["resilience"]["open_breakers"]);print("rate_limit",d["rate_limit"]["backend"]);print("tracing",d["tracing"]["enabled"])' \
    || echo "orchestrator não respondeu"
}

verify() {
  say "enfileirando job assíncrono"
  local key="deploy-$(date +%s)"
  local job
  job="$(curl -fsS -X POST -H 'Content-Type: application/json' -H "Idempotency-Key: $key" \
    -d '{"request_id":"deploy-verify","data":{"payload":"x"}}' \
    "$ORCHESTRATOR_URL/run-pipeline/async" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')"
  echo "job $job"
  for _ in $(seq 1 60); do
    local status
    status="$(curl -fsS "$ORCHESTRATOR_URL/pipeline/jobs/$job" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status"))')"
    if [ "$status" = "succeeded" ] || [ "$status" = "failed" ]; then
      curl -fsS "$ORCHESTRATOR_URL/pipeline/jobs/$job" | python3 -c '
import json,sys
d=json.load(sys.stdin); r=d.get("result") or {}
print("status", d.get("status"), "steps", r.get("steps"), "failed", r.get("failed_steps"))
'
      [ "$status" = "succeeded" ] || die "job terminou como $status"
      return 0
    fi
    sleep 3
  done
  die "job não terminou no tempo esperado"
}

case "${1:-}" in
  local) deploy_local ;;
  compose) deploy_compose ;;
  k8s) shift; deploy_k8s "${1:-}" ;;
  status) show_status ;;
  verify) verify ;;
  *) echo "uso: $0 {local|compose|k8s [--dry-run]|status|verify}" >&2; exit 2 ;;
esac
