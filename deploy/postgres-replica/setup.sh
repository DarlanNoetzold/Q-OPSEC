#!/usr/bin/env bash
# Sobe pgBouncer (porta 6432) e uma réplica física do Postgres 16 (porta 5433)
# na frente do primary já existente (container "postgres", porta 5432).
#
# Pré-requisitos verificados pelo script:
#   - container "postgres" rodando com wal_level=replica e max_wal_senders>0
#   - role de replicação com REPLICATION LOGIN
#   - regra de replicação no pg_hba.conf do primary para a rede Docker
#
# Uso:
#   ./deploy/postgres-replica/setup.sh            # aplica tudo e valida
#   ./deploy/postgres-replica/setup.sh --check    # só valida, não altera nada
#   ./deploy/postgres-replica/setup.sh --teardown # remove réplica e pgBouncer
#
# Variáveis: PRIMARY=postgres REPLICA=qopsec-pg-replica PGBOUNCER=qopsec-pgbouncer
#            REPL_USER=replicator REPL_PASSWORD=replpass DB_NAME=validation_send
set -euo pipefail

PRIMARY="${PRIMARY:-postgres}"
REPLICA="${REPLICA:-qopsec-pg-replica}"
PGBOUNCER="${PGBOUNCER:-qopsec-pgbouncer}"
REPL_USER="${REPL_USER:-replicator}"
REPL_PASSWORD="${REPL_PASSWORD:-replpass}"
DB_NAME="${DB_NAME:-validation_send}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
REPLICA_PORT="${REPLICA_PORT:-5433}"
PGBOUNCER_PORT="${PGBOUNCER_PORT:-6432}"
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/logs/pg-replica}"

MODE="apply"
case "${1:-}" in
  --check) MODE="check" ;;
  --teardown) MODE="teardown" ;;
  "") ;;
  *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
esac

say() { printf '\n== %s\n' "$*"; }

teardown() {
  say "removendo containers"
  docker rm -f "$REPLICA" >/dev/null 2>&1 || true
  docker rm -f "$PGBOUNCER" >/dev/null 2>&1 || true
  say "removendo $DATA_DIR"
  rm -rf "$DATA_DIR"
  say "pronto"
}

check_primary() {
  say "primary: wal_level e max_wal_senders"
  docker exec "$PRIMARY" psql -U "$DB_USER" -tAc "show wal_level; show max_wal_senders;"
  say "role de replicação"
  docker exec "$PRIMARY" psql -U "$DB_USER" -tAc \
    "select rolname, rolreplication from pg_roles where rolname='$REPL_USER';"
  say "regra de replicação no pg_hba.conf"
  if docker exec "$PRIMARY" grep -q "replication $REPL_USER 172.16.0.0/12" /var/lib/postgresql/data/pg_hba.conf; then
    echo "presente"
  else
    echo "AUSENTE"
    return 1
  fi
}

ensure_hba() {
  say "garantindo regra de replicação no pg_hba.conf"
  docker exec "$PRIMARY" bash -c \
    "grep -q 'host replication $REPL_USER 172.16.0.0/12' /var/lib/postgresql/data/pg_hba.conf || \
     echo 'host replication $REPL_USER 172.16.0.0/12 scram-sha-256' >> /var/lib/postgresql/data/pg_hba.conf"
  docker exec "$PRIMARY" psql -U "$DB_USER" -tAc "select pg_reload_conf();"
}

ensure_role() {
  say "garantindo role $REPL_USER"
  docker exec "$PRIMARY" psql -U "$DB_USER" -tAc \
    "select 1 from pg_roles where rolname='$REPL_USER'" | grep -q 1 || \
    docker exec "$PRIMARY" psql -U "$DB_USER" -c \
      "CREATE ROLE $REPL_USER WITH REPLICATION LOGIN PASSWORD '$REPL_PASSWORD';"
}

start_replica() {
  say "subindo réplica em $REPLICA_PORT"
  docker rm -f "$REPLICA" >/dev/null 2>&1 || true
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_DIR"
  chown 999:999 "$DATA_DIR"
  docker run -d --name "$REPLICA" --network host --user postgres \
    -e PGPASSWORD="$REPL_PASSWORD" \
    -v "$DATA_DIR:/var/lib/postgresql/data" \
    postgres:16 bash -c \
    "pg_basebackup -h 127.0.0.1 -p 5432 -U $REPL_USER -D /var/lib/postgresql/data -Fp -Xs -R -P && chmod 700 /var/lib/postgresql/data && postgres -p $REPLICA_PORT" >/dev/null
  for _ in $(seq 1 30); do
    if docker exec "$REPLICA" pg_isready -p "$REPLICA_PORT" >/dev/null 2>&1; then
      echo "réplica pronta"
      return 0
    fi
    sleep 2
  done
  echo "réplica não ficou pronta; últimos logs:" >&2
  docker logs "$REPLICA" 2>&1 | tail -20 >&2
  return 1
}

start_pgbouncer() {
  say "subindo pgBouncer em $PGBOUNCER_PORT"
  docker rm -f "$PGBOUNCER" >/dev/null 2>&1 || true
  docker run -d --name "$PGBOUNCER" --network host \
    -e DB_HOST=127.0.0.1 -e DB_PORT=5432 \
    -e DB_USER="$DB_USER" -e DB_PASSWORD="$DB_PASSWORD" -e DB_NAME="$DB_NAME" \
    -e POOL_MODE=transaction -e LISTEN_PORT="$PGBOUNCER_PORT" \
    -e AUTH_TYPE=scram-sha-256 -e MAX_CLIENT_CONN=200 -e DEFAULT_POOL_SIZE=20 \
    edoburu/pgbouncer:latest >/dev/null
  for _ in $(seq 1 20); do
    if docker exec "$PGBOUNCER" sh -c "nc -z 127.0.0.1 $PGBOUNCER_PORT" >/dev/null 2>&1; then
      echo "pgBouncer pronto"
      return 0
    fi
    sleep 2
  done
  echo "pgBouncer não ficou pronto; últimos logs:" >&2
  docker logs "$PGBOUNCER" 2>&1 | tail -20 >&2
  return 1
}

verify() {
  say "réplica é standby e recebe WAL"
  docker exec "$REPLICA" psql -U "$DB_USER" -p "$REPLICA_PORT" -tAc \
    "select pg_is_in_recovery();"
  say "escrita no primary aparece na réplica"
  docker exec "$PRIMARY" psql -U "$DB_USER" -d "$DB_NAME" -tAc \
    "create table if not exists kb007_probe(id int primary key, ts timestamptz default now());
     insert into kb007_probe(id) values (1) on conflict (id) do update set ts=now();" >/dev/null
  sleep 3
  docker exec "$REPLICA" psql -U "$DB_USER" -d "$DB_NAME" -p "$REPLICA_PORT" -tAc \
    "select count(*) from kb007_probe;"
  say "pgBouncer responde"
  local gw
  gw="$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')"
  docker exec "$PRIMARY" psql "postgresql://$DB_USER:$DB_PASSWORD@$gw:$PGBOUNCER_PORT/$DB_NAME" -tAc "select 1;"
}

case "$MODE" in
  teardown) teardown ;;
  check) check_primary ;;
  apply)
    ensure_role
    ensure_hba
    check_primary
    start_replica
    start_pgbouncer
    verify
    say "concluído: QOPSEC_DATABASE_READ_URL=postgresql://$DB_USER:$DB_PASSWORD@127.0.0.1:$REPLICA_PORT/$DB_NAME"
    echo "           QOPSEC_DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@127.0.0.1:$PGBOUNCER_PORT/$DB_NAME"
    ;;
esac
