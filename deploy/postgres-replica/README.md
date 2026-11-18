# Postgres: pgBouncer + réplica de leitura

Script único para o KB-007: coloca um pgBouncer (modo `transaction`, porta 6432) na frente do primary
e sobe uma réplica física (porta 5433) para o `get_read_db()` do `kms_service`.

## Por que existe

O `kms_service/database.py` já tem pool (`pool_size`, `max_overflow`, `pool_pre_ping`, `pool_recycle`)
e separa escrita de leitura (`QOPSEC_DATABASE_URL` / `QOPSEC_DATABASE_READ_URL`, `get_read_db`,
`pool_status`). Faltava exercitar isso contra Postgres real com bouncer e réplica.

## Pré-requisitos

- container `postgres` (Postgres 16) rodando na porta 5432;
- `wal_level=replica` e `max_wal_senders>0` no primary;
- role com `REPLICATION LOGIN`;
- regra de replicação no `pg_hba.conf` do primary para a rede Docker. O `host all all all
  scram-sha-256` **não** cobre conexões `replication` — sem a linha
  `host replication replicator 172.16.0.0/12 scram-sha-256` o `pg_basebackup` falha com
  `no pg_hba.conf entry for replication connection`.

## Uso

```bash
./deploy/postgres-replica/setup.sh --check     # só valida, não altera nada
./deploy/postgres-replica/setup.sh             # aplica e valida ponta a ponta
./deploy/postgres-replica/setup.sh --teardown  # remove réplica, pgBouncer e o data dir
```

O `--check` sai com código 1 quando a regra de replicação está ausente, então serve de gate em CI.

## O que o modo `apply` faz

1. cria a role `replicator` se não existir;
2. acrescenta a linha de replicação no `pg_hba.conf` do primary e recarrega a configuração;
3. roda `pg_basebackup -Fp -Xs -R` para `logs/pg-replica` e sobe a réplica na porta 5433;
4. sobe o `edoburu/pgbouncer` em modo `transaction` na porta 6432;
5. valida: `pg_is_in_recovery()` = `t`, escrita no primary visível na réplica, e `select 1` pelo
   pgBouncer.

## Variáveis

| Variável | Padrão | Uso |
| --- | --- | --- |
| `PRIMARY` | `postgres` | container do primary |
| `REPLICA` | `qopsec-pg-replica` | container da réplica |
| `PGBOUNCER` | `qopsec-pgbouncer` | container do bouncer |
| `REPL_USER` / `REPL_PASSWORD` | `replicator` / `replpass` | role de replicação |
| `DB_NAME` | `validation_send` | base usada na validação |
| `REPLICA_PORT` / `PGBOUNCER_PORT` | `5433` / `6432` | portas publicadas |
| `DATA_DIR` | `logs/pg-replica` | PGDATA da réplica |

## Ligando no KMS

```bash
export QOPSEC_DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:6432/validation_send"
export QOPSEC_DATABASE_READ_URL="postgresql://postgres:postgres@127.0.0.1:5433/validation_send"
```

Com isso `pool_status()` passa a reportar `replica_configured: true` e as leituras vão para a réplica.

## Limites

- a réplica é um container local, não um standby gerenciado: sem failover automático nem slot de
  replicação persistente;
- o pgBouncer em modo `transaction` não suporta prepared statements nomeados nem `SET` de sessão
  entre transações;
- as credenciais acima são de desenvolvimento; em produção use um secret store.
