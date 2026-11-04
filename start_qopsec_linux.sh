#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
#  Q-OPSEC - Inicia o ambiente no Linux (orquestrador + servicos)
#  Uso: bash start_qopsec_linux.sh [--skip-services]
# ===========================================================================
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPF="$INSTALL_DIR/qopsec_env/bin/python"

arg_skip=0
for a in "$@"; do [ "$a" = "--skip-services" ] && arg_skip=1; done

# 1) venv deve existir
[ -x "$DPF" ] || { echo "[ERRO] venv nao encontrado em $INSTALL_DIR/qopsec_env. Rode o install.sh." >&2; exit 1; }
[ -d "$INSTALL_DIR/logs" ] || mkdir -p "$INSTALL_DIR/logs"
PIDF="$INSTALL_DIR/logs/orchestrator.pid"

# 2) segredos do middleware.env
SECRETS_FILE="$HOME/.config/qopsec/middleware.env"
if [ -f "$SECRETS_FILE" ]; then
    # shellcheck disable=SC1090
    source "$SECRETS_FILE"
    echo "[OK] Segredos carregados de $SECRETS_FILE"
elif [ -f "$INSTALL_DIR/middleware.env" ]; then
    # shellcheck disable=SC1090
    source "$INSTALL_DIR/middleware.env"
    echo "[OK] Segredos carregados do middleware.env do repositorio"
else
    echo "[ERRO] Segredos nao encontrados. Rode o install.sh primeiro." >&2
    exit 1
fi

# 3) sobe o orquestrador se nao estiver rodando
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "[OK] Orquestrador ja esta rodando (PID $(cat "$PIDF"))"
else
    echo "[..] Iniciando orquestrador (porta 8090)..."
    # shellcheck disable=SC2024
    nohup "$DPF" orchestrator.py >"$INSTALL_DIR/logs/qopsec.out.log" 2>"$INSTALL_DIR/logs/qopsec.err.log" &
    echo $! >"$PIDF"
    echo "[OK] Orquestrador PID $(cat "$PIDF")"
    ok=0
    for _ in $(seq 1 24); do
        if curl -sf -o /dev/null --max-time 5 http://127.0.0.1:8090/architecture/status 2>/dev/null; then ok=1; break; fi
        sleep 5
    done
    [ "$ok" = "1" ] && echo "[OK] Orquestrador pronto" || echo "[AVISO] orquestrador nao respondeu a tempo (veja logs/qopsec.err.log)" >&2
fi

# 4) dispara servicos
if [ "$arg_skip" = "1" ]; then
    echo "[OK] --skip-services: servicos nao iniciados"
else
    echo "[..] Disparando /start/all..."
    if curl -sf -X POST "http://127.0.0.1:8090/start/all" >/dev/null 2>&1; then
        echo "[OK] Servicos iniciados"
    else
        echo "[AVISO] falha no /start/all" >&2
    fi
fi
