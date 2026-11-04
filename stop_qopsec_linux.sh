#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
#  Q-OPSEC - Para o ambiente no Linux
#  Uso: bash stop_qopsec_linux.sh [--remove-containers]
# ===========================================================================
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDF="$INSTALL_DIR/logs/orchestrator.pid"
arg_rm=0
for a in "$@"; do [ "$a" = "--remove-containers" ] && arg_rm=1; done

# 1) /stop/all
echo "[..] Parando servicos (/stop/all)..."
if curl -sf -X POST "http://127.0.0.1:8090/stop/all" >/dev/null 2>&1; then
    echo "[OK] /stop/all ok"
else
    echo "[AVISO] orquestrador nao respondeu ao /stop/all (provavelmente ja parado)" >&2
fi

# 2) encerra o orquestrador
if [ -f "$PIDF" ]; then
    OPID="$(cat "$PIDF")"
    if kill -0 "$OPID" 2>/dev/null; then
        kill "$OPID" 2>/dev/null || true
        sleep 2
        kill -0 "$OPID" 2>/dev/null && kill -9 "$OPID" 2>/dev/null || true
        echo "[OK] Orquestrador PID $OPID encerrado"
    fi
    rm -f "$PIDF"
fi

# 3) remove containers de infra (opcional)
if [ "$arg_rm" = "1" ]; then
    for c in qopsec-postgres qopsec-mongo qopsec-mosquitto; do
        if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
            docker rm -f "$c" >/dev/null 2>&1 || true
            echo "[OK] container $c removido"
        fi
    done
fi

echo "[OK] Q-OPSEC parado."
