#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/umbrel/projetos/Q-OPSEC/risk_service_V2
source /home/umbrel/projetos/Q-OPSEC/qopsec_env/bin/activate
cd /home/umbrel/projetos/Q-OPSEC/risk_service_V2
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
