import sys
import os
import subprocess
from pathlib import Path
import yaml

# Adiciona o diretório do projeto ao sys.path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR))

try:
    from src.dataset_generation.orchestrator import DatasetOrchestrator
    from src.common.logger import logger
except ImportError:
    print("Error: Could not import DatasetOrchestrator. Are you in the right directory?")
    sys.exit(1)

def run():
    config_path = BASE_DIR / "config" / "dataset_config.yaml"
    output_dir = BASE_DIR / "output"
    
    logger.info(f"PhD Autonomous Generator starting with config: {config_path}")
    
    try:
        # 1. Gerar Dataset
        orchestrator = DatasetOrchestrator(str(config_path), str(output_dir))
        orchestrator.run()
        
        # 2. Se a geração foi bem sucedida, disparar o treinamento automaticamente
        # (PhD Research flow: Generate -> Train -> Eval)
        logger.info("Dataset generation successful. Triggering model training...")
        
        train_script = BASE_DIR / "train_model.py"
        venv = "/home/umbrel/projetos/Q-OPSEC/qopsec_env/bin/python3"
        
        # Dispara o treino em um processo separado para não travar o dashboard
        subprocess.Popen(
            [venv, str(train_script)],
            cwd=str(BASE_DIR),
            stdout=open("/home/umbrel/projetos/Q-OPSEC/logs/risk_v2_auto_phd.log", "a"),
            stderr=subprocess.STDOUT
        )
        
        logger.info("Training triggered in background. Check logs/risk_v2_auto_phd.log")
        
    except Exception as e:
        logger.error(f"PhD Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run()
