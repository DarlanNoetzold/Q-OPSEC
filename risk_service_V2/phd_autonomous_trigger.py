import sys
import os
from pathlib import Path
import subprocess

# Garante que o diretório base do risk_service_V2 esteja no path
BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.dataset_generation.orchestrator import DatasetOrchestrator
    from src.common.logger import get_logger
    logger = get_logger("phd_trigger")
except ImportError as e:
    print(f"Error: Could not import DatasetOrchestrator: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Working directory: {os.getcwd()}")
    sys.exit(1)

def run():
    config_path = BASE_DIR / "config" / "dataset_config.yaml"
    output_dir = BASE_DIR / "output"
    
    logger.info(f"Autonomous Generator starting with config: {config_path}")
    
    try:
        # 1. Gerar Dataset
        orchestrator = DatasetOrchestrator(str(config_path), str(output_dir))
        orchestrator.run()
        
        # 2. Se a geração foi bem sucedida, disparar o treinamento automaticamente
        logger.info("Dataset generation successful. Triggering model training...")
        
        train_script = BASE_DIR / "train_model.py"
        venv = "/home/umbrel/projetos/Q-OPSEC/qopsec_env/bin/python3"
        
        # Dispara o treino em um processo separado
        with open("/home/umbrel/projetos/Q-OPSEC/logs/risk_v2_auto_phd.log", "a") as log_file:
            subprocess.Popen(
                [venv, str(train_script)],
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        
        logger.info("Training triggered in background. Check logs/risk_v2_auto_phd.log")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run()
