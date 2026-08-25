from src.dataset_generation.orchestrator import DatasetOrchestrator
import os
from pathlib import Path

def main():
    config_path = "dataset_config.yaml"
    output_dir = "output"
    
    print(f"Iniciando gerador com config: {config_path}")
    orchestrator = DatasetOrchestrator(config_path=config_path, output_dir=output_dir)
    orchestrator.run()

if __name__ == "__main__":
    main()
