import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

# Mock para os imports que falhariam sem o ambiente completo
class MockSettings:
    ml_registry_dir = "/mnt/c/Projetos/Q-OPSEC/classify_scheduler/model_registry"
    ml_registry_latest_file = "latest.json"

# Injeta mocks no sys.modules para o script carregar
class MockLog:
    def info(self, *args, **kwargs): print(f"INFO: {args} {kwargs}")
    def error(self, *args, **kwargs): print(f"ERROR: {args} {kwargs}")
    def get_logger(self): return self

# Mocking app structures
sys.modules['..core.config'] = type('obj', (object,), {'settings': MockSettings()})
sys.modules['structlog'] = MockLog()

# Agora importamos o ModelService para testar
from app.services.model_service import ModelService

async def test():
    ms = ModelService()
    print("Iniciando teste de carregamento...")
    try:
        await ms.load_latest_model(force=True)
        print(f"Modelo carregado: {ms.model_name}")
        
        # Teste de predicao
        test_data = {"feature1": 0.5, "feature2": 1.2} # Ajuste conforme real
        results, probs, tid = ms.predict(test_data)
        print(f"Predicao: {results}, Prob: {probs}, ID: {tid}")
        print("TESTE OK!")
    except Exception as e:
        print(f"ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
