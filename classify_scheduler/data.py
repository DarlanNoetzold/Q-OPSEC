import pandas as pd
from pathlib import Path

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" in df.columns:
        # manter “id” se necessário para auditoria; não é usado como feature
        pass
    return df