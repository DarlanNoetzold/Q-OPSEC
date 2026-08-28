import pandas as pd
from pathlib import Path

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" in df.columns:
        pass
    return df