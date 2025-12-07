import pandas as pd
from pathlib import Path

def save_parquet(df, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

def load_parquet(path):
    return pd.read_parquet(path)
