from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing

from src.config.config import PathsConfig


def ingest_california_housing(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = fetch_california_housing(as_frame=True)
    df = data.frame  # includes target column

    out_path = output_dir / "california_housing_raw.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Saved raw dataset to: {out_path}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    return out_path


if __name__ == "__main__":
    cfg = PathsConfig()
    ingest_california_housing(cfg.data_raw_dir)
