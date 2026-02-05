from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path = Path(__file__).resolve().parents[2]  # project root
    data_raw_dir: Path = base_dir / "data" / "raw"
    data_processed_dir: Path = base_dir / "data" / "processed"
    reports_dir: Path = base_dir / "reports"
    models_dir: Path = base_dir / "models"
