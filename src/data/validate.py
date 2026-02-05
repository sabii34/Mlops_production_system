import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config.config import PathsConfig


@dataclass
class ValidationReport:
    file_path: str
    rows: int
    cols: int
    missing_values_total: int
    missing_by_column: dict
    duplicate_rows: int
    passed: bool
    issues: list


def validate_raw_csv(csv_path: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    issues = []

    # Basic checks
    rows, cols = df.shape
    if rows == 0:
        issues.append("Dataset has 0 rows.")
    if cols == 0:
        issues.append("Dataset has 0 columns.")

    # Missing values
    missing_by_col = df.isnull().sum().to_dict()
    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        issues.append(f"Dataset has missing values: total={missing_total}")

    # Duplicates
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        issues.append(f"Dataset has duplicate rows: {dup_rows}")

    passed = len(issues) == 0

    report = ValidationReport(
        file_path=str(csv_path),
        rows=int(rows),
        cols=int(cols),
        missing_values_total=missing_total,
        missing_by_column=missing_by_col,
        duplicate_rows=dup_rows,
        passed=passed,
        issues=issues,
    )

    report_path = report_dir / "data_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print(f"✅ Validation report saved to: {report_path}")
    print(f"Passed: {passed}")
    if not passed:
        print("Issues found:")
        for i in issues:
            print(f" - {i}")

    return report_path


if __name__ == "__main__":
    cfg = PathsConfig()

    # Change filename if your raw file name is different
    raw_csv = cfg.data_raw_dir / "california_housing_raw.csv"

    validate_raw_csv(raw_csv, cfg.reports_dir)
