from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.config.config import PathsConfig


def build_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features)
        ]
    )

    return X, y, preprocessor


def transform_and_save(
    raw_csv: Path,
    processed_dir: Path,
    models_dir: Path,
    target_col: str = "MedHouseVal"
):
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv)

    X, y, preprocessor = build_preprocessor(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Save processed datasets
    pd.DataFrame(X_train_transformed).to_csv(
        processed_dir / "X_train.csv", index=False
    )
    pd.DataFrame(X_test_transformed).to_csv(
        processed_dir / "X_test.csv", index=False
    )
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    # Save preprocessing object
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")

    print("✅ Data transformation completed")
    print("Saved:")
    print(" - X_train.csv")
    print(" - X_test.csv")
    print(" - y_train.csv")
    print(" - y_test.csv")
    print(" - preprocessor.joblib")


if __name__ == "__main__":
    cfg = PathsConfig()
    raw_csv_path = cfg.data_raw_dir / "california_housing_raw.csv"

    transform_and_save(
        raw_csv=raw_csv_path,
        processed_dir=cfg.data_processed_dir,
        models_dir=cfg.models_dir
    )
