from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.config.config import PathsConfig


def train_and_track(
    processed_dir: Path,
    models_dir: Path,
    experiment_name: str = "california_housing_experiment",
):
    # Load processed data
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        # Log params & metrics
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Save model
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("✅ Model training completed")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    cfg = PathsConfig()

    train_and_track(
        processed_dir=cfg.data_processed_dir,
        models_dir=cfg.models_dir,
    )
