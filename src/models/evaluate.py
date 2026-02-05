import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.config.config import PathsConfig


def save_metrics(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_pred_vs_actual(y_true, y_pred, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals Plot")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def evaluate_and_report(processed_dir: Path, models_dir: Path, reports_dir: Path):
    # Load processed test data
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    # Load model
    model = joblib.load(models_dir / "model.joblib")

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": float(mse), "rmse": float(rmse), "r2": float(r2)}

    # Save metrics + plots
    metrics_path = reports_dir / "metrics.json"
    pred_plot_path = reports_dir / "pred_vs_actual.png"
    resid_plot_path = reports_dir / "residuals.png"

    save_metrics(metrics, metrics_path)
    plot_pred_vs_actual(y_test, y_pred, pred_plot_path)
    plot_residuals(y_test, y_pred, resid_plot_path)

    print("✅ Evaluation completed")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {pred_plot_path}")
    print(f"Saved: {resid_plot_path}")

    # Log to MLflow (logs to your latest active run if you start a new one)
    mlflow.set_experiment("california_housing_experiment")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(pred_plot_path))
        mlflow.log_artifact(str(resid_plot_path))

        print("✅ Logged evaluation artifacts to MLflow")


if __name__ == "__main__":
    cfg = PathsConfig()
    evaluate_and_report(cfg.data_processed_dir, cfg.models_dir, cfg.reports_dir)
