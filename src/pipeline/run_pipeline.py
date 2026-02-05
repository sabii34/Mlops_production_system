from src.config.config import PathsConfig
from src.data.ingest import ingest_california_housing
from src.data.validate import validate_raw_csv
from src.features.build_features import transform_and_save
from src.models.train import train_and_track
from src.models.evaluate import evaluate_and_report


def run_pipeline():
    cfg = PathsConfig()

    print("\n🚀 Starting End-to-End ML Pipeline\n")

    # Stage 1: Ingestion
    raw_csv = ingest_california_housing(cfg.data_raw_dir)

    # Stage 2: Validation
    validate_raw_csv(raw_csv, cfg.reports_dir)

    # Stage 3: Transformation
    transform_and_save(
        raw_csv=raw_csv,
        processed_dir=cfg.data_processed_dir,
        models_dir=cfg.models_dir
    )

    # Stage 4: Training
    train_and_track(
        processed_dir=cfg.data_processed_dir,
        models_dir=cfg.models_dir
    )

    # Stage 5: Evaluation
    evaluate_and_report(
        processed_dir=cfg.data_processed_dir,
        models_dir=cfg.models_dir,
        reports_dir=cfg.reports_dir
    )

    print("\n✅ Pipeline executed successfully!\n")


if __name__ == "__main__":
    run_pipeline()
