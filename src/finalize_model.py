"""Train and persist the final model using all labelled competition data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.config_loader import load_config
from src.load_data import load_data_from_kaggle
from src.logger import get_logger
from src.train import model_pipeline
from src.evaluate_tuned_model import resolve_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

config = load_config()
logger = get_logger(__name__)

def load_best_study()-> optuna.Study:
    tuning_config = config['tuning']
    model_name = tuning_config['model_name']

    storage_path = resolve_path(tuning_config['storage_path'])

    if not storage_path.is_file():
        raise FileNotFoundError(
            f"Optuna database was not found: {storage_path}"
        )
    storage_url = f'sqlite:///{storage_path.as_posix()}'
    study = optuna.load_study(
        study_name=tuning_config['study_name'],
        storage= storage_url,
    )
    if not study.trials:
        raise ValueError("The Optuna study contains no trials.")

    if study.best_trial is None:
        raise ValueError("The Optuna study has no completed best trial.")

    logger.info(
        "Loaded Optuna study '%s'. Best trial: %s | CV RMSLE: %.6f",
        tuning_config["study_name"],
        study.best_trial.number,
        study.best_value,
    )

    return study

def load_train_dataset()->tuple[pd.DataFrame, pd.Series]:
    """Load all labelled competition rows without creating a holdout split."""
    data_config = config["data"]

    dataset = data_config["dataset"]
    data_path = resolve_path(data_config["data_path"])
    target_column = data_config["target_column"]

    train_df, _ = load_data_from_kaggle(
        dataset=dataset,
        data_path=str(data_path),
    )
    if not isinstance(train_df, pd.DataFrame) or train_df.empty:
        raise ValueError("The labelled training dataset is empty or invalid.")

    if target_column not in train_df.columns:
        raise KeyError(
            f"Target column '{target_column}' is missing from training data."
        )

    if train_df[target_column].isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values."
        )

    X = train_df.drop(columns=target_column)
    y = train_df[target_column]

    y_values = y.to_numpy(dtype=np.float64)

    if not np.isfinite(y_values).all():
        raise ValueError("The target contains non-finite values.")

    if np.any(y_values < 0):
        raise ValueError("The target contains negative housing prices.")

    if X.empty or len(X) != len(y):
        raise ValueError("The training features or target are invalid.")

    logger.info(
        "Loaded full training dataset: %d rows, %d features.",
        X.shape[0],
        X.shape[1],
    )

    return X, y

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    study: optuna.Study,
)-> Any:

    """Build and fit the final pipeline using the best Optuna parameters."""
    tuning_config = config['tuning']
    model_name = tuning_config['model_name']
    best_params = study.best_params.copy()

    if model_name != "GradientBoostingRegressor":
        raise ValueError(
            "This finalization script currently supports "
            "'GradientBoostingRegressor' only."
        )

    random_state = tuning_config['random_state']

    final_estimator = GradientBoostingRegressor(
        **best_params,
        random_state=random_state
    )

    final_pipeline = model_pipeline(
        model_name=model_name,
        models= final_estimator,
        config=config
    )
    logger.info(
        "Fitting final %s pipeline on %d labelled rows.",
        model_name,
        len(X),
    )

    final_pipeline.fit(X,y)
    logger.info("Final model training completed successfully.")

    return final_pipeline

def save_final_artifacts(
    model: Any,
    study: optuna.Study,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Persist the final pipeline and its reproducibility metadata."""
    tuning_config = config['tuning']
    final_model_config = config['final_model_selection']

    model_path = resolve_path(
        final_model_config.get('final_model_artifact_path','models/gradient_boosting_final.joblib')

    )

    metadata_path = resolve_path(
        config['final_model_selection'].get('final_metadata_path','reports/final_model_metadata.json')
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    metadata = {
        "model_name": tuning_config["model_name"],
        "training_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target_column": config["data"]["target_column"],
        "optuna_study_name": tuning_config["study_name"],
        "best_trial_number": int(study.best_trial.number),
        "best_cv_rmsle": float(study.best_value),
        "best_params": study.best_params,
        "historical_validation_rmsle": 0.135991,
        "validation_score_note": (
            "Historical model-selection score from the holdout validation "
            "partition. It is not an independent evaluation of this final "
            "model, which was trained on all labelled rows."
        ),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(model_path.relative_to(PROJECT_ROOT)),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, allow_nan=False)
        file.write("\n")

    logger.info("Final model saved to: %s", model_path)
    logger.info("Final model metadata saved to: %s", metadata_path)


def verify_saved_model(
    X: pd.DataFrame,
    model_path: Path,
) -> None:
    """Verify that the saved artifact loads and produces valid predictions."""
    loaded_model = joblib.load(model_path)

    # Verify inference on a small sample without refitting the model.
    sample = X.head(min(5, len(X)))
    predictions = np.asarray(
        loaded_model.predict(sample),
        dtype=np.float64,
    ).reshape(-1)

    if len(predictions) != len(sample):
        raise ValueError("Saved model returned an unexpected prediction count.")

    if not np.isfinite(predictions).all():
        raise ValueError("Saved model returned non-finite predictions.")

    logger.info(
        "Artifact verification passed: %d sample predictions generated.",
        len(predictions),
    )

def main():
    """Execute the final model-training workflow."""
    logger.info("Starting final model-training workflow.")

    study = load_best_study()
    X,y = load_train_dataset()

    final_model = train_final_model(
        X=X,
        y=y,
        study=study
    )

    save_final_artifacts(model=final_model, study=study, X=X, y=y)

    model_path = resolve_path(
        config["final_model_selection"].get(
            "final_model_artifact_path",
            "models/gradient_boosting_final.joblib",
        )
    )

    verify_saved_model(X=X, model_path= model_path)
    logger.info("Final model workflow completed successfully.")


if __name__ == "__main__":
    main()






 



