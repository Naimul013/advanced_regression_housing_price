"""Generate and validate predictions for the Kaggle competition."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.evaluate_tuned_model import resolve_path
from src.load_data import load_data_from_kaggle
from src.logger import get_logger


config = load_config()
logger = get_logger(__name__)

def load_test_data()-> pd.DataFrame:
    data_config = config['data']
    dataset = data_config['dataset']
    data_path = resolve_path(
        data_config.get('data_path', 'data/raw')
    )
    _, test_df = load_data_from_kaggle(dataset=dataset, data_path=data_path)

    if not isinstance(test_df, pd.DataFrame) or test_df.empty:
        raise ValueError("Competition test data is empty or invalid.")

    logger.info("Loaded competition test data: %d rows.", len(test_df))

    return test_df

def load_final_model():
    """Load the final fitted pipeline from its configured artifact path."""
    model_config = config["final_model_selection"]
    model_path = resolve_path(
        model_config["final_model_artifact_path"]
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Final model artifact was not found: {model_path}. "
            "Run the final-model training workflow first."
        )

    model = joblib.load(model_path)

    logger.info("Loaded final model artifact: %s", model_path)

    return model

def generate_predictions(
    model,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Generate finite, non-negative predictions for the test rows."""
    submission_config = config["submission"]
    id_column = submission_config["id_column"]
    target_column = submission_config["target_column"]

    if id_column not in test_df.columns:
        raise KeyError(f"Test data is missing '{id_column}'.")

    if target_column in test_df.columns:
        raise ValueError(
            "The test dataset unexpectedly contains the target column."
        )

    if test_df[id_column].isna().any():
        raise ValueError("Test identifiers contain missing values.")

    if test_df[id_column].duplicated().any():
        raise ValueError("Test identifiers are not unique.")

    # Competition identifiers must not be used as model features.
    X_test = test_df.copy()

    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        missing = set(expected_features) - set(X_test.columns)
        unexpected = set(X_test.columns) - set(expected_features)

        if missing or unexpected:
            raise ValueError(
                f"Test feature mismatch. Missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        X_test = X_test.loc[:, expected_features]


    predictions = np.asarray(
        model.predict(X_test),
        dtype=np.float64,
    ).reshape(-1)

    if len(predictions) != len(test_df):
        raise ValueError("Prediction count does not match test row count.")

    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    negative_count = int(np.sum(predictions < 0))

    # House prices and RMSLE predictions must be non-negative.
    predictions = np.maximum(predictions, 0.0)

    logger.info(
        "Generated %d predictions; clipped %d negative predictions.",
        len(predictions),
        negative_count,
    )

    return predictions

def save_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> Path:
    """Create, save, and validate the Kaggle submission CSV."""
    submission_config = config["submission"]

    id_column = submission_config["id_column"]
    target_column = submission_config["target_column"]
    output_path = resolve_path(submission_config["output_path"])

    if len(predictions) != len(test_df):
        raise ValueError("Prediction count does not match test row count.")

    submission = pd.DataFrame(
        {
            id_column: test_df[id_column].to_numpy(),
            target_column: predictions,
        }
    )

    if list(submission.columns) != [id_column, target_column]:
        raise ValueError("Submission columns do not match the required schema.")

    if submission.isna().any().any():
        raise ValueError("Submission contains missing values.")

    if not np.isfinite(
        submission[target_column].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Submission contains non-finite predictions.")

    if (submission[target_column] < 0).any():
        raise ValueError("Submission contains negative predictions.")

    if submission[id_column].duplicated().any():
        raise ValueError("Submission contains duplicate identifiers.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    # Verify the actual file written to disk.
    saved = pd.read_csv(output_path)

    if list(saved.columns) != [id_column, target_column]:
        raise ValueError("Saved submission has an invalid schema.")

    if len(saved) != len(test_df):
        raise ValueError("Saved submission has an incorrect row count.")

    if not saved[id_column].equals(
        test_df[id_column].reset_index(drop=True)
    ):
        raise ValueError("Saved submission identifiers are out of order.")

    if saved.isna().any().any():
        raise ValueError("Saved submission contains missing values.")

    logger.info(
        "Submission saved and verified: %s | rows=%d",
        output_path,
        len(saved),
    )

    return output_path


def main() -> None:
    """Run the submission-generation workflow."""
    logger.info("Starting Kaggle submission workflow.")

    test_df = load_test_data()
    model = load_final_model()
    predictions = generate_predictions(model, test_df)

    output_path = save_submission(test_df, predictions)

    logger.info("Submission workflow completed: %s", output_path)


if __name__ == "__main__":
    main()