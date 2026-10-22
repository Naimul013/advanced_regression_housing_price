
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from src.config_loader import load_config
from src.load_data import load_data_from_kaggle
from src.logger import get_logger
from src.evaluate_tuned_model import resolve_path


logger = get_logger(__name__)


def count_invalid_values(data) -> tuple[int, int]:
    """Return the number of NaN and infinite numeric values."""
    if sparse.issparse(data):
        data = data.toarray()
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    array = np.asarray(data)

    try:
        numeric = array.astype(float)
    except (TypeError, ValueError):
        return -1, -1

    return (
        int(np.isnan(numeric).sum()),
        int(np.isinf(numeric).sum()),
    )


def print_original_missing_values(X: pd.DataFrame) -> None:
    """Report missing values in the original test data."""
    missing = X.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\nORIGINAL TEST DATA")
    print("=" * 70)
    print(f"Shape: {X.shape}")
    print(f"Duplicate column names: {X.columns.duplicated().any()}")
    print(f"Duplicate Id values: {X['Id'].duplicated().any() if 'Id' in X else 'Id unavailable'}")
    print(f"Columns containing NaNs: {len(missing)}")
    print(f"Total NaNs: {int(missing.sum())}")

    if not missing.empty:
        print("\nMissing-value counts:")
        print(missing.to_string())


def report_output(name: str, output) -> None:
    """Print output shape and numeric invalid-value counts."""
    if sparse.issparse(output):
        shape = output.shape
    else:
        shape = np.asarray(output).shape

    nan_count, inf_count = count_invalid_values(output)

    print(f"\n{name}")
    print(f"Output shape: {shape}")

    if nan_count == -1:
        print("Output contains non-numeric values.")
        return

    print(f"NaN count: {nan_count}")
    print(f"Infinite count: {inf_count}")


def get_source_columns(
    X: pd.DataFrame,
    selector,
) -> list[str] | None:
    """Resolve a ColumnTransformer column selector."""
    if isinstance(selector, str):
        return [selector]

    if isinstance(selector, slice):
        return list(X.columns[selector])

    if isinstance(selector, (list, tuple, np.ndarray, pd.Index)):
        if len(selector) == 0:
            return []

        if all(isinstance(item, str) for item in selector):
            return list(selector)

        if all(isinstance(item, (int, np.integer)) for item in selector):
            return list(X.columns[list(selector)])

    # Callable or other complex selectors are handled separately.
    return None


def diagnose_column_transformer(
    column_transformer,
    X: pd.DataFrame,
) -> None:
    """Inspect each ColumnTransformer branch for remaining NaNs."""
    print("\nCOLUMNTRANSFORMER BRANCH DIAGNOSTICS")
    print("=" * 70)

    for name, transformer, selector in column_transformer.transformers_:
        if transformer == "drop":
            continue

        print(f"\nBranch: {name}")

        try:
            if callable(selector):
                resolved = selector(X)
                if isinstance(resolved, (list, tuple, np.ndarray, pd.Index)):
                    if len(resolved) and all(
                        isinstance(item, (bool, np.bool_))
                        for item in resolved
                    ):
                        source_columns = list(X.columns[np.asarray(resolved)])
                    elif len(resolved) and all(
                        isinstance(item, (int, np.integer))
                        for item in resolved
                    ):
                        source_columns = list(X.columns[list(resolved)])
                    else:
                        source_columns = list(resolved)
                else:
                    print(f"Unsupported selector result: {resolved!r}")
                    continue
            else:
                source_columns = get_source_columns(X, selector)

            if source_columns is None:
                print(f"Could not resolve selector: {selector!r}")
                continue

            print(f"Source columns: {source_columns}")

            if transformer == "passthrough":
                output = X.loc[:, source_columns].to_numpy()
            else:
                output = transformer.transform(X.loc[:, source_columns])

            if sparse.issparse(output):
                output = output.toarray()
            elif isinstance(output, pd.DataFrame):
                output = output.to_numpy()

            output = np.asarray(output)

            try:
                numeric = output.astype(float)
            except (TypeError, ValueError):
                print("Branch output is not entirely numeric.")
                continue

            nan_mask = np.isnan(numeric)
            inf_mask = np.isinf(numeric)

            print(f"Output shape: {numeric.shape}")
            print(f"NaN count: {int(nan_mask.sum())}")
            print(f"Infinite count: {int(inf_mask.sum())}")

            if nan_mask.any():
                affected_rows = np.flatnonzero(nan_mask.any(axis=1))
                affected_columns = np.flatnonzero(nan_mask.any(axis=0))

                print(f"Affected test row indices: {affected_rows.tolist()}")

                print("Affected branch output columns:")
                for output_column in affected_columns:
                    rows = np.flatnonzero(nan_mask[:, output_column])
                    print(
                        f"  Output column {output_column}: "
                        f"{len(rows)} NaNs; rows={rows.tolist()}"
                    )

                print("Original input values for affected rows:")
                print(
                    X.iloc[affected_rows]
                    .loc[:, source_columns]
                    .to_string()
                )

        except Exception as exc:
            print(f"Branch diagnostic failed: {type(exc).__name__}: {exc}")


def main() -> None:
    """Load the existing model and diagnose inference preprocessing."""
    config = load_config()

    data_config = config["data"]
    target_column = data_config.get("target_column", "SalePrice")

    _,test_df = load_data_from_kaggle(
        dataset=data_config["dataset"],
        data_path=data_config["data_path"],
    )

    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("The data loader did not return a pandas DataFrame.")

    # Accommodate loaders that return both train and test data.
    if isinstance(test_df, tuple):
        raise TypeError(
            "The data loader returned a tuple. Unpack the test DataFrame "
            "using the existing loader's documented return structure."
        )

    # If the loader returns the competition training set, this is not
    # the intended inference dataset. Stop rather than predict incorrectly.
    if target_column in test_df.columns:
        raise ValueError(
            f"The loaded DataFrame contains target column {target_column!r}. "
            "Ensure the loader returns the competition test.csv DataFrame."
        )

    print_original_missing_values(test_df)

    model_path = resolve_path(
        config["final_model_selection"]["final_model_artifact_path"]
    )

    model = joblib.load(model_path)

    print(f"\nSaved model: {model_path}")
    print(f"Model type: {type(model).__name__}")

    if not hasattr(model, "named_steps"):
        raise TypeError("The saved model is not a scikit-learn Pipeline.")

    if "preprocessor" not in model.named_steps:
        raise KeyError("The saved model has no 'preprocessor' step.")

    preprocessor = model.named_steps["preprocessor"]

    # Keep Id because the saved model was trained with Id as an input feature.
    X_test = test_df.copy()

    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        missing_features = sorted(set(expected_features) - set(X_test.columns))
        extra_features = sorted(set(X_test.columns) - set(expected_features))

        print("\nFEATURE SCHEMA CHECK")
        print(f"Missing expected features: {missing_features}")
        print(f"Unexpected features: {extra_features}")

        if missing_features:
            raise ValueError(
                f"Test data is missing required features: {missing_features}"
            )

        X_test = X_test.loc[:, list(expected_features)]

    report_output("RAW MODEL INPUT", X_test)

    print("\nSTAGE-BY-STAGE PREPROCESSING")
    print("=" * 70)

    X_stage = X_test.copy()

    for stage_name, transformer in preprocessor.steps:
        print(f"\nRunning stage: {stage_name}")
        transformer.fit(X_stage)
        X_stage = transformer.transform(X_stage)
        report_output(f"After stage: {stage_name}", X_stage)

        if stage_name == "columntransformer":
            diagnose_column_transformer(transformer, X_test)

    nan_count, inf_count = count_invalid_values(X_stage)

    print("\nFINAL ESTIMATOR CHECK")
    print("=" * 70)
    print(f"Estimator: {type(model.named_steps['model']).__name__}")
    print(f"Final NaN count: {nan_count}")
    print(f"Final infinite count: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print(
            "\nDIAGNOSIS: Invalid numeric values remain after preprocessing. "
            "Inspect the branch diagnostics above."
        )
        return

    print("\nPreprocessing output contains no NaNs or infinities.")

    try:
        predictions = model.predict(X_test)
        print(f"Prediction successful. Number of predictions: {len(predictions)}")
        print(f"First five predictions: {predictions[:5]}")
    except Exception as exc:
        logger.exception("Prediction verification failed.")
        print(f"Prediction failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
