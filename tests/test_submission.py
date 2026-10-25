"""Tests for Kaggle submission validation."""

import numpy as np
import pandas as pd
import pytest

from src import submission


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame({"Id": [1461, 1462, 1463], "LotArea": [8000, 9000, 7500]})


@pytest.fixture(autouse=True)
def tmp_output(tmp_path, monkeypatch):
    """Redirect the submission file into a temporary directory."""
    path = tmp_path / "submission.csv"
    monkeypatch.setattr(submission, "resolve_path", lambda _: path)
    return path


def test_valid_submission_is_written_with_required_schema(test_df, tmp_output):
    out = submission.save_submission(test_df, np.array([150000.0, 200000.0, 175000.0]))
    saved = pd.read_csv(out)

    assert list(saved.columns) == ["Id", "SalePrice"]
    assert saved["Id"].tolist() == test_df["Id"].tolist()


def test_wrong_prediction_count_is_rejected(test_df):
    with pytest.raises(ValueError):
        submission.save_submission(test_df, np.array([150000.0]))


def test_negative_predictions_are_rejected(test_df):
    with pytest.raises(ValueError):
        submission.save_submission(test_df, np.array([1.0, -5.0, 3.0]))


def test_non_finite_predictions_are_rejected(test_df):
    with pytest.raises(ValueError):
        submission.save_submission(test_df, np.array([1.0, np.nan, 3.0]))
