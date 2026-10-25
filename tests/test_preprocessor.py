"""Unit tests for the custom preprocessing transformers."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import (
    LotFrontageImputer,
    MasVnrFeatureEngineer,
    SkewnessTransformer,
    StructuralFeatureEngineer,
    ordinal_pipeline,
)


def make_structural() -> StructuralFeatureEngineer:
    return StructuralFeatureEngineer(
        existence_feature={"GarageExist": "GarageYrBlt"},
        numerical_features=["GarageYrBlt", "GarageArea"],
        categorical_features=["GarageType"],
        absence_feature="None",
    )


def test_structural_creates_existence_flag_and_fills_missing():
    df = pd.DataFrame(
        {
            "GarageYrBlt": [2001.0, np.nan],
            "GarageArea": [400.0, np.nan],
            "GarageType": ["Attchd", np.nan],
        }
    )
    out = make_structural().fit(df).transform(df)

    assert out["GarageExist"].tolist() == [1, 0]
    assert out.loc[1, "GarageArea"] == 0
    assert out.loc[1, "GarageType"] == "None"


def test_structural_does_not_modify_input():
    df = pd.DataFrame(
        {"GarageYrBlt": [np.nan], "GarageArea": [np.nan], "GarageType": [np.nan]}
    )
    make_structural().fit(df).transform(df)
    assert df["GarageArea"].isna().all()


def test_structural_fit_raises_on_missing_column():
    df = pd.DataFrame({"GarageYrBlt": [1.0]})
    with pytest.raises(ValueError):
        make_structural().fit(df)


def test_masvnr_leaves_no_missing_values():
    df = pd.DataFrame(
        {
            "MasVnrArea": [120.0, 200.0, 0.0, np.nan, 1.0],
            "MasVnrType": ["BrkFace", "BrkFace", "BrkFace", np.nan, np.nan],
        }
    )
    out = MasVnrFeatureEngineer().fit(df).transform(df)

    assert out["MasVnrArea"].notna().all()
    assert out["MasVnrType"].notna().all()


def test_masvnr_fills_missing_type_when_area_is_large():
    train = pd.DataFrame(
        {
            "MasVnrArea": [120.0, 200.0, 180.0],
            "MasVnrType": ["BrkFace", "BrkFace", "Stone"],
        }
    )
    test = pd.DataFrame(
        {"MasVnrArea": [150.0], "MasVnrType": pd.Series([np.nan], dtype=object)}
    )
    transformer = MasVnrFeatureEngineer().fit(train)

    assert transformer.transform(test).loc[0, "MasVnrType"] == "BrkFace"


def test_lot_frontage_uses_neighborhood_median_then_overall_fallback():
    train = pd.DataFrame(
        {
            "Neighborhood": ["A", "A", "A", "B"],
            "LotFrontage": [60.0, 70.0, 80.0, 100.0],
        }
    )
    test = pd.DataFrame(
        {"Neighborhood": ["A", "Unseen"], "LotFrontage": [np.nan, np.nan]}
    )
    imputer = LotFrontageImputer("neighborhood_median", "overall_median").fit(train)
    out = imputer.transform(test)

    assert out.loc[0, "LotFrontage"] == 70.0  # median of neighbourhood A
    assert out.loc[1, "LotFrontage"] == 75.0  # overall median fallback


def test_skewness_applies_log1p():
    df = pd.DataFrame({"LotArea": [0.0, np.e - 1]})
    out = SkewnessTransformer(["LotArea"], "log1p").fit(df).transform(df)

    assert out["LotArea"].tolist() == pytest.approx([0.0, 1.0])


def test_skewness_rejects_unsupported_method():
    df = pd.DataFrame({"LotArea": [1.0]})
    with pytest.raises(ValueError):
        SkewnessTransformer(["LotArea"], "sqrt").fit(df)


def test_ordinal_pipeline_orders_categories_and_flags_unknown():
    order = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
    pipeline = ordinal_pipeline(["ExterQual"], order)
    train = pd.DataFrame({"ExterQual": ["TA", "Ex", np.nan]})
    pipeline.fit(train)

    out = pipeline.transform(pd.DataFrame({"ExterQual": ["Gd", "Po", "Weird"]}))
    assert out.ravel().tolist() == [4, 1, -1]
