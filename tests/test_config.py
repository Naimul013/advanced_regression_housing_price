"""Sanity checks that the YAML configuration is complete and consistent."""

from src.config_loader import load_config


def test_config_has_required_sections():
    config = load_config()
    for section in (
        "data", "split", "preprocessor", "models",
        "cross_validation", "tuning", "submission",
    ):
        assert section in config


def test_ordinal_features_have_orders():
    ordinal = load_config()["preprocessor"]["ordinal"]
    for name, spec in ordinal.items():
        assert spec["order"], f"ordinal group '{name}' has no category order"


def test_tuning_search_space_bounds_are_valid():
    space = load_config()["tuning"]["search_space"]
    for name, bounds in space.items():
        assert bounds["low"] < bounds["high"], f"invalid bounds for {name}"
