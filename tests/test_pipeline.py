"""
tests/test_pipeline.py
Unit tests for the AI4I 2020 ML pipeline components.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np

from model.features import engineer_features, get_feature_columns, get_target_column


@pytest.fixture
def sample_row():
    return pd.DataFrame([{
        "air_temp_c":           25.1,
        "process_temp_c":       36.4,
        "rotational_speed_rpm": 1551.0,
        "torque_nm":            42.8,
        "tool_wear_min":        108.0,
        "product_type_encoded": 1,
    }])


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "air_temp_c":           np.random.uniform(20, 35, n),
        "process_temp_c":       np.random.uniform(30, 50, n),
        "rotational_speed_rpm": np.random.uniform(1200, 2000, n),
        "torque_nm":            np.random.uniform(10, 70, n),
        "tool_wear_min":        np.random.uniform(0, 250, n),
        "product_type_encoded": np.random.randint(0, 3, n),
        "machine_failure":      np.random.randint(0, 2, n),
    })


class TestFeatureEngineering:

    def test_adds_engineered_columns(self, sample_row):
        result = engineer_features(sample_row)
        for col in ["power_w", "temp_diff_c", "tool_wear_flag",
                    "torque_deviation", "speed_torque_index",
                    "thermal_ratio", "wear_torque_interaction"]:
            assert col in result.columns, f"Missing: {col}"

    def test_does_not_mutate_input(self, sample_row):
        cols = list(sample_row.columns)
        engineer_features(sample_row)
        assert list(sample_row.columns) == cols

    def test_power_w_calculation(self, sample_row):
        result = engineer_features(sample_row)
        expected = 42.8 * (2 * np.pi * 1551.0 / 60)
        assert abs(result["power_w"].iloc[0] - expected) < 0.01

    def test_temp_diff_calculation(self, sample_row):
        result = engineer_features(sample_row)
        assert abs(result["temp_diff_c"].iloc[0] - (36.4 - 25.1)) < 0.01

    def test_torque_deviation_non_negative(self, sample_row):
        result = engineer_features(sample_row)
        assert result["torque_deviation"].iloc[0] >= 0

    def test_tool_wear_flag_below_threshold(self, sample_row):
        result = engineer_features(sample_row)
        assert result["tool_wear_flag"].iloc[0] == 0   # 108 < 200

    def test_tool_wear_flag_above_threshold(self, sample_row):
        row = sample_row.copy()
        row["tool_wear_min"] = 210.0
        result = engineer_features(row)
        assert result["tool_wear_flag"].iloc[0] == 1

    def test_higher_torque_means_higher_deviation(self, sample_row):
        low  = sample_row.copy(); low["torque_nm"]  = 35.0
        high = sample_row.copy(); high["torque_nm"] = 70.0
        dev_low  = engineer_features(low)["torque_deviation"].iloc[0]
        dev_high = engineer_features(high)["torque_deviation"].iloc[0]
        assert dev_high > dev_low


class TestFeatureColumns:

    def test_returns_list(self):
        assert isinstance(get_feature_columns(), list)

    def test_no_duplicates(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols))

    def test_target_not_in_features(self):
        assert get_target_column() not in get_feature_columns()

    def test_all_features_present_after_engineering(self, sample_row):
        engineered = engineer_features(sample_row)
        for col in get_feature_columns():
            assert col in engineered.columns, f"Missing after engineering: {col}"


class TestDataIntegrity:

    def test_no_nan_after_engineering(self, sample_df):
        result = engineer_features(sample_df)
        assert result[get_feature_columns()].isna().sum().sum() == 0

    def test_no_infinite_values(self, sample_df):
        result = engineer_features(sample_df)
        assert not np.isinf(result[get_feature_columns()].values).any()