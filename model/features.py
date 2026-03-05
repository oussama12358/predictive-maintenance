"""
model/features.py
-----------------
Feature engineering for the AI4I 2020 Predictive Maintenance Dataset.
Source: Stephan Matzka — HTW Berlin, Germany
DOI   : https://doi.org/10.24432/C5HS5C

Dataset columns (after load_ai4i.py cleaning):
  air_temp_c            : Air temperature in Celsius
  process_temp_c        : Process temperature in Celsius
  rotational_speed_rpm  : Rotational speed in RPM
  torque_nm             : Torque in Newton-meters
  tool_wear_min         : Tool wear in minutes
  product_type_encoded  : Product quality (L=0, M=1, H=2)
  machine_failure       : TARGET — binary failure label

CRITICAL RULE:
    This file is the single source of truth for all feature transformations.
    Used by both train.py and app/predictor.py — never transform elsewhere.
"""

import pandas as pd
import numpy as np

NOMINAL_TORQUE_NM     = 40.0
NOMINAL_SPEED_RPM     = 1500.0
TOOL_WEAR_WARNING_MIN = 200


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    Works on full training DataFrames and single-row inference DataFrames.
    """
    df = df.copy()

    # Power (W) = Torque x Angular velocity
    # P = T x (2pi x rpm / 60)
    df["power_w"] = df["torque_nm"] * (2 * np.pi * df["rotational_speed_rpm"] / 60)

    # Temperature differential: process should be ~10C above air in normal ops
    # Large deviations indicate heat dissipation failure (HDF mode)
    df["temp_diff_c"] = df["process_temp_c"] - df["air_temp_c"]

    # Tool wear severity flag: > 200 min is warning zone for TWF failure
    df["tool_wear_flag"] = (df["tool_wear_min"] >= TOOL_WEAR_WARNING_MIN).astype(int)

    # Torque deviation from nominal 40Nm
    df["torque_deviation"] = np.abs(df["torque_nm"] - NOMINAL_TORQUE_NM)

    # Speed-Torque stress index: captures combined mechanical stress (OSF mode)
    df["speed_torque_index"] = df["rotational_speed_rpm"] * df["torque_nm"]

    # Thermal efficiency ratio
    df["thermal_ratio"] = df["process_temp_c"] / (df["air_temp_c"] + 1e-6)

    # Worn tool + high torque = compound TWF + OSF risk
    df["wear_torque_interaction"] = df["tool_wear_min"] * df["torque_nm"]

    return df


def get_feature_columns() -> list:
    """
    Ordered feature list — contract between pipeline and serving API.
    """
    return [
        "air_temp_c",
        "process_temp_c",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "product_type_encoded",
        "power_w",
        "temp_diff_c",
        "tool_wear_flag",
        "torque_deviation",
        "speed_torque_index",
        "thermal_ratio",
        "wear_torque_interaction",
    ]


def get_target_column() -> str:
    return "machine_failure"


def get_failure_mode_columns() -> list:
    return ["twf", "hdf", "pwf", "osf", "rnf"]