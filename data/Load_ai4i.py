"""
data/load_ai4i.py
-----------------
Loader and preprocessor for the AI4I 2020 Predictive Maintenance Dataset.

Source  : UCI Machine Learning Repository
Author  : Stephan Matzka — HTW Berlin, Germany
DOI     : https://doi.org/10.24432/C5HS5C
Kaggle  : https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

Dataset schema (14 columns):
  UID                  : unique identifier [1, 10000]
  Product ID           : L/M/H quality variant + serial number
  Type                 : product quality type L=Low, M=Medium, H=High
  Air temperature [K]  : air temperature in Kelvin
  Process temperature [K]: process temperature in Kelvin
  Rotational speed [rpm]: rotational speed calculated from power of 2860W
  Torque [Nm]          : torque in Nm, normally distributed ~40Nm, sigma=10
  Tool wear [min]      : quality variant adds 5/3/2 minutes tool wear
  Machine failure      : binary target — 1 if any failure mode occurred
  TWF                  : Tool Wear Failure
  HDF                  : Heat Dissipation Failure
  PWF                  : Power Failure
  OSF                  : Overstrain Failure
  RNF                  : Random Failure

Usage:
    # Option 1: Auto-download via ucimlrepo (recommended)
    python data/load_ai4i.py

    # Option 2: Manual CSV
    Download ai4i2020.csv from Kaggle and place in data/raw/
    then run: python data/load_ai4i.py --source csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("data/raw/ai4i2020.csv")


def download_from_uci() -> pd.DataFrame:
    """
    Download AI4I 2020 directly from UCI using the official ucimlrepo package.
    Requires: pip install ucimlrepo
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo not installed. Run: pip install ucimlrepo\n"
            "Or download manually from Kaggle and use --source csv"
        )

    print(" Downloading AI4I 2020 from UCI ML Repository...")
    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)
    print(f"✅ Downloaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_from_csv(csv_path: str = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    """
    Load AI4I 2020 from a local CSV file.
    Download from: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at {path}\n"
            "Download from: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020\n"
            "Or run: python data/load_ai4i.py (auto-download via ucimlrepo)"
        )
    df = pd.read_csv(path)
    print(f"✅ Loaded CSV: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and types for the rest of the pipeline.

    Raw UCI column names have spaces and brackets — we normalize them
    to snake_case for clean Python attribute access.
    """
    # Normalize column names: lowercase, replace spaces/brackets with underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r"[\s\[\]()]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # Rename to clean standard names used across the pipeline
    rename_map = {
        "air_temperature_k"      : "air_temp_k",
        "process_temperature_k"  : "process_temp_k",
        "rotational_speed_rpm"   : "rotational_speed_rpm",
        "torque_nm"              : "torque_nm",
        "tool_wear_min"          : "tool_wear_min",
        "machine_failure"        : "machine_failure",
        "type"                   : "product_type",
    }

    # Apply only existing renames (handles slight UCI/Kaggle naming differences)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert product type to numeric (L=0, M=1, H=2)
    if "product_type" in df.columns:
        type_map = {"L": 0, "M": 1, "H": 2}
        df["product_type_encoded"] = df["product_type"].map(type_map).fillna(0).astype(int)

    # Convert Kelvin to Celsius (more intuitive for engineers)
    if "air_temp_k" in df.columns:
        df["air_temp_c"] = (df["air_temp_k"] - 273.15).round(2)
    if "process_temp_k" in df.columns:
        df["process_temp_c"] = (df["process_temp_k"] - 273.15).round(2)

    # Drop non-feature columns
    drop_cols = ["uid", "product_id", "air_temp_k", "process_temp_k", "product_type"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Ensure target is integer
    if "machine_failure" in df.columns:
        df["machine_failure"] = df["machine_failure"].astype(int)

    print(f"   Columns after cleaning: {list(df.columns)}")
    failure_rate = df["machine_failure"].mean()
    print(f"   Failure rate          : {failure_rate:.1%}  ({df['machine_failure'].sum()} failures)")
    print(f"   Failure breakdown:")
    for col in ["twf", "hdf", "pwf", "osf", "rnf"]:
        if col in df.columns:
            print(f"     {col.upper()}: {df[col].sum():4d} cases")

    return df


def save_dataset(df: pd.DataFrame, output_path: str = str(OUTPUT_PATH)) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n Dataset saved to: {path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load AI4I 2020 Predictive Maintenance Dataset")
    parser.add_argument("--source", choices=["uci", "csv"], default="uci",
                        help="uci = auto-download, csv = load from data/raw/ai4i2020.csv")
    parser.add_argument("--csv-path", default="data/raw/ai4i2020.csv",
                        help="Path to CSV file (only used with --source csv)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  AI4I 2020 — Dataset Loader")
    print("  Source: HTW Berlin / UCI ML Repository")
    print("="*55 + "\n")

    if args.source == "uci":
        df_raw = download_from_uci()
    else:
        df_raw = load_from_csv(args.csv_path)

    print("\nCleaning and standardizing columns...")
    df_clean = clean_and_standardize(df_raw)

    save_dataset(df_clean)