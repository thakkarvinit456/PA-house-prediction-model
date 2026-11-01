"""
Dataset Loader Module for House Price Prediction System
Dataset: Ames Housing Dataset (Dean De Cock, 2011)
Source: OpenML (ID: 543) / Kaggle Advanced Regression Techniques
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "house_prices.csv")

DATASET_METADATA = {
    "name": "Ames Housing Dataset",
    "author": "Dean De Cock (Truman State University)",
    "publication": "Journal of Statistics Education, Volume 19, Number 3 (2011)",
    "source_url": "https://www.openml.org/d/543",
    "target_variable": "SalePrice",
    "target_unit": "USD ($)",
    "task_type": "Supervised Machine Learning - Continuous Regression",
    "description": (
        "The Ames Housing dataset contains 81 residential real estate features "
        "describing houses sold in Ames, Iowa between 2006 and 2010. It serves as a modern, "
        "high-dimensional, and realistic replacement for the classic Boston Housing dataset, "
        "incorporating structural, spatial, environmental, and temporal property attributes."
    ),
    "primary_key": "Id"
}


def load_raw_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads the Ames Housing dataset from local CSV file.
    If the file does not exist locally, attempts to fetch it via OpenML.
    """
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print(f"Data file not found at {data_path}. Attempting to fetch from OpenML...")
        from sklearn.datasets import fetch_openml
        housing = fetch_openml(name="house_prices", as_frame=True, parser="auto")
        df = housing.frame
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Dataset successfully fetched and cached to {data_path}")

    # Standardize data types
    if "Id" in df.columns:
        df["Id"] = df["Id"].astype(int)
    if "SalePrice" in df.columns:
        df["SalePrice"] = pd.to_numeric(df["SalePrice"], errors="coerce")

    return df


def get_dataset_metadata() -> Dict[str, Any]:
    """Returns dataset documentation and origin metadata."""
    return DATASET_METADATA.copy()


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive statistical and structural summary of the dataset.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    
    missing_series = df.isnull().sum()
    missing_dict = missing_series[missing_series > 0].sort_values(ascending=False).to_dict()

    target_stats = {}
    if "SalePrice" in df.columns:
        sp = df["SalePrice"].dropna()
        target_stats = {
            "count": int(sp.count()),
            "mean": float(sp.mean()),
            "std": float(sp.std()),
            "min": float(sp.min()),
            "25%": float(sp.quantile(0.25)),
            "50% (median)": float(sp.median()),
            "75%": float(sp.quantile(0.75)),
            "max": float(sp.max()),
            "skewness": float(sp.skew()),
            "kurtosis": float(sp.kurtosis())
        }

    return {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "numerical_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_with_missing_values": len(missing_dict),
        "missing_values_breakdown": missing_dict,
        "target_statistics": target_stats
    }


if __name__ == "__main__":
    df = load_raw_data()
    print("=" * 60)
    print("DATASET SELECTION & UNDERSTANDING")
    print("=" * 60)
    meta = get_dataset_metadata()
    for k, v in meta.items():
        print(f"{k.capitalize()}: {v}")
    
    print("\nDataset Summary:")
    summary = get_data_summary(df)
    print(f"Shape: {summary['total_rows']} rows x {summary['total_columns']} columns")
    print(f"Numerical columns ({summary['numerical_features_count']}): {summary['numerical_features'][:5]}...")
    print(f"Categorical columns ({summary['categorical_features_count']}): {summary['categorical_features'][:5]}...")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print(f"Columns with missing values: {summary['columns_with_missing_values']}")
    print("\nTarget Variable (SalePrice) Statistics:")
    for k, v in summary["target_statistics"].items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
