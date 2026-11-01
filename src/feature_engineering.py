"""
Feature Engineering and Selection Module
Creates domain-specific derived features from Ames Housing structural attributes,
computes feature importance and correlation rankings, and selects the optimal feature set.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates high-value derived features based on real estate domain knowledge.
    This transformation operates deterministically on row-level attributes.
    """
    df_feat = df.copy()

    # 1. Total Square Footage (TotalSF) = Basement + 1st Floor + 2nd Floor
    bsmt = df_feat["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in df_feat.columns else 0
    flr1 = df_feat["1stFlrSF"].fillna(0) if "1stFlrSF" in df_feat.columns else 0
    flr2 = df_feat["2ndFlrSF"].fillna(0) if "2ndFlrSF" in df_feat.columns else 0
    df_feat["TotalSF"] = bsmt + flr1 + flr2

    # 2. Total Bathrooms (TotalBath) = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
    full_bath = df_feat["FullBath"].fillna(0) if "FullBath" in df_feat.columns else 0
    half_bath = df_feat["HalfBath"].fillna(0) if "HalfBath" in df_feat.columns else 0
    bsmt_full = df_feat["BsmtFullBath"].fillna(0) if "BsmtFullBath" in df_feat.columns else 0
    bsmt_half = df_feat["BsmtHalfBath"].fillna(0) if "BsmtHalfBath" in df_feat.columns else 0
    df_feat["TotalBath"] = full_bath + (0.5 * half_bath) + bsmt_full + (0.5 * bsmt_half)

    # 3. Total Porch & Deck Square Footage
    wood_deck = df_feat["WoodDeckSF"].fillna(0) if "WoodDeckSF" in df_feat.columns else 0
    open_porch = df_feat["OpenPorchSF"].fillna(0) if "OpenPorchSF" in df_feat.columns else 0
    enc_porch = df_feat["EnclosedPorch"].fillna(0) if "EnclosedPorch" in df_feat.columns else 0
    ssn_porch = df_feat["3SsnPorch"].fillna(0) if "3SsnPorch" in df_feat.columns else 0
    scr_porch = df_feat["ScreenPorch"].fillna(0) if "ScreenPorch" in df_feat.columns else 0
    df_feat["TotalPorchSF"] = wood_deck + open_porch + enc_porch + ssn_porch + scr_porch

    # 4. Temporal Features: House Age and Years Since Remodeling at time of sale
    if "YrSold" in df_feat.columns and "YearBuilt" in df_feat.columns:
        df_feat["HouseAge"] = (df_feat["YrSold"] - df_feat["YearBuilt"]).clip(lower=0)
    elif "YearBuilt" in df_feat.columns:
        df_feat["HouseAge"] = (2010 - df_feat["YearBuilt"]).clip(lower=0)

    if "YrSold" in df_feat.columns and "YearRemodAdd" in df_feat.columns:
        df_feat["RemodAge"] = (df_feat["YrSold"] - df_feat["YearRemodAdd"]).clip(lower=0)
    elif "YearRemodAdd" in df_feat.columns:
        df_feat["RemodAge"] = (2010 - df_feat["YearRemodAdd"]).clip(lower=0)

    # 5. Composite Quality & Condition Score
    if "OverallQual" in df_feat.columns and "OverallCond" in df_feat.columns:
        df_feat["OverallGrade"] = df_feat["OverallQual"] * df_feat["OverallCond"]

    # 6. Binary Amenity Indicators
    if "GarageArea" in df_feat.columns:
        df_feat["HasGarage"] = (df_feat["GarageArea"].fillna(0) > 0).astype(int)
    if "TotalBsmtSF" in df_feat.columns:
        df_feat["HasBsmt"] = (df_feat["TotalBsmtSF"].fillna(0) > 0).astype(int)
    if "Fireplaces" in df_feat.columns:
        df_feat["HasFireplace"] = (df_feat["Fireplaces"].fillna(0) > 0).astype(int)
    if "PoolArea" in df_feat.columns:
        df_feat["HasPool"] = (df_feat["PoolArea"].fillna(0) > 0).astype(int)

    # Drop non-predictive identifiers if present
    if "Id" in df_feat.columns:
        df_feat = df_feat.drop(columns=["Id"])

    return df_feat


def get_feature_correlations(df: pd.DataFrame, target_col: str = "SalePrice", top_n: int = 15) -> pd.Series:
    """Computes Pearson correlation coefficients of numerical features with the target."""
    numeric_df = df.select_dtypes(include=["number"])
    if target_col in numeric_df.columns:
        corr = numeric_df.corr()[target_col].drop(target_col, errors="ignore")
        return corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)
    return pd.Series(dtype=float)


if __name__ == "__main__":
    from data_loader import load_raw_data
    df = load_raw_data()
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Original feature count: {df.shape[1]}")
    df_feat = engineer_features(df)
    print(f"Post-engineering feature count: {df_feat.shape[1]}")
    print("\nTop 15 Features by Correlation with SalePrice:")
    corrs = get_feature_correlations(df_feat)
    for feat, val in corrs.items():
        print(f"  {feat:<20}: {val:+.4f}")
