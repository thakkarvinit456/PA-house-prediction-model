"""
Data Preprocessing Pipeline Module
Builds Scikit-learn Pipeline and ColumnTransformer for numerical imputation,
scaling, categorical imputation, and one-hot encoding.
Ensures zero data leakage: transformations are fitted exclusively on training data.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(MODEL_DIR, exist_ok=True)


def handle_outliers(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Treats extreme outliers as recommended by Dean De Cock (2011).
    Houses with GrLivArea > 4000 sq ft and SalePrice < $300,000 are anomalous partial sales.
    Applied strictly to training partition to prevent contamination of test set.
    """
    if is_training and "GrLivArea" in df.columns and "SalePrice" in df.columns:
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            print(f"Removed {outlier_count} documented Dean De Cock extreme outliers from training set.")
            return df[~outlier_mask].copy()
    return df.copy()


def build_preprocessor_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Constructs an industrial Scikit-Learn ColumnTransformer pipeline.
    Numerical: Median Imputation -> StandardScaler
    Categorical: 'Missing' Imputation -> OneHotEncoder(handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor


class HousingPreprocessor:
    """
    Wrapper class managing feature segregation, pipeline fitting, and transformation.
    """
    def __init__(self):
        self.pipeline: Optional[ColumnTransformer] = None
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names_out: List[str] = []

    def fit(self, X: pd.DataFrame):
        # Identify feature types
        self.numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        self.pipeline = build_preprocessor_pipeline(self.numerical_cols, self.categorical_cols)
        self.pipeline.fit(X)

        # Extract output feature names
        try:
            self.feature_names_out = self.pipeline.get_feature_names_out().tolist()
        except Exception:
            self.feature_names_out = [f"feat_{i}" for i in range(len(self.numerical_cols))]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: str = os.path.join(MODEL_DIR, "preprocessor.joblib")):
        joblib.dump(self, filepath)
        print(f"Preprocessor successfully saved to {filepath}")

    @staticmethod
    def load(filepath: str = os.path.join(MODEL_DIR, "preprocessor.joblib")) -> "HousingPreprocessor":
        return joblib.load(filepath)


if __name__ == "__main__":
    from data_loader import load_raw_data
    from feature_engineering import engineer_features

    print("=" * 60)
    print("PREPROCESSING PIPELINE INITIALIZATION")
    print("=" * 60)
    raw_df = load_raw_data()
    clean_df = handle_outliers(raw_df, is_training=True)
    feat_df = engineer_features(clean_df)

    X = feat_df.drop(columns=["SalePrice"])
    y = feat_df["SalePrice"]

    prep = HousingPreprocessor()
    X_trans = prep.fit_transform(X)

    print(f"Input shape: {X.shape}")
    print(f"Numerical features: {len(prep.numerical_cols)}")
    print(f"Categorical features: {len(prep.categorical_cols)}")
    print(f"Transformed output matrix shape: {X_trans.shape}")
    print(f"Number of encoded feature names: {len(prep.feature_names_out)}")
    prep.save()
