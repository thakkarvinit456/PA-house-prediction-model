"""
Model Training Module
Implements 80/20 train-test split, 5-fold cross-validation on training partition,
trains 5 industry-standard regression algorithms:
1. Linear Regression (with Ridge regularization for collinear stability)
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor
Saves trained models, fitted preprocessors, and cross-validation logs.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_raw_data
from feature_engineering import engineer_features
from preprocessing import HousingPreprocessor, handle_outliers

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evaluation_results"
)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


MODEL_THEORY = {
    "Linear Regression": {
        "concept": (
            "Ordinary Least Squares (OLS) models the linear relationship between independent predictors and target variable "
            "by minimizing the sum of squared residuals: J(w) = ||y - Xw||^2. L2 regularization (Ridge) is incorporated "
            "to control matrix condition number and stabilize multicollinear property dimensions."
        ),
        "suitability": (
            "Serves as an essential statistical baseline. Features like living area and garage size exhibit direct proportional "
            "pricing increments, which linear coefficients express cleanly."
        ),
        "pros": "Highly interpretable coefficients; computationally rapid; low memory footprint; clear statistical significance tests.",
        "cons": "Fails to capture complex non-linear interactions without manual polynomial expansion; sensitive to extreme outliers and collinearity."
    },
    "Decision Tree Regressor": {
        "concept": (
            "Recursively partitions the multi-dimensional feature space into axis-parallel orthogonal rectangles, "
            "minimizing intra-node variance (Mean Squared Error). At each split point, it selects the feature and threshold "
            "yielding the greatest reduction in variance."
        ),
        "suitability": (
            "Reflects the hierarchical human valuation process (e.g. Neighborhood -> Size -> Quality -> Condition). "
            "Naturally handles categorical encodings and non-linearities without scaling."
        ),
        "pros": "Non-parametric; invariant to monotonic transformations; captures non-linear relationships; zero scale dependence.",
        "cons": "High variance; prone to severe overfitting on training data; unstable splits where small data perturbations alter tree architecture."
    },
    "Random Forest Regressor": {
        "concept": (
            "An ensemble bagging technique that trains an array of unpruned de-correlated decision trees on bootstrap samples "
            "of the training data (bootstrap aggregating) while randomly sampling feature subsets at each split node. "
            "Final prediction is the arithmetic mean of all ensemble trees."
        ),
        "suitability": (
            "Ideal for real estate modeling where individual properties possess diverse idiosyncratic attributes. "
            "Ensemble averaging dampens individual tree variance and prevents overfitting."
        ),
        "pros": "Resilient to noise and outliers; minimal hyperparameter tuning required; built-in out-of-bag validation; delivers feature importances.",
        "cons": "Cannot extrapolate trends beyond the boundary range of training observations; larger memory footprint than single models."
    },
    "Gradient Boosting Regressor": {
        "concept": (
            "A sequential ensemble boosting framework that constructs trees additively in functional space. "
            "Each successive shallow tree is fitted to predict the negative gradient (pseudo-residuals) of the loss function "
            "with respect to previous ensemble predictions: F_m(x) = F_{m-1}(x) + nu * h_m(x)."
        ),
        "suitability": (
            "Recognized across competitive data science benchmarks as the premier paradigm for tabular real estate datasets, "
            "capturing subtle non-linear valuation premiums with minimal residual error."
        ),
        "pros": "Exceptional predictive accuracy; fine-grained optimization via shrinkage (learning rate); handles mixed feature types effectively.",
        "cons": "Sensitive to noisy targets and outliers; higher training runtime; prone to overfitting if tree count and learning rate are uncalibrated."
    },
    "XGBoost Regressor": {
        "concept": (
            "Extreme Gradient Boosting: an advanced, highly optimized scalable implementation of gradient boosted trees. "
            "Implements exact and approximate tree splits using second-order Taylor expansion gradients and Hessians, "
            "augmented by explicit L1 (alpha) and L2 (lambda) tree complexity regularization and cache-aware column subsampling."
        ),
        "suitability": (
            "Industry standard for high-dimensional tabular pricing problems. Effectively tames 300+ one-hot encoded variables "
            "using second-order curvature and shrinkage."
        ),
        "pros": "State-of-the-art accuracy; internal handling of sparsity and missing values; parallel multi-threaded tree construction; built-in regularization.",
        "cons": "Complex hyperparameter space requiring systematic tuning; requires careful feature alignment at inference time."
    }
}


def prepare_train_test_data(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HousingPreprocessor, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw data, performs 80/20 train-test split, treats outliers strictly on train partition,
    applies feature engineering, and fits the preprocessor exclusively on training data.
    """
    raw_df = load_raw_data()

    # Step 1: 80% Train, 20% Test Split before any preprocessing (avoids data leakage)
    train_df, test_df = train_test_split(raw_df, test_size=0.20, random_state=random_state)
    print(f"Initial Split -> Train: {train_df.shape[0]} samples, Test: {test_df.shape[0]} samples")

    # Step 2: Remove extreme documented outliers from training set only
    train_df = handle_outliers(train_df, is_training=True)
    print(f"Cleaned Train Set: {train_df.shape[0]} samples")

    # Step 3: Feature Engineering
    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)

    X_train_raw = train_feat.drop(columns=["SalePrice"])
    y_train = train_feat["SalePrice"].values

    X_test_raw = test_feat.drop(columns=["SalePrice"])
    y_test = test_feat["SalePrice"].values

    # Step 4: Fit preprocessor strictly on training features
    preprocessor = HousingPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    preprocessor.save(os.path.join(MODEL_DIR, "preprocessor.joblib"))

    # Save data cache for evaluation modules
    data_cache = {
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "feature_names": preprocessor.feature_names_out
    }
    joblib.dump(data_cache, os.path.join(MODEL_DIR, "train_test_data.joblib"))
    print(f"Preprocessed features: {X_train_trans.shape[1]} columns")

    return X_train_trans, X_test_trans, y_train, y_test, preprocessor, train_feat, test_feat


def get_model_instances(random_state: int = 42) -> Dict[str, Any]:
    """Returns initialized regression model instances."""
    return {
        "Linear Regression": Ridge(alpha=10.0, random_state=random_state),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=4, random_state=random_state, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=random_state),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.85, colsample_bytree=0.8, random_state=random_state, n_jobs=-1)
    }


def train_and_cross_validate_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Performs 5-Fold Cross Validation on the training set, trains models,
    evaluates train performance and generalizes overfitting/underfitting diagnoses.
    """
    models = get_model_instances(random_state=random_state)
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    training_results = {}

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION & MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTraining & Cross-Validating: {name}...")

        # 5-fold cross-validation on training data
        cv_scores = cross_validate(
            model, X_train, y_train, cv=kfold,
            scoring=["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"],
            n_jobs=-1, return_train_score=True
        )

        cv_rmse = float(-cv_scores["test_neg_root_mean_squared_error"].mean())
        cv_mae = float(-cv_scores["test_neg_mean_absolute_error"].mean())
        cv_r2 = float(cv_scores["test_r2"].mean())

        # Fit model on entire training partition
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)

        train_rmse = float(root_mean_squared_error(y_train, y_train_pred))
        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        train_r2 = float(r2_score(y_train, y_train_pred))

        # Diagnose Overfitting / Underfitting
        r2_gap = train_r2 - cv_r2
        if r2_gap > 0.15:
            fit_diagnosis = "Noticeable Overfitting (Train score substantially exceeds CV score)"
        elif train_r2 < 0.70:
            fit_diagnosis = "Underfitting (Model capacity insufficient for underlying variance)"
        else:
            fit_diagnosis = "Well-Balanced Generalization (Tight correspondence between Train and CV)"

        # Save model artifact
        safe_name = name.lower().replace(" ", "_")
        model_save_path = os.path.join(MODEL_DIR, f"{safe_name}.joblib")
        joblib.dump(model, model_save_path)

        training_results[name] = {
            "model_path": model_save_path,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "cv_rmse": cv_rmse,
            "cv_mae": cv_mae,
            "cv_r2": cv_r2,
            "fit_diagnosis": fit_diagnosis,
            "theory": MODEL_THEORY.get(name, {})
        }

        print(f"  Train RMSE: ${train_rmse:,.2f} | Train R2: {train_r2:.4f}")
        print(f"  CV-5 RMSE : ${cv_rmse:,.2f} | CV-5 R2 : {cv_r2:.4f}")
        print(f"  Diagnosis : {fit_diagnosis}")

    # Cache results
    joblib.dump(training_results, os.path.join(OUTPUT_DIR, "training_cv_results.joblib"))
    return training_results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, train_feat, test_feat = prepare_train_test_data()
    results = train_and_cross_validate_all(X_train, y_train)
    print("\nTraining and Cross-Validation Complete. Artifacts successfully saved.")
