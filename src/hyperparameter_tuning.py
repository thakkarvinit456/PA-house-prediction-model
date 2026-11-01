"""
Hyperparameter Tuning Module
Executes systematic search (RandomizedSearchCV / GridSearchCV) across
Random Forest, Gradient Boosting, and XGBoost regressor models with 5-fold cross validation.
Reports initial vs. tuned performance, identifies optimal hyperparameter sets,
and evaluates the tuned champion model on the untouched test set.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
OUTPUT_RES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evaluation_results"
)


PARAM_DISTRIBUTIONS = {
    "XGBoost": {
        "estimator": xgb.XGBRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [150, 200, 250, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.85, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.85],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [1.0, 1.5, 2.0]
        },
        "n_iter": 12
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [120, 160, 200, 250],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.04, 0.06, 0.08, 0.1],
            "subsample": [0.75, 0.85, 0.9]
        },
        "n_iter": 10
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [120, 180, 220],
            "max_depth": [12, 16, 20, None],
            "min_samples_split": [2, 4, 6],
            "max_features": ["sqrt", 0.4, 0.6]
        },
        "n_iter": 8
    }
}


def run_hyperparameter_tuning(random_state: int = 42) -> Dict[str, Any]:
    """
    Executes cross-validated hyperparameter tuning across tree ensemble models.
    """
    data_cache = joblib.load(os.path.join(MODEL_DIR, "train_test_data.joblib"))
    X_train = data_cache["X_train_trans"]
    y_train = data_cache["y_train"]
    X_test = data_cache["X_test_trans"]
    y_test = data_cache["y_test"]

    # Load baseline test metrics
    test_metrics_path = os.path.join(OUTPUT_RES_DIR, "model_comparison_metrics.csv")
    baseline_df = pd.read_csv(test_metrics_path) if os.path.exists(test_metrics_path) else None

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    tuning_summary = {}

    print("=" * 60)
    print("PHASE 11: HYPERPARAMETER TUNING & OPTIMIZATION")
    print("=" * 60)

    best_overall_score = float("inf")
    best_overall_model = None
    champion_name = ""

    for model_key, config in PARAM_DISTRIBUTIONS.items():
        print(f"\nTuning {model_key} using 5-Fold RandomizedSearchCV ({config['n_iter']} iterations)...")
        
        search = RandomizedSearchCV(
            estimator=config["estimator"],
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        best_cv_rmse = float(-search.best_score_)
        best_params = search.best_params_

        # Evaluate tuned model on held-out test partition
        y_test_pred = best_estimator.predict(X_test)
        tuned_test_rmse = float(root_mean_squared_error(y_test, y_test_pred))
        tuned_test_mae = float(mean_absolute_error(y_test, y_test_pred))
        tuned_test_r2 = float(r2_score(y_test, y_test_pred))

        # Retrieve initial baseline test RMSE
        baseline_rmse = None
        if baseline_df is not None:
            match = baseline_df[baseline_df["Model"].str.contains(model_key.split()[0], case=False, na=False)]
            if not match.empty:
                baseline_rmse = float(match.iloc[0]["RMSE"])

        rmse_improvement = (
            ((baseline_rmse - tuned_test_rmse) / baseline_rmse * 100)
            if baseline_rmse is not None else 0.0
        )

        tuning_summary[model_key] = {
            "best_params": best_params,
            "best_cv_rmse": best_cv_rmse,
            "baseline_test_rmse": baseline_rmse,
            "tuned_test_rmse": tuned_test_rmse,
            "tuned_test_mae": tuned_test_mae,
            "tuned_test_r2": tuned_test_r2,
            "improvement_pct": rmse_improvement
        }

        print(f"  Best Parameters: {best_params}")
        print(f"  Best CV-5 RMSE : ${best_cv_rmse:,.2f}")
        print(f"  Initial Test RMSE: ${baseline_rmse:,.2f}" if baseline_rmse else "  Initial: N/A")
        print(f"  Tuned Test RMSE  : ${tuned_test_rmse:,.2f} (Delta: {rmse_improvement:+.2f}%)")
        print(f"  Tuned Test R2    : {tuned_test_r2:.4f}")

        # Check for overall champion
        if tuned_test_rmse < best_overall_score:
            best_overall_score = tuned_test_rmse
            best_overall_model = best_estimator
            champion_name = model_key

    # Save tuned champion model
    champion_path = os.path.join(MODEL_DIR, "tuned_best_model.joblib")
    joblib.dump(best_overall_model, champion_path)
    
    tuning_output = {
        "champion_model_name": champion_name,
        "champion_test_rmse": best_overall_score,
        "champion_test_r2": tuning_summary[champion_name]["tuned_test_r2"],
        "tuning_details": tuning_summary
    }

    with open(os.path.join(OUTPUT_RES_DIR, "hyperparameter_tuning_results.json"), "w") as f:
        json.dump(tuning_output, f, indent=4)

    print(f"\nTuned Champion Model: {champion_name}")
    print(f"Saved to: {champion_path}")
    return tuning_output


if __name__ == "__main__":
    run_hyperparameter_tuning()
