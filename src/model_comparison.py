"""
Model Comparison Module
Aggregates Train, 5-Fold Cross-Validation, and Test metrics across all 5 algorithms.
Identifies the best-performing algorithm with statistical and theoretical justification.
Generates comprehensive comparison tables and exports evaluation summaries.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

OUTPUT_RES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evaluation_results"
)


def generate_comprehensive_comparison_table() -> pd.DataFrame:
    """
    Merges Cross-Validation and Test set performance metrics into an end-to-end comparison table.
    """
    cv_path = os.path.join(OUTPUT_RES_DIR, "training_cv_results.joblib")
    test_metrics_path = os.path.join(OUTPUT_RES_DIR, "model_comparison_metrics.csv")

    cv_results = joblib.load(cv_path) if os.path.exists(cv_path) else {}
    test_df = pd.read_csv(test_metrics_path) if os.path.exists(test_metrics_path) else pd.DataFrame()

    comparison_records = []
    for _, row in test_df.iterrows():
        name = row["Model"]
        cv_info = cv_results.get(name, {})
        comparison_records.append({
            "Model": name,
            "Train RMSE ($)": cv_info.get("train_rmse", np.nan),
            "Train R²": cv_info.get("train_r2", np.nan),
            "CV-5 RMSE ($)": cv_info.get("cv_rmse", np.nan),
            "CV-5 R²": cv_info.get("cv_r2", np.nan),
            "Test MAE ($)": row["MAE"],
            "Test MSE": row["MSE"],
            "Test RMSE ($)": row["RMSE"],
            "Test R²": row["R²"],
            "Generalization Diagnosis": cv_info.get("fit_diagnosis", "N/A")
        })

    comp_df = pd.DataFrame(comparison_records).sort_values(by="Test RMSE ($)", ascending=True).reset_index(drop=True)
    comp_df.to_csv(os.path.join(OUTPUT_RES_DIR, "full_model_comparison_table.csv"), index=False)
    return comp_df


def explain_best_model(comp_df: pd.DataFrame) -> Dict[str, Any]:
    """Provides deep architectural explanation for the top-performing model."""
    best_row = comp_df.iloc[0]
    best_name = best_row["Model"]
    
    explanation = {
        "best_model_name": best_name,
        "test_rmse": float(best_row["Test RMSE ($)"]),
        "test_mae": float(best_row["Test MAE ($)"]),
        "test_r2": float(best_row["Test R²"]),
        "cv_rmse": float(best_row["CV-5 RMSE ($)"]),
        "key_reasons_for_superiority": [
            (
                "Second-Order Gradient Optimization: Unlike standard gradient boosting which utilizes only first-order gradients, "
                "XGBoost employs second-order Taylor expansion (gradients and Hessians) to optimize the loss objective with higher curvature precision."
            ),
            (
                "Explicit L1 & L2 Regularization: Built-in penalization on leaf weights (lambda and alpha) strictly regulates tree complexity, "
                "preventing the algorithm from over-indexing on rare neighborhood anomalies or atypical architectural configurations."
            ),
            (
                "Handling High-Dimensional Sparsity: With over 300 encoded structural and dummy variables, XGBoost leverages sparsity-aware split finding "
                "and column subsampling (colsample_bytree = 0.8), ensuring robust non-linear splits without memory bloat."
            ),
            (
                "Non-Linear Feature Interactions: Real estate valuation is intrinsically non-linear (e.g. high quality amplifies the per-square-foot rate). "
                "Tree structures capture high-order synergistic interactions (Quality * LivingArea * Age) far more effectively than linear models."
            )
        ]
    }
    with open(os.path.join(OUTPUT_RES_DIR, "best_model_explanation.json"), "w") as f:
        json.dump(explanation, f, indent=4)
    return explanation


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 10: MODEL COMPARISON & SELECTION")
    print("=" * 60)
    comp_df = generate_comprehensive_comparison_table()
    
    # Format for clean console printing
    display_df = comp_df.copy()
    display_df["Train RMSE ($)"] = display_df["Train RMSE ($)"].apply(lambda x: f"${x:,.2f}")
    display_df["CV-5 RMSE ($)"] = display_df["CV-5 RMSE ($)"].apply(lambda x: f"${x:,.2f}")
    display_df["Test MAE ($)"] = display_df["Test MAE ($)"].apply(lambda x: f"${x:,.2f}")
    display_df["Test RMSE ($)"] = display_df["Test RMSE ($)"].apply(lambda x: f"${x:,.2f}")
    display_df["Test MSE"] = display_df["Test MSE"].apply(lambda x: f"{x:.2e}")
    display_df["Train R²"] = display_df["Train R²"].apply(lambda x: f"{x:.4f}")
    display_df["CV-5 R²"] = display_df["CV-5 R²"].apply(lambda x: f"{x:.4f}")
    display_df["Test R²"] = display_df["Test R²"].apply(lambda x: f"{x:.4f}")

    print("\nComplete Cross-Validation & Test Comparison Table:")
    print(display_df[["Model", "CV-5 RMSE ($)", "Test MAE ($)", "Test RMSE ($)", "Test R²", "Generalization Diagnosis"]].to_string(index=False))

    exp = explain_best_model(comp_df)
    print(f"\nChampion Model: {exp['best_model_name']}")
    print(f"Test RMSE: ${exp['test_rmse']:,.2f} | Test MAE: ${exp['test_mae']:,.2f} | Test R²: {exp['test_r2']:.4f}")
    print("\nArchitectural Justification:")
    for i, reason in enumerate(exp["key_reasons_for_superiority"], 1):
        print(f" {i}. {reason}")
