"""
Model Evaluation Module
Evaluates all trained models on the held-out test partition across standard regression metrics:
1. Mean Absolute Error (MAE)
2. Mean Squared Error (MSE)
3. Root Mean Squared Error (RMSE)
4. R-squared (R2 Score)
Generates diagnostic evaluation plots (Actual vs Predicted, Residual Distribution,
Residual vs Predicted, Model Comparison Bar Chart) and saves figures and metric summaries.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from typing import Dict, Any, Tuple

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
OUTPUT_VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "visualizations"
)
OUTPUT_RES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evaluation_results"
)
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
os.makedirs(OUTPUT_RES_DIR, exist_ok=True)

METRIC_EXPLANATIONS = {
    "MAE": {
        "full_name": "Mean Absolute Error",
        "formula": "MAE = (1/n) * sum(|y_i - y_hat_i|)",
        "explanation": (
            "Measures the average linear magnitude of errors in the same units as the target ($ USD). "
            "It is completely intuitive: an MAE of $15,000 means the model's prediction is, on average, "
            "off by $15,000 from the true transaction price. Unlike RMSE, it treats small and large errors proportionally."
        )
    },
    "MSE": {
        "full_name": "Mean Squared Error",
        "formula": "MSE = (1/n) * sum((y_i - y_hat_i)^2)",
        "explanation": (
            "Measures the average of squared prediction errors ($^2 USD^2). By squaring errors before averaging, "
            "MSE applies severe quadratic penalties to substantial forecasting misses. It serves as the primary "
            "differentiable loss function for gradient-based optimizers."
        )
    },
    "RMSE": {
        "full_name": "Root Mean Squared Error",
        "formula": "RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))",
        "explanation": (
            "The square root of MSE, restoring the error metric back to original dollar units ($ USD). "
            "Because errors are squared prior to square-rooting, RMSE gives higher weight to large outliers than MAE. "
            "A wide divergence between RMSE and MAE signals the presence of rare, high-magnitude prediction blunders."
        )
    },
    "R2": {
        "full_name": "Coefficient of Determination (R-squared)",
        "formula": "R^2 = 1 - (sum((y_i - y_hat_i)^2) / sum((y_i - y_bar)^2))",
        "explanation": (
            "Represents the proportion of variance in house sale price that is explained by the independent features. "
            "An R^2 of 0.90 indicates that 90% of the price variability across Ames properties is captured by the model, "
            "with only 10% attributable to unobserved factors, negotiating noise, or random error."
        )
    }
}


def load_cached_data_and_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads cached preprocessed test data and saved regression models."""
    data_cache = joblib.load(os.path.join(MODEL_DIR, "train_test_data.joblib"))
    
    model_names = [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "XGBoost Regressor"
    ]
    models = {}
    for name in model_names:
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODEL_DIR, f"{safe_name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return data_cache, models


def evaluate_models_on_test() -> pd.DataFrame:
    """
    Evaluates all trained models on the held-out 20% test dataset.
    Returns comparison dataframe with MAE, MSE, RMSE, and R2.
    """
    data_cache, models = load_cached_data_and_models()
    X_test = data_cache["X_test_trans"]
    y_test = data_cache["y_test"]

    records = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        records.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        })

    eval_df = pd.DataFrame(records).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    
    # Save results
    eval_df.to_csv(os.path.join(OUTPUT_RES_DIR, "model_comparison_metrics.csv"), index=False)
    with open(os.path.join(OUTPUT_RES_DIR, "model_comparison_metrics.json"), "w") as f:
        json.dump(records, f, indent=4)
        
    joblib.dump(predictions, os.path.join(OUTPUT_RES_DIR, "test_predictions.joblib"))
    return eval_df


def generate_evaluation_plots(eval_df: pd.DataFrame, best_model_name: str = "Gradient Boosting Regressor"):
    """
    Generates all 4 required diagnostic plots:
    1. Actual vs Predicted Price Plot
    2. Residual Distribution Plot
    3. Residual vs Predicted Plot
    4. Model Performance Comparison Chart
    """
    data_cache, models = load_cached_data_and_models()
    X_test = data_cache["X_test_trans"]
    y_test = data_cache["y_test"]

    best_model = models.get(best_model_name, list(models.values())[0])
    y_pred = best_model.predict(X_test)
    residuals = y_test - y_pred

    # 1. Actual vs Predicted Plot
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    ax1.scatter(y_test, y_pred, alpha=0.6, color="#2563EB", edgecolors="none", s=40, label="Test Observations")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linestyle="--", linewidth=2, label="Ideal Identity (y = x)")
    ax1.set_title(f"Actual vs. Predicted Sale Price ({best_model_name})", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Actual Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Predicted Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax1.xaxis.set_major_formatter("${x:,.0f}")
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    ax1.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_VIS_DIR, "eval_actual_vs_predicted.png"), dpi=200)
    plt.close(fig1)

    # 2. Residual Distribution Plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#8B5CF6", bins=35, ax=ax2)
    ax2.axvline(0, color="#EF4444", linestyle="--", linewidth=2, label="Zero Error Baseline")
    ax2.set_title(f"Residual Error Distribution ({best_model_name})", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Residual Error (Actual - Predicted) ($ USD)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="semibold")
    ax2.xaxis.set_major_formatter("${x:,.0f}")
    ax2.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_VIS_DIR, "eval_residual_distribution.png"), dpi=200)
    plt.close(fig2)

    # 3. Residual vs Predicted Plot
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(y_pred, residuals, alpha=0.6, color="#10B981", edgecolors="none", s=40)
    ax3.axhline(0, color="#EF4444", linestyle="--", linewidth=2, label="Zero Error Line")
    ax3.set_title(f"Residual vs. Predicted Price (Homoscedasticity Analysis)", fontsize=14, fontweight="bold", pad=12)
    ax3.set_xlabel("Predicted Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax3.set_ylabel("Residual Error ($ USD)", fontsize=11, fontweight="semibold")
    ax3.xaxis.set_major_formatter("${x:,.0f}")
    ax3.yaxis.set_major_formatter("${x:,.0f}")
    ax3.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_VIS_DIR, "eval_residual_vs_predicted.png"), dpi=200)
    plt.close(fig3)

    # 4. Model Comparison Bar Chart
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE Comparison
    sns.barplot(data=eval_df, x="Model", y="RMSE", color="#2563EB", ax=ax4a)
    ax4a.set_title("Test RMSE Comparison (Lower is Better)", fontsize=12, fontweight="bold")
    ax4a.set_ylabel("Root Mean Squared Error ($ USD)", fontsize=10)
    ax4a.yaxis.set_major_formatter("${x:,.0f}")
    ax4a.set_xticks(range(len(eval_df)))
    ax4a.set_xticklabels(eval_df["Model"], rotation=30, ha="right")
    
    # R2 Comparison
    sns.barplot(data=eval_df, x="Model", y="R²", color="#10B981", ax=ax4b)
    ax4b.set_title("Test R² Score Comparison (Higher is Better)", fontsize=12, fontweight="bold")
    ax4b.set_ylabel("Coefficient of Determination (R²)", fontsize=10)
    ax4b.set_ylim([0.7, 1.0])
    ax4b.set_xticks(range(len(eval_df)))
    ax4b.set_xticklabels(eval_df["Model"], rotation=30, ha="right")

    fig4.suptitle("Model Performance Comparison on 20% Held-Out Test Set", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_VIS_DIR, "eval_model_comparison_bars.png"), dpi=200)
    plt.close(fig4)
    print("All evaluation plots generated and saved successfully.")


# Plotly Interactive builders for Streamlit
def build_plotly_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={"x": "Actual Sale Price ($)", "y": "Predicted Sale Price ($)"},
        title=f"Actual vs. Predicted Price ({model_name})",
        opacity=0.7, color_discrete_sequence=["#2563EB"]
    )
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode="lines", name="Perfect Fit (y=x)",
        line=dict(color="#EF4444", dash="dash", width=2)
    ))
    fig.update_layout(template="plotly_white")
    return fig


def build_plotly_residuals(y_pred: np.ndarray, residuals: np.ndarray, model_name: str) -> go.Figure:
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted Sale Price ($)", "y": "Residual Error ($)"},
        title=f"Residual vs. Predicted Price ({model_name})",
        opacity=0.7, color_discrete_sequence=["#10B981"]
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444", annotation_text="Zero Residual")
    fig.update_layout(template="plotly_white")
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION ON TEST PARTITION")
    print("=" * 60)
    eval_df = evaluate_models_on_test()
    print("\nModel Evaluation Comparison Table (Untouched 20% Test Set):")
    print(eval_df.to_string(index=False))
    
    best_model = eval_df.iloc[0]["Model"]
    print(f"\nTop Performing Algorithm on Test Set: {best_model} (RMSE: ${eval_df.iloc[0]['RMSE']:,.2f}, R2: {eval_df.iloc[0]['R²']:.4f})")
    generate_evaluation_plots(eval_df, best_model_name=best_model)
