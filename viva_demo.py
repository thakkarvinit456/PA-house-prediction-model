"""
===============================================================================
ESTATEVALUE AI — VIVA EXAMINATION LIVE DEMONSTRATION SCRIPT
Author: Vinit Thakkar
Repository: thakkarvinit456/PA-house-prediction-model
Topic: End-to-End Ames House Price Prediction System
===============================================================================
Run with:
    python viva_demo.py
===============================================================================
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src/ is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from prediction import HousePricePredictor
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def print_divider(char="=", length=75):
    print(char * length)


def print_header():
    print_divider("=")
    print("   ESTATEVALUE AI: MACHINE LEARNING HOUSE PRICE PREDICTION SYSTEM")
    print("   Academic Viva Voce & Practical Examination Live Demonstration")
    print("   Candidate: Vinit Thakkar  |  Dataset: Ames Housing (Dean De Cock)")
    print_divider("=")
    print()


def show_dataset_summary():
    print("[1] DATASET UNDERSTANDING & CHARACTERISTICS")
    print_divider("-")
    print("  • Source Dataset       : Dean De Cock (Ames, Iowa - OpenML ID 543)")
    print("  • Total Records        : 1,460 Residential Property Transactions")
    print("  • Raw Feature Space    : 81 Columns (37 Numerical, 43 Categorical, 1 Target)")
    print("  • Target Variable      : SalePrice (Positively Skewed: skewness = +1.88)")
    print("  • Train/Test Split     : 80% Train (1,168 rows) / 20% Test (292 rows)")
    print("  • Transformed Features : 310 Columns (One-Hot Encoded with Zero Leakage)")
    print("  • Engineered Features  : TotalSF (r = +0.782), TotalBath (r = +0.632),")
    print("                           HouseAge (r = -0.523), RemodAge (r = -0.509)")
    print()


def show_benchmark_results():
    print("[2] EMPIRICAL BENCHMARK COMPARISON (Evaluated on Held-Out Test Set N=292)")
    print_divider("-")
    print(f"  {'Rank':<5} {'Model':<28} {'Test MAE':<14} {'Test RMSE':<14} {'Test R²':<10} {'Status'}")
    print_divider("-")
    print(f"  {'1':<5} {'Tuned Gradient Boosting':<28} {'$14,482.10':<14} {'$23,513.00':<14} {'0.9279':<10} {'CHAMPION (+17.3%)'}")
    print(f"  {'2':<5} {'XGBoost Regressor':<28} {'$14,885.54':<14} {'$24,202.35':<14} {'0.9236':<10} {'High Regularization'}")
    print(f"  {'3':<5} {'Linear Regression (Ridge)':<28} {'$17,504.65':<14} {'$27,551.72':<14} {'0.9010':<10} {'Linear Baseline'}")
    print(f"  {'4':<5} {'Random Forest Regressor':<28} {'$16,104.22':<14} {'$27,821.30':<14} {'0.8991':<10} {'Ensemble Bagging'}")
    print(f"  {'5':<5} {'Decision Tree Regressor':<28} {'$23,391.33':<14} {'$35,042.27':<14} {'0.8399':<10} {'High Variance'}")
    print_divider("-")
    print("  Key Finding: Tuned GBDT achieved highest variance explanation (R² = 92.79%)")
    print("               reducing prediction error by 17.27% over baseline.")
    print()


def run_live_predictions(predictor):
    print("[3] LIVE MODEL INFERENCE DEMONSTRATION")
    print_divider("-")

    test_cases = [
        {
            "name": "Case A: Standard Suburban Family Home (College Creek)",
            "inputs": {
                "GrLivArea": 1800,
                "TotalBsmtSF": 950,
                "1stFlrSF": 1100,
                "2ndFlrSF": 700,
                "LotArea": 10500,
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "YearRemodAdd": 2005,
                "FullBath": 2,
                "HalfBath": 1,
                "BedroomAbvGr": 3,
                "GarageCars": 2,
                "Neighborhood": "CollgCr",
                "KitchenQual": "Gd",
                "Fireplaces": 1,
                "CentralAir": "Y"
            }
        },
        {
            "name": "Case B: Luxury Executive Estate (Northridge)",
            "inputs": {
                "GrLivArea": 3200,
                "TotalBsmtSF": 1600,
                "1stFlrSF": 1700,
                "2ndFlrSF": 1500,
                "LotArea": 18000,
                "OverallQual": 9,
                "OverallCond": 6,
                "YearBuilt": 2008,
                "YearRemodAdd": 2012,
                "FullBath": 3,
                "HalfBath": 1,
                "BedroomAbvGr": 4,
                "GarageCars": 3,
                "Neighborhood": "NoRidge",
                "KitchenQual": "Ex",
                "Fireplaces": 2,
                "CentralAir": "Y"
            }
        },
        {
            "name": "Case C: Budget Starter Home (North Ames)",
            "inputs": {
                "GrLivArea": 1150,
                "TotalBsmtSF": 600,
                "1stFlrSF": 1150,
                "2ndFlrSF": 0,
                "LotArea": 7500,
                "OverallQual": 5,
                "OverallCond": 5,
                "YearBuilt": 1975,
                "YearRemodAdd": 1985,
                "FullBath": 1,
                "HalfBath": 0,
                "BedroomAbvGr": 3,
                "GarageCars": 1,
                "Neighborhood": "NAmes",
                "KitchenQual": "TA",
                "Fireplaces": 0,
                "CentralAir": "Y"
            }
        }
    ]

    for tc in test_cases:
        t0 = time.perf_counter()
        res = predictor.predict(tc["inputs"])
        latency_ms = (time.perf_counter() - t0) * 1000

        metrics = res.get("derived_metrics", {})
        price_sqft_usd = metrics.get("PricePerSqFt_USD", 0.0)
        price_sqft_inr = metrics.get("PricePerSqFt_INR", 0.0)
        total_sf = metrics.get("TotalSF", 0.0)
        house_age = metrics.get("HouseAge", 0)

        print(f"  ▶ {tc['name']}")
        print(f"    Inputs        : {tc['inputs']['GrLivArea']} sq ft living, Qual {tc['inputs']['OverallQual']}/10, Built {tc['inputs']['YearBuilt']}, Neighborhood: {tc['inputs']['Neighborhood']}")
        print(f"    Valuation USD : ${res['predicted_price_usd']:,.2f}")
        print(f"    Valuation INR : ₹{res['predicted_price_inr']:,.2f} (at ₹85/$)")
        print(f"    Rate / sq ft  : ${price_sqft_usd:,.2f}/sq ft  (₹{price_sqft_inr:,.2f}/sq ft)")
        print(f"    Total SF      : {total_sf:,.0f} sq ft  |  House Age: {house_age} yrs  |  Inference Latency: {latency_ms:.2f} ms")
        print()


def show_viva_defense_points():
    print("[4] TOP 3 CORE VIVA DEFENSE ARGUMENTS (For Examiner Q&A)")
    print_divider("-")
    print("  Q1: How did you ensure ZERO DATA LEAKAGE in your pipeline?")
    print("  Ans: The 80/20 train/test split was executed strictly FIRST.")
    print("       Imputers, Scalers, and One-Hot encoders were fitted EXCLUSIVELY")
    print("       on the training set (N=1,168) and only transformed on the test set (N=292).")
    print()
    print("  Q2: What was your most impactful Feature Engineering decision?")
    print("  Ans: Synthesizing 'TotalSF' = TotalBsmtSF + 1stFlrSF + 2ndFlrSF.")
    print("       Total square footage directly drives consumer utility and exhibited")
    print("       the highest Pearson correlation with SalePrice (r = +0.7823),")
    print("       surpassing individual floor areas.")
    print()
    print("  Q3: Why did Gradient Boosting outperform Random Forest and Decision Tree?")
    print("  Ans: Decision Trees overfit (R² = 0.8399). Random Forest bags independent")
    print("       trees, which reduces variance but cannot actively correct residual bias.")
    print("       Gradient Boosting sequentially minimizes the pseudo-residuals of")
    print("       previous estimators via gradient descent in function space, achieving")
    print("       superior predictive fidelity (R² = 0.9279).")
    print_divider("=")
    print("   VIVA DEMO READY! Website is accessible at: http://localhost:8000")
    print("   GitHub Repository: https://github.com/thakkarvinit456/PA-house-prediction-model")
    print_divider("=")


def main():
    print_header()
    show_dataset_summary()
    show_benchmark_results()

    print("Loading fitted preprocessor and Champion Tuned GBDT model...")
    t0 = time.time()
    predictor = HousePricePredictor(model_name="Tuned Champion (Gradient Boosting)")
    print(f"Model loaded successfully in {time.time() - t0:.2f}s.\n")

    run_live_predictions(predictor)
    show_viva_defense_points()


if __name__ == "__main__":
    main()
