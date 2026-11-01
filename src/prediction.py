"""
Final Prediction Engine Module
Provides an end-to-end inference pipeline for predicting house prices from raw user input.
Applies feature engineering, preprocessing transformation, model inference,
and dual-currency valuation in both USD ($) and INR (₹).
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from data_loader import load_raw_data
from feature_engineering import engineer_features
from preprocessing import HousingPreprocessor

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

# Standardized exchange rate benchmark
USD_TO_INR_RATE = 85.0


class HousePricePredictor:
    """
    Production-ready inference service for the House Price Prediction system.
    """
    def __init__(self, model_name: str = "Tuned Champion (Gradient Boosting)"):
        self.preprocessor = HousingPreprocessor.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
        self.model_name = model_name
        self.model = self._load_model(model_name)
        
        # Load dataset prototype template for missing defaults
        raw_df = load_raw_data()
        self.numerical_defaults = raw_df.select_dtypes(include=["number"]).median().to_dict()
        self.categorical_defaults = {
            col: raw_df[col].mode()[0] if not raw_df[col].mode().empty else "None"
            for col in raw_df.select_dtypes(exclude=["number"]).columns
        }

    def _load_model(self, model_name: str):
        model_map = {
            "Tuned Champion (Gradient Boosting)": os.path.join(MODEL_DIR, "tuned_best_model.joblib"),
            "XGBoost Regressor": os.path.join(MODEL_DIR, "xgboost_regressor.joblib"),
            "Gradient Boosting Regressor": os.path.join(MODEL_DIR, "gradient_boosting_regressor.joblib"),
            "Random Forest Regressor": os.path.join(MODEL_DIR, "random_forest_regressor.joblib"),
            "Linear Regression": os.path.join(MODEL_DIR, "linear_regression.joblib"),
            "Decision Tree Regressor": os.path.join(MODEL_DIR, "decision_tree_regressor.joblib")
        }
        path = model_map.get(model_name, os.path.join(MODEL_DIR, "tuned_best_model.joblib"))
        if not os.path.exists(path):
            path = os.path.join(MODEL_DIR, "tuned_best_model.joblib")
        return joblib.load(path)

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def predict(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw dictionary of property attributes, aligns with schema,
        runs feature engineering and pipeline preprocessing, and predicts valuation.
        """
        # Build baseline record from typical dataset medians/modes
        row = {}
        for col, default_val in self.numerical_defaults.items():
            if col != "SalePrice":
                row[col] = user_inputs.get(col, default_val)

        for col, default_val in self.categorical_defaults.items():
            row[col] = user_inputs.get(col, default_val)

        # Explicitly overwrite supplied user inputs
        for k, v in user_inputs.items():
            row[k] = v

        input_df = pd.DataFrame([row])

        # Step 1: Feature Engineering
        engineered_df = engineer_features(input_df)

        # Step 2: Preprocessing pipeline transformation
        trans_matrix = self.preprocessor.transform(engineered_df)

        # Step 3: Model Prediction
        pred_usd = float(self.model.predict(trans_matrix)[0])
        pred_usd = max(25000.0, pred_usd)  # Ground truth sanity floor
        pred_inr = pred_usd * USD_TO_INR_RATE

        # Derived metrics for transparency
        total_sf = float(engineered_df["TotalSF"].iloc[0])
        total_bath = float(engineered_df["TotalBath"].iloc[0])
        house_age = int(engineered_df["HouseAge"].iloc[0])
        price_per_sqft_usd = pred_usd / max(total_sf, 100.0)
        price_per_sqft_inr = pred_inr / max(total_sf, 100.0)

        return {
            "predicted_price_usd": round(pred_usd, 2),
            "predicted_price_inr": round(pred_inr, 2),
            "currency_code_primary": "USD ($)",
            "currency_code_converted": "INR (₹)",
            "exchange_rate_applied": USD_TO_INR_RATE,
            "currency_explanation": (
                f"The Ames Housing dataset is natively priced in United States Dollars ($ USD). "
                f"For international interpretation, INR valuations are converted at the current benchmark "
                f"exchange rate of 1 USD = ₹{USD_TO_INR_RATE:.2f}."
            ),
            "model_used": self.model_name,
            "derived_metrics": {
                "TotalSF": total_sf,
                "TotalBath": total_bath,
                "HouseAge": house_age,
                "PricePerSqFt_USD": round(price_per_sqft_usd, 2),
                "PricePerSqFt_INR": round(price_per_sqft_inr, 2)
            }
        }


if __name__ == "__main__":
    predictor = HousePricePredictor()
    sample_house = {
        "OverallQual": 8,
        "GrLivArea": 2100,
        "TotalBsmtSF": 1100,
        "1stFlrSF": 1100,
        "2ndFlrSF": 1000,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "GarageCars": 2,
        "GarageArea": 550,
        "YearBuilt": 2005,
        "YearRemodAdd": 2006,
        "Neighborhood": "CollgCr",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "CentralAir": "Y",
        "Fireplaces": 1
    }

    print("=" * 60)
    print("FINAL PREDICTION ENGINE TEST")
    print("=" * 60)
    result = predictor.predict(sample_house)
    print(f"Model: {result['model_used']}")
    print(f"Estimated Price (USD): ${result['predicted_price_usd']:,.2f}")
    print(f"Estimated Price (INR): ₹{result['predicted_price_inr']:,.2f}")
    print(f"Rate Applied: 1 USD = ₹{result['exchange_rate_applied']:.2f}")
    print(f"Price / Sq Ft: ${result['derived_metrics']['PricePerSqFt_USD']:.2f} (₹{result['derived_metrics']['PricePerSqFt_INR']:.2f})")
    print(f"Explanation: {result['currency_explanation']}")
