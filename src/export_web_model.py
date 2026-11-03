import sys
import os
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import HousingPreprocessor
from data_loader import load_raw_data

model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
prep = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
ridge = joblib.load(os.path.join(model_dir, "linear_regression.joblib"))

num_pipe = prep.pipeline.named_transformers_["num"]
scaler = num_pipe.named_steps["scaler"]
num_cols = prep.numerical_cols

cat_pipe = prep.pipeline.named_transformers_["cat"]
ohe = cat_pipe.named_steps["onehot"]
cat_cols = prep.categorical_cols
cat_feature_names = ohe.get_feature_names_out(cat_cols)

coef_num = ridge.coef_[:len(num_cols)]
coef_cat = ridge.coef_[len(num_cols):]

# Compute unstandardized weights
weights_dict = {
    "intercept": float(ridge.intercept_),
    "numerical": {},
    "categorical": {}
}

for col, coef, mean, scale in zip(num_cols, coef_num, scaler.mean_, scaler.scale_):
    weights_dict["numerical"][col] = {
        "unit_weight": float(coef / scale),
        "mean": float(mean),
        "coef": float(coef)
    }

for feat_name, coef in zip(cat_feature_names, coef_cat):
    weights_dict["categorical"][feat_name] = float(coef)

out_path = os.path.join(model_dir, "web_model_weights.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(weights_dict, f, indent=2)

print(f"Web model weights successfully exported to {out_path}")
print(f"Intercept: ${weights_dict['intercept']:,.2f}")
print(f"Numerical features: {len(weights_dict['numerical'])}")
print(f"Categorical features: {len(weights_dict['categorical'])}")
