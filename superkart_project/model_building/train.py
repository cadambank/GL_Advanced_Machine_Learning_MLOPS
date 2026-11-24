"""Step 3 - Train, tune, evaluate and register the best SuperKart sales model."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATASET_REPO = os.getenv("HF_DATASET_REPO", "cadambank/superkart-sales-data")
MODEL_REPO = os.getenv("HF_MODEL_REPO", "cadambank/superkart-sales-model")
TOKEN = os.getenv("HF_TOKEN")
OUT_DIR = "superkart_project/model_building"
MODEL_FILE = "superkart_sales_model_v1.joblib"

NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
CATEGORICAL = ["Product_Sugar_Content", "Product_Type", "Store_Size",
               "Store_Location_City_Type", "Store_Type"]


def load(name):
    """Read one split file from the Hugging Face dataset hub."""
    return pd.read_csv(f"hf://datasets/{DATASET_REPO}/splits/{name}", storage_options={"token": TOKEN})


def evaluate(model, X, y):
    """Regression metrics for one data set."""
    pred = model.predict(X)
    return {"RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
            "MAE": round(float(mean_absolute_error(y, pred)), 2),
            "R2": round(float(r2_score(y, pred)), 4),
            "MAPE": round(float(mean_absolute_percentage_error(y, pred)), 4)}


# 1. Load train / test data from the HF dataset hub
Xtrain, Xtest = load("Xtrain.csv"), load("Xtest.csv")
ytrain, ytest = load("ytrain.csv").squeeze("columns"), load("ytest.csv").squeeze("columns")
print("Train:", Xtrain.shape, "| Test:", Xtest.shape)

# 2. Define the preprocessing, the models and their parameter grids
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", "passthrough", NUMERIC),
])
candidates = {
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=1),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [8, 12, None],
         "model__min_samples_leaf": [1, 5]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=42, n_jobs=1, objective="reg:squarederror"),
        {"model__n_estimators": [100, 200, 300],
         "model__max_depth": [3, 5, 7],
         "model__learning_rate": [0.05, 0.1],
         "model__subsample": [0.8, 1.0]},
    ),
}

# 3. Tune each model with 5-fold cross-validation and evaluate it
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results, best_models = {}, {}
for name, (estimator, grid) in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(Xtrain, ytrain)
    best_models[name] = search.best_estimator_
    results[name] = {
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "cv_rmse": round(float(-search.best_score_), 2),
        "train": evaluate(search.best_estimator_, Xtrain, ytrain),
        "test": evaluate(search.best_estimator_, Xtest, ytest),
    }
    print(f"\n{name}: best params {results[name]['best_params']}")
    print(f"  CV RMSE {results[name]['cv_rmse']} | train {results[name]['train']} | test {results[name]['test']}")

# 4. Select the best model on cross-validated RMSE (the test set is not used for selection)
best_name = min(results, key=lambda n: results[n]["cv_rmse"])
best_model = best_models[best_name]
print(f"\nSelected model: {best_name}")

# 5. Save the model and metrics locally
os.makedirs(OUT_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(OUT_DIR, MODEL_FILE))
metrics = {"selected_model": best_name, "model_file": MODEL_FILE, "results": results}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# 6. Register the best model on the Hugging Face model hub
t = results[best_name]["test"]
card = f"""---
library_name: sklearn
tags: [tabular-regression, sales-forecasting, mlops]
---
# SuperKart Sales Forecast Model

Predicts `Product_Store_Sales_Total` for a product in a store.

* **Algorithm:** {best_name} inside a scikit-learn Pipeline (one-hot encoding + model)
* **Best hyper-parameters:** `{results[best_name]['best_params']}`
* **Test metrics:** RMSE {t['RMSE']} | MAE {t['MAE']} | R2 {t['R2']} | MAPE {t['MAPE']:.2%}
* **Training data:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})

```python
import joblib
from huggingface_hub import hf_hub_download
model = joblib.load(hf_hub_download("{MODEL_REPO}", "{MODEL_FILE}"))
```
"""
api = HfApi(token=TOKEN)
api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)
api.upload_file(path_or_fileobj=os.path.join(OUT_DIR, MODEL_FILE), path_in_repo=MODEL_FILE,
                repo_id=MODEL_REPO, commit_message=f"Register best model ({best_name})")
api.upload_file(path_or_fileobj=os.path.join(OUT_DIR, "metrics.json"), path_in_repo="metrics.json",
                repo_id=MODEL_REPO, commit_message="Upload evaluation metrics")
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                repo_id=MODEL_REPO, commit_message="Update model card")
print(f"Registered model -> https://huggingface.co/{MODEL_REPO}")
