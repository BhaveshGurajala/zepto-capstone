"""Reload the saved Titanic pipeline and predict on raw, unprocessed passenger data.

    python predict.py

The saved object is the full Pipeline (imputers + one-hot encoder + scaler +
tuned Random Forest), so it takes raw rows exactly as they appear in titanic.csv.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

HERE = Path(__file__).parent
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

pipeline = joblib.load(HERE / "models" / "titanic_best_pipeline.joblib")
print("Loaded:", type(pipeline).__name__, "->", [name for name, _ in pipeline.steps])

# Raw new passengers: text categories, prices in GBP, one missing age
new = pd.DataFrame([
    {"pclass": 1, "sex": "female", "age": 29,     "sibsp": 0, "parch": 0, "fare": 110.0, "embarked": "C"},
    {"pclass": 3, "sex": "male",   "age": 24,     "sibsp": 0, "parch": 0, "fare": 7.25,  "embarked": "S"},
    {"pclass": 3, "sex": "female", "age": np.nan, "sibsp": 1, "parch": 3, "fare": 25.47, "embarked": "Q"},
])
new["P(survive)"] = pipeline.predict_proba(new[FEATURES])[:, 1].round(3)
new["prediction"] = np.where(pipeline.predict(new[FEATURES]) == 1, "survived", "died")
print(new.to_string(index=False))

# Sanity check on every raw row of the committed CSV
df = pd.read_csv(HERE / "titanic.csv").dropna(subset=["embarked"])
print(f"\nAccuracy on all {len(df)} rows of titanic.csv (train + test): "
      f"{accuracy_score(df['survived'], pipeline.predict(df[FEATURES])):.3f}")
