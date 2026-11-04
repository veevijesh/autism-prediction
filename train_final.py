import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("train.csv").replace("?", np.nan)
y = df["Class/ASD"]
features = [f"A{i}_Score" for i in range(1, 11)]
X = df[features].astype(int)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

# Honest estimate: 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
s = cross_validate(pipeline, X, y, cv=cv, scoring=["recall", "precision", "roc_auc"])
print("Cross-validated (5-fold) results for the deployed model:")
print(f"  recall    {s['test_recall'].mean():.3f}")
print(f"  precision {s['test_precision'].mean():.3f}")
print(f"  ROC-AUC   {s['test_roc_auc'].mean():.3f}")

# Final model: fit on all 800 labeled respondents and save
pipeline.fit(X, y)
with open("autism_model_v2.pkl", "wb") as f:
    pickle.dump({"pipeline": pipeline, "features": features}, f)
print("Saved model to 'autism_model_v2.pkl'")