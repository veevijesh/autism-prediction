import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("train.csv").replace("?", np.nan)
y = df["Class/ASD"]
print("People:", len(df), "| positives:", int(y.sum()), f"({y.mean():.1%})")
print("Accuracy if we always guess 'no ASD':", round(1 - y.mean(), 3))
print()

A = [f"A{i}_Score" for i in range(1, 11)]
background_num = ["age"]
background_cat = ["gender", "ethnicity", "jaundice", "austim",
                  "contry_of_res", "used_app_before", "relation"]

feature_sets = {
    "Questionnaire only (10)": (A, []),
    "Background only (8)": (background_num, background_cat),
    "All features (18)": (A + background_num, background_cat),
}
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for fs_name, (num_cols, cat_cols) in feature_sets.items():
    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), num_cols)]
    if cat_cols:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore"))]),
                      cat_cols))
    for m_name, m in models.items():
        pipe = Pipeline([("prep", ColumnTransformer(parts)), ("model", m)])
        s = cross_validate(pipe, df[num_cols + cat_cols], y, cv=skf,
                           scoring=["recall", "precision", "accuracy", "roc_auc"])
        rows.append({
            "Features": fs_name, "Model": m_name,
            "Recall": round(s["test_recall"].mean(), 3),
            "Precision": round(s["test_precision"].mean(), 3),
            "Accuracy": round(s["test_accuracy"].mean(), 3),
            "ROC-AUC": round(s["test_roc_auc"].mean(), 3),
        })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv("results.csv", index=False)