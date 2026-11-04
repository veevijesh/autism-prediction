import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv("autism_data.csv")
df = df.replace(["?", "null", "None", "nan", "NaN"], np.nan)

# Target Column Identification
target_col = None
for col in ["Class/ASD", "Class/ASD Traits", "austim", "class"]:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("Target column not found in dataset.")

# Drop unneeded ID columns
drop_cols = [c for c in ["ID", "id", "age_desc", "result"] if c in df.columns]
df = df.drop(columns=drop_cols)

# Separate Target y (1 = positive ASD, 0 = negative)
y_raw = df[target_col].astype(str).str.strip().str.lower()
y = y_raw.isin(["1", "yes", "true", "y"]).astype(int)
X = df.drop(columns=[target_col]).copy()

# Binary Mapping Helper
binary_map = {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0, "f": 0, "m": 1}

# Clean Numeric Features First
for col in X.columns:
    if "A" in col and "_Score" in col:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)
    elif col == "age":
        X[col] = pd.to_numeric(X[col], errors="coerce")

# Train/Test Split BEFORE target-encoding categorical variables to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute median age calculated strictly from X_train
age_median = X_train["age"].median()
X_train["age"] = X_train["age"].fillna(age_median)
X_test["age"] = X_test["age"].fillna(age_median)

# Explicit Categorical Encoding (learned on Train, applied to Test)
category_maps = {}
for col in X_train.columns:
    if col in ["age"] or ("A" in col and "_Score" in col):
        continue

    # Fill NaN with 'unknown' and convert to string to prevent float vs str comparison errors
    cleaned_train = X_train[col].fillna("unknown").astype(str).str.strip().str.lower()
    cleaned_test = X_test[col].fillna("unknown").astype(str).str.strip().str.lower()

    if set(cleaned_train.unique()).issubset(set(binary_map.keys()).union({"nan", "unknown"})):
        X_train[col] = cleaned_train.map(binary_map).fillna(0).astype(int)
        X_test[col] = cleaned_test.map(binary_map).fillna(0).astype(int)
        category_maps[col] = binary_map
    else:
        # Categorical string columns (ethnicity, contry_of_res, relation)
        uniques = sorted(list(cleaned_train.unique()))
        mapping = {val: idx for idx, val in enumerate(uniques)}
        X_train[col] = cleaned_train.map(mapping).fillna(0).astype(int)
        X_test[col] = cleaned_test.map(mapping).fillna(0).astype(int)
        category_maps[col] = mapping

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    class_weight="balanced", 
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("--- EVALUATION REPORT ---")
preds = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

# Save Artifacts
with open("autism_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": list(X_train.columns),
            "category_maps": category_maps,
            "age_median": age_median,
            "classes": list(model.classes_),
        },
        f,
    )

print("✅ Saved robust model pipeline to 'autism_model.pkl' successfully!")