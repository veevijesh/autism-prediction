import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
df = pd.read_csv("autism_data.csv")

# 2. Clean missing indicators
df = df.replace(["?", "null", "None", "nan", "NaN"], np.nan)

# Drop redundant ID / result columns
drop_cols = [c for c in ["ID", "id", "age_desc", "result"] if c in df.columns]
df = df.drop(columns=drop_cols)

# 3. Clean numeric columns safely
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > (len(df) * 0.5):
            df[col] = converted

# 4. Fill missing values SAFELY
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode_val)

# 5. Target Column Handling (Guarantees binary 0 and 1 classes)
target_col = [
    c for c in df.columns if "class" in c.lower() or "asd" in c.lower()
]
target = target_col[0] if target_col else df.columns[-1]

# Convert target to string & standardize
raw_target = df[target].astype(str).str.strip().str.lower()
target_map = {
    "1": 1,
    "0": 0,
    "1.0": 1,
    "0.0": 0,
    "yes": 1,
    "no": 0,
    "y": 1,
    "n": 0,
    "true": 1,
    "false": 0,
}

y = raw_target.map(target_map)

# Fallback if unmapped values exist
if y.isnull().any():
    le_target = LabelEncoder()
    y = pd.Series(
        le_target.fit_transform(raw_target), index=df.index
    ).astype(int)

X = df.drop(columns=[target]).copy()

# 6. Encode Categorical Features
feature_encoders = {}
for col in X.select_dtypes(include=["object", "category", "string"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 10. Evaluate
preds = model.predict(X_test_scaled)
acc = accuracy_score(y_test, preds)

print(f"\nRandom Forest Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, preds))

# 11. Save artifacts
with open("autism_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": list(X.columns),
            "feature_encoders": feature_encoders,
            "classes": list(model.classes_),
        },
        f,
    )

print("Saved model pipeline to 'autism_model.pkl' successfully!")