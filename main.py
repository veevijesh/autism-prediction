"""Train with: python main.py (adult questionnaire CSV, pre-scored by default)."""
import argparse
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "autism_model.pkl"
VERSION = "adult-items-only-v2"

# Adult AQ-10 key. Yes represents agreement, No represents disagreement.
# A True entry awards one point for Yes, a False entry awards one for No.
ITEMS = (
    ("Do small sounds often catch your attention when others miss them?", True),
    ("Do you usually concentrate more on the whole picture than on small details?", False),
    ("Do you find it easy to do more than one task at the same time?", False),
    ("After an interruption, can you quickly switch back to what you were doing?", False),
    ("Do you find it easy to read between the lines when someone is talking?", False),
    ("Can you tell when someone listening to you is getting bored?", False),
    ("When reading a story, do you find it difficult to work out characters' intentions?", True),
    ("Do you like collecting information about categories of things?", True),
    ("Can you easily tell what someone is thinking or feeling by looking at their face?", False),
    ("Do you find it difficult to work out what other people intend?", True),
)
SCORES = [f"A{i}_Score" for i in range(1, 11)]
FIELDS = {
    "gender": "Gender", "ethnicity": "Ethnicity", "jaundice": "Born with jaundice?",
    "austim": "Family history of autism?", "contry_of_res": "Country of residence",
    "used_app_before": "Previously used a screening app?", "relation": "Completed by",
}
ALIASES = {"jundice": "jaundice", "family_pdd": "austim", "country_of_res": "contry_of_res"}
BINARY = {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}


def point(item_index, yes):
    return int(yes == ITEMS[item_index][1])


def binary(series, label, allow_words):
    cleaned = series.astype("string").str.strip().str.lower()
    mapped = pd.to_numeric(cleaned, errors="coerce")
    if allow_words:
        mapped = mapped.fillna(cleaned.map(BINARY))
    if not mapped.isin([0, 1]).all():
        raise ValueError(f"{label} contains missing or unrecognised values. Check the CSV encoding.")
    return mapped.astype(int)


def train(csv_path, raw_answers=False, target=None):
    df = pd.read_csv(csv_path, encoding="utf-8-sig", na_values=["?", "null", "None"])
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns=ALIASES)
    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns after normalisation. Check the CSV headers.")
    targets = [k for k in ("class/asd", "class/asd traits", "class", "class/asd_traits") if k in df]
    if target is None and len(targets) != 1:
        raise ValueError("Specify the outcome column with --target. Do not use family history.")
    target = target.strip().lower() if target else targets[0]
    if target not in df or target in {*FIELDS, *ALIASES, "age", "result", *(s.lower() for s in SCORES)}:
        raise ValueError(f"Invalid outcome column: {target!r}")
    y = binary(df[target], target, True)
    X = pd.DataFrame(index=df.index)
    for i, col in enumerate(SCORES):
        if col.lower() not in df:
            raise ValueError(f"Missing required column {col}")
        values = binary(df[col.lower()], col, raw_answers)
        X[col] = values.map(lambda yes: point(i, bool(yes))) if raw_answers else values
    if "result" in df:
        totals = pd.to_numeric(df["result"], errors="coerce")
        matches = totals.eq(X.sum(axis=1))
        print(f"CSV result matches item sum on {matches.sum()}/{len(df)} rows. "
              "If result is the documented item total, investigate any mismatch before use.")
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 2 or len(y) < 6:
        raise ValueError("Need enough examples of both classes for an 80/20 stratified split.")
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    if min(train_y.nunique(), test_y.nunique()) < 2:
        raise ValueError("Both classes must appear in training and evaluation. Add minority examples.")

    # Background controls use training-only metadata. The forest sees only item points.
    background = df.loc[train_x.index]
    age_default = None
    if "age" in background:
        ages = pd.to_numeric(background["age"], errors="coerce").dropna()
        if not ages.empty:
            age_default = float(ages.median())
    maps = {}
    for col in FIELDS:
        if col in background:
            values = background[col].dropna().astype(str).str.strip().str.lower()
            maps[col] = {v: i for i, v in enumerate(sorted(set(values) - {""}))}
    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    pos = list(model.classes_).index(1)
    auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, pos])
    report = classification_report(test_y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_y, pred, labels=[0, 1]).ravel()
    print(report, f"ROC-AUC: {auc:.3f}; missed positive labels: {fn}/{fn + tp}")
    return dict(version=VERSION, model=model, features=list(train_x.columns), maps=maps,
                age_default=age_default, auc=auc, report=report, confusion=[[tn, fp], [fn, tp]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "autism_data.csv")
    parser.add_argument("--target", help="Explicit binary outcome header if not recognised")
    parser.add_argument("--raw-answers", action="store_true",
                        help="Only for raw adult item responses where 1 means Yes and 0 means No")
    args = parser.parse_args()
    bundle = train(args.csv, args.raw_answers, args.target)
    with NamedTemporaryFile(dir=ROOT, delete=False) as tmp:
        pickle.dump(bundle, tmp)
    Path(tmp.name).replace(MODEL_PATH)
    print(f"Saved {MODEL_PATH}")
