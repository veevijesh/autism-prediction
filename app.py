"""Start with: python -m streamlit run app.py
Requires autism_model_v2.pkl, created by: python train_final.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "autism_model_v2.pkl"

# Adult AQ-10 key. True: a "Yes" earns one point. False: a "No" earns one point.
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
LABELS = [f"Q{i}" for i in range(1, 11)]


def point(item_index, yes):
    return int(yes == ITEMS[item_index][1])


st.set_page_config(page_title="Adult questionnaire explorer", layout="wide")
st.title("Adult autism questionnaire explorer")
st.write("Educational demonstration using adapted yes/no questions. This questionnaire "
         "cannot diagnose autism or rule it out.")


@st.cache_resource
def load_bundle(path, stamp):
    # The file timestamp is part of the cache key, so retraining reloads the model.
    with path.open("rb") as stream:
        return pickle.load(stream)


def per_answer_contributions(pipeline, row):
    """Exact log-odds contribution of each answer for a scaler + logistic regression pipeline."""
    z = pipeline.named_steps["scaler"].transform(row)[0]
    weights = pipeline.named_steps["model"].coef_[0]
    return pd.Series(weights * z, index=LABELS, name="Contribution")


def shap_contributions(pipeline, row):
    """SHAP values (in log-odds) for one respondent, or None if shap is not installed."""
    try:
        import shap
    except ImportError:
        return None
    z = pipeline.named_steps["scaler"].transform(row)
    # After StandardScaler the training mean of every feature is 0, so a zero row is the baseline.
    explainer = shap.LinearExplainer(pipeline.named_steps["model"], np.zeros((1, z.shape[1])))
    raw = explainer.shap_values(z)
    if isinstance(raw, list):  # Older SHAP versions return one array per class.
        raw = raw[-1]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[..., -1]
    return pd.Series(raw[0], index=LABELS, name="SHAP value")


bundle = None
try:
    stamp = MODEL_PATH.stat().st_mtime_ns
    loaded = load_bundle(MODEL_PATH, stamp)
    if isinstance(loaded, dict) and "pipeline" in loaded and "features" in loaded:
        bundle = loaded
    else:
        st.warning("Unrecognised model format. Run python train_final.py to rebuild it.")
except FileNotFoundError:
    st.info("The questionnaire works now. Run python train_final.py to enable the model tab.")
except (ImportError, AttributeError, EOFError, ValueError, pickle.UnpicklingError):
    st.warning("Could not load this model. Rebuild it with python train_final.py in this environment.")

assessment, explanation = st.tabs(["Questionnaire", "Model explanation"])

with assessment:
    choices = []
    for i, (question, _) in enumerate(ITEMS):
        choices.append(st.selectbox(f"{i + 1}. {question}", ["Choose an answer", "Yes", "No"], key=SCORES[i]))

    if st.button("Show my results", type="primary"):
        if "Choose an answer" in choices:
            st.warning("Please answer all ten questions.")
        else:
            points = [point(i, choice == "Yes") for i, choice in enumerate(choices)]
            st.session_state["last_submission"] = {"choices": choices.copy(), "points": points}

    submitted = st.session_state.get("last_submission")
    if submitted:
        points = submitted["points"]
        st.subheader("Last submitted results")
        st.caption("After editing an answer, select Show my results again to update these results.")
        st.metric("Answers scoring one point", f"{sum(points)} / 10")
        st.caption("This count describes the answers you gave. It is separate from the model output. "
                   "If you have concerns, you can discuss them with a qualified clinician at any score.")
        st.dataframe(pd.DataFrame({"Question": LABELS, "Answer": submitted["choices"], "Point": points}),
                     hide_index=True)
        if bundle:
            st.write("Open Model explanation to see how the trained model classified these answers.")

with explanation:
    if not bundle:
        st.info("A trained model is required for this tab. Run python train_final.py first.")
    elif not submitted:
        st.info("Complete and submit the questionnaire first.")
    else:
        pipeline = bundle["pipeline"]
        row = pd.DataFrame([submitted["points"]], columns=bundle["features"])
        classes = list(pipeline.classes_)
        output = pipeline.predict_proba(row)[0, classes.index(1)]
        label = pipeline.predict(row)[0]
        st.metric("Predicted dataset label", str(label))
        st.write(f"Model output for the positive dataset label: {output:.1%}.")
        st.caption("This estimates agreement with the training labels. It is not a person's "
                   "probability of autism. The model uses class weighting, so this number is "
                   "not calibrated to real-world prevalence.")

        st.subheader("What influenced this prediction?")
        try:
            contributions = shap_contributions(pipeline, row)
        except Exception:
            contributions = None
        if contributions is None:
            st.info("Install shap (pip install shap) to see SHAP values. Showing the exact "
                    "linear contributions instead.")
            contributions = per_answer_contributions(pipeline, row)
        st.caption("SHAP values: each bar shows how much one answer pushed the result toward "
                   "label 1 (positive) or label 0 (negative), compared with an average respondent. "
                   "Values are in log-odds and add up to the model's output.")
        st.bar_chart(contributions.sort_values().to_frame())

        st.subheader("How the model weighs each question overall")
        model = pipeline.named_steps["model"]
        per_point = pd.Series(model.coef_[0] / pipeline.named_steps["scaler"].scale_,
                              index=LABELS, name="Weight")
        st.caption("Change in log-odds for label 1 when that question scores a point "
                   "(larger means the model relies on it more).")
        st.bar_chart(per_point.sort_values(ascending=False).to_frame())
        st.caption("For cross-validated performance of the model and alternatives, "
                   "open the Model Comparison page in the sidebar.")