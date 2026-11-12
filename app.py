"""Start with: python -m streamlit run app.py"""
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from main import FIELDS, ITEMS, MODEL_PATH, SCORES, VERSION, point

st.set_page_config(page_title="Adult questionnaire explorer", layout="wide")
st.title("Adult autism questionnaire explorer")
st.write("Educational demonstration using adapted yes/no questions. This questionnaire "
         "cannot diagnose autism or rule it out.")


@st.cache_resource
def load_bundle(path, stamp):
    # The file timestamp is part of the cache key so retraining reloads the model.
    with path.open("rb") as stream:
        return pickle.load(stream)


def show_contributions(model, row, positive_index):
    try:
        import shap
    except ImportError:
        st.info("Install shap to view the per-answer model explanation.")
        return
    raw = shap.TreeExplainer(model).shap_values(row)
    if isinstance(raw, list):  # Older SHAP versions return one array per class.
        values = raw[positive_index][0]
    elif np.ndim(raw) == 3:
        values = raw[0, :, positive_index]
    else:
        st.info("This SHAP version did not return class-specific contributions.")
        return
    contributions = pd.Series(values, index=row.columns, name="Contribution")
    st.caption("Positive contributions raise the model output for label 1 relative to its baseline.")
    st.bar_chart(contributions.sort_values().to_frame())


bundle = None
stamp = None
try:
    stamp = MODEL_PATH.stat().st_mtime_ns
    loaded = load_bundle(MODEL_PATH, stamp)
    if isinstance(loaded, dict) and loaded.get("version") == VERSION:
        bundle = loaded
    else:
        st.warning("Old model format. Run python main.py --csv train.csv to enable the model tabs.")
except FileNotFoundError:
    st.info("The questionnaire works now. Run python main.py --csv train.csv to enable the model tabs.")
except (ImportError, AttributeError, EOFError, ValueError, pickle.UnpicklingError):
    st.warning("Could not load this model. Rebuild it with python main.py --csv train.csv in this environment.")

assessment, explanation, analytics = st.tabs(["Questionnaire", "Model explanation", "Model analytics"])
with assessment:
    # Start with a complete input record. Item defaults are overwritten on valid submission.
    inputs = dict.fromkeys(SCORES, 0)
    if bundle:
        inputs.update(dict.fromkeys(bundle["maps"], -1))
        if bundle["age_default"] is not None:
            inputs["age"] = float(bundle["age_default"])

    choices = []
    for i, (question, _) in enumerate(ITEMS):
        choices.append(st.selectbox(f"{i + 1}. {question}", ["Choose an answer", "Yes", "No"], key=SCORES[i]))
    if bundle and ("age" in inputs or bundle["maps"]):
        with st.expander("Optional background"):
            st.caption("Background information is excluded from both the score and the model.")
            if "age" in inputs:
                inputs["age"] = st.number_input("Age in years", min_value=0.0, step=1.0,
                                                value=max(0.0, inputs["age"]))
                st.caption("Defaults to the median age in the training split.")
            for col, mapping in bundle["maps"].items():
                value = st.selectbox(FIELDS[col], [None, *mapping],
                                     format_func=lambda v: "Unspecified" if v is None else str(v),
                                     key=col)
                inputs[col] = mapping.get(value, -1)

    if st.button("Show my results", type="primary"):
        if "Choose an answer" in choices:
            st.warning("Please answer all ten questions.")
        else:
            points = [point(i, choice == "Yes") for i, choice in enumerate(choices)]
            inputs.update(zip(SCORES, points))
            st.session_state["last_submission"] = {
                "choices": choices.copy(), "inputs": inputs.copy(), "points": points.copy()
            }

    submitted = st.session_state.get("last_submission")
    if submitted:
        points = submitted["points"]
        st.subheader("Last submitted results")
        st.caption("After editing an answer, select Show my results again to update these results.")
        st.metric("Answers scoring one point", f"{sum(points)} / 10")
        st.caption("This count describes the answers you gave. It is separate from the model output. "
                   "If you have concerns, you can discuss them with a qualified clinician at any score.")
        st.dataframe(pd.DataFrame({"Question": SCORES, "Answer": submitted["choices"], "Point": points}),
                     hide_index=True)
        if bundle:
            st.write("Open Model explanation to see how the trained model classified these answers.")

with explanation:
    if not bundle:
        st.info("A freshly trained model is required for this tab.")
    elif not submitted:
        st.info("Complete and submit the current questionnaire first.")
    else:
        model = bundle["model"]
        row = pd.DataFrame([submitted["inputs"]], columns=bundle["features"])
        pos = list(model.classes_).index(1)
        output = model.predict_proba(row)[0, pos]
        label = model.predict(row)[0]
        st.metric("Predicted dataset label", str(label))
        st.write(f"Model output for the positive dataset label: {output:.1%}.")
        st.caption("This estimates agreement with the training labels. It is not a person's probability of autism.")
        st.subheader("What influenced this model prediction?")
        show_contributions(model, row, pos)

with analytics:
    if bundle:
        st.subheader("Held-out evaluation against the CSV labels")
        st.metric("ROC-AUC", f"{bundle['auc']:.3f}")
        st.code(bundle["report"])
        st.dataframe(pd.DataFrame(bundle["confusion"], index=["Actual 0", "Actual 1"],
                                  columns=["Predicted 0", "Predicted 1"]))
        st.subheader("Global feature importance")
        st.bar_chart(pd.Series(bundle["model"].feature_importances_, index=bundle["features"],
                               name="Importance").sort_values(ascending=False).to_frame())
    else:
        st.info("Train the model to see its held-out evaluation.")
