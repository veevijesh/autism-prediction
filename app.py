import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Autism Spectrum Screening Tool", page_icon="🧠", layout="wide")

st.title("🧠 Autism Spectrum Trait Screening Tool")
st.write(
    "A preliminary screening demo based on the AQ-10 questionnaire. The model is a Logistic "
    "Regression trained on 800 labeled adult respondents (Kaggle Autism Prediction dataset)."
)
st.info(
    "⚠️ Educational screening demonstration for adults (18+). It does not replace a clinical diagnosis."
)


@st.cache_resource
def load_bundle():
    with open("autism_model_v2.pkl", "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
pipeline = bundle["pipeline"]
features = bundle["features"]

questions = {
    "A1_Score": "A1. Do you easily notice small sounds or details that others miss?",
    "A2_Score": "A2. Do you usually concentrate more on the whole picture rather than small details?",
    "A3_Score": "A3. Do you find it easy to do more than one task at the same time?",
    "A4_Score": "A4. If there is an interruption, can you switch back to what you were doing easily?",
    "A5_Score": "A5. Do you find it easy to 'read between the lines' when someone is talking?",
    "A6_Score": "A6. Do you know how to tell if someone listening to you is getting bored?",
    "A7_Score": "A7. When reading a story, do you find it difficult to work out characters' intentions?",
    "A8_Score": "A8. Do you like to collect information about categories of things (e.g., cars, birds, plants)?",
    "A9_Score": "A9. Do you find it easy to understand what someone is feeling just by looking at their face?",
    "A10_Score": "A10. Do you find it difficult to make new friends?",
}

# AQ-10 scoring key: a point is earned by "Yes" on these questions and by "No" on the rest
yes_earns_point = ["A1_Score", "A7_Score", "A8_Score", "A10_Score"]

tab1, tab2 = st.tabs(["📋 Screening Questionnaire", "📊 Model Insights"])

with tab1:
    st.subheader("AQ-10 Screening Questionnaire")
    scores = {}
    for col, text in questions.items():
        choice = st.selectbox(text, options=["No", "Yes"], key=col)
        if col in yes_earns_point:
            scores[col] = 1 if choice == "Yes" else 0
        else:
            scores[col] = 1 if choice == "No" else 0

    if st.button("Predict ASD Trait Likelihood", type="primary"):
        x_new = pd.DataFrame([scores])[features]
        prob = pipeline.predict_proba(x_new)[0][1] * 100
        prediction = int(pipeline.predict(x_new)[0])
        points = sum(scores.values())

        st.markdown("---")
        if prediction == 1:
            st.error(f"**Higher likelihood of ASD traits** (model score: {prob:.1f}%)")
        else:
            st.success(f"**Lower likelihood of ASD traits** (model score: {prob:.1f}%)")
        st.write(f"AQ-10 points earned: **{points} out of 10**")
        st.caption(
            "The model score is the output of a Logistic Regression and is not a clinical "
            "probability. A higher score means the answers resemble those of respondents "
            "labeled positive in the training data."
        )

with tab2:
    st.subheader("What the model relies on")
    coefs = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"Question": features, "Weight": coefs}).sort_values(
        "Weight", ascending=False
    )
    st.caption("A larger positive weight means a point on that question pushes the score up more.")
    st.bar_chart(coef_df.set_index("Question"))
    st.dataframe(coef_df)

    st.subheader("Cross-validated performance")
    try:
        res = pd.read_csv("results.csv")
        row = res[
            (res["Features"] == "Questionnaire only (10)")
            & (res["Model"] == "Logistic Regression")
        ]
        st.dataframe(row)
        st.write(
            "5-fold stratified cross-validation on 800 respondents. "
            "See the Model Comparison page for all models and feature sets."
        )
    except FileNotFoundError:
        st.write("Run experiments.py to generate results.csv.")