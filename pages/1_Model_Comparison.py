import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="🧪", layout="wide")
st.title("🧪 Model Comparison & Limitations")
st.write(
    "5-fold stratified cross-validation on 800 labeled respondents "
    "(161 positive, 20.1%) from the Kaggle Autism Prediction training set. "
    "The label is the dataset's own Class/ASD column."
)

res = pd.read_csv("results.csv")
st.dataframe(res)

st.subheader("Recall (share of true positives caught)")
st.bar_chart(
    res.pivot(index="Features", columns="Model", values="Recall"),
    stack=False, horizontal=True,
)

st.subheader("ROC-AUC (how well positives are ranked above negatives)")
st.bar_chart(
    res.pivot(index="Features", columns="Model", values="ROC-AUC"),
    stack=False, horizontal=True,
)

st.subheader("Key findings")
st.markdown(
    "- Always predicting 'no ASD' scores about 0.80 accuracy, so accuracy alone is misleading. Recall, precision and ROC-AUC are the metrics that matter.\n"
    "- The 10 questionnaire items alone give the best results for Logistic Regression (recall 0.88, ROC-AUC 0.91).\n"
    "- Background information alone is informative (ROC-AUC about 0.83) but weaker than the questionnaire.\n"
    "- Adding background information does not improve Logistic Regression and slightly improves Random Forest.\n"
    "- Precision is about 0.5 to 0.6 for every model, so roughly half of the people flagged are true positives. The tool works as a first-pass screen, not a diagnosis."
)

st.subheader("Limitations")
st.markdown(
    "- 800 adult respondents from a Kaggle competition dataset, so results may not generalize to other populations.\n"
    "- The dataset has inconsistencies (for example, ages below 18 labeled '18 and more'), so it should be treated as a demonstration dataset.\n"
    "- Models use default settings and a 0.5 decision threshold, with no tuning.\n"
    "- Cross-validation on 800 rows gives noisy estimates, so small differences between models should not be over-read.\n"
    "- This is a screening demonstration, not a diagnostic tool."
)