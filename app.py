import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Autism Spectrum Screening Tool",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Autism Spectrum Trait Screening Tool")
st.write(
    "An ML-powered preliminary screening application based on the AQ-10 behavioral assessment."
)

st.info(
    "⚠️ Disclaimer: Educational and screening purposes only. Does not replace clinical diagnosis."
)


# Load saved model pipeline
@st.cache_resource
def load_pipeline():
    with open("autism_model.pkl", "rb") as f:
        return pickle.load(f)


pipeline = load_pipeline()
model = pipeline["model"]
scaler = pipeline["scaler"]
feature_names = pipeline["feature_names"]
feature_encoders = pipeline["feature_encoders"]
classes = pipeline.get("classes", [0, 1])

# Create Tabs for FYP Presentation
tab1, tab2 = st.tabs(["📋 Screening Questionnaire", "📊 Model Analytics & Insights"])

# --- TAB 1: SCREENING TOOL ---
with tab1:
    aq10_questions = {
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

    demographic_labels = {
        "gender": "Gender",
        "ethnicity": "Ethnicity / Race",
        "jaundice": "Born with Jaundice (Yellowing of skin/eyes at birth)?",
        "austim": "Family History of Autism Spectrum Disorder (ASD)?",
        "contry_of_res": "Country of Residence",
        "used_app_before": "Have you used this screening application before?",
        "relation": "Who is completing this assessment?",
        "age": "Age (in years)",
    }

    value_display_mappings = {
        "gender": {"f": "Female", "m": "Male"},
        "jaundice": {"yes": "Yes", "no": "No"},
        "austim": {"yes": "Yes", "no": "No"},
        "used_app_before": {"yes": "Yes", "no": "No"},
    }

    st.subheader("1. AQ-10 Behavioral Screening Questionnaire")
    user_inputs = {}

    for col in feature_names:
        if col in aq10_questions:
            choice = st.selectbox(
                aq10_questions[col], options=["No (0)", "Yes (1)"]
            )
            user_inputs[col] = 1 if choice == "Yes (1)" else 0

    st.markdown("---")
    st.subheader("2. Participant Demographic Information")

    for col in feature_names:
        if col not in aq10_questions:
            label = demographic_labels.get(col, col.replace("_", " ").title())

            if col in feature_encoders:
                raw_classes = list(feature_encoders[col].classes_)
                mapping = value_display_mappings.get(col, {})
                display_options = [mapping.get(c, str(c).title()) for c in raw_classes]
                option_map = dict(zip(display_options, raw_classes))

                selected_display = st.selectbox(label, display_options)
                raw_val = option_map[selected_display]

                user_inputs[col] = int(
                    feature_encoders[col].transform([raw_val])[0]
                )
            else:
                val = st.number_input(
                    label, min_value=0, max_value=100, value=25
                )
                user_inputs[col] = val

    if st.button("Predict ASD Trait Likelihood", type="primary"):
        input_df = pd.DataFrame([user_inputs])[feature_names]
        input_scaled = scaler.transform(input_df)

        prediction = int(model.predict(input_scaled)[0])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            if 1 in classes and len(probs) > list(classes).index(1):
                prob_asd = probs[list(classes).index(1)] * 100
            else:
                prob_asd = probs[-1] * 100
        else:
            prob_asd = 100.0 if prediction == 1 else 0.0

        st.markdown("---")
        if prediction == 1:
            st.error(
                f"**High Likelihood of ASD Traits** (Probability Score: {prob_asd:.1f}%)"
            )
            st.write(
                "The model identified behavioral responses consistent with autism spectrum trait indicators."
            )
        else:
            st.success(
                f"**Low Likelihood of ASD Traits** (Probability Score: {prob_asd:.1f}%)"
            )
            st.write(
                "The model did not identify significant behavioral patterns aligned with autism spectrum indicators."
            )

# --- TAB 2: MODEL PERFORMANCE FOR FYP PRESENTATION ---
with tab2:
    st.subheader("📊 Machine Learning Model Analytics")
    st.write(
        "This section highlights the trained Random Forest Classifier pipeline features and importance metrics."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Model Architecture", value="Random Forest Classifier")
        st.metric(label="Ensemble Estimators", value="100 Decision Trees")
    with col2:
        st.metric(label="Scaling Standardizer", value="StandardScaler (Z-Score)")
        st.metric(label="Total Features Analyzed", value=len(feature_names))

    st.markdown("---")
    st.subheader("🌲 Feature Importance Breakdown")
    st.caption("Shows which survey questions contributed most heavily to the model's predictions.")

    importances = model.feature_importances_
    feat_df = pd.DataFrame(
        {"Feature Name": feature_names, "Importance Score": importances}
    ).sort_values("Importance Score", ascending=False)

    st.bar_chart(feat_df.set_index("Feature Name"))
    st.dataframe(feat_df, use_container_width=True)