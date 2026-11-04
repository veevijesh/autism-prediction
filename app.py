import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(
    page_title="Explainable Autism Spectrum Screening Tool",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Explainable ML Autism Trait Screening System")
st.write(
    "An advanced clinical decision support tool using Random Forest classification and SHAP (SHapley Additive exPlanations)."
)
st.info(
    "⚠️ Disclaimer: Educational and preliminary screening purposes only. Does not replace clinical medical diagnosis."
)

@st.cache_resource
def load_pipeline():
    with open("autism_model.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()
model = pipeline["model"]
scaler = pipeline["scaler"]
feature_names = pipeline["feature_names"]
category_maps = pipeline["category_maps"]

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

# Session state initialization
if "evaluated" not in st.session_state:
    st.session_state["evaluated"] = False
if "user_inputs" not in st.session_state:
    st.session_state["user_inputs"] = {}

tab1, tab2, tab3 = st.tabs([
    "📋 Patient Assessment", 
    "🔍 Explainable AI (XAI) Diagnosis", 
    "📊 Dataset & Model Analytics"
])

# --- TAB 1: QUESTIONNAIRE ---
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

    # AQ-10 Clinical Scoring Standard:
    # "Yes" indicates trait for A1, A7, A8, A10
    # "No" indicates trait (reverse-scored) for A2, A3, A4, A5, A6, A9
    yes_is_trait = ["A1_Score", "A7_Score", "A8_Score", "A10_Score"]

    st.subheader("1. Behavioral Diagnostic Indicators (AQ-10)")
    user_inputs = {}

    col_a, col_b = st.columns(2)
    for idx, (col, q_text) in enumerate(aq10_questions.items()):
        if col in feature_names:
            target_container = col_a if idx % 2 == 0 else col_b
            choice = target_container.selectbox(
                q_text, options=["No", "Yes"], key=col
            )
            
            # Apply clinical scoring key (1 = Trait Present, 0 = Trait Absent)
            if col in yes_is_trait:
                user_inputs[col] = 1 if choice == "Yes" else 0
            else:
                user_inputs[col] = 1 if choice == "No" else 0

    st.markdown("---")
    st.subheader("2. Demographic & Medical Background")

    demographics = {
        "age": st.number_input("Age (Years)", min_value=0, max_value=100, value=25, key="age"),
        "gender": st.selectbox("Gender", options=["Female", "Male"], key="gender"),
        "ethnicity": st.selectbox("Ethnicity / Race", options=["Asian", "White-European", "Middle Eastern", "Black", "Others"], key="ethnicity"),
        "jaundice": st.selectbox("Born with Jaundice?", options=["No", "Yes"], key="jaundice"),
        "austim": st.selectbox("Family History of ASD?", options=["No", "Yes"], key="austim"),
        "contry_of_res": st.selectbox("Country of Residence", options=["United Kingdom", "United States", "India", "Others"], key="contry_of_res"),
        "used_app_before": st.selectbox("Used Screening App Before?", options=["No", "Yes"], key="used_app_before"),
        "relation": st.selectbox("Assessment completed by", options=["Self", "Parent", "Relative", "Others"], key="relation"),
    }

    for demo_col, val in demographics.items():
        if demo_col in feature_names:
            if demo_col == "age":
                user_inputs[demo_col] = float(val)
            else:
                str_val = str(val).strip().lower()
                mapping = category_maps.get(demo_col, {})
                user_inputs[demo_col] = mapping.get(str_val, 0)

    predict_btn = st.button("Generate Diagnostic Risk Assessment", type="primary")

    if predict_btn:
        st.session_state["evaluated"] = True
        st.session_state["user_inputs"] = user_inputs
        st.success("✅ Assessment submitted! Switch to the **Explainable AI Diagnosis** tab to view results.")

# --- TAB 2: EXPLAINABLE AI DIAGNOSIS ---
with tab2:
    if st.session_state.get("evaluated", False):
        inputs = st.session_state["user_inputs"]
        input_df = pd.DataFrame([inputs])[feature_names]
        input_scaled = scaler.transform(input_df)

        # Get exact class 1 probability
        probs = model.predict_proba(input_scaled)[0]
        prob_asd = probs[1] * 100

        prediction = 1 if prob_asd >= 45.0 else 0

        st.subheader("Diagnostic Risk Output")
        if prediction == 1:
            st.error(f"⚠️ **High ASD Trait Likelihood Detected** (Risk Score: {prob_asd:.1f}%)")
            st.write("The ML model detected significant behavioral pattern alignment with autism spectrum trait indicators.")
        else:
            st.success(f"✅ **Low ASD Trait Likelihood Detected** (Risk Score: {prob_asd:.1f}%)")
            st.write("The model did not identify significant behavioral patterns aligned with autism spectrum indicators.")

        st.markdown("---")
        st.subheader("🔍 Local Explainability (SHAP Contribution Analysis)")

        # Compute SHAP values safely across SHAP library versions
        raw_shap = explainer(input_scaled)
        
        # Handle SHAP dimensionality variations (2D/3D arrays)
        if isinstance(raw_shap.values, list):
            vals_array = raw_shap.values[1][0]
            base_val = raw_shap.base_values[1][0]
        elif len(raw_shap.values.shape) == 3:
            vals_array = raw_shap.values[0, :, 1]
            base_val = raw_shap.base_values[0, 1]
        else:
            vals_array = raw_shap.values[0]
            base_val = raw_shap.base_values[0]

        shap_exp = shap.Explanation(
            values=vals_array,
            base_values=base_val,
            data=input_df.iloc[0].values,
            feature_names=feature_names
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.bar(shap_exp, show=False)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Please fill in the assessment in Tab 1 and click **Generate Diagnostic Risk Assessment**.")

# --- TAB 3: MODEL ANALYTICS ---
with tab3:
    st.subheader("📊 Model Performance & Global Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame(
        {"Feature": feature_names, "Importance Score": importances}
    ).sort_values("Importance Score", ascending=False)

    st.bar_chart(feat_df.set_index("Feature"))
    st.dataframe(feat_df, use_container_width=True)