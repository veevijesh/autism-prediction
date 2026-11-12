# Autism Spectrum Trait Screening Tool

A Streamlit app that screens for autism spectrum (ASD) traits from the 10-item adult AQ-10 questionnaire. It returns a model output from a Logistic Regression, explains each prediction with SHAP, and includes a Model Comparison page with cross-validated results for two models and three feature sets.

> **This is an educational screening demonstration. It is not a diagnostic tool.** It cannot diagnose autism or rule it out. If you have concerns, speak with a qualified clinician.

## What it does

- Takes the 10 AQ-10 answers (Yes/No), applies the AQ-10 scoring key, and shows how many answers score a point (out of 10).
- Returns a model output for the positive dataset label from a Logistic Regression trained on the 10 item scores.
- Explains each prediction with **SHAP**: a bar chart showing how much each answer pushed that respondent's result toward label 1 or label 0.
- Shows how the model weighs each question overall.
- Includes a **Model Comparison** page with 5-fold cross-validated results for Logistic Regression and Random Forest across three feature sets.

## Data

- Kaggle "Autism Prediction" competition training set: https://www.kaggle.com/competitions/autismdiagnosis/data
- 800 adult respondents, 161 positive (20.1%). The label is the dataset's own `Class/ASD` column.
- The data file is not included in this repository. Download `train.csv` from the Kaggle page and place it in the project folder.

## Results (5-fold stratified cross-validation)

Always predicting "no ASD" scores about 0.80 accuracy, so accuracy alone is misleading. Recall, precision, and ROC-AUC are the metrics that matter.

| Features | Model | Recall | Precision | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Questionnaire only (10) | Logistic Regression | 0.882 | 0.525 | 0.815 | 0.909 |
| Questionnaire only (10) | Random Forest | 0.782 | 0.542 | 0.821 | 0.889 |
| Background only (8) | Logistic Regression | 0.733 | 0.472 | 0.779 | 0.834 |
| Background only (8) | Random Forest | 0.546 | 0.481 | 0.790 | 0.828 |
| All features (18) | Logistic Regression | 0.819 | 0.550 | 0.826 | 0.898 |
| All features (18) | Random Forest | 0.726 | 0.602 | 0.846 | 0.905 |

**Key findings**

- On the questionnaire alone, Logistic Regression had the highest recall (0.88) and ROC-AUC (0.91), and it is the simplest model, so it was chosen as the deployed model.
- Background information alone is informative (ROC-AUC about 0.83) but weaker than the questionnaire.
- Adding background information does not improve Logistic Regression and slightly improves Random Forest.
- Precision is about 0.47 to 0.60 for every model, so roughly half of the people flagged are true positives. The tool works as a first-pass screen, not a diagnosis.
- Differences between models are small relative to the noise expected from 800 rows, so they should not be over-read.

## How it works

```
train.csv -> 10 item scores -> StandardScaler + Logistic Regression (class_weight="balanced")
          -> 5-fold stratified cross-validation -> fit on all 800 rows -> autism_model_v2.pkl
                                                                              |
                         Streamlit app loads the model, scores answers, and explains them with SHAP
```

- **Features:** only the 10 item scores. Demographic and background columns are excluded from the deployed model to limit reliance on them. This reduces dependence on those variables but does not prove the model is fair.
- **Preprocessing inside the pipeline:** scaling is re-fit within each cross-validation fold, so nothing leaks from the validation data.
- **Explanations:** SHAP `LinearExplainer` on the Logistic Regression. After scaling, the training mean of every feature is 0, so the baseline is the average respondent.
- **Model reloading:** the app's cache key includes the model file's timestamp, so a retrained model is picked up automatically without restarting the server.

## Project structure

```
.
├── app.py                       # Streamlit app: questionnaire, prediction, SHAP explanation
├── pages/
│   └── 1_Model_Comparison.py    # Model comparison page (reads results.csv)
├── train_final.py               # Cross-validates and saves the deployed Logistic Regression
├── experiments.py               # Compares 2 models x 3 feature sets, writes results.csv
├── results.csv                  # Output of experiments.py
├── main.py                      # Earlier Random Forest prototype, kept for reference
└── README.md
```

## Setup and usage

```bash
git clone https://github.com/veevijesh/autism-prediction.git
cd autism-prediction
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install pandas numpy scikit-learn streamlit shap
```

1. Download `train.csv` from the Kaggle page above and place it in the project folder.
2. Train and save the model:
   ```bash
   python train_final.py
   ```
3. (Optional) Regenerate the comparison table:
   ```bash
   python experiments.py
   ```
4. Run the app:
   ```bash
   python -m streamlit run app.py
   ```

If `autism_model_v2.pkl` is missing, the questionnaire still works and the model tab shows instructions to train it.

## Limitations

- 800 adult respondents from a Kaggle competition dataset, so results may not generalise to other populations.
- The labels are the dataset's own, not clinical ground truth, so the metrics describe agreement with those labels, not diagnostic accuracy.
- The dataset has inconsistencies (for example, ages below 18 labeled "18 and more"), so it should be treated as a demonstration dataset.
- Models use default settings and a 0.5 decision threshold, with no hyperparameter or threshold tuning.
- Cross-validation on 800 rows gives noisy estimates, and fold-level variation is not reported.
- The model output is not a probability of autism, and because of class weighting it is not calibrated to real-world prevalence.
- `autism_model_v2.pkl` is a pickle file. Only load pickle files from sources you trust.

## Possible next steps

- Report standard deviations across folds and tune the decision threshold for a recall-focused screen.
- Compare against a simple rule baseline (for example, item total of 6 or more).
- Evaluate performance by subgroup to check fairness.
- Validate on an independent dataset with clinically assigned labels.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, SHAP