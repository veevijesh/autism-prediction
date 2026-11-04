# Autism Spectrum Trait Screening Tool

A Streamlit app that screens for autism spectrum (ASD) traits from the 10-item AQ-10 questionnaire, with a model comparison page.

## What it does
- Takes the 10 AQ-10 answers, applies the AQ-10 scoring key, and returns a model score from a Logistic Regression
- Shows which questions the model relies on most
- Includes a Model Comparison page with cross-validated results for two models and three feature sets

**This is an educational screening demonstration. It is not a diagnostic tool.**

## Data
- Kaggle "Autism Prediction" competition training set: https://www.kaggle.com/competitions/autismdiagnosis/data
- 800 adult respondents, 161 positive (20.1%), label from the dataset's `Class/ASD` column
- The data file is not included in this repository. Download `train.csv` from the Kaggle page and place it in the project folder.

## Results (5-fold stratified cross-validation)
| Features | Model | Recall | Precision | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Questionnaire only (10) | Logistic Regression | 0.882 | 0.525 | 0.815 | 0.909 |
| Questionnaire only (10) | Random Forest | 0.782 | 0.542 | 0.821 | 0.889 |
| Background only (8) | Logistic Regression | 0.733 | 0.472 | 0.779 | 0.834 |
| Background only (8) | Random Forest | 0.546 | 0.481 | 0.790 | 0.828 |
| All features (18) | Logistic Regression | 0.819 | 0.550 | 0.826 | 0.898 |
| All features (18) | Random Forest | 0.726 | 0.602 | 0.846 | 0.905 |

## Findings
- Always predicting "no ASD" scores about 0.80 accuracy, so recall, precision and ROC-AUC are the metrics that matter.
- Logistic Regression on the 10 questionnaire items alone gave the best recall and ROC-AUC, so it is the deployed model. It also avoids using ethnicity or country of residence.
- Background information alone is informative (ROC-AUC about 0.83) but weaker than the questionnaire.
- Precision is about 0.5 to 0.6 for every model, so the tool works as a first-pass screen only.

## Limitations
- 800 adult respondents from a Kaggle competition, so results may not generalize
- The dataset has inconsistencies (for example, ages below 18 labeled "18 and more")
- Default model settings and a 0.5 threshold, with no tuning
- Cross-validation on 800 rows gives noisy estimates
- A screening demonstration, not a diagnostic tool

## How to run
~~~
pip install -r requirements.txt
python experiments.py
python train_final.py
streamlit run app.py
~~~

`main.py` is the earlier prototype, kept for reference.