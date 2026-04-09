# 📡 Telecom Customer Churn Predictor

A production-ready machine learning pipeline that predicts whether a telecom customer will churn based on their account, demographic, and service-usage attributes. The project covers the full ML lifecycle — from EDA and feature engineering to hyperparameter tuning, model evaluation, and deployment via an interactive Streamlit web app.

---

## 📌 Problem Statement

Given a telecom customer's profile and service usage, predict whether they will **churn (leave the company)**.  
This is a **supervised binary classification** problem trained on ~7,000 real-world telecom customer records.

---

## 🗂️ Project Structure

```
telecom-churn-predictor/
│
├── app.py                              # Streamlit web application
├── model/
│   └── best_model_pipeline.pkl         # Serialized model artifact (pipeline, or {"model": pipeline, ...})
├── data/
│   └── telecom customer churn.csv      # Raw dataset (~7k records)
├── notebook/
│   └── telecom_churn_predictor.ipynb   # End-to-end ML notebook
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source   | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows     | 7,043 |
| Features | 19 (mix of binary, categorical, and numerical) |
| Target   | `Churn` — Yes / No |
| Class balance | ~73% No Churn / ~27% Churn |

**Feature descriptions:**

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Binary categorical | Customer gender: Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior citizen |
| `Partner` | Binary categorical | Whether the customer has a partner |
| `Dependents` | Binary categorical | Whether the customer has dependents |
| `tenure` | Numerical (int) | Months the customer has been with the company (0–72) |
| `PhoneService` | Binary categorical | Whether the customer has phone service |
| `MultipleLines` | Categorical | Multiple phone lines: Yes / No / No phone service |
| `InternetService` | Categorical | Internet type: DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Online security add-on: Yes / No / No internet service |
| `OnlineBackup` | Categorical | Online backup add-on: Yes / No / No internet service |
| `DeviceProtection` | Categorical | Device protection add-on: Yes / No / No internet service |
| `TechSupport` | Categorical | Tech support add-on: Yes / No / No internet service |
| `StreamingTV` | Categorical | TV streaming add-on: Yes / No / No internet service |
| `StreamingMovies` | Categorical | Movie streaming add-on: Yes / No / No internet service |
| `Contract` | Categorical | Contract type: Month-to-month / One year / Two year |
| `PaperlessBilling` | Binary categorical | Whether the customer uses paperless billing |
| `PaymentMethod` | Categorical | Payment method: 4 categories |
| `MonthlyCharges` | Numerical (float) | Monthly charge amount in USD |
| `TotalCharges` | Numerical (float) | Cumulative charges over tenure (originally stored as string) |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Removed **22 duplicate rows**
- Converted `TotalCharges` from `object` to `float64`; dropped **11 blank-space entries** that were masking as non-null strings
- Encoded the binary target: `Yes → 1`, `No → 0`
- No outliers detected in the three continuous features (IQR-based check)

### 2. Feature Engineering

| New Feature | Rationale |
|-------------|-----------|
| `NumAddons` | Count of active add-on services (0–6) — customers with more add-ons have higher switching costs and lower churn risk |
| `IsNewCustomer` | 1 if tenure ≤ 3 months, else 0 — new customers are at highest risk of early churn |

### 3. Preprocessing Pipelines

| Pipeline | Numerical step | Categorical step | Used for |
|----------|---------------|-----------------|----------|
| `preprocessor_with_scaling` | `StandardScaler` | `OneHotEncoder` | Scale-sensitive models (LR, KNN, SVC) |
| `preprocessor_no_scaling` | `passthrough` | `OneHotEncoder` | Tree-based models (DT, RF, XGBoost, LightGBM) |

`OneHotEncoder(handle_unknown='ignore')` is used for all categorical features to handle unseen categories gracefully at inference time.

### 4. Model Selection
Seven models were benchmarked via **5-fold stratified cross-validation** on the training set. Primary metric is **PR-AUC** (chosen over accuracy and ROC-AUC due to class imbalance):

| Model | CV PR-AUC | CV Recall | CV F1 |
|-------|-----------|-----------|-------|
| Logistic Regression | ~0.65 | ~0.80 | ~0.62 |
| KNN | ~0.50 | ~0.53 | ~0.54 |
| SVC | ~0.60 | ~0.77 | ~0.62 |
| Decision Tree | ~0.37 | ~0.49 | ~0.49 |
| Random Forest | ~0.60 | ~0.47 | ~0.53 |
| XGBoost | ~0.61 | ~0.60 | ~0.58 |
| LightGBM | ~0.65 | ~0.58 | ~0.61 |

Class imbalance was handled per model type: `class_weight='balanced'` for sklearn models, `scale_pos_weight` for XGBoost, and `is_unbalance=True` for LightGBM.

### 5. Hyperparameter Tuning
`RandomizedSearchCV` (100 iterations, 5-fold stratified CV, scoring = `recall`) was applied to three candidate models: **Logistic Regression**, **XGBoost**, and **LightGBM**.

### 6. Threshold Tuning
Rather than using the default 0.5 decision threshold, the Precision–Recall curve is plotted across all thresholds using `cross_val_predict` on the training set. Based on this analysis, a threshold of **0.40** is selected to improve recall on the minority (churn) class while keeping precision at an acceptable level for retention campaigns.

### 7. Final Evaluation

The best-performing model is selected by **CV Recall** and evaluated once on the held-out test set at threshold = 0.40:

> **Traceability note:** The following values are tied to the current notebook run and exported model artifact.  
> If you re-run training, treat these numbers as historical until you regenerate and update them from the notebook outputs.

| Metric | Score |
|--------|-------|
| PR-AUC | 0.6415 |
| ROC-AUC | 0.8462 |
| Recall (Churned) | 0.89 |
| Precision (Churned) | 0.47 |
| F1 (Churned) | 0.61 |
| Accuracy | 0.70 |

The model effectively catches high-risk churners (Recall = 0.89) at a threshold of 0.40, trading some precision for maximum coverage — a practical balance for designing targeted retention campaigns.

### 8. Artifact Contract (Training ↔ App)

The Streamlit app supports loading `model/best_model_pipeline.pkl` in either of these formats:

1. A direct scikit-learn pipeline/model object with `predict_proba`.
2. A dictionary containing at least:
   - `model`: the pipeline/model object used for inference.

At startup, `app.py` validates that:
- the artifact file exists and is readable,
- dict artifacts include the `model` key,
- the loaded model exposes `predict_proba`.

During prediction, if the loaded model exposes `feature_names_in_`, the app validates required feature names and then reorders input columns to the model's expected order. This avoids avoidable order-only failures while still protecting against train/serve schema drift.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
git clone https://github.com/amr-salah-shaheen/telecom-churn-predictor.git
cd telecom-churn-predictor
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Re-train the Model

Open and run all cells in the notebook:

```bash
jupyter notebook notebook/telecom_churn_predictor.ipynb
```

This will regenerate `model/best_model_pipeline.pkl`.

---

## 🖥️ Web App

The Streamlit app accepts a customer's full profile and returns an estimated churn probability in real-time.

**Input fields:**
- Account & services: Internet service type, contract type, payment method, monthly charges, tenure, total charges
- Demographics: Senior citizen, partner, dependents
- Billing: Paperless billing, phone service, multiple lines
- Add-ons: Online security, online backup, device protection, tech support, streaming TV, streaming movies

**Output:** Churn risk label (Churn Risk Detected / Likely to Stay), churn probability percentage, and a visual probability gauge bar.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Web App | `streamlit` |
| Visualisation | `matplotlib`, `seaborn` |
| Model Serialisation | `joblib`, `json` |

---"# telecom-churn-predictor" 
"# telecom-churn-predictor" 
