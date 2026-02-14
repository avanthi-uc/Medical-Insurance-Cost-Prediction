
# 🏥 Medical Insurance Cost Prediction

An end-to-end Machine Learning project to predict individual medical insurance charges using demographic and health-related features.

---

## 🚀 Project Overview

This project builds a complete regression pipeline to estimate medical insurance costs based on:

- Age
- Gender
- BMI
- Smoking Status
- Number of Children
- Region

The best-performing model was deployed using Streamlit in an interactive web application.

---

## 📊 Business Use Cases

- Personalized insurance premium estimation  
- Risk assessment for insurance providers  
- Cost transparency for policyholders  
- Financial planning support  

---

## 🔬 Machine Learning Pipeline

### 1️⃣ Data Preprocessing
- Handled missing & duplicate values
- Encoded categorical features
- Scaled numeric variables
- Performed feature engineering

### 2️⃣ Model Building
Trained multiple regression models:

- Linear Regression
- Random Forest
- XGBoost (Best Performing Model)
- Ridge Regression
- Lasso Regression
- Gradient Boost Regressor

Evaluated using:
- RMSE
- MAE
- R² Score

---

## 📈 Exploratory Data Analysis (EDA)

Key Insights:

- Charges are right-skewed
- Smoking has the strongest impact on insurance costs
- Age positively correlates with charges
- Obese smokers pay significantly higher premiums

---

## 🧠 Risk Logic Added

The app includes a custom risk classification system:

- 🔴 High Risk → Smoker + Age > 40 + BMI > 29.9
- 🟡 Moderate Risk → Any 2 risk factors
- 🟢 Low Risk → 0–1 risk factors

---

## 💻 Streamlit Application Features

- Interactive EDA Dashboard
- Multivariate Analysis
- Outlier Detection
- Correlation Analysis
- Insurance Cost Prediction
- Health Profile Analyzer

---

## 📂 Project Structure

```
medical-insurance-ml/
│
├── models/
├── data/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
- MLflow

---

## 📎 Presentation

Project presentation available in repository:
`Medical_Insurance_ML_Project_Presentation.pptx`

---



