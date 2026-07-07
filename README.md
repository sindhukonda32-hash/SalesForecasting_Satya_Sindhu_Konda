# 📈 End-to-End Sales Forecasting & Demand Intelligence System

## 📌 Project Overview

This project was developed as part of a Data Science Internship Final Assessment. It provides a complete end-to-end solution for analyzing historical sales data, forecasting future demand, detecting sales anomalies, segmenting products based on demand patterns, and presenting business insights through an interactive Streamlit dashboard.

---

## 🎯 Objectives

- Analyze historical sales data
- Forecast future sales using multiple forecasting models
- Compare forecasting model performance
- Detect unusual sales behavior
- Segment products based on demand
- Build an interactive business dashboard

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost

---

## 📂 Project Structure

```
SalesForecasting/

├── analysis.ipynb
├── app.py
├── requirements.txt
├── README.md
├── summary.pdf
├── train.csv
├── vgsales.csv
└── charts/
```

---

## 📊 Project Workflow

### Task 1 – Data Preprocessing
- Data loading
- Data cleaning
- Missing value handling
- Feature engineering
- Exploratory Data Analysis (EDA)

### Task 2 – Time Series Analysis
- Monthly sales aggregation
- Seasonal decomposition
- Stationarity testing using ADF Test

### Task 3 – Sales Forecasting
Implemented and compared three forecasting models:

- SARIMA
- Prophet
- XGBoost

Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

### Task 4 – Forecast Analysis
- Category-wise forecasting
- Region-wise forecasting
- Business interpretation

### Task 5 – Anomaly Detection
Implemented:
- Isolation Forest
- Z-Score Analysis

Purpose:
- Detect unusual sales behavior
- Identify abnormal business patterns

### Task 6 – Demand Segmentation
Implemented:
- K-Means Clustering
- PCA Visualization

Purpose:
- Segment products into different demand groups
- Support inventory and marketing decisions

### Task 7 – Interactive Dashboard

Developed using Streamlit.

Dashboard Features:
- Business Overview
- Sales Dashboard
- Forecast Explorer
- Anomaly Detection
- Demand Segmentation
- Interactive Charts
- KPI Cards
- Region & Category Filters

### Task 8 – Business Summary

The project concludes with business insights and recommendations based on forecasting, anomaly detection, and demand segmentation results.

---

## 📈 Forecasting Models

| Model | Purpose |
|--------|----------|
| SARIMA | Statistical Time Series Forecasting |
| Prophet | Trend & Seasonality Forecasting |
| XGBoost | Machine Learning Forecasting |

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 📊 Dashboard Modules

- Home Dashboard
- Sales Dashboard
- Forecast Explorer
- Anomaly Detection
- Demand Segmentation
- About Project

---

## 💼 Business Applications

- Sales Forecasting
- Inventory Planning
- Demand Prediction
- Business Intelligence
- Market Analysis
- Decision Support System

---

## 🔮 Future Improvements

- Real-time forecasting
- Cloud deployment
- Automated model retraining
- Live business analytics
- Advanced forecasting models

---

## 👨‍💻 Developer

**Satya Sindhu Konda**

B.Tech – Artificial Intelligence & Machine Learning

Python Developer | AI & ML Enthusiast

GitHub:
https://github.com/sindhukonda32-hash

LinkedIn:
https://www.linkedin.com/in/sindhu-konda-ab5089325/

---

## ⭐ Acknowledgement

This project was developed as part of a Data Science Internship Final Assessment to demonstrate practical skills in data analysis, machine learning, forecasting, visualization, and dashboard development.