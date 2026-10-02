# Customer Churn Prediction & Analytics Platform

> An end-to-end customer churn analytics platform built with **PySpark, Databricks, Delta Lake, Spark MLlib, Python, and Streamlit** to transform raw telecom customer data into churn predictions, customer risk segments, revenue-at-risk insights, and actionable business intelligence.

## 🚀 Live Application

**[Open the Customer Churn Analytics Platform](https://customer-churn-analytics-v2-9fvref367tgldad8p6imkt.streamlit.app/)**

> **Live Demo:** The application is deployed using Streamlit Community Cloud and provides an interactive business-facing interface for exploring customer churn analytics.

---

## 📌 Project Overview

Customer churn is a major business challenge for subscription-based companies.

When customers leave, organizations lose recurring revenue and must spend additional resources acquiring new customers. Traditional reporting often identifies churn only after the customer has already left.

This project was developed to move from **reactive churn reporting to proactive churn analytics**.

The platform processes customer data through a structured **Bronze → Silver → Gold Medallion Architecture**, performs business-focused feature engineering, trains a **Random Forest classification model**, evaluates customer churn risk, and converts model outputs into business-oriented insights.

The complete solution combines:

**Data Engineering + Machine Learning + Business Intelligence + Application Development**

---

## 🎯 Business Problem

The objective is not simply to predict whether a customer will churn.

The platform is designed to answer practical business questions such as:

- Which customers are at higher risk of churn?
- What customer characteristics are associated with churn?
- Which contract types have higher churn exposure?
- How do monthly charges relate to churn risk?
- Which customers contribute to potential revenue loss?
- How can customers be segmented into actionable risk groups?
- What business insights can be derived from churn predictions?

The final output transforms raw customer records into information that can support customer-retention analysis.

---

## 🚀 Project Highlights

- End-to-end customer churn analytics pipeline
- Medallion Architecture: **Bronze → Silver → Gold**
- PySpark-based data processing
- Databricks development environment
- Delta Lake data storage
- Data validation and cleaning
- Business-focused feature engineering
- Random Forest churn prediction
- Model evaluation using Accuracy, F1 Score, and ROC-AUC
- Customer risk segmentation
- Feature importance analysis
- Revenue-at-risk analysis
- Executive business dashboard
- Streamlit application
- Streamlit Cloud deployment
- MLOps components including monitoring, testing, and model lifecycle management

---

## 📊 Project Snapshot

| Metric | Result |
|---|---:|
| Customers Analyzed | **7,043** |
| Original Dataset Columns | **21** |
| Architecture | **Bronze → Silver → Gold** |
| Machine Learning Model | **Random Forest** |
| Accuracy | **80.30%** |
| F1 Score | **79.20%** |
| ROC-AUC | **85.85%** |
| Prediction Output | **Customer Churn Risk** |
| Business Output | **Executive Analytics Dashboard** |

---

# 🏗️ Solution Architecture

The project follows a **Medallion Architecture** to progressively transform raw customer data into an ML-ready analytical dataset.

```text
                    RAW CUSTOMER DATA
                           │
                           ▼
                ┌─────────────────────┐
                │   BRONZE LAYER      │
                │ Raw Data Ingestion  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  DATA VALIDATION     │
                │ Schema & Quality     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    SILVER LAYER     │
                │ Clean & Standardize │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ BUSINESS ANALYTICS  │
                │ EDA & Churn Insights│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     GOLD LAYER      │
                │ Feature Engineering │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  MACHINE LEARNING   │
                │     Spark MLlib     │
                │   Random Forest     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ MODEL EVALUATION    │
                │ Accuracy / F1 / AUC │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ CUSTOMER RISK       │
                │ High / Medium / Low │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ BUSINESS INSIGHTS   │
                │ Revenue at Risk     │
                │ Personas            │
                │ Recommendations     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ STREAMLIT APP       │
                │ Interactive Platform│
                └─────────────────────┘

# 📊 Executive Dashboard

The project includes an executive-oriented analytics layer designed to convert churn predictions into understandable business information.

### Dashboard capabilities

- Executive KPI summary
- Customer risk distribution
- Contract-wise churn analysis
- Revenue-at-risk analysis
- Customer segmentation
- Business insights
- Retention-oriented recommendations

The dashboard provides a high-level view of customer churn patterns and risk exposure for business analysis.

---

# 💼 Business Impact

The project connects machine-learning predictions with business-oriented analysis.

### Key business outputs

- Identification of customers with higher churn risk
- Customer risk segmentation
- Revenue-at-risk estimation
- Contract-level churn analysis
- Customer behavior analysis
- Retention-oriented recommendations
- Executive-level KPI reporting

The objective is to move beyond model accuracy and translate analytical results into information that can support customer-retention decisions.

---

# 🖥️ Streamlit Application

The Streamlit application provides the interactive presentation layer of the project.

It allows users to explore the project's analytics through a web-based interface without directly interacting with the underlying notebooks.

### Application Features

- 📊 Executive Overview
- 👥 Customer Risk Segmentation
- 💰 Revenue-at-Risk Analysis
- 📈 Churn & Business Insights
- 🔎 Customer Analytics
- 📁 Dataset Upload
- ✅ Data Quality Validation
- 🤖 AI-assisted business analysis

---

## 🚀 Live Application

### Customer Churn Analytics Platform

**[Open the Live Application →](https://customer-churn-analytics-v2-9fvref367tgldad8p6imkt.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud**.

---

# 🧪 Data Quality Validation

The application includes validation checks before processing uploaded customer datasets.

### Validation checks include

- Required columns
- Duplicate records
- Duplicate customer IDs
- Missing customer IDs
- Null values
- Numeric field validation
- Tenure range validation
- Negative monthly-charge validation
- Valid categorical values
- Churn column presence

This validation layer helps identify data-quality problems before the dataset enters the analytics workflow.

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | PySpark |
| Data Platform | Databricks |
| Storage | Delta Lake |
| Machine Learning | Spark MLlib |
| ML Algorithm | Random Forest |
| Query Language | SQL |
| Data Analysis | Spark DataFrame API |
| Visualization | Databricks Visualizations |
| Application | Streamlit |
| Version Control | Git & GitHub |
| Deployment | Streamlit Community Cloud |

---

# 📁 Repository Structure

```text
customer-churn-analytics-v2/
│
├── app/
│   ├── streamlit_app.py
│   ├── analytics.py
│   ├── ui_sections.py
│   └── ui_rag.py
│
├── notebooks/
│   ├── 01_Data_Ingestion.py
│   ├── 02_Data_Validation.py
│   ├── 03_Data_Cleaning.py
│   ├── 04_Business_Insights_Analysis.py
│   ├── 05_Gold_Layer_&_Feature_Engineering.py
│   ├── 06_Feature_Validation_&_Selection.py
│   ├── 07_Machine_Learning_Pipeline.py
│   ├── 08_Executive_Dashboard_&_Business_Intelligence.py
│   └── 09_Model_Validation_&_Pred_Analysis.py
│
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── mlops/
│   └── validation/
│
├── data/
├── tests/
├── scripts/
│
├── requirements.txt
├── requirements-rag.txt
├── pyproject.toml
├── README.md
└── .gitignore
```
---

## 👨‍💻 Author
Shivek Singh
Data Analyst | Data Science Enthusiast Technologies

Python · PySpark · Databricks · Spark MLlib · Delta Lake · SQL · Streamlit
