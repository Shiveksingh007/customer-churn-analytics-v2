# Customer Churn Prediction & Analytics Platform
## Project Structure

This project follows a modular structure inspired by real-world Data Engineering and Machine Learning projects. The repository is organized to separate notebooks, datasets, reusable source code, reports, dashboards, and project assets.

---

## Repository Structure

```text
Customer-Churn-Prediction-Analytics-Platform/
│
├── architecture/
│   └── Project architecture diagrams
│
├── dashboard/
│   └── Dashboard screenshots and business visualizations
│
├── data/
│   ├── raw/
│   │   └── Original Telco Customer Churn dataset
│   │
│   └── processed/
│       └── Optional processed sample datasets
│
├── models/
│   ├── Feature Importance
│   └── Model Information
│
├── notebooks/
│   ├── 01_Data_Ingestion.ipynb
│   ├── 02_Data_Validation.ipynb
│   ├── 03_Data_Cleaning.ipynb
│   ├── 04_Business_Insights_Analysis.ipynb
│   ├── 05_Gold_Layer_&_Feature_Engineering.ipynb
│   ├── 06_Feature_Validation_&_Selection.ipynb
│   ├── 07_Machine_Learning_Pipeline.ipynb
│   ├── 08_Executive_Dashboard_&_Business_Intelligence.ipynb
│   └── 09_Model_Validation_&_Prediction_Analysis.ipynb
│
├── presentation/
│   └── Final project presentation
│
├── reports/
│   ├── Project Report
│   └── Model Validation Report
│
├── screenshots/
│   └── Project screenshots used in documentation
│
├── src/
│   ├── config/
│   ├── features/
│   ├── logging/
│   ├── models/
│   ├── preprocessing/
│   ├── utils/
│   └── validation/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── Project_Structure.md
```

---

# Notebook Workflow

| Notebook | Description |
|-----------|-------------|
| Notebook 01 | Data Ingestion (Bronze Layer) |
| Notebook 02 | Data Validation |
| Notebook 03 | Data Cleaning (Silver Layer) |
| Notebook 04 | Business Insights & Exploratory Analysis |
| Notebook 05 | Gold Layer & Feature Engineering |
| Notebook 06 | Feature Validation & Selection |
| Notebook 07 | Machine Learning Pipeline |
| Notebook 08 | Executive Dashboard & Business Intelligence |
| Notebook 09 | Model Validation & Prediction Analysis |

---

# Source Code Modules

The `src/` directory contains reusable Python modules used throughout the notebooks.

| Folder | Purpose |
|---------|---------|
| config | Project configuration and settings |
| features | Feature engineering utilities |
| logging | Logging utilities |
| models | Model-related helper functions |
| preprocessing | Data preprocessing functions |
| utils | Common utility functions |
| validation | Data validation functions |

---

# Project Architecture

The project follows a Medallion Architecture:

Raw Dataset

↓

Bronze Layer

↓

Silver Layer

↓

Gold Layer

↓

Machine Learning Pipeline

↓

Prediction Report

↓

Executive Dashboard

---

# Tech Stack

- Python
- PySpark
- Databricks
- Spark MLlib
- Delta Lake
- SQL

---

# Project Status

**Current Status:** Completed Development & Validation

Next Phase:

- Documentation
- GitHub Packaging
- Presentation