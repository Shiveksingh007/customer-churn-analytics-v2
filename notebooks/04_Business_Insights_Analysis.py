# Databricks notebook source
# MAGIC %md
# MAGIC # Telecom Customer Intelligence Platform
# MAGIC
# MAGIC ## Notebook 04 – Business Insights Analysis
# MAGIC
# MAGIC ### Objective
# MAGIC
# MAGIC Analyze customer behavior using the Silver dataset and generate actionable business insights.
# MAGIC
# MAGIC ### Input
# MAGIC
# MAGIC Silver Layer
# MAGIC
# MAGIC ### Output
# MAGIC
# MAGIC Business KPIs, Customer Insights and Executive Recommendations

# COMMAND ----------

# Initialization

import sys

PROJECT_ROOT = "/Workspace/Users/thakurshivek777@gmail.com/Customer-Churn-Platform"
SRC_PATH = f"{PROJECT_ROOT}/src"

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print(SRC_PATH)
print(sys.path)

# COMMAND ----------

from pyspark.sql import functions as F

from config.settings import SILVER_TABLE
from features.churn_analysis import analyze_churn

# COMMAND ----------

silver_df = spark.read.table(SILVER_TABLE)

display(silver_df.limit(5))

# COMMAND ----------

total_customers = silver_df.count()

churned_customers = silver_df.filter(
    F.col("Churn") == "Yes"
).count()

active_customers = silver_df.filter(
    F.col("Churn") == "No"
).count()

churn_rate = (
    churned_customers / total_customers
) * 100

avg_monthly = silver_df.select(
    F.avg("MonthlyCharges")
).first()[0]

avg_tenure = silver_df.select(
    F.avg("tenure")
).first()[0]

# COMMAND ----------

print("=" * 70)
print("EXECUTIVE KPI DASHBOARD")
print("=" * 70)

print(f"Total Customers          : {total_customers}")
print(f"Active Customers         : {active_customers}")
print(f"Churned Customers        : {churned_customers}")
print(f"Churn Rate (%)           : {churn_rate:.2f}")
print(f"Average Monthly Charges  : ${avg_monthly:.2f}")
print(f"Average Tenure (Months)  : {avg_tenure:.2f}")

# COMMAND ----------

# Analyze Gender Distribution
gender_summary = (
    silver_df
    .groupBy("gender", "Churn")
    .count()
    .orderBy("gender", "Churn")
)

display(gender_summary)

# COMMAND ----------

# Calculate Churn Rate by Gender
gender_churn = (
    silver_df
    .groupBy("gender")
    .agg(
        F.count("*").alias("Total_Customers"),
        F.sum(
            F.when(F.col("Churn") == "Yes", 1).otherwise(0)
        ).alias("Churned_Customers")
    )
)

gender_churn = (
    gender_churn
    .withColumn(
        "Churn_Rate",
        F.round(
            (F.col("Churned_Customers") / F.col("Total_Customers")) * 100,
            2
        )
    )
)

display(gender_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC **Question**
# MAGIC
# MAGIC Does gender influence customer churn?
# MAGIC
# MAGIC **Observation**
# MAGIC
# MAGIC (We'll write this after seeing the output.)
# MAGIC
# MAGIC **Recommendation**
# MAGIC
# MAGIC (We'll decide based on evidence.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Question
# MAGIC
# MAGIC Do senior citizens churn more than non-senior customers?

# COMMAND ----------

senior_churn = (
    silver_df
    .groupBy("SeniorCitizen")
    .agg(
        F.count("*").alias("Total_Customers"),
        F.sum(
            F.when(F.col("Churn") == "Yes", 1).otherwise(0)
        ).alias("Churned_Customers")
    )
)

# COMMAND ----------

senior_churn = (
    senior_churn
    .withColumn(
        "Churn_Rate",
        F.round(
            (F.col("Churned_Customers") / F.col("Total_Customers")) * 100,
            2
        )
    )
)

display(senior_churn)

# COMMAND ----------

display(senior_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC Do senior citizens churn more than non-senior customers?
# MAGIC
# MAGIC ### Evidence
# MAGIC
# MAGIC - Non-Senior Churn Rate : **23.61%**
# MAGIC - Senior Citizen Churn Rate : **41.68%**
# MAGIC
# MAGIC Difference : **18.07 percentage points**
# MAGIC
# MAGIC ### Business Insight
# MAGIC
# MAGIC Senior citizens are significantly more likely to churn than non-senior customers. This indicates that age group is a strong factor associated with customer churn.
# MAGIC
# MAGIC ### Business Impact
# MAGIC
# MAGIC 🔴 High
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Develop targeted retention strategies for senior customers, such as simplified support channels, loyalty benefits, personalized service plans, and proactive customer outreach.

# COMMAND ----------

contract_churn = analyze_churn(silver_df, "Contract")
display(contract_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC Does contract type influence customer churn?
# MAGIC
# MAGIC ### Evidence
# MAGIC
# MAGIC | Contract | Churn Rate |
# MAGIC |----------|-----------:|
# MAGIC | Month-to-month | 42.71% |
# MAGIC | One year | 11.27% |
# MAGIC | Two year | 2.83% |
# MAGIC
# MAGIC ### Business Insight
# MAGIC
# MAGIC Contract type is one of the strongest indicators of customer churn. Customers on month-to-month contracts churn at a significantly higher rate than customers on long-term contracts.
# MAGIC
# MAGIC ### Business Impact
# MAGIC
# MAGIC 🔴 Very High
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Increase customer retention by encouraging month-to-month customers to migrate to annual or two-year contracts through discounts, loyalty rewards, bundled services, or exclusive offers.

# COMMAND ----------

# MAGIC %md
# MAGIC # Internet Service Analysis
# MAGIC
# MAGIC ## Business Question
# MAGIC
# MAGIC Does the type of internet service influence customer churn?

# COMMAND ----------

internet_churn = analyze_churn(
    silver_df,
    "InternetService"
)

display(internet_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC Does Internet Service influence customer churn?
# MAGIC
# MAGIC ### Evidence
# MAGIC
# MAGIC | Internet Service | Churn Rate |
# MAGIC |-----------------|-----------:|
# MAGIC | Fiber optic | 41.89% |
# MAGIC | DSL | 18.96% |
# MAGIC | No Internet | 7.40% |
# MAGIC
# MAGIC ### Business Insight
# MAGIC
# MAGIC Customers using Fiber Optic internet experience the highest churn rate, more than double that of DSL users.
# MAGIC
# MAGIC ### Business Impact
# MAGIC
# MAGIC 🔥 Critical
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Investigate Fiber Optic service quality, pricing, customer support, and network stability. Prioritize retention campaigns for Fiber Optic customers.

# COMMAND ----------

# MAGIC %md
# MAGIC # Online Security Analysis
# MAGIC
# MAGIC ## Business Question
# MAGIC
# MAGIC Does having Online Security reduce customer churn?

# COMMAND ----------

security_churn = analyze_churn(
    silver_df,
    "OnlineSecurity"
)

display(security_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC Does Online Security reduce churn?
# MAGIC
# MAGIC ### Evidence
# MAGIC
# MAGIC Without Online Security : 41.77%
# MAGIC
# MAGIC With Online Security : 14.61%
# MAGIC
# MAGIC ### Business Insight
# MAGIC
# MAGIC Customers without Online Security churn nearly three times more often than customers who subscribe to the service.
# MAGIC
# MAGIC ### Business Impact
# MAGIC
# MAGIC 🔥 Critical
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Bundle Online Security into premium internet plans or provide discounted introductory offers.

# COMMAND ----------

# MAGIC %md
# MAGIC # Tech Support Analysis
# MAGIC
# MAGIC ## Business Question
# MAGIC
# MAGIC Does Tech Support influence customer retention?

# COMMAND ----------

tech_support_churn = analyze_churn(
    silver_df,
    "TechSupport"
)

display(tech_support_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC Does Tech Support improve customer retention?
# MAGIC
# MAGIC ### Evidence
# MAGIC
# MAGIC Without Tech Support : 41.64%
# MAGIC
# MAGIC With Tech Support : 15.17%
# MAGIC
# MAGIC ### Business Insight
# MAGIC
# MAGIC Customers who subscribe to Tech Support are significantly more likely to remain with the company.
# MAGIC
# MAGIC ### Business Impact
# MAGIC
# MAGIC 🔥 Critical
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Promote Tech Support bundles and improve awareness among new customers.

# COMMAND ----------

# MAGIC %md
# MAGIC # Device Protection Analysis
# MAGIC
# MAGIC ## Business Question
# MAGIC
# MAGIC Does Device Protection reduce churn?

# COMMAND ----------

device_churn = analyze_churn(
    silver_df,
    "DeviceProtection"
)

display(device_churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC Does Device Protection reduce churn?
# MAGIC
# MAGIC ### Evidence
# MAGIC
# MAGIC Without Device Protection : 39.13%
# MAGIC
# MAGIC With Device Protection : 22.50%
# MAGIC
# MAGIC ### Business Insight
# MAGIC
# MAGIC Customers without Device Protection show considerably higher churn than customers who subscribe to the service.
# MAGIC
# MAGIC ### Business Impact
# MAGIC
# MAGIC 🔴 High
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Offer Device Protection during onboarding and contract renewal.

# COMMAND ----------

# MAGIC %md
# MAGIC # Executive Business Findings

# COMMAND ----------

# MAGIC %md
# MAGIC | Feature | Highest Risk Category | Churn Rate | Priority | Recommendation |
# MAGIC |---------|-----------------------|-----------:|----------|----------------|
# MAGIC | Gender | Female | 26.92% | 🟢 Low | No Action |
# MAGIC | Senior Citizen | Yes | 41.68% | 🔴 High | Target Retention Campaign |
# MAGIC | Contract | Month-to-month | 42.71% | 🔥 Critical | Convert to Long-Term Contracts |
# MAGIC | Internet Service | Fiber Optic | 41.89% | 🔥 Critical | Improve Fiber Service Quality |
# MAGIC | Online Security | No | 41.77% | 🔥 Critical | Bundle Security Services |
# MAGIC | Tech Support | No | 41.64% | 🔥 Critical | Promote Tech Support Plans |
# MAGIC | Device Protection | No | 39.13% | 🔴 High | Offer Protection Bundles |

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Risk Profiling
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Identify customer profiles that have the highest probability of churn by combining multiple business attributes.

# COMMAND ----------

contract_internet = (
    silver_df
    .groupBy("Contract", "InternetService")
    .agg(
        F.count("*").alias("Customers"),
        F.sum(
            F.when(F.col("Churn") == "Yes", 1).otherwise(0)
        ).alias("Churned")
    )
    .withColumn(
        "Churn_Rate",
        F.round(
            (F.col("Churned") / F.col("Customers")) * 100,
            2
        )
    )
    .orderBy(F.desc("Churn_Rate"))
)

display(contract_internet)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC Which combination of Contract and Internet Service has the highest churn?
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC - Month-to-month + Fiber Optic customers have the highest churn rate (54.61%).
# MAGIC - Long-term contracts dramatically reduce churn regardless of internet service.
# MAGIC - Two-year contract customers show extremely low churn.
# MAGIC
# MAGIC ### Business Recommendation
# MAGIC
# MAGIC Prioritize converting Month-to-month Fiber customers into One-year or Two-year contracts using discounts and loyalty offers.

# COMMAND ----------

senior_contract = (
    silver_df
    .groupBy("SeniorCitizen", "Contract")
    .agg(
        F.count("*").alias("Customers"),
        F.sum(
            F.when(F.col("Churn") == "Yes", 1).otherwise(0)
        ).alias("Churned")
    )
    .withColumn(
        "Churn_Rate",
        F.round(
            (F.col("Churned") / F.col("Customers")) * 100,
            2
        )
    )
    .orderBy(F.desc("Churn_Rate"))
)

display(senior_contract)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC How does Contract Type affect Senior Citizens?
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC Senior Citizens with Month-to-month contracts have the highest churn rate (54.65%).
# MAGIC
# MAGIC Long-term contracts significantly reduce churn even for Senior Citizens.
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Create special long-term plans and loyalty discounts specifically for Senior Citizens.

# COMMAND ----------

support_security = (
    silver_df
    .groupBy("TechSupport", "OnlineSecurity")
    .agg(
        F.count("*").alias("Customers"),
        F.sum(
            F.when(F.col("Churn") == "Yes", 1).otherwise(0)
        ).alias("Churned")
    )
    .withColumn(
        "Churn_Rate",
        F.round(
            (F.col("Churned") / F.col("Customers")) * 100,
            2
        )
    )
    .orderBy(F.desc("Churn_Rate"))
)

display(support_security)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insight
# MAGIC
# MAGIC ### Business Question
# MAGIC
# MAGIC How does Contract Type affect Senior Citizens?
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC Senior Citizens with Month-to-month contracts have the highest churn rate (54.65%).
# MAGIC
# MAGIC Long-term contracts significantly reduce churn even for Senior Citizens.
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Create special long-term plans and loyalty discounts specifically for Senior Citizens.

# COMMAND ----------

# MAGIC %md
# MAGIC # Executive Customer Personas
# MAGIC
# MAGIC ## 🔥 Highest Risk Customer
# MAGIC
# MAGIC - Senior Citizen
# MAGIC - Month-to-month Contract
# MAGIC - Fiber Optic Internet
# MAGIC - No Online Security
# MAGIC - No Tech Support
# MAGIC - No Device Protection
# MAGIC
# MAGIC Expected churn risk:
# MAGIC Very High (>50%)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟢 Lowest Risk Customer
# MAGIC
# MAGIC - Two-Year Contract
# MAGIC - DSL or No Internet
# MAGIC - Online Security = Yes
# MAGIC - Tech Support = Yes
# MAGIC
# MAGIC Expected churn risk:
# MAGIC Very Low (<10%)