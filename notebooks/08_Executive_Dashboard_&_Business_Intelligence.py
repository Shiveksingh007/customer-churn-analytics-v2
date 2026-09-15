# Databricks notebook source
# MAGIC %md
# MAGIC ### Telecom Customer Intelligence Platform
# MAGIC
# MAGIC ### Notebook 08 — Executive Dashboard & Business Intelligence
# MAGIC
# MAGIC #### Objective
# MAGIC
# MAGIC Load the prediction report generated in Notebook 07 and create
# MAGIC an executive dashboard for customer churn analysis.
# MAGIC
# MAGIC Input:
# MAGIC - workspace.default.customer_churn_predictions
# MAGIC
# MAGIC Output:
# MAGIC - Executive KPIs
# MAGIC - Risk Distribution
# MAGIC - High Risk Customers
# MAGIC - Business Insights

# COMMAND ----------

from pyspark.sql import functions as F

print("=" * 60)
print("NOTEBOOK 08 STARTED")
print("=" * 60)

# COMMAND ----------

prediction_df = spark.table(
    "workspace.default.customer_churn_predictions"
)

print("=" * 60)
print("PREDICTION REPORT LOADED")
print("=" * 60)

print(f"Rows    : {prediction_df.count()}")
print(f"Columns : {len(prediction_df.columns)}")

display(prediction_df.limit(10))

# COMMAND ----------

total_customers = prediction_df.count()

print("=" * 60)
print("TOTAL CUSTOMERS")
print("=" * 60)
print(total_customers)

# COMMAND ----------

risk_summary = (

    prediction_df

    .groupBy("Risk_Level")

    .count()

    .orderBy("Risk_Level")

)

display(risk_summary)

# COMMAND ----------

prediction_df.select(

    F.round(

        F.avg("Churn_Percentage"),

        2

    ).alias("Average_Churn_Probability (%)")

).show()

# COMMAND ----------

high_risk = prediction_df.filter(
    F.col("Risk_Level") == "High"
)

print("=" * 60)
print("HIGH RISK CUSTOMERS")
print("=" * 60)

print(high_risk.count())

display(high_risk)

# COMMAND ----------

high = prediction_df.filter(F.col("Risk_Level")=="High").count()

medium = prediction_df.filter(F.col("Risk_Level")=="Medium").count()

low = prediction_df.filter(F.col("Risk_Level")=="Low").count()

print("=" * 60)
print("EXECUTIVE KPI SUMMARY")
print("=" * 60)

print(f"Total Customers        : {total_customers}")
print(f"High Risk Customers    : {high}")
print(f"Medium Risk Customers  : {medium}")
print(f"Low Risk Customers     : {low}")

# COMMAND ----------

display(
    prediction_df
    .groupBy("Risk_Level")
    .count()
    .orderBy("Risk_Level")
)

# COMMAND ----------

contract_risk = (
    prediction_df
    .groupBy("Contract", "Risk_Level")
    .count()
    .orderBy("Contract", "Risk_Level")
)

display(contract_risk)

# COMMAND ----------

monthly_charge_analysis = (
    prediction_df
    .groupBy("Risk_Level")
    .agg(
        F.round(
            F.avg("MonthlyCharges"),
            2
        ).alias("Average_Monthly_Charges")
    )
)

display(monthly_charge_analysis)

# COMMAND ----------

tenure_analysis = (
    prediction_df
    .groupBy("Risk_Level")
    .agg(
        F.round(
            F.avg("tenure"),
            2
        ).alias("Average_Tenure")
    )
)

display(tenure_analysis)

# COMMAND ----------

display(
    prediction_df
    .filter(F.col("Risk_Level") == "High")
    .orderBy(F.col("Churn_Percentage").desc())
    .limit(20)
)

# COMMAND ----------

print("=" * 60)
print("BUSINESS INSIGHTS SUMMARY")
print("=" * 60)

print(f"Total Customers          : {total_customers}")
print(f"High Risk Customers      : {high}")
print(f"Medium Risk Customers    : {medium}")
print(f"Low Risk Customers       : {low}")

print("\nDashboard Status : READY")

# COMMAND ----------

# MAGIC %md
# MAGIC # Executive Recommendations
# MAGIC
# MAGIC ## High Risk Customers
# MAGIC - Contact immediately through the retention team.
# MAGIC - Offer contract upgrade discounts.
# MAGIC - Provide personalized retention offers.
# MAGIC
# MAGIC ## Medium Risk Customers
# MAGIC - Promote bundled services.
# MAGIC - Offer Tech Support and Online Security packages.
# MAGIC - Run targeted email/SMS campaigns.
# MAGIC
# MAGIC ## Low Risk Customers
# MAGIC - Enroll in loyalty and rewards programs.
# MAGIC - Continue regular engagement campaigns.
# MAGIC - Monitor periodically for changes in behavior.

# COMMAND ----------

print("=" * 60)
print("EXECUTIVE DASHBOARD SUMMARY")
print("=" * 60)

print(f"Total Customers             : {total_customers}")
print(f"High Risk Customers         : {high}")
print(f"Medium Risk Customers       : {medium}")
print(f"Low Risk Customers          : {low}")

print("\nKey Findings")
print("- High-risk customers require immediate retention efforts.")
print("- Medium-risk customers are suitable for targeted marketing.")
print("- Low-risk customers should be retained through loyalty programs.")

# COMMAND ----------

assert total_customers == prediction_df.count()
assert high + medium + low == total_customers

print("=" * 60)
print("NOTEBOOK VALIDATION PASSED")
print("=" * 60)

# COMMAND ----------

print("=" * 60)
print("NOTEBOOK 08 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Prediction Report      : SUCCESS")
print("Executive Dashboard    : SUCCESS")
print("Business Insights      : SUCCESS")
print("Project Status         : READY FOR FINAL DASHBOARD / DOCUMENTATION")

# COMMAND ----------

# MAGIC %md
# MAGIC # Dashboard Visualizations
# MAGIC
# MAGIC The following visualizations provide business-friendly insights for
# MAGIC executive decision-making.

# COMMAND ----------

risk_summary = (
    prediction_df
    .groupBy("Risk_Level")
    .count()
)

display(risk_summary)

# COMMAND ----------

contract_dashboard = (
    prediction_df
    .groupBy("Contract", "Risk_Level")
    .count()
)

display(contract_dashboard)

# COMMAND ----------

monthly_dashboard = (
    prediction_df
    .groupBy("Risk_Level")
    .agg(
        F.round(
            F.avg("MonthlyCharges"),
            2
        ).alias("Average_Monthly_Charges")
    )
)

display(monthly_dashboard)

# COMMAND ----------

tenure_dashboard = (
    prediction_df
    .groupBy("Risk_Level")
    .agg(
        F.round(
            F.avg("tenure"),
            2
        ).alias("Average_Tenure")
    )
)

display(tenure_dashboard)

# COMMAND ----------

top_high_risk = (
    prediction_df
    .filter(F.col("Risk_Level") == "High")
    .orderBy(F.col("Churn_Percentage").desc())
)

display(top_high_risk)

# COMMAND ----------

high_risk_revenue = (
    prediction_df
    .filter(F.col("Risk_Level") == "High")
    .agg(
        F.round(
            F.sum("MonthlyCharges"),
            2
        ).alias("Monthly_Revenue_At_Risk")
    )
)

display(high_risk_revenue)

# COMMAND ----------

monthly_revenue = (
    prediction_df
    .filter(F.col("Risk_Level") == "High")
    .agg(F.sum("MonthlyCharges"))
    .first()[0]
)

annual_revenue = monthly_revenue * 12

print("=" * 60)
print("REVENUE IMPACT")
print("=" * 60)

print(f"Monthly Revenue At Risk : ₹{monthly_revenue:.2f}")
print(f"Annual Revenue At Risk  : ₹{annual_revenue:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Revenue Impact Analysis
# MAGIC
# MAGIC ## Business Value
# MAGIC
# MAGIC The churn prediction model identifies customers who are most likely to leave.
# MAGIC
# MAGIC By proactively retaining High Risk customers, the company can reduce revenue loss and improve customer lifetime value.
# MAGIC
# MAGIC This analysis helps prioritize retention campaigns based on potential financial impact.
# MAGIC

# COMMAND ----------

persona_summary = (
    prediction_df
    .groupBy("Risk_Level", "Contract")
    .count()
    .orderBy("Risk_Level", F.desc("count"))
)

display(persona_summary)

# COMMAND ----------

customer_profile = (
    prediction_df
    .groupBy("Risk_Level")
    .agg(
        F.round(F.avg("MonthlyCharges"), 2).alias("Avg_Monthly_Charges"),
        F.round(F.avg("tenure"), 2).alias("Avg_Tenure"),
        F.count("*").alias("Customers")
    )
)

display(customer_profile)

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Personas
# MAGIC
# MAGIC ## 🔴 High Risk Customers
# MAGIC
# MAGIC Typical Characteristics
# MAGIC
# MAGIC - Mostly Month-to-Month contracts
# MAGIC - Higher Monthly Charges
# MAGIC - Lower Average Tenure
# MAGIC - Immediate retention action recommended
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟠 Medium Risk Customers
# MAGIC
# MAGIC Typical Characteristics
# MAGIC
# MAGIC - Moderate Monthly Charges
# MAGIC - Medium Tenure
# MAGIC - Suitable for promotional campaigns
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟢 Low Risk Customers
# MAGIC
# MAGIC Typical Characteristics
# MAGIC
# MAGIC - Longer Customer Tenure
# MAGIC - Lower Churn Probability
# MAGIC - Good candidates for loyalty programs