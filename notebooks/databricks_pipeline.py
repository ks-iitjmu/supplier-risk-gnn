# Databricks notebook source
# MAGIC %md
# MAGIC # Supplier risk lakehouse pipeline on Databricks
# MAGIC Runs the same bronze / silver / gold code as the local version and saves every layer
# MAGIC as a Unity Catalog table in `workspace.supplier_risk`.
# MAGIC
# MAGIC 1. Run cell 1 to create the schema and the landing volume.
# MAGIC 2. Upload the nine CSVs from `data/raw/` into a folder named `raw` in the volume
# MAGIC    (Catalog > workspace > supplier_risk > landing).
# MAGIC 3. Run the remaining cells.

# COMMAND ----------

# Cell 1: schema and landing volume
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.supplier_risk")
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.supplier_risk.landing")
print("Upload the CSVs to /Volumes/workspace/supplier_risk/landing/raw/")

# COMMAND ----------

# Cell 2: check the upload
files = sorted(f.name for f in dbutils.fs.ls("/Volumes/workspace/supplier_risk/landing/raw/"))
print(len(files), "files:", files)
assert len([f for f in files if f.endswith(".csv")]) == 9, "Expected 9 CSV files"

# COMMAND ----------

# Cell 3: import the pipeline from this Git folder
import os
import sys

repo_root = os.path.dirname(os.getcwd())   # the notebook runs from <repo>/notebooks
sys.path.append(os.path.join(repo_root, "pipeline"))
from spark_pipeline import Lake, bronze, silver, gold

lake = Lake(spark, "/Volumes/workspace/supplier_risk/landing", "uc",
            schema="workspace.supplier_risk")

# COMMAND ----------

bronze(lake)

# COMMAND ----------

silver(lake)

# COMMAND ----------

gold(lake)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Tables created by the pipeline
# MAGIC SHOW TABLES IN workspace.supplier_risk

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Gold layer: suppliers with the strongest risk signals
# MAGIC SELECT supplier_id, category, n_invoices, near_limit_share, p90_price_ratio,
# MAGIC        unshipped_share, top_partner_share, shared_bank_count, shared_director_count,
# MAGIC        confirmed_fraud
# MAGIC FROM workspace.supplier_risk.gold_supplier_features
# MAGIC ORDER BY confirmed_fraud DESC, shared_bank_count DESC, near_limit_share DESC
# MAGIC LIMIT 20
