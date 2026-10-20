# Databricks notebook source
# MAGIC %md
# MAGIC # Supplier risk pipeline on Databricks
# MAGIC Runs the same bronze / silver / gold code as the local version, writing Delta tables.
# MAGIC
# MAGIC **Before running:** create a Unity Catalog volume (for example `workspace.default.procurement`)
# MAGIC and upload the nine CSV files from `data/raw/` into a folder named `raw` inside it.

# COMMAND ----------

import sys

REPO = "/Workspace/Users/<your-email>/supplier-risk-gnn"   # path of the Git folder
BASE = "/Volumes/workspace/default/procurement"             # volume containing raw/

sys.path.append(f"{REPO}/pipeline")
from spark_pipeline import Lake, bronze, silver, gold

lake = Lake(spark, BASE, "delta")

# COMMAND ----------

bronze(lake)

# COMMAND ----------

silver(lake)

# COMMAND ----------

gold(lake)

# COMMAND ----------

display(lake.read("gold", "supplier_features").orderBy("confirmed_fraud", ascending=False).limit(20))
