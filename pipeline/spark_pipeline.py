"""Lakehouse pipeline in PySpark: raw CSV -> bronze -> silver -> gold.

bronze  raw source extracts stored as-is, plus ingestion metadata
silver  typed, deduplicated, enriched (market prices, approval limits, goods
        receipts) and addresses resolved to a canonical key
gold    supplier feature table, graph edge tables and audit labels

Runs locally or on Databricks:
  local       python pipeline/spark_pipeline.py                  (Parquet files under data/)
  Databricks  notebooks/databricks_pipeline.py                   (Unity Catalog tables)
"""
import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]
RAW_TABLES = ["suppliers", "supplier_directors", "buyers", "products", "invoices",
              "shipments", "tenders", "bids", "audit_labels"]
SERVICE_CATEGORIES = ["Facility Services", "Logistics"]
NEAR_LIMIT_LOW = 0.85      # an invoice in [85%, 100%) of the approval limit is "near limit"
UNSHIPPED_BELOW = 0.5      # less than half the invoiced quantity received


def spark_session() -> SparkSession:
    return (SparkSession.builder.appName("supplier-risk-pipeline")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


class Lake:
    """Reads and writes medallion layers.

    fmt "parquet" or "delta": files under <base>/<layer>/<name>  (local runs)
    fmt "uc": Unity Catalog tables named <schema>.<layer>_<name>  (Databricks)
    Raw CSVs are always read from <base>/raw/.
    """

    def __init__(self, spark, base: str, fmt: str, schema: str = None):
        if fmt == "uc" and not schema:
            raise ValueError("fmt='uc' needs schema, e.g. 'workspace.supplier_risk'")
        self.spark, self.base, self.fmt, self.schema = spark, base.rstrip("/"), fmt, schema

    def write(self, df: DataFrame, layer: str, name: str):
        w = df.write.mode("overwrite")
        if self.fmt == "uc":
            w.option("overwriteSchema", "true").saveAsTable(f"{self.schema}.{layer}_{name}")
        else:
            w.format(self.fmt).save(f"{self.base}/{layer}/{name}")

    def read(self, layer: str, name: str) -> DataFrame:
        if self.fmt == "uc":
            return self.spark.table(f"{self.schema}.{layer}_{name}")
        return self.spark.read.format(self.fmt).load(f"{self.base}/{layer}/{name}")


# ----------------------------------------------------------------------- bronze
def bronze(lake: Lake):
    for t in RAW_TABLES:
        df = (lake.spark.read.option("header", True).csv(f"{lake.base}/raw/{t}.csv")
              .withColumn("_ingested_at", F.current_timestamp())
              .withColumn("_source_file", F.lit(f"{t}.csv")))
        lake.write(df, "bronze", t)
        print(f"[bronze] {t:20s} {df.count():>9,} rows")


# ------------------------------------------------------------ address resolution
def parse_address(col: str) -> list:
    """Split messy address strings into house number, street, pincode and unit.

    '#12 M.G. Rd, Bangalore 560 038' and 'No. 12, Mahatma Gandhi Road, Bengaluru 560038'
    both parse to house '12', street 'MG ROAD', pin '560038'.
    """
    s = F.upper(F.trim(F.col(col)))
    unit = F.regexp_extract(s, r"^(?:UNIT |OFFICE )?(\d+[A-F]),", 1)
    s = F.regexp_replace(s, r"^(?:UNIT |OFFICE )?\d+[A-F],\s*", "")
    house = F.regexp_extract(s, r"^(?:NO\. |#)?(\d+)", 1)
    rest = F.regexp_replace(s, r"^(?:NO\. |#)?\d+,?\s+", "")
    street = F.split(rest, ",").getItem(0)
    street = F.regexp_replace(street, r"\.", "")
    street = F.regexp_replace(street, r"\bMAHATMA GANDHI\b|\bM G\b", "MG")
    street = F.regexp_replace(street, r"\bRD\b", "ROAD")
    street = F.regexp_replace(street, r"\bSEC\b", "SECTOR")
    street = F.regexp_replace(street, r"\bPH\b", "PHASE")
    street = F.trim(F.regexp_replace(street, r"\s+", " "))
    pin = F.concat(F.regexp_extract(s, r"(\d{3})\s?(\d{3})\s*$", 1),
                   F.regexp_extract(s, r"(\d{3})\s?(\d{3})\s*$", 2))
    return [house.alias("house_no"), street.alias("street"), pin.alias("pin"),
            unit.alias("unit")]


def fix_street_typos(df: DataFrame, max_edits: int = 2) -> DataFrame:
    """Fuzzy matching for typos. Within the same pincode, house number and unit,
    map each street spelling to the most common spelling within `max_edits` edits."""
    block = ["pin", "house_no", "unit"]
    freq = df.groupBy(*block, "street").agg(F.count("*").alias("freq"))
    pairs = (freq.withColumnRenamed("street", "s").withColumnRenamed("freq", "f")
             .join(freq.withColumnRenamed("street", "t").withColumnRenamed("freq", "g"), block)
             .where(F.levenshtein("s", "t") <= max_edits))
    best = Window.partitionBy(*block, "s").orderBy(F.desc("g"), F.asc("t"))
    canon = (pairs.withColumn("r", F.row_number().over(best)).where("r = 1")
             .select(*block, F.col("s").alias("street"), F.col("t").alias("street_canonical")))
    return (df.join(canon, block + ["street"], "left")
            .withColumn("street_fixed", F.col("street") != F.col("street_canonical"))
            .withColumn("street", F.col("street_canonical")).drop("street_canonical"))


def resolve_address(df: DataFrame) -> DataFrame:
    """Canonical keys: suppliers in one business park share a building_key but
    differ in unit; shell companies at a virtual office share the full address_key."""
    df = fix_street_typos(df.select("*", *parse_address("address")))
    building = F.concat_ws("|", "house_no", "street", "pin")
    return (df.withColumn("building_key", building)
            .withColumn("address_key", F.concat_ws("|", building, F.col("unit"))))


# ----------------------------------------------------------------------- silver
def silver(lake: Lake):
    b = {t: lake.read("bronze", t).drop("_ingested_at", "_source_file") for t in RAW_TABLES}

    suppliers = (b["suppliers"]
                 .withColumn("registration_date", F.to_date("registration_date"))
                 .dropDuplicates(["supplier_id"]))
    suppliers = resolve_address(suppliers)

    directors = b["supplier_directors"].dropDuplicates(["supplier_id", "director_din"])
    buyers = b["buyers"].withColumn("approval_limit", F.col("approval_limit").cast("double"))
    products = b["products"].withColumn("market_unit_price",
                                        F.col("market_unit_price").cast("double"))

    receipts = (b["shipments"]
                .groupBy("invoice_id")
                .agg(F.sum(F.col("quantity_received").cast("double")).alias("quantity_received"),
                     F.count("*").alias("n_shipments")))

    invoices = (b["invoices"]
                .dropDuplicates(["invoice_id"])
                .withColumn("invoice_date", F.to_date("invoice_date"))
                .withColumn("quantity", F.col("quantity").cast("double"))
                .withColumn("unit_price", F.col("unit_price").cast("double"))
                .withColumn("amount", F.col("amount").cast("double"))
                .join(buyers.select("buyer_id", "approval_limit"), "buyer_id", "left")
                .join(products.select("product_id", "market_unit_price"), "product_id", "left")
                .join(receipts, "invoice_id", "left")
                .fillna({"quantity_received": 0.0, "n_shipments": 0})
                .withColumn("price_ratio", F.col("unit_price") / F.col("market_unit_price"))
                .withColumn("received_ratio", F.col("quantity_received") / F.col("quantity"))
                .withColumn("is_near_limit",
                            (F.col("amount") >= NEAR_LIMIT_LOW * F.col("approval_limit")) &
                            (F.col("amount") < F.col("approval_limit")))
                .withColumn("is_unshipped", F.col("received_ratio") < UNSHIPPED_BELOW))

    tenders = (b["tenders"].withColumn("tender_date", F.to_date("tender_date"))
               .withColumn("estimated_value", F.col("estimated_value").cast("double")))
    bids = (b["bids"]
            .withColumn("bid_amount", F.col("bid_amount").cast("double"))
            .withColumn("is_winner", F.col("is_winner") == "True")
            .join(tenders.select("tender_id", "buyer_id", "estimated_value"), "tender_id", "left")
            .withColumn("bid_to_estimate", F.col("bid_amount") / F.col("estimated_value")))

    labels = b["audit_labels"].withColumn("confirmed_fraud",
                                          F.col("confirmed_fraud").cast("int"))

    for name, df in [("suppliers", suppliers), ("supplier_directors", directors),
                     ("buyers", buyers), ("products", products), ("invoices", invoices),
                     ("tenders", tenders), ("bids", bids), ("audit_labels", labels)]:
        lake.write(df, "silver", name)
        print(f"[silver] {name:20s} {df.count():>9,} rows")


# ------------------------------------------------------------------------- gold
def gold(lake: Lake):
    sup = lake.read("silver", "suppliers")
    inv = lake.read("silver", "invoices")
    bids = lake.read("silver", "bids")
    dirs = lake.read("silver", "supplier_directors")
    labels = lake.read("silver", "audit_labels")
    as_of = inv.agg(F.max("invoice_date")).first()[0]

    # Same-day bursts: several invoices from one supplier to one buyer on one day
    burst = Window.partitionBy("supplier_id", "buyer_id", "invoice_date")
    inv = inv.withColumn("in_burst", F.count("*").over(burst) > 1)

    per_buyer = inv.groupBy("supplier_id", "buyer_id").agg(F.count("*").alias("n"))
    concentration = (per_buyer.groupBy("supplier_id")
                     .agg((F.max("n") / F.sum("n")).alias("top_buyer_share")))

    inv_feats = inv.groupBy("supplier_id").agg(
        F.count("*").alias("n_invoices"),
        F.sum("amount").alias("total_amount"),
        F.countDistinct("buyer_id").alias("n_buyers"),
        F.avg(F.col("is_near_limit").cast("double")).alias("near_limit_share"),
        F.avg(F.col("in_burst").cast("double")).alias("burst_share"),
        F.percentile_approx("price_ratio", 0.5).alias("median_price_ratio"),
        F.percentile_approx("price_ratio", 0.9).alias("p90_price_ratio"),
        F.avg(F.col("is_unshipped").cast("double")).alias("unshipped_share"),
    )

    bid_feats = bids.groupBy("supplier_id").agg(
        F.count("*").alias("n_bids"),
        F.avg(F.col("is_winner").cast("double")).alias("win_rate"),
        F.avg("bid_to_estimate").alias("mean_bid_to_estimate"),
    )

    # Cover-bid signal: how close a supplier's losing bids are to the winning bid
    winning = bids.where("is_winner").select("tender_id", F.col("bid_amount").alias("winning_bid"))
    lost = (bids.where(~F.col("is_winner")).join(winning, "tender_id")
            .withColumn("gap", (F.col("bid_amount") - F.col("winning_bid")) / F.col("winning_bid")))
    gap_feats = lost.groupBy("supplier_id").agg(
        F.avg("gap").alias("mean_lost_bid_gap"),
        F.avg((F.col("gap") <= 0.10).cast("double")).alias("close_loss_share"))

    # Bidding-partner concentration: does a supplier keep bidding with the same rivals?
    co_bid = (bids.alias("a").join(bids.alias("b"), "tender_id")
              .where(F.col("a.supplier_id") != F.col("b.supplier_id"))
              .groupBy(F.col("a.supplier_id").alias("supplier_id"),
                       F.col("b.supplier_id").alias("partner"))
              .agg(F.count("*").alias("n_together")))
    partner_feats = (co_bid.groupBy("supplier_id")
                     .agg(F.max("n_together").alias("max_partner_tenders"),
                          F.count("*").alias("n_bid_partners"))
                     .join(bid_feats.select("supplier_id", "n_bids"), "supplier_id")
                     .withColumn("top_partner_share", F.col("max_partner_tenders") / F.col("n_bids"))
                     .select("supplier_id", "n_bid_partners", "top_partner_share"))

    # Simple link counts (how many OTHER suppliers share each attribute)
    def shared_count(df, key, name):
        w = Window.partitionBy(key)
        return (df.withColumn("k", F.count("*").over(w) - 1)
                .groupBy("supplier_id").agg(F.max("k").alias(name)))

    n_dirs = dirs.groupBy("supplier_id").agg(F.count("*").alias("n_directors"))
    links = (shared_count(sup, "bank_account", "shared_bank_count")
             .join(shared_count(sup, "address_key", "shared_address_count"), "supplier_id")
             .join(shared_count(sup, "building_key", "shared_building_count"), "supplier_id")
             .join(shared_count(dirs, "director_din", "shared_director_count"),
                   "supplier_id", "left"))

    features = (sup.select("supplier_id", "category", "registration_date")
                .withColumn("supplier_age_days", F.datediff(F.lit(as_of), "registration_date"))
                .withColumn("is_service", F.col("category").isin(SERVICE_CATEGORIES).cast("int"))
                .join(inv_feats, "supplier_id", "left")
                .join(concentration, "supplier_id", "left")
                .join(bid_feats, "supplier_id", "left")
                .join(gap_feats, "supplier_id", "left")
                .join(partner_feats, "supplier_id", "left")
                .join(n_dirs, "supplier_id", "left")
                .join(links, "supplier_id", "left")
                .join(labels, "supplier_id", "left")
                .fillna(0))

    # Graph edge tables
    edges = {
        "edges_supplier_bank": sup.select("supplier_id", "bank_account"),
        "edges_supplier_address": sup.select("supplier_id", "address_key"),
        "edges_supplier_director": dirs.select("supplier_id", "director_din"),
        "edges_supplier_buyer": inv.groupBy("supplier_id", "buyer_id")
                                   .agg(F.count("*").alias("n_invoices"),
                                        F.sum("amount").alias("amount")),
        "edges_co_bid": (bids.alias("a")
                         .join(bids.alias("b"), "tender_id")
                         .where(F.col("a.supplier_id") < F.col("b.supplier_id"))
                         .groupBy(F.col("a.supplier_id").alias("supplier_a"),
                                  F.col("b.supplier_id").alias("supplier_b"))
                         .agg(F.count("*").alias("n_tenders_together"))),
    }

    lake.write(features, "gold", "supplier_features")
    print(f"[gold] {'supplier_features':24s} {features.count():>7,} rows, "
          f"{len(features.columns)} columns")
    for name, df in edges.items():
        lake.write(df, "gold", name)
        print(f"[gold] {name:24s} {df.count():>7,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data"))
    ap.add_argument("--format", default="parquet", choices=["parquet", "delta"])
    ap.add_argument("--layers", default="bronze,silver,gold")
    a = ap.parse_args()
    spark = spark_session()
    spark.sparkContext.setLogLevel("WARN")
    lake = Lake(spark, a.base, a.format)
    steps = {"bronze": bronze, "silver": silver, "gold": gold}
    for layer in a.layers.split(","):
        steps[layer](lake)
    spark.stop()
