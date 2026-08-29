import sys
import re

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    DecimalType,
    TimestampType
)

from delta.tables import DeltaTable


# =========================================================
# Glue initialization
# =========================================================

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME"]
)

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(args["JOB_NAME"], args)

spark.conf.set(
    "spark.sql.session.timeZone",
    "UTC"
)


# =========================================================
# Configuration
# =========================================================

BRONZE_PATH = (
    "s3://nw-globalpartners-project/"
    "bronze/order_item_options/"
)

SILVER_PATH = (
    "s3://nw-globalpartners-project/"
    "silver/order_item_options/"
)

LOGICAL_KEY = [
    "ORDER_ID",
    "LINEITEM_ID",
    "OPTION_GROUP_NAME",
    "OPTION_NAME"
]

EXPECTED_BUSINESS_COLUMNS = [
    "ORDER_ID",
    "LINEITEM_ID",
    "OPTION_GROUP_NAME",
    "OPTION_NAME",
    "OPTION_PRICE",
    "OPTION_QUANTITY"
]

EXPECTED_METADATA_COLUMNS = [
    "_INGESTED_AT",
    "_SOURCE_TABLE",
    "_LOAD_TYPE"
]

EXPECTED_COLUMNS = (
    EXPECTED_BUSINESS_COLUMNS
    + EXPECTED_METADATA_COLUMNS
)


# =========================================================
# Helper: UPPER_SNAKE_CASE
# =========================================================

def upper_snake_case(column_name):

    has_leading_underscore = (
        column_name.startswith("_")
    )

    cleaned = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        column_name
    )

    cleaned = (
        cleaned
        .strip("_")
        .upper()
    )

    if has_leading_underscore:
        cleaned = "_" + cleaned

    return cleaned


# =========================================================
# Read Bronze
# =========================================================

print("=" * 80)
print("Reading ORDER_ITEM_OPTIONS Bronze Delta table")
print(f"Bronze path: {BRONZE_PATH}")
print("=" * 80)

bronze_df = (
    spark.read
    .format("delta")
    .load(BRONZE_PATH)
)

bronze_count = bronze_df.count()

print(
    f"Bronze ORDER_ITEM_OPTIONS row count: "
    f"{bronze_count}"
)

print("Bronze schema:")
bronze_df.printSchema()


# =========================================================
# Normalize column names
# =========================================================

normalized_names = [
    upper_snake_case(column_name)
    for column_name in bronze_df.columns
]

if len(normalized_names) != len(set(normalized_names)):

    raise RuntimeError(
        "Column normalization created duplicate "
        "column names."
    )

for old_name, new_name in zip(
    bronze_df.columns,
    normalized_names
):

    if old_name != new_name:

        bronze_df = (
            bronze_df
            .withColumnRenamed(
                old_name,
                new_name
            )
        )


# =========================================================
# Validate schema
# =========================================================

actual_columns = set(bronze_df.columns)
expected_columns = set(EXPECTED_COLUMNS)

missing_columns = (
    expected_columns
    - actual_columns
)

unexpected_columns = (
    actual_columns
    - expected_columns
)

if missing_columns:

    raise RuntimeError(
        "ORDER_ITEM_OPTIONS is missing "
        f"expected columns: "
        f"{sorted(missing_columns)}"
    )

if unexpected_columns:

    raise RuntimeError(
        "ORDER_ITEM_OPTIONS contains unexpected "
        f"columns: {sorted(unexpected_columns)}"
    )


# =========================================================
# Preserve values needed for cast validation
# =========================================================

working_df = (
    bronze_df

    .withColumn(
        "__ORIGINAL_OPTION_PRICE",
        F.col("OPTION_PRICE")
    )

    .withColumn(
        "__ORIGINAL_OPTION_QUANTITY",
        F.col("OPTION_QUANTITY")
    )

    .withColumn(
        "__ORIGINAL_INGESTED_AT",
        F.col("_INGESTED_AT")
    )
)


# =========================================================
# Enforce Silver types
# =========================================================

silver_source_df = (
    working_df

    .withColumn(
        "ORDER_ID",
        F.col("ORDER_ID").cast(
            StringType()
        )
    )

    .withColumn(
        "LINEITEM_ID",
        F.col("LINEITEM_ID").cast(
            StringType()
        )
    )

    .withColumn(
        "OPTION_GROUP_NAME",
        F.col("OPTION_GROUP_NAME").cast(
            StringType()
        )
    )

    .withColumn(
        "OPTION_NAME",
        F.col("OPTION_NAME").cast(
            StringType()
        )
    )

    .withColumn(
        "OPTION_PRICE",
        F.col("OPTION_PRICE").cast(
            DecimalType(18, 0)
        )
    )

    .withColumn(
        "OPTION_QUANTITY",
        F.col("OPTION_QUANTITY").cast(
            DecimalType(18, 0)
        )
    )

    .withColumn(
        "_INGESTED_AT",
        F.col("_INGESTED_AT").cast(
            TimestampType()
        )
    )

    .withColumn(
        "_SOURCE_TABLE",
        F.col("_SOURCE_TABLE").cast(
            StringType()
        )
    )

    .withColumn(
        "_LOAD_TYPE",
        F.col("_LOAD_TYPE").cast(
            StringType()
        )
    )
)


# =========================================================
# Detect failed casts
# =========================================================

cast_failure_df = (
    silver_source_df
    .filter(

        (
            F.col(
                "__ORIGINAL_OPTION_PRICE"
            ).isNotNull()
            &
            F.col(
                "OPTION_PRICE"
            ).isNull()
        )

        |

        (
            F.col(
                "__ORIGINAL_OPTION_QUANTITY"
            ).isNotNull()
            &
            F.col(
                "OPTION_QUANTITY"
            ).isNull()
        )

        |

        (
            F.col(
                "__ORIGINAL_INGESTED_AT"
            ).isNotNull()
            &
            F.col(
                "_INGESTED_AT"
            ).isNull()
        )
    )
)

if (
    cast_failure_df
    .limit(1)
    .count()
    > 0
):

    print(
        "ERROR: values failed Silver type "
        "conversion."
    )

    cast_failure_df.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "ORDER_ITEM_OPTIONS type "
        "enforcement failed."
    )


silver_source_df = (
    silver_source_df
    .drop(
        "__ORIGINAL_OPTION_PRICE",
        "__ORIGINAL_OPTION_QUANTITY",
        "__ORIGINAL_INGESTED_AT"
    )
)


# =========================================================
# Remove exact duplicate rows
# =========================================================

before_dedupe_count = (
    silver_source_df.count()
)

silver_source_df = (
    silver_source_df
    .dropDuplicates()
)

after_dedupe_count = (
    silver_source_df.count()
)

duplicates_removed = (
    before_dedupe_count
    - after_dedupe_count
)

print(
    f"Exact duplicate rows removed: "
    f"{duplicates_removed}"
)


# =========================================================
# Validate logical-key NULLs
# =========================================================

null_key_condition = (
    F.col("ORDER_ID").isNull()
    |
    F.col("LINEITEM_ID").isNull()
    |
    F.col("OPTION_GROUP_NAME").isNull()
    |
    F.col("OPTION_NAME").isNull()
)

null_key_df = (
    silver_source_df
    .filter(null_key_condition)
)

if (
    null_key_df
    .limit(1)
    .count()
    > 0
):

    print(
        "ERROR: NULL values found "
        "in logical key."
    )

    null_key_df.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "ORDER_ITEM_OPTIONS contains "
        "NULL logical-key values."
    )


# =========================================================
# Validate logical-key uniqueness
#
# At this point exact duplicates are gone.
#
# If a key still occurs multiple times, the rows disagree
# on OPTION_PRICE, OPTION_QUANTITY, or metadata/business
# attributes.
#
# We intentionally fail instead of picking one.
# =========================================================

conflicting_keys_df = (
    silver_source_df

    .groupBy(
        *LOGICAL_KEY
    )

    .count()

    .filter(
        F.col("count") > 1
    )
)

conflicting_key_count = (
    conflicting_keys_df.count()
)

if conflicting_key_count > 0:

    print(
        "ERROR: conflicting logical keys found."
    )

    print(
        f"Number of conflicting logical keys: "
        f"{conflicting_key_count}"
    )

    conflicting_keys_df.show(
        20,
        truncate=False
    )

    # Show the actual source records for several
    # conflicting keys to make debugging easier.
    example_conflicts = (
        conflicting_keys_df
        .select(*LOGICAL_KEY)
        .limit(20)
    )

    conflicting_records_df = (
        silver_source_df
        .join(
            example_conflicts,
            on=LOGICAL_KEY,
            how="inner"
        )
        .orderBy(
            *LOGICAL_KEY
        )
    )

    print(
        "Example conflicting source records:"
    )

    conflicting_records_df.show(
        100,
        truncate=False
    )

    raise RuntimeError(
        "ORDER_ITEM_OPTIONS logical-key "
        "uniqueness validation failed."
    )


# =========================================================
# Clean source
# =========================================================

clean_source_count = (
    silver_source_df.count()
)

print(
    f"Clean source row count: "
    f"{clean_source_count}"
)

print(
    "ORDER_ITEM_OPTIONS schema ready "
    "for Silver:"
)

silver_source_df.printSchema()


# =========================================================
# Delta create / MERGE
# =========================================================

if DeltaTable.isDeltaTable(
    spark,
    SILVER_PATH
):

    print(
        "Existing ORDER_ITEM_OPTIONS "
        "Silver Delta table found."
    )

    print(
        "Running Delta MERGE..."
    )

    silver_delta = (
        DeltaTable.forPath(
            spark,
            SILVER_PATH
        )
    )

    merge_condition = """
        target.ORDER_ID = source.ORDER_ID
        AND
        target.LINEITEM_ID = source.LINEITEM_ID
        AND
        target.OPTION_GROUP_NAME = source.OPTION_GROUP_NAME
        AND
        target.OPTION_NAME = source.OPTION_NAME
    """

    (
        silver_delta
        .alias("target")

        .merge(
            silver_source_df.alias("source"),
            merge_condition
        )

        .whenMatchedUpdateAll()

        .whenNotMatchedInsertAll()

        .whenNotMatchedBySourceDelete()

        .execute()
    )

    print(
        "ORDER_ITEM_OPTIONS Delta MERGE "
        "completed successfully."
    )


else:

    print(
        "No Silver ORDER_ITEM_OPTIONS "
        "Delta table exists."
    )

    print(
        "Creating initial Silver table..."
    )

    (
        silver_source_df.write
        .format("delta")
        .mode("overwrite")
        .save(SILVER_PATH)
    )

    print(
        "Initial ORDER_ITEM_OPTIONS "
        "Silver Delta table created."
    )


# =========================================================
# Post-write validation
# =========================================================

silver_result_df = (
    spark.read
    .format("delta")
    .load(SILVER_PATH)
)

silver_count = (
    silver_result_df.count()
)

print(
    f"Final Silver row count: "
    f"{silver_count}"
)


# Full snapshot + NOT MATCHED BY SOURCE DELETE means
# these should match exactly.
if silver_count != clean_source_count:

    raise RuntimeError(
        "ORDER_ITEM_OPTIONS post-MERGE "
        "row-count validation failed. "
        f"Clean source = {clean_source_count}, "
        f"Silver = {silver_count}"
    )


# =========================================================
# Final Silver key validation
# =========================================================

post_merge_duplicates = (
    silver_result_df

    .groupBy(
        *LOGICAL_KEY
    )

    .count()

    .filter(
        F.col("count") > 1
    )
)

if (
    post_merge_duplicates
    .limit(1)
    .count()
    > 0
):

    raise RuntimeError(
        "ORDER_ITEM_OPTIONS Silver contains "
        "duplicate logical keys after MERGE."
    )


print("")
print("=" * 80)
print(
    "ORDER_ITEM_OPTIONS SILVER JOB SUCCESSFUL"
)
print("=" * 80)

print(
    f"Bronze rows:              "
    f"{bronze_count}"
)

print(
    f"Exact duplicates removed: "
    f"{duplicates_removed}"
)

print(
    f"Clean source rows:        "
    f"{clean_source_count}"
)

print(
    f"Final Silver rows:        "
    f"{silver_count}"
)

print(
    f"Silver location:          "
    f"{SILVER_PATH}"
)


job.commit()