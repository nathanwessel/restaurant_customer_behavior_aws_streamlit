import sys
import re

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    FloatType,
    IntegerType,
    TimestampType
)
from pyspark.storagelevel import StorageLevel

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

job.init(
    args["JOB_NAME"],
    args
)


# Normalize parsed timestamps to UTC.
spark.conf.set(
    "spark.sql.session.timeZone",
    "UTC"
)


# =========================================================
# Configuration
# =========================================================

BRONZE_PATH = (
    "s3://nw-globalpartners-project/"
    "bronze/order_items/"
)

SILVER_PATH = (
    "s3://nw-globalpartners-project/"
    "silver/order_items/"
)


LOGICAL_KEY = [
    "ORDER_ID",
    "LINEITEM_ID",
    "RESTAURANT_ID",
    "CREATION_TIME_UTC"
]


EXPECTED_BUSINESS_COLUMNS = [
    "APP_NAME",
    "RESTAURANT_ID",
    "CREATION_TIME_UTC",
    "ORDER_ID",
    "USER_ID",
    "PRINTED_CARD_NUMBER",
    "IS_LOYALTY",
    "CURRENCY",
    "LINEITEM_ID",
    "ITEM_CATEGORY",
    "ITEM_NAME",
    "ITEM_PRICE",
    "ITEM_QUANTITY"
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
# Helper: safely parse SQL Server datetimeoffset
#
# SQL Server datetimeoffset can arrive through JDBC/Delta
# as a string containing an explicit timezone offset.
#
# We try several common representations and return NULL
# if none can be parsed. A later validation step turns
# that NULL into a job failure when the original value
# was non-null.
# =========================================================

def parse_datetimeoffset(column_name):

    value = F.trim(
        F.col(column_name).cast("string")
    )

    return F.coalesce(

        # Let Spark try its standard timestamp parser first.
        F.try_to_timestamp(value),

        # Example:
        # 2026-08-29 12:34:56.1234567 +00:00
        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ss.SSSSSSS XXX"
            )
        ),

        # Same, without space before timezone offset.
        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ss.SSSSSSSXXX"
            )
        ),

        # Microseconds
        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ss.SSSSSS XXX"
            )
        ),

        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ss.SSSSSSXXX"
            )
        ),

        # Milliseconds
        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ss.SSS XXX"
            )
        ),

        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ss.SSSXXX"
            )
        ),

        # No fractional seconds
        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ss XXX"
            )
        ),

        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd HH:mm:ssXXX"
            )
        ),

        # ISO representations using T
        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd'T'HH:mm:ss.SSSSSSSXXX"
            )
        ),

        F.try_to_timestamp(
            value,
            F.lit(
                "yyyy-MM-dd'T'HH:mm:ssXXX"
            )
        )
    )


# =========================================================
# Read Bronze
# =========================================================

print("=" * 80)
print("Reading ORDER_ITEMS Bronze Delta table")
print(f"Bronze path: {BRONZE_PATH}")
print("=" * 80)


bronze_df = (
    spark.read
    .format("delta")
    .load(BRONZE_PATH)
)


bronze_count = bronze_df.count()


print(
    f"Bronze ORDER_ITEMS row count: "
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


print(
    "Columns after UPPER_SNAKE_CASE normalization:"
)

print(bronze_df.columns)


# =========================================================
# Validate expected schema
# =========================================================

actual_columns = set(
    bronze_df.columns
)

expected_columns = set(
    EXPECTED_COLUMNS
)


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
        "ORDER_ITEMS is missing expected columns: "
        f"{sorted(missing_columns)}"
    )


if unexpected_columns:

    raise RuntimeError(
        "ORDER_ITEMS contains unexpected columns: "
        f"{sorted(unexpected_columns)}"
    )


# =========================================================
# Preserve original values for cast validation
# =========================================================

working_df = (
    bronze_df

    .withColumn(
        "__ORIGINAL_CREATION_TIME_UTC",
        F.col("CREATION_TIME_UTC")
    )

    .withColumn(
        "__ORIGINAL_ITEM_PRICE",
        F.col("ITEM_PRICE")
    )

    .withColumn(
        "__ORIGINAL_ITEM_QUANTITY",
        F.col("ITEM_QUANTITY")
    )

    .withColumn(
        "__ORIGINAL_INGESTED_AT",
        F.col("_INGESTED_AT")
    )
)


# =========================================================
# Enforce Silver data types
# =========================================================

silver_source_df = (
    working_df

    .withColumn(
        "APP_NAME",
        F.col("APP_NAME").cast(
            StringType()
        )
    )

    .withColumn(
        "RESTAURANT_ID",
        F.col("RESTAURANT_ID").cast(
            StringType()
        )
    )

    # SQL Server datetimeoffset -> Spark TIMESTAMP
    #
    # Because spark.sql.session.timeZone = UTC,
    # timestamps containing offsets are normalized
    # to their UTC instant.
    .withColumn(
        "CREATION_TIME_UTC",
        parse_datetimeoffset(
            "CREATION_TIME_UTC"
        )
    )

    .withColumn(
        "ORDER_ID",
        F.col("ORDER_ID").cast(
            StringType()
        )
    )

    .withColumn(
        "USER_ID",
        F.col("USER_ID").cast(
            StringType()
        )
    )

    .withColumn(
        "PRINTED_CARD_NUMBER",
        F.col("PRINTED_CARD_NUMBER").cast(
            StringType()
        )
    )

    .withColumn(
        "IS_LOYALTY",
        F.col("IS_LOYALTY").cast(
            StringType()
        )
    )

    .withColumn(
        "CURRENCY",
        F.col("CURRENCY").cast(
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
        "ITEM_CATEGORY",
        F.col("ITEM_CATEGORY").cast(
            StringType()
        )
    )

    .withColumn(
        "ITEM_NAME",
        F.col("ITEM_NAME").cast(
            StringType()
        )
    )

    # SQL Server REAL maps naturally to Spark FLOAT.
    .withColumn(
        "ITEM_PRICE",
        F.col("ITEM_PRICE").cast(
            FloatType()
        )
    )

    .withColumn(
        "ITEM_QUANTITY",
        F.col("ITEM_QUANTITY").cast(
            IntegerType()
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
# Validate type conversions
# =========================================================

cast_failure_df = (
    silver_source_df

    .filter(

        (
            F.col(
                "__ORIGINAL_CREATION_TIME_UTC"
            ).isNotNull()
            &
            F.col(
                "CREATION_TIME_UTC"
            ).isNull()
        )

        |

        (
            F.col(
                "__ORIGINAL_ITEM_PRICE"
            ).isNotNull()
            &
            F.col(
                "ITEM_PRICE"
            ).isNull()
        )

        |

        (
            F.col(
                "__ORIGINAL_ITEM_QUANTITY"
            ).isNotNull()
            &
            F.col(
                "ITEM_QUANTITY"
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
        "ERROR: ORDER_ITEMS contains values "
        "that could not be safely converted."
    )

    cast_failure_df.select(
        "CREATION_TIME_UTC",
        "__ORIGINAL_CREATION_TIME_UTC",
        "ITEM_PRICE",
        "__ORIGINAL_ITEM_PRICE",
        "ITEM_QUANTITY",
        "__ORIGINAL_ITEM_QUANTITY",
        "_INGESTED_AT",
        "__ORIGINAL_INGESTED_AT"
    ).show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "ORDER_ITEMS Silver type "
        "enforcement failed."
    )


# Remove temporary validation fields.
silver_source_df = (
    silver_source_df
    .drop(
        "__ORIGINAL_CREATION_TIME_UTC",
        "__ORIGINAL_ITEM_PRICE",
        "__ORIGINAL_ITEM_QUANTITY",
        "__ORIGINAL_INGESTED_AT"
    )
)


# =========================================================
# Remove exact duplicates
# =========================================================

before_dedupe_count = (
    silver_source_df.count()
)


silver_source_df = (
    silver_source_df
    .dropDuplicates()
)


# Persist because the cleaned source is used repeatedly
# for validation and the MERGE.
silver_source_df = (
    silver_source_df
    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
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

    F.col("RESTAURANT_ID").isNull()

    |

    F.col("CREATION_TIME_UTC").isNull()
)


null_key_df = (
    silver_source_df
    .filter(
        null_key_condition
    )
)


if (
    null_key_df
    .limit(1)
    .count()
    > 0
):

    print(
        "ERROR: NULL values found "
        "in ORDER_ITEMS logical key."
    )

    null_key_df.select(
        *LOGICAL_KEY
    ).show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "ORDER_ITEMS contains NULL "
        "logical-key values."
    )


# =========================================================
# Validate logical-key uniqueness
#
# Exact duplicates have already been removed.
#
# Therefore, multiple rows with the same:
#
# ORDER_ID
# LINEITEM_ID
# RESTAURANT_ID
# CREATION_TIME_UTC
#
# indicate conflicting records.
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
        "ERROR: conflicting ORDER_ITEMS "
        "logical keys found."
    )

    print(
        f"Number of conflicting logical keys: "
        f"{conflicting_key_count}"
    )


    conflicting_keys_df.show(
        20,
        truncate=False
    )


    # Retrieve sample complete records so we can
    # determine why the proposed logical key
    # isn't sufficient.
    example_conflicts = (
        conflicting_keys_df
        .select(
            *LOGICAL_KEY
        )
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
        "ORDER_ITEMS logical-key "
        "uniqueness validation failed."
    )


# =========================================================
# Clean source count
# =========================================================

clean_source_count = (
    silver_source_df.count()
)


print(
    f"Clean ORDER_ITEMS source count: "
    f"{clean_source_count}"
)


print(
    "ORDER_ITEMS schema ready for Silver:"
)


silver_source_df.printSchema()


# =========================================================
# Create / MERGE Silver Delta
# =========================================================

if DeltaTable.isDeltaTable(
    spark,
    SILVER_PATH
):

    print(
        "Existing ORDER_ITEMS "
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
        target.RESTAURANT_ID = source.RESTAURANT_ID
        AND
        target.CREATION_TIME_UTC = source.CREATION_TIME_UTC
    """


    (
        silver_delta
        .alias("target")

        .merge(
            silver_source_df.alias("source"),
            merge_condition
        )

        # Same logical row exists:
        # refresh all current attributes.
        .whenMatchedUpdateAll()

        # New logical row:
        # add it.
        .whenNotMatchedInsertAll()

        # Because Bronze represents a complete current
        # RDS snapshot, anything remaining in Silver but
        # absent from Bronze has been deleted upstream.
        .whenNotMatchedBySourceDelete()

        .execute()
    )


    print(
        "ORDER_ITEMS Delta MERGE "
        "completed successfully."
    )


else:

    print(
        "No Silver ORDER_ITEMS "
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
        "Initial ORDER_ITEMS "
        "Silver Delta table created."
    )


# =========================================================
# Post-MERGE validation
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
    f"Final Silver ORDER_ITEMS row count: "
    f"{silver_count}"
)


# Since our source is a complete snapshot and the MERGE
# deletes rows absent from source, these counts must match.
if silver_count != clean_source_count:

    raise RuntimeError(
        "ORDER_ITEMS post-MERGE row-count "
        "validation failed. "
        f"Clean source = {clean_source_count}, "
        f"Silver = {silver_count}"
    )


# =========================================================
# Final logical-key validation
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

    print(
        "ERROR: duplicate logical keys found "
        "in final Silver table."
    )

    post_merge_duplicates.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "ORDER_ITEMS Silver contains "
        "duplicate logical keys after MERGE."
    )


# =========================================================
# Success summary
# =========================================================

print("")
print("=" * 80)
print(
    "ORDER_ITEMS SILVER JOB SUCCESSFUL"
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


silver_source_df.unpersist()

job.commit()