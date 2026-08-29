import sys
import re

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    ShortType,
    ByteType,
    StringType,
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

job.init(
    args["JOB_NAME"],
    args
)


# Keep all timestamps consistent.
spark.conf.set(
    "spark.sql.session.timeZone",
    "UTC"
)


# =========================================================
# Configuration
# =========================================================

BRONZE_PATH = (
    "s3://nw-globalpartners-project/"
    "bronze/date_dim/"
)

SILVER_PATH = (
    "s3://nw-globalpartners-project/"
    "silver/date_dim/"
)

LOGICAL_KEY = [
    "DATE_KEY"
]


# Expected source/business columns.
#
# We explicitly define this because Silver is supposed to
# enforce a known schema rather than silently accept bad
# upstream schema changes.
EXPECTED_BUSINESS_COLUMNS = [
    "DATE_KEY",
    "YEAR",
    "MONTH",
    "WEEK",
    "DAY_OF_WEEK",
    "IS_WEEKEND",
    "IS_HOLIDAY",
    "HOLIDAY_NAME"
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
# Helper: normalize column names
# =========================================================

def upper_snake_case(column_name):
    """
    Convert a column name to UPPER_SNAKE_CASE while
    preserving a leading underscore for ingestion metadata.

    Examples:
        date_key       -> DATE_KEY
        Day Of Week    -> DAY_OF_WEEK
        _ingested_at   -> _INGESTED_AT
    """

    has_leading_underscore = column_name.startswith("_")

    cleaned = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        column_name
    )

    cleaned = cleaned.strip("_").upper()

    if has_leading_underscore:
        cleaned = "_" + cleaned

    return cleaned


# =========================================================
# Read Bronze Delta
# =========================================================

print("=" * 80)
print("Reading DATE_DIM Bronze Delta table")
print(f"Bronze path: {BRONZE_PATH}")
print("=" * 80)


bronze_df = (
    spark.read
    .format("delta")
    .load(BRONZE_PATH)
)


bronze_count = bronze_df.count()

print(
    f"Bronze DATE_DIM row count: "
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


# Make sure normalization itself does not accidentally
# cause two different source columns to acquire one name.
if len(normalized_names) != len(set(normalized_names)):

    raise RuntimeError(
        "Column-name normalization created duplicate "
        "column names. Silver job stopped."
    )


for old_name, new_name in zip(
    bronze_df.columns,
    normalized_names
):

    if old_name != new_name:

        bronze_df = bronze_df.withColumnRenamed(
            old_name,
            new_name
        )


print(
    "Columns after UPPER_SNAKE_CASE normalization:"
)

print(bronze_df.columns)


# =========================================================
# Validate expected columns
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
        "DATE_DIM Bronze is missing expected columns: "
        f"{sorted(missing_columns)}"
    )


if unexpected_columns:

    raise RuntimeError(
        "DATE_DIM Bronze contains unexpected columns: "
        f"{sorted(unexpected_columns)}. "
        "Review the schema change before allowing it "
        "into Silver."
    )


# =========================================================
# Preserve pre-cast values temporarily
#
# This lets us distinguish:
#
#    genuine source NULL
#
# from:
#
#    non-null source value that failed conversion
# =========================================================

working_df = (
    bronze_df
    .withColumn(
        "__ORIGINAL_DATE_KEY",
        F.col("DATE_KEY")
    )
    .withColumn(
        "__ORIGINAL_YEAR",
        F.col("YEAR")
    )
    .withColumn(
        "__ORIGINAL_MONTH",
        F.col("MONTH")
    )
    .withColumn(
        "__ORIGINAL_WEEK",
        F.col("WEEK")
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
        "DATE_KEY",
        F.col("DATE_KEY").cast(
            DateType()
        )
    )

    .withColumn(
        "YEAR",
        F.col("YEAR").cast(
            ShortType()
        )
    )

    .withColumn(
        "MONTH",
        F.col("MONTH").cast(
            ByteType()
        )
    )

    .withColumn(
        "WEEK",
        F.col("WEEK").cast(
            ByteType()
        )
    )

    .withColumn(
        "DAY_OF_WEEK",
        F.col("DAY_OF_WEEK").cast(
            StringType()
        )
    )

    .withColumn(
        "IS_WEEKEND",
        F.col("IS_WEEKEND").cast(
            StringType()
        )
    )

    .withColumn(
        "IS_HOLIDAY",
        F.col("IS_HOLIDAY").cast(
            StringType()
        )
    )

    .withColumn(
        "HOLIDAY_NAME",
        F.col("HOLIDAY_NAME").cast(
            StringType()
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
# Detect failed type conversions
#
# A legitimate NULL remains acceptable unless it is one of
# our key columns.
#
# But:
#
#    source value = "not-a-date"
#    DATE cast    = NULL
#
# should fail instead of silently entering Silver.
# =========================================================

cast_failure_df = (
    silver_source_df
    .filter(

        (
            F.col("__ORIGINAL_DATE_KEY").isNotNull()
            & F.col("DATE_KEY").isNull()
        )

        |

        (
            F.col("__ORIGINAL_YEAR").isNotNull()
            & F.col("YEAR").isNull()
        )

        |

        (
            F.col("__ORIGINAL_MONTH").isNotNull()
            & F.col("MONTH").isNull()
        )

        |

        (
            F.col("__ORIGINAL_WEEK").isNotNull()
            & F.col("WEEK").isNull()
        )

        |

        (
            F.col(
                "__ORIGINAL_INGESTED_AT"
            ).isNotNull()
            & F.col(
                "_INGESTED_AT"
            ).isNull()
        )
    )
)


if cast_failure_df.limit(1).count() > 0:

    print(
        "ERROR: DATE_DIM contains values "
        "that could not be safely cast."
    )

    cast_failure_df.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "DATE_DIM Silver type enforcement failed."
    )


# Remove temporary validation columns.
silver_source_df = (
    silver_source_df
    .drop(
        "__ORIGINAL_DATE_KEY",
        "__ORIGINAL_YEAR",
        "__ORIGINAL_MONTH",
        "__ORIGINAL_WEEK",
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
# Validate logical key is not NULL
# =========================================================

null_key_df = (
    silver_source_df
    .filter(
        F.col("DATE_KEY").isNull()
    )
)


if null_key_df.limit(1).count() > 0:

    print(
        "ERROR: DATE_DIM contains NULL DATE_KEY values."
    )

    null_key_df.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "DATE_DIM logical-key validation failed: "
        "DATE_KEY contains NULL."
    )


# =========================================================
# Validate logical key uniqueness
#
# Exact duplicates were already removed.
#
# Therefore, if DATE_KEY appears more than once here,
# those rows disagree on some other attribute and should
# NOT be arbitrarily collapsed.
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


if conflicting_keys_df.limit(1).count() > 0:

    print(
        "ERROR: DATE_DIM contains conflicting "
        "records for the same DATE_KEY."
    )

    conflicting_keys_df.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "DATE_DIM logical-key uniqueness "
        "validation failed."
    )


clean_source_count = (
    silver_source_df.count()
)


print(
    f"Clean DATE_DIM source count: "
    f"{clean_source_count}"
)

print(
    "Schema ready for Silver:"
)

silver_source_df.printSchema()


# =========================================================
# Write / MERGE Silver Delta
# =========================================================

if DeltaTable.isDeltaTable(
    spark,
    SILVER_PATH
):

    print(
        "Existing Silver Delta table found."
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


    merge_condition = (
        "target.DATE_KEY = "
        "source.DATE_KEY"
    )


    (
        silver_delta.alias("target")

        .merge(
            silver_source_df.alias("source"),
            merge_condition
        )

        # Existing DATE_KEY:
        # update all current values.
        .whenMatchedUpdateAll()

        # New DATE_KEY:
        # insert it.
        .whenNotMatchedInsertAll()

        # DATE_KEY exists in old Silver but is absent
        # from today's full Bronze snapshot:
        # delete it.
        .whenNotMatchedBySourceDelete()

        .execute()
    )


    print(
        "Delta MERGE completed successfully."
    )


else:

    print(
        "No Silver Delta table exists yet."
    )

    print(
        "Creating initial DATE_DIM Silver table..."
    )


    (
        silver_source_df.write
        .format("delta")
        .mode("overwrite")
        .save(SILVER_PATH)
    )


    print(
        "Initial Silver Delta table created."
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
    f"Final Silver DATE_DIM row count: "
    f"{silver_count}"
)


# Because this is a full source snapshot and the MERGE
# deletes rows absent from source, Silver should contain
# exactly the same number of clean records.
if silver_count != clean_source_count:

    raise RuntimeError(
        "DATE_DIM post-MERGE row-count validation "
        f"failed. Clean source = {clean_source_count}, "
        f"Silver = {silver_count}"
    )


# Check the key one more time after the MERGE.
silver_duplicate_key_df = (
    silver_result_df
    .groupBy(
        "DATE_KEY"
    )
    .count()
    .filter(
        F.col("count") > 1
    )
)


if (
    silver_duplicate_key_df
    .limit(1)
    .count()
    > 0
):

    raise RuntimeError(
        "DATE_DIM Silver contains duplicate "
        "DATE_KEY values after MERGE."
    )


print("")
print("=" * 80)
print("DATE_DIM SILVER JOB SUCCESSFUL")
print("=" * 80)

print(
    f"Bronze rows:             "
    f"{bronze_count}"
)

print(
    f"Exact duplicates removed:"
    f" {duplicates_removed}"
)

print(
    f"Clean source rows:       "
    f"{clean_source_count}"
)

print(
    f"Final Silver rows:       "
    f"{silver_count}"
)

print(
    f"Silver location:         "
    f"{SILVER_PATH}"
)


job.commit()