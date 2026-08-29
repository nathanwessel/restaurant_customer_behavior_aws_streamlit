import sys
import json
import boto3

from datetime import datetime, timezone

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel


# ---------------------------------------------------------
# Glue initialization
# ---------------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "load_type"
    ]
)

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session

job = Job(glue_context)
job.init(args["JOB_NAME"], args)

spark.conf.set("spark.sql.session.timeZone", "UTC")


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AWS_REGION = "us-east-1"

SECRET_ID = "globalpartners/rds/sqlserver"

JDBC_URL = (
    "jdbc:sqlserver://"
    "database-1.cif8yi242dar.us-east-1.rds.amazonaws.com:1433;"
    "databaseName=GlobalPartners;"
    "encrypt=true;"
    "trustServerCertificate=true;"
)

JDBC_DRIVER = "com.microsoft.sqlserver.jdbc.SQLServerDriver"

BRONZE_ROOT = "s3://nw-globalpartners-project/bronze"

LOAD_TYPE = args["load_type"].upper()

VALID_LOAD_TYPES = {
    "FULL_INITIAL",
    "FULL_DAILY"
}

if LOAD_TYPE not in VALID_LOAD_TYPES:
    raise ValueError(
        f"Invalid load_type '{LOAD_TYPE}'. "
        f"Expected one of {VALID_LOAD_TYPES}"
    )


TABLES = {
    "date_dim": "dbo.date_dim",
    "order_item_options": "dbo.order_item_options",
    "order_items": "dbo.order_items"
}


# ---------------------------------------------------------
# Retrieve SQL Server credentials
# ---------------------------------------------------------

secrets_client = boto3.client(
    "secretsmanager",
    region_name=AWS_REGION
)

secret_response = secrets_client.get_secret_value(
    SecretId=SECRET_ID
)

secret = json.loads(secret_response["SecretString"])

sql_username = secret["username"]
sql_password = secret["password"]


# ---------------------------------------------------------
# Use one ingestion timestamp for the whole batch
# ---------------------------------------------------------

batch_timestamp = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d %H:%M:%S.%f"
)

print(f"Starting GlobalPartners Bronze ingestion")
print(f"Load type: {LOAD_TYPE}")
print(f"Batch timestamp UTC: {batch_timestamp}")


# ---------------------------------------------------------
# Read each SQL Server table and replace its Bronze snapshot
# ---------------------------------------------------------

for bronze_table_name, source_table_name in TABLES.items():

    bronze_path = f"{BRONZE_ROOT}/{bronze_table_name}/"

    print("")
    print("=" * 80)
    print(f"Source table: {source_table_name}")
    print(f"Bronze path: {bronze_path}")
    print("=" * 80)

    # -----------------------------------------------------
    # JDBC read
    # -----------------------------------------------------

    source_df = (
        spark.read
        .format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", source_table_name)
        .option("user", sql_username)
        .option("password", sql_password)
        .option("driver", JDBC_DRIVER)
        .option("queryTimeout", "600")
        .load()
    )

    # Cache so count + write do not cause separate JDBC reads.
    source_df = source_df.persist(
        StorageLevel.MEMORY_AND_DISK
    )

    source_count = source_df.count()

    print(
        f"Rows read from {source_table_name}: "
        f"{source_count}"
    )

    # -----------------------------------------------------
    # Add Bronze ingestion metadata
    # -----------------------------------------------------

    bronze_df = (
        source_df
        .withColumn(
            "_INGESTED_AT",
            F.to_timestamp(F.lit(batch_timestamp))
        )
        .withColumn(
            "_SOURCE_TABLE",
            F.lit(f"GlobalPartners.{source_table_name}")
        )
        .withColumn(
            "_LOAD_TYPE",
            F.lit(LOAD_TYPE)
        )
    )

    print("Schema being written to Bronze:")
    bronze_df.printSchema()

    # -----------------------------------------------------
    # Replace current-state Bronze Delta snapshot
    # -----------------------------------------------------

    (
        bronze_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(bronze_path)
    )

    # -----------------------------------------------------
    # Validate resulting Delta table
    # -----------------------------------------------------

    bronze_check_df = (
        spark.read
        .format("delta")
        .load(bronze_path)
    )

    bronze_count = bronze_check_df.count()

    print(
        f"Rows written to {bronze_path}: "
        f"{bronze_count}"
    )

    if bronze_count != source_count:
        raise RuntimeError(
            f"Row count validation failed for "
            f"{source_table_name}. "
            f"Source count = {source_count}, "
            f"Bronze count = {bronze_count}"
        )

    source_df.unpersist()

    print(
        f"SUCCESS: {source_table_name} "
        f"was loaded to Bronze."
    )


print("")
print("All three Bronze tables loaded successfully.")

job.commit()