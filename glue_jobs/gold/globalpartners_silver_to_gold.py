import sys
from datetime import datetime, timezone

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.storagelevel import StorageLevel


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
# Paths
# =========================================================

SILVER_ORDER_ITEMS = (
    "s3://nw-globalpartners-project/"
    "silver/order_items/"
)

SILVER_ORDER_ITEM_OPTIONS = (
    "s3://nw-globalpartners-project/"
    "silver/order_item_options/"
)

GOLD_ROOT = (
    "s3://nw-globalpartners-project/gold"
)

GOLD_PATHS = {
    "customer_clv":
        f"{GOLD_ROOT}/customer_clv/",

    "customer_rfm":
        f"{GOLD_ROOT}/customer_rfm/",

    "churn_indicators":
        f"{GOLD_ROOT}/churn_indicators/",

    "sales_trends_daily":
        f"{GOLD_ROOT}/sales_trends_daily/",

    "sales_trends_weekly":
        f"{GOLD_ROOT}/sales_trends_weekly/",

    "sales_trends_monthly":
        f"{GOLD_ROOT}/sales_trends_monthly/",

    "loyalty_program_impact":
        f"{GOLD_ROOT}/loyalty_program_impact/",

    "restaurant_performance":
        f"{GOLD_ROOT}/restaurant_performance/",

    "discount_effectiveness":
        f"{GOLD_ROOT}/discount_effectiveness/"
}


# =========================================================
# Document business assumptions
# =========================================================

print("=" * 90)
print("GLOBALPARTNERS SILVER -> GOLD")
print("=" * 90)

print("""
BUSINESS ASSUMPTIONS

1. USER_ID identifies a customer.

2. One order is identified by:
       ORDER_ID + RESTAURANT_ID

3. ORDER_ITEM_OPTIONS joins to ORDER_ITEMS on:
       ORDER_ID + LINEITEM_ID

4. IMPORTANT:
       ITEM_PRICE is ALREADY multiplied by ITEM_QUANTITY.

5. Therefore:
       LINE_REVENUE =
           ITEM_PRICE
           + SUM(OPTION_PRICE * OPTION_QUANTITY)

6. An order is considered DISCOUNTED if any joined
   ORDER_ITEM_OPTIONS record has OPTION_PRICE = 0.

7. Historical CLV means observed lifetime revenue.
   It is NOT a prediction of future customer value.

8. RFM uses the six months ending on ANALYSIS_DATE.

9. ANALYSIS_DATE is the maximum order date present
   in the Silver source data.

10. A USER_ID is considered a loyalty member when
    that USER_ID's latest order has IS_LOYALTY = TRUE.

11. Churn status:
       DAYS_SINCE_LAST_ORDER > 45 -> AT_RISK
       otherwise                  -> NOT_AT_RISK

12. NULL, empty, and whitespace-only USER_ID values:
       - remain in non-customer business metrics
       - are excluded from CLV, RFM, churn, and loyalty

13. RFM scoring:
       FREQUENCY_6M = 0 -> F_SCORE = 1
       MONETARY_6M  = 0 -> M_SCORE = 1

    Customers with positive frequency / monetary value
    are scored from 2 through 5.

    CUME_DIST is used so equal values receive equal scores.

14. VIP RFM segment:
       R_SCORE >= 4
       F_SCORE >= 4
       M_SCORE >= 4
       R_SCORE + F_SCORE + M_SCORE >= 14

15. Restaurant average daily/weekly orders are calculated
    across each RESTAURANT_ID's observed first-to-last-order
    calendar span, including zero-order days/weeks inside
    that observed span.
""")


# =========================================================
# Read Silver Delta
# =========================================================

order_items = (
    spark.read
    .format("delta")
    .load(SILVER_ORDER_ITEMS)
    .persist(StorageLevel.MEMORY_AND_DISK)
)

order_options = (
    spark.read
    .format("delta")
    .load(SILVER_ORDER_ITEM_OPTIONS)
    .persist(StorageLevel.MEMORY_AND_DISK)
)

order_items_count = order_items.count()
order_options_count = order_options.count()

print(
    f"Silver ORDER_ITEMS rows: "
    f"{order_items_count}"
)

print(
    f"Silver ORDER_ITEM_OPTIONS rows: "
    f"{order_options_count}"
)


# =========================================================
# Critical join validation
#
# ORDER_ITEM_OPTIONS only contains:
#   ORDER_ID
#   LINEITEM_ID
#
# Therefore ORDER_ID + LINEITEM_ID must uniquely identify
# one ORDER_ITEMS record before options can be joined safely.
# =========================================================

unsafe_item_join_keys = (
    order_items
    .groupBy(
        "ORDER_ID",
        "LINEITEM_ID"
    )
    .count()
    .filter(
        F.col("count") > 1
    )
)

unsafe_item_join_key_count = (
    unsafe_item_join_keys.count()
)

if unsafe_item_join_key_count > 0:

    print(
        "ERROR: ORDER_ID + LINEITEM_ID is not "
        "unique in ORDER_ITEMS."
    )

    print(
        "ORDER_ITEM_OPTIONS cannot be safely joined."
    )

    unsafe_item_join_keys.show(
        50,
        truncate=False
    )

    raise RuntimeError(
        "Gold job stopped because "
        "ORDER_ID + LINEITEM_ID is not unique "
        "enough to safely join ORDER_ITEM_OPTIONS."
    )

print(
    "VALIDATION PASSED: ORDER_ID + LINEITEM_ID "
    "uniquely identifies ORDER_ITEMS records."
)


# =========================================================
# Revenue-required field validation
# =========================================================

if (
    order_items
    .filter(
        F.col("ITEM_PRICE").isNull()
    )
    .limit(1)
    .count()
    > 0
):
    raise RuntimeError(
        "ITEM_PRICE contains NULL values. "
        "Revenue cannot be calculated safely."
    )

if (
    order_options
    .filter(
        F.col("OPTION_PRICE").isNull()
        |
        F.col("OPTION_QUANTITY").isNull()
    )
    .limit(1)
    .count()
    > 0
):
    raise RuntimeError(
        "OPTION_PRICE or OPTION_QUANTITY "
        "contains NULL values."
    )


# =========================================================
# Normalize IS_LOYALTY
#
# Expected source strings:
# TRUE
# FALSE
# =========================================================

order_items = (
    order_items
    .withColumn(
        "LOYALTY_FLAG",
        F.when(
            F.upper(
                F.trim(
                    F.col("IS_LOYALTY")
                )
            ) == "TRUE",
            F.lit(1)
        )
        .when(
            F.upper(
                F.trim(
                    F.col("IS_LOYALTY")
                )
            ) == "FALSE",
            F.lit(0)
        )
        .otherwise(
            F.lit(None)
        )
    )
)

invalid_loyalty = (
    order_items
    .filter(
        F.col("IS_LOYALTY").isNotNull()
        &
        F.col("LOYALTY_FLAG").isNull()
    )
)

if invalid_loyalty.limit(1).count() > 0:

    print(
        "Unexpected IS_LOYALTY values:"
    )

    invalid_loyalty.select(
        "IS_LOYALTY"
    ).distinct().show(
        truncate=False
    )

    raise RuntimeError(
        "Unexpected IS_LOYALTY values."
    )


# =========================================================
# Aggregate options to one record per line item
# =========================================================

option_fact = (
    order_options

    .withColumn(
        "OPTION_EXTENDED_REVENUE",
        (
            F.col("OPTION_PRICE")
            .cast("decimal(20,2)")
            *
            F.col("OPTION_QUANTITY")
            .cast("decimal(20,2)")
        ).cast("decimal(24,2)")
    )

    .groupBy(
        "ORDER_ID",
        "LINEITEM_ID"
    )

    .agg(
        F.sum(
            "OPTION_EXTENDED_REVENUE"
        ).cast(
            "decimal(24,2)"
        ).alias(
            "OPTION_REVENUE"
        ),

        F.max(
            F.when(
                F.col("OPTION_PRICE") == 0,
                F.lit(1)
            ).otherwise(
                F.lit(0)
            )
        ).alias(
            "HAS_ZERO_PRICE_OPTION"
        ),

        F.count(
            F.lit(1)
        ).alias(
            "OPTION_COUNT"
        )
    )
)


# =========================================================
# Build shared line-level fact
#
# IMPORTANT:
# ITEM_PRICE is already quantity-extended.
#
# LINE_REVENUE =
#     ITEM_PRICE
#     + OPTION_REVENUE
# =========================================================

line_fact = (
    order_items.alias("i")

    .join(
        option_fact.alias("o"),
        on=[
            "ORDER_ID",
            "LINEITEM_ID"
        ],
        how="left"
    )

    .withColumn(
        "OPTION_REVENUE",
        F.coalesce(
            F.col("OPTION_REVENUE"),
            F.lit(0).cast(
                "decimal(24,2)"
            )
        )
    )

    .withColumn(
        "HAS_ZERO_PRICE_OPTION",
        F.coalesce(
            F.col("HAS_ZERO_PRICE_OPTION"),
            F.lit(0)
        )
    )

    .withColumn(
        "LINE_REVENUE",
        (
            F.col("ITEM_PRICE")
            .cast("decimal(24,2)")
            +
            F.col("OPTION_REVENUE")
        ).cast(
            "decimal(24,2)"
        )
    )
    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
)


# =========================================================
# Validate customer and loyalty consistency inside an order
#
# One order =
# ORDER_ID + RESTAURANT_ID
# =========================================================

order_consistency = (
    line_fact

    .groupBy(
        "ORDER_ID",
        "RESTAURANT_ID"
    )

    .agg(
        F.countDistinct(
            "USER_ID"
        ).alias(
            "USER_ID_COUNT"
        ),

        F.countDistinct(
            "LOYALTY_FLAG"
        ).alias(
            "LOYALTY_VALUE_COUNT"
        )
    )

    .filter(
        (F.col("USER_ID_COUNT") > 1)
        |
        (F.col("LOYALTY_VALUE_COUNT") > 1)
    )
)

if order_consistency.limit(1).count() > 0:

    print(
        "Orders contain inconsistent USER_ID "
        "or IS_LOYALTY values across line items."
    )

    order_consistency.show(
        50,
        truncate=False
    )

    raise RuntimeError(
        "Order-level customer/loyalty "
        "consistency validation failed."
    )


# =========================================================
# Build shared order-level fact
#
# ASSUMPTION:
# One order =
# ORDER_ID + RESTAURANT_ID
# =========================================================

print(
    "ORDER ASSUMPTION: one order is identified by "
    "ORDER_ID + RESTAURANT_ID."
)

order_fact = (
    line_fact

    .groupBy(
        "ORDER_ID",
        "RESTAURANT_ID"
    )

    .agg(
        F.first(
            "USER_ID",
            ignorenulls=True
        ).alias(
            "USER_ID"
        ),

        F.first(
            "LOYALTY_FLAG",
            ignorenulls=True
        ).alias(
            "LOYALTY_FLAG"
        ),

        F.min(
            "CREATION_TIME_UTC"
        ).alias(
            "ORDER_TIMESTAMP"
        ),

        F.sum(
            "LINE_REVENUE"
        ).cast(
            "decimal(30,2)"
        ).alias(
            "ORDER_REVENUE"
        ),

        F.max(
            "HAS_ZERO_PRICE_OPTION"
        ).alias(
            "HAS_DISCOUNT"
        ),

        F.sum(
            F.coalesce(
                F.col("ITEM_QUANTITY"),
                F.lit(0)
            )
        ).alias(
            "TOTAL_ITEM_QUANTITY"
        )
    )

    .withColumn(
        "ORDER_DATE",
        F.to_date(
            "ORDER_TIMESTAMP"
        )
    )

    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
)

order_fact_count = order_fact.count()

print(
    f"Derived order count "
    f"(ORDER_ID + RESTAURANT_ID): "
    f"{order_fact_count}"
)


# =========================================================
# USER_ID data-quality diagnostics
#
# Anonymous/unidentified orders remain valid for:
# - sales trends
# - restaurant performance
# - discount effectiveness
#
# They are excluded from customer-level metrics.
# =========================================================

null_user_orders = (
    order_fact
    .filter(
        F.col("USER_ID").isNull()
    )
    .count()
)

blank_user_orders = (
    order_fact
    .filter(
        F.col("USER_ID").isNotNull()
        &
        (
            F.length(
                F.trim(
                    F.col("USER_ID")
                )
            ) == 0
        )
    )
    .count()
)

print(
    f"Orders with NULL USER_ID: "
    f"{null_user_orders}"
)

print(
    f"Orders with blank/whitespace USER_ID: "
    f"{blank_user_orders}"
)

print(
    "NULL/blank USER_ID orders remain in "
    "non-customer Gold metrics but are excluded "
    "from CLV, RFM, churn, and loyalty."
)


# =========================================================
# Customer-eligible orders
# =========================================================

customer_orders = (
    order_fact

    .filter(
        F.col("USER_ID").isNotNull()
        &
        (
            F.length(
                F.trim(
                    F.col("USER_ID")
                )
            ) > 0
        )
    )

    .withColumn(
        "USER_ID",
        F.trim(
            F.col("USER_ID")
        )
    )
)

customer_order_count = (
    customer_orders.count()
)

customer_count = (
    customer_orders
    .select("USER_ID")
    .distinct()
    .count()
)

print(
    f"Customer-eligible orders: "
    f"{customer_order_count}"
)

print(
    f"Distinct valid USER_ID values: "
    f"{customer_count}"
)


# =========================================================
# ANALYSIS_DATE
#
# Use maximum source order date,
# NOT current_date().
# =========================================================

analysis_date = (
    order_fact
    .agg(
        F.max(
            "ORDER_DATE"
        ).alias(
            "ANALYSIS_DATE"
        )
    )
    .first()[
        "ANALYSIS_DATE"
    ]
)

if analysis_date is None:
    raise RuntimeError(
        "Unable to determine ANALYSIS_DATE."
    )

print(
    f"ANALYSIS_DATE: "
    f"{analysis_date}"
)

generated_at = (
    datetime.now(
        timezone.utc
    )
    .strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
)


# =========================================================
# Helper: add common Gold metadata
# =========================================================

def add_gold_metadata(df):

    return (
        df
        .withColumn(
            "_GENERATED_AT",
            F.to_timestamp(
                F.lit(
                    generated_at
                )
            )
        )
        .withColumn(
            "_ANALYSIS_DATE",
            F.lit(
                analysis_date
            ).cast(
                "date"
            )
        )
    )


# =========================================================
# Helper: overwrite current-state Parquet Gold dataset
# =========================================================

def write_gold(
    df,
    dataset_name
):

    output_path = (
        GOLD_PATHS[
            dataset_name
        ]
    )

    final_df = (
        add_gold_metadata(
            df
        )
    )

    row_count = (
        final_df.count()
    )

    print("")
    print(
        f"Writing Gold dataset: "
        f"{dataset_name}"
    )
    print(
        f"Rows: {row_count}"
    )
    print(
        f"Path: {output_path}"
    )

    (
        final_df
        .write
        .mode("overwrite")
        .parquet(
            output_path
        )
    )

    print(
        f"SUCCESS: "
        f"{dataset_name}"
    )


# =========================================================
# Customer base
# =========================================================

customer_base = (
    customer_orders

    .groupBy(
        "USER_ID"
    )

    .agg(
        F.sum(
            "ORDER_REVENUE"
        ).cast(
            "decimal(30,2)"
        ).alias(
            "LIFETIME_REVENUE"
        ),

        F.count(
            F.lit(1)
        ).alias(
            "LIFETIME_ORDER_COUNT"
        ),

        F.min(
            "ORDER_DATE"
        ).alias(
            "FIRST_ORDER_DATE"
        ),

        F.max(
            "ORDER_DATE"
        ).alias(
            "LAST_ORDER_DATE"
        )
    )

    .withColumn(
        "LIFETIME_AVG_ORDER_VALUE",
        (
            F.col(
                "LIFETIME_REVENUE"
            )
            /
            F.col(
                "LIFETIME_ORDER_COUNT"
            )
        ).cast(
            "decimal(30,2)"
        )
    )
)


# =========================================================
# GOLD 1: Historical Customer Lifetime Value
#
# NTILE(5):
# tile 1 -> LOW
# tiles 2-4 -> MEDIUM
# tile 5 -> HIGH
# =========================================================

clv_window = (
    Window
    .orderBy(
        F.col(
            "LIFETIME_REVENUE"
        ).asc()
    )
)

customer_clv = (
    customer_base

    .withColumn(
        "CLV_QUINTILE",
        F.ntile(5).over(
            clv_window
        )
    )

    .withColumn(
        "CLV_SEGMENT",
        F.when(
            F.col(
                "CLV_QUINTILE"
            ) == 5,
            F.lit(
                "HIGH"
            )
        )
        .when(
            F.col(
                "CLV_QUINTILE"
            ) == 1,
            F.lit(
                "LOW"
            )
        )
        .otherwise(
            F.lit(
                "MEDIUM"
            )
        )
    )
)

write_gold(
    customer_clv,
    "customer_clv"
)


# =========================================================
# GOLD 2: Customer RFM
#
# R = Recency
# F = Frequency
# M = Monetary
#
# RFM window:
# six months ending on ANALYSIS_DATE
#
# Higher score always means better.
# =========================================================

six_month_orders = (
    customer_orders

    .filter(
        F.col("ORDER_DATE")
        >=
        F.add_months(
            F.lit(
                analysis_date
            ),
            -6
        )
    )

    .filter(
        F.col("ORDER_DATE")
        <=
        F.lit(
            analysis_date
        )
    )
)

six_month_metrics = (
    six_month_orders

    .groupBy(
        "USER_ID"
    )

    .agg(
        F.count(
            F.lit(1)
        ).alias(
            "FREQUENCY_6M"
        ),

        F.sum(
            "ORDER_REVENUE"
        ).cast(
            "decimal(30,2)"
        ).alias(
            "MONETARY_6M"
        )
    )
)

rfm = (
    customer_base

    .select(
        "USER_ID",
        "LAST_ORDER_DATE"
    )

    .join(
        six_month_metrics,
        on="USER_ID",
        how="left"
    )

    .withColumn(
        "FREQUENCY_6M",
        F.coalesce(
            F.col(
                "FREQUENCY_6M"
            ),
            F.lit(0)
        )
    )

    .withColumn(
        "MONETARY_6M",
        F.coalesce(
            F.col(
                "MONETARY_6M"
            ),
            F.lit(0)
            .cast(
                "decimal(30,2)"
            )
        )
    )

    .withColumn(
        "RECENCY_DAYS",
        F.datediff(
            F.lit(
                analysis_date
            ),
            F.col(
                "LAST_ORDER_DATE"
            )
        )
    )
)


# =========================================================
# Recency score
#
# Smaller RECENCY_DAYS is better.
#
# Ordering descending means older customers occur first
# and more recent customers occur toward the high end of
# the cumulative distribution.
#
# CUME_DIST keeps equal RECENCY_DAYS values together.
# =========================================================

r_window = (
    Window
    .orderBy(
        F.col(
            "RECENCY_DAYS"
        ).desc()
    )
)

rfm_with_r = (
    rfm

    .withColumn(
        "__R_PERCENTILE",
        F.cume_dist().over(
            r_window
        )
    )

    .withColumn(
        "R_SCORE",
        F.least(
            F.lit(5),
            F.greatest(
                F.lit(1),
                F.ceil(
                    F.col(
                        "__R_PERCENTILE"
                    )
                    *
                    F.lit(5)
                )
            )
        ).cast(
            "int"
        )
    )

    .drop(
        "__R_PERCENTILE"
    )
)


# =========================================================
# Frequency score
#
# Explicit business rule:
#
# FREQUENCY_6M = 0 -> F_SCORE = 1
#
# Positive-frequency customers:
# scores 2 through 5.
#
# CUME_DIST keeps equal frequency values together.
# =========================================================

f_positive_window = (
    Window
    .orderBy(
        F.col(
            "FREQUENCY_6M"
        ).asc()
    )
)

f_scores = (
    rfm

    .filter(
        F.col(
            "FREQUENCY_6M"
        ) > 0
    )

    .select(
        "USER_ID",
        "FREQUENCY_6M"
    )

    .withColumn(
        "__F_PERCENTILE",
        F.cume_dist().over(
            f_positive_window
        )
    )

    .withColumn(
        "F_SCORE",
        (
            F.lit(1)
            +
            F.ceil(
                F.col(
                    "__F_PERCENTILE"
                )
                *
                F.lit(4)
            )
        ).cast(
            "int"
        )
    )

    .select(
        "USER_ID",
        "F_SCORE"
    )
)


# =========================================================
# Monetary score
#
# Explicit business rule:
#
# MONETARY_6M = 0 -> M_SCORE = 1
#
# Positive-monetary customers:
# scores 2 through 5.
#
# CUME_DIST keeps equal monetary values together.
# =========================================================

m_positive_window = (
    Window
    .orderBy(
        F.col(
            "MONETARY_6M"
        ).asc()
    )
)

m_scores = (
    rfm

    .filter(
        F.col(
            "MONETARY_6M"
        ) > 0
    )

    .select(
        "USER_ID",
        "MONETARY_6M"
    )

    .withColumn(
        "__M_PERCENTILE",
        F.cume_dist().over(
            m_positive_window
        )
    )

    .withColumn(
        "M_SCORE",
        (
            F.lit(1)
            +
            F.ceil(
                F.col(
                    "__M_PERCENTILE"
                )
                *
                F.lit(4)
            )
        ).cast(
            "int"
        )
    )

    .select(
        "USER_ID",
        "M_SCORE"
    )
)


# =========================================================
# Combine RFM scores and assign segments
# =========================================================

customer_rfm = (
    rfm_with_r

    .join(
        f_scores,
        on="USER_ID",
        how="left"
    )

    .join(
        m_scores,
        on="USER_ID",
        how="left"
    )

    .withColumn(
        "F_SCORE",
        F.coalesce(
            F.col(
                "F_SCORE"
            ),
            F.lit(1)
        )
    )

    .withColumn(
        "M_SCORE",
        F.coalesce(
            F.col(
                "M_SCORE"
            ),
            F.lit(1)
        )
    )

    .withColumn(
        "RFM_SEGMENT",

        F.when(
            (
                F.col(
                    "R_SCORE"
                ) >= 4
            )
            &
            (
                F.col(
                    "F_SCORE"
                ) >= 4
            )
            &
            (
                F.col(
                    "M_SCORE"
                ) >= 4
            )
            &
            (
                (
                    F.col("R_SCORE")
                    +
                    F.col("F_SCORE")
                    +
                    F.col("M_SCORE")
                ) >= 14
            ),
            F.lit(
                "VIP"
            )
        )

        .when(
            (
                F.col(
                    "R_SCORE"
                ) >= 4
            )
            &
            (
                F.col(
                    "F_SCORE"
                ) <= 2
            ),
            F.lit(
                "NEW_CUSTOMER"
            )
        )

        .when(
            (
                F.col(
                    "R_SCORE"
                ) <= 2
            )
            &
            (
                F.col(
                    "F_SCORE"
                ) <= 2
            ),
            F.lit(
                "CHURN_RISK"
            )
        )

        .otherwise(
            F.lit(
                "OTHER"
            )
        )
    )
)


# =========================================================
# RFM validation
# =========================================================

invalid_zero_frequency = (
    customer_rfm

    .filter(
        (
            F.col(
                "FREQUENCY_6M"
            ) == 0
        )
        &
        (
            F.col(
                "F_SCORE"
            ) != 1
        )
    )
)

if (
    invalid_zero_frequency
    .limit(1)
    .count()
    > 0
):

    invalid_zero_frequency.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "RFM validation failed: "
        "FREQUENCY_6M = 0 must have F_SCORE = 1."
    )


invalid_zero_monetary = (
    customer_rfm

    .filter(
        (
            F.col(
                "MONETARY_6M"
            ) == 0
        )
        &
        (
            F.col(
                "M_SCORE"
            ) != 1
        )
    )
)

if (
    invalid_zero_monetary
    .limit(1)
    .count()
    > 0
):

    invalid_zero_monetary.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "RFM validation failed: "
        "MONETARY_6M = 0 must have M_SCORE = 1."
    )


print(
    "RFM VALIDATION PASSED: "
    "zero-frequency users have F_SCORE = 1 "
    "and zero-monetary users have M_SCORE = 1."
)

print(
    "RFM VIP RULE: "
    "R_SCORE >= 4, F_SCORE >= 4, M_SCORE >= 4, "
    "and R_SCORE + F_SCORE + M_SCORE >= 14."
)

print("RFM segment counts:")
(
    customer_rfm
    .groupBy("RFM_SEGMENT")
    .count()
    .orderBy(
        F.col("count").desc()
    )
    .show(
        truncate=False
    )
)

write_gold(
    customer_rfm,
    "customer_rfm"
)


# =========================================================
# GOLD 3: Churn indicators
# =========================================================

order_sequence_window = (
    Window
    .partitionBy(
        "USER_ID"
    )
    .orderBy(
        "ORDER_TIMESTAMP",
        "ORDER_ID",
        "RESTAURANT_ID"
    )
)

customer_order_gaps = (
    customer_orders

    .withColumn(
        "PREVIOUS_ORDER_DATE",
        F.lag(
            "ORDER_DATE"
        ).over(
            order_sequence_window
        )
    )

    .withColumn(
        "GAP_DAYS",
        F.datediff(
            F.col(
                "ORDER_DATE"
            ),
            F.col(
                "PREVIOUS_ORDER_DATE"
            )
        )
    )
)

average_gaps = (
    customer_order_gaps

    .groupBy(
        "USER_ID"
    )

    .agg(
        F.avg(
            "GAP_DAYS"
        ).alias(
            "AVG_DAYS_BETWEEN_ORDERS"
        )
    )
)

spend_periods = (
    customer_orders

    .withColumn(
        "DAYS_AGO",
        F.datediff(
            F.lit(
                analysis_date
            ),
            F.col(
                "ORDER_DATE"
            )
        )
    )

    .groupBy(
        "USER_ID"
    )

    .agg(
        F.sum(
            F.when(
                (
                    F.col(
                        "DAYS_AGO"
                    ) >= 0
                )
                &
                (
                    F.col(
                        "DAYS_AGO"
                    ) < 30
                ),
                F.col(
                    "ORDER_REVENUE"
                )
            ).otherwise(
                F.lit(0)
            )
        ).cast(
            "decimal(30,2)"
        ).alias(
            "CURRENT_30_DAY_SPEND"
        ),

        F.sum(
            F.when(
                (
                    F.col(
                        "DAYS_AGO"
                    ) >= 30
                )
                &
                (
                    F.col(
                        "DAYS_AGO"
                    ) < 60
                ),
                F.col(
                    "ORDER_REVENUE"
                )
            ).otherwise(
                F.lit(0)
            )
        ).cast(
            "decimal(30,2)"
        ).alias(
            "PREVIOUS_30_DAY_SPEND"
        )
    )
)

churn_indicators = (
    customer_base

    .select(
        "USER_ID",
        "LAST_ORDER_DATE",
        "LIFETIME_ORDER_COUNT"
    )

    .join(
        average_gaps,
        on="USER_ID",
        how="left"
    )

    .join(
        spend_periods,
        on="USER_ID",
        how="left"
    )

    .withColumn(
        "DAYS_SINCE_LAST_ORDER",
        F.datediff(
            F.lit(
                analysis_date
            ),
            F.col(
                "LAST_ORDER_DATE"
            )
        )
    )

    .withColumn(
        "SPEND_CHANGE_PCT",
        F.when(
            F.col(
                "PREVIOUS_30_DAY_SPEND"
            ) != 0,
            (
                (
                    F.col(
                        "CURRENT_30_DAY_SPEND"
                    )
                    -
                    F.col(
                        "PREVIOUS_30_DAY_SPEND"
                    )
                )
                /
                F.col(
                    "PREVIOUS_30_DAY_SPEND"
                )
                *
                F.lit(100)
            ).cast(
                "decimal(12,2)"
            )
        )
    )

    .withColumn(
        "CHURN_STATUS",
        F.when(
            F.col(
                "DAYS_SINCE_LAST_ORDER"
            ) > 45,
            F.lit(
                "AT_RISK"
            )
        )
        .otherwise(
            F.lit(
                "NOT_AT_RISK"
            )
        )
    )
)

write_gold(
    churn_indicators,
    "churn_indicators"
)


# =========================================================
# Add consistent order timestamp/date back to line fact
# for category-level sales trends.
# =========================================================

line_sales_fact = (
    line_fact

    .join(
        order_fact.select(
            "ORDER_ID",
            "RESTAURANT_ID",
            "ORDER_TIMESTAMP",
            "ORDER_DATE"
        ),
        on=[
            "ORDER_ID",
            "RESTAURANT_ID"
        ],
        how="inner"
    )
)


# =========================================================
# Sales trend builder
#
# Each physical table has one time grain only.
# =========================================================

def build_sales_trend(
    source_df,
    period_column
):

    return (
        source_df

        .withColumn(
            "PERIOD_START_DATE",
            period_column
        )

        .groupBy(
            "PERIOD_START_DATE",
            "RESTAURANT_ID",
            "ITEM_CATEGORY"
        )

        .agg(
            F.sum(
                "LINE_REVENUE"
            ).cast(
                "decimal(30,2)"
            ).alias(
                "TOTAL_REVENUE"
            ),

            F.countDistinct(
                "ORDER_ID",
                "RESTAURANT_ID"
            ).alias(
                "ORDER_COUNT"
            ),

            F.sum(
                F.coalesce(
                    F.col(
                        "ITEM_QUANTITY"
                    ),
                    F.lit(0)
                )
            ).alias(
                "ITEM_QUANTITY"
            )
        )

        .withColumn(
            "AVERAGE_ORDER_VALUE",
            (
                F.col(
                    "TOTAL_REVENUE"
                )
                /
                F.col(
                    "ORDER_COUNT"
                )
            ).cast(
                "decimal(30,2)"
            )
        )
    )


# =========================================================
# GOLD 4a: Daily sales trends
# =========================================================

sales_daily = build_sales_trend(
    line_sales_fact,
    F.col(
        "ORDER_DATE"
    )
)

write_gold(
    sales_daily,
    "sales_trends_daily"
)


# =========================================================
# GOLD 4b: Weekly sales trends
# =========================================================

sales_weekly = build_sales_trend(
    line_sales_fact,
    F.to_date(
        F.date_trunc(
            "week",
            F.col(
                "ORDER_TIMESTAMP"
            )
        )
    )
)

write_gold(
    sales_weekly,
    "sales_trends_weekly"
)


# =========================================================
# GOLD 4c: Monthly sales trends
# =========================================================

sales_monthly = build_sales_trend(
    line_sales_fact,
    F.trunc(
        F.col(
            "ORDER_DATE"
        ),
        "month"
    )
)

write_gold(
    sales_monthly,
    "sales_trends_monthly"
)


# =========================================================
# GOLD 5: Loyalty program impact
#
# Membership is determined from USER_ID's latest order.
# =========================================================

latest_order_window = (
    Window
    .partitionBy(
        "USER_ID"
    )
    .orderBy(
        F.col(
            "ORDER_TIMESTAMP"
        ).desc(),
        F.col(
            "ORDER_ID"
        ).desc(),
        F.col(
            "RESTAURANT_ID"
        ).desc()
    )
)

latest_customer_order = (
    customer_orders

    .withColumn(
        "LATEST_ORDER_RANK",
        F.row_number().over(
            latest_order_window
        )
    )

    .filter(
        F.col(
            "LATEST_ORDER_RANK"
        ) == 1
    )

    .select(
        "USER_ID",
        F.col(
            "LOYALTY_FLAG"
        ).alias(
            "LATEST_LOYALTY_FLAG"
        )
    )
)

customer_loyalty_stats = (
    customer_base

    .join(
        latest_customer_order,
        on="USER_ID",
        how="inner"
    )

    .withColumn(
        "LOYALTY_STATUS",
        F.when(
            F.col(
                "LATEST_LOYALTY_FLAG"
            ) == 1,
            F.lit(
                "LOYALTY_MEMBER"
            )
        )
        .otherwise(
            F.lit(
                "NON_MEMBER"
            )
        )
    )

    .withColumn(
        "IS_REPEAT_USER",
        F.when(
            F.col(
                "LIFETIME_ORDER_COUNT"
            ) > 1,
            F.lit(1)
        )
        .otherwise(
            F.lit(0)
        )
    )
)

loyalty_program_impact = (
    customer_loyalty_stats

    .groupBy(
        "LOYALTY_STATUS"
    )

    .agg(
        F.count(
            F.lit(1)
        ).alias(
            "USER_COUNT"
        ),

        F.sum(
            "LIFETIME_ORDER_COUNT"
        ).alias(
            "ORDER_COUNT"
        ),

        F.sum(
            "LIFETIME_REVENUE"
        ).cast(
            "decimal(30,2)"
        ).alias(
            "TOTAL_REVENUE"
        ),

        F.avg(
            "LIFETIME_REVENUE"
        ).cast(
            "decimal(30,2)"
        ).alias(
            "AVG_LIFETIME_REVENUE_PER_USER"
        ),

        F.sum(
            "IS_REPEAT_USER"
        ).alias(
            "REPEAT_USER_COUNT"
        )
    )

    .withColumn(
        "AVG_ORDER_VALUE",
        (
            F.col(
                "TOTAL_REVENUE"
            )
            /
            F.col(
                "ORDER_COUNT"
            )
        ).cast(
            "decimal(30,2)"
        )
    )

    .withColumn(
        "REPEAT_USER_RATE_PCT",
        (
            F.col(
                "REPEAT_USER_COUNT"
            )
            /
            F.col(
                "USER_COUNT"
            )
            *
            F.lit(100)
        ).cast(
            "decimal(10,2)"
        )
    )
)

write_gold(
    loyalty_program_impact,
    "loyalty_program_impact"
)


# =========================================================
# GOLD 6: Restaurant performance
#
# Average daily/weekly orders are calculated across each
# restaurant's observed first-to-last-order calendar span.
#
# This means zero-order days/weeks BETWEEN the first and
# last observed order are included in the denominator.
#
# The restaurant is not penalized for time before its first
# observed order or after its last observed order.
# =========================================================

restaurant_performance = (
    order_fact

    .groupBy(
        "RESTAURANT_ID"
    )

    .agg(
        F.sum(
            "ORDER_REVENUE"
        ).cast(
            "decimal(30,2)"
        ).alias(
            "TOTAL_REVENUE"
        ),

        F.count(
            F.lit(1)
        ).alias(
            "ORDER_COUNT"
        ),

        F.countDistinct(
            "USER_ID"
        ).alias(
            "USER_COUNT"
        ),

        F.min(
            "ORDER_DATE"
        ).alias(
            "FIRST_ORDER_DATE"
        ),

        F.max(
            "ORDER_DATE"
        ).alias(
            "LAST_ORDER_DATE"
        )
    )

    .withColumn(
        "ACTIVE_CALENDAR_DAYS",
        (
            F.datediff(
                F.col("LAST_ORDER_DATE"),
                F.col("FIRST_ORDER_DATE")
            )
            +
            F.lit(1)
        ).cast(
            "int"
        )
    )

    .withColumn(
        "FIRST_WEEK_START",
        F.to_date(
            F.date_trunc(
                "week",
                F.col("FIRST_ORDER_DATE")
            )
        )
    )

    .withColumn(
        "LAST_WEEK_START",
        F.to_date(
            F.date_trunc(
                "week",
                F.col("LAST_ORDER_DATE")
            )
        )
    )

    .withColumn(
        "ACTIVE_CALENDAR_WEEKS",
        (
            F.floor(
                F.datediff(
                    F.col("LAST_WEEK_START"),
                    F.col("FIRST_WEEK_START")
                )
                /
                F.lit(7)
            )
            +
            F.lit(1)
        ).cast(
            "int"
        )
    )

    .withColumn(
        "AVERAGE_ORDER_VALUE",
        (
            F.col("TOTAL_REVENUE")
            /
            F.col("ORDER_COUNT")
        ).cast(
            "decimal(30,2)"
        )
    )

    .withColumn(
        "REVENUE_PER_USER",
        F.when(
            F.col("USER_COUNT") > 0,
            (
                F.col("TOTAL_REVENUE")
                /
                F.col("USER_COUNT")
            ).cast(
                "decimal(30,2)"
            )
        )
    )

    .withColumn(
        "AVG_DAILY_ORDERS",
        F.when(
            F.col("ACTIVE_CALENDAR_DAYS") > 0,
            (
                F.col("ORDER_COUNT")
                /
                F.col("ACTIVE_CALENDAR_DAYS")
            ).cast(
                "decimal(18,4)"
            )
        )
    )

    .withColumn(
        "AVG_WEEKLY_ORDERS",
        F.when(
            F.col("ACTIVE_CALENDAR_WEEKS") > 0,
            (
                F.col("ORDER_COUNT")
                /
                F.col("ACTIVE_CALENDAR_WEEKS")
            ).cast(
                "decimal(18,4)"
            )
        )
    )
)

restaurant_rank_window = (
    Window
    .orderBy(
        F.col(
            "TOTAL_REVENUE"
        ).desc()
    )
)

restaurant_performance = (
    restaurant_performance

    .withColumn(
        "REVENUE_RANK",
        F.dense_rank().over(
            restaurant_rank_window
        )
    )
)


# ---------------------------------------------------------
# Restaurant-average validation
# ---------------------------------------------------------

invalid_restaurant_spans = (
    restaurant_performance
    .filter(
        (F.col("ACTIVE_CALENDAR_DAYS") <= 0)
        |
        (F.col("ACTIVE_CALENDAR_WEEKS") <= 0)
    )
)

if invalid_restaurant_spans.limit(1).count() > 0:
    invalid_restaurant_spans.show(
        20,
        truncate=False
    )

    raise RuntimeError(
        "Restaurant performance validation failed: "
        "calendar day/week spans must be positive."
    )


print(
    "RESTAURANT PERFORMANCE VALIDATION PASSED: "
    "AVG_DAILY_ORDERS and AVG_WEEKLY_ORDERS use each "
    "restaurant's first-to-last-order calendar span, "
    "including zero-order periods within that span."
)

write_gold(
    restaurant_performance,
    "restaurant_performance"
)


# =========================================================
# GOLD 7: Discount effectiveness
#
# Current business rule:
#
# If ANY option belonging to the order has OPTION_PRICE = 0,
# classify the order as DISCOUNTED.
#
# This compares revenue/order behavior only.
# =========================================================

discount_effectiveness = (
    order_fact

    .withColumn(
        "DISCOUNT_STATUS",
        F.when(
            F.col(
                "HAS_DISCOUNT"
            ) == 1,
            F.lit(
                "DISCOUNTED"
            )
        )
        .otherwise(
            F.lit(
                "NON_DISCOUNTED"
            )
        )
    )

    .groupBy(
        "DISCOUNT_STATUS"
    )

    .agg(
        F.count(
            F.lit(1)
        ).alias(
            "ORDER_COUNT"
        ),

        F.sum(
            "ORDER_REVENUE"
        ).cast(
            "decimal(30,2)"
        ).alias(
            "TOTAL_REVENUE"
        ),

        F.countDistinct(
            "USER_ID"
        ).alias(
            "USER_COUNT"
        )
    )

    .withColumn(
        "AVERAGE_ORDER_VALUE",
        (
            F.col(
                "TOTAL_REVENUE"
            )
            /
            F.col(
                "ORDER_COUNT"
            )
        ).cast(
            "decimal(30,2)"
        )
    )
)

write_gold(
    discount_effectiveness,
    "discount_effectiveness"
)


# =========================================================
# Success
# =========================================================

print("")
print("=" * 90)
print("GLOBALPARTNERS GOLD JOB SUCCESSFUL")
print("=" * 90)

print(
    f"ANALYSIS_DATE: "
    f"{analysis_date}"
)

print(
    f"ORDER_ITEMS rows: "
    f"{order_items_count}"
)

print(
    f"ORDER_ITEM_OPTIONS rows: "
    f"{order_options_count}"
)

print(
    f"Derived orders: "
    f"{order_fact_count}"
)

print(
    f"Valid USER_ID values: "
    f"{customer_count}"
)

print(
    "Nine current-state Parquet Gold "
    "datasets successfully generated."
)

line_fact.unpersist()
order_fact.unpersist()
order_items.unpersist()
order_options.unpersist()

job.commit()
