# APP_BUILD = "clv-histogram-user-id-count-v9"
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import s3fs
import streamlit as st


# =============================================================================
# App configuration
# =============================================================================

st.set_page_config(
    page_title="Global Partners Customer & Sales Analytics",
    page_icon="📊",
    layout="wide",
)

px.defaults.template = "plotly_white"

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.65rem;
        }
        .topic-label {
            color: #5f6b7a;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.1rem;
        }
        .question-box {
            padding: 0.85rem 1rem;
            border-left: 4px solid #ff9900;
            background: #f7f8fa;
            border-radius: 0.25rem;
            margin-bottom: 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# S3 configuration
# =============================================================================

S3_BUCKET = "nw-globalpartners-project"
GOLD_PREFIX = "gold"
AWS_REGION = "us-east-1"

GOLD_DATASETS = {
    "customer_clv": "customer_clv",
    "customer_rfm": "customer_rfm",
    "churn_indicators": "churn_indicators",
    "sales_trends_daily": "sales_trends_daily",
    "sales_trends_weekly": "sales_trends_weekly",
    "sales_trends_monthly": "sales_trends_monthly",
    "loyalty_program_impact": "loyalty_program_impact",
    "restaurant_performance": "restaurant_performance",
    "discount_effectiveness": "discount_effectiveness",
}


def get_s3_storage_options() -> Optional[Dict]:
    """
    Local:
        If Streamlit secrets are absent, s3fs uses the normal AWS credential
        chain, including credentials configured by the AWS CLI.

    Streamlit Community Cloud:
        Configure an [aws] secrets section with:
            aws_access_key_id
            aws_secret_access_key
            region_name

        An optional aws_session_token is also supported.

    No credentials are stored in this source file.

    SSL certificate verification is intentionally disabled for S3 access
    in this project environment, matching AWS CLI --no-verify-ssl behavior.
    """
    try:
        aws = st.secrets["aws"]
    except Exception:
        return None

    key = aws.get("aws_access_key_id")
    secret = aws.get("aws_secret_access_key")
    token = aws.get("aws_session_token")
    region = aws.get("region_name", AWS_REGION)

    if not key or not secret:
        return None

    options = {
        "key": key,
        "secret": secret,
        "client_kwargs": {"region_name": region, "verify": False},
    }

    if token:
        options["token"] = token

    return options


@st.cache_resource
def get_s3_filesystem() -> s3fs.S3FileSystem:
    options = get_s3_storage_options()
    if options:
        return s3fs.S3FileSystem(**options)

    # Uses the normal AWS credential chain locally.
    return s3fs.S3FileSystem(
        anon=False,
        client_kwargs={"region_name": AWS_REGION, "verify": False},
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_gold_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load every Parquet part file underneath one Gold S3 prefix.

    Gold is current-state Parquet, so a 10-minute Streamlit cache avoids
    repeatedly downloading the same files while navigating the app.
    """
    if dataset_name not in GOLD_DATASETS:
        raise ValueError(f"Unknown Gold dataset: {dataset_name}")

    fs = get_s3_filesystem()

    prefix = (
        f"{S3_BUCKET}/{GOLD_PREFIX}/"
        f"{GOLD_DATASETS[dataset_name]}"
    )

    parquet_files = sorted(
        path
        for path in fs.find(prefix)
        if path.endswith(".parquet")
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found at s3://{prefix}/"
        )

    frames = []
    for path in parquet_files:
        with fs.open(path, "rb") as file_obj:
            frames.append(
                pd.read_parquet(
                    file_obj,
                    engine="pyarrow",
                )
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_or_stop(dataset_name: str) -> pd.DataFrame:
    try:
        return load_gold_dataset(dataset_name)
    except Exception as exc:
        st.error(
            f"Unable to load `{dataset_name}` from S3."
        )
        st.exception(exc)
        st.stop()


# =============================================================================
# Formatting helpers
# =============================================================================

def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )
    return df


def dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )
    return df


def money(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def integer(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{int(round(float(value))):,}"


def percentage(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.1f}%"


def short_restaurant_id(value) -> str:
    """Display a RESTAURANT_ID as ...abc while preserving the full ID in data."""
    if pd.isna(value):
        return "N/A"

    value = str(value)
    if len(value) <= 3:
        return value

    return f"...{value[-3:]}"



def remove_blank_user_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude rows where USER_ID is NULL, empty, or whitespace-only.

    This prevents empty-string USER_ID values from being treated as real
    customers in customer-level analytics.
    """
    if "USER_ID" not in df.columns:
        return df.copy()

    cleaned = df.copy()
    user_text = cleaned["USER_ID"].astype("string")
    valid_user = user_text.notna() & user_text.str.strip().ne("")
    return cleaned.loc[valid_user].copy()



def topic_header(number: int, title: str, question: str) -> None:
    st.markdown(
        f'<div class="topic-label">Topic {number}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<div class="question-box"><strong>Question:</strong> {question}</div>',
        unsafe_allow_html=True,
    )


def plot(fig) -> None:
    fig.update_layout(
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =============================================================================
# Sidebar navigation
# =============================================================================

st.sidebar.title("Global Partners")
st.sidebar.caption(
    "Customer and sales analytics from current-state Gold Parquet data in S3."
)

PAGE_OPTIONS = [
    "Overview",
    "1. Customer Lifetime Value",
    "2. Customer Segmentation & Behavior",
    "3. Churn Indicators",
    "4. Sales Trends Monitoring",
    "5. Loyalty Program Impact",
    "6. Top-Performing Locations",
    "7. Pricing & Discount Effectiveness",
]

page = st.sidebar.radio(
    "Navigate",
    PAGE_OPTIONS,
)

st.sidebar.divider()
st.sidebar.caption(
    "Gold data cache: 10 minutes"
)
st.sidebar.caption(
    f"s3://{S3_BUCKET}/{GOLD_PREFIX}/"
)


# =============================================================================
# Overview
# =============================================================================

if page == "Overview":
    st.title("Global Partners Analytics Overview")
    st.markdown(
        """
        Each summary below maps directly to one of the seven required
        business topics. Use the sidebar for the full analysis.
        """
    )
    st.caption(
        "Customer-level views exclude NULL, empty, and whitespace-only USER_ID values."
    )

    clv = remove_blank_user_ids(
        numeric(
            load_or_stop("customer_clv"),
            ["LIFETIME_REVENUE"],
        )
    )

    rfm = remove_blank_user_ids(
        load_or_stop("customer_rfm")
    )

    churn = remove_blank_user_ids(
        numeric(
            load_or_stop("churn_indicators"),
            ["DAYS_SINCE_LAST_ORDER"],
        )
    )

    monthly = numeric(
        dates(
            load_or_stop("sales_trends_monthly"),
            ["PERIOD_START_DATE"],
        ),
        ["TOTAL_REVENUE"],
    )

    loyalty = numeric(
        load_or_stop("loyalty_program_impact"),
        ["USER_COUNT"],
    )

    restaurants = numeric(
        load_or_stop("restaurant_performance"),
        ["TOTAL_REVENUE", "REVENUE_RANK"],
    )

    discount = numeric(
        load_or_stop("discount_effectiveness"),
        ["ORDER_COUNT"],
    )

    avg_clv = clv["LIFETIME_REVENUE"].mean()

    vip_count = int(
        (rfm["RFM_SEGMENT"] == "VIP").sum()
    )

    at_risk_count = int(
        (churn["CHURN_STATUS"] == "AT_RISK").sum()
    )

    latest_month = monthly["PERIOD_START_DATE"].max()
    latest_month_revenue = (
        monthly.loc[
            monthly["PERIOD_START_DATE"] == latest_month,
            "TOTAL_REVENUE",
        ].sum()
    )

    loyalty_member_users = loyalty.loc[
        loyalty["LOYALTY_STATUS"] == "LOYALTY_MEMBER",
        "USER_COUNT",
    ].sum()
    total_loyalty_users = loyalty["USER_COUNT"].sum()
    loyalty_share = (
        100 * loyalty_member_users / total_loyalty_users
        if total_loyalty_users
        else float("nan")
    )

    if "REVENUE_RANK" in restaurants.columns:
        ranked = restaurants.sort_values(
            ["REVENUE_RANK", "TOTAL_REVENUE"],
            ascending=[True, False],
        )
    else:
        ranked = restaurants.sort_values(
            "TOTAL_REVENUE",
            ascending=False,
        )

    top_restaurant = (
        short_restaurant_id(
            ranked.iloc[0]["RESTAURANT_ID"]
        )
        if not ranked.empty
        else "N/A"
    )

    discounted_orders = discount.loc[
        discount["DISCOUNT_STATUS"] == "DISCOUNTED",
        "ORDER_COUNT",
    ].sum()
    total_orders = discount["ORDER_COUNT"].sum()
    discounted_share = (
        100 * discounted_orders / total_orders
        if total_orders
        else float("nan")
    )

    row1 = st.columns(4)
    row1[0].metric(
        "Topic 1 — Avg historical CLV",
        money(avg_clv),
    )
    row1[1].metric(
        "Topic 2 — VIP USER_IDs",
        integer(vip_count),
    )
    row1[2].metric(
        "Topic 3 — At-risk USER_IDs",
        integer(at_risk_count),
    )
    if "_ANALYSIS_DATE" in monthly.columns:
        overview_analysis_date = pd.to_datetime(
            monthly["_ANALYSIS_DATE"],
            errors="coerce",
        ).max()
    else:
        overview_analysis_date = pd.NaT

    if pd.notna(overview_analysis_date):
        latest_period_label = (
            "Topic 4 — "
            f"{latest_month.strftime('%b %Y')} revenue through "
            f"{overview_analysis_date.strftime('%b %d').replace(' 0', ' ')}"
        )
    else:
        latest_period_label = "Topic 4 — Latest period revenue"

    row1[3].metric(
        latest_period_label,
        money(latest_month_revenue),
    )

    row2 = st.columns(3)
    row2[0].metric(
        "Topic 5 — Loyalty member share",
        percentage(loyalty_share),
    )
    row2[1].metric(
        "Topic 6 — #1 RESTAURANT_ID",
        top_restaurant,
    )
    row2[2].metric(
        "Topic 7 — Discounted order share",
        percentage(discounted_share),
    )

    st.subheader(
        "Topic 4 — How is revenue changing over time across top-performing RESTAURANT_IDs?"
    )

    overview_monthly = monthly.copy()
    overview_monthly["RESTAURANT_ID"] = (
        overview_monthly["RESTAURANT_ID"]
        .astype(str)
    )

    # Revenue is additive across ITEM_CATEGORY, so it is safe to aggregate
    # TOTAL_REVENUE across categories for this overview trend.
    restaurant_totals = (
        overview_monthly
        .groupby(
            "RESTAURANT_ID",
            as_index=False,
        )["TOTAL_REVENUE"]
        .sum()
        .sort_values(
            "TOTAL_REVENUE",
            ascending=False,
        )
    )

    top_5_restaurants = (
        restaurant_totals
        .head(5)["RESTAURANT_ID"]
        .tolist()
    )

    overview_trend = (
        overview_monthly[
            overview_monthly["RESTAURANT_ID"]
            .isin(top_5_restaurants)
        ]
        .groupby(
            [
                "PERIOD_START_DATE",
                "RESTAURANT_ID",
            ],
            as_index=False,
        )["TOTAL_REVENUE"]
        .sum()
        .sort_values(
            "PERIOD_START_DATE"
        )
    )

    overview_trend[
        "RESTAURANT_ID_DISPLAY"
    ] = overview_trend[
        "RESTAURANT_ID"
    ].apply(
        short_restaurant_id
    )

    fig = px.line(
        overview_trend,
        x="PERIOD_START_DATE",
        y="TOTAL_REVENUE",
        color="RESTAURANT_ID_DISPLAY",
        markers=True,
        custom_data=["RESTAURANT_ID"],
        title="Monthly revenue trend — top 5 RESTAURANT_IDs by total revenue",
        labels={
            "PERIOD_START_DATE": "Month",
            "TOTAL_REVENUE": "Revenue",
            "RESTAURANT_ID_DISPLAY": "RESTAURANT_ID",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "Month=%{x|%b %Y}<br>"
            "Revenue=$%{y:,.2f}<br>"
            "RESTAURANT_ID=%{customdata[0]}"
            "<extra></extra>"
        )
    )
    fig.update_yaxes(
        tickprefix="$"
    )
    plot(fig)


# =============================================================================
# Topic 1 — Customer Lifetime Value
# =============================================================================

elif page == "1. Customer Lifetime Value":
    topic_header(
        1,
        "Customer Lifetime Value (CLV)",
        (
            "How much total revenue has each USER_ID generated over their "
            "observed relationship with the business?"
        ),
    )

    st.caption(
        "CLV here is historical observed lifetime revenue, not a prediction "
        "of future customer value. HIGH = top 20%, MEDIUM = middle 60%, "
        "LOW = bottom 20% using quintiles."
    )

    df = remove_blank_user_ids(
        numeric(
            dates(
                load_or_stop("customer_clv"),
                ["FIRST_ORDER_DATE", "LAST_ORDER_DATE"],
            ),
            [
                "LIFETIME_REVENUE",
                "LIFETIME_ORDER_COUNT",
                "LIFETIME_AVG_ORDER_VALUE",
                "CLV_QUINTILE",
            ],
        )
    )

    total_users = df["USER_ID"].nunique()
    avg_lifetime = df["LIFETIME_REVENUE"].mean()
    median_lifetime = df["LIFETIME_REVENUE"].median()
    total_lifetime = df["LIFETIME_REVENUE"].sum()

    clv_p90 = df["LIFETIME_REVENUE"].quantile(0.90)
    clv_p95 = df["LIFETIME_REVENUE"].quantile(0.95)
    clv_p99 = df["LIFETIME_REVENUE"].quantile(0.99)

    cols = st.columns(4)
    cols[0].metric(
        "USER_IDs",
        integer(total_users),
    )
    cols[1].metric(
        "Average historical lifetime revenue",
        money(avg_lifetime),
    )
    cols[2].metric(
        "Median historical lifetime revenue",
        money(median_lifetime),
    )
    cols[3].metric(
        "Total observed lifetime revenue",
        money(total_lifetime),
    )

    percentile_cols = st.columns(3)
    percentile_cols[0].metric(
        "90th percentile historical CLV",
        money(clv_p90),
    )
    percentile_cols[1].metric(
        "95th percentile historical CLV",
        money(clv_p95),
    )
    percentile_cols[2].metric(
        "99th percentile historical CLV",
        money(clv_p99),
    )

    left, right = st.columns(2)

    with left:
        st.subheader(
            "How are USER_IDs grouped into High, Medium, and Low CLV?"
        )
        segment_order = ["HIGH", "MEDIUM", "LOW"]
        segment_counts = (
            df["CLV_SEGMENT"]
            .value_counts()
            .reindex(segment_order)
            .fillna(0)
            .rename_axis("CLV_SEGMENT")
            .reset_index(name="USER_COUNT")
        )
        segment_counts["SHARE_PCT"] = (
            100
            * segment_counts["USER_COUNT"]
            / segment_counts["USER_COUNT"].sum()
        )

        fig = px.bar(
            segment_counts,
            x="CLV_SEGMENT",
            y="USER_COUNT",
            hover_data={"SHARE_PCT": ":.1f"},
            title="CLV segment distribution",
            labels={
                "CLV_SEGMENT": "CLV segment",
                "USER_COUNT": "USER_ID count",
                "SHARE_PCT": "Share (%)",
            },
        )
        plot(fig)

    with right:
        st.subheader(
            "What does the historical lifetime revenue distribution look like?"
        )

        clv_distribution = df[
            df["LIFETIME_REVENUE"] <= clv_p99
        ].copy()

        st.caption(
            "Topic 1 CLV distribution shows the bottom 99% of USER_IDs "
            "so the main customer distribution remains readable. The top "
            "1% is excluded from this chart axis only; all USER_IDs remain "
            "included in the CLV KPIs, segments, percentiles, and Top-10 view."
        )

        fig = px.histogram(
            clv_distribution,
            x="LIFETIME_REVENUE",
            nbins=40,
            title="Historical CLV distribution — bottom 99% of USER_IDs",
            labels={
                "LIFETIME_REVENUE": "Historical lifetime revenue",
                "count": "USER_ID count",
            },
        )
        fig.update_xaxes(tickprefix="$")
        fig.update_yaxes(title_text="USER_ID count")
        plot(fig)

    st.subheader(
        "Which USER_IDs have generated the most historical lifetime revenue?"
    )

    top_users = (
        df.sort_values(
            "LIFETIME_REVENUE",
            ascending=False,
        )
        .head(10)
        .copy()
    )
    top_users["USER_ID"] = top_users["USER_ID"].astype(str)

    fig = px.bar(
        top_users.sort_values("LIFETIME_REVENUE"),
        x="LIFETIME_REVENUE",
        y="USER_ID",
        orientation="h",
        title="Top 10 USER_IDs by historical lifetime revenue",
        labels={
            "LIFETIME_REVENUE": "Lifetime revenue",
            "USER_ID": "USER_ID",
        },
        hover_data=[
            "LIFETIME_ORDER_COUNT",
            "LIFETIME_AVG_ORDER_VALUE",
            "CLV_SEGMENT",
        ],
    )
    fig.update_xaxes(tickprefix="$")
    plot(fig)

    st.dataframe(
        top_users[
            [
                "USER_ID",
                "LIFETIME_REVENUE",
                "LIFETIME_ORDER_COUNT",
                "LIFETIME_AVG_ORDER_VALUE",
                "FIRST_ORDER_DATE",
                "LAST_ORDER_DATE",
                "CLV_QUINTILE",
                "CLV_SEGMENT",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Topic 2 — Customer Segmentation & Behavior
# =============================================================================

elif page == "2. Customer Segmentation & Behavior":
    topic_header(
        2,
        "Customer Segmentation & Behavior",
        (
            "How can USER_IDs be grouped based on spending and activity "
            "to support campaign targeting?"
        ),
    )

    st.caption(
        "RFM window: 6 months ending on the maximum source date. "
        "Recency = days since last purchase; Frequency = orders in the "
        "last 6 months; Monetary = spend in the last 6 months. "
        "R_SCORE, F_SCORE, M_SCORE, and RFM_SEGMENT are read directly "
        "from the Gold dataset. FREQUENCY_6M = 0 receives F_SCORE = 1; "
        "MONETARY_6M = 0 receives M_SCORE = 1. VIP requires R, F, and M "
        "scores of at least 4 plus a combined R+F+M score of at least 14."
    )

    df = remove_blank_user_ids(
        numeric(
            dates(
                load_or_stop("customer_rfm"),
                ["LAST_ORDER_DATE"],
            ),
            [
                "RECENCY_DAYS",
                "FREQUENCY_6M",
                "MONETARY_6M",
                "R_SCORE",
                "F_SCORE",
                "M_SCORE",
            ],
        )
    )

    segment_order = [
        "VIP",
        "NEW_CUSTOMER",
        "CHURN_RISK",
        "OTHER",
    ]

    segment_counts = (
        df["RFM_SEGMENT"]
        .value_counts()
        .reindex(segment_order)
        .fillna(0)
    )

    cols = st.columns(4)
    for col, segment in zip(cols, segment_order):
        col.metric(
            segment.replace("_", " ").title(),
            integer(segment_counts.loc[segment]),
        )

    st.sidebar.subheader("Topic 2 filters")
    chosen_segments = st.sidebar.multiselect(
        "RFM segment",
        segment_order,
        default=segment_order,
        key="rfm_segment_filter",
    )

    filtered = df[
        df["RFM_SEGMENT"].isin(chosen_segments)
    ].copy()

    left, right = st.columns(2)

    with left:
        st.subheader(
            "How many USER_IDs fall into each RFM segment?"
        )
        counts = (
            df["RFM_SEGMENT"]
            .value_counts()
            .reindex(segment_order)
            .fillna(0)
            .rename_axis("RFM_SEGMENT")
            .reset_index(name="USER_COUNT")
        )
        fig = px.bar(
            counts,
            x="RFM_SEGMENT",
            y="USER_COUNT",
            title="RFM segment distribution",
            labels={
                "RFM_SEGMENT": "RFM segment",
                "USER_COUNT": "USER_ID count",
            },
        )
        plot(fig)

    with right:
        st.subheader(
            "How do frequency and monetary value differ across segments?"
        )
        if filtered.empty:
            st.info("Select at least one RFM segment.")
        else:
            fig = px.scatter(
                filtered,
                x="FREQUENCY_6M",
                y="MONETARY_6M",
                color="RFM_SEGMENT",
                hover_name="USER_ID",
                hover_data=[
                    "RECENCY_DAYS",
                    "R_SCORE",
                    "F_SCORE",
                    "M_SCORE",
                ],
                title="Frequency vs monetary value",
                labels={
                    "FREQUENCY_6M": "Orders in last 6 months",
                    "MONETARY_6M": "Spend in last 6 months",
                    "RFM_SEGMENT": "RFM segment",
                },
            )
            fig.update_yaxes(tickprefix="$")
            plot(fig)

    st.subheader(
        "What are the Recency, Frequency, Monetary values and quintile scores for each USER_ID?"
    )

    table = filtered.sort_values(
        ["RFM_SEGMENT", "MONETARY_6M"],
        ascending=[True, False],
    )

    st.dataframe(
        table[
            [
                "USER_ID",
                "LAST_ORDER_DATE",
                "RECENCY_DAYS",
                "FREQUENCY_6M",
                "MONETARY_6M",
                "R_SCORE",
                "F_SCORE",
                "M_SCORE",
                "RFM_SEGMENT",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Topic 3 — Churn Indicators
# =============================================================================

elif page == "3. Churn Indicators":
    topic_header(
        3,
        "Churn Indicators",
        (
            "Which USER_IDs show inactivity or spending behavior that "
            "analysts can use to identify at-risk customers?"
        ),
    )

    st.caption(
        "AT_RISK = more than 45 days since the USER_ID's last order. "
        "Spend change compares the current 30-day period with the previous "
        "30-day period."
    )

    df = remove_blank_user_ids(
        numeric(
            dates(
                load_or_stop("churn_indicators"),
                ["LAST_ORDER_DATE"],
            ),
            [
                "LIFETIME_ORDER_COUNT",
                "AVG_DAYS_BETWEEN_ORDERS",
                "CURRENT_30_DAY_SPEND",
                "PREVIOUS_30_DAY_SPEND",
                "SPEND_CHANGE_PCT",
                "DAYS_SINCE_LAST_ORDER",
            ],
        )
    )

    total_users = len(df)
    at_risk = int(
        (df["CHURN_STATUS"] == "AT_RISK").sum()
    )
    not_at_risk = int(
        (df["CHURN_STATUS"] == "NOT_AT_RISK").sum()
    )
    at_risk_rate = (
        100 * at_risk / total_users
        if total_users
        else float("nan")
    )
    avg_days_since = df["DAYS_SINCE_LAST_ORDER"].mean()

    cols = st.columns(4)
    cols[0].metric(
        "AT_RISK USER_IDs",
        integer(at_risk),
    )
    cols[1].metric(
        "NOT_AT_RISK USER_IDs",
        integer(not_at_risk),
    )
    cols[2].metric(
        "At-risk rate",
        percentage(at_risk_rate),
    )
    cols[3].metric(
        "Average days since last order",
        f"{avg_days_since:,.1f}"
        if not pd.isna(avg_days_since)
        else "N/A",
    )

    st.subheader(
        "How is the customer base split by churn status?"
    )

    churn_counts = (
        df["CHURN_STATUS"]
        .value_counts()
        .rename_axis("CHURN_STATUS")
        .reset_index(name="USER_COUNT")
    )

    fig = px.bar(
        churn_counts,
        x="CHURN_STATUS",
        y="USER_COUNT",
        title="Churn status distribution",
        labels={
            "CHURN_STATUS": "Churn status",
            "USER_COUNT": "USER_ID count",
        },
    )
    plot(fig)

    st.subheader(
        "Which USER_IDs are currently AT_RISK, and what do their activity indicators show?"
    )

    risk_table = (
        df[df["CHURN_STATUS"] == "AT_RISK"]
        .sort_values(
            "DAYS_SINCE_LAST_ORDER",
            ascending=False,
        )
        .copy()
    )

    risk_table["SPEND_CHANGE_PCT"] = risk_table[
        "SPEND_CHANGE_PCT"
    ].apply(
        lambda value: (
            "N/A"
            if pd.isna(value)
            else f"{value:,.2f}%"
        )
    )

    st.dataframe(
        risk_table[
            [
                "USER_ID",
                "LAST_ORDER_DATE",
                "DAYS_SINCE_LAST_ORDER",
                "AVG_DAYS_BETWEEN_ORDERS",
                "CURRENT_30_DAY_SPEND",
                "PREVIOUS_30_DAY_SPEND",
                "SPEND_CHANGE_PCT",
                "LIFETIME_ORDER_COUNT",
                "CHURN_STATUS",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Topic 4 — Sales Trends Monitoring
# =============================================================================

elif page == "4. Sales Trends Monitoring":
    topic_header(
        4,
        "Sales Trends Monitoring",
        (
            "How are revenue and order patterns changing over time, "
            "by RESTAURANT_ID and menu category?"
        ),
    )

    st.caption(
        "Gold formula: AVERAGE_ORDER_VALUE = TOTAL_REVENUE / ORDER_COUNT. "
        "Because the Gold trend tables are grouped by PERIOD_START_DATE + "
        "RESTAURANT_ID + ITEM_CATEGORY, for one selected ITEM_CATEGORY this "
        "means revenue attributed to that category divided by distinct orders "
        "containing that category. ORDER_COUNT is non-additive across categories, "
        "so order count and AOV are shown only when exactly one ITEM_CATEGORY "
        "is selected."
    )

    st.sidebar.subheader("Topic 4 filters")

    grain = st.sidebar.selectbox(
        "Time grain",
        ["Daily", "Weekly", "Monthly"],
    )

    dataset_for_grain = {
        "Daily": "sales_trends_daily",
        "Weekly": "sales_trends_weekly",
        "Monthly": "sales_trends_monthly",
    }

    df = load_or_stop(
        dataset_for_grain[grain]
    )

    df = numeric(
        dates(
            df,
            ["PERIOD_START_DATE"],
        ),
        [
            "TOTAL_REVENUE",
            "ORDER_COUNT",
            "ITEM_QUANTITY",
            "AVERAGE_ORDER_VALUE",
        ],
    )

    restaurant_values = sorted(
        df["RESTAURANT_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    category_values = sorted(
        df["ITEM_CATEGORY"]
        .fillna("UNKNOWN")
        .astype(str)
        .unique()
        .tolist()
    )

    df["RESTAURANT_ID"] = df[
        "RESTAURANT_ID"
    ].astype(str)

    df["ITEM_CATEGORY"] = (
        df["ITEM_CATEGORY"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    selected_restaurants = st.sidebar.multiselect(
        "RESTAURANT_ID",
        restaurant_values,
        default=restaurant_values,
        key=f"sales_restaurants_{grain}",
    )

    selected_categories = st.sidebar.multiselect(
        "ITEM_CATEGORY",
        category_values,
        default=category_values,
        key=f"sales_categories_{grain}",
    )

    min_date = df["PERIOD_START_DATE"].min().date()
    max_date = df["PERIOD_START_DATE"].max().date()

    selected_dates = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"sales_dates_{grain}",
    )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates

    filtered = df[
        df["RESTAURANT_ID"].isin(
            selected_restaurants
        )
        & df["ITEM_CATEGORY"].isin(
            selected_categories
        )
        & (
            df["PERIOD_START_DATE"].dt.date
            >= start_date
        )
        & (
            df["PERIOD_START_DATE"].dt.date
            <= end_date
        )
    ].copy()

    if filtered.empty:
        st.warning(
            "No Gold rows match the current filters."
        )
        st.stop()

    total_revenue = filtered["TOTAL_REVENUE"].sum()
    total_quantity = filtered["ITEM_QUANTITY"].sum()

    # ORDER_COUNT is non-additive across ITEM_CATEGORY because one order can
    # contain multiple categories. It is safe to aggregate only when exactly
    # one category is selected.
    order_count_is_safe = (
        len(selected_categories) == 1
    )

    if order_count_is_safe:
        total_orders = filtered["ORDER_COUNT"].sum()
        aggregate_aov = (
            total_revenue / total_orders
            if total_orders
            else float("nan")
        )
    else:
        total_orders = float("nan")
        aggregate_aov = float("nan")

    cols = st.columns(4)
    cols[0].metric(
        "Revenue",
        money(total_revenue),
    )
    cols[1].metric(
        "Order count",
        integer(total_orders),
    )
    cols[2].metric(
        "Average order value",
        money(aggregate_aov),
    )
    cols[3].metric(
        "Item quantity",
        integer(total_quantity),
    )

    if not order_count_is_safe:
        st.info(
            "Order count and average order value are shown as N/A when "
            "multiple ITEM_CATEGORY values are selected. ORDER_COUNT in "
            "the Gold trend tables is distinct within each category, so "
            "summing it across categories would double-count orders that "
            "contain items from more than one category. Select exactly one "
            "ITEM_CATEGORY to enable those metrics and the order trend."
        )

    st.subheader(
        f"What does {grain.lower()} revenue look like over time?"
    )

    revenue_over_time = (
        filtered.groupby(
            "PERIOD_START_DATE",
            as_index=False,
        )["TOTAL_REVENUE"]
        .sum()
        .sort_values("PERIOD_START_DATE")
    )

    fig = px.line(
        revenue_over_time,
        x="PERIOD_START_DATE",
        y="TOTAL_REVENUE",
        markers=True,
        title=f"{grain} revenue trend",
        labels={
            "PERIOD_START_DATE": "Period",
            "TOTAL_REVENUE": "Revenue",
        },
    )
    fig.update_yaxes(tickprefix="$")
    plot(fig)

    if order_count_is_safe:
        st.subheader(
            f"What does {grain.lower()} order volume look like over time for the selected category?"
        )

        orders_over_time = (
            filtered.groupby(
                "PERIOD_START_DATE",
                as_index=False,
            )["ORDER_COUNT"]
            .sum()
            .sort_values("PERIOD_START_DATE")
        )

        fig = px.line(
            orders_over_time,
            x="PERIOD_START_DATE",
            y="ORDER_COUNT",
            markers=True,
            title=f"{grain} order trend",
            labels={
                "PERIOD_START_DATE": "Period",
                "ORDER_COUNT": "Orders",
            },
        )
        plot(fig)

    left, right = st.columns(2)

    with left:
        st.subheader(
            "Which menu categories generate the most revenue?"
        )

        category_revenue = (
            filtered.groupby(
                "ITEM_CATEGORY",
                as_index=False,
            )["TOTAL_REVENUE"]
            .sum()
            .sort_values(
                "TOTAL_REVENUE",
                ascending=False,
            )
        )

        fig = px.bar(
            category_revenue,
            x="ITEM_CATEGORY",
            y="TOTAL_REVENUE",
            title="Revenue by menu category",
            labels={
                "ITEM_CATEGORY": "Item category",
                "TOTAL_REVENUE": "Revenue",
            },
        )
        fig.update_yaxes(tickprefix="$")
        plot(fig)

    with right:
        st.subheader(
            "Which RESTAURANT_IDs generate the most revenue in the selected period?"
        )

        restaurant_revenue = (
            filtered.groupby(
                "RESTAURANT_ID",
                as_index=False,
            )["TOTAL_REVENUE"]
            .sum()
            .sort_values(
                "TOTAL_REVENUE",
                ascending=False,
            )
            .head(10)
        )

        fig = px.bar(
            restaurant_revenue.sort_values(
                "TOTAL_REVENUE"
            ),
            x="TOTAL_REVENUE",
            y="RESTAURANT_ID",
            orientation="h",
            title="Top RESTAURANT_IDs by revenue",
            labels={
                "TOTAL_REVENUE": "Revenue",
                "RESTAURANT_ID": "RESTAURANT_ID",
            },
        )
        fig.update_xaxes(tickprefix="$")
        plot(fig)


# =============================================================================
# Topic 5 — Loyalty Program Impact
# =============================================================================

elif page == "5. Loyalty Program Impact":
    topic_header(
        5,
        "Loyalty Program Impact",
        (
            "How do loyalty members and non-members compare in spend "
            "and engagement?"
        ),
    )

    st.caption(
        "A USER_ID is considered a loyalty member when that USER_ID's "
        "latest order has IS_LOYALTY = TRUE."
    )

    df = numeric(
        load_or_stop("loyalty_program_impact"),
        [
            "USER_COUNT",
            "ORDER_COUNT",
            "TOTAL_REVENUE",
            "AVG_LIFETIME_REVENUE_PER_USER",
            "REPEAT_USER_COUNT",
            "AVG_ORDER_VALUE",
            "REPEAT_USER_RATE_PCT",
        ],
    )

    total_users = df["USER_COUNT"].sum()
    member_users = df.loc[
        df["LOYALTY_STATUS"] == "LOYALTY_MEMBER",
        "USER_COUNT",
    ].sum()
    member_share = (
        100 * member_users / total_users
        if total_users
        else float("nan")
    )

    st.metric(
        "Loyalty member share of USER_IDs",
        percentage(member_share),
    )

    statuses = [
        status
        for status in [
            "LOYALTY_MEMBER",
            "NON_MEMBER",
        ]
        if status in set(df["LOYALTY_STATUS"])
    ]

    status_cols = st.columns(
        max(len(statuses), 1)
    )

    for col, status in zip(
        status_cols,
        statuses,
    ):
        row = df.loc[
            df["LOYALTY_STATUS"] == status
        ].iloc[0]

        with col:
            st.subheader(
                status.replace("_", " ").title()
            )
            metric_cols = st.columns(3)
            metric_cols[0].metric(
                "USER_IDs",
                integer(row["USER_COUNT"]),
            )
            metric_cols[1].metric(
                "Orders",
                integer(row["ORDER_COUNT"]),
            )
            metric_cols[2].metric(
                "Total revenue",
                money(row["TOTAL_REVENUE"]),
            )

            metric_cols = st.columns(3)
            metric_cols[0].metric(
                "Avg lifetime revenue / USER_ID",
                money(
                    row[
                        "AVG_LIFETIME_REVENUE_PER_USER"
                    ]
                ),
            )
            metric_cols[1].metric(
                "Average order value",
                money(row["AVG_ORDER_VALUE"]),
            )
            metric_cols[2].metric(
                "Repeat USER_ID rate",
                percentage(
                    row["REPEAT_USER_RATE_PCT"]
                ),
            )

    left, right = st.columns(2)

    with left:
        st.subheader(
            "How does average spend compare?"
        )
        spend_compare = df.melt(
            id_vars=["LOYALTY_STATUS"],
            value_vars=[
                "AVG_LIFETIME_REVENUE_PER_USER",
                "AVG_ORDER_VALUE",
            ],
            var_name="METRIC",
            value_name="VALUE",
        )
        spend_compare["METRIC"] = spend_compare[
            "METRIC"
        ].map(
            {
                "AVG_LIFETIME_REVENUE_PER_USER":
                    "Avg lifetime revenue / USER_ID",
                "AVG_ORDER_VALUE":
                    "Average order value",
            }
        )

        fig = px.bar(
            spend_compare,
            x="LOYALTY_STATUS",
            y="VALUE",
            color="METRIC",
            barmode="group",
            title="Average spend comparison",
            labels={
                "LOYALTY_STATUS": "Loyalty status",
                "VALUE": "Revenue",
                "METRIC": "Metric",
            },
        )
        fig.update_yaxes(tickprefix="$")
        plot(fig)

    with right:
        st.subheader(
            "How do repeat orders compare?"
        )
        fig = px.bar(
            df,
            x="LOYALTY_STATUS",
            y="REPEAT_USER_RATE_PCT",
            title="Repeat USER_ID rate",
            labels={
                "LOYALTY_STATUS": "Loyalty status",
                "REPEAT_USER_RATE_PCT": "Repeat USER_ID rate (%)",
            },
        )
        fig.update_yaxes(ticksuffix="%")
        plot(fig)

    st.subheader(
        "How do total revenue and order volume compare?"
    )

    comparison = df[
        [
            "LOYALTY_STATUS",
            "USER_COUNT",
            "ORDER_COUNT",
            "TOTAL_REVENUE",
            "AVG_LIFETIME_REVENUE_PER_USER",
            "AVG_ORDER_VALUE",
            "REPEAT_USER_COUNT",
            "REPEAT_USER_RATE_PCT",
        ]
    ].copy()

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Topic 6 — Top-Performing Locations
# =============================================================================

elif page == "6. Top-Performing Locations":
    topic_header(
        6,
        "Top-Performing Locations",
        (
            "Which RESTAURANT_IDs are the best- and worst-performing "
            "based primarily on total revenue?"
        ),
    )

    df = numeric(
        dates(
            load_or_stop("restaurant_performance"),
            [
                "FIRST_ORDER_DATE",
                "LAST_ORDER_DATE",
            ],
        ),
        [
            "TOTAL_REVENUE",
            "ORDER_COUNT",
            "USER_COUNT",
            "AVERAGE_ORDER_VALUE",
            "REVENUE_PER_USER",
            "AVG_DAILY_ORDERS",
            "AVG_WEEKLY_ORDERS",
            "ACTIVE_CALENDAR_DAYS",
            "ACTIVE_CALENDAR_WEEKS",
            "REVENUE_RANK",
        ],
    )

    st.caption(
        "Topic 6 order-rate formulas: AVG_DAILY_ORDERS = ORDER_COUNT / "
        "ACTIVE_CALENDAR_DAYS and AVG_WEEKLY_ORDERS = ORDER_COUNT / "
        "ACTIVE_CALENDAR_WEEKS. Each denominator spans the restaurant's "
        "first observed order through its last observed order, including "
        "zero-order days/weeks within that span."
    )

    df["RESTAURANT_ID"] = df[
        "RESTAURANT_ID"
    ].astype(str)

    df = df.sort_values(
        ["REVENUE_RANK", "TOTAL_REVENUE"],
        ascending=[True, False],
    )

    st.sidebar.subheader("Topic 6 filters")
    focus_restaurant = st.sidebar.selectbox(
        "Focus RESTAURANT_ID",
        ["All"] + df["RESTAURANT_ID"].tolist(),
        format_func=lambda value: (
            value
            if value == "All"
            else short_restaurant_id(value)
        ),
    )

    top_row = df.iloc[0]

    cols = st.columns(3)
    cols[0].metric(
        "#1 RESTAURANT_ID by revenue",
        short_restaurant_id(
            top_row["RESTAURANT_ID"]
        ),
    )
    cols[1].metric(
        "Restaurants",
        integer(df["RESTAURANT_ID"].nunique()),
    )
    cols[2].metric(
        "Average revenue per restaurant",
        money(df["TOTAL_REVENUE"].mean()),
    )

    if focus_restaurant != "All":
        focus = df.loc[
            df["RESTAURANT_ID"] == focus_restaurant
        ].iloc[0]

        st.subheader(
            "How is RESTAURANT_ID "
            f"{short_restaurant_id(focus_restaurant)} performing?"
        )
        st.caption(
            f"Full RESTAURANT_ID: {focus_restaurant}"
        )

        focus_cols = st.columns(4)
        focus_cols[0].metric(
            "Revenue rank",
            integer(focus["REVENUE_RANK"]),
        )
        focus_cols[1].metric(
            "Total revenue",
            money(focus["TOTAL_REVENUE"]),
        )
        focus_cols[2].metric(
            "Average order value",
            money(focus["AVERAGE_ORDER_VALUE"]),
        )
        focus_cols[3].metric(
            "Revenue per USER_ID",
            money(focus["REVENUE_PER_USER"]),
        )

        focus_cols = st.columns(4)
        focus_cols[0].metric(
            "Orders",
            integer(focus["ORDER_COUNT"]),
        )
        focus_cols[1].metric(
            "USER_IDs",
            integer(focus["USER_COUNT"]),
        )
        focus_cols[2].metric(
            "Average daily orders",
            f"{focus['AVG_DAILY_ORDERS']:,.2f}",
        )
        focus_cols[3].metric(
            "Average weekly orders",
            f"{focus['AVG_WEEKLY_ORDERS']:,.2f}",
        )

    df["RESTAURANT_ID_DISPLAY"] = df[
        "RESTAURANT_ID"
    ].apply(
        short_restaurant_id
    )

    left, right = st.columns(2)

    with left:
        st.subheader(
            "Which RESTAURANT_IDs generate the most revenue?"
        )

        top_10 = (
            df.head(10)
            .sort_values("TOTAL_REVENUE")
        )

        fig = px.bar(
            top_10,
            x="TOTAL_REVENUE",
            y="RESTAURANT_ID_DISPLAY",
            orientation="h",
            custom_data=["RESTAURANT_ID"],
            title="Top 10 RESTAURANT_IDs by revenue",
            labels={
                "TOTAL_REVENUE": "Revenue",
                "RESTAURANT_ID_DISPLAY": "RESTAURANT_ID",
            },
        )
        fig.update_traces(
            hovertemplate=(
                "Revenue=$%{x:,.2f}<br>"
                "RESTAURANT_ID=%{customdata[0]}"
                "<extra></extra>"
            )
        )
        fig.update_xaxes(tickprefix="$")
        plot(fig)

    with right:
        st.subheader(
            "Which RESTAURANT_IDs generate the least revenue?"
        )

        bottom_10 = (
            df.sort_values(
                "TOTAL_REVENUE",
                ascending=True,
            )
            .head(10)
            .sort_values("TOTAL_REVENUE")
        )

        fig = px.bar(
            bottom_10,
            x="TOTAL_REVENUE",
            y="RESTAURANT_ID_DISPLAY",
            orientation="h",
            custom_data=["RESTAURANT_ID"],
            title="Bottom 10 RESTAURANT_IDs by revenue",
            labels={
                "TOTAL_REVENUE": "Revenue",
                "RESTAURANT_ID_DISPLAY": "RESTAURANT_ID",
            },
        )
        fig.update_traces(
            hovertemplate=(
                "Revenue=$%{x:,.2f}<br>"
                "RESTAURANT_ID=%{customdata[0]}"
                "<extra></extra>"
            )
        )
        fig.update_xaxes(tickprefix="$")
        plot(fig)

    st.subheader(
        "How do total revenue and average order value relate across RESTAURANT_IDs?"
    )

    fig = px.scatter(
        df,
        x="AVERAGE_ORDER_VALUE",
        y="TOTAL_REVENUE",
        size="ORDER_COUNT",
        hover_name="RESTAURANT_ID_DISPLAY",
        custom_data=["RESTAURANT_ID"],
        hover_data=[
            "REVENUE_RANK",
            "USER_COUNT",
            "REVENUE_PER_USER",
            "AVG_DAILY_ORDERS",
            "AVG_WEEKLY_ORDERS",
        ],
        title="Restaurant revenue vs average order value",
        labels={
            "AVERAGE_ORDER_VALUE": "Average order value",
            "TOTAL_REVENUE": "Total revenue",
            "ORDER_COUNT": "Order count",
        },
    )
    fig.update_xaxes(tickprefix="$")
    fig.update_yaxes(tickprefix="$")
    plot(fig)

    st.subheader(
        "What are the full revenue, order, USER_ID, and orders-per-day/week metrics for each RESTAURANT_ID?"
    )

    st.dataframe(
        df[
            [
                "REVENUE_RANK",
                "RESTAURANT_ID_DISPLAY",
                "RESTAURANT_ID",
                "TOTAL_REVENUE",
                "AVERAGE_ORDER_VALUE",
                "ORDER_COUNT",
                "FIRST_ORDER_DATE",
                "LAST_ORDER_DATE",
                "ACTIVE_CALENDAR_DAYS",
                "AVG_DAILY_ORDERS",
                "ACTIVE_CALENDAR_WEEKS",
                "AVG_WEEKLY_ORDERS",
                "USER_COUNT",
                "REVENUE_PER_USER",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


# =============================================================================
# Topic 7 — Pricing & Discount Effectiveness
# =============================================================================

elif page == "7. Pricing & Discount Effectiveness":
    topic_header(
        7,
        "Pricing & Discount Effectiveness",
        (
            "How do orders classified as discounted compare with "
            "non-discounted orders in revenue and order behavior?"
        ),
    )

    st.caption(
        "Business rule: an order is classified as DISCOUNTED if any joined "
        "option has OPTION_PRICE = 0. Otherwise it is NON_DISCOUNTED."
    )

    df = numeric(
        load_or_stop("discount_effectiveness"),
        [
            "ORDER_COUNT",
            "TOTAL_REVENUE",
            "USER_COUNT",
            "AVERAGE_ORDER_VALUE",
        ],
    )

    discounted_orders = df.loc[
        df["DISCOUNT_STATUS"] == "DISCOUNTED",
        "ORDER_COUNT",
    ].sum()

    total_orders = df["ORDER_COUNT"].sum()

    discounted_order_share = (
        100 * discounted_orders / total_orders
        if total_orders
        else float("nan")
    )

    cols = st.columns(3)
    cols[0].metric(
        "Discounted orders",
        integer(discounted_orders),
    )
    cols[1].metric(
        "Total orders",
        integer(total_orders),
    )
    cols[2].metric(
        "Discounted order share",
        percentage(discounted_order_share),
    )

    top_left, top_right = st.columns(2)

    with top_left:
        st.subheader(
            "How does revenue compare between discounted and non-discounted orders?"
        )
        fig = px.bar(
            df,
            x="DISCOUNT_STATUS",
            y="TOTAL_REVENUE",
            title="Revenue by discount status",
            labels={
                "DISCOUNT_STATUS": "Discount status",
                "TOTAL_REVENUE": "Revenue",
            },
        )
        fig.update_yaxes(tickprefix="$")
        plot(fig)

    with top_right:
        st.subheader(
            "How does order volume compare between discounted and non-discounted orders?"
        )
        fig = px.bar(
            df,
            x="DISCOUNT_STATUS",
            y="ORDER_COUNT",
            title="Orders by discount status",
            labels={
                "DISCOUNT_STATUS": "Discount status",
                "ORDER_COUNT": "Orders",
            },
        )
        plot(fig)

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.subheader(
            "How does average order value compare?"
        )
        fig = px.bar(
            df,
            x="DISCOUNT_STATUS",
            y="AVERAGE_ORDER_VALUE",
            title="Average order value by discount status",
            labels={
                "DISCOUNT_STATUS": "Discount status",
                "AVERAGE_ORDER_VALUE": "Average order value",
            },
        )
        fig.update_yaxes(tickprefix="$")
        plot(fig)

    with bottom_right:
        st.subheader(
            "How many unique USER_IDs appear in each group?"
        )
        fig = px.bar(
            df,
            x="DISCOUNT_STATUS",
            y="USER_COUNT",
            title="USER_ID count by discount status",
            labels={
                "DISCOUNT_STATUS": "Discount status",
                "USER_COUNT": "USER_ID count",
            },
        )
        plot(fig)

    st.dataframe(
        df[
            [
                "DISCOUNT_STATUS",
                "ORDER_COUNT",
                "TOTAL_REVENUE",
                "AVERAGE_ORDER_VALUE",
                "USER_COUNT",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
