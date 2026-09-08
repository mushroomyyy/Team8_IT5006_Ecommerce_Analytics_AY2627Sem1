from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.set_page_config(
    page_title="Olist E-Commerce EDA Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
CUSTOM_DATA_DIRECTORY = ""
IS_HOSTED_RELEASE = os.getenv("OLIST_HOSTED_RELEASE") == "1"
LOCAL_DATA_DIRECTORIES = (
    APP_DIR / "data" / "Olist_CSV",
    APP_DIR / "Olist_CSV",
    APP_DIR.parent / "data" / "Olist_CSV",
)
HOSTED_DATA_DIRECTORY = (
    Path(tempfile.gettempdir()) / "olist_eda_dashboard" / "v1" / "Olist_CSV"
)
BUNDLED_DATA_ARCHIVE = APP_DIR.parent / "streamlit_release" / "data" / "olist_csv.zip"

REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

BLUE = "#397DA8"
ORANGE = "#E67E22"
NAVY = "#16324F"
TEAL = "#2A9D8F"
LIGHT_BLUE = "#BFD7EA"
PLOT_TEMPLATE = "plotly_white"


st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(180deg, #F7FAFC 0%, #FFFFFF 35%);}
      [data-testid="stSidebar"] {background: #F0F5F9;}
      [data-testid="stMetric"] {
        background: white;
        border: 1px solid #DDE7EF;
        border-radius: 14px;
        min-width: 0;
        padding: 10px 12px;
        box-shadow: 0 3px 12px rgba(22, 50, 79, 0.06);
      }
      [data-testid="stMetricLabel"] p {
        font-size: 0.88rem;
        line-height: 1.2;
      }
      [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        line-height: 1.2;
        white-space: nowrap;
      }
      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.2rem 0 1rem 0;
      }
      .kpi-card {
        min-width: 0;
        background: white;
        border: 1px solid #DDE7EF;
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
        box-shadow: 0 3px 12px rgba(22, 50, 79, 0.06);
      }
      .kpi-label {
        color: #29445D;
        font-size: 0.82rem;
        line-height: 1.2;
        margin-bottom: 0.4rem;
      }
      .kpi-value {
        color: #16324F;
        font-size: clamp(1rem, 1.6vw, 1.6rem);
        font-weight: 500;
        line-height: 1.15;
        white-space: nowrap;
      }
      .kpi-value-compact {
        font-size: clamp(0.95rem, 1.3vw, 1.35rem);
        letter-spacing: -0.015em;
      }
      @media (max-width: 1100px) {
        .kpi-grid {grid-template-columns: repeat(3, minmax(0, 1fr));}
      }
      @media (max-width: 620px) {
        .kpi-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
      }
      .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #16324F 0%, #24577A 65%, #2A9D8F 100%);
        color: white;
        margin-bottom: 1rem;
      }
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {margin: 0.45rem 0 0 0; opacity: 0.92;}
      .callout {
        border-left: 5px solid #E67E22;
        background: #FFF7ED;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0 1rem 0;
      }
      .small-note {color: #526777; font-size: 0.88rem;}
      .kpi-subheader {
        color: #29445D;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.2rem 0 0.6rem 0;
      }
      .kpi-help-wrap {
        position: relative;
        display: inline-flex;
        margin-left: 0.3rem;
        vertical-align: middle;
      }
      .kpi-help-icon {
        display: flex;
        width: 16px;
        height: 16px;
        color: #16324F;
        cursor: pointer;
      }
      .kpi-help-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 135%;
        left: 50%;
        transform: translateX(-50%);
        background: #F0F5F9;
        color: #16324F;
        font-size: 14px;
        font-weight: 400;
        line-height: 1.4;
        text-align: left;
        white-space: normal;
        width: max-content;
        max-width: 220px;
        padding: 6px 12px;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.16);
        transition: opacity 0.15s ease;
        z-index: 999;
      }
      .kpi-help-wrap:hover .kpi-help-tooltip,
      .kpi-help-wrap:focus-within .kpi-help-tooltip {
        visibility: visible;
        opacity: 1;
      }
      div[data-baseweb="tab-list"] {gap: 0.75rem;}
      button[data-baseweb="tab"] {
        font-weight: 600;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def path_from_text(value: str) -> Path:
    """Create a platform-native path and tolerate paths pasted with quotes."""
    cleaned = value.strip().strip('"').strip("'")
    return Path(os.path.expandvars(cleaned)).expanduser()


def first_existing_data_dir() -> Path | None:
    """Use an available local dataset; otherwise allow the hosted fallback."""
    candidates = []
    if os.getenv("OLIST_DATA_DIR"):
        candidates.append(path_from_text(os.environ["OLIST_DATA_DIR"]))
    if CUSTOM_DATA_DIRECTORY:
        candidates.append(path_from_text(CUSTOM_DATA_DIRECTORY))
    candidates.extend(LOCAL_DATA_DIRECTORIES)
    for candidate in candidates:
        if candidate.is_dir() and not missing_files(candidate):
            return candidate
    return None


def missing_files(data_dir: Path) -> list[str]:
    return [filename for filename in REQUIRED_FILES.values() if not (data_dir / filename).is_file()]


@st.cache_resource(show_spinner="Loading Olist data...")
def resolve_data_dir() -> Path:
    """Resolve local data when present, or extract the bundled hosted data once."""
    local_data_dir = first_existing_data_dir()
    if local_data_dir is not None:
        return local_data_dir
    if HOSTED_DATA_DIRECTORY.is_dir() and not missing_files(HOSTED_DATA_DIRECTORY):
        return HOSTED_DATA_DIRECTORY

    if not BUNDLED_DATA_ARCHIVE.is_file():
        raise RuntimeError("The bundled Olist data package is unavailable.")

    HOSTED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLED_DATA_ARCHIVE) as archive:
        archive_members = {
            Path(member).name: member
            for member in archive.namelist()
            if not member.endswith("/")
        }
        unavailable = [
            filename for filename in REQUIRED_FILES.values() if filename not in archive_members
        ]
        if unavailable:
            raise RuntimeError("The bundled data package is incomplete.")
        for filename in REQUIRED_FILES.values():
            with archive.open(archive_members[filename]) as source:
                with (HOSTED_DATA_DIRECTORY / filename).open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    return HOSTED_DATA_DIRECTORY


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0088
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def ordered_join(values) -> str:
    return ", ".join(sorted({str(value) for value in values.dropna()}))


@st.cache_data(show_spinner="Preparing Olist tables and engineered features...")
def prepare_data(data_dir_text: str):
    data_dir = Path(data_dir_text)

    orders = pd.read_csv(
        data_dir / REQUIRED_FILES["orders"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    items = pd.read_csv(
        data_dir / REQUIRED_FILES["order_items"],
        parse_dates=["shipping_limit_date"],
    )
    customers = pd.read_csv(data_dir / REQUIRED_FILES["customers"])
    products = pd.read_csv(data_dir / REQUIRED_FILES["products"])
    sellers = pd.read_csv(data_dir / REQUIRED_FILES["sellers"])
    payments = pd.read_csv(data_dir / REQUIRED_FILES["payments"])
    reviews = pd.read_csv(data_dir / REQUIRED_FILES["reviews"])
    translation = pd.read_csv(data_dir / REQUIRED_FILES["translation"])
    geolocation = pd.read_csv(
        data_dir / REQUIRED_FILES["geolocation"],
        usecols=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
    )

    raw_tables = {
        "Orders": orders,
        "Order items": items,
        "Customers": customers,
        "Products": products,
        "Sellers": sellers,
        "Payments": payments,
        "Reviews": reviews,
        "Geolocation": geolocation,
        "Category translation": translation,
    }
    inventory = pd.DataFrame(
        [
            {
                "Table": name,
                "Rows": len(frame),
                "Columns": frame.shape[1],
                "Missing cells (%)": 100 * frame.isna().sum().sum() / frame.size,
            }
            for name, frame in raw_tables.items()
        ]
    ).sort_values("Rows", ascending=False)

    orphan_checks = {
        "Order items → orders": (~items["order_id"].isin(orders["order_id"])).sum(),
        "Order items → products": (~items["product_id"].isin(products["product_id"])).sum(),
        "Order items → sellers": (~items["seller_id"].isin(sellers["seller_id"])).sum(),
        "Payments → orders": (~payments["order_id"].isin(orders["order_id"])).sum(),
        "Reviews → orders": (~reviews["order_id"].isin(orders["order_id"])).sum(),
        "Orders → customers": (~orders["customer_id"].isin(customers["customer_id"])).sum(),
    }
    items_per_order = items.groupby("order_id").size()
    payments_per_order = payments.groupby("order_id").size()
    reviews_per_order = reviews.groupby("order_id").size()
    joined_order_rows = len(orders[["order_id"]].merge(items[["order_id"]], on="order_id", how="left"))
    quality_checks = pd.DataFrame(
        [
            (
                "Foreign-key orphan rows",
                f"{sum(orphan_checks.values()):,}",
                "Across six tested table relationships",
            ),
            (
                "Order-to-item join fan-out",
                f"{joined_order_rows / len(orders):.2f}×",
                "Order counts must use unique order_id",
            ),
            (
                "Items per order (median / maximum)",
                f"{items_per_order.median():.0f} / {items_per_order.max():,}",
                "Joining items multiplies order-level rows",
            ),
            (
                "Payments per order (median / maximum)",
                f"{payments_per_order.median():.0f} / {payments_per_order.max():,}",
                "Payments are aggregated before joining",
            ),
            (
                "Reviews per order (median / maximum)",
                f"{reviews_per_order.median():.0f} / {reviews_per_order.max():,}",
                "Reviews are aggregated before joining",
            ),
            (
                "Orders without line items",
                f"{(~orders['order_id'].isin(items['order_id'])).sum():,}",
                "Mostly cancelled or unavailable orders",
            ),
        ],
        columns=["Validation check", "Result", "Modelling implication"],
    )

    category_map = translation.set_index("product_category_name")[
        "product_category_name_english"
    ]
    products = products.copy()
    products["category"] = (
        products["product_category_name"].map(category_map)
        .combine_first(products["product_category_name"])
        .fillna("Unknown")
    )
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    item_detail = (
        items.merge(
            products[["product_id", "category", "product_weight_g", "product_volume_cm3"]],
            on="product_id",
            how="left",
        )
        .merge(
            sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]],
            on="seller_id",
            how="left",
        )
        .sort_values(["order_id", "order_item_id"])
    )
    item_detail["category"] = item_detail["category"].fillna("Unknown")

    item_agg = (
        item_detail.groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            seller_count=("seller_id", "nunique"),
            order_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            total_weight_g=("product_weight_g", "sum"),
            total_volume_cm3=("product_volume_cm3", "sum"),
            primary_category=("category", "first"),
            primary_seller_state=("seller_state", "first"),
            seller_states=("seller_state", ordered_join),
        )
    )
    item_agg["freight_ratio"] = item_agg["freight_value"] / item_agg["order_value"].replace(0, np.nan)

    customer_detail = customers[[
        "customer_id",
        "customer_unique_id",
        "customer_state",
        "customer_city",
        "customer_zip_code_prefix",
    ]]

    order_customer = orders[["order_id", "customer_id"]].merge(
        customer_detail, on="customer_id", how="left"
    )
    route_pairs = (
        item_detail[["order_id", "seller_id", "seller_state"]]
        .drop_duplicates()
        .merge(order_customer[["order_id", "customer_state"]], on="order_id", how="left")
    )
    route_pairs["same_state"] = route_pairs["seller_state"].eq(route_pairs["customer_state"])
    route_agg = (
        route_pairs.groupby("order_id", as_index=False)
        .agg(route_legs=("seller_id", "count"), same_state_legs=("same_state", "sum"))
    )
    route_agg["route_type"] = np.select(
        [
            route_agg["same_state_legs"].eq(route_agg["route_legs"]),
            route_agg["same_state_legs"].eq(0),
        ],
        ["All same-state", "All interstate"],
        default="Mixed",
    )

    valid_geo = geolocation[
        geolocation["geolocation_lat"].between(-34, 6)
        & geolocation["geolocation_lng"].between(-74, -34)
    ]
    geo_by_zip = (
        valid_geo.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(latitude=("geolocation_lat", "median"), longitude=("geolocation_lng", "median"))
    )
    customer_coords = customer_detail[["customer_id", "customer_zip_code_prefix"]].merge(
        geo_by_zip.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "latitude": "customer_latitude",
                "longitude": "customer_longitude",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    seller_coords = sellers[["seller_id", "seller_zip_code_prefix"]].merge(
        geo_by_zip.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "latitude": "seller_latitude",
                "longitude": "seller_longitude",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )
    distance_pairs = (
        items[["order_id", "seller_id"]]
        .drop_duplicates()
        .merge(seller_coords, on="seller_id", how="left")
        .merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .merge(customer_coords, on="customer_id", how="left")
    )
    distance_pairs["distance_km"] = haversine_km(
        distance_pairs["seller_latitude"],
        distance_pairs["seller_longitude"],
        distance_pairs["customer_latitude"],
        distance_pairs["customer_longitude"],
    )
    distance_agg = (
        distance_pairs.groupby("order_id", as_index=False)
        .agg(
            max_distance_km=("distance_km", "max"),
            mean_distance_km=("distance_km", "mean"),
        )
    )

    payment_first = (
        payments.sort_values(["order_id", "payment_sequential"])
        .groupby("order_id", as_index=False)
        .agg(primary_payment_type=("payment_type", "first"), payment_value=("payment_value", "sum"))
    )
    review_agg = (
        reviews.groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "mean"),
            has_review_comment=("review_comment_message", lambda values: values.notna().any()),
        )
    )

    order_df = (
        orders.merge(customer_detail, on="customer_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(route_agg[["order_id", "route_type"]], on="order_id", how="left")
        .merge(distance_agg, on="order_id", how="left")
        .merge(payment_first, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="left")
    )
    order_df["purchase_date"] = order_df["order_purchase_timestamp"].dt.normalize()
    order_df["purchase_month"] = order_df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    order_df["delivery_deviation_days"] = (
        order_df["order_delivered_customer_date"].dt.normalize()
        - order_df["order_estimated_delivery_date"].dt.normalize()
    ).dt.days
    order_df["delivery_lead_days"] = (
        order_df["order_delivered_customer_date"] - order_df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    order_df["estimated_window_days"] = (
        order_df["order_estimated_delivery_date"] - order_df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    order_df["purchase_to_approval_days"] = (
        order_df["order_approved_at"].dt.normalize()
        - order_df["order_purchase_timestamp"].dt.normalize()
    ).dt.days
    order_df["approval_to_carrier_days"] = (
        order_df["order_delivered_carrier_date"].dt.normalize()
        - order_df["order_approved_at"].dt.normalize()
    ).dt.days
    order_df["carrier_to_customer_days"] = (
        order_df["order_delivered_customer_date"].dt.normalize()
        - order_df["order_delivered_carrier_date"].dt.normalize()
    ).dt.days
    order_df["valid_delivery_target"] = (
        order_df["order_status"].eq("delivered")
        & order_df["delivery_deviation_days"].notna()
    )
    order_df["late"] = np.where(
        order_df["valid_delivery_target"],
        order_df["delivery_deviation_days"].gt(0).astype(int),
        np.nan,
    )
    order_df["review_group"] = pd.cut(
        order_df["review_score"],
        bins=[0, 2, 4, 5],
        labels=["Low (1-2)", "Mid (3-4)", "High (5)"],
    )

    # Leakage-safe seller history: only deliveries completed before the next purchase.
    seller_outcomes = (
        items[["order_id", "seller_id"]]
        .drop_duplicates()
        .merge(
            orders[[
                "order_id",
                "order_status",
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]],
            on="order_id",
            how="left",
        )
    )
    seller_outcomes["delivery_deviation_days"] = (
        seller_outcomes["order_delivered_customer_date"].dt.normalize()
        - seller_outcomes["order_estimated_delivery_date"].dt.normalize()
    ).dt.days
    seller_outcomes["late"] = seller_outcomes["delivery_deviation_days"].gt(0).astype(int)
    seller_outcomes = seller_outcomes[
        seller_outcomes["order_status"].eq("delivered")
        & seller_outcomes["order_purchase_timestamp"].notna()
        & seller_outcomes["order_delivered_customer_date"].notna()
        & seller_outcomes["delivery_deviation_days"].notna()
    ].copy()

    completion_events = (
        seller_outcomes.groupby(["seller_id", "order_delivered_customer_date"], as_index=False)
        .agg(
            completed_at_event=("order_id", "nunique"),
            late_at_event=("late", "sum"),
            deviation_at_event=("delivery_deviation_days", "sum"),
        )
        .sort_values(["seller_id", "order_delivered_customer_date"])
    )
    completion_events["prior_completed_orders"] = completion_events.groupby("seller_id")[
        "completed_at_event"
    ].cumsum()
    completion_events["prior_late_orders"] = completion_events.groupby("seller_id")[
        "late_at_event"
    ].cumsum()
    completion_events["prior_deviation_sum"] = completion_events.groupby("seller_id")[
        "deviation_at_event"
    ].cumsum()
    completion_events["prior_late_rate"] = (
        completion_events["prior_late_orders"] / completion_events["prior_completed_orders"]
    )
    completion_events["prior_mean_deviation"] = (
        completion_events["prior_deviation_sum"] / completion_events["prior_completed_orders"]
    )
    history_events = completion_events[[
        "seller_id",
        "order_delivered_customer_date",
        "prior_completed_orders",
        "prior_late_rate",
        "prior_mean_deviation",
    ]].rename(columns={"order_delivered_customer_date": "history_available_timestamp"})

    current_orders = seller_outcomes[[
        "order_id",
        "seller_id",
        "order_purchase_timestamp",
        "delivery_deviation_days",
        "late",
    ]].sort_values(["order_purchase_timestamp", "seller_id"])
    history_events = history_events.sort_values(["history_available_timestamp", "seller_id"])
    seller_history = pd.merge_asof(
        current_orders,
        history_events,
        by="seller_id",
        left_on="order_purchase_timestamp",
        right_on="history_available_timestamp",
        direction="backward",
        allow_exact_matches=False,
    )
    seller_history["prior_completed_orders"] = seller_history["prior_completed_orders"].fillna(0).astype(int)
    seller_history["new_seller"] = seller_history["prior_completed_orders"].eq(0)

    order_history = (
        seller_history.groupby("order_id", as_index=False)
        .agg(
            minimum_seller_history=("prior_completed_orders", "min"),
            maximum_prior_late_rate=("prior_late_rate", "max"),
            maximum_prior_mean_deviation=("prior_mean_deviation", "max"),
            new_seller_count=("new_seller", "sum"),
        )
    )
    order_df = order_df.merge(order_history, on="order_id", how="left")

    customer_months = (
        order_df[["customer_unique_id", "purchase_month"]]
        .dropna()
        .drop_duplicates()
    )
    repeat_origin_months = pd.concat(
        [
            customer_months.assign(
                purchase_month=customer_months["purchase_month"] - pd.DateOffset(months=offset)
            )
            for offset in (1, 2, 3)
        ],
        ignore_index=True,
    ).drop_duplicates()
    repeat_origin_months["repeat_next_3m"] = 1
    customer_months = customer_months.merge(
        repeat_origin_months,
        on=["customer_unique_id", "purchase_month"],
        how="left",
    )
    customer_months["repeat_next_3m"] = customer_months["repeat_next_3m"].fillna(0)
    repeat_monthly = (
        customer_months.groupby("purchase_month", as_index=False)
        .agg(
            customers=("customer_unique_id", "nunique"),
            repeat_customers=("repeat_next_3m", "sum"),
            repeat_rate=("repeat_next_3m", "mean"),
        )
    )
    repeat_monthly["repeat_rate_pct"] = 100 * repeat_monthly["repeat_rate"]

    return order_df, seller_history, inventory, quality_checks, repeat_monthly


def chart_style(fig, height=440):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=65, b=25),
        font=dict(family="Arial", color=NAVY),
        title_font=dict(size=19, color=NAVY),
        legend_title_text="",
    )
    return fig


def empty_state(message: str):
    st.info(message)


st.markdown(
    """
    <div class="hero">
      <h1>IT5006 Olist E-Commerce EDA Dashboard (Team 8)</h1>
      <p>Interactive exploratory data analysis of orders, delivery performance, geography, sellers and customer experience.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if IS_HOSTED_RELEASE:
    try:
        data_dir = resolve_data_dir()
        order_df, seller_history, inventory, quality_checks, repeat_monthly = prepare_data(
            str(data_dir.resolve())
        )
    except Exception:
        st.error("The dashboard data could not be loaded. Please try again later.")
        st.stop()
else:
    with st.sidebar:
        st.header("Data and filters")
        initial_dir = first_existing_data_dir() or resolve_data_dir()
        with st.expander("Data location", expanded=False):
            st.caption(
                "The bundled dataset is used automatically. To use another copy, paste "
                "the folder containing all nine CSV files here."
            )
            data_dir_text = st.text_input(
                "Folder containing the nine Olist CSV files",
                value=str(initial_dir),
            )

    data_dir = path_from_text(data_dir_text)
    missing = missing_files(data_dir)
    if missing:
        st.error(
            f"The selected data folder `{data_dir}` is missing {len(missing)} required file(s)."
        )
        st.code("\n".join(missing), language="text")
        st.markdown(
            "Place the files in `data/Olist_CSV/`, paste the correct folder under "
            "**Data location**, or change `CUSTOM_DATA_DIRECTORY` near the top of `app.py`."
        )
        st.stop()

    try:
        order_df, seller_history, inventory, quality_checks, repeat_monthly = prepare_data(
            str(data_dir.resolve())
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()


with st.sidebar:
    if IS_HOSTED_RELEASE:
        st.header("Filters")
    min_date = order_df["purchase_date"].min().date()
    max_date = order_df["purchase_date"].max().date()
    analysis_start = max(min_date, pd.Timestamp("2017-01-01").date())
    analysis_end = min(max_date, pd.Timestamp("2018-08-31").date())
    selected_dates = st.date_input(
        "Date range",
        value=(analysis_start, analysis_end),
        min_value=min_date,
        max_value=max_date,
        disabled=st.session_state.get("use_complete_window", True),
    )
    use_complete_window = st.toggle(
        "Use recommended date range of Jan 2017 to Aug 2018",
        value=True,
        key="use_complete_window",
        help=(
            "Records before January 2017 and after August 2018 are sparse and incomplete. "
            "While we have made all dates available for analysis, preliminary observations indicate "
            "that analysis beyond the recommended date range is potentially misleading."
        ),
    )

    if use_complete_window:
        selected_start, selected_end = analysis_start, analysis_end
    elif len(selected_dates) == 2:
        selected_start, selected_end = selected_dates
    else:
        selected_start = selected_end = selected_dates[0]

    customer_states = st.multiselect(
        "Customer states",
        sorted(order_df["customer_state"].dropna().unique()),
        placeholder="All states",
    )
    seller_states = st.multiselect(
        "Primary seller states",
        sorted(order_df["primary_seller_state"].dropna().unique()),
        placeholder="All states",
    )
    route_types = st.multiselect(
        "Route types",
        ["All same-state", "All interstate", "Mixed"],
        placeholder="All route types",
    )
    categories = st.multiselect(
        "Primary product categories",
        sorted(order_df["primary_category"].dropna().unique()),
        placeholder="All categories",
    )

    distance_max = int(np.ceil(order_df["max_distance_km"].quantile(0.995) / 100) * 100)
    distance_range = st.slider(
        "Seller-customer distance (km)",
        min_value=0,
        max_value=max(distance_max, 100),
        value=(0, max(distance_max, 100)),
        step=50,
    )
    include_missing_distance = st.checkbox("Include orders without coordinates", value=True)
    order_scope = st.radio(
        "Order scope",
        ["All orders", "Delivered orders only"],
        horizontal=False,
    )


filtered = order_df[
    order_df["purchase_date"].between(pd.Timestamp(selected_start), pd.Timestamp(selected_end))
].copy()
if customer_states:
    filtered = filtered[filtered["customer_state"].isin(customer_states)]
if seller_states:
    filtered = filtered[filtered["primary_seller_state"].isin(seller_states)]
if route_types:
    filtered = filtered[filtered["route_type"].isin(route_types)]
if categories:
    filtered = filtered[filtered["primary_category"].isin(categories)]

distance_mask = filtered["max_distance_km"].between(*distance_range)
if include_missing_distance:
    distance_mask |= filtered["max_distance_km"].isna()
filtered = filtered[distance_mask]
if order_scope == "Delivered orders only":
    filtered = filtered[filtered["order_status"].eq("delivered")]

delivery_filtered = filtered[filtered["valid_delivery_target"]].copy()
review_filtered = filtered[filtered["review_score"].notna()].copy()
history_filtered = seller_history[seller_history["order_id"].isin(filtered["order_id"])].copy()

with st.sidebar:
    st.divider()
    st.caption(f"{len(filtered):,} orders match the current filters")
    if not IS_HOSTED_RELEASE and st.button("Clear cached data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


if filtered.empty:
    st.warning("No orders match the selected filters. Widen the sidebar selections.")
    st.stop()


metrics = [
    ("Total Orders", f"{filtered['order_id'].nunique():,}"),
    ("Total Customers", f"{filtered['customer_unique_id'].nunique():,}"),
    ("Total Order Value", f"R$ {filtered['order_value'].sum():,.0f}"),
    (
        "Late-delivery Rate",
        f"{delivery_filtered['late'].mean():.1%}" if not delivery_filtered.empty else "N/A",
    ),
    (
        "Median Delivery Deviation",
        f"{delivery_filtered['delivery_deviation_days'].median():.0f} days"
        if not delivery_filtered.empty
        else "N/A",
    ),
    (
        "Average Review Score",
        f"{review_filtered['review_score'].mean():.2f} / 5"
        if not review_filtered.empty
        else "N/A",
    ),
]
KPI_HELP = {
    "Median Delivery Deviation": (
        "How many days early or late a typical delivered order arrives, compared to the "
        "estimated delivery date. Negative means early, positive means late."
    ),
}
KPI_HELP_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" class="kpi-help-icon"><circle cx="12" cy="12" r="10"></circle>'
    '<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>'
    '<line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
)
metric_cards = "".join(
    f'<div class="kpi-card"><div class="kpi-label">{label}'
    + (
        f'<span class="kpi-help-wrap">{KPI_HELP_ICON}'
        f'<span class="kpi-help-tooltip">{KPI_HELP[label]}</span></span>'
        if label in KPI_HELP
        else ""
    )
    + "</div>"
    f'<div class="kpi-value{" kpi-value-compact" if label == "Order value" else ""}">'
    f'{value}</div></div>'
    for label, value in metrics
)
st.markdown(
    '<div class="kpi-subheader">Totals for orders matching the current filters</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="kpi-grid">{metric_cards}</div>', unsafe_allow_html=True)


overview_tab, delivery_tab, geography_tab, seller_tab, review_tab, data_tab = st.tabs(
    [
        "Executive overview",
        "Delivery promises",
        "Geography",
        "Seller reliability",
        "Customer behavior",
        "Data quality",
    ]
)


with overview_tab:
    monthly = (
        filtered.groupby("purchase_month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            revenue=("order_value", "sum"),
            average_order_value=("order_value", "mean"),
        )
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["purchase_month"],
            y=monthly["revenue"],
            name="Revenue",
            marker_color=BLUE,
            customdata=monthly[["orders"]],
            hovertemplate="Revenue: R$ %{y:,.0f}<br>Orders: %{customdata[0]:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["purchase_month"],
            y=monthly["average_order_value"],
            name="Average order value",
            mode="lines+markers",
            line=dict(color=ORANGE, width=3),
            hovertemplate="Average order value: R$ %{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Revenue (R$)", secondary_y=False)
    fig.update_yaxes(title_text="Average order value (R$)", secondary_y=True)
    fig.update_layout(title="Monthly Revenue and Average Order Value")
    st.plotly_chart(chart_style(fig), use_container_width=True)

    top_categories = (
        filtered.groupby("primary_category")["item_count"].sum().nlargest(6).index
    )
    category_monthly = (
        filtered[filtered["primary_category"].isin(top_categories)]
        .groupby(["purchase_month", "primary_category"], as_index=False)
        .agg(items_sold=("item_count", "sum"))
    )
    fig = px.line(
        category_monthly,
        x="purchase_month",
        y="items_sold",
        color="primary_category",
        markers=True,
        title="Monthly Items Sold for Leading Product Categories",
        labels={
            "purchase_month": "Purchase month",
            "items_sold": "Items sold",
            "primary_category": "Category",
        },
    )
    st.plotly_chart(chart_style(fig, 480), use_container_width=True)

    categories_summary = (
        filtered.groupby("primary_category", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            item_quantity=("item_count", "sum"),
            order_value=("order_value", "sum"),
        )
    )
    left, right = st.columns(2)
    items_ranked = categories_summary.nlargest(12, "item_quantity").sort_values("item_quantity")
    fig = px.bar(
        items_ranked,
        x="item_quantity",
        y="primary_category",
        orientation="h",
        color_discrete_sequence=[TEAL],
        labels={
            "item_quantity": "Items sold",
            "primary_category": "Category",
        },
        title="Top Categories by Items Sold",
        hover_data=["orders", "item_quantity", "order_value"],
    )
    left.plotly_chart(chart_style(fig), use_container_width=True)

    value_ranked = categories_summary.nlargest(12, "order_value").sort_values("order_value")
    fig = px.bar(
        value_ranked,
        x="order_value",
        y="primary_category",
        orientation="h",
        color_discrete_sequence=[TEAL],
        labels={
            "order_value": "Order value (R$)",
            "primary_category": "Category",
        },
        title="Top Categories by Order Value",
        hover_data=["orders", "item_quantity", "order_value"],
    )
    right.plotly_chart(chart_style(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    status_summary = filtered["order_status"].value_counts().rename_axis("status").reset_index(name="orders")
    status_summary["share_pct"] = 100 * status_summary["orders"] / status_summary["orders"].sum()
    status_summary = status_summary.sort_values("share_pct")
    fig = px.bar(
        status_summary,
        x="share_pct",
        y="status",
        orientation="h",
        text=status_summary["share_pct"].map(lambda value: f"{value:.1f}%"),
        title="Order Status Distribution",
        color_discrete_sequence=[BLUE],
        labels={"share_pct": "Share of orders (%)", "status": "Order status"},
        hover_data={"orders": ":,", "share_pct": ":.1f"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 110])
    c1.plotly_chart(chart_style(fig, 390), use_container_width=True)

    weekday = filtered.assign(
        weekday=filtered["order_purchase_timestamp"].dt.day_name(),
        hour=filtered["order_purchase_timestamp"].dt.hour,
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = weekday.pivot_table(
        index="weekday", columns="hour", values="order_id", aggfunc="nunique", fill_value=0
    ).reindex(weekday_order)
    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale=[[0, "#EDF4F8"], [0.5, BLUE], [1, NAVY]],
        labels=dict(x="Hour", y="Weekday", color="Orders"),
        title="Number of Orders by Weekday and Hour",
    )
    c2.plotly_chart(chart_style(fig, 390), use_container_width=True)


with delivery_tab:
    if delivery_filtered.empty:
        empty_state("No delivered orders with valid promise dates match the filters.")
    else:
        c1, c2 = st.columns([1.25, 0.75])
        low_clip = delivery_filtered["delivery_deviation_days"].quantile(0.01)
        high_clip = delivery_filtered["delivery_deviation_days"].quantile(0.99)
        deviation_plot = delivery_filtered[
            delivery_filtered["delivery_deviation_days"].between(low_clip, high_clip)
        ]
        fig = px.histogram(
            deviation_plot,
            x="delivery_deviation_days",
            nbins=55,
            color_discrete_sequence=[BLUE],
            title="Delivery Deviation Distribution (1st-99th Percentile)",
            labels={"delivery_deviation_days": "Days early (-) / late (+)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color=ORANGE, annotation_text="Promised date")
        c1.plotly_chart(chart_style(fig), use_container_width=True)

        late_mix = pd.DataFrame(
            {
                "Delivery status": ["By promised date", "Late"],
                "Orders": [delivery_filtered["late"].eq(0).sum(), delivery_filtered["late"].eq(1).sum()],
            }
        )
        late_mix["share_pct"] = 100 * late_mix["Orders"] / late_mix["Orders"].sum()
        fig = px.bar(
            late_mix,
            x="Delivery status",
            y="share_pct",
            text=late_mix["share_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Delivery Promise Outcome Distribution",
            labels={"share_pct": "Share of delivered orders (%)"},
            hover_data={"Orders": ":,", "share_pct": ":.1f"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, 105])
        c2.plotly_chart(chart_style(fig), use_container_width=True)
        st.caption(
            "Delivery deviation is the difference between the actual delivery date and estimated delivery date. "
            "Negative values indicate early delivery, while positive values indicate late delivery."
        )

        route_order = ["All same-state", "All interstate", "Mixed"]
        route_summary = (
            delivery_filtered.dropna(subset=["route_type"])
            .groupby("route_type", as_index=False)
            .agg(
                delivered_orders=("order_id", "nunique"),
                late_rate=("late", "mean"),
            )
        )
        late_route_summary = (
            delivery_filtered[delivery_filtered["late"].eq(1)]
            .dropna(subset=["route_type"])
            .groupby("route_type", as_index=False)
            .agg(median_days_late=("delivery_deviation_days", "median"))
        )
        route_summary = route_summary.merge(late_route_summary, on="route_type", how="left")
        route_summary["late_rate_pct"] = 100 * route_summary["late_rate"]
        route_summary["route_type"] = pd.Categorical(
            route_summary["route_type"], categories=route_order, ordered=True
        )
        route_summary = route_summary.sort_values("route_type")
        route_summary["route_label"] = route_summary["route_type"].map(
            {
                "All same-state": "Same-state",
                "All interstate": "Interstate",
                "Mixed": "Mixed seller routes",
            }
        )
        fig = px.bar(
            route_summary,
            x="route_label",
            y="late_rate_pct",
            text=route_summary["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Late-delivery Rate by Route Type",
            labels={"route_label": "Route type", "late_rate_pct": "Late rate (%)"},
            hover_data={
                "delivered_orders": ":,",
                "median_days_late": ":.1f",
                "route_type": False,
            },
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        if not route_summary.empty:
            fig.update_yaxes(range=[0, route_summary["late_rate_pct"].max() * 1.18])
        st.plotly_chart(chart_style(fig, 440), use_container_width=True)
        st.caption(
            "Same-state routes are those with sellers located entirely within the customer’s state. "
            "Interstate routes are those with sellers entirely outside the customer's state. "
            "Mixed orders contain both."
        )

        monthly_delivery = (
            delivery_filtered.groupby("purchase_month", as_index=False)
            .agg(
                delivered_orders=("order_id", "nunique"),
                late_rate=("late", "mean"),
            )
        )
        monthly_delivery["late_rate_pct"] = 100 * monthly_delivery["late_rate"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=monthly_delivery["purchase_month"],
                y=monthly_delivery["delivered_orders"],
                name="Delivered orders",
                marker_color=BLUE,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_delivery["purchase_month"],
                y=monthly_delivery["late_rate_pct"],
                name="Late-delivery rate",
                mode="lines+markers",
                line=dict(color=ORANGE, width=3),
                hovertemplate="Late-delivery rate: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Delivered orders", secondary_y=False)
        fig.update_yaxes(title_text="Late-delivery rate (%)", secondary_y=True)
        fig.update_layout(title="Monthly Delivered Orders and Late-delivery Rate")
        st.plotly_chart(chart_style(fig, 460), use_container_width=True)

        stage_columns = [
            "purchase_to_approval_days",
            "approval_to_carrier_days",
            "carrier_to_customer_days",
        ]
        stage_data = delivery_filtered.dropna(subset=stage_columns).copy()
        stage_data = stage_data[(stage_data[stage_columns] >= 0).all(axis=1)]
        stage_data["total_stage_days"] = stage_data[stage_columns].sum(axis=1)
        stage_data = stage_data[stage_data["total_stage_days"].gt(0)]
        for stage in stage_columns:
            stage_data[stage] = 100 * stage_data[stage] / stage_data["total_stage_days"]
        stage_summary = (
            stage_data.assign(
                delivery_outcome=np.where(stage_data["late"].eq(1), "Late", "By promised date")
            )
            .groupby("delivery_outcome", as_index=False)[stage_columns]
            .mean()
            .melt(
                id_vars="delivery_outcome",
                var_name="delivery_stage",
                value_name="share_pct",
            )
        )
        stage_summary["delivery_stage"] = stage_summary["delivery_stage"].map(
            {
                "purchase_to_approval_days": "Purchase to approval",
                "approval_to_carrier_days": "Approval to carrier",
                "carrier_to_customer_days": "Carrier to customer",
            }
        )

        window_labels = ["0–10", "11–20", "21–30", "31–40", "41+"]
        window_data = delivery_filtered.copy()
        window_data["estimated_window_band"] = pd.cut(
            window_data["estimated_window_days"],
            bins=[0, 10, 20, 30, 40, np.inf],
            labels=window_labels,
            include_lowest=True,
        )
        window_summary = (
            window_data.groupby("estimated_window_band", observed=True)
            .agg(
                delivered_orders=("order_id", "nunique"),
                late_rate=("late", "mean"),
            )
            .reset_index()
        )
        window_summary["late_rate_pct"] = 100 * window_summary["late_rate"]

        c1, c2 = st.columns(2)
        fig = px.bar(
            stage_summary,
            x="delivery_outcome",
            y="share_pct",
            color="delivery_stage",
            barmode="stack",
            text=stage_summary["share_pct"].map(lambda value: f"{value:.0f}%"),
            color_discrete_sequence=[LIGHT_BLUE, ORANGE, BLUE],
            title="Delivery Time by Stage: On-time versus Late",
            labels={
                "delivery_outcome": "Delivery outcome",
                "share_pct": "Average share of delivery time (%)",
                "delivery_stage": "Delivery stage",
            },
        )
        fig.update_traces(textposition="inside")
        c1.plotly_chart(chart_style(fig, 480), use_container_width=True)

        fig = px.bar(
            window_summary,
            x="estimated_window_band",
            y="late_rate_pct",
            text=window_summary["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Late-delivery Rate by Estimated Delivery Window",
            labels={
                "estimated_window_band": "Estimated delivery window (days)",
                "late_rate_pct": "Late rate (%)",
            },
            hover_data={"delivered_orders": ":,", "late_rate_pct": ":.1f"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        if not window_summary.empty:
            fig.update_yaxes(range=[0, window_summary["late_rate_pct"].max() * 1.18])
        c2.plotly_chart(chart_style(fig, 480), use_container_width=True)

        distance_bins = [0, 100, 300, 600, 1000, 2000, np.inf]
        distance_labels = ["0-100", "101-300", "301-600", "601-1,000", "1,001-2,000", "2,000+"]
        distance_data = delivery_filtered.copy()
        distance_data["distance_band"] = pd.cut(
            distance_data["max_distance_km"],
            bins=distance_bins,
            labels=distance_labels,
            include_lowest=True,
        )
        distance_summary = (
            distance_data.groupby("distance_band", observed=True)
            .agg(
                orders=("order_id", "nunique"),
                late_rate=("late", "mean"),
                median_deviation=("delivery_deviation_days", "median"),
            )
            .reset_index()
        )
        distance_summary["late_rate_pct"] = 100 * distance_summary["late_rate"]

        deviation_scope = st.radio(
            "Delivery-deviation chart scope",
            ["Late deliveries only", "All delivered orders"],
            horizontal=True,
            help="This selection applies to the right-hand severity chart. The late-rate chart always uses all delivered orders.",
        )
        deviation_data = (
            distance_data[distance_data["late"].eq(1)].copy()
            if deviation_scope == "Late deliveries only"
            else distance_data.copy()
        )
        deviation_summary = (
            deviation_data.groupby("distance_band", observed=True)
            .agg(
                orders=("order_id", "nunique"),
                median_deviation=("delivery_deviation_days", "median"),
            )
            .reset_index()
        )

        c1, c2 = st.columns(2)
        fig = px.bar(
            distance_summary,
            x="distance_band",
            y="late_rate_pct",
            text=distance_summary["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Late-delivery Rate by Seller-Customer Distance",
            labels={"distance_band": "Maximum distance (km)", "late_rate_pct": "Late rate (%)"},
            hover_data=["orders"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        c1.plotly_chart(chart_style(fig), use_container_width=True)

        if deviation_summary.empty:
            c2.info("No orders match the selected delivery-deviation scope.")
        else:
            late_only = deviation_scope == "Late deliveries only"
            fig = px.bar(
                deviation_summary,
                x="distance_band",
                y="median_deviation",
                text=deviation_summary["median_deviation"].map(
                    lambda value: f"{value:.1f} days"
                ),
                color_discrete_sequence=[BLUE],
                title=(
                    "Median Delay Among Late Orders by Distance"
                    if late_only
                    else "Median Delivery Deviation by Distance"
                ),
                labels={
                    "distance_band": "Maximum distance (km)",
                    "median_deviation": (
                        "Median days late"
                        if late_only
                        else "Days early (-) / late (+)"
                    ),
                },
                hover_data=["orders"],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            if not late_only:
                fig.add_hline(y=0, line_color=NAVY)
            c2.plotly_chart(chart_style(fig), use_container_width=True)

        with st.expander("Pre-outcome order attributes and late-delivery signal", expanded=False):
            feature_options = {
                "Order value": "order_value",
                "Freight value": "freight_value",
                "Product weight": "total_weight_g",
                "Items per order": "item_count",
            }
            feature_label = st.selectbox(
                "Order attribute",
                list(feature_options),
                key="continuous_feature",
            )
            feature_column = feature_options[feature_label]
            feature_data = delivery_filtered.dropna(subset=[feature_column]).copy()
            if feature_column == "item_count":
                feature_data["feature_band"] = pd.cut(
                    feature_data[feature_column],
                    bins=[0, 1, 2, 3, np.inf],
                    labels=["1 item", "2 items", "3 items", "4+ items"],
                    include_lowest=True,
                )
            else:
                feature_data["feature_band"] = pd.qcut(
                    feature_data[feature_column], q=5, duplicates="drop"
                )
                category_count = len(feature_data["feature_band"].cat.categories)
                feature_data["feature_band"] = feature_data["feature_band"].cat.rename_categories(
                    [f"Q{index + 1}" for index in range(category_count)]
                )
            feature_summary = (
                feature_data.groupby("feature_band", observed=True)
                .agg(
                    orders=("order_id", "nunique"),
                    late_rate=("late", "mean"),
                    minimum=(feature_column, "min"),
                    maximum=(feature_column, "max"),
                )
                .reset_index()
            )
            feature_summary["late_rate_pct"] = 100 * feature_summary["late_rate"]
            fig = px.bar(
                feature_summary,
                x="feature_band",
                y="late_rate_pct",
                text=feature_summary["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
                color_discrete_sequence=[BLUE],
                title=f"Late-delivery Rate by {feature_label}",
                labels={"feature_band": f"{feature_label} band", "late_rate_pct": "Late rate (%)"},
                hover_data={"orders": ":,", "minimum": ":,.1f", "maximum": ":,.1f"},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            if not feature_summary.empty:
                fig.update_yaxes(range=[0, feature_summary["late_rate_pct"].max() * 1.18])
            st.plotly_chart(chart_style(fig, 430), use_container_width=True)
            st.caption(
                "Q1 is the lowest-value group and Q5 the highest; item counts use explicit basket-size bands."
            )

        st.subheader("Historical cohort explorer")
        st.caption("Interactive historical comparison only; this is not yet a predictive model.")
        s1, s2, s3 = st.columns(3)
        max_scenario_distance = int(max(100, np.ceil(delivery_filtered["max_distance_km"].max() / 100) * 100))
        minimum_distance = s1.slider(
            "Minimum distance (km)", 0, max_scenario_distance, 0, 100, key="scenario_distance"
        )
        max_window = int(max(1, np.ceil(delivery_filtered["estimated_window_days"].max())))
        maximum_promise_window = s2.slider(
            "Maximum estimated window (days)", 1, max_window, max_window, key="scenario_window"
        )
        minimum_prior_late = s3.slider(
            "Minimum prior seller late rate (%)",
            0,
            50,
            0,
            1,
            key="scenario_seller_rate",
        )
        scenario = delivery_filtered[
            delivery_filtered["max_distance_km"].ge(minimum_distance)
            & delivery_filtered["estimated_window_days"].le(maximum_promise_window)
            & delivery_filtered["maximum_prior_late_rate"].fillna(0).ge(minimum_prior_late / 100)
        ]
        scenario_metrics = st.columns(4)
        scenario_metrics[0].metric("Comparable orders", f"{len(scenario):,}")
        scenario_metrics[1].metric(
            "Observed late rate", f"{scenario['late'].mean():.1%}" if len(scenario) else "N/A"
        )
        scenario_metrics[2].metric(
            "Median deviation",
            f"{scenario['delivery_deviation_days'].median():.0f} days" if len(scenario) else "N/A",
        )
        scenario_metrics[3].metric(
            "Average review", f"{scenario['review_score'].mean():.2f} / 5" if len(scenario) else "N/A"
        )


with geography_tab:
    c1, c2 = st.columns(2)
    customer_state_summary = (
        filtered.groupby("customer_state", as_index=False)
        .agg(orders=("order_id", "nunique"), order_value=("order_value", "sum"))
        .sort_values("orders", ascending=False)
    )
    fig = px.bar(
        customer_state_summary.head(15).sort_values("orders"),
        x="orders",
        y="customer_state",
        orientation="h",
        color_discrete_sequence=[BLUE],
        title="Top Customer States",
        labels={"orders": "Orders", "customer_state": "Customer state"},
        hover_data=["order_value"],
    )
    c1.plotly_chart(chart_style(fig), use_container_width=True)

    seller_state_summary = (
        filtered.groupby("primary_seller_state", as_index=False)
        .agg(orders=("order_id", "nunique"), order_value=("order_value", "sum"))
        .sort_values("orders", ascending=False)
    )
    fig = px.bar(
        seller_state_summary.head(15).sort_values("orders"),
        x="orders",
        y="primary_seller_state",
        orientation="h",
        color_discrete_sequence=[TEAL],
        title="Top Primary Seller States",
        labels={"orders": "Orders", "primary_seller_state": "Seller state"},
        hover_data=["order_value"],
    )
    c2.plotly_chart(chart_style(fig), use_container_width=True)

    if not delivery_filtered.empty:
        routes = (
            delivery_filtered.groupby(["primary_seller_state", "customer_state"], as_index=False)
            .agg(orders=("order_id", "nunique"), late_rate=("late", "mean"))
        )
        route_slider_max = max(10, min(1000, int(routes["orders"].max())))
        minimum_route_orders = st.slider(
            "Minimum orders per state-to-state route",
            10,
            route_slider_max,
            min(100, route_slider_max),
            10,
        )
        routes = routes[routes["orders"].ge(minimum_route_orders)].copy()
        routes["route"] = routes["primary_seller_state"] + " → " + routes["customer_state"]
        routes["late_rate_pct"] = 100 * routes["late_rate"]
        routes = routes.nlargest(20, "orders").sort_values("orders")
        fig = px.bar(
            routes,
            x="orders",
            y="route",
            orientation="h",
            text=routes["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Late-delivery Rate Call-outs for High-volume Routes",
            labels={"orders": "Delivered orders", "route": "Seller → Customer", "late_rate_pct": "Late rate (%)"},
            hover_data={"late_rate_pct": ":.1f"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        if not routes.empty:
            fig.update_xaxes(range=[0, routes["orders"].max() * 1.18])
        st.plotly_chart(chart_style(fig, 550), use_container_width=True)


with seller_tab:
    if history_filtered.empty:
        empty_state("No seller-history observations match the current filters.")
    else:
        history_filtered["experience_band"] = pd.cut(
            history_filtered["prior_completed_orders"],
            bins=[-1, 0, 4, 19, 49, np.inf],
            labels=["0 (new seller)", "1-4", "5-19", "20-49", "50+"],
        )
        history_filtered["risk_band"] = np.select(
            [
                history_filtered["prior_completed_orders"].eq(0),
                history_filtered["prior_completed_orders"].between(1, 4),
                history_filtered["prior_late_rate"].eq(0),
                history_filtered["prior_late_rate"].le(0.05),
                history_filtered["prior_late_rate"].le(0.10),
                history_filtered["prior_late_rate"].gt(0.10),
            ],
            [
                "New seller",
                "Fewer than 5 prior orders",
                "0% prior late",
                ">0-5% prior late",
                ">5-10% prior late",
                ">10% prior late",
            ],
            default="Unknown",
        )
        experience = (
            history_filtered.groupby("experience_band", observed=True)
            .agg(observations=("order_id", "count"), current_late_rate=("late", "mean"))
            .reset_index()
        )
        experience["late_rate_pct"] = 100 * experience["current_late_rate"]
        risk_order = [
            "New seller",
            "Fewer than 5 prior orders",
            "0% prior late",
            ">0-5% prior late",
            ">5-10% prior late",
            ">10% prior late",
        ]
        risk = (
            history_filtered.groupby("risk_band", as_index=False)
            .agg(observations=("order_id", "count"), current_late_rate=("late", "mean"))
        )
        risk["risk_band"] = pd.Categorical(risk["risk_band"], categories=risk_order, ordered=True)
        risk = risk.sort_values("risk_band")
        risk["late_rate_pct"] = 100 * risk["current_late_rate"]

        c1, c2 = st.columns(2)
        fig = px.bar(
            experience,
            x="experience_band",
            y="late_rate_pct",
            text=experience["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Late-delivery Rate by Prior Seller Experience",
            labels={"experience_band": "Completed orders before purchase", "late_rate_pct": "Late rate (%)"},
            hover_data=["observations"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        c1.plotly_chart(chart_style(fig), use_container_width=True)

        fig = px.bar(
            risk,
            x="risk_band",
            y="late_rate_pct",
            text=risk["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Late-delivery Rate by Historical Seller Risk",
            labels={"risk_band": "Seller history available at purchase", "late_rate_pct": "Late rate (%)"},
            hover_data=["observations"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        c2.plotly_chart(chart_style(fig), use_container_width=True)

        st.subheader("Seller performance explorer")
        minimum_orders = st.slider("Minimum delivered seller-orders", 1, 100, 20)
        seller_performance = (
            history_filtered.groupby("seller_id", as_index=False)
            .agg(
                delivered_orders=("order_id", "count"),
                late_rate=("late", "mean"),
                median_deviation=("delivery_deviation_days", "median"),
            )
        )
        seller_performance = seller_performance[seller_performance["delivered_orders"].ge(minimum_orders)]
        seller_performance["late_rate_pct"] = 100 * seller_performance["late_rate"]
        fig = px.scatter(
            seller_performance,
            x="delivered_orders",
            y="late_rate_pct",
            size="delivered_orders",
            color_discrete_sequence=[BLUE],
            hover_name="seller_id",
            title="Seller Volume and Late-delivery Rate",
            labels={
                "delivered_orders": "Delivered seller-orders",
                "late_rate_pct": "Late rate (%)",
                "median_deviation": "Median deviation (days)",
            },
            hover_data={"late_rate_pct": ":.1f", "median_deviation": ":.1f"},
        )
        st.plotly_chart(chart_style(fig, 520), use_container_width=True)


with review_tab:
    st.subheader("Three-month Repeat-purchase Rate")
    repeat_plot = repeat_monthly[
        repeat_monthly["purchase_month"].between(
            pd.Timestamp(selected_start).to_period("M").to_timestamp(),
            min(
                pd.Timestamp(selected_end).to_period("M").to_timestamp(),
                pd.Timestamp("2018-05-01"),
            ),
        )
    ].copy()
    if repeat_plot.empty:
        empty_state("No complete three-month repeat-purchase cohorts match the date selection.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=repeat_plot["purchase_month"],
                y=repeat_plot["customers"],
                name="Customers",
                marker_color=LIGHT_BLUE,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=repeat_plot["purchase_month"],
                y=repeat_plot["repeat_rate_pct"],
                name="Repeat-purchase rate",
                mode="lines+markers+text",
                text=repeat_plot["repeat_rate_pct"].map(lambda value: f"{value:.1f}%"),
                textposition="top center",
                line=dict(color=ORANGE, width=3),
                customdata=repeat_plot[["repeat_customers"]],
                hovertemplate=(
                    "Repeat-purchase rate: %{y:.2f}%<br>"
                    "Repeat customers: %{customdata[0]:,.0f}<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Customers in monthly cohort", secondary_y=False)
        fig.update_yaxes(title_text="Repeat-purchase rate (%)", secondary_y=True)
        fig.update_layout(title="Customers Purchasing Again Within the Next Three Months")
        st.plotly_chart(chart_style(fig, 470), use_container_width=True)
        st.caption(
            "Only cohorts through May 2018 are shown so every customer has a complete "
            "three-month opportunity to purchase again."
        )

    st.divider()
    st.subheader("Customer Review Outcomes")
    if review_filtered.empty:
        empty_state("No reviewed orders match the current filters.")
    else:
        c1, c2 = st.columns(2)
        review_counts = (
            review_filtered["review_score"].round().astype(int).value_counts().sort_index()
            .rename_axis("review_score").reset_index(name="orders")
        )
        fig = px.bar(
            review_counts,
            x="review_score",
            y="orders",
            color="review_score",
            color_continuous_scale=[[0, "#B23A48"], [0.5, ORANGE], [1, TEAL]],
            title="Review-score Distribution",
            labels={"review_score": "Review score", "orders": "Orders"},
        )
        fig.update_layout(coloraxis_showscale=False)
        c1.plotly_chart(chart_style(fig), use_container_width=True)

        review_delivery = review_filtered[review_filtered["valid_delivery_target"]].copy()
        score_late = (
            review_delivery.groupby("review_group", observed=True)
            .agg(orders=("order_id", "nunique"), late_rate=("late", "mean"))
            .reset_index()
        )
        score_late["late_rate_pct"] = 100 * score_late["late_rate"]
        fig = px.bar(
            score_late,
            x="review_group",
            y="late_rate_pct",
            text=score_late["late_rate_pct"].map(lambda value: f"{value:.1f}%"),
            color_discrete_sequence=[BLUE],
            title="Late-delivery Rate by Review Group",
            labels={"review_group": "Review group", "late_rate_pct": "Late rate (%)"},
            hover_data=["orders"],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        c2.plotly_chart(chart_style(fig), use_container_width=True)

        comment_coverage = review_filtered["has_review_comment"].mean()
        low_review_rate = review_filtered["review_score"].le(2).mean()
        k1, k2, k3 = st.columns(3)
        k1.metric("Low reviews (1-2)", f"{low_review_rate:.1%}")
        k2.metric("Reviews with written comments", f"{comment_coverage:.1%}")
        k3.metric(
            "On-time low reviews",
            f"{review_delivery.loc[review_delivery['review_score'].le(2), 'late'].eq(0).mean():.1%}",
        )
        st.caption(
            "Review text is diagnostic, not a pre-outcome predictor: it is written after the customer experience."
        )


with data_tab:
    st.subheader("Observation Completeness by Month")
    monthly_coverage = (
        order_df.dropna(subset=["purchase_month", "purchase_date"])
        .groupby("purchase_month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            observed_days=("purchase_date", "nunique"),
        )
    )
    monthly_coverage["orders_per_observed_day"] = (
        monthly_coverage["orders"] / monthly_coverage["observed_days"]
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly_coverage["purchase_month"],
            y=monthly_coverage["observed_days"],
            name="Observed days",
            marker_color=LIGHT_BLUE,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_coverage["purchase_month"],
            y=monthly_coverage["orders_per_observed_day"],
            name="Orders per observed day",
            mode="lines+markers",
            line=dict(color=ORANGE, width=3),
        ),
        secondary_y=True,
    )
    fig.add_vrect(
        x0="2017-01-01",
        x1="2018-09-01",
        fillcolor=TEAL,
        opacity=0.07,
        line_width=0,
        annotation_text="Recommended complete-month window",
        annotation_position="top left",
    )
    fig.update_yaxes(title_text="Observed purchase days", secondary_y=False)
    fig.update_yaxes(title_text="Orders per observed day", secondary_y=True)
    fig.update_layout(title="Coverage and Normalised Order Activity")
    st.plotly_chart(chart_style(fig, 470), use_container_width=True)
    st.caption(
        "Sparse 2016 records and the truncated September–October 2018 tail are excluded "
        "from the recommended January 2017–August 2018 analysis window."
    )

    st.subheader("Source-table inventory")
    inventory_display = inventory.copy()
    inventory_display["Rows"] = inventory_display["Rows"].map(lambda value: f"{value:,}")
    inventory_display["Missing cells (%)"] = inventory_display["Missing cells (%)"].map(
        lambda value: f"{value:.2f}%"
    )
    st.dataframe(inventory_display, use_container_width=True, hide_index=True)

    st.subheader("Join Integrity and Fan-out")
    st.dataframe(quality_checks, use_container_width=True, hide_index=True)
    st.caption(
        "Payments and reviews are aggregated before joining, while order-level counts use "
        "unique order_id to avoid multiplication from line items."
    )

    st.subheader("Feature timing and leakage guardrails")
    feature_dictionary = pd.DataFrame(
        [
            ("Purchase timing", "Before outcome", "Candidate predictor"),
            ("Estimated delivery window", "Before outcome", "Candidate predictor"),
            ("Seller/customer location and distance", "Before outcome", "Candidate predictor"),
            ("Basket, product, price and freight", "Before outcome", "Candidate predictor"),
            ("Seller history completed before purchase", "Before outcome", "Candidate predictor"),
            ("Actual carrier handover", "After purchase", "Exclude from order-time model"),
            ("Actual customer delivery", "Outcome", "Target construction only"),
            ("Review score and comments", "After outcome", "Diagnostic analysis only"),
        ],
        columns=["Feature", "Availability", "Modelling use"],
    )
    st.dataframe(feature_dictionary, use_container_width=True, hide_index=True)

    st.subheader("Definitions")
    st.markdown(
        """
        - **Late delivery:** actual customer delivery calendar date is after the estimated delivery calendar date.
        - **Delivery deviation:** actual date minus estimated date; negative is early and positive is late.
        - **Distance:** straight-line Haversine distance using median postcode-prefix coordinates.
        - **Seller history:** only deliveries completed before the current order was purchased.
        - **New seller:** no completed seller orders were observable before the current purchase.
        """
    )
