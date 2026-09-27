"""Locate and load the Olist tables (model_dev_v2.ipynb cells 3, 5 and 6)."""
import os
import zipfile

import pandas as pd

TABLES = {
    'orders': 'olist_orders_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'payments': 'olist_order_payments_dataset.csv',
    'reviews': 'olist_order_reviews_dataset.csv',
    'geolocation': 'olist_geolocation_dataset.csv',
    'category_translation': 'product_category_name_translation.csv',
}

DATE_COLS = {
    'orders': ['order_purchase_timestamp', 'order_approved_at',
               'order_delivered_carrier_date', 'order_delivered_customer_date',
               'order_estimated_delivery_date'],
    'order_items': ['shipping_limit_date'],
    'reviews': ['review_creation_date', 'review_answer_timestamp'],
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, 'Olist_CSV')
BUNDLED_ARCHIVE = os.path.join(PROJECT_ROOT, 'streamlit_release', 'data', 'olist_csv.zip')


def resolve_data_dir(data_dir=None):
    """Return a folder of Olist CSVs, extracting the bundled zip on first use."""
    data_dir = data_dir or LOCAL_DATA_DIR
    if not os.path.isdir(data_dir) and os.path.isfile(BUNDLED_ARCHIVE):
        print(f'Extracting {BUNDLED_ARCHIVE} -> {data_dir}')
        with zipfile.ZipFile(BUNDLED_ARCHIVE) as archive:
            archive.extractall(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Olist data directory not found: {data_dir}')
    return data_dir


def load_olist_data(data_dir=None, verbose=True):
    """Load all Olist CSV files into a dictionary of DataFrames.

    Product categories are translated to English, with 'Unknown' for missing values.
    """
    data_dir = resolve_data_dir(data_dir)
    data = {}
    for name, filename in TABLES.items():
        data[name] = pd.read_csv(os.path.join(data_dir, filename),
                                 parse_dates=DATE_COLS.get(name))
        if verbose:
            print(f'Loaded {name}: {data[name].shape[0]:,} rows × {data[name].shape[1]} cols')

    products = data['products'].merge(data['category_translation'],
                                      on='product_category_name', how='left')
    products['product_category_name'] = (products['product_category_name_english']
                                         .combine_first(products['product_category_name'])
                                         .fillna('Unknown'))
    data['products'] = products.drop(columns=['product_category_name_english'])
    return data
