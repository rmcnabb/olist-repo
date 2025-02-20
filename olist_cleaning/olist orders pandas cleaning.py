import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(file_path):
    df = pd.read_csv(file_path)
    logging.info(f'Data loaded from {file_path}')
    return df

# Load the Order Items dataset
order_items = load_data(r"C:\Users\ross_\OneDrive\Documents\Brazilian e-commerce datasets\olist_order_items_dataset.csv")

# Check the first few rows
print(order_items.head())

# Get a summary of the dataset
print(order_items.info())

# Describe numerical columns
print(order_items.describe())

# Check for duplicates
duplicate_pairs = order_items.duplicated(subset=['order_id', 'order_item_id'], keep=False)

if duplicate_pairs.any():
    logging.warning('Duplicate (order_id, order_item_id) pairs found.')
    # View duplicates
    print(order_items[duplicate_pairs])
    # Decide on a strategy to handle duplicates (e.g., remove them)
    order_items = order_items.drop_duplicates(subset=['order_id', 'order_item_id'])
    logging.info('Duplicate pairs removed.')
else:
    logging.info('No duplicate (order_id, order_item_id) pairs found.')
# Check for missing values
missing_values = order_items[['order_id', 'product_id', 'seller_id', 'price', 'freight_value']].isnull().sum()
print('Missing values per column:\n', missing_values)

# Handle missing values
def handle_missing_values(df, columns):
    for col in columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            logging.warning(f'{missing_count} missing values found in {col}.')
            # Strategy: Drop rows with missing critical values
            df = df.dropna(subset=[col])
            logging.info(f'Rows with missing {col} have been dropped.')
    return df

order_items = handle_missing_values(order_items, ['order_id', 'product_id', 'seller_id', 'price', 'freight_value'])

# Load Products and Sellers datasets
products = load_data(r"C:\Users\ross_\OneDrive\Documents\Brazilian e-commerce datasets\olist_products_dataset.csv")
sellers = load_data(r"C:\Users\ross_\OneDrive\Documents\Brazilian e-commerce datasets\olist_sellers_dataset.csv")

# Check if all product_ids in order_items exist in products
missing_products = set(order_items['product_id']) - set(products['product_id'])

if missing_products:
    logging.warning(f'{len(missing_products)} product_ids in Order Items not found in Products dataset.')
    # Decide on a strategy (e.g., remove these records)
    order_items = order_items[~order_items['product_id'].isin(missing_products)]
    logging.info('Records with missing product_ids have been removed.')
else:
    logging.info('All product_ids are valid.')
# Check if all seller_ids in order_items exist in sellers
missing_sellers = set(order_items['seller_id']) - set(sellers['seller_id'])

if missing_sellers:
    logging.warning(f'{len(missing_sellers)} seller_ids in Order Items not found in Sellers dataset.')
    # Decide on a strategy (e.g., remove these records)
    order_items = order_items[~order_items['seller_id'].isin(missing_sellers)]
    logging.info('Records with missing seller_ids have been removed.')
else:
    logging.info('All seller_ids are valid.')
# Function to Detect Outliers 
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Function to log the summary instead of removing them
def log_outliers(df, column):
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, column)
    if not outliers.empty:
        logging.warning(f'{len(outliers)} outliers detected in {column}.')
        logging.info(f'Lower Bound: {lower_bound}, Upper Bound: {upper_bound}')
        logging.info(f'Outliers Summary:\n{outliers.describe()}')
        # Alternatively, return or display the outliers for further analysis
        print(f"Outliers detected in {column}:")
        print(outliers)
    else:
        logging.info(f'No outliers detected in {column}.')
    return outliers

# Apply the function to detect and log outliers in 'price' without removing them
outliers_in_price = log_outliers(order_items, 'price')

# Calculate total freight per order by summing freight_value
freight_per_order = order_items.groupby('order_id')['freight_value'].sum().reset_index()
freight_per_order.rename(columns={'freight_value': 'total_freight_value'}, inplace=True)

# Calculate total order price by summing price * quantity
# Ensure that the 'quantity' column exists. If not, adjust accordingly.
if 'quantity' in order_items.columns:
    # Handle missing or zero quantities
    order_items['quantity'] = order_items['quantity'].fillna(1)  # Assuming default quantity as 1
    order_items['quantity'] = order_items['quantity'].replace(0, 1)  # Replace zero with 1 to avoid null total price

    # Calculate total price per order considering quantity
    order_items['total_item_price'] = order_items['price'] * order_items['quantity']
    price_per_order = order_items.groupby('order_id')['total_item_price'].sum().reset_index()
    price_per_order.rename(columns={'total_item_price': 'total_order_price'}, inplace=True)
else:
    logging.info("No 'quantity' column found. Calculating total order price by summing 'price'.")
    price_per_order = order_items.groupby('order_id')['price'].sum().reset_index()
    price_per_order.rename(columns={'price': 'total_order_price'}, inplace=True)

# Merge total freight back to order_items
order_items = order_items.merge(freight_per_order, on='order_id', how='left')

# Merge total order price back to order_items
order_items = order_items.merge(price_per_order, on='order_id', how='left')

# Calculate proportion of freight per item
order_items['freight_proportion'] = order_items['freight_value'] / order_items['total_freight_value']

# Calculate proportion of item price
order_items['item_proportion'] = order_items['price'] / order_items['total_order_price']

# Handle potential division by zero or missing values
order_items['freight_proportion'] = order_items['freight_proportion'].replace([np.inf, -np.inf], np.nan)
order_items['item_proportion'] = order_items['item_proportion'].replace([np.inf, -np.inf], np.nan)

order_items['freight_proportion'] = order_items['freight_proportion'].fillna(0)
order_items['item_proportion'] = order_items['item_proportion'].fillna(0)

# Check for anomalies in freight_proportion
anomalies = order_items[(order_items['freight_proportion'] < 0) | (order_items['freight_proportion'] > 1)]

if not anomalies.empty:
    logging.warning(f'{len(anomalies)} anomalies detected in freight allocation.')
    # Optionally, inspect or handle anomalies
    print(f"Anomalies detected in freight_proportion:")
    print(anomalies[['order_id', 'freight_value', 'total_freight_value', 'freight_proportion']])
else:
    logging.info('Freight allocation appears consistent.')


# Load Orders dataset
orders = load_data(r"C:\Users\ross_\OneDrive\Documents\Brazilian e-commerce datasets\olist_orders_dataset.csv")

# Check for missing order_ids
missing_orders = set(order_items['order_id']) - set(orders['order_id'])

if missing_orders:
    logging.warning(f'{len(missing_orders)} order_ids in Order Items not found in Orders dataset.')
    # Decide on a strategy (e.g., remove these records)
    order_items = order_items[~order_items['order_id'].isin(missing_orders)]
    logging.info('Records with missing order_ids have been removed.')
else:
    logging.info('All order_ids are valid.')

# Optional: Merge Orders data into Order Items for enriched analysis
order_items = order_items.merge(orders, on='order_id', how='left')

# Define expected data types
expected_dtypes = {
    'order_id': 'object',
    'order_item_id': 'int64',
    'product_id': 'object',
    'seller_id': 'object',
    'shipping_limit_date': 'datetime64[ns]',
    'price': 'float64',
    'freight_value': 'float64'
}

# Convert data types
order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'])

for column, dtype in expected_dtypes.items():
    if column != 'shipping_limit_date':  # Already converted
        order_items[column] = order_items[column].astype(dtype)
        logging.info(f'Column {column} converted to {dtype}.')

# Check for missing dates
if order_items['shipping_limit_date'].isnull().any():
    logging.warning('Missing values detected in shipping_limit_date.')
    order_items = order_items.dropna(subset=['shipping_limit_date'])
    logging.info('Rows with missing shipping_limit_date have been dropped.')

# Ensure that shipping_limit_date is after order purchase date
# Assuming 'order_purchase_timestamp' is in the merged Orders dataset
order_items['order_purchase_timestamp'] = pd.to_datetime(order_items['order_purchase_timestamp'])

# Compare dates
invalid_dates = order_items[order_items['shipping_limit_date'] < order_items['order_purchase_timestamp']]

if not invalid_dates.empty:
    logging.warning(f'{len(invalid_dates)} records with shipping_limit_date before order_purchase_timestamp.')
    # Decide on a strategy (e.g., correct dates if possible, or remove records)
    order_items = order_items[order_items['shipping_limit_date'] >= order_items['order_purchase_timestamp']]
    logging.info('Invalid date records have been removed.')
else:
    logging.info('All shipping_limit_dates are valid.')
#Save Cleaned Data 
def save_clean_data(df, output_path):
    df.to_csv(output_path, index=False)
    logging.info(f'Clean data saved to {output_path}')

save_clean_data(order_items, 'clean_order_items_dataset.csv')
