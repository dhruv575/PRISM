import pandas as pd
import os
from datetime import datetime # Import datetime

# Define data path relative to this file's location
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
STOCK_DATA_PATH = os.path.join(DATA_DIR, "stock_data.csv")
RF_DATA_PATH = os.path.join(DATA_DIR, "risk_free_data.csv")
TICKERS_PATH = os.path.join(DATA_DIR, "tickers_by_sector.json") # Added for potential future use
FACTORS_PATH = os.path.join(DATA_DIR, "factors_data.csv") # Path for Fama-French factors

def load_data():
    """Loads stock and risk-free rate data from CSV files.
       Stock data is pivoted to wide format (date index, ticker columns).
       Handles duplicate date/ticker entries by averaging returns.
    """
    try:
        # Load stock data (long format)
        stock_data_long = pd.read_csv(STOCK_DATA_PATH, parse_dates=['date']) # Use lowercase 'date'

        # Check for duplicates before pivoting
        duplicates = stock_data_long[stock_data_long.duplicated(subset=['date', 'ticker'], keep=False)]
        if not duplicates.empty:
            print(f"Warning: Found {len(duplicates)} duplicate date/ticker entries in stock_data.csv.")
            print("Aggregating returns using 'mean'. First few duplicates:")
            print(duplicates.head())

        # Pivot to wide format using pivot_table with mean aggregation
        stock_data_df = stock_data_long.pivot_table(
            index='date', 
            columns='ticker', 
            values='ret', 
            aggfunc='mean' # Aggregate duplicates by taking the mean return
        )

        # Load risk-free data
        # Use lowercase 'date', set as index directly
        rf_data_df = pd.read_csv(RF_DATA_PATH, parse_dates=['date'], index_col='date')

        print("Data loaded. Stock data pivoted successfully (duplicates averaged).")
        return stock_data_df, rf_data_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure '{STOCK_DATA_PATH}' and '{RF_DATA_PATH}' exist.")
        return None, None
    except KeyError as e:
        print(f"Error processing data: Missing expected column - {e}")
        print("Please ensure CSV files have 'date', 'ticker', 'ret' (for stock) and 'date', 'rf' (for risk-free)." )
        return None, None
    except Exception as e: # Catch other potential errors during pivoting etc.
        print(f"An unexpected error occurred during data loading: {e}")
        return None, None

def load_factor_data():
    """Loads Fama-French factor data from CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with market (mktrf), size (smb), and value (hml) factors,
                     or None if loading fails.
    """
    try:
        # Load factor data
        factor_data = pd.read_csv(FACTORS_PATH, parse_dates=['date'], index_col='date')
        
        # Ensure required columns exist
        required_columns = ['mktrf', 'smb', 'hml']
        missing_columns = [col for col in required_columns if col not in factor_data.columns]
        
        if missing_columns:
            print(f"Missing required columns in factors data: {missing_columns}")
            print(f"Available columns: {factor_data.columns.tolist()}")
            return None
            
        print(f"Factor data loaded successfully with {len(factor_data)} entries.")
        return factor_data
    except FileNotFoundError:
        print(f"Factor data file not found: {FACTORS_PATH}")
        print("CAPM and FF3 factor analysis will not be available.")
        return None
    except Exception as e:
        print(f"Error loading factor data: {e}")
        return None

# --- Data filtering function (should work with pivoted data) ---
def filter_data(stock_df, rf_df, start_date_str=None, end_date_str=None, tickers=None):
    """Filters stock (wide format) and risk-free data based on date range and tickers.

    Args:
        stock_df (pd.DataFrame): DataFrame with stock returns (Date index, tickers as columns).
        rf_df (pd.DataFrame): DataFrame with risk-free rates (Date index, 'rf' column).
        start_date_str (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to None (start of data).
        end_date_str (str, optional): End date in 'YYYY-MM-DD' format. Defaults to None (end of data).
        tickers (list, optional): List of ticker symbols to include. Defaults to None (all tickers).

    Returns:
        pd.DataFrame: Combined DataFrame with filtered stock returns and risk-free rate ('rf' column),
                      or None if filtering results in an empty DataFrame.
    """
    filtered_stock_df = stock_df.copy()
    filtered_rf_df = rf_df.copy()

    # Convert date strings to datetime objects
    start_date = pd.to_datetime(start_date_str) if start_date_str else None
    end_date = pd.to_datetime(end_date_str) if end_date_str else None

    # Filter by date
    if start_date:
        filtered_stock_df = filtered_stock_df[filtered_stock_df.index >= start_date]
        filtered_rf_df = filtered_rf_df[filtered_rf_df.index >= start_date]
    if end_date:
        filtered_stock_df = filtered_stock_df[filtered_stock_df.index <= end_date]
        filtered_rf_df = filtered_rf_df[filtered_rf_df.index <= end_date]

    # Filter by tickers
    if tickers:
        # Ensure only requested tickers that exist in the dataframe are selected
        valid_tickers = [t for t in tickers if t in filtered_stock_df.columns]
        if not valid_tickers:
            print(f"Warning: None of the requested tickers {tickers} found in the data.")
            return None
        # Select only valid tickers that exist in columns
        filtered_stock_df = filtered_stock_df[valid_tickers]
    else:
        # If no tickers specified, use all available tickers from the wide format
        valid_tickers = filtered_stock_df.columns.tolist()

    # Combine stock data and risk-free rate
    combined_df = filtered_stock_df.join(filtered_rf_df, how='inner') # Use inner join to ensure dates match

    # Ensure 'rf' column exists (already correct based on rf_data.csv header)
    if 'rf' not in combined_df.columns:
        print("Warning: Risk-free rate column ('rf') not found after join.")
        # Attempt rename (as fallback, though likely unnecessary now)
        if 'Daily Treasury Yield Curve Rate' in combined_df.columns:
            print("Renaming 'Daily Treasury Yield Curve Rate' to 'rf'")
            combined_df = combined_df.rename(columns={'Daily Treasury Yield Curve Rate': 'rf'})
        else:
            print("Could not find 'rf' or alternative name.")
            return None

    # Reorder columns to have tickers first, then 'rf'
    # Ensure 'rf' is included if it exists
    final_columns = valid_tickers + [col for col in ['rf'] if col in combined_df.columns]
    combined_df = combined_df[final_columns]

    if combined_df.empty:
        print("Warning: Filtering resulted in an empty DataFrame.")
        return None

    return combined_df

def filter_factors(factor_df, start_date_str=None, end_date_str=None):
    """Filters factor data based on date range.

    Args:
        factor_df (pd.DataFrame): DataFrame with factor data (Date index).
        start_date_str (str, optional): Start date in 'YYYY-MM-DD' format. 
        end_date_str (str, optional): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Filtered factor data DataFrame.
    """
    if factor_df is None:
        return None
        
    filtered_factor_df = factor_df.copy()

    # Convert date strings to datetime objects
    start_date = pd.to_datetime(start_date_str) if start_date_str else None
    end_date = pd.to_datetime(end_date_str) if end_date_str else None

    # Filter by date
    if start_date:
        filtered_factor_df = filtered_factor_df[filtered_factor_df.index >= start_date]
    if end_date:
        filtered_factor_df = filtered_factor_df[filtered_factor_df.index <= end_date]

    return filtered_factor_df 