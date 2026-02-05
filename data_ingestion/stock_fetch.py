import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="AAPL", period="7d", interval="1d", min_years=2):
    """
    Fetch stock data from yfinance
    FIX: Fetch 2-3 years of data (enough for Prophet, not too slow)
    """
    print("Fetching data...")
    # FIX: Fetch 2 years of data (enough for Prophet, much faster than 5 years)
    data = yf.download(ticker, period="2y", interval="1d")
    
    # Verify we have enough data
    if len(data) > 0:
        data_days = (data.index[-1] - data.index[0]).days
        data_years = data_days / 365.25
        print(f"Fetched {len(data)} days of data ({data_years:.1f} years)")
        
        # If still less than min_years, try 3 years (not max - too slow)
        if data_years < min_years:
            print(f"Data is only {data_years:.1f} years. Trying 3 years...")
            data = yf.download(ticker, period="3y", interval="1d")
            data_days = (data.index[-1] - data.index[0]).days
            data_years = data_days / 365.25
            print(f"Fetched {len(data)} days of data ({data_years:.1f} years)")
    data.reset_index(inplace=True)
    
    # Flatten multi-level columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[1] == ticker else col[0] for col in data.columns]
    
    # Auto-detect datetime column (yfinance usually returns 'Date' or index name)
    datetime_col = None
    for col in ['Date', 'Datetime', 'datetime', 'date', 'timestamp', 'Timestamp']:
        if col in data.columns:
            datetime_col = col
            break
    
    # If no datetime column found, check index
    if datetime_col is None:
        if pd.api.types.is_datetime64_any_dtype(data.index):
            data = data.reset_index()
            datetime_col = data.columns[0]  # First column after reset_index
    
    # Rename datetime column to 'Datetime' for consistency
    if datetime_col and datetime_col != 'Datetime':
        data = data.rename(columns={datetime_col: 'Datetime'})
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = ['Datetime'] + [col for col in required_cols if col in data.columns]
    
    return data[available_cols]

if __name__ == "__main__":
    df = fetch_stock_data()
    print("Data fetched:")
    print(df.head())
