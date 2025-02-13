import yfinance as yf
import pandas as pd
import os

def load_etf_data(symbol, start_date='2005-01-01', end_date='2023-12-31'):
    """Download and process ETF data"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # Convert timezone from UTC to US/Eastern
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert('US/Eastern')
        
        # Resample to weekly data
        weekly_df = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        return weekly_df
    except Exception as e:
        print(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of None

def save_signals(df, symbol, strategy_name):
    """Save signal data to CSV"""
    ticker_dir = os.path.join('data', 'processed', symbol)
    os.makedirs(ticker_dir, exist_ok=True)
    output_file = os.path.join(ticker_dir, f'weekly_{strategy_name}_signals.csv')
    df.to_csv(output_file)
    print(f"Data saved to {output_file}")

def load_vix_data(start_date='2005-01-01', end_date='2023-12-31'):
    """
    Load VIX index data. Downloads from Yahoo Finance if local file doesn't exist.
    """
    try:
        # Try to load from local file first
        vix_path = os.path.join('data', 'raw', 'VIX.csv')
        
        if os.path.exists(vix_path):
            vix_df = pd.read_csv(vix_path)
            vix_df['Date'] = pd.to_datetime(vix_df['Date'], utc=True)
            vix_df.set_index('Date', inplace=True)
        else:
            # If file doesn't exist, download from Yahoo Finance
            print("VIX data not found locally. Downloading from Yahoo Finance...")
            vix_ticker = yf.Ticker('^VIX')
            vix_df = vix_ticker.history(start=start_date, end=end_date)
            
            # Convert timezone from UTC to US/Eastern
            vix_df.index = pd.to_datetime(vix_df.index, utc=True)
            vix_df.index = vix_df.index.tz_convert('US/Eastern')
            
            # Resample to weekly data
            vix_df = vix_df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # Save to file for future use
            os.makedirs(os.path.dirname(vix_path), exist_ok=True)
            vix_df.to_csv(vix_path)
            print(f"VIX data saved to {vix_path}")
        
        return vix_df
    except Exception as e:
        print(f"Error loading VIX data: {e}")
        return None 