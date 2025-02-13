import yfinance as yf
import pandas as pd
import os

def load_etf_data(symbol, start_date='2005-01-01', end_date='2023-12-31'):
    """Download and process ETF data"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        # Convert timezone from UTC to US/Eastern
        df.index = pd.to_datetime(df.index)
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
        return None

def save_signals(df, symbol, strategy_name):
    """Save signal data to CSV"""
    ticker_dir = os.path.join('data', 'processed', symbol)
    os.makedirs(ticker_dir, exist_ok=True)
    output_file = os.path.join(ticker_dir, f'weekly_{strategy_name}_signals.csv')
    df.to_csv(output_file)
    print(f"Data saved to {output_file}") 