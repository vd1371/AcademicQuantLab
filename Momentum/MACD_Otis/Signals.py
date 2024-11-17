import pandas as pd
import yfinance as yf
import numpy as np
import os

def apply_stop_loss(df, stop_loss_pct=0.03):
    """
    Apply stop loss to positions immediately when threshold is breached
    Uses intraweek high/low prices to check for stop loss triggers
    """
    position = 0
    entry_price = 0
    portfolio_value = df['Portfolio_Value'].iloc[0]
    
    for i in range(len(df)):
        if df['Position'].iloc[i] != 0 and position == 0:
            # Enter new position
            position = df['Position'].iloc[i]
            entry_price = df['Close'].iloc[i]
            portfolio_value = df['Portfolio_Value'].iloc[i]
        elif position != 0:
            # Check for stop loss using High and Low prices
            if position == 1:  # Long position
                lowest_price = df['Low'].iloc[i]
                loss_pct = (lowest_price - entry_price) / entry_price
                if loss_pct < -stop_loss_pct:
                    # Stop loss triggered - use the stop loss price for return calculation
                    stop_price = entry_price * (1 - stop_loss_pct)
                    df.loc[df.index[i], 'Close'] = stop_price  # Assume execution at stop price
                    df.loc[df.index[i], 'Position'] = 0
                    position = 0
                    entry_price = 0
                    
            else:  # Short position
                highest_price = df['High'].iloc[i]
                loss_pct = (entry_price - highest_price) / entry_price
                if loss_pct < -stop_loss_pct:
                    # Stop loss triggered - use the stop loss price for return calculation
                    stop_price = entry_price * (1 + stop_loss_pct)
                    df.loc[df.index[i], 'Close'] = stop_price  # Assume execution at stop price
                    df.loc[df.index[i], 'Position'] = 0
                    position = 0
                    entry_price = 0
            
            # Check for regular position change
            if df['Position'].iloc[i] != position and position != 0:
                position = df['Position'].iloc[i]
                entry_price = df['Close'].iloc[i] if position != 0 else 0
                portfolio_value = df['Portfolio_Value'].iloc[i]
    
    return df

def calculate_strategy_returns(df):
    """Calculate strategy returns with position changes"""
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Portfolio_Returns'] = df['Strategy_Returns']
    df['Portfolio_Value'] = df['Portfolio_Value'].iloc[0] * (1 + df['Portfolio_Returns']).cumprod()
    df['Position_Change'] = df['Position'].diff()
    return df

def get_macd_signals(symbol='^GSPC', start_date='2005-01-01', end_date='2023-12-31', initial_capital=1_000_000):
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download S&P 500 data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Convert timezone from UTC to US/Eastern
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert('US/Eastern')
    
    # Resample to weekly data (last trading day of the week)
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # Calculate weekly MACD
    exp1 = weekly_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = weekly_df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Create signals on weekly data
    weekly_df['MACD'] = macd
    weekly_df['Signal_Line'] = signal
    weekly_df['MACD_Histogram'] = macd - signal
    
    # Generate buy/sell signals
    weekly_df['Position'] = 0
    weekly_df['Position'] = weekly_df['Position'].mask(macd > signal, 1)
    weekly_df['Position'] = weekly_df['Position'].mask(macd < signal, -1)
    
    # Shift positions by 1 week to implement signal lag
    weekly_df['Position'] = weekly_df['Position'].shift(1)
    
    # Initialize Portfolio Value
    weekly_df['Portfolio_Value'] = initial_capital
    
    # Apply stop loss with 2%
    weekly_df = apply_stop_loss(weekly_df, stop_loss_pct=0.02)
    
    # Recalculate returns after stop loss
    weekly_df = calculate_strategy_returns(weekly_df)
    
    # Save DataFrame to CSV
    output_file = f'data/weekly_macd_signals_{symbol.replace("^", "")}.csv'
    weekly_df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    
    return weekly_df

def get_macd_signals_zero_cross(symbol='^GSPC', start_date='2005-01-01', end_date='2023-12-31', initial_capital=1_000_000):
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download S&P 500 data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Convert timezone from UTC to US/Eastern
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert('US/Eastern')
    
    # Resample to weekly data (last trading day of the week)
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # Calculate weekly MACD
    exp1 = weekly_df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = weekly_df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Create signals on weekly data
    weekly_df['MACD'] = macd
    weekly_df['Signal_Line'] = signal
    weekly_df['MACD_Histogram'] = macd - signal
    
    # Generate buy/sell signals with zero-line condition
    weekly_df['Position'] = 0
    
    # Previous position to maintain when no new signal
    prev_position = 0
    
    for i in range(len(weekly_df)):
        # Buy signal: MACD crosses above signal line AND MACD is above zero
        if (macd.iloc[i] > signal.iloc[i]) and (macd.iloc[i] > 0):
            weekly_df.iloc[i, weekly_df.columns.get_loc('Position')] = 1
            prev_position = 1
        # Sell signal: MACD crosses below signal line AND MACD crosses below zero
        elif (macd.iloc[i] < signal.iloc[i]) and (macd.iloc[i] < 0):
            weekly_df.iloc[i, weekly_df.columns.get_loc('Position')] = -1
            prev_position = -1
        else:
            # Maintain previous position when no new signal
            weekly_df.iloc[i, weekly_df.columns.get_loc('Position')] = prev_position
    
    # Shift positions by 1 week to implement signal lag
    weekly_df['Position'] = weekly_df['Position'].shift(1)
    
    # Initialize Portfolio Value
    weekly_df['Portfolio_Value'] = initial_capital
    
    # Apply stop loss with 2%
    weekly_df = apply_stop_loss(weekly_df, stop_loss_pct=0.02)
    
    # Recalculate returns after stop loss
    weekly_df = calculate_strategy_returns(weekly_df)
    
    # Save DataFrame to CSV
    output_file = f'data/weekly_macd_zero_cross_{symbol.replace("^", "")}.csv'
    weekly_df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    
    return weekly_df

def get_vpvma_signals(symbol='^GSPC', start_date='2005-01-01', end_date='2023-12-31', initial_capital=1_000_000):
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download S&P 500 data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Download VIX data
    vix = yf.Ticker('^VIX')
    vix_df = vix.history(start=start_date, end=end_date)
    
    # Convert timezone from UTC to US/Eastern
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert('US/Eastern')
    vix_df.index = pd.to_datetime(vix_df.index)
    vix_df.index = vix_df.index.tz_convert('US/Eastern')
    
    # Resample to weekly data
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # Resample VIX to weekly data (using average VIX for the week)
    weekly_vix = vix_df.resample('W').agg({
        'Close': 'mean'  # Using mean of VIX values for the week
    })
    
    # Calculate typical price
    weekly_df['Typical_Price'] = (weekly_df['High'] + weekly_df['Low'] + weekly_df['Close']) / 3
    
    # Use VIX as volatility proxy (divide by 100 to convert percentage to decimal)
    weekly_df['Volatility'] = weekly_vix['Close'] / 100
    
    # Calculate volume-weighted typical price
    weekly_df['Vol_Weighted_Price'] = weekly_df['Typical_Price'] * weekly_df['Volume']
    
    # Calculate short-term (12-week) and long-term (26-week) VPVMA
    short_window = 12
    long_window = 26
    signal_window = 9
    
    # Calculate volume-weighted moving averages
    vwp_short = weekly_df['Vol_Weighted_Price'].rolling(window=short_window).sum() / \
                weekly_df['Volume'].rolling(window=short_window).sum()
    vwp_long = weekly_df['Vol_Weighted_Price'].rolling(window=long_window).sum() / \
               weekly_df['Volume'].rolling(window=long_window).sum()
    
    # Multiply by volatility and apply EMA smoothing
    vpvma = (vwp_short * weekly_df['Volatility']).ewm(span=short_window, adjust=False).mean()
    vpvma_long = (vwp_long * weekly_df['Volatility']).ewm(span=long_window, adjust=False).mean()
    
    # Calculate VPVMA (similar to MACD)
    weekly_df['VPVMA'] = vpvma - vpvma_long
    weekly_df['VPVMA_Signal'] = weekly_df['VPVMA'].ewm(span=signal_window, adjust=False).mean()
    weekly_df['VPVMA_Histogram'] = weekly_df['VPVMA'] - weekly_df['VPVMA_Signal']
    
    # Generate buy/sell signals
    weekly_df['Position'] = 0
    weekly_df['Position'] = weekly_df['Position'].mask(
        (weekly_df['VPVMA'] > weekly_df['VPVMA_Signal']), 1)
    weekly_df['Position'] = weekly_df['Position'].mask(
        (weekly_df['VPVMA'] < weekly_df['VPVMA_Signal']), -1)
    
    # Shift positions by 1 week to implement signal lag
    weekly_df['Position'] = weekly_df['Position'].shift(1)
    
    # Initialize Portfolio Value
    weekly_df['Portfolio_Value'] = initial_capital
    
    # Apply stop loss with 2%
    weekly_df = apply_stop_loss(weekly_df, stop_loss_pct=0.02)
    
    # Recalculate returns after stop loss
    weekly_df = calculate_strategy_returns(weekly_df)
    
    # Save DataFrame to CSV
    output_file = f'data/weekly_vpvma_signals_{symbol.replace("^", "")}.csv'
    weekly_df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    
    return weekly_df

def get_vpvma_signals_zero_cross(symbol='^GSPC', start_date='2005-01-01', end_date='2023-12-31', initial_capital=1_000_000):
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download S&P 500 data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    # Download VIX data
    vix = yf.Ticker('^VIX')
    vix_df = vix.history(start=start_date, end=end_date)
    
    # Convert timezone from UTC to US/Eastern
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert('US/Eastern')
    vix_df.index = pd.to_datetime(vix_df.index)
    vix_df.index = vix_df.index.tz_convert('US/Eastern')
    
    # Resample to weekly data
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # Resample VIX to weekly data
    weekly_vix = vix_df.resample('W').agg({
        'Close': 'mean'
    })
    
    # Calculate typical price
    weekly_df['Typical_Price'] = (weekly_df['High'] + weekly_df['Low'] + weekly_df['Close']) / 3
    
    # Use VIX as volatility proxy
    weekly_df['Volatility'] = weekly_vix['Close'] / 100
    
    # Calculate volume-weighted typical price
    weekly_df['Vol_Weighted_Price'] = weekly_df['Typical_Price'] * weekly_df['Volume']
    
    # Calculate short-term and long-term VPVMA
    short_window = 12
    long_window = 26
    signal_window = 9
    
    # Calculate volume-weighted moving averages
    vwp_short = weekly_df['Vol_Weighted_Price'].rolling(window=short_window).sum() / \
                weekly_df['Volume'].rolling(window=short_window).sum()
    vwp_long = weekly_df['Vol_Weighted_Price'].rolling(window=long_window).sum() / \
               weekly_df['Volume'].rolling(window=long_window).sum()
    
    # Multiply by volatility and apply EMA smoothing
    vpvma = (vwp_short * weekly_df['Volatility']).ewm(span=short_window, adjust=False).mean()
    vpvma_long = (vwp_long * weekly_df['Volatility']).ewm(span=long_window, adjust=False).mean()
    
    # Calculate VPVMA
    weekly_df['VPVMA'] = vpvma - vpvma_long
    weekly_df['VPVMA_Signal'] = weekly_df['VPVMA'].ewm(span=signal_window, adjust=False).mean()
    weekly_df['VPVMA_Histogram'] = weekly_df['VPVMA'] - weekly_df['VPVMA_Signal']
    
    # Generate buy/sell signals with zero-line condition
    weekly_df['Position'] = 0
    
    # Previous position to maintain when no new signal
    prev_position = 0
    
    for i in range(len(weekly_df)):
        # Buy signal: VPVMA crosses above signal line AND VPVMA is above zero
        if (weekly_df['VPVMA'].iloc[i] > weekly_df['VPVMA_Signal'].iloc[i]) and (weekly_df['VPVMA'].iloc[i] > 0):
            weekly_df.iloc[i, weekly_df.columns.get_loc('Position')] = 1
            prev_position = 1
        # Sell signal: VPVMA crosses below signal line AND VPVMA crosses below zero
        elif (weekly_df['VPVMA'].iloc[i] < weekly_df['VPVMA_Signal'].iloc[i]) and (weekly_df['VPVMA'].iloc[i] < 0):
            weekly_df.iloc[i, weekly_df.columns.get_loc('Position')] = -1
            prev_position = -1
        else:
            # Maintain previous position when no new signal
            weekly_df.iloc[i, weekly_df.columns.get_loc('Position')] = prev_position
    
    # Shift positions by 1 week to implement signal lag
    weekly_df['Position'] = weekly_df['Position'].shift(1)
    
    # Initialize Portfolio Value
    weekly_df['Portfolio_Value'] = initial_capital
    
    # Apply stop loss with 2%
    weekly_df = apply_stop_loss(weekly_df, stop_loss_pct=0.02)
    
    # Recalculate returns after stop loss
    weekly_df = calculate_strategy_returns(weekly_df)
    
    # Save DataFrame to CSV
    output_file = f'data/weekly_vpvma_zero_cross_{symbol.replace("^", "")}.csv'
    weekly_df.to_csv(output_file)
    print(f"Data saved to {output_file}")
    
    return weekly_df
