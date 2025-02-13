import pandas as pd
import numpy as np

def calculate_vpvma(df, vix_df):
    """Calculate VPVMA indicator"""
    # Calculate Volume Price Trend
    pct_change = df['Close'].pct_change().fillna(0)  # Fill first NaN with 0
    vpt = (pct_change * df['Volume']).cumsum()
    
    # Calculate VIX-adjusted moving average
    vix_ma = vix_df['Close'].rolling(window=2).mean().bfill()  # Changed to bfill()
    window = max(2, int(vix_ma.mean()))  # Ensure minimum window of 2
    vpvma = vpt.rolling(window=window, min_periods=1).mean()
    
    return pd.DataFrame({
        'VPVMA': vpvma,
        'VPT': vpt
    }, index=df.index)

def generate_vpvma_signals(df, vix_df):
    """Generate VPVMA trading signals"""
    # Calculate VPVMA
    vpvma_data = calculate_vpvma(df, vix_df)
    
    # Generate signals
    signals = pd.DataFrame(index=df.index)
    signals['Date'] = df.index
    signals['Close'] = df['Close']
    signals['VPVMA'] = vpvma_data['VPVMA']  # Only select the VPVMA column
    
    # Generate trading signals (1 for buy, 0 for hold/sell)
    signals['Signal'] = 0
    signals.loc[df['Close'] > vpvma_data['VPVMA'], 'Signal'] = 1  # Update comparison to use VPVMA column
    
    # Calculate portfolio value
    signals['Portfolio_Value'] = 1000000.0  # Initial capital as float
    position = 0
    
    for i in range(1, len(signals)):
        prev_value = signals.loc[signals.index[i-1], 'Portfolio_Value']
        returns = signals.loc[signals.index[i], 'Close'] / signals.loc[signals.index[i-1], 'Close'] - 1
        
        if signals.loc[signals.index[i], 'Signal'] == 1 and position == 0:  # Buy signal
            position = 1
            signals.loc[signals.index[i], 'Portfolio_Value'] = prev_value * (1 + returns)
        elif signals.loc[signals.index[i], 'Signal'] == 0 and position == 1:  # Sell signal
            position = 0
            signals.loc[signals.index[i], 'Portfolio_Value'] = prev_value
        else:  # Hold
            signals.loc[signals.index[i], 'Portfolio_Value'] = prev_value * (1 + (returns if position == 1 else 0))
    
    return signals 