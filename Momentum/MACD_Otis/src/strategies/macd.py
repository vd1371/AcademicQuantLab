import pandas as pd
import numpy as np
import os

def calculate_macd(df):
    """Calculate MACD indicators"""
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal,
        'MACD_Hist': macd - signal
    }, index=df.index)

def generate_macd_signals(df):
    """Generate MACD trading signals"""
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    # Generate signals
    signals = pd.DataFrame(index=df.index)
    signals['Date'] = df.index
    signals['Close'] = df['Close']
    signals['MACD'] = macd
    signals['Signal_Line'] = signal_line
    
    # Generate trading signals (1 for buy, 0 for hold/sell)
    signals['Signal'] = 0
    signals.loc[macd > signal_line, 'Signal'] = 1
    
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