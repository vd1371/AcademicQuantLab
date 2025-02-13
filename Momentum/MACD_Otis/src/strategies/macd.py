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

def generate_macd_signals(df, zero_cross=False):
    """Generate MACD trading signals"""
    macd_data = calculate_macd(df)
    
    if zero_cross:
        # Zero-cross strategy
        df['Position'] = 0
        df['Position'] = df['Position'].mask(macd_data['MACD'] > 0, 1)
        df['Position'] = df['Position'].mask(macd_data['MACD'] < 0, -1)
    else:
        # Signal line cross strategy
        df['Position'] = 0
        df['Position'] = df['Position'].mask(macd_data['MACD'] > macd_data['Signal'], 1)
        df['Position'] = df['Position'].mask(macd_data['MACD'] < macd_data['Signal'], -1)
    
    df['Position'] = df['Position'].shift(1)  # Implement signal lag
    return df 