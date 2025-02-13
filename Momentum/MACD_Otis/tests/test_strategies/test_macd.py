import pytest
import pandas as pd
import numpy as np
from src.strategies.macd import calculate_macd, generate_macd_signals

def create_sample_data():
    return pd.DataFrame({
        'Close': [100, 102, 104, 103, 102, 103, 105, 107, 108],
        'Open': [99, 100, 102, 104, 103, 102, 103, 105, 107],
        'High': [101, 103, 105, 104, 103, 104, 106, 108, 109],
        'Low': [98, 99, 101, 102, 101, 101, 102, 104, 106],
        'Volume': [1000] * 9
    })

def test_calculate_macd():
    df = create_sample_data()
    result = calculate_macd(df)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['MACD', 'Signal', 'MACD_Hist'])
    assert len(result) == len(df)
    assert not result['MACD'].isna().any()

def test_generate_macd_signals_standard():
    df = create_sample_data()
    result = generate_macd_signals(df, zero_cross=False)
    
    assert 'Position' in result.columns
    assert result['Position'].isin([1, 0, -1]).all()
    assert result['Position'].iloc[0] == 0  # First position should be 0 due to shift

def test_generate_macd_signals_zero_cross():
    df = create_sample_data()
    result = generate_macd_signals(df, zero_cross=True)
    
    assert 'Position' in result.columns
    assert result['Position'].isin([1, 0, -1]).all() 