import pytest
import pandas as pd
import numpy as np
from src.strategies.vpvma import calculate_vpvma, generate_vpvma_signals

def create_sample_data():
    return pd.DataFrame({
        'Close': [100, 102, 104, 103, 102, 103, 105, 107, 108],
        'Volume': [1000, 1200, 800, 1100, 900, 1300, 1000, 1100, 900]
    })

def create_sample_vix():
    return pd.DataFrame({
        'Close': [15, 16, 14, 17, 18, 15, 14, 13, 15]
    })

def test_calculate_vpvma():
    df = create_sample_data()
    vix_df = create_sample_vix()
    result = calculate_vpvma(df, vix_df)
    
    assert isinstance(result, pd.DataFrame)
    assert 'VPVMA' in result.columns
    assert len(result) == len(df)

def test_generate_vpvma_signals():
    df = create_sample_data()
    vix_df = create_sample_vix()
    result = generate_vpvma_signals(df, vix_df)
    
    assert 'Position' in result.columns
    assert result['Position'].isin([1, 0, -1]).all() 