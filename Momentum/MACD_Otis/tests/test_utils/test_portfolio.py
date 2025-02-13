import pytest
import pandas as pd
import os
from src.utils.portfolio import construct_portfolio_performance

def create_mock_data(tmp_path):
    # Create mock strategy statistics
    stats_df = pd.DataFrame({
        'ETF': ['TEST1', 'TEST2'],
        'Strategy': ['MACD_Standard', 'VPVMA_Standard']
    }).set_index('ETF')
    
    os.makedirs(os.path.join(tmp_path, 'data/results'), exist_ok=True)
    stats_df.to_csv(os.path.join(tmp_path, 'data/results/strategy_statistics.csv'))
    
    # Create mock signal data
    for etf in ['TEST1', 'TEST2']:
        os.makedirs(os.path.join(tmp_path, 'data/processed', etf), exist_ok=True)
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Portfolio_Value': [1000000] * 10
        })
        df.to_csv(os.path.join(tmp_path, 'data/processed', etf, 'weekly_macd_standard_signals.csv'))

def test_construct_portfolio_performance(tmp_path):
    create_mock_data(tmp_path)
    portfolio_value, etf_values = construct_portfolio_performance(['TEST1', 'TEST2'], 
                                                                data_dir=os.path.join(tmp_path, 'data/results'))
    
    assert isinstance(portfolio_value, pd.Series)
    assert isinstance(etf_values, dict)
    assert len(etf_values) == 2
    assert all(isinstance(v, pd.Series) for v in etf_values.values()) 