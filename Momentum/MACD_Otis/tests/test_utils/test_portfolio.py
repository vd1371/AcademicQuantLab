import pytest
import pandas as pd
import os
from src.utils.portfolio import construct_portfolio_performance

def create_mock_data(tmp_path):
    # Create mock strategy statistics
    stats_df = pd.DataFrame({
        'ETF': ['TEST1', 'TEST2'],
        'Strategy': ['macd_standard', 'vpvma_standard']
    })
    
    # Create directories
    results_dir = os.path.join(tmp_path, 'data', 'results')
    processed_dir = os.path.join(tmp_path, 'data', 'processed')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save strategy statistics with ETF as index
    stats_df.set_index('ETF', inplace=True)
    stats_df.to_csv(os.path.join(results_dir, 'strategy_statistics.csv'))
    
    # Create mock signal data
    dates = pd.date_range('2023-01-01', periods=10)
    for etf in ['TEST1', 'TEST2']:
        etf_dir = os.path.join(processed_dir, etf)
        os.makedirs(etf_dir, exist_ok=True)
        
        df = pd.DataFrame({
            'Date': dates,
            'Portfolio_Value': [1000000] * 10,
            'ETF': [etf] * 10
        })
        
        # Save signal data for the strategy specified in stats_df
        strategy = stats_df.loc[etf, 'Strategy']
        df.to_csv(os.path.join(etf_dir, f'weekly_{strategy}_signals.csv'), index=False)

@pytest.fixture
def mock_data_path(tmp_path):
    create_mock_data(tmp_path)
    return tmp_path

def test_construct_portfolio_performance(tmp_path, monkeypatch):
    # Create mock data
    create_mock_data(tmp_path)
    
    # Patch the data directory paths
    def mock_path_handling(path):
        if path.startswith('data/processed'):
            return os.path.join(tmp_path, path)
        return path
    
    def safe_path_join(*args):
        # Store the original join function
        original_join = os.path.join
        # Temporarily restore original join to avoid recursion
        monkeypatch.undo()
        try:
            # Use the original join function
            result = original_join(*args)
            # Apply any mock path handling if needed
            return mock_path_handling(result)
        finally:
            # Restore the mock
            monkeypatch.setattr(os.path, 'join', safe_path_join)

    monkeypatch.setattr(os.path, 'join', safe_path_join)
    
    # Run test
    portfolio_value, etf_values = construct_portfolio_performance(
        ['TEST1', 'TEST2'],
        data_dir=os.path.join(tmp_path, 'data', 'results')
    )
    
    assert portfolio_value is not None
    assert isinstance(portfolio_value, pd.Series)
    assert isinstance(etf_values, dict)
    assert len(etf_values) == 2
    assert all(isinstance(v, pd.Series) for v in etf_values.values()) 