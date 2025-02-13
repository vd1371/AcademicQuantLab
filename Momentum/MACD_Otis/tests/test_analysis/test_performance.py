import pytest
import pandas as pd
import numpy as np
from src.analysis.performance import calculate_performance_metrics

def test_calculate_performance_metrics():
    df = pd.DataFrame({
        'Portfolio_Value': [1000000, 1100000, 1050000, 1200000, 1150000],
        'Portfolio_Returns': [0, 0.1, -0.045, 0.143, -0.042],
        'Position': [0, 1, 1, -1, 0]
    })
    
    metrics = calculate_performance_metrics(df)
    
    assert isinstance(metrics, pd.Series)
    assert 'Number of Trades' in metrics
    assert 'Win Ratio' in metrics
    assert 'Total Return' in metrics
    assert 'Sharpe Ratio' in metrics
    assert 'Maximum Drawdown' in metrics 