import pytest
import pandas as pd
import os
from src.utils.data_loader import load_etf_data, save_signals

def test_load_etf_data(mocker):
    mock_history = pd.DataFrame({
        'Open': [100] * 5,
        'High': [105] * 5,
        'Low': [95] * 5,
        'Close': [102] * 5,
        'Volume': [1000] * 5
    }, index=pd.date_range('2023-01-01', periods=5))
    
    mock_ticker = mocker.Mock()
    mock_ticker.history.return_value = mock_history
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)
    
    result = load_etf_data('TEST')
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_save_signals(tmp_path):
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=5),
        'Close': [100, 101, 102, 103, 104],
        'Position': [0, 1, 1, -1, 0]
    })
    
    save_signals(df, 'TEST', 'macd')
    assert os.path.exists(os.path.join('data', 'processed', 'TEST', 'weekly_macd_signals.csv')) 