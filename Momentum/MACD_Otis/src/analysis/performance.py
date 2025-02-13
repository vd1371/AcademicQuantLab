import pandas as pd
import numpy as np

def calculate_performance_metrics(df):
    """Calculate performance metrics for a strategy"""
    returns = df['Portfolio_Returns'].dropna()
    
    metrics = {
        'Number of Trades': len(df[df['Position'] != df['Position'].shift(1)]),
        'Win Ratio': f"{(returns[returns > 0].count() / returns.count() * 100):.2f}%",
        'Total Return': f"{(df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0] - 1) * 100:.2f}%",
        'Annual Return': f"{((df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) ** (252/len(df)) - 1) * 100:.2f}%",
        'Sharpe Ratio': f"{np.sqrt(52) * returns.mean() / returns.std():.2f}",
        'Maximum Drawdown': f"{((df['Portfolio_Value'] - df['Portfolio_Value'].cummax()) / df['Portfolio_Value'].cummax()).min() * 100:.2f}%",
        'Initial Portfolio Value': f"${df['Portfolio_Value'].iloc[0]:,.2f}",
        'Final Portfolio Value': f"${df['Portfolio_Value'].iloc[-1]:,.2f}",
    }
    
    return pd.Series(metrics) 