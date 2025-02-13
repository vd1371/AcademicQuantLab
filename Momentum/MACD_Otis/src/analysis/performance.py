import pandas as pd
import numpy as np

def calculate_performance_metrics(df):
    """Calculate comprehensive performance metrics for a strategy"""
    # Convert Portfolio_Value to returns
    returns = df['Portfolio_Value'].pct_change().fillna(0)
    
    # Calculate basic return metrics
    total_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Calculate volatility and risk metrics
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(52)  # Assuming weekly data
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Calculate win rate
    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Number of Trades': total_trades
    } 