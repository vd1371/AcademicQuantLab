import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from Signals import (
    get_macd_signals,
    get_macd_signals_zero_cross,
    get_vpvma_signals,
    get_vpvma_signals_zero_cross
)
import os

def calculate_performance_metrics(df):
    """Calculate various trading performance metrics"""
    
    # Number of Trades
    trades = df['Position_Change'].fillna(0)
    num_trades = len(trades[trades != 0])
    
    # Win Ratio - only count returns when position changes
    position_changes = df[df['Position_Change'] != 0]
    winning_trades = len(position_changes[position_changes['Strategy_Returns'] > 0])
    win_ratio = winning_trades / num_trades if num_trades > 0 else 0
    
    # Profit & Loss
    cumulative_returns = (1 + df['Strategy_Returns']).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    
    # Calculate annual return
    years = (df.index[-1] - df.index[0]).days / 365.25
    annual_return = (cumulative_returns.iloc[-1] ** (1/years)) - 1
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    mean_returns = df['Strategy_Returns'].mean()
    std_returns = df['Strategy_Returns'].std()
    sharpe_ratio = np.sqrt(52) * mean_returns / std_returns if std_returns != 0 else 0
    
    # Maximum Drawdown calculation
    portfolio_value = df['Portfolio_Value']
    peak = portfolio_value.expanding(min_periods=1).max()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Portfolio Value metrics
    initial_value = df['Portfolio_Value'].iloc[0]
    final_value = df['Portfolio_Value'].iloc[-1]
    portfolio_return = ((final_value - initial_value) / initial_value) * 100
    
    return {
        'Number of Trades': num_trades,
        'Win Ratio': f"{win_ratio:.2%}",
        'Total Return': f"{total_return:.2f}%",
        'Annual Return': f"{annual_return*100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2f}%",
        'Initial Portfolio Value': f"${initial_value:,.2f}",
        'Final Portfolio Value': f"${final_value:,.2f}",
        'Portfolio Return': f"{portfolio_return:.2f}%"
    }

def plot_macd_signals(df):
    """Plot MACD signals and price movements"""
    # Create figure with secondary y-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Plot price
    ax1.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.6)
    
    # Plot buy/sell signals
    buy_signals = df[df['Position_Change'] == 1].index
    sell_signals = df[df['Position_Change'] == -2].index  # From 1 to -1
    ax1.scatter(buy_signals, df.loc[buy_signals, 'Close'], marker='^', color='green', label='Buy Signal')
    ax1.scatter(sell_signals, df.loc[sell_signals, 'Close'], marker='v', color='red', label='Sell Signal')
    
    ax1.set_title('Price Movement and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Plot MACD
    ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax2.plot(df.index, df['Signal_Line'], label='Signal Line', color='orange')
    ax2.bar(df.index, df['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
    ax2.set_title('MACD Indicator')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_performance(df):
    """Plot strategy performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cumulative returns comparison
    strategy_cum_returns = (1 + df['Strategy_Returns']).cumprod()
    market_cum_returns = (1 + df['Returns']).cumprod()
    ax1.plot(df.index, strategy_cum_returns, label='Strategy Returns', color='blue')
    ax1.plot(df.index, market_cum_returns, label='Market Returns', color='gray', alpha=0.6)
    ax1.set_title('Cumulative Returns')
    ax1.legend()
    
    # Monthly returns heatmap
    monthly_returns = df['Strategy_Returns'].groupby([df.index.year, df.index.month]).sum().unstack()
    sns.heatmap(monthly_returns, ax=ax2, cmap='RdYlGn', center=0, annot=True, fmt='.2%')
    ax2.set_title('Monthly Returns Heatmap')
    
    # Rolling Sharpe ratio (252-day)
    rolling_sharpe = (df['Strategy_Returns'].rolling(252).mean() / 
                     df['Strategy_Returns'].rolling(252).std() * np.sqrt(252))
    ax3.plot(df.index, rolling_sharpe)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('Rolling Sharpe Ratio (252-day)')
    
    # Drawdown analysis
    strategy_cum_returns = (1 + df['Strategy_Returns']).cumprod()
    rolling_max = strategy_cum_returns.expanding().max()
    drawdowns = (strategy_cum_returns - rolling_max) / rolling_max
    ax4.fill_between(df.index, drawdowns, 0, color='red', alpha=0.3)
    ax4.set_title('Drawdown Analysis')
    
    plt.tight_layout()
    plt.show()

def get_trade_info(df, strategy_name):
    """Extract trade information from the signals DataFrame"""
    trades = []
    position = 0
    entry_price = 0
    entry_date = None
    
    for date, row in df.iterrows():
        if row['Position_Change'] != 0:
            # Case 1: Opening a new position from neutral
            if position == 0:
                position = row['Position']
                entry_price = row['Close']
                entry_date = date
            # Case 2: Direct switch between long and short positions
            elif (position == 1 and row['Position'] == -1) or (position == -1 and row['Position'] == 1):
                # Close current position
                exit_price = row['Close']
                pnl = position * (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Position': 'Long' if position == 1 else 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'PnL %': pnl
                })
                # Open new position
                position = row['Position']
                entry_price = row['Close']
                entry_date = date
            # Case 3: Closing a position to neutral
            elif row['Position'] == 0:
                exit_price = row['Close']
                pnl = position * (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Position': 'Long' if position == 1 else 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'PnL %': pnl
                })
                position = 0
    
    trades_df = pd.DataFrame(trades)
    
    # Save trade information to CSV with strategy name
    output_file = f'data/trade_info_{strategy_name}.csv'
    trades_df.to_csv(output_file, index=False)
    print(f"\nTrade information for {strategy_name} saved to {output_file}")
    print(f"Total trades: {len(trades_df)}")
    print(f"Long trades: {len(trades_df[trades_df['Position'] == 'Long'])}")
    print(f"Short trades: {len(trades_df[trades_df['Position'] == 'Short'])}")
    
    return trades_df

if __name__ == "__main__":
    # Original MACD strategy
    signals_df = get_macd_signals()
    
    # Calculate market metrics
    market_df = signals_df.copy()
    market_df['Position'] = 1  # Market is always long
    market_df['Position_Change'] = 0  # Market maintains position
    market_df['Strategy_Returns'] = market_df['Returns']  # Use market returns directly
    market_df['Portfolio_Value'] = 1_000_000  # Set initial portfolio value
    market_df['Portfolio_Returns'] = market_df['Returns']  # Use market returns for portfolio
    market_df['Portfolio_Value'] = market_df['Portfolio_Value'].iloc[0] * (1 + market_df['Portfolio_Returns']).cumprod()
    market_metrics = calculate_performance_metrics(market_df)
    
    # Calculate strategy metrics
    trades_df = get_trade_info(signals_df, 'macd_original')
    metrics = calculate_performance_metrics(signals_df)
    
    # Zero-cross MACD strategy
    signals_df_zero = get_macd_signals_zero_cross()
    trades_df_zero = get_trade_info(signals_df_zero, 'macd_zero_cross')
    metrics_zero = calculate_performance_metrics(signals_df_zero)
    
    # VPVMA strategy
    signals_df_vpvma = get_vpvma_signals()
    trades_df_vpvma = get_trade_info(signals_df_vpvma, 'vpvma')
    metrics_vpvma = calculate_performance_metrics(signals_df_vpvma)
    
    # VPVMA zero-cross strategy
    signals_df_vpvma_zero = get_vpvma_signals_zero_cross()
    trades_df_vpvma_zero = get_trade_info(signals_df_vpvma_zero, 'vpvma_zero_cross')
    metrics_vpvma_zero = calculate_performance_metrics(signals_df_vpvma_zero)
    
    # Collect all metrics in a dictionary
    all_metrics = {
        'Market (Buy & Hold)': market_metrics,
        'Original MACD': metrics,
        'Zero-Cross MACD': metrics_zero,
        'VPVMA': metrics_vpvma,
        'VPVMA Zero-Cross': metrics_vpvma_zero
    }
    
    # Convert to DataFrame and display
    metrics_df = pd.DataFrame(all_metrics).T
    print("\nStrategy Performance Comparison:")
    print(metrics_df.to_string())
    
    # Save metrics to CSV
    os.makedirs('data', exist_ok=True)
    metrics_df.to_csv('data/strategy_metrics_comparison.csv')
    
    # Plot results for all strategies
    plt.figure(figsize=(15, 10))
    plt.title("Strategy Comparison")
    
    # Calculate and plot market returns (buy and hold)
    market_returns = (1 + signals_df['Returns']).cumprod()
    plt.plot(signals_df.index, market_returns, label='Market (Buy & Hold)', color='gray', linestyle='--', alpha=0.7)
    
    # Plot strategy returns
    plt.plot(signals_df.index, (1 + signals_df['Strategy_Returns']).cumprod(), label='Original MACD')
    plt.plot(signals_df_zero.index, (1 + signals_df_zero['Strategy_Returns']).cumprod(), label='Zero-Cross MACD')
    plt.plot(signals_df_vpvma.index, (1 + signals_df_vpvma['Strategy_Returns']).cumprod(), label='VPVMA')
    plt.plot(signals_df_vpvma_zero.index, (1 + signals_df_vpvma_zero['Strategy_Returns']).cumprod(), label='VPVMA Zero-Cross')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.yscale('log')  # Using log scale for better visualization of returns
    plt.tight_layout()
    
    # Save plot to data folder
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
