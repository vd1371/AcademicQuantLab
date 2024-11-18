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
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

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

def get_trade_info(df, strategy_name, ticker):
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
    
    # Create ticker-specific directory
    ticker_dir = os.path.join('data', ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Save trade information to CSV in ticker directory
    output_file = os.path.join(ticker_dir, f'trade_info_{strategy_name}.csv')
    trades_df.to_csv(output_file, index=False)
    print(f"\nTrade information for {strategy_name} saved to {output_file}")
    print(f"Total trades: {len(trades_df)}")
    print(f"Long trades: {len(trades_df[trades_df['Position'] == 'Long'])}")
    print(f"Short trades: {len(trades_df[trades_df['Position'] == 'Short'])}")
    
    return trades_df

def process_etf(etf):
    """Process a single ETF and return its results"""
    try:
        # Create ETF-specific directory
        ticker_dir = os.path.join('data', etf)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Download and process data using functions from Signals.py
        df = yf.Ticker(etf)
        df_hist = df.history(start='2005-01-01', end='2023-12-31')
        vix = yf.Ticker('^VIX')
        vix_df = vix.history(start='2005-01-01', end='2023-12-31')
        
        # Dictionary to store results for different strategies
        results = {}
        sharpe_ratios = {}
        
        # Test different MACD strategies
        strategies = {
            'MACD_Standard': lambda: get_macd_signals(df=df_hist.copy(), symbol=etf),
            'MACD_Zero_Cross': lambda: get_macd_signals_zero_cross(df=df_hist.copy(), symbol=etf),
            'VPVMA_Standard': lambda: get_vpvma_signals(df=df_hist.copy(), vix_df=vix_df.copy(), symbol=etf),
            'VPVMA_Zero_Cross': lambda: get_vpvma_signals_zero_cross(df=df_hist.copy(), vix_df=vix_df.copy(), symbol=etf)
        }
        
        for name, strategy_func in strategies.items():
            # Apply strategy
            signals_df = strategy_func()
            
            # Calculate metrics
            metrics = calculate_performance_metrics(signals_df)
            results[name] = metrics
            sharpe_ratios[name] = float(metrics['Sharpe Ratio'].replace(',', ''))
            
            # Save trade information
            get_trade_info(signals_df, name, etf)
        
        # Determine best strategy based on Sharpe ratio
        best_strategy = max(sharpe_ratios.items(), key=lambda x: x[1])[0]
        
        print(f"\nProcessed {etf}")
        return results, best_strategy, sharpe_ratios
        
    except Exception as e:
        print(f"Error processing {etf}: {str(e)}")
        return None

if __name__ == "__main__":
    # List of ETFs to analyze
    etfs = [
        'EEM',  # Emerging Markets
        'VWO',  # Emerging Markets
        'FXI',  # China Large-Cap
        'AAXJ', # Asia ex-Japan
        'EWJ',  # Japan
        'ACWX', # All Country World ex-US
        'CHIX', # China Technology
        'CQQQ', # China Technology
        'EWZ',  # Brazil
        'ERUS', # Russia
        'EWC',  # Canada
        'EWU',  # United Kingdom
        'VGK',  # Europe
        'VPL'   # Pacific
    ]
    
    # Store results for all ETFs
    all_results = {}
    best_strategies = {}
    all_sharpe_ratios = {}
    
    # Process ETFs in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_etf, etf): etf for etf in etfs}
        for future in as_completed(futures):
            etf = futures[future]
            try:
                result = future.result()
                if result:
                    results, best_strategy, sharpe_ratios = result
                    best_strategies[etf] = best_strategy
                    all_sharpe_ratios[etf] = sharpe_ratios
                    all_results[etf] = results  # Store all results
            except Exception as e:
                print(f"Error processing {etf}: {str(e)}")
    
    # Create summary of best strategies and key stats
    summary_dir = os.path.join('data', 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save best strategies summary
    with open(os.path.join(summary_dir, 'best_strategies.txt'), 'w') as f:
        f.write("Best Strategy by ETF\n")
        f.write("=" * 50 + "\n\n")
        for etf, strategy in best_strategies.items():
            f.write(f"{etf}: {strategy} (Sharpe: {all_sharpe_ratios[etf][strategy]:.2f})\n")
            f.write(f"Key Statistics:\n")
            f.write("-" * 30 + "\n")
            for metric, value in all_results[etf][strategy].items():
                f.write(f"{metric}: {value}\n")
            f.write("\n")
    
    # Create DataFrame of all Sharpe ratios
    sharpe_df = pd.DataFrame(all_sharpe_ratios).T
    sharpe_df.to_csv(os.path.join(summary_dir, 'sharpe_ratios.csv'))
    
    # Create summary statistics DataFrame
    summary_stats = []
    for etf in etfs:
        if etf in all_results:
            best_strat = best_strategies[etf]
            stats = all_results[etf][best_strat]
            stats['ETF'] = etf
            stats['Strategy'] = best_strat
            summary_stats.append(stats)
    
    stats_df = pd.DataFrame(summary_stats)
    stats_df.set_index('ETF', inplace=True)
    stats_df.to_csv(os.path.join(summary_dir, 'strategy_statistics.csv'))
    
    print("\nAnalysis complete. Results saved in data/summary folder.")
