import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_returns_comparison(perf_df, output_dir='data/visualizations'):
    """Plot comparison of returns across ETFs"""
    plt.figure(figsize=(15, 8))
    
    # Create bar plot for annual returns
    annual_returns = pd.to_numeric(perf_df['Annual Return'].str.rstrip('%')) / 100
    
    # Split into sectors and bonds
    sector_mask = annual_returns.index.str.startswith('XL')
    
    # Plot sectors and bonds with different colors
    plt.bar(np.where(sector_mask)[0], annual_returns[sector_mask], color='blue', alpha=0.6, label='Sector ETFs')
    plt.bar(np.where(~sector_mask)[0], annual_returns[~sector_mask], color='green', alpha=0.6, label='Bond ETFs')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(len(annual_returns)), annual_returns.index, rotation=45)
    plt.title('Annual Returns by ETF')
    plt.ylabel('Annual Return')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{output_dir}/annual_returns.png')
    plt.close()

def plot_risk_return_scatter(perf_df, output_dir='data/visualizations'):
    """Create risk-return scatter plot"""
    plt.figure(figsize=(12, 8))
    
    # Convert string percentages to float
    returns = pd.to_numeric(perf_df['Annual Return'].str.rstrip('%')) / 100
    volatility = pd.to_numeric(perf_df['Annualized Volatility'].str.rstrip('%')) / 100
    sharpe = pd.to_numeric(perf_df['Sharpe Ratio'])
    
    # Create scatter plot
    sector_mask = returns.index.str.startswith('XL')
    
    plt.scatter(volatility[sector_mask], returns[sector_mask], 
               s=sharpe[sector_mask]*100, alpha=0.6, 
               c='blue', label='Sector ETFs')
    plt.scatter(volatility[~sector_mask], returns[~sector_mask], 
               s=sharpe[~sector_mask]*100, alpha=0.6,
               c='green', label='Bond ETFs')
    
    # Add labels for each point
    for idx, (vol, ret) in enumerate(zip(volatility, returns)):
        plt.annotate(volatility.index[idx], (vol, ret), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annual Return')
    plt.title('Risk-Return Profile (bubble size = Sharpe ratio)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{output_dir}/risk_return_profile.png')
    plt.close()

def plot_win_rates(perf_df, output_dir='data/visualizations'):
    """Plot win rates across ETFs"""
    plt.figure(figsize=(15, 8))
    
    # Convert win rates to numeric
    win_rates = pd.to_numeric(perf_df['Win Rate'].str.rstrip('%')) / 100
    
    # Split into sectors and bonds
    sector_mask = win_rates.index.str.startswith('XL')
    
    # Plot sectors and bonds
    plt.bar(np.where(sector_mask)[0], win_rates[sector_mask], color='blue', alpha=0.6, label='Sector ETFs')
    plt.bar(np.where(~sector_mask)[0], win_rates[~sector_mask], color='green', alpha=0.6, label='Bond ETFs')
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% Threshold')
    plt.xticks(range(len(win_rates)), win_rates.index, rotation=45)
    plt.title('Win Rates by ETF')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{output_dir}/win_rates.png')
    plt.close()

def plot_drawdowns(perf_df, output_dir='data/visualizations'):
    """Plot maximum drawdowns"""
    plt.figure(figsize=(15, 8))
    
    # Convert drawdowns to numeric
    drawdowns = pd.to_numeric(perf_df['Max Drawdown'].str.rstrip('%')) / 100
    
    # Split into sectors and bonds
    sector_mask = drawdowns.index.str.startswith('XL')
    
    # Plot sectors and bonds
    plt.bar(np.where(sector_mask)[0], drawdowns[sector_mask], color='blue', alpha=0.6, label='Sector ETFs')
    plt.bar(np.where(~sector_mask)[0], drawdowns[~sector_mask], color='green', alpha=0.6, label='Bond ETFs')
    
    plt.xticks(range(len(drawdowns)), drawdowns.index, rotation=45)
    plt.title('Maximum Drawdowns by ETF')
    plt.ylabel('Maximum Drawdown')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{output_dir}/max_drawdowns.png')
    plt.close()

def plot_cumulative_returns(signals_dict, output_dir='data/visualizations'):
    """Plot cumulative returns for each ticker"""
    # Create separate plots for sector ETFs and bond ETFs
    sector_dir = os.path.join(output_dir, 'sector_cumulative')
    bond_dir = os.path.join(output_dir, 'bond_cumulative')
    os.makedirs(sector_dir, exist_ok=True)
    os.makedirs(bond_dir, exist_ok=True)
    
    for etf, signals in signals_dict.items():
        plt.figure(figsize=(15, 8))
        
        # Convert portfolio value to cumulative returns
        cum_returns = signals['Portfolio_Value'] / signals['Portfolio_Value'].iloc[0] - 1
        
        # Plot cumulative returns
        plt.plot(signals.index, cum_returns, linewidth=2)
        
        # Add buy/sell markers
        buy_signals = signals[signals['Signal'] == 1].index
        sell_signals = signals[signals['Signal'] == 0].index
        
        plt.scatter(buy_signals, cum_returns[buy_signals], 
                   color='green', marker='^', s=100, label='Buy Signal')
        plt.scatter(sell_signals, cum_returns[sell_signals], 
                   color='red', marker='v', s=100, label='Sell Signal')
        
        plt.title(f'Cumulative Returns for {etf}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to appropriate directory
        save_dir = sector_dir if etf.startswith('XL') else bond_dir
        plt.savefig(os.path.join(save_dir, f'{etf}_cumulative_returns.png'))
        plt.close()
        
    # Create summary plots for sectors and bonds
    plt.figure(figsize=(15, 8))
    for etf, signals in signals_dict.items():
        if etf.startswith('XL'):  # Only sector ETFs
            cum_returns = signals['Portfolio_Value'] / signals['Portfolio_Value'].iloc[0] - 1
            plt.plot(signals.index, cum_returns, label=etf, alpha=0.7)
    
    plt.title('Cumulative Returns - Sector ETFs')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sector_cumulative_summary.png'))
    plt.close()
    
    plt.figure(figsize=(15, 8))
    for etf, signals in signals_dict.items():
        if not etf.startswith('XL'):  # Only bond ETFs
            cum_returns = signals['Portfolio_Value'] / signals['Portfolio_Value'].iloc[0] - 1
            plt.plot(signals.index, cum_returns, label=etf, alpha=0.7)
    
    plt.title('Cumulative Returns - Bond ETFs')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bond_cumulative_summary.png'))
    plt.close() 