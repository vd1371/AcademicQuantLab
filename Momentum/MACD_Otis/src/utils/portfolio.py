import pandas as pd
import os

def construct_portfolio_performance(etf_list, data_dir='data/results'):
    """Construct portfolio performance from weekly signals"""
    # Load strategy statistics
    stats = pd.read_csv(os.path.join(data_dir, 'strategy_statistics.csv'))
    stats.set_index('ETF', inplace=True)
    
    # Initialize portfolio metrics
    portfolio_values = []
    
    # Process each ETF
    for etf in etf_list:
        if etf in stats.index:
            best_strategy = stats.loc[etf, 'Strategy']
            file_name = f'weekly_{best_strategy.lower()}_signals.csv'
            
            # Load weekly data
            file_path = os.path.join('data', 'processed', etf, file_name)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df['ETF'] = etf
                portfolio_values.append(df[['Date', 'Portfolio_Value', 'ETF']])
    
    # Combine and process portfolio values
    if portfolio_values:
        all_values = pd.concat(portfolio_values, ignore_index=True)
        portfolio_value = all_values.groupby('Date')['Portfolio_Value'].sum()
        
        # Calculate individual ETF contributions
        etf_values = {
            etf: all_values[all_values['ETF'] == etf].set_index('Date')['Portfolio_Value']
            for etf in etf_list if etf in all_values['ETF'].unique()
        }
        
        return portfolio_value, etf_values
    
    return None, None 