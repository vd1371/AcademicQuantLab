import pandas as pd
import os

def construct_portfolio_performance(etf_list, data_dir='data/processed'):
    """Construct portfolio performance from weekly signals"""
    # Load strategy statistics
    stats_file = os.path.join('data/results', 'strategy_statistics.csv')
    if not os.path.exists(stats_file):
        print(f"Strategy statistics file not found at {stats_file}")
        return None, None
        
    stats = pd.read_csv(stats_file, index_col='ETF')
    
    # Initialize portfolio metrics
    portfolio_values = []
    
    # Process each ETF
    for etf in etf_list:
        if etf in stats.index:
            strategy_type = stats.loc[etf, 'Strategy'].lower().split('_')[0]  # Get just 'macd' or 'vpvma'
            file_name = f'weekly_{strategy_type}_signals.csv'
            
            # Load weekly data
            file_path = os.path.join(data_dir, etf, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    df['ETF'] = etf
                    portfolio_values.append(df[['Date', 'Portfolio_Value', 'ETF']])
                except Exception as e:
                    print(f"Error loading data for {etf}: {str(e)}")
                    continue
            else:
                print(f"Signal file not found: {file_path}")
    
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
    
    print("No portfolio values found")
    return None, None 