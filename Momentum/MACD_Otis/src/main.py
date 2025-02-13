from strategies import macd, vpvma
from utils import data_loader, portfolio
from analysis import performance, visualization
import pandas as pd
import os

# Define ETF lists
SECTOR_ETFS = ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE']
BOND_ETFS = ['AGG', 'BND', 'TLT', 'IEF', 'LQD', 'HYG', 'EMB', 'MUB', 'BNDX', 'VCIT']

def main():
    # Load VIX data first
    vix_df = data_loader.load_vix_data()
    
    all_etfs = SECTOR_ETFS + BOND_ETFS
    results = {}
    
    # Create a list to store strategy statistics and performance metrics
    strategy_stats = []
    performance_results = []
    best_signals_dict = {}  # Store best signals for each ETF
    
    for etf in all_etfs:
        # Process each ETF
        df = data_loader.load_etf_data(etf)
        if df is not None:
            # Run strategies
            macd_signals = macd.generate_macd_signals(df.copy())
            vpvma_signals = vpvma.generate_vpvma_signals(df.copy(), vix_df)
            
            # Save signals
            data_loader.save_signals(macd_signals, etf, 'macd')
            data_loader.save_signals(vpvma_signals, etf, 'vpvma')
            
            # Calculate strategy performance using final portfolio value
            initial_capital = 1000000
            macd_performance = macd_signals['Portfolio_Value'].iloc[-1] / initial_capital - 1
            vpvma_performance = vpvma_signals['Portfolio_Value'].iloc[-1] / initial_capital - 1
            
            # Choose the better strategy
            best_strategy = 'macd' if macd_performance > vpvma_performance else 'vpvma'
            best_signals = macd_signals if best_strategy == 'macd' else vpvma_signals
            
            # Calculate performance metrics for the best strategy
            best_signals.set_index('Date', inplace=True)
            metrics = performance.calculate_performance_metrics(best_signals)
            metrics['ETF'] = etf
            metrics['Best Strategy'] = best_strategy
            performance_results.append(metrics)
            
            # Store best signals for plotting
            best_signals_dict[etf] = best_signals
            
            # Add to strategy statistics
            strategy_stats.append({
                'ETF': etf,
                'Strategy': best_strategy
            })
    
    # Create and save strategy statistics DataFrame
    stats_df = pd.DataFrame(strategy_stats)
    os.makedirs('data/results', exist_ok=True)
    stats_df.to_csv('data/results/strategy_statistics.csv', index=False)
    
    # Create and save performance results
    perf_df = pd.DataFrame(performance_results)
    perf_df.set_index('ETF', inplace=True)
    
    # Format the results for better readability
    formatted_perf = perf_df.copy()
    formatted_perf['Total Return'] = formatted_perf['Total Return'].map('{:.2%}'.format)
    formatted_perf['Annual Return'] = formatted_perf['Annual Return'].map('{:.2%}'.format)
    formatted_perf['Annualized Volatility'] = formatted_perf['Annualized Volatility'].map('{:.2%}'.format)
    formatted_perf['Sharpe Ratio'] = formatted_perf['Sharpe Ratio'].map('{:.2f}'.format)
    formatted_perf['Max Drawdown'] = formatted_perf['Max Drawdown'].map('{:.2%}'.format)
    formatted_perf['Win Rate'] = formatted_perf['Win Rate'].map('{:.2%}'.format)
    
    # Save performance results
    formatted_perf.to_csv('data/results/strategy_performance.csv')
    print("\nStrategy Performance Summary:")
    print(formatted_perf)
    
    # Create visualizations directory
    os.makedirs('data/visualizations', exist_ok=True)
    
    # Generate visualizations
    visualization.plot_returns_comparison(formatted_perf)
    visualization.plot_risk_return_scatter(formatted_perf)
    visualization.plot_win_rates(formatted_perf)
    visualization.plot_drawdowns(formatted_perf)
    visualization.plot_cumulative_returns(best_signals_dict)
    
    print("\nVisualizations have been saved to data/visualizations/")
    print("Individual cumulative return plots can be found in:")
    print("- data/visualizations/sector_cumulative/")
    print("- data/visualizations/bond_cumulative/")
    
    # Now construct portfolios
    sector_portfolio, sector_etf_values = portfolio.construct_portfolio_performance(SECTOR_ETFS)
    bond_portfolio, bond_etf_values = portfolio.construct_portfolio_performance(BOND_ETFS)

if __name__ == "__main__":
    main() 