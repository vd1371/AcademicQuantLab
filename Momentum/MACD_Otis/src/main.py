from strategies import macd, vpvma
from utils import data_loader, portfolio
from analysis import performance
import pandas as pd
import os

# Define ETF lists
SECTOR_ETFS = ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE']
BOND_ETFS = ['AGG', 'BND', 'TLT', 'IEF', 'LQD', 'HYG', 'EMB', 'MUB', 'BNDX', 'VCIT']

def main():
    all_etfs = SECTOR_ETFS + BOND_ETFS
    results = {}
    
    for etf in all_etfs:
        # Process each ETF
        df = data_loader.load_etf_data(etf)
        if df is not None:
            # Run strategies
            macd_signals = macd.generate_macd_signals(df.copy())
            vpvma_signals = vpvma.generate_vpvma_signals(df.copy())
            
            # Save signals
            data_loader.save_signals(macd_signals, etf, 'macd')
            data_loader.save_signals(vpvma_signals, etf, 'vpvma')
    
    # Construct portfolios
    sector_portfolio, sector_etf_values = portfolio.construct_portfolio_performance(SECTOR_ETFS)
    bond_portfolio, bond_etf_values = portfolio.construct_portfolio_performance(BOND_ETFS)
    
    # Generate and save results
    # Add result saving logic here

if __name__ == "__main__":
    main() 