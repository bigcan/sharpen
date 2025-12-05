import data_loader
import backtest
import config
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    print(f"Running Experiment: {config.EXPERIMENT_NAME}")
    
    # 1. Load Data
    try:
        df = data_loader.download_data()
        df = data_loader.preprocess_data(df)
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        return

    # 2. Generate Windows
    windows = data_loader.get_rolling_windows(df)
    
    # 3. Run Backtest
    results_df = backtest.run_backtest(df, windows)
    
    # 4. Analyze Results
    if not results_df.empty:
        print("\n=== Summary Metrics ===")
        print(f"Average Single Sharpe:  {results_df['single_sharpe'].mean():.3f}")
        print(f"Average Synapse Sharpe: {results_df['synapse_sharpe'].mean():.3f}")
        print(f"Net Improvement:        {results_df['synapse_sharpe'].mean() - results_df['single_sharpe'].mean():+.3f}")
        
        # Plot Cumulative Returns (concatenated from windows)
        # Note: This is a simplified plot of window-by-window returns, not a true continuous equity curve
        # Constructing a true equity curve requires stitching the daily portfolio values.
        # For this report, we'll plot the Sharpe comparison bar chart.
        
        plt.figure(figsize=(12, 6))
        x = range(len(results_df))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], results_df['single_sharpe'], width, label='Single Agent', color='gray')
        plt.bar([i + width/2 for i in x], results_df['synapse_sharpe'], width, label='Synapse', color='green')
        
        plt.xlabel('Window')
        plt.ylabel('Sharpe Ratio')
        plt.title('Synapse vs Single Agent - Rolling Backtest (DJIA 2015-2024)')
        plt.xticks(x, results_df['test_start'].dt.strftime('%Y-%m'), rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(config.RESULTS_DIR, "sharpe_comparison.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
