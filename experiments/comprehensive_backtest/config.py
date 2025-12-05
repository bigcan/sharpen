import os

# Experiment Settings
EXPERIMENT_NAME = "synapse_comprehensive_backtest"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data Universe (DJIA 30)
TICKERS = [
    "AAPL", "MSFT", "JPM", "V", "RTX", "PG", "GS", "NKE", "DIS", "AXP",
    "HD", "INTC", "WMT", "IBM", "MRK", "UNH", "KO", "CAT", "TRV", "JNJ",
    "CVX", "MCD", "VZ", "CSCO", "XOM", "BA", "MMM", "PFE", "WBA", "DD" 
    # Note: Dow components change over time, this is a representative list for the period.
    # For a stricter backtest, we might want to handle survivorship bias, but for now this is sufficient.
]

# Date Ranges
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Rolling Window Settings
TRAIN_WINDOW_SIZE = 365 * 2  # 2 Years
TEST_WINDOW_SIZE = 365 // 2  # 6 Months
ROLLING_STEP_SIZE = 365 // 2 # 6 Months

# Environment Settings
INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST_PCT = 0.001 # 0.1%

# Model Hyperparameters
TRAIN_EPOCHS = 30
LEARNING_RATE = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64

# Technical Indicators
INDICATORS = ['macd', 'rsi_14', 'cci_14', 'dx_14']
