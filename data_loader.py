import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_stock_data_from_csv(file_path, price_col='Close'):
    """
    Load stock data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        price_col: Column name for price data
        
    Returns:
        Numpy array of prices
    """
    try:
        df = pd.read_csv(file_path)
        if price_col in df.columns:
            return df[price_col].values
        else:
            print(f"Column '{price_col}' not found. Available columns: {df.columns.tolist()}")
            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_synthetic_data(days=1000, volatility=0.01, trend=0.0001, start_price=100.0):
    """
    Generate synthetic stock price data.
    
    Args:
        days: Number of trading days to generate
        volatility: Daily price volatility
        trend: Daily price trend
        start_price: Initial price
        
    Returns:
        Numpy array of synthetic prices
    """
    # Generate random daily returns
    daily_returns = np.random.normal(trend, volatility, days)
    
    # Add some momentum (autocorrelation)
    for i in range(1, len(daily_returns)):
        daily_returns[i] += daily_returns[i-1] * 0.1
    
    # Add occasional market shocks
    shock_indices = np.random.choice(days, size=int(days*0.05), replace=False)
    for idx in shock_indices:
        daily_returns[idx] += np.random.choice([-1, 1]) * volatility * 5
    
    # Calculate price series
    price_series = start_price * (1 + np.cumsum(daily_returns))
    
    # Ensure prices are positive
    return np.maximum(price_series, 0.01)

def add_market_regimes(prices, regime_length=100, bear_modifier=0.8, bull_modifier=1.2):
    """
    Add bull and bear market regimes to the price data.
    
    Args:
        prices: Original price array
        regime_length: Average length of each market regime
        bear_modifier: Price multiplier for bear markets
        bull_modifier: Price multiplier for bull markets
        
    Returns:
        Modified price array with market regimes
    """
    modified_prices = prices.copy()
    days = len(prices)
    
    # Generate random regimes
    num_regimes = max(days // regime_length, 1)
    regime_types = np.random.choice(['bull', 'bear', 'sideways'], size=num_regimes)
    
    start_idx = 0
    for i, regime in enumerate(regime_types):
        end_idx = min(start_idx + regime_length + np.random.randint(-20, 20), days)
        
        if regime == 'bull':
            # Upward trend
            trend = np.linspace(1, bull_modifier, end_idx - start_idx)
            modified_prices[start_idx:end_idx] *= trend
        elif regime == 'bear':
            # Downward trend
            trend = np.linspace(1, bear_modifier, end_idx - start_idx)
            modified_prices[start_idx:end_idx] *= trend
        # Sideways market keeps the original trend
        
        start_idx = end_idx
    
    return modified_prices

def visualize_data(prices, window_size=20):
    """
    Visualize stock price data with moving average.
    
    Args:
        prices: Array of price data
        window_size: Moving average window size
    """
    days = np.arange(len(prices))
    
    # Calculate moving average
    ma = np.convolve(prices, np.ones(window_size)/window_size, mode='valid')
    ma_days = np.arange(window_size-1, len(prices))
    
    plt.figure(figsize=(12, 6))
    plt.plot(days, prices, 'b-', label='Stock Price')
    plt.plot(ma_days, ma, 'r-', label=f'{window_size}-day Moving Average')
    
    plt.title('Stock Price Data')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Calculate daily returns
    returns = np.diff(prices) / prices[:-1]
    
    plt.figure(figsize=(12, 6))
    plt.hist(returns, bins=50, alpha=0.75)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('stock_data_visualization.png')
    plt.show()
    
    # Print statistics
    print(f"Data Statistics:")
    print(f"Number of days: {len(prices)}")
    print(f"Starting price: ${prices[0]:.2f}")
    print(f"Ending price: ${prices[-1]:.2f}")
    print(f"Min price: ${np.min(prices):.2f}")
    print(f"Max price: ${np.max(prices):.2f}")
    print(f"Price change: {(prices[-1]/prices[0] - 1)*100:.2f}%")
    print(f"Daily volatility: {np.std(returns)*100:.2f}%")
    
    return {
        'mean_return': np.mean(returns),
        'volatility': np.std(returns),
        'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
        'max_drawdown': calculate_max_drawdown(prices)
    }

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Array of prices
        
    Returns:
        Maximum drawdown value
    """
    peak = prices[0]
    max_drawdown = 0
    
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into training, validation, and testing sets.
    
    Args:
        data: Price data array
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        train_data, val_data, test_data
    """
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    return train_data, val_data, test_data