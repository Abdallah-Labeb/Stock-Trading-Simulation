# Reinforcement Learning for Stock Trading

A comprehensive reinforcement learning framework for stock trading using Q-Learning.

## Project Overview

This project implements a reinforcement learning agent for stock trading. The agent uses Q-Learning to learn optimal trading strategies from historical stock price data. The system includes:

- A flexible stock trading environment
- A Q-Learning agent with state discretization
- Data handling utilities for real and synthetic data
- Comprehensive performance evaluation and visualization
- Training and evaluation pipelines

## Files and Components

- `environment.py` - The stock trading environment
- `agent.py` - Q-Learning agent implementation
- `data_loader.py` - Utilities for loading and processing data
- `train.py` - Training functions
- `evaluate.py` - Evaluation and performance analysis
- `visualization.py` - Advanced visualizations and dashboards
- `main.py` - Main application entry point

## Installation

```bash
# Clone the repository
git clone https://github.com/username/rl-stock-trading.git
cd rl-stock-trading

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
tqdm
```

## Usage

### Training a model

```bash
python main.py --mode train --data synthetic --episodes 500 --visualize --save-model model.pkl
```

### Using real data

```bash
python main.py --mode train --data path/to/stock_data.csv --price-col Close
```

### Evaluating a trained model

```bash
python main.py --mode evaluate --data synthetic --load-model model.pkl --visualize
```

### Analyzing agent behavior

```bash
python main.py --mode analyze --data synthetic --load-model model.pkl
```

## Command Line Arguments

- `--mode`: Operating mode (`train`, `evaluate`, or `analyze`)
- `--data`: Data source (path to CSV file or "synthetic")
- `--price-col`: Column name for price data in CSV
- `--initial-balance`: Initial balance for trading
- `--fee`: Transaction fee percentage
- `--episodes`: Number of training episodes
- `--lr`: Learning rate
- `--gamma`: Discount factor
- `--epsilon`: Initial exploration rate
- `--epsilon-decay`: Exploration rate decay
- `--min-epsilon`: Minimum exploration rate
- `--save-model`: Path to save the trained model
- `--load-model`: Path to load a trained model
- `--render`: Render the environment during training
- `--visualize`: Visualize data and results

## Environment

The environment simulates a stock trading scenario with the following features:

- Variable initial balance
- Transaction fees
- Multiple actions: Buy, Sell, Hold
- Realistic reward calculation based on portfolio value
- Technical indicators for informed decision making

## Agent

The Q-Learning agent includes:

- State discretization for handling continuous state space
- Epsilon-greedy exploration strategy with decay
- Learning rate and discount factor parameters
- Action-value function approximation

## Performance Visualization

The project includes advanced visualization tools:

- Portfolio value and trading decisions visualization
- Comparison against buy-and-hold baseline
- Performance metrics (Sharpe ratio, max drawdown, etc.)
- Trading pattern analysis
- Q-value visualization
- Comprehensive performance dashboard

## Example

```python
from data_loader import generate_synthetic_data
from environment import StockTradingEnv
from agent import QLearningAgent
from train import train_agent
from evaluate import evaluate_against_baseline

# Generate data
data = generate_synthetic_data(days=1000)

# Create environment and agent
env = StockTradingEnv(data, initial_balance=10000)
agent = QLearningAgent(action_space=3)

# Train the agent
train_agent(env, agent, episodes=500)

# Evaluate performance
eval_env = StockTradingEnv(data, initial_balance=10000)
performance = evaluate_against_baseline(eval_env, agent, data)
```

## Future Work

- Implement Deep Q-Learning for better state representation
- Add more technical indicators
- Support for multi-asset trading
- Portfolio optimization
- Risk management strategies
- Real-time trading integration

## License

MIT License

## Credits

Created by [Your Name]