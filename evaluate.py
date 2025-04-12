import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def evaluate_against_baseline(env, agent, data, initial_balance=10000, episodes=1):
    """
    Evaluate the agent against a buy-and-hold baseline.
    
    Args:
        env: Stock trading environment
        agent: Trained RL agent
        data: Stock price data
        initial_balance: Starting balance
        episodes: Number of evaluation episodes
        
    Returns:
        Metrics comparing agent performance to baseline
    """
    # Calculate buy and hold performance
    buy_and_hold_value = calculate_buy_and_hold(data, initial_balance)
    
    # Run agent evaluation
    agent_metrics = run_agent_evaluation(env, agent, episodes)
    
    # Compare performance
    comparison = compare_performance(agent_metrics, buy_and_hold_value, data, initial_balance)
    
    # Visualize comparison
    visualize_comparison(agent_metrics, buy_and_hold_value, data)
    
    return comparison

def calculate_buy_and_hold(data, initial_balance):
    """
    Calculate the performance of a buy-and-hold strategy.
    
    Args:
        data: Stock price data
        initial_balance: Starting balance
        
    Returns:
        Array of portfolio values over time
    """
    # Buy as many shares as possible at the beginning
    shares = initial_balance // data[0]
    remaining_cash = initial_balance - shares * data[0]
    
    # Calculate portfolio value at each time step
    portfolio_values = [initial_balance]  # Initial value
    for i in range(1, len(data)):
        value = shares * data[i] + remaining_cash
        portfolio_values.append(value)
    
    return portfolio_values

def run_agent_evaluation(env, agent, episodes=1):
    """
    Run the agent through multiple evaluation episodes.
    
    Args:
        env: Stock trading environment
        agent: Trained RL agent
        episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of agent performance metrics
    """
    # Set agent to evaluation mode (no exploration)
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = 0
    
    cumulative_returns = []
    final_values = []
    transaction_counts = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Collect metrics
        final_values.append(env.total_value)
        transaction_counts.append(len(env.transaction_history))
        
        # Get performance metrics
        metrics = env.visualize_performance()
        cumulative_returns.append(metrics['cumulative_return'])
    
    # Restore exploration rate
    agent.exploration_rate = original_exploration_rate
    
    return {
        'final_values': final_values,
        'mean_final_value': np.mean(final_values),
        'cumulative_returns': cumulative_returns,
        'mean_return': np.mean(cumulative_returns),
        'transaction_counts': transaction_counts,
        'mean_transactions': np.mean(transaction_counts)
    }

def compare_performance(agent_metrics, buy_and_hold_values, data, initial_balance):
    """
    Compare agent performance to buy-and-hold strategy.
    
    Args:
        agent_metrics: Dictionary of agent performance metrics
        buy_and_hold_values: Array of buy-and-hold portfolio values
        data: Stock price data
        initial_balance: Starting balance
        
    Returns:
        Dictionary of comparison metrics
    """
    # Calculate buy-and-hold return
    buy_hold_return = (buy_and_hold_values[-1] - initial_balance) / initial_balance
    
    # Get agent's mean return
    agent_return = agent_metrics['mean_return']
    
    # Calculate return difference
    return_difference = agent_return - buy_hold_return
    
    # Calculate stock market return
    market_return = (data[-1] - data[0]) / data[0]
    
    # Calculate risk-adjusted returns
    # For buy-and-hold
    buy_hold_returns = np.diff(buy_and_hold_values) / buy_and_hold_values[:-1]
    buy_hold_sharpe = np.mean(buy_hold_returns) / np.std(buy_hold_returns) * np.sqrt(252) if np.std(buy_hold_returns) > 0 else 0
    
    # For agent (use the mean across episodes)
    agent_sharpe = np.mean(agent_metrics.get('sharpe_ratios', [0]))
    
    results = {
        'agent_return': agent_return,
        'buy_hold_return': buy_hold_return,
        'return_difference': return_difference,
        'market_return': market_return,
        'agent_final_value': agent_metrics['mean_final_value'],
        'buy_hold_final_value': buy_and_hold_values[-1],
        'agent_sharpe': agent_sharpe,
        'buy_hold_sharpe': buy_hold_sharpe,
        'agent_transactions': agent_metrics['mean_transactions']
    }
    
    # Print summary
    print("\nPerformance Comparison:")
    print(f"Agent Return: {agent_return*100:.2f}%")
    print(f"Buy-and-Hold Return: {buy_hold_return*100:.2f}%")
    print(f"Return Difference: {return_difference*100:.2f}%")
    print(f"Market Return: {market_return*100:.2f}%")
    print(f"Agent Final Value: ${agent_metrics['mean_final_value']:.2f}")
    print(f"Buy-and-Hold Final Value: ${buy_and_hold_values[-1]:.2f}")
    
    return results

def visualize_comparison(agent_metrics, buy_and_hold_values, data):
    """
    Visualize comparison between agent and buy-and-hold strategy.
    
    Args:
        agent_metrics: Dictionary of agent performance metrics
        buy_and_hold_values: Array of buy-and-hold portfolio values
        data: Stock price data
    """
    # Normalize data for comparison
    buy_hold_norm = np.array(buy_and_hold_values) / buy_and_hold_values[0]
    stock_price_norm = data / data[0]
    
    plt.figure(figsize=(12, 6))
    
    # Plot normalized stock price
    plt.plot(range(len(data)), stock_price_norm, 'k-', label='Stock Price', alpha=0.5)
    
    # Plot buy-and-hold strategy
    plt.plot(range(len(buy_hold_norm)), buy_hold_norm, 'b-', label='Buy-and-Hold')
    
    # Plot agent performance (final point)
    agent_final = agent_metrics['mean_final_value'] / buy_and_hold_values[0]
    plt.scatter(len(data)-1, agent_final, color='red', s=100, marker='*', label='RL Agent')
    
    plt.title('Performance Comparison (Normalized)')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('performance_comparison.png')
    plt.show()

def analyze_trading_patterns(env):
    """
    Analyze the trading patterns of the agent.
    
    Args:
        env: Environment after agent interaction
        
    Returns:
        Dictionary of pattern analysis
    """
    if not env.transaction_history:
        return {"error": "No transactions found"}
    
    transactions = env.transaction_history
    price_data = env.asset_price_history
    
    # Convert transactions to DataFrame for analysis
    trans_data = []
    for t in transactions:
        trans_data.append({
            'step': t['step'],
            'price': t['price'],
            'action': 'Buy' if t['action'] == 0 else 'Sell',
            'shares': t.get('shares', 0)
        })
    
    df_trans = pd.DataFrame(trans_data)
    
    # Calculate metrics
    buy_prices = [t['price'] for t in transactions if t['action'] == 0]
    sell_prices = [t['price'] for t in transactions if t['action'] == 1]
    
    avg_buy_price = np.mean(buy_prices) if buy_prices else 0
    avg_sell_price = np.mean(sell_prices) if sell_prices else 0
    
    # Calculate average holding period
    holding_periods = []
    buy_steps = [t['step'] for t in transactions if t['action'] == 0]
    sell_steps = [t['step'] for t in transactions if t['action'] == 1]
    
    if len(buy_steps) > 0 and len(sell_steps) > 0:
        # Match buys and sells (simple approach)
        for sell_step in sell_steps:
            # Find the most recent buy before this sell
            previous_buys = [step for step in buy_steps if step < sell_step]
            if previous_buys:
                most_recent_buy = max(previous_buys)
                holding_periods.append(sell_step - most_recent_buy)
    
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0
    
    # Calculate profit per trade
    profit_per_trade = avg_sell_price - avg_buy_price if avg_buy_price > 0 else 0
    
    # Identify market timing patterns
    price_changes = np.diff(price_data)
    up_days = np.sum(price_changes > 0)
    down_days = np.sum(price_changes < 0)
    
    # Calculate accuracy of buying on dips and selling on rises
    buys_on_dips = 0
    sells_on_rises = 0
    
    for i, t in enumerate(transactions):
        if t['action'] == 0:  # Buy
            if i > 0 and t['price'] < price_data[t['step']-1]:
                buys_on_dips += 1
        elif t['action'] == 1:  # Sell
            if i > 0 and t['price'] > price_data[t['step']-1]:
                sells_on_rises += 1
    
    buy_dip_accuracy = buys_on_dips / len(buy_prices) if buy_prices else 0
    sell_rise_accuracy = sells_on_rises / len(sell_prices) if sell_prices else 0
    
    # Visual analysis
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(price_data)), price_data, 'k-', alpha=0.6)
    
    buy_idxs = [t['step'] for t in transactions if t['action'] == 0]
    sell_idxs = [t['step'] for t in transactions if t['action'] == 1]
    
    buy_prices = [price_data[idx] for idx in buy_idxs]
    sell_prices = [price_data[idx] for idx in sell_idxs]
    
    plt.scatter(buy_idxs, buy_prices, color='green', marker='^', s=100, label='Buy')
    plt.scatter(sell_idxs, sell_prices, color='red', marker='v', s=100, label='Sell')
    
    plt.title('Agent Trading Pattern Analysis')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('trading_pattern_analysis.png')
    plt.show()
    
    # Print analysis
    print("\nTrading Pattern Analysis:")
    print(f"Total Trades: {len(transactions)}")
    print(f"Buy Trades: {len(buy_prices)}")
    print(f"Sell Trades: {len(sell_prices)}")
    print(f"Average Buy Price: ${avg_buy_price:.2f}")
    print(f"Average Sell Price: ${avg_sell_price:.2f}")
    print(f"Average Profit per Trade: ${profit_per_trade:.2f}")
    print(f"Average Holding Period: {avg_holding_period:.2f} days")
    print(f"Buy on Dip Accuracy: {buy_dip_accuracy*100:.2f}%")
    print(f"Sell on Rise Accuracy: {sell_rise_accuracy*100:.2f}%")
    
    return {
        'total_trades': len(transactions),
        'buy_trades': len(buy_prices),
        'sell_trades': len(sell_prices),
        'avg_buy_price': avg_buy_price,
        'avg_sell_price': avg_sell_price,
        'avg_profit_per_trade': profit_per_trade,
        'avg_holding_period': avg_holding_period,
        'buy_dip_accuracy': buy_dip_accuracy,
        'sell_rise_accuracy': sell_rise_accuracy
    }

def visualize_q_values(agent, env):
    """
    Visualize the agent's learned Q-values.
    
    Args:
        agent: Trained RL agent
        env: Trading environment
    """
    if not agent.q_table:
        print("Q-table is empty, no visualization possible.")
        return
    
    # Extract key state components for analysis
    price_rel_sma_values = defaultdict(list)
    momentum_values = defaultdict(list)
    
    for state, q_values in agent.q_table.items():
        price_rel_sma, _, _, momentum_disc, _ = state
        
        # Record maximum Q-value for each action
        action_indices = np.argsort(q_values)[::-1]  # Sort in descending order
        best_action = action_indices[0]
        
        price_rel_sma_values[best_action].append(price_rel_sma)
        momentum_values[best_action].append(momentum_disc)
    
    # Visualize price relative to SMA vs best action
    plt.figure(figsize=(12, 6))
    
    # For each action, plot histogram of price_rel_sma values
    labels = ['Buy', 'Sell', 'Hold']
    colors = ['green', 'red', 'blue']
    
    for action in range(3):
        if price_rel_sma_values[action]:
            plt.hist(price_rel_sma_values[action], bins=10, alpha=0.5, 
                     color=colors[action], label=labels[action])
    
    plt.title('Best Actions Based on Price Relative to SMA')
    plt.xlabel('Price Relative to SMA (Discretized)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('q_values_price_sma.png')
    plt.show()
    
    # Visualize momentum vs best action
    plt.figure(figsize=(12, 6))
    
    for action in range(3):
        if momentum_values[action]:
            plt.hist(momentum_values[action], bins=5, alpha=0.5,
                     color=colors[action], label=labels[action])
    
    plt.title('Best Actions Based on Price Momentum')
    plt.xlabel('Momentum (Discretized)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('q_values_momentum.png')
    plt.show()
    
    # Analyze which factors most influence decisions
    # Count how often each factor is the dominant reason for an action
    print("\nQ-Value Analysis:")
    
    # Calculate average Q-value for each action
    action_q_values = [[] for _ in range(3)]
    for q_values in agent.q_table.values():
        for i, q in enumerate(q_values):
            action_q_values[i].append(q)
    
    for i, action in enumerate(labels):
        avg_q = np.mean(action_q_values[i]) if action_q_values[i] else 0
        print(f"Average Q-value for {action}: {avg_q:.4f}")
    
    # Analyze state-action patterns
    print("\nState-Action Patterns:")
    
    # Price relative to SMA
    for action in range(3):
        values = price_rel_sma_values[action]
        if values:
            mean_val = np.mean(values)
            print(f"{labels[action]} action tends to occur when price relative to SMA is: {mean_val:.2f}")
    
    # Momentum
    for action in range(3):
        values = momentum_values[action]
        if values:
            mean_val = np.mean(values)
            momentum_desc = "strongly negative" if mean_val < -1.5 else \
                           "negative" if mean_val < -0.5 else \
                           "neutral" if mean_val < 0.5 else \
                           "positive" if mean_val < 1.5 else "strongly positive"
            print(f"{labels[action]} action tends to occur when momentum is: {momentum_desc} ({mean_val:.2f})")