import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StockTradingEnv:
    """
    A stock trading environment for reinforcement learning.
    """
    
    def __init__(self, data, initial_balance=10000, transaction_fee_percent=0.001):
        """
        Initialize the environment.
        
        Args:
            data: Historical stock price data
            initial_balance: Starting cash balance
            transaction_fee_percent: Percentage fee for each transaction
        """
        self.stock_data = data
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reset()
    
    def reset(self):
        """Reset the environment to its initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.balance
        self.transaction_history = []
        self.returns_history = []
        self.asset_price_history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current state observation.
        
        Returns:
            Dictionary containing the current state
        """
        price = self.stock_data[self.current_step]
        
        # Include price history for technical indicators
        history_window = 10
        if self.current_step >= history_window:
            price_history = self.stock_data[self.current_step-history_window:self.current_step]
        else:
            price_history = self.stock_data[:self.current_step]
            # Pad with first price if needed
            if len(price_history) < history_window:
                padding = [self.stock_data[0]] * (history_window - len(price_history))
                price_history = np.concatenate([padding, price_history])
        
        # Calculate simple moving average (SMA)
        if len(price_history) > 0:
            sma = np.mean(price_history)
        else:
            sma = price
        
        # Calculate price momentum (rate of change)
        if len(price_history) > 5:
            momentum = (price - price_history[-5]) / price_history[-5]
        else:
            momentum = 0
            
        # Calculate volatility (standard deviation of returns)
        if len(price_history) > 1:
            returns = np.diff(price_history) / price_history[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0
            
        obs = {
            'price': price,
            'balance': self.balance,
            'shares': self.shares_held,
            'total_value': self.total_value,
            'price_history': price_history,
            'sma': sma,
            'momentum': momentum,
            'volatility': volatility
        }
        return obs
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Action to take (0: Buy, 1: Sell, 2: Hold)
            
        Returns:
            next_state, reward, done, info
        """
        # Execute the action
        self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1
        
        # Get the new observation
        next_observation = self._get_observation()
        
        # Calculate reward (change in portfolio value)
        current_price = next_observation['price']
        new_total_value = self.balance + self.shares_held * current_price
        reward = new_total_value - self.total_value
        
        # Update total value
        old_value = self.total_value
        self.total_value = new_total_value
        
        # Store return percentage
        self.returns_history.append((self.total_value - old_value) / old_value if old_value > 0 else 0)
        
        # Store asset price
        self.asset_price_history.append(current_price)
        
        # Information dictionary
        info = {
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_value': self.total_value
        }
        
        return next_observation, reward, done, info
    
    def _take_action(self, action):
        """
        Execute the specified action in the environment.
        
        Args:
            action: 0 (Buy), 1 (Sell), or 2 (Hold)
        """
        current_price = self.stock_data[self.current_step]
        transaction = {'step': self.current_step, 'price': current_price, 'action': action}
        
        if action == 0:  # Buy
            max_affordable_shares = self.balance // (current_price * (1 + self.transaction_fee_percent))
            if max_affordable_shares > 0:
                shares_to_buy = max_affordable_shares
                transaction_cost = shares_to_buy * current_price * (1 + self.transaction_fee_percent)
                self.balance -= transaction_cost
                self.shares_held += shares_to_buy
                transaction.update({
                    'shares': shares_to_buy,
                    'cost': transaction_cost,
                    'new_balance': self.balance,
                    'new_shares': self.shares_held
                })
                self.transaction_history.append(transaction)
                
        elif action == 1:  # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held
                transaction_value = shares_to_sell * current_price * (1 - self.transaction_fee_percent)
                self.balance += transaction_value
                self.shares_held = 0
                transaction.update({
                    'shares': shares_to_sell,
                    'value': transaction_value,
                    'new_balance': self.balance,
                    'new_shares': self.shares_held
                })
                self.transaction_history.append(transaction)
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        print(f"Step: {self.current_step}")
        print(f"Price: ${self.stock_data[self.current_step]:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held}")
        print(f"Total value: ${self.total_value:.2f}")
        print("-" * 40)
    
    def visualize_performance(self, baseline_data=None):
        """
        Visualize the performance of the agent.
        
        Args:
            baseline_data: Buy and hold performance data for comparison
        """
        steps = list(range(len(self.returns_history)))
        
        # Plot portfolio value vs stock price
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
        
        # Portfolio value over time
        portfolio_values = [self.initial_balance]
        for i, ret in enumerate(self.returns_history):
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        ax1.plot(steps, portfolio_values[:-1], 'b-', label='Portfolio Value')
        
        # Normalize stock price for comparison
        normalized_price = [self.asset_price_history[0]]
        for price in self.asset_price_history:
            normalized_price.append(price / self.asset_price_history[0] * self.initial_balance)
        
        ax1.plot(steps, normalized_price[:-1], 'r-', label='Stock Price (Normalized)')
        
        # Buy and Hold strategy if provided
        if baseline_data is not None:
            ax1.plot(steps, baseline_data, 'g-', label='Buy & Hold')
        
        ax1.set_title('Portfolio Value vs Stock Price')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Transaction history
        buy_x = []
        buy_y = []
        sell_x = []
        sell_y = []
        
        for transaction in self.transaction_history:
            if transaction['action'] == 0:  # Buy
                buy_x.append(transaction['step'])
                buy_y.append(self.asset_price_history[transaction['step']])
            elif transaction['action'] == 1:  # Sell
                sell_x.append(transaction['step'])
                sell_y.append(self.asset_price_history[transaction['step']])
        
        ax2.plot(steps, self.asset_price_history[:-1], 'k-', label='Stock Price')
        ax2.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy')
        ax2.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell')
        
        ax2.set_title('Trading Decisions')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Stock Price ($)')
        ax2.legend()
        ax2.grid(True)
        
        # Returns distribution
        ax3.hist(self.returns_history, bins=50, alpha=0.75)
        ax3.set_title('Distribution of Returns')
        ax3.set_xlabel('Return')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('trading_performance.png')
        plt.show()
        
        # Return metrics
        cumulative_return = (portfolio_values[-2] - self.initial_balance) / self.initial_balance
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns_array = np.array(self.returns_history)
        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        
        # Maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        metrics = {
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.transaction_history)
        }
        
        return metrics