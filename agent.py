import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class QLearningAgent:
    """
    Q-Learning agent for stock trading.
    """
    
    def __init__(self, action_space=3, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize the Q-Learning agent.
        
        Args:
            action_space: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which to decay exploration
            min_exploration_rate: Minimum exploration rate
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table as a defaultdict to handle new states automatically
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))
        
        # Metrics for tracking learning progress
        self.q_values_history = []
        self.exploration_history = []
        self.action_frequency = {0: 0, 1: 0, 2: 0}  # Buy, Sell, Hold
    
    def discretize_state(self, observation):
        """
        Convert continuous observation to discrete state representation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Tuple representing discrete state
        """
        price = observation['price']
        balance = observation['balance']
        shares = observation['shares']
        
        # Technical indicators
        sma = observation.get('sma', price)
        momentum = observation.get('momentum', 0)
        volatility = observation.get('volatility', 0)
        
        # Discretize price relative to SMA
        if sma > 0:
            price_rel_sma = round((price / sma - 1) * 20)  # Normalize around 0
        else:
            price_rel_sma = 0
            
        # Discretize amount of cash available (as percentage of initial balance)
        if balance > 0:
            balance_disc = int(balance / 1000)  # Each $1000 is a different state
        else:
            balance_disc = 0
            
        # Discretize shares held
        shares_disc = min(int(shares / 10), 10)  # Cap at 10 states for shares
        
        # Discretize momentum into 5 bins
        if momentum > 0.05:
            momentum_disc = 2  # Strong positive
        elif momentum > 0.01:
            momentum_disc = 1  # Positive
        elif momentum < -0.05:
            momentum_disc = -2  # Strong negative
        elif momentum < -0.01:
            momentum_disc = -1  # Negative
        else:
            momentum_disc = 0  # Neutral
            
        # Discretize volatility into 3 bins
        if volatility > 0.02:
            volatility_disc = 2  # High
        elif volatility > 0.01:
            volatility_disc = 1  # Medium
        else:
            volatility_disc = 0  # Low
        
        # Return the state tuple
        return (price_rel_sma, balance_disc, shares_disc, momentum_disc, volatility_disc)
    
    def get_action(self, observation):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Selected action
        """
        state = self.discretize_state(observation)
        
        # Exploration: choose random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space)
        
        # Exploitation: choose best action based on Q-values
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values based on experience.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode is done
        """
        # Discretize states
        state_disc = self.discretize_state(state)
        next_state_disc = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_disc][action]
        
        # Next state's maximum Q-value
        next_max_q = np.max(self.q_table[next_state_disc]) if not done else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        # Update Q-table
        self.q_table[state_disc][action] = new_q
        
        # Track action frequency
        self.action_frequency[action] += 1
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(
                self.min_exploration_rate, 
                self.exploration_rate * self.exploration_decay
            )
            self.exploration_history.append(self.exploration_rate)
            
            # Store average Q-values
            avg_q = np.mean([np.max(q_values) for q_values in self.q_table.values()])
            self.q_values_history.append(avg_q)
    
    def visualize_learning(self):
        """
        Visualize the agent's learning progress.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot exploration rate over time
        episodes = range(len(self.exploration_history))
        ax1.plot(episodes, self.exploration_history, 'b-')
        ax1.set_title('Exploration Rate Over Time')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Exploration Rate (ε)')
        ax1.grid(True)
        
        # Plot average Q-values over time
        if self.q_values_history:
            ax2.plot(episodes, self.q_values_history, 'r-')
            ax2.set_title('Average Maximum Q-Value Over Time')
            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('Average Max Q-Value')
            ax2.grid(True)
        
        # Plot action frequency
        actions = list(self.action_frequency.keys())
        frequencies = list(self.action_frequency.values())
        ax3.bar(actions, frequencies, color=['green', 'red', 'blue'])
        ax3.set_title('Action Frequency')
        ax3.set_xlabel('Action (0: Buy, 1: Sell, 2: Hold)')
        ax3.set_ylabel('Frequency')
        ax3.set_xticks(actions)
        ax3.set_xticklabels(['Buy', 'Sell', 'Hold'])
        
        plt.tight_layout()
        plt.savefig('learning_visualization.png')
        plt.show()