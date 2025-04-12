import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def train_agent(env, agent, episodes=1000, max_steps=None, print_interval=100, render=False):
    """
    Train the agent in the environment.
    
    Args:
        env: Stock trading environment
        agent: RL agent
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode (None for full episode)
        print_interval: How often to print results
        render: Whether to render the environment
        
    Returns:
        rewards_history, portfolio_values
    """
    rewards_history = []
    portfolio_values = []
    episode_lengths = []
    training_start_time = time.time()
    
    # Progress bar for training
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Agent selects an action
            action = agent.get_action(state)
            
            # Environment executes the action
            next_state, reward, done, info = env.step(action)
            
            # Agent learns from the experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1
            
            # Optionally render the environment
            if render and episode % print_interval == 0:
                env.render()
            
            # Check if maximum steps reached
            if max_steps is not None and steps >= max_steps:
                done = True
        
        # Store episode results
        rewards_history.append(total_reward)
        portfolio_values.append(env.total_value)
        episode_lengths.append(steps)
        
        # Print progress
        if episode % print_interval == 0 or episode == episodes - 1:
            avg_reward = np.mean(rewards_history[-print_interval:])
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}, "
                  f"Final Value = ${env.total_value:.2f}, "
                  f"Exploration Rate = {agent.exploration_rate:.4f}")
    
    training_time = time.time() - training_start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} steps")
    
    # Visualize training progress
    visualize_training(rewards_history, portfolio_values, agent.exploration_history)
    
    return rewards_history, portfolio_values

def visualize_training(rewards, portfolio_values, exploration_rates=None):
    """
    Visualize the training progress.
    
    Args:
        rewards: List of episode rewards
        portfolio_values: List of final portfolio values
        exploration_rates: List of exploration rates
    """
    episodes = range(len(rewards))
    
    # Create subplots
    if exploration_rates is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot episode rewards
    ax1.plot(episodes, rewards, 'b-')
    # Add a smoothed line for clarity
    if len(rewards) > 50:
        window_size = min(50, len(rewards) // 10)
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = range(window_size-1, len(rewards))
        ax1.plot(smoothed_episodes, smoothed, 'r-', linewidth=2, 
                 label=f'Moving Avg ({window_size} episodes)')
        ax1.legend()
    
    ax1.set_title('Episode Rewards During Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot portfolio values
    ax2.plot(episodes, portfolio_values, 'g-')
    ax2.set_title('Final Portfolio Value During Training')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True)
    
    # Plot exploration rate
    if exploration_rates is not None:
        ax3.plot(range(len(exploration_rates)), exploration_rates, 'r-')
        ax3.set_title('Exploration Rate (ε) During Training')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Exploration Rate')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

def evaluate_agent(env, agent, episodes=10, verbose=True):
    """
    Evaluate the trained agent.
    
    Args:
        env: Stock trading environment
        agent: Trained RL agent
        episodes: Number of evaluation episodes
        verbose: Whether to print detailed results
    """
    returns = []
    portfolio_values = []
    
    # Set agent to evaluation mode (no exploration)
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            episode_return += reward
            state = next_state
        
        returns.append(episode_return)
        portfolio_values.append(env.total_value)
        
        if verbose:
            print(f"Episode {episode}: Final Return = {episode_return:.2f}, "
                  f"Final Value = ${env.total_value:.2f}")
    
    # Calculate performance metrics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_value = np.mean(portfolio_values)
    
    print(f"\nEvaluation Results (over {episodes} episodes):")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Mean Final Portfolio Value: ${mean_value:.2f}")
    
    # Restore exploration rate
    agent.exploration_rate = original_exploration_rate
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_value': mean_value,
        'all_returns': returns,
        'all_values': portfolio_values
    }