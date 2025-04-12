import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import pickle

from data_loader import load_stock_data_from_csv, generate_synthetic_data, add_market_regimes, visualize_data, split_data
from environment import StockTradingEnv
from agent import QLearningAgent
from train import train_agent, evaluate_agent
from evaluate import evaluate_against_baseline, analyze_trading_patterns, visualize_q_values

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Trading RL Agent')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'analyze'],
                        help='Mode: train, evaluate, or analyze')
    
    parser.add_argument('--data', type=str, default='synthetic',
                        help='Data source: path to CSV file or "synthetic"')
    
    parser.add_argument('--price-col', type=str, default='Close',
                        help='Column name for price data in CSV')
    
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial balance for trading')
    
    parser.add_argument('--fee', type=float, default=0.001,
                        help='Transaction fee percentage')
    
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor')
    
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial exploration rate')
    
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Exploration rate decay')
    
    parser.add_argument('--min-epsilon', type=float, default=0.01,
                        help='Minimum exploration rate')
    
    parser.add_argument('--save-model', type=str, default='model.pkl',
                        help='Path to save the trained model')
    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load a trained model')
    
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during training')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize data and results')
    
    return parser.parse_args()

def main():
    """Main function to run the application."""
    args = parse_args()
    
    # Load or generate data
    if args.data == 'synthetic':
        print("Generating synthetic data...")
        raw_data = generate_synthetic_data(days=1000, volatility=0.01, trend=0.0001)
        data = add_market_regimes(raw_data)
    else:
        print(f"Loading data from {args.data}...")
        data = load_stock_data_from_csv(args.data, args.price_col)
        if data is None:
            print("Failed to load data. Exiting.")
            return
    
    # Visualize data if requested
    if args.visualize:
        data_stats = visualize_data(data)
        print("Data visualization completed.")
    
    # Split data
    train_data, val_data, test_data = split_data(data)
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create environment and agent for the appropriate dataset
    if args.mode == 'train':
        env = StockTradingEnv(train_data, args.initial_balance, args.fee)
        agent = QLearningAgent(
            action_space=3,
            learning_rate=args.lr,
            discount_factor=args.gamma,
            exploration_rate=args.epsilon,
            exploration_decay=args.epsilon_decay,
            min_exploration_rate=args.min_epsilon
        )
    else:
        env = StockTradingEnv(test_data, args.initial_balance, args.fee)
        
        if args.load_model:
            print(f"Loading model from {args.load_model}...")
            with open(args.load_model, 'rb') as f:
                agent = pickle.load(f)
        else:
            print("No model specified for evaluation. Creating a new agent.")
            agent = QLearningAgent(action_space=3)
    
    # Execute the requested mode
    if args.mode == 'train':
        print(f"Training agent for {args.episodes} episodes...")
        train_rewards, train_values = train_agent(
            env, agent, episodes=args.episodes, render=args.render
        )
        
        # Save the trained model
        if args.save_model:
            print(f"Saving model to {args.save_model}...")
            with open(args.save_model, 'wb') as f:
                pickle.dump(agent, f)
        
        # Evaluate on validation data
        print("\nEvaluating on validation data...")
        val_env = StockTradingEnv(val_data, args.initial_balance, args.fee)
        eval_results = evaluate_agent(val_env, agent, episodes=5)
        
        # Visualize agent's learning progress
        agent.visualize_learning()
        
    elif args.mode == 'evaluate':
        print("Evaluating agent performance...")
        evaluation_metrics = evaluate_against_baseline(env, agent, test_data, args.initial_balance)
        
        # Run a single episode for visualization
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Visualize performance
        metrics = env.visualize_performance()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
    elif args.mode == 'analyze':
        print("Analyzing agent behavior...")
        
        # Run a single episode
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Analyze trading patterns
        pattern_analysis = analyze_trading_patterns(env)
        
        # Visualize Q-values
        visualize_q_values(agent, env)
    
    print("Done!")

if __name__ == "__main__":
    main()