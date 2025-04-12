import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def create_interactive_dashboard(env, agent, price_data, initial_balance):
    """
    Create a comprehensive dashboard of trading performance.
    
    Args:
        env: Trading environment after an episode
        agent: The trained agent
        price_data: Historical price data
        initial_balance: Initial account balance
    """
    # Create a Pandas DataFrame for easier manipulation
    if not env.transaction_history:
        print("No transactions found. Run the agent first.")
        return
    
    # Generate date range (assuming daily data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=len(price_data))
    date_range = pd.date_range(start=start_date, end=end_date, periods=len(price_data))
    
    # Create main DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Price': price_data,
    })
    
    # Add portfolio value over time
    portfolio_values = [initial_balance]
    for ret in env.returns_history:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # Ensure portfolio_values has the same length as df
    if len(portfolio_values) > len(df):
        portfolio_values = portfolio_values[:len(df)]
    elif len(portfolio_values) < len(df):
        # Pad with the last value
        padding = [portfolio_values[-1]] * (len(df) - len(portfolio_values))
        portfolio_values.extend(padding)
    
    df['PortfolioValue'] = portfolio_values
    
    # Add buy-and-hold strategy
    shares_bought = initial_balance // price_data[0]
    remaining_cash = initial_balance - shares_bought * price_data[0]
    df['BuyHoldValue'] = df['Price'] * shares_bought + remaining_cash
    
    # Add trading signals
    df['Signal'] = 'Hold'  # Default to Hold
    
    for transaction in env.transaction_history:
        if transaction['step'] < len(df):
            if transaction['action'] == 0:  # Buy
                df.loc[transaction['step'], 'Signal'] = 'Buy'
            elif transaction['action'] == 1:  # Sell
                df.loc[transaction['step'], 'Signal'] = 'Sell'
    
    # Calculate returns
    df['DailyReturn'] = df['Price'].pct_change()
    df['PortfolioReturn'] = df['PortfolioValue'].pct_change()
    df['BuyHoldReturn'] = df['BuyHoldValue'].pct_change()
    
    # Calculate performance metrics
    portfolio_return = (df['PortfolioValue'].iloc[-1] - initial_balance) / initial_balance
    buyhold_return = (df['BuyHoldValue'].iloc[-1] - initial_balance) / initial_balance
    
    portfolio_sharpe = np.mean(df['PortfolioReturn'].dropna()) / np.std(df['PortfolioReturn'].dropna()) * np.sqrt(252) if np.std(df['PortfolioReturn'].dropna()) > 0 else 0
    buyhold_sharpe = np.mean(df['BuyHoldReturn'].dropna()) / np.std(df['BuyHoldReturn'].dropna()) * np.sqrt(252) if np.std(df['BuyHoldReturn'].dropna()) > 0 else 0
    
    # Calculate drawdowns
    df['PortfolioDD'] = calculate_drawdown_series(df['PortfolioValue'])
    df['BuyHoldDD'] = calculate_drawdown_series(df['BuyHoldValue'])
    
    # Create the dashboard
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 2)
    
    # Plot 1: Price and Portfolio Value
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['Date'], df['Price'], 'k-', alpha=0.6, label='Stock Price')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['Date'], df['PortfolioValue'], 'b-', label='Portfolio Value')
    ax1_twin.plot(df['Date'], df['BuyHoldValue'], 'g-', label='Buy & Hold')
    
    # Add buy/sell markers
    buy_points = df[df['Signal'] == 'Buy']
    sell_points = df[df['Signal'] == 'Sell']
    
    ax1.scatter(buy_points['Date'], buy_points['Price'], marker='^', color='green', s=100, label='Buy')
    ax1.scatter(sell_points['Date'], sell_points['Price'], marker='v', color='red', s=100, label='Sell')
    
    ax1.set_title('Stock Price and Portfolio Value', fontsize=14)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.set_xlabel('')
    ax1_twin.set_ylabel('Portfolio Value ($)', fontsize=12)
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
    
    # Plot 2: Portfolio Performance Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Normalize values for comparison
    df['NormPortfolio'] = df['PortfolioValue'] / df['PortfolioValue'].iloc[0]
    df['NormBuyHold'] = df['BuyHoldValue'] / df['BuyHoldValue'].iloc[0]
    df['NormPrice'] = df['Price'] / df['Price'].iloc[0]
    
    ax2.plot(df['Date'], df['NormPortfolio'], 'b-', label='RL Agent')
    ax2.plot(df['Date'], df['NormBuyHold'], 'g-', label='Buy & Hold')
    ax2.plot(df['Date'], df['NormPrice'], 'k-', alpha=0.3, label='Stock Price')
    
    ax2.set_title('Normalized Performance Comparison', fontsize=14)
    ax2.set_ylabel('Normalized Value', fontsize=12)
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    
    # Plot 3: Drawdowns
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(df['Date'], 0, df['PortfolioDD'] * 100, color='blue', alpha=0.3, label='RL Agent')
    ax3.fill_between(df['Date'], 0, df['BuyHoldDD'] * 100, color='green', alpha=0.3, label='Buy & Hold')
    
    ax3.set_title('Drawdowns', fontsize=14)
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.legend()
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right')
    ax3.invert_yaxis()  # Invert y-axis to show drawdowns as negative
    
    # Plot 4: Rolling Returns
    ax4 = fig.add_subplot(gs[2, 0])
    window = 20  # 20-day rolling return
    df['RollingPortfolioReturn'] = df['PortfolioValue'].pct_change(window)
    df['RollingBuyHoldReturn'] = df['BuyHoldValue'].pct_change(window)
    
    ax4.plot(df['Date'][window:], df['RollingPortfolioReturn'][window:] * 100, 'b-', label='RL Agent')
    ax4.plot(df['Date'][window:], df['RollingBuyHoldReturn'][window:] * 100, 'g-', label='Buy & Hold')
    
    ax4.set_title(f'{window}-Day Rolling Returns', fontsize=14)
    ax4.set_ylabel('Return (%)', fontsize=12)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax4.legend()
    ax4.grid(True)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right')
    
    # Plot 5: Trade Analysis
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Calculate trade outcomes
    trades = []
    buy_price = None
    buy_index = None
    
    for i, row in df.iterrows():
        if row['Signal'] == 'Buy':
            buy_price = row['Price']
            buy_index = i
        elif row['Signal'] == 'Sell' and buy_price is not None:
            profit = (row['Price'] - buy_price) / buy_price * 100
            trades.append({
                'BuyDate': df.loc[buy_index, 'Date'],
                'SellDate': row['Date'],
                'BuyPrice': buy_price,
                'SellPrice': row['Price'],
                'Profit': profit,
                'HoldDays': (i - buy_index)
            })
            buy_price = None
    
    if trades:
        trade_df = pd.DataFrame(trades)
        
        # Plot trade outcomes
        profits = trade_df['Profit']
        ax5.bar(range(len(profits)), profits, color=['green' if p > 0 else 'red' for p in profits])
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        ax5.set_title('Trade Outcomes', fontsize=14)
        ax5.set_xlabel('Trade Number', fontsize=12)
        ax5.set_ylabel('Profit/Loss (%)', fontsize=12)
        ax5.grid(True, axis='y')
        
        # Add average line
        avg_profit = np.mean(profits)
        ax5.axhline(y=avg_profit, color='blue', linestyle='--', label=f'Avg: {avg_profit:.2f}%')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "No completed trades found", horizontalalignment='center',
                 verticalalignment='center', transform=ax5.transAxes)
    
    # Plot 6: Q-values Analysis
    ax6 = fig.add_subplot(gs[3, 0])
    
    # Count actions for different price-SMA relationships
    price_rel_states = [-2, -1, 0, 1, 2]  # Example discretized states
    actions = [0, 1, 2]  # Buy, Sell, Hold
    action_labels = ['Buy', 'Sell', 'Hold']
    action_matrix = np.zeros((len(price_rel_states), len(actions)))
    
    for state, q_values in agent.q_table.items():
        price_rel_sma = state[0]
        # Simplify price_rel_sma to fit into our 5 buckets
        bucket = min(max(price_rel_sma // 5, -2), 2)  # Map to -2, -1, 0, 1, 2
        bucket_idx = bucket + 2  # Shift to 0-4 index
        best_action = np.argmax(q_values)
        if 0 <= bucket_idx < len(price_rel_states):
            action_matrix[bucket_idx, best_action] += 1
    
    # Normalize by row
    row_sums = action_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.zeros_like(action_matrix)
    for i in range(len(price_rel_states)):
        if row_sums[i] > 0:
            norm_matrix[i] = action_matrix[i] / row_sums[i]
    
    # Create heatmap
    sns.heatmap(norm_matrix, annot=True, cmap='YlGnBu', cbar=True,
                xticklabels=action_labels,
                yticklabels=['Strong Undervalued', 'Undervalued', 'Fair Value', 'Overvalued', 'Strong Overvalued'],
                ax=ax6)
    
    ax6.set_title('Agent Decision Making by Price-SMA Relationship', fontsize=14)
    ax6.set_xlabel('Action', fontsize=12)
    ax6.set_ylabel('Price relative to SMA', fontsize=12)
    
    # Plot 7: Performance Metrics Table
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Calculate various metrics
    max_dd_portfolio = df['PortfolioDD'].max() * 100
    max_dd_buyhold = df['BuyHoldDD'].max() * 100
    
    # Volatility
    vol_portfolio = df['PortfolioReturn'].std() * np.sqrt(252) * 100
    vol_buyhold = df['BuyHoldReturn'].std() * np.sqrt(252) * 100
    
    # Count trades
    num_buys = len(df[df['Signal'] == 'Buy'])
    num_sells = len(df[df['Signal'] == 'Sell'])
    total_trades = num_buys + num_sells
    
    # Win rate
    if trades:
        win_rate = len([t for t in trades if t['Profit'] > 0]) / len(trades) * 100
        avg_profit_per_trade = np.mean([t['Profit'] for t in trades])
        avg_hold_days = np.mean([t['HoldDays'] for t in trades])
    else:
        win_rate = 0
        avg_profit_per_trade = 0
        avg_hold_days = 0
    
    metrics_table = [
        ['Metric', 'RL Agent', 'Buy & Hold'],
        ['Total Return (%)', f'{portfolio_return * 100:.2f}', f'{buyhold_return * 100:.2f}'],
        ['Sharpe Ratio', f'{portfolio_sharpe:.2f}', f'{buyhold_sharpe:.2f}'],
        ['Max Drawdown (%)', f'{max_dd_portfolio:.2f}', f'{max_dd_buyhold:.2f}'],
        ['Annualized Volatility (%)', f'{vol_portfolio:.2f}', f'{vol_buyhold:.2f}'],
        ['Total Trades', f'{total_trades}', '1'],
        ['Win Rate (%)', f'{win_rate:.2f}', 'N/A'],
        ['Avg Profit per Trade (%)', f'{avg_profit_per_trade:.2f}', 'N/A'],
        ['Avg Holding Period (days)', f'{avg_hold_days:.2f}', 'N/A']
    ]
    
    table = ax7.table(cellText=metrics_table, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Highlighting the better performance
    for i in range(2, 6):  # Rows with comparable metrics
        if float(metrics_table[i][1].split()[0]) > float(metrics_table[i][2].split()[0]):
            table[(i, 1)].set_facecolor('#c6efce')  # Light green
        elif float(metrics_table[i][1].split()[0]) < float(metrics_table[i][2].split()[0]):
            table[(i, 2)].set_facecolor('#c6efce')  # Light green
    
    # Special case for Max Drawdown (lower is better)
    if float(metrics_table[3][1].split()[0]) < float(metrics_table[3][2].split()[0]):
        table[(3, 1)].set_facecolor('#c6efce')  # Light green
    else:
        table[(3, 2)].set_facecolor('#c6efce')  # Light green
    
    plt.suptitle('RL Trading Agent Performance Dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return metrics dictionary
    return {
        'portfolio_return': portfolio_return,
        'buyhold_return': buyhold_return,
        'portfolio_sharpe': portfolio_sharpe,
        'buyhold_sharpe': buyhold_sharpe,
        'max_dd_portfolio': max_dd_portfolio,
        'max_dd_buyhold': max_dd_buyhold,
        'volatility_portfolio': vol_portfolio,
        'volatility_buyhold': vol_buyhold,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit_per_trade': avg_profit_per_trade,
        'avg_hold_days': avg_hold_days
    }

def calculate_drawdown_series(values):
    """
    Calculate a time series of drawdowns from peak.
    
    Args:
        values: Series of values (e.g., portfolio values)
        
    Returns:
        Series of drawdown values (0 to 1)
    """
    # Calculate running maximum
    running_max = values.cummax()
    
    # Calculate drawdown
    drawdown = (running_max - values) / running_max
    
    return drawdown

def create_3d_qvalue_visualization(agent):
    """
    Create a 3D visualization of Q-values.
    
    Args:
        agent: Trained RL agent
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract state components and Q-values
    states = []
    max_q_values = []
    best_actions = []
    
    for state, q_values in agent.q_table.items():
        # We'll use price_rel_sma and momentum as the x, y coordinates
        price_rel_sma, _, _, momentum, _ = state
        
        max_q = np.max(q_values)
        best_action = np.argmax(q_values)
        
        states.append((price_rel_sma, momentum))
        max_q_values.append(max_q)
        best_actions.append(best_action)
    
    # Convert to numpy arrays
    states = np.array(states)
    max_q_values = np.array(max_q_values)
    best_actions = np.array(best_actions)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for actions
    colors = ['green', 'red', 'blue']
    action_labels = ['Buy', 'Sell', 'Hold']
    
    # Create scatter plot
    for action in range(3):
        mask = best_actions == action
        if np.any(mask):
            ax.scatter(states[mask, 0], states[mask, 1], max_q_values[mask],
                      c=colors[action], label=action_labels[action], alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Price relative to SMA')
    ax.set_ylabel('Price Momentum')
    ax.set_zlabel('Max Q-Value')
    ax.set_title('3D Visualization of Q-values by State', fontsize=14)
    ax.legend()
    
    plt.savefig('3d_qvalue_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_learning_curve_comparison(reward_histories, labels, hyperparams=None):
    """
    Compare learning curves for different hyperparameter settings.
    
    Args:
        reward_histories: List of reward histories from different runs
        labels: List of labels for each run
        hyperparams: Dictionary of hyperparameters for each run (optional)
    """
    plt.figure(figsize=(12, 6))
    
    for i, rewards in enumerate(reward_histories):
        # Smooth rewards for visualization
        window_size = min(50, len(rewards) // 10)
        if window_size > 1:
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            episodes = range(window_size-1, len(rewards))
            plt.plot(episodes, smoothed, label=labels[i], linewidth=2)
        else:
            plt.plot(range(len(rewards)), rewards, label=labels[i], linewidth=2)
    
    plt.title('Learning Curve Comparison', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    if hyperparams:
        # Add hyperparameter details as text
        param_text = '\n'.join([f"{label}: {', '.join([f'{k}={v}' for k, v in params.items()])}" 
                               for label, params in zip(labels, hyperparams)])
        plt.figtext(0.5, 0.01, param_text, ha='center', fontsize=10, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig('learning_curve_comparison.png', dpi=300)
    plt.show()

def create_animated_trading_visualization(env, price_data, save_path='trading_animation.gif'):
    """
    Create an animated visualization of the trading process.
    
    Args:
        env: Trading environment after an episode
        price_data: Stock price data
        save_path: Path to save the animation
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        print("Matplotlib animation module not available. Cannot create animation.")
        return
    
    if not env.transaction_history:
        print("No transactions found. Run the agent first.")
        return
    
    # Create dataframe
    df = pd.DataFrame({
        'Step': range(len(price_data)),
        'Price': price_data
    })
    
    # Add transaction data
    df['Signal'] = 'None'
    for trans in env.transaction_history:
        step = trans['step']
        if step < len(df):
            df.loc[step, 'Signal'] = 'Buy' if trans['action'] == 0 else 'Sell'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    def update(frame):
        ax.clear()
        
        # Plot price data up to current frame
        ax.plot(df['Step'][:frame+1], df['Price'][:frame+1], 'k-')
        
        # Plot buy/sell signals up to current frame
        buys = df[(df['Step'] <= frame) & (df['Signal'] == 'Buy')]
        sells = df[(df['Step'] <= frame) & (df['Signal'] == 'Sell')]
        
        ax.scatter(buys['Step'], buys['Price'], color='green', s=100, marker='^')
        ax.scatter(sells['Step'], sells['Price'], color='red', s=100, marker='v')
        
        # Add labels and title
        ax.set_title(f'Trading Simulation (Day {frame})', fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='k', lw=2, label='Stock Price'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Buy'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=10, label='Sell')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
    anim = animation.FuncAnimation(fig, update, frames=len(df), repeat=True)
    
    try:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    
    plt.close(fig)