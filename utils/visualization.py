import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, window=100):
    """
    Plot episode rewards as a scatter plot and moving average as a line.
    Args:
        rewards (list): List of episode rewards.
        window (int): Window size for moving average.
    """
    plt.figure()
    plt.scatter(range(len(rewards)), rewards, label='Reward', color='tab:blue', alpha=0.2, s=10, zorder=1)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}-episode MA', color='tab:orange', linewidth=2, zorder=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Rewards')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_q_table(q_table):
    """
    Plot Q-table as heatmap (for tabular Q-learning).
    Args:
        q_table (dict): Q-table mapping state tuples to action values.
    """
    # Example: plot Q(player_sum, dealer_card, usable_ace=0) for action=0 (stand)
    player_sums = range(4, 22)
    dealer_cards = range(1, 11)
    q_matrix = np.zeros((len(player_sums), len(dealer_cards)))
    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            q_matrix[i, j] = q_table.get((ps, dc, 0), [0, 0, 0])[0]  # action=0
    plt.figure()
    plt.imshow(q_matrix, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Q-value (stand)')
    plt.xlabel('Dealer Showing')
    plt.ylabel('Player Sum')
    plt.title('Q-table Heatmap (Stand, Usable Ace=0)')
    plt.show()

def plot_dqn_q_values(q_net, states, action_names=None):
    """
    Plot DQN Q-values for selected states as bar charts.
    Args:
        q_net (nn.Module): Trained DQN network.
        states (list): List of state vectors to visualize.
        action_names (list): Optional list of action names.
    """
    import torch
    for idx, state in enumerate(states):
        q_vals = q_net(torch.FloatTensor(state).unsqueeze(0)).detach().numpy().flatten()
        plt.figure()
        plt.bar(range(len(q_vals)), q_vals)
        plt.xlabel('Action')
        plt.ylabel('Q-value')
        plt.title(f'DQN Q-values for State {idx+1}: {state}')
        if action_names:
            plt.xticks(range(len(q_vals)), action_names)
        plt.show() 