"""
MP6 Submission: Blackjack AI with Q-Learning
Author: [Your Name]
Date: [Current Date]

This file contains the training and runtime execution code for a Q-learning based Blackjack AI agent.
The implementation is based on the following sources:

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
   - Q-learning algorithm implementation
   - State-action value function updates
   - Epsilon-greedy exploration strategy

2. OpenAI Gym Blackjack environment (modified)
   - State representation
   - Action space definition
   - Reward structure
   - Game mechanics

The code includes:
- Data preparation for training
- Q-learning agent implementation
- Training loop
- Policy saving and loading
- Runtime execution for playing games
"""

import numpy as np
import matplotlib.pyplot as plt
from src.blackjack_ai.environment import BlackjackEnv
from src.blackjack_ai.agents import QLearningAgent

def train_agent(episodes=100000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Train the Q-learning agent on the Blackjack environment.
    
    Parameters:
    - episodes: Number of training episodes
    - alpha: Learning rate
    - gamma: Discount factor
    - epsilon: Exploration rate
    
    Returns:
    - Trained agent
    - Training statistics
    """
    env = BlackjackEnv()
    agent = QLearningAgent(env.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon)
    
    # Training statistics
    rewards = []
    wins = 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Choose action using epsilon-greedy policy
            action = agent.choose_action(state)
            
            # Take action and observe result
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-values
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done and reward > 0:
                wins += 1
        
        rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10000 == 0:
            win_rate = wins / (episode + 1)
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Average reward: {np.mean(rewards[-10000:]):.2f}")
            print(f"Win rate: {win_rate:.2%}")
    
    return agent, rewards

def save_policy(agent, filename='trained_policy.npy'):
    """Save the learned policy to a file."""
    np.save(filename, agent.q_table)

def load_policy(filename='trained_policy.npy'):
    """Load a saved policy."""
    q_table = np.load(filename)
    env = BlackjackEnv()
    agent = QLearningAgent(env.action_space.n)
    agent.q_table = q_table
    return agent

def play_game(agent, num_games=1000):
    """
    Play games using the trained agent and collect statistics.
    
    Parameters:
    - agent: Trained Q-learning agent
    - num_games: Number of games to play
    
    Returns:
    - Game statistics
    """
    env = BlackjackEnv()
    wins = 0
    rewards = []
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        game_reward = 0
        
        while not done:
            action = agent.choose_action(state, epsilon=0)  # No exploration
            state, reward, done, _ = env.step(action)
            game_reward += reward
            
            if done and reward > 0:
                wins += 1
        
        rewards.append(game_reward)
    
    return {
        'win_rate': wins / num_games,
        'average_reward': np.mean(rewards),
        'std_reward': np.std(rewards)
    }

def plot_training_results(rewards, window=1000):
    """Plot training results."""
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(rewards, np.ones(window)/window, mode='valid'))
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Training parameters
    EPISODES = 100000
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.1
    
    # Train the agent
    print("Training agent...")
    agent, rewards = train_agent(
        episodes=EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON
    )
    
    # Save the trained policy
    save_policy(agent)
    
    # Plot training results
    plot_training_results(rewards)
    
    # Evaluate the trained agent
    print("\nEvaluating agent...")
    stats = play_game(agent)
    print(f"Win rate: {stats['win_rate']:.2%}")
    print(f"Average reward: {stats['average_reward']:.2f}")
    print(f"Reward standard deviation: {stats['std_reward']:.2f}") 