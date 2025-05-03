import numpy as np
from typing import Tuple, Dict
import pickle
import os
from scipy import stats
import random

class QLearningAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 1.0, exploration_decay: float = 0.995,
                 min_exploration: float = 0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Initialize Q-table
        self.q_table = {}
        
        # Track performance metrics
        self.performance_history = {
            'wins': [],
            'losses': [],
            'pushes': [],
            'exploration_rate': []
        }

    def _get_state_key(self, state: Tuple) -> str:
        """Convert state tuple to string key for Q-table"""
        return str(state)

    def get_action(self, state: Tuple, is_training: bool = True) -> int:
        """Choose an action based on current state"""
        state_key = self._get_state_key(state)
        
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(3)  # 3 actions: stand, hit, double down

        # Exploration vs Exploitation
        if is_training and np.random.random() < self.exploration_rate:
            return np.random.randint(0, 3)  # Random action
        else:
            return np.argmax(self.q_table[state_key])

    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool):
        """Update Q-values based on experience"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(3)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(3)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state_key][action] = new_q

        # Update exploration rate
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay

    def update_performance(self, result: str):
        """Update performance tracking"""
        if result == 'win':
            self.performance_history['wins'].append(1)
            self.performance_history['losses'].append(0)
            self.performance_history['pushes'].append(0)
        elif result == 'loss':
            self.performance_history['wins'].append(0)
            self.performance_history['losses'].append(1)
            self.performance_history['pushes'].append(0)
        else:  # push
            self.performance_history['wins'].append(0)
            self.performance_history['losses'].append(0)
            self.performance_history['pushes'].append(1)
        
        self.performance_history['exploration_rate'].append(self.exploration_rate)

    def get_performance_metrics(self) -> Dict:
        """Return current performance metrics with statistical analysis"""
        if not self.performance_history['wins']:
            return None

        total_games = len(self.performance_history['wins'])
        wins = sum(self.performance_history['wins'])
        losses = sum(self.performance_history['losses'])
        pushes = sum(self.performance_history['pushes'])
        win_rate = wins / total_games if total_games > 0 else 0

        # Statistical analysis
        # Expected win rate with perfect basic strategy is about 42%
        expected_win_rate = 0.42
        
        # Binomial test against random chance (0.5)
        random_p_value = stats.binomtest(wins, total_games, p=0.5).pvalue
        
        # Binomial test against expected win rate
        expected_p_value = stats.binomtest(wins, total_games, p=expected_win_rate).pvalue
        
        # Confidence interval for win rate
        ci = stats.binomtest(wins, total_games).proportion_ci(confidence_level=0.95)
        
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'random_p_value': random_p_value,
            'expected_p_value': expected_p_value,
            'confidence_interval': ci,
            'expected_win_rate': expected_win_rate
        }

    def save_policy(self, filename: str):
        """Save the current Q-table and performance history"""
        data = {
            'q_table': self.q_table,
            'exploration_rate': self.exploration_rate,
            'performance_history': self.performance_history
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_policy(self, filename: str) -> bool:
        """Load a saved Q-table and performance history"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.exploration_rate = data['exploration_rate']
                self.performance_history = data['performance_history']
            return True
        except:
            return False

    def get_action_and_reason(self, state, is_training=True):
        action = self.get_action(state, is_training)
        if is_training and random.random() < self.exploration_rate:
            reason = f"Exploring: Random action (epsilon = {self.exploration_rate:.4f})"
        else:
            q_values = self.q_table[state]
            reason = f"Exploiting: Chosen action with Q-values: {q_values}"
        return action, reason

    def get_win_loss_stats(self):
        """Return only win/loss/push counts and rates (additive, does not affect bankroll or other stats)"""
        if not self.performance_history['wins']:
            return None
        total_games = len(self.performance_history['wins'])
        wins = sum(self.performance_history['wins'])
        losses = sum(self.performance_history['losses'])
        pushes = sum(self.performance_history['pushes'])
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'loss_rate': losses / total_games if total_games > 0 else 0,
            'push_rate': pushes / total_games if total_games > 0 else 0
        }

class BasicStrategyAgent:
    """Implements basic strategy for blackjack"""
    def __init__(self):
        # Basic strategy table: (player_sum, dealer_card, has_usable_ace) -> action
        # Actions: 0 = Stand, 1 = Hit, 2 = Double Down
        self.strategy_table = self._create_strategy_table()
        
        # Track performance metrics
        self.performance_history = {
            'wins': [],
            'losses': [],
            'pushes': []
        }

    def _create_strategy_table(self):
        """Create the basic strategy table"""
        table = {}
        
        # Hard totals
        for player_sum in range(4, 22):
            for dealer_card in range(2, 12):
                # Always hit if sum <= 11
                if player_sum <= 11:
                    table[(player_sum, dealer_card, 0)] = 1
                # Double down on 11 vs dealer 2-10
                elif player_sum == 11 and dealer_card != 1:
                    table[(player_sum, dealer_card, 0)] = 2
                # Stand on 17 and up
                elif player_sum >= 17:
                    table[(player_sum, dealer_card, 0)] = 0
                # 12-16 vs dealer 2-6: stand, otherwise hit
                elif 12 <= player_sum <= 16 and 2 <= dealer_card <= 6:
                    table[(player_sum, dealer_card, 0)] = 0
                else:
                    table[(player_sum, dealer_card, 0)] = 1
        
        # Soft totals (with usable ace)
        for player_sum in range(12, 22):
            for dealer_card in range(2, 12):
                # Always hit soft 17 or below
                if player_sum <= 17:
                    table[(player_sum, dealer_card, 1)] = 1
                # Stand on soft 19 and up
                elif player_sum >= 19:
                    table[(player_sum, dealer_card, 1)] = 0
                # Soft 18: stand vs dealer 2-8, hit vs 9-A
                elif player_sum == 18:
                    table[(player_sum, dealer_card, 1)] = 0 if dealer_card <= 8 else 1
        
        return table

    def get_action(self, state: Tuple, is_training: bool = False) -> int:
        """Get action based on basic strategy"""
        player_sum, dealer_card, has_usable_ace = state[:3]
        return self.strategy_table.get((player_sum, dealer_card, has_usable_ace), 1)  # Default to hit

    def update_performance(self, result: str):
        """Update performance tracking"""
        if result == 'win':
            self.performance_history['wins'].append(1)
            self.performance_history['losses'].append(0)
            self.performance_history['pushes'].append(0)
        elif result == 'loss':
            self.performance_history['wins'].append(0)
            self.performance_history['losses'].append(1)
            self.performance_history['pushes'].append(0)
        else:  # push
            self.performance_history['wins'].append(0)
            self.performance_history['losses'].append(0)
            self.performance_history['pushes'].append(1)

    def get_performance_metrics(self) -> Dict:
        """Return current performance metrics with statistical analysis"""
        if not self.performance_history['wins']:
            return None

        total_games = len(self.performance_history['wins'])
        wins = sum(self.performance_history['wins'])
        losses = sum(self.performance_history['losses'])
        pushes = sum(self.performance_history['pushes'])
        win_rate = wins / total_games if total_games > 0 else 0

        # Statistical analysis
        # Expected win rate with perfect basic strategy is about 42%
        expected_win_rate = 0.42
        
        # Binomial test against random chance (0.5)
        random_p_value = stats.binomtest(wins, total_games, p=0.5).pvalue
        
        # Binomial test against expected win rate
        expected_p_value = stats.binomtest(wins, total_games, p=expected_win_rate).pvalue
        
        # Confidence interval for win rate
        ci = stats.binomtest(wins, total_games).proportion_ci(confidence_level=0.95)
        
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'random_p_value': random_p_value,
            'expected_p_value': expected_p_value,
            'confidence_interval': ci,
            'expected_win_rate': expected_win_rate
        }

    def get_win_loss_stats(self):
        """Return only win/loss/push counts and rates (additive, does not affect bankroll or other stats)"""
        if not self.performance_history['wins']:
            return None
        total_games = len(self.performance_history['wins'])
        wins = sum(self.performance_history['wins'])
        losses = sum(self.performance_history['losses'])
        pushes = sum(self.performance_history['pushes'])
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'loss_rate': losses / total_games if total_games > 0 else 0,
            'push_rate': pushes / total_games if total_games > 0 else 0
        }

class RandomAgent:
    """A random agent for baseline comparison (chooses actions randomly)"""
    def __init__(self):
        self.performance_history = {
            'wins': [],
            'losses': [],
            'pushes': []
        }
    def get_action(self, state, is_training=False):
        return random.randint(0, 2)  # 0=Stand, 1=Hit, 2=Double Down
    def update_performance(self, result: str):
        if result == 'win':
            self.performance_history['wins'].append(1)
            self.performance_history['losses'].append(0)
            self.performance_history['pushes'].append(0)
        elif result == 'loss':
            self.performance_history['wins'].append(0)
            self.performance_history['losses'].append(1)
            self.performance_history['pushes'].append(0)
        else:
            self.performance_history['wins'].append(0)
            self.performance_history['losses'].append(0)
            self.performance_history['pushes'].append(1)
    def get_win_loss_stats(self):
        if not self.performance_history['wins']:
            return None
        total_games = len(self.performance_history['wins'])
        wins = sum(self.performance_history['wins'])
        losses = sum(self.performance_history['losses'])
        pushes = sum(self.performance_history['pushes'])
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'loss_rate': losses / total_games if total_games > 0 else 0,
            'push_rate': pushes / total_games if total_games > 0 else 0
        }

def calculate_statistics(wins: list, total_games: int, expected_win_rate: float = 0.42):
    """Calculate statistical metrics for win rates"""
    win_rate = sum(wins) / total_games
    
    # Binomial test against expected win rate
    p_value = stats.binomtest(sum(wins), total_games, p=expected_win_rate).pvalue
    
    # Calculate 95% confidence interval
    ci = stats.proportion_confint(sum(wins), total_games, alpha=0.05, method='wilson')
    
    return {
        'win_rate': win_rate,
        'p_value': p_value,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'total_games': total_games
    }

def compare_agents(agent1_wins: list, agent2_wins: list):
    """Perform statistical comparison between two agents"""
    # Chi-square test for independence
    contingency = np.array([
        [sum(agent1_wins), len(agent1_wins) - sum(agent1_wins)],
        [sum(agent2_wins), len(agent2_wins) - sum(agent2_wins)]
    ])
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'agent1_win_rate': sum(agent1_wins) / len(agent1_wins),
        'agent2_win_rate': sum(agent2_wins) / len(agent2_wins)
    }

def analyze_performance(agent, name: str = "Agent"):
    """Comprehensive performance analysis for an agent"""
    wins = agent.performance_history['wins']
    total_games = len(wins)
    
    # Basic statistics
    stats_vs_expected = calculate_statistics(wins, total_games)
    stats_vs_random = calculate_statistics(wins, total_games, expected_win_rate=0.5)
    
    # Bankroll analysis
    bankroll = agent.performance_history['bankroll']
    final_bankroll = bankroll[-1] if bankroll else 0
    bankroll_change = final_bankroll - bankroll[0] if bankroll else 0
    
    return {
        'name': name,
        'stats_vs_expected': stats_vs_expected,
        'stats_vs_random': stats_vs_random,
        'final_bankroll': final_bankroll,
        'bankroll_change': bankroll_change,
        'total_games': total_games
    }