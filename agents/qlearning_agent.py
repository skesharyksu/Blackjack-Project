import numpy as np
from collections import defaultdict

class QLearningAgent:
    """
    Tabular Q-learning agent for Blackjack.
    Args:
        state_space_size (tuple): Shape of discretized state space.
        action_space_size (int): Number of possible actions.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        min_epsilon (float): Minimum exploration rate.
    """
    def __init__(self, state_space_size=(32, 11, 2), action_space_size=3, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

    def _discretize(self, state):
        # Discretize state for tabular Q (player_sum, dealer_card, usable_ace)
        player_sum = min(max(int(state[0]), 0), self.state_space_size[0] - 1)
        dealer_card = min(max(int(state[1]), 0), self.state_space_size[1] - 1)
        usable_ace = int(state[2])
        return (player_sum, dealer_card, usable_ace)

    def select_action(self, state, epsilon=None):
        """
        Args:
            state (np.ndarray): [player_sum, dealer_card, usable_ace]
            epsilon (float): Exploration rate (optional).
        Returns:
            int: action (0=stand, 1=hit, 2=double)
        """
        eps = self.epsilon if epsilon is None else epsilon
        s = self._discretize(state)
        if np.random.rand() < eps:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.q_table[s])

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update rule.
        """
        s = self._discretize(state)
        ns = self._discretize(next_state)
        best_next = np.max(self.q_table[ns])
        td_target = reward + (0 if done else self.gamma * best_next)
        td_error = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.alpha * td_error
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay) 