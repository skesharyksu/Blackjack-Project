import numpy as np
import random
from collections import defaultdict
from typing import Tuple, Dict, Any

class QLearningAgent:
    def __init__(self, learning_rate: float = 0.05, 
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.9995):
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.005

    def get_action(self, state, is_training=True):
        if is_training and random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward if done else reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save_policy(self, filename="q_learning_policy.npy"):
        policy_dict = dict(self.q_table)
        np.save(filename, policy_dict)

    def load_policy(self, filename="q_learning_policy.npy"):
        try:
            policy_dict = np.load(filename, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(3), policy_dict)
            return True
        except FileNotFoundError:
            print(f"Policy file {filename} not found.")
            return False

    def get_action_and_reason(self, state, is_training=True):
        action = self.get_action(state, is_training)
        if is_training and random.random() < self.epsilon:
            reason = f"Exploring: Random action (epsilon = {self.epsilon:.4f})"
        else:
            q_values = self.q_table[state]
            reason = f"Exploiting: Chosen action with Q-values: {q_values}"
        return action, reason