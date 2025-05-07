import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks.dqn_networks import DQNNet
from utils.replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network agent for Blackjack.
    Args:
        state_dim (int): State dimension.
        action_dim (int): Number of actions.
        hidden_layers (list): Hidden layer sizes.
        activation (nn.Module): Activation function.
        lr (float): Learning rate.
        gamma (float): Discount factor.
        epsilon_start (float): Initial exploration rate.
        epsilon_end (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        buffer_size (int): Replay buffer size.
        batch_size (int): Batch size for updates.
        target_update (int): Target network update frequency.
    """
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64], activation=nn.ReLU, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, buffer_size=10000, batch_size=64, target_update=100):
        self.q_net = DQNNet(state_dim, action_dim, hidden_layers, activation)
        self.target_net = DQNNet(state_dim, action_dim, hidden_layers, activation)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        self.action_dim = action_dim

    def select_action(self, state, deterministic=False, softmax_temp=None):
        """
        Selects an action using epsilon-greedy or softmax policy.
        Args:
            state (np.ndarray): State vector.
            deterministic (bool): If True, always pick argmax.
            softmax_temp (float): If set, use softmax policy with this temperature.
        Returns:
            int: action
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        if softmax_temp is not None:
            q_values = self.q_net(state_t).detach().numpy().flatten()
            exp_q = np.exp(q_values / softmax_temp)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(self.action_dim, p=probs)
        if deterministic or np.random.rand() > self.epsilon:
            with torch.no_grad():
                return self.q_net(state_t).argmax().item()
        else:
            return np.random.randint(self.action_dim)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay) 