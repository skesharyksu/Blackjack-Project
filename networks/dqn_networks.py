import torch.nn as nn

class DQNNet(nn.Module):
    """
    Flexible DQN network architecture for Blackjack.
    Args:
        input_dim (int): State dimension.
        output_dim (int): Number of actions.
        hidden_layers (list): List of hidden layer sizes.
        activation (nn.Module): Activation function class (e.g., nn.ReLU).
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) 