# Blackjack AI: A Q-Learning Approach
## Final Project Report

### 1. Introduction

#### Project Overview
This project implements an AI agent that learns to play Blackjack using Q-learning, a model-free reinforcement learning algorithm. The system simulates the game environment, allowing the agent to learn optimal strategies through trial and error. The implementation includes a custom Blackjack environment, Q-learning algorithm, and tools for tracking and visualizing the learning process.

#### Problem Statement
Blackjack presents an interesting challenge for AI systems due to its combination of chance and strategy. While basic strategy exists, developing an AI that can learn and adapt its strategy through experience is valuable for both educational and practical purposes. The challenge lies in creating an agent that can learn optimal decisions without explicit programming of game rules or strategies.

#### Scope and Challenges
The project scope includes:
- Implementing a custom Blackjack environment
- Developing a Q-learning agent
- Training the agent through simulation
- Evaluating performance against basic strategy

Key challenges:
- State space complexity in Blackjack
- Balancing exploration and exploitation
- Efficient learning with sparse rewards
- Measuring and comparing performance

### 2. Background & Related Work

#### Existing Approaches
Traditional approaches to Blackjack include:
- Basic strategy tables
- Card counting systems
- Monte Carlo methods
- Rule-based expert systems

#### How This Project Differs
This project differs from existing approaches by:
- Using Q-learning for strategy development
- Learning through experience rather than pre-programmed rules
- Adapting to different game conditions
- Providing real-time visualization of learning progress

### 3. Dataset and Processing

#### Data Sources
The project uses simulated game data generated through:
- Custom Blackjack environment
- Random game play for initial exploration
- Policy-guided play during training

#### Preprocessing and Filtering
- State representation: (player_sum, dealer_card, usable_ace)
- Action space: {hit, stand, double}
- Reward structure: {-2, -1, 0, 1} for different outcomes
- Normalization of state values

### 4. Methodology

#### Baseline Model
The Q-learning implementation uses a tabular approach with the following key components:

```python
class QLearningAgent:
    def __init__(self, learning_rate: float = 0.05, 
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.9995):
        self.q_table = defaultdict(lambda: np.zeros(3))  # 3 actions: hit, stand, double
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.005
```

Key features:
- Adaptive exploration rate with decay
- Tabular Q-table for state-action values
- Three possible actions: hit, stand, double down
- Learning rate of 0.05 for stable updates
- Discount factor of 0.95 for future reward consideration

#### State Representation
The environment uses a compact state representation:

```python
def _get_state(self):
    """Return the current state of the game"""
    dealer_showing = self.dealer_hand[0].value
    return (self.player_sum, dealer_showing, int(self.player_has_usable_ace))
```

State components:
1. Player's current sum (4-21)
2. Dealer's showing card (1-10)
3. Whether player has a usable ace (0 or 1)

#### Training Process
The training loop implements the Q-learning update rule:

```python
def update(self, state, action, reward, next_state, done):
    best_next_action = np.argmax(self.q_table[next_state])
    td_target = reward if done else reward + self.gamma * self.q_table[next_state][best_next_action]
    td_error = td_target - self.q_table[state][action]
    self.q_table[state][action] += self.lr * td_error
```

Key aspects:
- Temporal Difference (TD) learning
- Epsilon-greedy exploration
- Adaptive learning rate
- Experience replay buffer

### 5. Technical Implementation

#### Environment Design
The Blackjack environment implements casino rules:

```python
class BlackjackEnv:
    def __init__(self, num_decks: int = 4):
        self.deck = Deck(num_decks)
        self.player_hand: List[Card] = []
        self.dealer_hand: List[Card] = []
        self.game_over = False
        self.player_sum = 0
        self.dealer_sum = 0
        self.player_has_usable_ace = False
        self.dealer_has_usable_ace = False
```

Features:
- Multiple deck support
- Proper ace handling
- Dealer strategy implementation
- Natural blackjack detection

#### Action Space
The environment supports three actions:

```python
def step(self, action):
    if action == 0:  # Stick
        return self._dealer_play()
    elif action == 1:  # Hit
        self.player_hand.append(self.deck.deal())
        self._calculate_hand_value()
    elif action == 2:  # Double down
        if len(self.player_hand) > 2:
            return self._get_state(), -1, False, info
```

#### Reward Structure
The reward system is designed to encourage optimal play:

```python
def _dealer_play(self):
    if self.dealer_sum > 21:
        return self._get_state(), 1, True, info    # Win
    elif self.dealer_sum > self.player_sum:
        return self._get_state(), -1, True, info   # Loss
    elif self.dealer_sum < self.player_sum:
        return self._get_state(), 1, True, info    # Win
    else:
        return self._get_state(), 0, True, info    # Draw
```

Rewards:
- +1 for winning
- -1 for losing
- 0 for drawing
- -2 for busting after double down

#### Performance Optimization
Key optimizations include:

1. Efficient State Representation:
```python
# Using tuples for immutable state representation
state = (player_sum, dealer_showing, int(has_usable_ace))
```

2. Memory-Efficient Q-Table:
```python
# Using defaultdict for sparse state representation
self.q_table = defaultdict(lambda: np.zeros(3))
```

3. Batch Processing:
```python
# Vectorized operations for Q-updates
td_error = td_target - self.q_table[state][action]
self.q_table[state][action] += self.lr * td_error
```

### 6. Results

#### Performance Metrics
- Training convergence analysis
- Win rate progression
- Policy stability measures
- Computational efficiency

#### Observations
- Learning curve characteristics
- Policy quality assessment
- Resource utilization
- Scalability considerations

### 7. Conclusions & Future Work

#### Key Achievements
- Successful implementation of Q-learning for Blackjack
- Demonstrated learning capability
- Efficient state representation
- Practical training framework

#### Future Improvements
- Deep Q-learning implementation
- Multi-agent training
- Real-time visualization
- Web interface development

### 8. Challenges Encountered

#### Development Challenges
- State space complexity
- Exploration-exploitation balance
- Training stability
- Performance optimization

#### Solutions
- Adaptive exploration strategy
- Experience replay implementation
- Hyperparameter tuning
- Efficient data structures

### 9. References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Thorp, E. O. (1966). Beat the Dealer: A Winning Strategy for the Game of Twenty-One. Vintage.
3. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
4. OpenAI Gym: A toolkit for developing and comparing reinforcement learning algorithms. 