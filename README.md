# Blackjack AI

A Q-learning AI agent that learns to play Blackjack optimally through reinforcement learning.

## Features
- Implementation of a Q-learning agent for Blackjack
- Clean visualization of gameplay and decision-making process
- Automatic training through multiple hands
- Saves and loads trained policies

## Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -e .
   ```
5. Start Jupyter:
   ```bash
   jupyter notebook
   ```
6. Open `blackjack_training.ipynb` to run the main AI training and evaluation notebook for the CIS 730 project.
7. Open `blackjack_modular_training.ipynb` to run the enhancements notebook, which includes DQN, improved statistical analysis, and enhanced visualizations for the CIS 732 project.

## Project Structure
- `src/blackjack_ai/` - Core implementation of the Blackjack environment and Q-learning agent
- `blackjack_training.ipynb` - Jupyter notebook for training and visualizing the AI
- `trained_policy.npy` - Pre-trained Q-learning policy 

## Agent Comparison & Statistical Analysis

This project compares three agents:
- **Q-Learning Agent:** Learns optimal play through reinforcement learning.
- **Basic Strategy Agent:** Plays using a deterministic, hardcoded basic strategy table.
- **Random Agent:** Chooses actions randomly.

### Statistical Evaluation
After running a series of games, the win rates of each agent are compared.  
A **binomial test** is used to calculate the p-value for each agent's win rate against the theoretical win rate for perfect basic strategy (42%).

**Null Hypothesis (H₀):**  
There is no significant difference in win rate between the Q-learning agent and the basic strategy or random agent.

**Alternative Hypothesis (H₁):**  
The Q-learning agent has a significantly higher win rate than the basic strategy or random agent.

**P-value Interpretation:**  
- A low p-value (< 0.05) means the agent's win rate is significantly different from 42%.
- A high p-value (≥ 0.05) means the agent's win rate is not significantly different from 42%.

### Example Output
```
Q-Learning Agent:
  Wins:   377 (37.7%)
  Losses: 497
  Pushes: 126
  p-value vs 42% win rate: 0.0059

Basic Strategy Agent:
  Wins:   390 (39.0%)
  Losses: 482
  Pushes: 128
  p-value vs 42% win rate: 0.0546

Random Agent:
  Wins:   239 (23.9%)
  Losses: 684
  Pushes: 77
  p-value vs 42% win rate: 0.0000
``` 

## Notebooks
- `blackjack_training.ipynb`: Main notebook for training, evaluating, and visualizing the Q-learning agent and baseline agents. **This notebook is the primary deliverable for CIS 730.**
- `blackjack_modular_training.ipynb`: Enhanced notebook with modular code, additional agent comparisons (including DQN and RandomAgent), advanced statistical analysis, and improved reward visualizations. **This notebook contains the enhancements and is the primary deliverable for CIS 732.** 
