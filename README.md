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
6. Open `blackjack_training.ipynb` to run the AI

## Project Structure
- `src/blackjack_ai/` - Core implementation of the Blackjack environment and Q-learning agent
- `blackjack_training.ipynb` - Jupyter notebook for training and visualizing the AI
- `trained_policy.npy` - Pre-trained Q-learning policy 