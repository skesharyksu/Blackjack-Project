import numpy as np
import random
from types import SimpleNamespace

class BlackjackEnv:
    """
    Blackjack environment for RL agents.
    Args:
        num_decks (int): Number of decks to use in the shoe.
    """
    def __init__(self, num_decks=4):
        self.num_decks = num_decks
        self.deck = self._init_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.done = False
        self.reward = 0
        # Add Gym-like spaces for RL agent compatibility
        self.observation_space = SimpleNamespace(shape=(3,))
        self.action_space = SimpleNamespace(n=3)

    def _init_deck(self):
        deck = [r for r in range(1, 11)] * 4 * self.num_decks  # 1=Ace, 2-10
        deck += [10] * 12 * self.num_decks  # J, Q, K as 10
        random.shuffle(deck)
        return deck

    def _draw_card(self):
        if not self.deck:
            self.deck = self._init_deck()
        return self.deck.pop()

    def _hand_value(self, hand):
        total = sum(hand)
        ace_count = hand.count(1)
        while ace_count > 0 and total + 10 <= 21:
            total += 10
            ace_count -= 1
        return total

    def reset(self):
        """Resets the environment and deals initial cards."""
        self.deck = self._init_deck()
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        self.done = False
        self.reward = 0
        return self._get_state()

    def _get_state(self):
        player_sum = self._hand_value(self.player_hand)
        dealer_show = self.dealer_hand[0]
        usable_ace = int(1 in self.player_hand and player_sum <= 21)
        return np.array([player_sum, dealer_show, usable_ace], dtype=np.float32)

    def step(self, action):
        """
        Takes an action: 0=stand, 1=hit, 2=double down.
        Returns: next_state, reward, done, info
        """
        if self.done:
            return self._get_state(), self.reward, self.done, {}
        info = {}
        if action == 1:  # Hit
            self.player_hand.append(self._draw_card())
            player_sum = self._hand_value(self.player_hand)
            if player_sum > 21:
                self.done = True
                self.reward = -1
            return self._get_state(), self.reward, self.done, info
        elif action == 2:  # Double down
            if len(self.player_hand) > 2:
                self.done = True
                self.reward = -1
                return self._get_state(), self.reward, self.done, info
            self.player_hand.append(self._draw_card())
            player_sum = self._hand_value(self.player_hand)
            if player_sum > 21:
                self.done = True
                self.reward = -2
                return self._get_state(), self.reward, self.done, info
            # Stand after double down
            return self._dealer_play(double=True)
        else:  # Stand
            return self._dealer_play(double=False)

    def _dealer_play(self, double=False):
        player_sum = self._hand_value(self.player_hand)
        while self._hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self._draw_card())
        dealer_sum = self._hand_value(self.dealer_hand)
        self.done = True
        if player_sum > 21:
            self.reward = -2 if double else -1
        elif dealer_sum > 21 or player_sum > dealer_sum:
            self.reward = 2 if double else 1
        elif player_sum < dealer_sum:
            self.reward = -2 if double else -1
        else:
            self.reward = 0
        return self._get_state(), self.reward, self.done, {} 