from typing import Tuple, Dict, List, Any
from .card import Card, Deck

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
        self.history = []  # Track game history for analysis

    def reset(self):
        """Reset the game environment and deal initial cards"""
        self.player_hand = []
        self.dealer_hand = []
        self.game_over = False

        # Deal two cards to player and dealer
        self.player_hand.append(self.deck.deal())
        self.dealer_hand.append(self.deck.deal())
        self.player_hand.append(self.deck.deal())
        self.dealer_hand.append(self.deck.deal())

        # Calculate initial sums
        self._calculate_hand_value()

        # Check for natural blackjack
        if self.player_sum == 21:
            self.game_over = True

        return self._get_state()

    def _calculate_hand_value(self):
        """Calculate the value of player and dealer hands, accounting for aces"""
        # Calculate player hand
        self.player_sum = 0
        num_aces = 0
        for card in self.player_hand:
            if card.rank == 'A':
                num_aces += 1
            self.player_sum += card.value

        # Adjust for aces if needed
        self.player_has_usable_ace = False
        while self.player_sum > 21 and num_aces > 0:
            self.player_sum -= 10
            num_aces -= 1

        if num_aces > 0 and self.player_sum <= 21:
            self.player_has_usable_ace = True

        # Calculate dealer hand
        self.dealer_sum = 0
        num_aces = 0
        for card in self.dealer_hand:
            if card.rank == 'A':
                num_aces += 1
            self.dealer_sum += card.value

        # Adjust for aces if needed
        self.dealer_has_usable_ace = False
        while self.dealer_sum > 21 and num_aces > 0:
            self.dealer_sum -= 10
            num_aces -= 1

        if num_aces > 0 and self.dealer_sum <= 21:
            self.dealer_has_usable_ace = True

    def _get_state(self):
        """Return the current state of the game"""
        dealer_showing = self.dealer_hand[0].value
        return (self.player_sum, dealer_showing, int(self.player_has_usable_ace))

    def step(self, action):
        """Take an action in the game"""
        info = {}

        if self.game_over:
            return self._get_state(), 0, True, info

        if action == 0:  # Stand
            return self._dealer_play()

        elif action == 1:  # Hit
            # Deal one more card to player
            self.player_hand.append(self.deck.deal())
            self._calculate_hand_value()

            # Show updated hands (all player cards, one dealer card)
            info['player_hand'] = self.format_hand(self.player_hand)
            info['dealer_hand'] = self.format_hand(self.dealer_hand, show_all=False)

            if self.player_sum > 21:
                self.game_over = True
                info['dealer_hand'] = self.format_hand(self.dealer_hand, show_all=True)
                self.history.append({
                    'result': 'loss',
                    'player_sum': self.player_sum,
                    'dealer_sum': self.dealer_sum
                })
                return self._get_state(), -1, True, info

            # Defensive: too many cards, force game over
            if len(self.player_hand) > 10:
                self.game_over = True
                info['dealer_hand'] = self.format_hand(self.dealer_hand, show_all=True)
                self.history.append({
                    'result': 'loss',
                    'player_sum': self.player_sum,
                    'dealer_sum': self.dealer_sum
                })
                return self._get_state(), -1, True, info

            return self._get_state(), 0, False, info

        elif action == 2:  # Double down
            if len(self.player_hand) > 2:
                # Can't double down after hitting, end the game to avoid infinite loop
                self.game_over = True
                info['player_hand'] = self.format_hand(self.player_hand)
                info['dealer_hand'] = self.format_hand(self.dealer_hand, show_all=True)
                self.history.append({
                    'result': 'loss',
                    'player_sum': self.player_sum,
                    'dealer_sum': self.dealer_sum
                })
                return self._get_state(), -1, True, info

            # Deal one card and show the complete hand
            self.player_hand.append(self.deck.deal())
            self._calculate_hand_value()

            # Show all player cards but only one dealer card
            info['player_hand'] = self.format_hand(self.player_hand)
            info['dealer_hand'] = self.format_hand(self.dealer_hand, show_all=False)

            if self.player_sum > 21:
                self.game_over = True
                info['dealer_hand'] = self.format_hand(self.dealer_hand, show_all=True)
                self.history.append({
                    'result': 'loss',
                    'player_sum': self.player_sum,
                    'dealer_sum': self.dealer_sum
                })
                return self._get_state(), -2, True, info

            # Now dealer plays (since double down forces stand)
            next_state, reward, done, dealer_info = self._dealer_play()
            info.update(dealer_info)
            return next_state, reward * 2, done, info

    def _dealer_play(self):
        """Execute dealer's strategy and determine the game outcome"""
        self.game_over = True
        info = {}

        while self.dealer_sum < 17:
            self.dealer_hand.append(self.deck.deal())
            self._calculate_hand_value()

        info['player_hand'] = self.format_hand(self.player_hand, show_all=True)
        info['dealer_hand'] = self.format_hand(self.dealer_hand, show_all=True)

        # Calculate reward based on outcome
        if self.dealer_sum > 21:
            self.history.append({
                'result': 'win',
                'player_sum': self.player_sum,
                'dealer_sum': self.dealer_sum
            })
            return self._get_state(), 1, True, info
        elif self.dealer_sum > self.player_sum:
            self.history.append({
                'result': 'loss',
                'player_sum': self.player_sum,
                'dealer_sum': self.dealer_sum
            })
            return self._get_state(), -1, True, info
        elif self.dealer_sum < self.player_sum:
            self.history.append({
                'result': 'win',
                'player_sum': self.player_sum,
                'dealer_sum': self.dealer_sum
            })
            return self._get_state(), 1, True, info
        else:
            self.history.append({
                'result': 'push',
                'player_sum': self.player_sum,
                'dealer_sum': self.dealer_sum
            })
            return self._get_state(), 0, True, info

    def format_hand(self, hand, show_all=False):
        """Format a hand of cards into a string"""
        if show_all or hand == self.player_hand:  # Always show all player cards
            return ", ".join(str(card) for card in hand)
        else:
            return str(hand[0])

    def get_statistics(self):
        """Return game statistics"""
        if not self.history:
            return None

        total_games = len(self.history)
        wins = sum(1 for game in self.history if game['result'] == 'win')
        losses = sum(1 for game in self.history if game['result'] == 'loss')
        pushes = sum(1 for game in self.history if game['result'] == 'push')
        
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': wins / total_games if total_games > 0 else 0
        }

    def get_win_loss_stats(self):
        """Return only win/loss/push counts and rates (additive, does not affect bankroll or other stats)"""
        if not self.history:
            return None
        total_games = len(self.history)
        wins = sum(1 for game in self.history if game['result'] == 'win')
        losses = sum(1 for game in self.history if game['result'] == 'loss')
        pushes = sum(1 for game in self.history if game['result'] == 'push')
        return {
            'total_games': total_games,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': wins / total_games if total_games > 0 else 0,
            'loss_rate': losses / total_games if total_games > 0 else 0,
            'push_rate': pushes / total_games if total_games > 0 else 0
        }