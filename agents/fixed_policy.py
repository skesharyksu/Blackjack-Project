class FixedPolicyAgent:
    """
    Fixed policy (basic strategy) agent for Blackjack.
    Uses a hardcoded table for actions based on player sum, dealer card, and usable ace.
    """
    def __init__(self):
        self.strategy = self._create_strategy_table()

    def _create_strategy_table(self):
        # (player_sum, dealer_card, usable_ace) -> action (0=stand, 1=hit, 2=double)
        table = {}
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                # Hard totals
                if player_sum <= 11:
                    table[(player_sum, dealer_card, 0)] = 1  # Always hit
                elif player_sum == 11 and dealer_card != 1:
                    table[(player_sum, dealer_card, 0)] = 2  # Double if possible
                elif player_sum >= 17:
                    table[(player_sum, dealer_card, 0)] = 0  # Stand
                elif 12 <= player_sum <= 16 and 2 <= dealer_card <= 6:
                    table[(player_sum, dealer_card, 0)] = 0  # Stand
                else:
                    table[(player_sum, dealer_card, 0)] = 1  # Hit
        # Soft totals (usable ace)
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                if player_sum <= 17:
                    table[(player_sum, dealer_card, 1)] = 1  # Hit
                elif player_sum >= 19:
                    table[(player_sum, dealer_card, 1)] = 0  # Stand
                elif player_sum == 18:
                    table[(player_sum, dealer_card, 1)] = 0 if dealer_card <= 8 else 1
        return table

    def select_action(self, state):
        """
        Args:
            state (np.ndarray): [player_sum, dealer_card, usable_ace]
        Returns:
            int: action (0=stand, 1=hit, 2=double)
        """
        player_sum, dealer_card, usable_ace = int(state[0]), int(state[1]), int(state[2])
        return self.strategy.get((player_sum, dealer_card, usable_ace), 1)  # Default to hit 