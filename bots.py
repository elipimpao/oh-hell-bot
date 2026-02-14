"""Rule-based bots for Oh Hell.

Cards are integers 0-51: suit = card // 13, rank = card % 13.
"""

import random
from game import Phase


class RandomBot:
    """Plays uniformly at random from legal actions."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def bid(self, game, player):
        legal = game.get_legal_bids(player)
        return self.rng.choice(legal)

    def play(self, game, player):
        legal = game.get_legal_plays(player)
        return self.rng.choice(legal)

    def act(self, game, player):
        if game.phase == Phase.BIDDING:
            return self.bid(game, player)
        else:
            return self.play(game, player)


class HeuristicBot:
    """Simple heuristic bot.

    Bidding: counts high cards as likely tricks.
    Playing: plays high when needing tricks, ducks when satisfied.
    """

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def bid(self, game, player):
        hand = game.hands[player]
        trump = game.trump_suit
        estimated_tricks = 0

        for card in hand:
            suit = card // 13
            rank = card % 13
            if suit == trump:
                # Trump: A=1.0, K=0.8, Q=0.5
                if rank == 12:
                    estimated_tricks += 1.0
                elif rank == 11:
                    estimated_tricks += 0.8
                elif rank == 10:
                    estimated_tricks += 0.5
            else:
                # Off-suit: only A is likely a trick
                if rank == 12:
                    estimated_tricks += 0.7
                elif rank == 11:
                    estimated_tricks += 0.3

        bid = round(estimated_tricks)
        bid = max(0, min(bid, game.hand_size))

        legal = game.get_legal_bids(player)
        if bid not in legal:
            # Pick the closest legal bid
            bid = min(legal, key=lambda b: abs(b - estimated_tricks))
        return bid

    def play(self, game, player):
        hand = game.hands[player]
        legal = game.get_legal_plays(player)
        trump = game.trump_suit
        tricks_needed = game.bids[player] - game.tricks_won[player]

        if len(legal) == 1:
            return legal[0]

        if not game.current_trick:
            # Leading
            return self._choose_lead(legal, trump, tricks_needed, game)
        else:
            return self._choose_follow(legal, game, player, tricks_needed)

    def _choose_lead(self, legal, trump, tricks_needed, game):
        non_trump = [c for c in legal if c // 13 != trump]
        trump_cards = [c for c in legal if c // 13 == trump]

        if tricks_needed > 0:
            # Need tricks — lead high
            if trump_cards and game.trump_broken:
                return max(trump_cards, key=lambda c: c % 13)
            if non_trump:
                return max(non_trump, key=lambda c: c % 13)
            return max(legal, key=lambda c: c % 13)
        else:
            # Have enough tricks — lead low to duck
            if non_trump:
                return min(non_trump, key=lambda c: c % 13)
            return min(legal, key=lambda c: c % 13)

    def _choose_follow(self, legal, game, player, tricks_needed):
        trick = game.current_trick
        lead_suit = trick[0][1] // 13
        trump = game.trump_suit

        # Determine current best card in trick
        best_card = trick[0][1]
        for _, card in trick[1:]:
            if card // 13 == trump and best_card // 13 != trump:
                best_card = card
            elif card // 13 == best_card // 13 and card % 13 > best_card % 13:
                best_card = card

        # Cards that can win
        winners = []
        for c in legal:
            if c // 13 == trump and best_card // 13 != trump:
                winners.append(c)
            elif c // 13 == best_card // 13 and c % 13 > best_card % 13:
                winners.append(c)

        if tricks_needed > 0:
            # Want to win
            if winners:
                return min(winners, key=lambda c: c % 13)  # cheapest winner
            return min(legal, key=lambda c: c % 13)  # can't win, dump low
        else:
            # Don't want to win — play low, avoid winning
            losers = [c for c in legal if c not in winners]
            if losers:
                return min(losers, key=lambda c: c % 13)
            # Forced to win — play the lowest winner
            return min(legal, key=lambda c: c % 13)

    def act(self, game, player):
        if game.phase == Phase.BIDDING:
            return self.bid(game, player)
        else:
            return self.play(game, player)


class SmartBot:
    """Card-counting heuristic bot.

    Bidding: evaluates hand strength with suit length, protection,
    void suits, and position awareness (dealer sees other bids).
    Playing: tracks cards played to identify guaranteed winners,
    detects opponent voids, and uses finesse ducking.
    """

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self._known_voids = set()   # (player, suit) pairs
        self._last_round_key = None

    def act(self, game, player):
        # Reset tracking on new round
        round_key = (id(game), game.round_index)
        if round_key != self._last_round_key:
            self._known_voids = set()
            self._last_round_key = round_key

        # Learn voids from current trick
        if game.phase == Phase.PLAYING and len(game.current_trick) > 1:
            lead_suit = game.current_trick[0][1] // 13
            for p, card in game.current_trick[1:]:
                if card // 13 != lead_suit:
                    self._known_voids.add((p, lead_suit))

        if game.phase == Phase.BIDDING:
            return self.bid(game, player)
        else:
            return self.play(game, player)

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def bid(self, game, player):
        hand = game.hands[player]
        trump = game.trump_suit
        np_ = game.num_players

        by_suit = {s: [] for s in range(4)}
        for card in hand:
            by_suit[card // 13].append(card)
        for s in by_suit:
            by_suit[s].sort(key=lambda c: c % 13, reverse=True)

        est = 0.0
        tc = by_suit[trump]

        # Trump cards
        for card in tc:
            rank = card % 13
            if rank == 12:    est += 1.0
            elif rank == 11:  est += 0.85
            elif rank == 10:  est += 0.55
            elif rank >= 8:   est += 0.25
        # Long trump bonus (extra trump beyond 2 likely win by exhaustion)
        if len(tc) >= 3:
            est += (len(tc) - 2) * 0.25

        # Off-suit cards
        for s in range(4):
            if s == trump:
                continue
            sc = by_suit[s]
            if not sc:
                # Void suit — can trump in
                if tc:
                    est += 0.25
                continue
            for card in sc:
                rank = card % 13
                if rank == 12:
                    est += 0.85
                elif rank == 11:
                    est += 0.55 if len(sc) >= 2 else 0.3
                elif rank == 10:
                    est += 0.3 if len(sc) >= 3 else 0.1
            # Long suit bonus (5+ cards — later cards win by exhaustion)
            if len(sc) >= 5:
                est += (len(sc) - 4) * 0.15

        # Adjust for player count
        adj = {2: 1.1, 3: 1.0, 4: 0.92, 5: 0.85}
        est *= adj.get(np_, 1.0)

        # Dealer position: can see all other bids
        if game.current_bidder_index == np_ - 1:
            total = sum(b for b in game.bids if b >= 0)
            if total < game.hand_size * 0.6:
                est *= 1.1
            elif total > game.hand_size:
                est *= 0.85

        bid = round(est)
        bid = max(0, min(bid, game.hand_size))
        legal = game.get_legal_bids(player)
        if bid not in legal:
            bid = min(legal, key=lambda b: abs(b - est))
        return bid

    # ------------------------------------------------------------------
    # Card counting helpers
    # ------------------------------------------------------------------

    def _is_top_remaining(self, card, game, player):
        """True if card is the highest unseen in its suit."""
        played = game.cards_played_this_round
        in_trick = {c for _, c in game.current_trick}
        hand = set(game.hands[player])
        suit = card // 13
        for r in range(card % 13 + 1, 13):
            c = suit * 13 + r
            if (c not in played and c not in in_trick
                    and c not in hand and c != game.trump_card):
                return False
        return True

    def _remaining_in_suit(self, suit, game, player):
        """Count unseen cards in a suit (in opponents' hands or deck)."""
        played = game.cards_played_this_round
        in_trick = {c for _, c in game.current_trick}
        hand = set(game.hands[player])
        count = 0
        for r in range(13):
            c = suit * 13 + r
            if (c not in played and c not in in_trick
                    and c not in hand and c != game.trump_card):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Playing
    # ------------------------------------------------------------------

    def play(self, game, player):
        legal = game.get_legal_plays(player)
        if len(legal) == 1:
            return legal[0]

        trump = game.trump_suit
        tricks_needed = game.bids[player] - game.tricks_won[player]

        if not game.current_trick:
            return self._choose_lead(legal, game, player, trump, tricks_needed)
        else:
            return self._choose_follow(legal, game, player, trump, tricks_needed)

    def _choose_lead(self, legal, game, player, trump, tricks_needed):
        non_trump = [c for c in legal if c // 13 != trump]
        trump_cards = [c for c in legal if c // 13 == trump]

        if tricks_needed > 0:
            # Guaranteed non-trump winners (top remaining in their suit)
            safe_nt = [c for c in non_trump
                       if self._is_top_remaining(c, game, player)]

            # Filter out suits where an opponent is known void (could trump)
            really_safe = [c for c in safe_nt
                           if not any((p, c // 13) in self._known_voids
                                      for p in range(game.num_players)
                                      if p != player)]
            pool = really_safe or safe_nt
            if pool:
                # Prefer suits with more remaining cards (opponents likely
                # still have that suit and can't trump)
                return max(pool,
                           key=lambda c: self._remaining_in_suit(
                               c // 13, game, player))

            # Guaranteed trump winners
            if trump_cards and game.trump_broken:
                safe_tr = [c for c in trump_cards
                           if self._is_top_remaining(c, game, player)]
                if safe_tr:
                    return min(safe_tr, key=lambda c: c % 13)

            # No guaranteed winners — lead high, avoiding known-void suits
            safe = [c for c in non_trump
                    if not any((p, c // 13) in self._known_voids
                               for p in range(game.num_players)
                               if p != player)]
            if safe:
                return max(safe, key=lambda c: c % 13)
            if non_trump:
                return max(non_trump, key=lambda c: c % 13)
            return max(legal, key=lambda c: c % 13)
        else:
            # Don't need tricks — lead low to duck
            if non_trump:
                return min(non_trump, key=lambda c: c % 13)
            return min(legal, key=lambda c: c % 13)

    def _choose_follow(self, legal, game, player, trump, tricks_needed):
        trick = game.current_trick
        lead_suit = trick[0][1] // 13

        # Current best card in trick
        best_card = trick[0][1]
        for _, card in trick[1:]:
            if card // 13 == trump and best_card // 13 != trump:
                best_card = card
            elif card // 13 == best_card // 13 and card % 13 > best_card % 13:
                best_card = card

        # Separate winners and losers
        winners = []
        for c in legal:
            if c // 13 == trump and best_card // 13 != trump:
                winners.append(c)
            elif c // 13 == best_card // 13 and c % 13 > best_card % 13:
                winners.append(c)
        losers = [c for c in legal if c not in winners]

        if tricks_needed > 0:
            if winners:
                return min(winners, key=lambda c: c % 13)  # cheapest winner
            return min(legal, key=lambda c: c % 13)  # can't win, dump low
        else:
            # Don't want tricks — finesse duck: dump highest loser to shed
            # dangerous middle cards that could accidentally win later
            if losers:
                return max(losers, key=lambda c: c % 13)
            # Forced to win — play lowest winner
            return min(legal, key=lambda c: c % 13)
