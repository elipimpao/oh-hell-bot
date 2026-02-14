# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython-accelerated Oh Hell card game engine.

Drop-in replacement for OhHellGame from game.py.
Cards are integers 0-51: suit = card // 13, rank = card % 13.
"""

import random


cdef class OhHellGame:
    """Cython-accelerated Oh Hell game state manager.

    Supports 2-5 players. Rounds go from max_hand_size down to 1 and back up.
    """

    cdef public int num_players
    cdef public int max_hand_size
    cdef public object rng
    cdef public list round_schedule
    cdef public int round_index
    cdef public list scores
    cdef public int dealer
    cdef public int hand_size
    cdef public list hands
    cdef public int trump_card
    cdef public int trump_suit
    cdef public int phase
    cdef public list bids
    cdef public list tricks_won
    cdef public int current_bidder_index
    cdef public list bid_order
    cdef public list current_trick
    cdef public object lead_suit
    cdef public bint trump_broken
    cdef public set cards_played_this_round
    cdef public list cards_played_by
    cdef public int trick_number
    cdef public int trick_leader

    def __init__(self, int num_players, rng=None):
        assert 2 <= num_players <= 5, "Oh Hell supports 2-5 players"
        self.num_players = num_players
        self.max_hand_size = min((52 - 1) // num_players, 10 if num_players == 5 else 12)
        self.rng = rng if rng is not None else random.Random()

        cdef list down = list(range(self.max_hand_size, 0, -1))
        cdef list up = list(range(2, self.max_hand_size + 1))
        self.round_schedule = down + up

        self.reset()

    def reset(self):
        """Start a new game."""
        self.scores = [0] * self.num_players
        self.dealer = self.rng.randint(0, self.num_players - 1)
        self.round_index = 0
        self._start_round()

    cdef _start_round(self):
        """Deal cards and set up a new round."""
        cdef int hand_size = <int>self.round_schedule[self.round_index]
        cdef int i, dealt_count
        self.hand_size = hand_size

        cdef list deck = list(range(52))
        self.rng.shuffle(deck)
        self.hands = [
            sorted(deck[i * hand_size:(i + 1) * hand_size])
            for i in range(self.num_players)
        ]
        dealt_count = self.num_players * hand_size
        self.trump_card = <int>deck[dealt_count]
        self.trump_suit = self.trump_card // 13

        self.phase = 0  # BIDDING
        self.bids = [-1] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_bidder_index = 0
        self.bid_order = [(self.dealer + 1 + i) % self.num_players
                          for i in range(self.num_players)]

        self.current_trick = []
        self.lead_suit = None
        self.trump_broken = False
        self.cards_played_this_round = set()
        self.cards_played_by = [set() for _ in range(self.num_players)]
        self.trick_number = 0
        self.trick_leader = <int>self.bid_order[0]

    @property
    def current_player(self):
        """Return which player should act next."""
        cdef int n_played
        if self.phase == 0:  # BIDDING
            return <int>self.bid_order[self.current_bidder_index]
        else:
            n_played = len(self.current_trick)
            return (self.trick_leader + n_played) % self.num_players

    def get_legal_bids(self, player=None):
        """Return list of legal bid values for the current bidder."""
        if player is not None:
            assert player == self.current_player
        cdef int p = self.current_player
        cdef int hs = self.hand_size
        cdef list all_bids = list(range(0, hs + 1))
        cdef int current_total, forbidden, b

        if self.current_bidder_index == self.num_players - 1:
            current_total = 0
            for b in self.bids:
                if b >= 0:
                    current_total += b
            forbidden = hs - current_total
            if 0 <= forbidden <= hs:
                all_bids = [b for b in all_bids if b != forbidden]

        return all_bids

    def place_bid(self, int bid):
        """Place a bid for the current bidder."""
        cdef int player = self.current_player
        self.bids[player] = bid
        self.current_bidder_index += 1
        if self.current_bidder_index >= self.num_players:
            self.phase = 1  # PLAYING

    def get_legal_plays(self, player=None):
        """Return list of legal cards for the current player."""
        if player is not None:
            assert player == self.current_player
        cdef int p = self.current_player
        cdef list hand = <list>self.hands[p]
        cdef int trump = self.trump_suit
        cdef int card, lead_s
        cdef Py_ssize_t i, n = len(hand)
        cdef list result

        if not self.current_trick:
            # Leading
            if self.trump_broken:
                return list(hand)
            result = []
            for i in range(n):
                card = <int>hand[i]
                if card // 13 != trump:
                    result.append(card)
            if result:
                return result
            return list(hand)
        else:
            # Following
            lead_s = <int>(<tuple>self.current_trick[0])[1] // 13
            result = []
            for i in range(n):
                card = <int>hand[i]
                if card // 13 == lead_s:
                    result.append(card)
            if result:
                return result
            return list(hand)

    def play_card(self, int card):
        """Play a card for the current player. Returns dict with game events."""
        cdef int player = self.current_player
        cdef int winner
        cdef list round_scores

        (<list>self.hands[player]).remove(card)
        self.current_trick.append((player, card))
        self.cards_played_this_round.add(card)
        (<set>self.cards_played_by[player]).add(card)

        # Track trump breaking
        if card // 13 == self.trump_suit and not self.trump_broken:
            if len(self.current_trick) > 1:
                self.trump_broken = True

        events = {"trick_complete": False, "round_complete": False, "game_over": False}

        if len(self.current_trick) == self.num_players:
            winner = self._resolve_trick()
            self.tricks_won[winner] = <int>self.tricks_won[winner] + 1
            self.trick_number += 1
            events["trick_complete"] = True
            events["trick_winner"] = winner

            if self.trick_number >= self.hand_size:
                round_scores = self._score_round()
                events["round_complete"] = True
                events["round_scores"] = round_scores
                events["round_bids"] = list(self.bids)
                events["round_tricks_won"] = list(self.tricks_won)
                events["round_hand_size"] = self.hand_size

                self.round_index += 1
                self.dealer = (self.dealer + 1) % self.num_players
                if self.round_index >= len(self.round_schedule):
                    events["game_over"] = True
                else:
                    self._start_round()
            else:
                self.trick_leader = winner
                self.current_trick = []
                self.lead_suit = None

        return events

    cdef int _resolve_trick(self):
        """Determine the winner of the current trick."""
        cdef int best_player, best_card, player, card
        cdef int best_suit, card_suit
        cdef Py_ssize_t i, n
        cdef tuple entry

        entry = <tuple>self.current_trick[0]
        best_player = <int>entry[0]
        best_card = <int>entry[1]

        n = len(self.current_trick)
        for i in range(1, n):
            entry = <tuple>self.current_trick[i]
            player = <int>entry[0]
            card = <int>entry[1]
            card_suit = card // 13
            best_suit = best_card // 13

            if card_suit == self.trump_suit and best_suit != self.trump_suit:
                best_player = player
                best_card = card
            elif card_suit == best_suit and (card % 13) > (best_card % 13):
                best_player = player
                best_card = card

        return best_player

    cdef list _score_round(self):
        """Score the round. Returns list of points earned this round."""
        cdef int p
        cdef list round_scores = [0] * self.num_players
        for p in range(self.num_players):
            if <int>self.tricks_won[p] == <int>self.bids[p]:
                round_scores[p] = 10 + <int>self.bids[p]
        for p in range(self.num_players):
            self.scores[p] = <int>self.scores[p] + <int>round_scores[p]
        return round_scores

    def get_state(self, int player):
        """Return observation dict from a player's perspective."""
        return {
            "hand": list(<list>self.hands[player]),
            "trump_suit": self.trump_suit,
            "trump_card": self.trump_card,
            "phase": self.phase,
            "bids": list(self.bids),
            "tricks_won": list(self.tricks_won),
            "scores": list(self.scores),
            "current_trick": list(self.current_trick),
            "cards_played_this_round": set(self.cards_played_this_round),
            "cards_played_by": [set(cp) for cp in self.cards_played_by],
            "trump_broken": self.trump_broken,
            "hand_size": self.hand_size,
            "dealer": self.dealer,
            "current_player": self.current_player,
            "trick_number": self.trick_number,
            "round_index": self.round_index,
            "num_rounds": len(self.round_schedule),
        }

    @property
    def is_game_over(self):
        return self.round_index >= len(self.round_schedule)
