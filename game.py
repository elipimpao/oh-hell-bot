"""Oh Hell card game engine. Pure game logic, no ML dependencies.

Cards are represented as integers 0-51:
    suit = card // 13  (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)
    rank = card % 13   (0=2, 1=3, ..., 12=Ace)
"""

from enum import IntEnum
from collections import namedtuple
import random

# Card namedtuple kept for GUI/display backward compatibility
Card = namedtuple("Card", ["suit", "rank"])

class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

SUIT_NAMES = ["♣", "♦", "♥", "♠"]
RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

def card_str(card):
    """Format card for display. Accepts int (0-51) or Card namedtuple."""
    if isinstance(card, int):
        return f"{RANK_NAMES[card % 13]}{SUIT_NAMES[card // 13]}"
    return f"{RANK_NAMES[card.rank]}{SUIT_NAMES[card.suit]}"

def card_index(card):
    """Convert card to integer index 0-51. Identity for int cards."""
    if isinstance(card, int):
        return card
    return card.suit * 13 + card.rank

def index_to_card(idx):
    """Convert integer index 0-51 to card int."""
    return idx

def make_card(suit, rank):
    """Create a card int from suit (0-3) and rank (0-12)."""
    return suit * 13 + rank

FULL_DECK = list(range(52))

# Precomputed card -> index lookup (identity for int cards)
CARD_IDX = {i: i for i in range(52)}

class Phase(IntEnum):
    BIDDING = 0
    PLAYING = 1

class OhHellGame:
    """Full Oh Hell game state manager.

    Supports 2-5 players. Rounds go from max_hand_size down to 1 and back up.
    Max hand size = (52 - 1) // num_players.
    """

    def __init__(self, num_players, rng=None):
        assert 2 <= num_players <= 5, "Oh Hell supports 2-5 players"
        self.num_players = num_players
        self.max_hand_size = min((52 - 1) // num_players, 10 if num_players == 5 else 12)
        self.rng = rng or random.Random()

        # Build the round schedule: max, max-1, ..., 2, 1, 2, ..., max-1, max
        down = list(range(self.max_hand_size, 0, -1))
        up = list(range(2, self.max_hand_size + 1))
        self.round_schedule = down + up

        self.reset()

    def reset(self):
        """Start a new game."""
        self.scores = [0] * self.num_players
        self.dealer = self.rng.randint(0, self.num_players - 1)
        self.round_index = 0
        self._start_round()

    def _start_round(self):
        """Deal cards and set up a new round."""
        hand_size = self.round_schedule[self.round_index]
        self.hand_size = hand_size

        # Shuffle and deal
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.hands = [
            sorted(deck[i * hand_size:(i + 1) * hand_size])
            for i in range(self.num_players)
        ]
        dealt_count = self.num_players * hand_size
        self.trump_card = deck[dealt_count]  # flip one card for trump
        self.trump_suit = self.trump_card // 13

        # Round state
        self.phase = Phase.BIDDING
        self.bids = [-1] * self.num_players  # -1 = not yet bid
        self.tricks_won = [0] * self.num_players
        self.current_bidder_index = 0  # index into bid order
        self.bid_order = [(self.dealer + 1 + i) % self.num_players for i in range(self.num_players)]

        # Trick state
        self.current_trick = []  # list of (player, card_int)
        self.lead_suit = None
        self.trump_broken = False
        self.cards_played_this_round = set()
        self.cards_played_by = [set() for _ in range(self.num_players)]
        self.trick_number = 0
        self.trick_leader = self.bid_order[0]  # left of dealer leads first trick

    @property
    def current_player(self):
        """Return which player should act next."""
        if self.phase == Phase.BIDDING:
            return self.bid_order[self.current_bidder_index]
        else:
            # During playing, current player is determined by trick position
            n_played = len(self.current_trick)
            return (self.trick_leader + n_played) % self.num_players

    def get_legal_bids(self, player=None):
        """Return list of legal bid values for the current bidder.

        The dealer (last bidder) cannot bid a value that makes total bids == hand_size.
        """
        if player is not None:
            assert player == self.current_player, f"Not player {player}'s turn to bid"
        player = self.current_player
        all_bids = list(range(0, self.hand_size + 1))

        # Dealer restriction: last bidder can't make total == hand_size
        is_last_bidder = self.current_bidder_index == self.num_players - 1
        if is_last_bidder:
            current_total = sum(b for b in self.bids if b >= 0)
            forbidden = self.hand_size - current_total
            if 0 <= forbidden <= self.hand_size:
                all_bids = [b for b in all_bids if b != forbidden]

        return all_bids

    def place_bid(self, bid):
        """Place a bid for the current bidder."""
        player = self.current_player
        legal = self.get_legal_bids()
        assert bid in legal, f"Illegal bid {bid} for player {player}. Legal: {legal}"

        self.bids[player] = bid
        self.current_bidder_index += 1

        if self.current_bidder_index >= self.num_players:
            self.phase = Phase.PLAYING

    def get_legal_plays(self, player=None):
        """Return list of legal cards for the current player to play.

        Rules:
        - Must follow lead suit if possible
        - Cannot lead trump unless trump is broken or hand is all trump
        """
        if player is not None:
            assert player == self.current_player, f"Not player {player}'s turn to play"
        player = self.current_player
        hand = self.hands[player]
        trump = self.trump_suit

        if not self.current_trick:
            # Leading the trick
            if self.trump_broken:
                return list(hand)
            else:
                non_trump = [c for c in hand if c // 13 != trump]
                if non_trump:
                    return non_trump
                else:
                    # Hand is all trump — must lead trump
                    return list(hand)
        else:
            # Following — must follow lead suit if possible
            lead_suit = self.current_trick[0][1] // 13
            follow = [c for c in hand if c // 13 == lead_suit]
            if follow:
                return follow
            else:
                return list(hand)

    def play_card(self, card):
        """Play a card for the current player. Returns dict with game events."""
        player = self.current_player
        legal = self.get_legal_plays()
        assert card in legal, f"Illegal play {card_str(card)} for player {player}. Legal: {[card_str(c) for c in legal]}"

        self.hands[player].remove(card)
        self.current_trick.append((player, card))
        self.cards_played_this_round.add(card)
        self.cards_played_by[player].add(card)

        # Track trump breaking
        if card // 13 == self.trump_suit and not self.trump_broken:
            if len(self.current_trick) > 1:  # not leading, played trump
                self.trump_broken = True

        events = {"trick_complete": False, "round_complete": False, "game_over": False}

        # Check if trick is complete
        if len(self.current_trick) == self.num_players:
            winner = self._resolve_trick()
            self.tricks_won[winner] += 1
            self.trick_number += 1
            events["trick_complete"] = True
            events["trick_winner"] = winner

            # Check if round is complete
            if self.trick_number >= self.hand_size:
                round_scores = self._score_round()
                events["round_complete"] = True
                events["round_scores"] = round_scores
                # Capture state before _start_round() resets it
                events["round_bids"] = list(self.bids)
                events["round_tricks_won"] = list(self.tricks_won)
                events["round_hand_size"] = self.hand_size

                # Advance to next round
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

    def _resolve_trick(self):
        """Determine the winner of the current trick."""
        best_player = self.current_trick[0][0]
        best_card = self.current_trick[0][1]

        for player, card in self.current_trick[1:]:
            if card // 13 == self.trump_suit and best_card // 13 != self.trump_suit:
                # Trump beats non-trump
                best_player, best_card = player, card
            elif card // 13 == best_card // 13 and card % 13 > best_card % 13:
                # Same suit, higher rank wins
                best_player, best_card = player, card

        return best_player

    def _score_round(self):
        """Score the round. Returns list of points earned this round."""
        round_scores = [0] * self.num_players
        for p in range(self.num_players):
            if self.tricks_won[p] == self.bids[p]:
                round_scores[p] = 10 + self.bids[p]
        for p in range(self.num_players):
            self.scores[p] += round_scores[p]
        return round_scores

    def get_state(self, player):
        """Return observation dict from a player's perspective."""
        return {
            "hand": list(self.hands[player]),
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


# Try to use Cython-accelerated engine if available
try:
    from game_fast import OhHellGame as OhHellGame  # noqa: F811
except ImportError:
    pass
