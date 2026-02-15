"""GameSession — Mode 1: Full game simulation with configurable seats."""

import random
import uuid

from game import OhHellGame, Phase, card_str
from gui.bot_manager import create_bot


class GameSession:
    """Wraps OhHellGame with seat management and bot auto-play."""

    def __init__(self, config):
        self.id = str(uuid.uuid4())
        self.config = config
        self.num_players = config["num_players"]
        self.dev_mode = config.get("dev_mode", False)
        self.auto_play_speed = config.get("auto_play_speed", 1.0)

        # Create game engine
        self.game = OhHellGame(self.num_players, rng=random.Random())

        # Track seats
        self.human_seats = set()
        self.bots = {}
        self.seat_labels = []

        for i, seat in enumerate(config["seats"]):
            seat_type = seat["type"]
            if seat_type == "human":
                self.human_seats.add(i)
                self.seat_labels.append("You" if i == config.get("human_seat", 0) else f"Human {i}")
            else:
                self.bots[i] = create_bot(seat_type, seat.get("snapshot_path"))
                if seat_type == "nn":
                    snap_name = (seat.get("snapshot_path") or "NN").replace("\\", "/").split("/")[-1]
                    self.seat_labels.append(snap_name)
                else:
                    label_map = {"random": "RandomBot", "heuristic": "HeuristicBot", "smart": "SmartBot"}
                    self.seat_labels.append(label_map.get(seat_type, seat_type))

        # Apply custom settings
        custom_hand_size = config.get("custom_hand_size")
        max_cards = config.get("max_cards")
        custom_hands = config.get("custom_hands")
        custom_trump_card = config.get("custom_trump_card")

        if max_cards:
            # Build custom round schedule: max_cards down to 1 and back up
            max_allowed = min((52 - 1) // num_players, 10 if num_players == 5 else 12)
            mc = min(max_cards, max_allowed)
            down = list(range(mc, 0, -1))
            up = list(range(2, mc + 1))
            self.game.round_schedule = down + up
            self.game.max_hand_size = mc
            self.game.round_index = 0
            self.game._start_round()

        if custom_hand_size:
            self.game.round_schedule = [custom_hand_size]
            self.game.round_index = 0
            self.game._start_round()

        if custom_hands:
            for seat_str, cards in custom_hands.items():
                seat_idx = int(seat_str)
                self.game.hands[seat_idx] = sorted(cards)
            if custom_hands:
                self.game.hand_size = max(len(h) for h in self.game.hands)

        if custom_trump_card is not None:
            self.game.trump_card = custom_trump_card
            self.game.trump_suit = custom_trump_card // 13

        # Event history for the current round
        self.trick_history = []
        self._pending_trick = []

    def handle_message(self, msg):
        """Process a frontend message and return response."""
        action = msg.get("action")

        if action == "get_state":
            return {"type": "state_update", "events": [], "state": self.get_full_state()}

        elif action == "bid":
            return self._handle_bid(msg["value"])

        elif action == "play_card":
            return self._handle_play(msg["card"])

        elif action == "advance":
            return self._advance_one()

        elif action == "auto_play":
            return self._run_until_human()

        return {"type": "error", "message": f"Unknown action: {action}"}

    def _handle_bid(self, bid_value):
        """Human places a bid, then auto-advance through bot turns."""
        player = self.game.current_player
        self.game.place_bid(bid_value)
        events = [{"type": "bid", "player": player, "value": bid_value, "state": self.get_full_state()}]
        more_events = self._auto_advance_bots()
        events.extend(more_events)
        return {"type": "state_update", "events": events, "state": self.get_full_state()}

    def _handle_play(self, card_int):
        """Human plays a card, then auto-advance through bot turns."""
        player = self.game.current_player
        pre_trick = list(self.game.current_trick)
        game_events = self.game.play_card(card_int)
        self._process_game_events(game_events)
        event = {"type": "play", "player": player, "card": card_int, "game_events": game_events, "state": self.get_full_state()}
        if game_events.get("trick_complete"):
            event["trick_cards"] = [[p, c] for p, c in pre_trick] + [[player, card_int]]
        events = [event]
        if not game_events.get("round_complete") and not game_events.get("game_over"):
            more_events = self._auto_advance_bots()
            events.extend(more_events)
        return {"type": "state_update", "events": events, "state": self.get_full_state()}

    def _advance_one(self):
        """Advance one bot action (for spectator mode)."""
        if self.game.is_game_over:
            return {"type": "state_update", "events": [], "state": self.get_full_state()}

        cp = self.game.current_player
        if cp in self.human_seats:
            return {"type": "state_update", "events": [], "state": self.get_full_state()}

        events = self._execute_bot_action(cp)
        return {"type": "state_update", "events": events, "state": self.get_full_state()}

    def _run_until_human(self):
        """Run bot actions until next human turn or game over."""
        events = self._auto_advance_bots()
        return {"type": "state_update", "events": events, "state": self.get_full_state()}

    def _auto_advance_bots(self):
        """Execute bot actions until a human needs to act or game over."""
        events = []
        while not self.game.is_game_over:
            cp = self.game.current_player
            if cp in self.human_seats:
                break
            bot_events = self._execute_bot_action(cp)
            events.extend(bot_events)
            if bot_events and bot_events[-1].get("game_events", {}).get("round_complete"):
                break
        return events

    def _execute_bot_action(self, player):
        """Execute one bot action and return events with state snapshots."""
        bot = self.bots[player]
        events = []

        if self.game.phase == Phase.BIDDING:
            bid = bot.act(self.game, player)
            self.game.place_bid(bid)
            events.append({"type": "bid", "player": player, "value": bid, "state": self.get_full_state()})
        else:
            card = bot.act(self.game, player)
            pre_trick = list(self.game.current_trick)
            game_events = self.game.play_card(card)
            self._process_game_events(game_events)
            event = {"type": "play", "player": player, "card": card, "game_events": game_events, "state": self.get_full_state()}
            if game_events.get("trick_complete"):
                event["trick_cards"] = [[p, c] for p, c in pre_trick] + [[player, card]]
            events.append(event)

        return events

    def _process_game_events(self, game_events):
        """Track trick history from game events."""
        if game_events.get("trick_complete"):
            self._pending_trick = []
        if game_events.get("round_complete"):
            self.trick_history = []

    def get_full_state(self):
        """Build complete state dict for the frontend."""
        game = self.game

        state = {
            "phase": int(game.phase) if not game.is_game_over else -1,
            "hand_size": game.hand_size if not game.is_game_over else 0,
            "trump_card": game.trump_card if not game.is_game_over else -1,
            "trump_suit": game.trump_suit if not game.is_game_over else -1,
            "dealer": game.dealer,
            "current_player": game.current_player if not game.is_game_over else -1,
            "bids": list(game.bids) if not game.is_game_over else [],
            "tricks_won": list(game.tricks_won) if not game.is_game_over else [],
            "scores": list(game.scores),
            "current_trick": [[p, c] for p, c in game.current_trick] if not game.is_game_over else [],
            "round_index": game.round_index,
            "num_rounds": len(game.round_schedule),
            "game_over": game.is_game_over,
            "num_players": game.num_players,
            "seat_labels": self.seat_labels,
            "human_seats": list(self.human_seats),
            "dev_mode": self.dev_mode,
        }

        state["hands"] = {}
        state["hand_counts"] = {}
        if not game.is_game_over:
            for seat in range(game.num_players):
                if seat in self.human_seats or self.dev_mode:
                    state["hands"][str(seat)] = list(game.hands[seat])
                state["hand_counts"][str(seat)] = len(game.hands[seat])

        if not game.is_game_over and game.current_player in self.human_seats:
            if game.phase == Phase.BIDDING:
                state["legal_bids"] = game.get_legal_bids()
            else:
                state["legal_plays"] = game.get_legal_plays()

        if not game.is_game_over:
            state["round_hand_size"] = game.round_schedule[game.round_index]
        state["round_schedule"] = game.round_schedule

        return state
