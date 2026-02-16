"""AdvisorSession — Mode 2: Mirror a real game and get AI recommendations."""

import random
import uuid

import torch

from game import OhHellGame, Phase, card_str
from network import OhHellNetwork
from gui.bot_manager import get_obs_and_mask, OBS_SIZE


class AdvisorSession:
    """Tracks a real game's state from partial information and provides NN recommendations."""

    def __init__(self, config):
        self.id = str(uuid.uuid4())
        self.config = config
        self.num_players = config["num_players"]
        self.my_seat = 0  # Always seat 0
        self.device = torch.device("cpu")

        # Load network (optional — can load later)
        self.network = None
        snapshot_path = config.get("advisor_snapshot")
        if snapshot_path:
            try:
                self._load_network(snapshot_path)
            except Exception as e:
                print(f"Warning: Failed to load snapshot: {e}")
                self.network = None

        # Game state (manually tracked)
        self.hand = []
        self.trump_suit = -1
        self.trump_card = -1
        self.dealer = 0
        self.bids = [-1] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.scores = [0] * self.num_players
        self.current_trick = []  # [(player, card)]
        self.cards_seen = set()
        self.cards_played_by = [set() for _ in range(self.num_players)]
        self.trump_broken = False
        self.trick_leader = -1
        self.trick_number = 0
        self.round_index = 0
        self.hand_size = 0

    def _load_network(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        hidden_dim = checkpoint.get("hidden_dim", 512)
        network = OhHellNetwork(OBS_SIZE, hidden_dim=hidden_dim).to(self.device)
        if "state_dict" in checkpoint:
            network.load_state_dict(checkpoint["state_dict"])
        elif "network" in checkpoint:
            network.load_state_dict(checkpoint["network"])
        else:
            network.load_state_dict(checkpoint)
        network.eval()
        self.network = network

    def handle_message(self, msg):
        """Process a message from the frontend."""
        action = msg.get("action")

        if action == "get_state":
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "set_hand":
            self.hand = sorted(msg["cards"])
            self.hand_size = len(self.hand)
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "set_trump":
            self.trump_card = msg["card"]
            self.trump_suit = msg["card"] // 13
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "set_dealer":
            self.dealer = msg["player"]
            self.trick_leader = (self.dealer + 1) % self.num_players
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "record_bid":
            player = msg["player"]
            bid = msg["value"]
            self.bids[player] = bid
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "undo_bid":
            player = msg["player"]
            self.bids[player] = -1
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "record_play":
            player = msg["player"]
            card = msg["card"]
            self.current_trick.append((player, card))
            self.cards_seen.add(card)
            self.cards_played_by[player].add(card)
            if card in self.hand:
                self.hand.remove(card)
            # Auto-detect trump broken
            if card // 13 == self.trump_suit and not self.trump_broken:
                if len(self.current_trick) > 1:
                    self.trump_broken = True
            # Auto-complete trick
            result = {"type": "advisor_state", "state": self._get_state_for_frontend()}
            if len(self.current_trick) == self.num_players:
                winner = self._resolve_trick()
                self.tricks_won[winner] += 1
                self.trick_number += 1
                self.trick_leader = winner
                trick_cards = list(self.current_trick)
                self.current_trick = []
                result["trick_complete"] = True
                result["trick_winner"] = winner
                result["trick_cards"] = [[p, c] for p, c in trick_cards]
                result["state"] = self._get_state_for_frontend()
            return result

        elif action == "undo_play":
            player = msg["player"]
            card = msg["card"]
            # Remove from current trick
            self.current_trick = [(p, c) for p, c in self.current_trick
                                  if not (p == player and c == card)]
            self.cards_seen.discard(card)
            self.cards_played_by[player].discard(card)
            # Restore to hand if it was my card
            if player == self.my_seat and card not in self.hand:
                self.hand.append(card)
                self.hand.sort()
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "new_round":
            self._reset_round()
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "get_recommendation":
            phase = msg["phase"]
            return self._get_recommendation(phase)

        elif action == "load_snapshot":
            self._load_network(msg["path"])
            return {"type": "snapshot_loaded", "path": msg["path"]}

        elif action == "set_scores":
            self.scores = msg["scores"]
            return {"type": "advisor_state", "state": self._get_state_for_frontend()}

        elif action == "score_round":
            # Calculate and apply round scores
            round_scores = [0] * self.num_players
            for p in range(self.num_players):
                if self.bids[p] >= 0 and self.tricks_won[p] == self.bids[p]:
                    round_scores[p] = 10 + self.bids[p]
            for p in range(self.num_players):
                self.scores[p] += round_scores[p]
            self.round_index += 1
            result = {"type": "round_scored", "round_scores": round_scores,
                      "scores": list(self.scores), "state": self._get_state_for_frontend()}
            return result

        return {"type": "error", "message": f"Unknown action: {action}"}

    def _reset_round(self):
        """Reset for a new round."""
        self.hand = []
        self.bids = [-1] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        self.cards_seen = set()
        self.cards_played_by = [set() for _ in range(self.num_players)]
        self.trump_broken = False
        self.trump_card = -1
        self.trump_suit = -1
        self.trick_leader = -1
        self.trick_number = 0
        self.hand_size = 0

    def _resolve_trick(self):
        """Determine the winner of the current trick."""
        best_player = self.current_trick[0][0]
        best_card = self.current_trick[0][1]
        for player, card in self.current_trick[1:]:
            if card // 13 == self.trump_suit and best_card // 13 != self.trump_suit:
                best_player, best_card = player, card
            elif card // 13 == best_card // 13 and card % 13 > best_card % 13:
                best_player, best_card = player, card
        return best_player

    def _get_recommendation(self, phase):
        """Build synthetic game state and run NN inference."""
        if self.network is None:
            return {"type": "error", "message": "No model loaded."}

        if self.trump_suit < 0:
            return {"type": "error", "message": "Trump card not set yet."}

        try:
            return self._build_and_infer(phase)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"type": "error", "message": f"Recommendation failed: {type(e).__name__}: {e}"}

    def _build_and_infer(self, phase):
        """Build synthetic game state and run NN inference (inner)."""
        # Construct synthetic OhHellGame
        game = OhHellGame(self.num_players, rng=random.Random(0))
        game.dealer = self.dealer
        game.hand_size = self.hand_size
        game.trump_suit = self.trump_suit
        game.trump_card = self.trump_card if self.trump_card >= 0 else self.trump_suit * 13
        game.trump_broken = self.trump_broken
        game.hands = [[] for _ in range(self.num_players)]
        game.hands[self.my_seat] = sorted(self.hand)
        # For bid recommendations, remove user's own bid if already recorded
        # (can happen during rejoin when events arrive out of order)
        bids = list(self.bids)
        if phase == "bid" and bids[self.my_seat] >= 0:
            bids[self.my_seat] = -1
        game.bids = bids
        game.tricks_won = list(self.tricks_won)
        game.cards_played_this_round = set(self.cards_seen)
        game.cards_played_by = [set(s) for s in self.cards_played_by]
        # For play recommendations, remove user's card from current_trick if already
        # recorded (can happen during rejoin when card_played fires before player_turn)
        current_trick = list(self.current_trick)
        if phase == "play":
            user_play = [(p, c) for p, c in current_trick if p == self.my_seat]
            if user_play:
                card = user_play[0][1]
                current_trick = [(p, c) for p, c in current_trick if p != self.my_seat]
                game.cards_played_this_round.discard(card)
                game.cards_played_by[self.my_seat].discard(card)
                if card not in game.hands[self.my_seat]:
                    game.hands[self.my_seat].append(card)
                    game.hands[self.my_seat].sort()
        game.current_trick = current_trick
        game.phase = Phase.BIDDING if phase == "bid" else Phase.PLAYING
        game.trick_leader = self.trick_leader if self.trick_leader >= 0 else (self.dealer + 1) % self.num_players
        game.bid_order = [(self.dealer + 1 + i) % self.num_players for i in range(self.num_players)]
        bids_placed = sum(1 for b in game.bids if b >= 0)
        game.current_bidder_index = bids_placed
        game.trick_number = self.trick_number
        game.round_index = 0

        # Get observation and mask
        obs, mask = get_obs_and_mask(game, self.my_seat)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(obs_t, mask_t)
            masked = logits.masked_fill(mask_t == 0, float("-inf"))
            probs = torch.softmax(masked, dim=-1).squeeze(0).cpu().numpy()

        if phase == "bid":
            legal = game.get_legal_bids(self.my_seat)
            recs = [{"bid": int(b), "prob": float(probs[52 + b])} for b in legal]
            recs.sort(key=lambda x: -x["prob"])
            return {"type": "bid_recommendation", "recommendations": recs,
                    "value": float(value.item())}
        else:
            legal = game.get_legal_plays(self.my_seat)
            recs = [{"card": int(c), "prob": float(probs[c])} for c in legal]
            recs.sort(key=lambda x: -x["prob"])
            return {"type": "play_recommendation", "recommendations": recs,
                    "value": float(value.item())}

    def _get_state_for_frontend(self):
        """Build state dict for the frontend."""
        # Determine current phase
        all_bids_placed = all(b >= 0 for b in self.bids)
        phase = "playing" if all_bids_placed else "bidding"

        return {
            "num_players": self.num_players,
            "my_seat": self.my_seat,
            "hand": list(self.hand),
            "hand_size": self.hand_size,
            "trump_card": self.trump_card,
            "trump_suit": self.trump_suit,
            "dealer": self.dealer,
            "bids": list(self.bids),
            "tricks_won": list(self.tricks_won),
            "scores": list(self.scores),
            "current_trick": [[p, c] for p, c in self.current_trick],
            "cards_seen": list(self.cards_seen),
            "trump_broken": self.trump_broken,
            "trick_leader": self.trick_leader,
            "trick_number": self.trick_number,
            "round_index": self.round_index,
            "phase": phase,
        }
