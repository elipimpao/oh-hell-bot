"""Bot creation, snapshot loading, and thread-safe observation building."""

import os
import numpy as np
import torch

from game import OhHellGame, Phase
from bots import RandomBot, HeuristicBot, SmartBot
from network import OhHellNetwork
from env import MAX_PLAYERS, MAX_BID

OBS_SIZE = 453
ACTION_SIZE = 52 + MAX_BID + 1  # 65


def create_bot(seat_type, snapshot_path=None, device=None):
    """Create a bot instance from a seat type string.

    Returns a bot with an .act(game, player) method.
    """
    if seat_type == "random":
        return RandomBot()
    elif seat_type == "heuristic":
        return HeuristicBot()
    elif seat_type == "smart":
        return SmartBot()
    elif seat_type == "nn":
        return load_nn_bot(snapshot_path, device or torch.device("cpu"))
    else:
        raise ValueError(f"Unknown seat type: {seat_type}")


def load_nn_bot(path, device):
    """Load a .pt snapshot and return an NNBot."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get("hidden_dim", 512)
    network = OhHellNetwork(OBS_SIZE, hidden_dim=hidden_dim).to(device)
    # Support both snapshot format ("state_dict") and checkpoint format ("network")
    if "state_dict" in checkpoint:
        network.load_state_dict(checkpoint["state_dict"])
    elif "network" in checkpoint:
        network.load_state_dict(checkpoint["network"])
    else:
        # Assume the file is a raw state_dict
        network.load_state_dict(checkpoint)
    network.eval()
    return NNBot(network, device)


class NNBot:
    """Bot wrapper using a neural network for decisions."""

    def __init__(self, network, device):
        self.network = network
        self.device = device

    def act(self, game, player):
        obs, mask = get_obs_and_mask(game, player)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.network(obs_t, mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        if game.phase == Phase.BIDDING:
            return action - 52
        else:
            return action

    def get_probabilities(self, game, player):
        """Get full probability distribution and state value."""
        obs, mask = get_obs_and_mask(game, player)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(obs_t, mask_t)
            masked = logits.masked_fill(mask_t == 0, float("-inf"))
            probs = torch.softmax(masked, dim=-1).squeeze(0).cpu().numpy()

        return probs, value.item()


def get_obs_and_mask(game, player):
    """Thread-safe observation and mask builder.

    Same logic as train.py's get_obs_and_mask_for_bot but allocates fresh
    arrays per call (fine for interactive use, avoids global buffer issues).
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    np_ = game.num_players

    # Hand (52) — offset 0
    for card in game.hands[player]:
        obs[card] = 1.0

    # Trump suit (4) — offset 52
    obs[52 + game.trump_suit] = 1.0

    # Phase (2) — offset 56
    obs[56 + int(game.phase)] = 1.0

    # Player count one-hot (4) — offset 58
    obs[58 + np_ - 2] = 1.0

    # Bids (5) — offset 62
    max_hs = game.max_hand_size
    inv_hs = 1.0 / max(max_hs, 1)
    obs[62:62 + MAX_PLAYERS] = -1.0
    for i in range(np_):
        p = (player + i) % np_
        b = game.bids[p]
        if b >= 0:
            obs[62 + i] = b * inv_hs

    # Tricks won (5) — offset 67
    inv_hand = 1.0 / max(game.hand_size, 1)
    for i in range(np_):
        p = (player + i) % np_
        obs[67 + i] = game.tricks_won[p] * inv_hand

    # Trick deficit (1) — offset 72
    my_bid = game.bids[player]
    obs[72] = (my_bid - game.tricks_won[player]) * inv_hs if my_bid >= 0 else 0.0

    # Current trick cards (52) — offset 73
    for _, card in game.current_trick:
        obs[73 + card] = 1.0

    # Lead suit (4) — offset 125
    if game.phase == Phase.PLAYING and game.current_trick:
        obs[125 + game.current_trick[0][1] // 13] = 1.0

    # Per-player cards played (5*52) — offset 129
    for i in range(np_):
        p = (player + i) % np_
        base = 129 + i * 52
        for card in game.cards_played_by[p]:
            obs[base + card] = 1.0

    # Cards unseen (52) — offset 389
    obs[389:441] = 1.0
    for card in game.hands[player]:
        obs[389 + card] = 0.0
    for card in game.cards_played_this_round:
        obs[389 + card] = 0.0
    obs[389 + game.trump_card] = 0.0

    # Trump broken (1) — offset 441
    if game.trump_broken:
        obs[441] = 1.0

    # Hand size normalized (1) — offset 442
    obs[442] = game.hand_size * inv_hs

    # Position relative to dealer (5) — offset 443
    obs[443 + (player - game.dealer) % np_] = 1.0

    # Position within trick (5) — offset 448
    if game.phase == Phase.PLAYING:
        trick_pos = len(game.current_trick)
        if trick_pos < MAX_PLAYERS:
            obs[448 + trick_pos] = 1.0

    # Action mask
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    if game.phase == Phase.BIDDING:
        for bid in game.get_legal_bids():
            mask[52 + bid] = 1.0
    else:
        for card in game.get_legal_plays():
            mask[card] = 1.0

    return obs, mask


def list_snapshots(project_root):
    """Scan snapshot directories and return list of available .pt files."""
    dirs = [
        "snapshots",
        os.path.join("league_snapshots", "main"),
    ]
    results = []
    for dir_path in dirs:
        full_path = os.path.join(project_root, dir_path)
        if not os.path.isdir(full_path):
            continue
        for fname in sorted(os.listdir(full_path)):
            if fname.endswith(".pt"):
                results.append({
                    "path": os.path.join(dir_path, fname),
                    "name": fname,
                    "directory": dir_path,
                })
    return results
