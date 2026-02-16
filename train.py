"""PPO training loop for Oh Hell bot. CleanRL-style single-file implementation.

Trains across all player counts (2-5) simultaneously for a general-purpose agent.
Uses multiprocessing with local rollout collection — each worker runs a CPU copy of
the agent network to collect complete rollouts independently, eliminating per-step IPC.
After a warmup phase against heuristic opponents, switches to mixed self-play.
"""

import os
import sys
import csv
import json
import time
import signal
import random
import argparse
import multiprocessing as mp
from collections import deque

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from game import OhHellGame, Phase, card_index, index_to_card
from env import OhHellEnv, MAX_PLAYERS, MAX_BID
from network import OhHellNetwork
from bots import RandomBot, HeuristicBot, SmartBot


PLAYER_COUNTS = [2, 3, 4, 5]
PLAYER_COUNT_WEIGHTS = [0.15, 0.20, 0.30, 0.35]  # favor 4p/5p games

_B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _to_base36(n):
    """Encode a non-negative integer as an uppercase base-36 string."""
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(_B36[n % 36])
        n //= 36
    return "".join(reversed(digits))


def _step_tag(step):
    """Format a step count as a compact string like '127M' or '4200k'."""
    if step >= 1_000_000:
        return f"{step // 1_000_000}M"
    elif step >= 1_000:
        return f"{step // 1_000}k"
    return str(step)


# ---- Output helpers ----


def dashboard_print(text):
    """Print dashboard block (plain scrolling output)."""
    print(text, flush=True)


def event_print(text):
    """Print a non-dashboard message (snapshot, eval, checkpoint)."""
    print(text, flush=True)


def format_dashboard(global_step, sp_active, opp_str, avg_reward,
                     pg_loss, v_loss, entropy, sps,
                     opponent_pool, cell_win_rates, cell_win_tracker,
                     pc_win_tracker, pc_win_rates, pc_base_weights,
                     worker_params, min_pc_samples, writer,
                     exploiter_mode=False):
    """Build the full dashboard string and log TensorBoard scalars."""
    mode = "SP" if sp_active else "H"
    lines = []

    # Line 1: step info
    lines.append(
        f"Step {global_step:>9,d} [{mode}:{opp_str}] | "
        f"rew {avg_reward:.3f} | pg {pg_loss:.4f} | vf {v_loss:.4f} | "
        f"ent {entropy:.4f} | SPS {sps}")

    if exploiter_mode:
        # Exploiter: compact display — just per-player-count win rates vs the snapshot
        pc_parts = []
        valid_rates = []
        for pc in PLAYER_COUNTS:
            if len(pc_win_tracker[pc]) >= min_pc_samples:
                rate = pc_win_rates.get(pc, sum(pc_win_tracker[pc]) / len(pc_win_tracker[pc]))
                pc_parts.append(f"{pc}p:{rate*100:.0f}%")
                valid_rates.append(rate)
                writer.add_scalar(f"train/pc_{pc}p_win_rate", rate, global_step)
            else:
                pc_parts.append(f"{pc}p:--")
        avg_str = f"{sum(valid_rates)/len(valid_rates)*100:.0f}%" if valid_rates else "--"
        lines.append(f"  Win vs target: {' '.join(pc_parts)} | avg: {avg_str}")
        return "\n".join(lines)

    # Line 2: PFSP hardest/easiest
    if len(opponent_pool) > 1:
        ranked = []
        for e in opponent_pool:
            wd = e.get("win_data", [])
            if len(wd) > 0:
                total_raw = sum(rw for _, rw, _ in wd)
                total_games = sum(g for _, _, g in wd)
                if total_games > 0:
                    ranked.append((e["id"], 100.0 * total_raw / total_games))
        ranked.sort(key=lambda x: x[1])
        if ranked:
            hard_str = ", ".join(f"{n}({r:.0f}%)" for n, r in ranked[:3])
            easy_str = ", ".join(f"{n}({r:.0f}%)" for n, r in ranked[-3:])
            lines.append(f"  PFSP hardest: {hard_str} | easiest: {easy_str}")

    # Line 3: PC win rates and adaptive weights
    current_weights = worker_params.get("player_count_weights") or pc_base_weights
    pc_parts = []
    for i, pc in enumerate(PLAYER_COUNTS):
        if len(pc_win_tracker[pc]) >= min_pc_samples:
            rate = pc_win_rates.get(pc, sum(pc_win_tracker[pc]) / len(pc_win_tracker[pc]))
            pc_parts.append(f"{pc}p:{rate*100:.0f}%")
            writer.add_scalar(f"train/pc_{pc}p_win_rate", rate, global_step)
        else:
            pc_parts.append(f"{pc}p:--")
        writer.add_scalar(f"train/pc_{pc}p_weight", current_weights[i], global_step)
    wt_parts = [f"{pc}p:{w*100:.0f}%" for pc, w in zip(PLAYER_COUNTS, current_weights)]
    lines.append(f"  PC win: {' '.join(pc_parts)} | wt: {' '.join(wt_parts)}")

    # Lines 4-8: Full 12-cell win rate grid
    bot_labels = [("Smart", "smart"), ("Heur", "heuristic"), ("Random", "random")]
    lines.append(f"  {'':8s}  {'2p':>6s}  {'3p':>6s}  {'4p':>6s}  {'5p':>6s}  | {'avg':>5s}")
    lines.append(f"  {'':8s}  {'----':>6s}  {'----':>6s}  {'----':>6s}  {'----':>6s}  | {'-----':>5s}")
    for label, bot_key in bot_labels:
        cells = []
        valid_rates = []
        for pc in PLAYER_COUNTS:
            key = (bot_key, pc)
            tracker = cell_win_tracker.get(key)
            if tracker and len(tracker) >= 5:
                rate = cell_win_rates.get(key)
                if rate is None:
                    rate = sum(tracker) / len(tracker)
                cells.append(f"{rate*100:5.0f}%")
                valid_rates.append(rate)
                writer.add_scalar(f"train/forced_{pc}p_{bot_key}_win_rate", rate, global_step)
            else:
                cells.append(f"{'--':>6s}")
        avg = sum(valid_rates) / len(valid_rates) if valid_rates else None
        avg_str = f"{avg*100:4.0f}%" if avg is not None else "  --"
        lines.append(f"  {label:8s}  {cells[0]:>6s}  {cells[1]:>6s}  {cells[2]:>6s}  {cells[3]:>6s}  | {avg_str:>5s}")

    # Bottom row: column averages
    col_avgs = []
    for pc in PLAYER_COUNTS:
        rates = []
        for _, bot_key in bot_labels:
            key = (bot_key, pc)
            tracker = cell_win_tracker.get(key)
            if tracker and len(tracker) >= 5:
                rate = cell_win_rates.get(key)
                if rate is None:
                    rate = sum(tracker) / len(tracker)
                rates.append(rate)
        if rates:
            col_avgs.append(f"{sum(rates)/len(rates)*100:5.0f}%")
        else:
            col_avgs.append(f"{'--':>6s}")
    all_valid = [r for (_, bk) in bot_labels for pc in PLAYER_COUNTS
                 for r in ([cell_win_rates.get((bk, pc))]
                           if cell_win_rates.get((bk, pc)) is not None else [])]
    grand_avg = f"{sum(all_valid)/len(all_valid)*100:4.0f}%" if all_valid else "  --"
    lines.append(f"  {'avg':8s}  {col_avgs[0]:>6s}  {col_avgs[1]:>6s}  {col_avgs[2]:>6s}  {col_avgs[3]:>6s}  | {grand_avg:>5s}")

    # Log conditional forced rates to TensorBoard
    pc_bot_rates = worker_params.get("forced_pc_bot_rates")
    if pc_bot_rates is not None:
        for fpc, bot_rates in pc_bot_rates.items():
            for bot, cond_rate in bot_rates.items():
                writer.add_scalar(f"train/forced_{fpc}p_{bot}_cond_rate",
                                  cond_rate, global_step)

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Oh Hell PPO agent")
    # Config file
    parser.add_argument("--config", type=str, default="config.toml",
                        help="Path to TOML config file")
    # Training
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Subprocess workers for env stepping (0 = auto)")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--steps-per-rollout", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden-dim", type=int, default=512)
    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=1_000_000,
                        help="Evaluate every N steps")
    parser.add_argument("--eval-games", type=int, default=50,
                        help="Games per player-count per opponent-type")
    # Self-play
    parser.add_argument("--self-play-start", type=int, default=500_000,
                        help="Start self-play after N steps")
    parser.add_argument("--snapshot-interval", type=int, default=50_000,
                        help="Save self-play snapshot every N steps")
    parser.add_argument("--snapshot-dir", type=str, default=None,
                        help="Directory for persistent self-play snapshot files")
    parser.add_argument("--sp-fraction", type=float, default=0.5,
                        help="(Deprecated) PFSP handles opponent selection automatically")
    # Checkpoints
    parser.add_argument("--checkpoint-interval", type=int, default=100_000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Path to snapshot/checkpoint to load model weights from "
                             "(no optimizer state, step counter starts at 0)")
    # PFSP parameters
    parser.add_argument("--pfsp-exploration-bonus", type=float, default=None)
    parser.add_argument("--pfsp-staleness-divisor", type=float, default=None)
    parser.add_argument("--pfsp-max-staleness-multiplier", type=float, default=None)
    parser.add_argument("--pfsp-reward-window", type=int, default=None)
    parser.add_argument("--pfsp-pool-size", type=int, default=None)
    parser.add_argument("--pfsp-log-interval", type=int, default=None)
    # League / exploiter
    parser.add_argument("--load-dirs", type=str, default="",
                        help="Comma-separated additional snapshot dirs to scan")
    parser.add_argument("--rescan-interval", type=int, default=0,
                        help="Re-scan snapshot dirs every N updates (0 = disabled)")
    parser.add_argument("--exploiter-mode", action="store_true", default=False,
                        help="Exploiter mode: no fixed bots, pool from loaded snapshots only")
    # Opponent parameters
    parser.add_argument("--opp-homogeneous-rate", type=float, default=None)
    parser.add_argument("--opp-primary-bias", type=float, default=None)
    parser.add_argument("--nn-temperature", type=float, default=None)
    parser.add_argument("--forced-smart", type=float, default=None)
    parser.add_argument("--forced-heuristic", type=float, default=None)
    parser.add_argument("--forced-random", type=float, default=None)
    parser.add_argument("--forced-adaptation-rate", type=float, default=None,
                        help="Blend rate for forced cell adaptation (0=static, 1=fully adaptive)")
    parser.add_argument("--forced-min-cell-rate", type=float, default=None,
                        help="Minimum unconditional rate per (bot,pc) cell")
    parser.add_argument("--forced-win-window", type=int, default=None,
                        help="Rolling window size for per-cell win rate tracking")

    parser.add_argument("--pc-base-weights", type=float, nargs=4, default=None,
                        help="Base player count weights for [2p, 3p, 4p, 5p]")
    parser.add_argument("--pc-adaptation-rate", type=float, default=None,
                        help="Blend rate toward weakness-based weights (0=static, 1=fully adaptive)")
    parser.add_argument("--pc-min-weight", type=float, default=None,
                        help="Minimum weight per player count (prevents starvation)")
    parser.add_argument("--pc-win-window", type=int, default=None,
                        help="Rolling window size for per-PC win rate tracking")

    return parser.parse_args()


# ---- Config loading (3-tier: defaults < config.toml < CLI args) ----

# Maps TOML paths to argparse attribute names
CONFIG_MAP = {
    "training.total_timesteps": "total_timesteps",
    "training.num_envs": "num_envs",
    "training.num_workers": "num_workers",
    "training.steps_per_rollout": "steps_per_rollout",
    "training.lr": "lr",
    "training.gamma": "gamma",
    "training.gae_lambda": "gae_lambda",
    "training.clip_eps": "clip_eps",
    "training.epochs": "epochs",
    "training.minibatch_size": "minibatch_size",
    "training.ent_coef": "ent_coef",
    "training.vf_coef": "vf_coef",
    "training.max_grad_norm": "max_grad_norm",
    "training.anneal_lr": "anneal_lr",
    "training.seed": "seed",
    "training.device": "device",
    "training.hidden_dim": "hidden_dim",
    "pfsp.exploration_bonus": "pfsp_exploration_bonus",
    "pfsp.staleness_divisor": "pfsp_staleness_divisor",
    "pfsp.max_staleness_multiplier": "pfsp_max_staleness_multiplier",
    "pfsp.reward_window": "pfsp_reward_window",
    "pfsp.pool_size": "pfsp_pool_size",
    "pfsp.log_interval": "pfsp_log_interval",
    "opponents.homogeneous_rate": "opp_homogeneous_rate",
    "opponents.primary_bias": "opp_primary_bias",
    "opponents.nn_temperature": "nn_temperature",
    "opponents.forced_homogeneous.smart": "forced_smart",
    "opponents.forced_homogeneous.heuristic": "forced_heuristic",
    "opponents.forced_homogeneous.random": "forced_random",
    "opponents.forced_homogeneous.adaptation_rate": "forced_adaptation_rate",
    "opponents.forced_homogeneous.min_cell_rate": "forced_min_cell_rate",
    "opponents.forced_homogeneous.win_window": "forced_win_window",
    "player_counts.base_weights": "pc_base_weights",
    "player_counts.adaptation_rate": "pc_adaptation_rate",
    "player_counts.min_weight": "pc_min_weight",
    "player_counts.win_window": "pc_win_window",
    "selfplay.self_play_start": "self_play_start",
    "selfplay.snapshot_interval": "snapshot_interval",
    "selfplay.snapshot_dir": "snapshot_dir",
    "evaluation.eval_interval": "eval_interval",
    "evaluation.eval_games": "eval_games",
    "checkpoints.checkpoint_interval": "checkpoint_interval",
    "checkpoints.save_dir": "save_dir",
    "checkpoints.run_name": "run_name",
    "checkpoints.resume": "resume",
    "league.load_dirs": "load_dirs",
    "league.rescan_interval": "rescan_interval",
    "league.exploiter_mode": "exploiter_mode",
    "league.init_weights": "init_weights",
}

# Defaults for new parameters (not covered by existing argparse defaults)
NEW_DEFAULTS = {
    "pfsp_exploration_bonus": 2.0,
    "pfsp_staleness_divisor": 50.0,
    "pfsp_max_staleness_multiplier": 3.0,
    "pfsp_reward_window": 20,
    "pfsp_pool_size": 50,
    "pfsp_log_interval": 20,
    "opp_homogeneous_rate": 0.25,
    "opp_primary_bias": 0.5,
    "nn_temperature": 1.0,
    "forced_smart": 0.05,
    "forced_heuristic": 0.03,
    "forced_random": 0.02,
    "forced_adaptation_rate": 0.5,
    "forced_min_cell_rate": 0.002,
    "forced_win_window": 50,
    "snapshot_dir": "snapshots",
    "pc_base_weights": [0.15, 0.20, 0.30, 0.35],
    "pc_adaptation_rate": 0.5,
    "pc_min_weight": 0.05,
    "pc_win_window": 50,
    "load_dirs": "",
    "rescan_interval": 0,
    "exploiter_mode": False,
}


def apply_config(args):
    """Apply TOML config with 3-tier precedence: defaults < TOML < CLI."""
    # Detect which args were explicitly passed on the command line
    cli_explicit = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            cli_explicit.add(arg.lstrip("-").replace("-", "_"))

    # Load TOML config if available
    toml_values = {}
    config_path = getattr(args, "config", "config.toml")
    if config_path and os.path.exists(config_path):
        if tomllib is None:
            print(f"Warning: tomli/tomllib not available, skipping {config_path}")
        else:
            with open(config_path, "rb") as f:
                raw = tomllib.load(f)
            for toml_path, attr_name in CONFIG_MAP.items():
                parts = toml_path.split(".")
                val = raw
                for p in parts:
                    if isinstance(val, dict) and p in val:
                        val = val[p]
                    else:
                        val = None
                        break
                if val is not None:
                    toml_values[attr_name] = val
            print(f"Loaded config from {config_path}")

    # Apply: CLI explicit > TOML > argparse default / NEW_DEFAULTS
    for attr_name in set(CONFIG_MAP.values()):
        if attr_name in cli_explicit:
            continue  # CLI takes priority
        if attr_name in toml_values:
            setattr(args, attr_name, toml_values[attr_name])

    # Fill in new parameters that have no argparse default
    for attr_name, default_val in NEW_DEFAULTS.items():
        current = getattr(args, attr_name, None)
        if current is None:
            setattr(args, attr_name, default_val)

    # Handle empty string as None for resume/run_name
    if getattr(args, "resume", None) == "":
        args.resume = None
    if getattr(args, "run_name", None) == "":
        args.run_name = None

    # Parse load_dirs from comma-separated string to list
    raw_ld = getattr(args, "load_dirs", "")
    if isinstance(raw_ld, list):
        args.load_dirs = [d.strip() for d in raw_ld if d.strip()]
    elif isinstance(raw_ld, str):
        args.load_dirs = [d.strip() for d in raw_ld.split(",") if d.strip()]
    else:
        args.load_dirs = []

    return args


class NNBot:
    """Bot wrapper that uses a neural network to make decisions."""

    def __init__(self, network, device):
        self.network = network
        self.device = device

    def act(self, game, player):
        obs, mask = get_obs_and_mask_for_bot(game, player)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.network(obs_t, mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        if game.phase == Phase.BIDDING:
            return action - 52
        else:
            return index_to_card(action)


# Preallocated buffers for get_obs_and_mask_for_bot (one per process)
_BOT_OBS_SIZE = 453
_BOT_ACTION_SIZE = 52 + MAX_BID + 1  # 65
_bot_obs_buf = np.zeros(_BOT_OBS_SIZE, dtype=np.float32)
_bot_mask_buf = np.zeros(_BOT_ACTION_SIZE, dtype=np.float32)


def get_obs_and_mask_for_bot(game, player):
    """Build obs and action mask for a given player in a game.

    Standalone function usable by NNBot and the GUI.
    Writes into preallocated buffers to avoid allocations.
    """
    obs = _bot_obs_buf
    obs[:] = 0.0
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
    ct = game.current_trick
    for _, card in ct:
        obs[73 + card] = 1.0

    # Lead suit (4) — offset 125
    if game.phase == Phase.PLAYING and ct:
        obs[125 + ct[0][1] // 13] = 1.0

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
        trick_pos = len(ct)
        if trick_pos < MAX_PLAYERS:
            obs[448 + trick_pos] = 1.0

    # Action mask
    mask = _bot_mask_buf
    mask[:] = 0.0
    if game.phase == Phase.BIDDING:
        for bid in game.get_legal_bids():
            mask[52 + bid] = 1.0
    else:
        for card in game.get_legal_plays():
            mask[card] = 1.0

    return obs.copy(), mask.copy()


def evaluate_agent(network, device, num_players, num_games, opponent_type="heuristic"):
    """Run evaluation games for a specific player count and return metrics.

    Uses greedy (argmax) action selection for deterministic evaluation.
    """
    scores = []
    winning_scores = []
    bid_hits = 0
    total_rounds = 0
    wins = 0

    for g in range(num_games):
        if opponent_type == "random":
            opps = [RandomBot(rng=random.Random(g + 1000 * i)) for i in range(num_players - 1)]
        elif opponent_type == "smart":
            opps = [SmartBot(rng=random.Random(g + 1000 * i)) for i in range(num_players - 1)]
        else:
            opps = [HeuristicBot(rng=random.Random(g + 1000 * i)) for i in range(num_players - 1)]

        env = OhHellEnv(num_players=num_players, opponents=opps, seed=g + 50000)
        obs, info = env.reset()
        done = False

        while not done:
            mask = info["action_mask"]
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = network(obs_t, mask_t)
                action = logits.argmax(dim=-1).item()

            obs, reward, done, _, info = env.step(action)

            # Track bid hits per round (reward > 0 iff agent hit their bid)
            if info.get("round_complete", False):
                total_rounds += 1
                if reward > 0:
                    bid_hits += 1

        final_scores = info.get("final_scores", env.game.scores)
        agent_score = final_scores[0]
        scores.append(agent_score)
        winning_scores.append(max(final_scores))

        if agent_score == max(final_scores):
            wins += 1

    avg_score = np.mean(scores)
    win_rate = wins / num_games
    bid_accuracy = bid_hits / max(total_rounds, 1)
    avg_winning_score = np.mean(winning_scores)

    return {
        "avg_score": avg_score,
        "win_rate": win_rate,
        "bid_accuracy": bid_accuracy,
        "avg_winning_score": avg_winning_score,
        "max_winning_score": max(winning_scores),
        "min_winning_score": min(winning_scores),
    }


def evaluate_all_configs(network, device, num_games_per, opponent_type="heuristic"):
    """Evaluate across all player counts. Returns per-config and aggregate metrics."""
    all_metrics = {}
    for np_ in PLAYER_COUNTS:
        all_metrics[np_] = evaluate_agent(network, device, np_, num_games_per, opponent_type)

    # Aggregate: average across player counts
    agg = {
        "avg_score": np.mean([m["avg_score"] for m in all_metrics.values()]),
        "win_rate": np.mean([m["win_rate"] for m in all_metrics.values()]),
        "bid_accuracy": np.mean([m["bid_accuracy"] for m in all_metrics.values()]),
        "avg_winning_score": np.mean([m["avg_winning_score"] for m in all_metrics.values()]),
        "max_winning_score": max(m["max_winning_score"] for m in all_metrics.values()),
        "min_winning_score": min(m["min_winning_score"] for m in all_metrics.values()),
    }
    return all_metrics, agg


def make_env_with_random_players(seed, base_seed, auto_opponents=True,
                                 player_count_weights=None):
    """Create an env with a weighted-random player count (2-5).

    Opponents are a mix of SmartBot, HeuristicBot, and RandomBot for training
    diversity.  Player counts weighted toward 4p/5p: [0.15, 0.20, 0.30, 0.35].
    When auto_opponents=False, opponents are still created (for fallback) but
    the env won't auto-play them — the caller handles opponent actions externally.
    """
    rng = random.Random(seed)
    weights = player_count_weights if player_count_weights is not None else PLAYER_COUNT_WEIGHTS
    num_players = rng.choices(PLAYER_COUNTS, weights=weights, k=1)[0]
    opps = []
    for j in range(num_players - 1):
        opp_rng = random.Random(base_seed + seed * 100 + j)
        roll = rng.random()
        if roll < 0.15:
            opps.append(RandomBot(rng=opp_rng))
        elif roll < 0.55:
            opps.append(HeuristicBot(rng=opp_rng))
        else:
            opps.append(SmartBot(rng=opp_rng))
    return OhHellEnv(num_players=num_players, opponents=opps, seed=seed,
                     auto_opponents=auto_opponents)


def _make_bot(bot_type, rng):
    """Create a bot of the given type."""
    if bot_type == "random":
        return RandomBot(rng=rng)
    elif bot_type == "heuristic":
        return HeuristicBot(rng=rng)
    else:
        return SmartBot(rng=rng)


# ---- PFSP (Prioritized Fictitious Self-Play) opponent pool ----

def create_opponent_pool(reward_window=20):
    """Create initial opponent pool with fixed bot entries."""
    return [
        {"id": "RandomBot", "type": "random", "state_dict": None,
         "step": 0, "rewards": deque(maxlen=reward_window),
         "win_data": deque(maxlen=reward_window), "last_used": 0},
        {"id": "HeuristicBot", "type": "heuristic", "state_dict": None,
         "step": 0, "rewards": deque(maxlen=reward_window),
         "win_data": deque(maxlen=reward_window), "last_used": 0},
        {"id": "SmartBot", "type": "smart", "state_dict": None,
         "step": 0, "rewards": deque(maxlen=reward_window),
         "win_data": deque(maxlen=reward_window), "last_used": 0},
    ]


def load_snapshot_dir(snapshot_dir, hidden_dim, pool_size, reward_window):
    """Load self-play snapshots from a directory into opponent pool entries.

    Scans snapshot_dir for .pt files, validates hidden_dim compatibility,
    and returns a list of pool entries (up to pool_size, most recent by step).

    Accepts two file formats:
      - Snapshot format: {"state_dict": ..., "hidden_dim": ..., "step": ...}
      - Checkpoint format: {"network": ..., "hidden_dim": ..., "global_step": ...}
    """
    import glob as glob_mod

    if not os.path.isdir(snapshot_dir):
        return []

    pt_files = glob_mod.glob(os.path.join(snapshot_dir, "*.pt"))
    if not pt_files:
        return []

    candidates = []
    for filepath in pt_files:
        filename = os.path.basename(filepath)
        try:
            data = torch.load(filepath, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  Warning: Failed to load {filename}: {e}")
            continue

        # Extract state_dict and step, handling both formats
        if "state_dict" in data:
            state_dict = data["state_dict"]
            step = data.get("step", 0)
            file_hidden_dim = data.get("hidden_dim", None)
        elif "network" in data:
            state_dict = data["network"]
            step = data.get("global_step", 0)
            file_hidden_dim = data.get("hidden_dim", None)
        else:
            print(f"  Warning: Skipping {filename} — no 'state_dict' or 'network' key")
            continue

        if file_hidden_dim is not None and file_hidden_dim != hidden_dim:
            print(f"  Warning: Skipping {filename} — hidden_dim mismatch "
                  f"(file={file_hidden_dim}, current={hidden_dim})")
            continue

        # Extract run tag from filename (e.g. "EXP_TAGLK7_1M.pt" -> "TAGLK7")
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")
        run_tag = parts[1] if len(parts) >= 3 else stem

        candidates.append((step, state_dict, filename, run_tag))

    if not candidates:
        return []

    # Sort by step descending, take most recent pool_size
    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) > pool_size:
        skipped = len(candidates) - pool_size
        candidates = candidates[:pool_size]
        print(f"  Snapshot dir: {skipped} older snapshots skipped (pool_size={pool_size})")

    # Build pool entries sorted ascending by step (oldest first)
    candidates.reverse()
    entries = []
    seen_keys = set()
    for step, state_dict, filename, run_tag in candidates:
        key = (run_tag, step)
        if key in seen_keys:
            print(f"  Warning: Skipping duplicate step {step} from {filename}")
            continue
        seen_keys.add(key)

        is_exp = filename.upper().startswith("EXP") or "exploiter" in filename.lower()
        prefix = "EXP" if is_exp else "snap"
        entries.append({
            "id": f"{prefix}_{run_tag}_{_step_tag(step)}", "type": "snapshot",
            "state_dict": state_dict, "step": step, "run_tag": run_tag,
            "rewards": deque(maxlen=reward_window),
            "win_data": deque(maxlen=reward_window),
            "last_used": 0,
        })

    return entries


def load_snapshot_dirs(dirs, hidden_dim, pool_size, reward_window,
                       existing_keys=None):
    """Load snapshots from multiple directories, deduplicating by (run_tag, step).

    Returns up to pool_size new entries (not already in existing_keys),
    sorted ascending by step.
    """
    if existing_keys is None:
        existing_keys = set()
    all_entries = []
    for d in dirs:
        entries = load_snapshot_dir(d, hidden_dim, pool_size * 2, reward_window)
        for entry in entries:
            key = (entry.get("run_tag", ""), entry["step"])
            if key not in existing_keys:
                all_entries.append(entry)
                existing_keys.add(key)
    # Category-aware truncation: EXP (exploiter) entries get a guaranteed
    # sub-quota so they aren't crowded out by higher-step MAIN entries.
    exp = [e for e in all_entries if e["id"].startswith("EXP")]
    non_exp = [e for e in all_entries if not e["id"].startswith("EXP")]
    exp.sort(key=lambda e: e["step"], reverse=True)
    non_exp.sort(key=lambda e: e["step"], reverse=True)

    exp_quota = pool_size // 3
    non_exp_quota = pool_size - min(len(exp), exp_quota)
    kept = non_exp[:non_exp_quota] + exp[:exp_quota]
    kept.sort(key=lambda e: e["step"])  # oldest first (original contract)
    return kept


def pfsp_select(pool, update_num, exploration_bonus=2.0,
                staleness_divisor=50.0, max_staleness_mult=3.0):
    """Select opponent using Prioritized Fictitious Self-Play.

    Uses player-count-adjusted win rate as the difficulty signal.
    Weight proportional to (1 - normalized_win_rate + eps)^2.
    Lower win rate = harder opponent = higher selection weight.
    Falls back to reward-based signal if no win data exists yet.
    """
    if not pool:
        return None

    win_rates = []
    for entry in pool:
        if len(entry.get("win_data", [])) > 0:
            total_adj = sum(a for a, _, _ in entry["win_data"])
            total_games = sum(g for _, _, g in entry["win_data"])
            if total_games > 0:
                win_rates.append(total_adj / total_games)
            else:
                win_rates.append(None)
        elif len(entry["rewards"]) > 0:
            # Fallback to reward for entries with no win data yet
            win_rates.append(sum(entry["rewards"]) / len(entry["rewards"]))
        else:
            win_rates.append(None)

    valid = [r for r in win_rates if r is not None]
    if not valid:
        return random.choice(pool)

    max_r = max(valid)
    min_r = min(valid)
    range_r = max_r - min_r if max_r > min_r else 1.0

    weights = []
    for i, entry in enumerate(pool):
        wr = win_rates[i]
        if wr is None:
            base_weight = exploration_bonus
        else:
            norm = (wr - min_r) / range_r
            base_weight = (1.0 - norm + 0.1) ** 2

        staleness = update_num - entry["last_used"]
        staleness_mult = 1.0 + min(staleness / staleness_divisor,
                                   max_staleness_mult)
        weights.append(base_weight * staleness_mult)

    return random.choices(pool, weights=weights, k=1)[0]


# ---- Multiprocessing environment workers ----

def _env_worker(remote, parent_remote, env_count, seed_offset, base_seed,
                hidden_dim):
    """Subprocess worker that collects complete rollouts locally.

    Each worker has its own CPU copy of the agent network and runs T steps
    independently, eliminating per-step IPC with the main process.

    Done signal is set at round boundaries (not just game over) so GAE
    treats each round as an independent episode.

    Supports multi-opponent tables: each collect command includes a list of
    opponent configs (bot types and/or NN snapshots). Each opponent seat in
    each env is independently assigned to a config, so a single 4-player
    game might have snapshot_A, HeuristicBot, and snapshot_B as opponents.
    """
    # Ignore Ctrl+C in workers — main process handles graceful shutdown
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent_remote.close()

    # Multiple NN opponents (lazy-allocated)
    sp_networks = []
    # Per-env seat assignments: {player_idx: config_list_index}
    env_seat_configs = [{} for _ in range(env_count)]
    # Map from opp_configs index -> sp_networks list index (for NN configs)
    nn_idx_map = {}
    # Current opp_configs for seat assignment on env reset
    current_opp_configs = [("heuristic", None)]
    # Bot instances per config index (only for non-NN configs)
    config_bots = {0: HeuristicBot(rng=random.Random(base_seed))}

    # Create envs (all use auto_opponents=False for unified handling)
    envs = []
    for i in range(env_count):
        envs.append(make_env_with_random_players(seed_offset + i, base_seed,
                                                 auto_opponents=False))

    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    # Agent network for local inference (CPU)
    agent_network = OhHellNetwork(obs_dim, hidden_dim=hidden_dim)
    agent_network.eval()

    # Track which config indices were actually seated during a rollout
    configs_used = set()
    # Forced homogeneous bot tables (bypass PFSP)
    forced_bots = {}        # env_idx -> bot_type string
    forced_bot_inst = {}    # env_idx -> bot instance
    # Worker params (updated each collect command)
    w_params = {
        "forced_rates": {"smart": 0.05, "heuristic": 0.03, "random": 0.02},
        "homogeneous_rate": 0.25,
        "primary_bias": 0.5,
        "nn_temperature": 1.0,
    }
    # Win tracking per rollout
    config_win_data = {}    # cfg_idx_or_string -> [(adjusted, raw_won)]
    pc_wins = {}            # player_count -> [raw_won]
    forced_cell_wins = {}   # (bot_type_str, pc) -> [raw_won, ...]

    def record_game_result(j):
        """Record win/loss before reset_env. Attributes to seated configs."""
        scores = envs[j].game.scores
        pc = envs[j].game.num_players
        won = 1.0 if scores[0] == max(scores) else 0.0
        adjusted = won * pc  # baseline = 1.0 for random play at any pc

        # Per-player-count (all tables including forced)
        pc_wins.setdefault(pc, []).append(won)

        if j in forced_bots:
            ftype = forced_bots[j]
            config_win_data.setdefault(ftype, []).append((adjusted, won))
            forced_cell_wins.setdefault((ftype, pc), []).append(won)
            return

        # PFSP tables: attribute to all seated configs
        for cfg_idx in set(env_seat_configs[j].values()):
            config_win_data.setdefault(cfg_idx, []).append((adjusted, won))

    def assign_seats(j):
        """Assign each opponent seat to a config from current_opp_configs.

        First checks for forced homogeneous bot tables (bypass PFSP).
        Then: configurable % of tables are homogeneous (all config 0).
        Otherwise: seat 1 = config 0, remaining seats biased toward primary.
        """
        np_ = envs[j].game.num_players
        n_configs = len(current_opp_configs)
        seat_map = {}

        # Check for forced homogeneous bot table
        # Use per-PC conditional rates if available, else flat rates
        pc = envs[j].game.num_players
        pc_bot_rates = w_params.get("forced_pc_bot_rates")
        if pc_bot_rates is not None and pc in pc_bot_rates:
            rates = pc_bot_rates[pc]
        else:
            rates = w_params.get("forced_rates", {})

        roll = random.random()
        cumulative = 0.0
        forced_type = None
        for bot_type, rate in rates.items():
            cumulative += rate
            if roll < cumulative:
                forced_type = bot_type
                break

        if forced_type is not None:
            forced_bots[j] = forced_type
            forced_bot_inst[j] = _make_bot(
                forced_type, random.Random(base_seed + j))
            for p in range(1, np_):
                seat_map[p] = -1  # sentinel: use forced bot
            env_seat_configs[j] = seat_map
            return

        # Normal PFSP-based assignment
        forced_bots.pop(j, None)
        forced_bot_inst.pop(j, None)
        homogeneous = random.random() < w_params["homogeneous_rate"]
        for p in range(1, np_):
            if homogeneous or p == 1 or n_configs == 1:
                seat_map[p] = 0
            elif random.random() < w_params["primary_bias"]:
                seat_map[p] = 0
            else:
                seat_map[p] = random.randint(0, n_configs - 1)
        env_seat_configs[j] = seat_map
        configs_used.update(seat_map.values())

    def _step_bot_opponent(j, bot):
        """Have a bot take one action in the given env."""
        player = envs[j].game.current_player
        action = bot.act(envs[j].game, player)
        if envs[j].game.phase == Phase.BIDDING:
            act_idx = action + 52
        else:
            act_idx = card_index(action)
        envs[j].step_opponent(act_idx)

    def play_to_agent_turn(j):
        """Play opponents using config-driven bots until agent's turn.

        Forced tables use the forced bot instance directly.
        PFSP tables use the config-driven bot or NN snapshot.
        """
        nn_temp = w_params["nn_temperature"]
        while envs[j].needs_opponent_action():
            if j in forced_bots:
                _step_bot_opponent(j, forced_bot_inst[j])
            else:
                player = envs[j].game.current_player
                cfg_idx = env_seat_configs[j].get(player, 0)
                cfg_type = current_opp_configs[cfg_idx][0]
                if cfg_type != "nn":
                    _step_bot_opponent(j, config_bots[cfg_idx])
                else:
                    sp_idx = nn_idx_map[cfg_idx]
                    o, m = get_obs_and_mask_for_bot(envs[j].game, player)
                    with torch.no_grad():
                        obs_t = torch.from_numpy(o).unsqueeze(0)
                        mask_t = torch.from_numpy(m).unsqueeze(0)
                        logits, _ = sp_networks[sp_idx](obs_t, mask_t)
                        if nn_temp != 1.0:
                            logits = logits / nn_temp
                        action = torch.distributions.Categorical(
                            logits=logits).sample().item()
                    envs[j].step_opponent(action)

    # Reset all envs and assign initial seats
    obs_arr = np.empty((env_count, obs_dim), dtype=np.float32)
    mask_arr = np.empty((env_count, act_dim), dtype=np.float32)
    for j, env in enumerate(envs):
        env.reset()
        assign_seats(j)
        play_to_agent_turn(j)
        obs_arr[j] = env._get_obs()
        mask_arr[j] = env._get_action_mask()

    # Seed counter for env resets
    seed_counter = seed_offset + env_count

    def reset_env(j):
        """Create a new env, reset it, and assign seats.

        Uses config-driven bots (from PFSP selection) to advance the game
        to the agent's turn, since auto_opponents is False.
        """
        nonlocal seed_counter
        seed_counter += 1
        pc_weights = w_params.get("player_count_weights", None)
        envs[j] = make_env_with_random_players(
            seed_counter, base_seed, auto_opponents=False,
            player_count_weights=pc_weights)
        envs[j].reset()
        assign_seats(j)
        play_to_agent_turn(j)
        obs = envs[j]._get_obs()
        info = {"action_mask": envs[j]._get_action_mask()}
        return obs, info

    # Signal ready
    remote.send("ready")

    try:
        while True:
            cmd = remote.recv()
            if cmd is None:
                break

            if cmd[0] == "collect":
                agent_sd, opp_configs, T = cmd[1], cmd[2], cmd[3]
                if len(cmd) > 4:
                    w_params.update(cmd[4])

                # Update agent weights
                agent_network.load_state_dict(agent_sd)

                # Update opponent configs and load NN weights
                configs_used.clear()
                forced_bots.clear()
                forced_bot_inst.clear()
                config_win_data.clear()
                pc_wins.clear()
                forced_cell_wins.clear()
                current_opp_configs = opp_configs
                nn_configs = [(i, sd) for i, (t, sd) in enumerate(opp_configs)
                              if t == "nn"]
                # Ensure enough sp_networks are allocated
                while len(sp_networks) < len(nn_configs):
                    net = OhHellNetwork(obs_dim, hidden_dim=hidden_dim)
                    net.eval()
                    sp_networks.append(net)
                # Load state dicts and build index map
                nn_idx_map.clear()
                for sp_idx, (cfg_idx, sd) in enumerate(nn_configs):
                    sp_networks[sp_idx].load_state_dict(sd)
                    nn_idx_map[cfg_idx] = sp_idx

                # Create bot instances for non-NN configs so PFSP
                # bot selection actually controls what plays
                config_bots.clear()
                for i, (cfg_type, _) in enumerate(opp_configs):
                    if cfg_type in ("random", "heuristic", "smart"):
                        config_bots[i] = _make_bot(
                            cfg_type, random.Random(base_seed + i))

                # Reassign seats for all envs with new configs
                for j in range(env_count):
                    assign_seats(j)

                # Allocate rollout buffers
                r_obs = np.empty((T, env_count, obs_dim), dtype=np.float32)
                r_mask = np.empty((T, env_count, act_dim), dtype=np.float32)
                r_act = np.empty((T, env_count), dtype=np.int64)
                r_logp = np.empty((T, env_count), dtype=np.float32)
                r_val = np.empty((T, env_count), dtype=np.float32)
                r_rew = np.empty((T, env_count), dtype=np.float32)
                r_done = np.empty((T, env_count), dtype=np.float32)

                for step in range(T):
                    # Store current state
                    r_obs[step] = obs_arr
                    r_mask[step] = mask_arr

                    # Batched agent forward pass (CPU)
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs_arr)
                        mask_t = torch.from_numpy(mask_arr)
                        logits, value = agent_network(obs_t, mask_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        actions = dist.sample()
                        log_probs = dist.log_prob(actions)

                    r_act[step] = actions.numpy()
                    r_logp[step] = log_probs.numpy()
                    r_val[step] = value.squeeze(-1).numpy()

                    # Step all envs
                    rew_arr = np.zeros(env_count, dtype=np.float32)
                    done_arr = np.zeros(env_count, dtype=np.float32)

                    # Phase 1: Apply agent actions
                    for j in range(env_count):
                        obs, reward, done, _, info = envs[j].step(
                            int(r_act[step, j]))
                        rew_arr[j] = reward
                        round_done = info.get("round_complete", False) or done
                        done_arr[j] = float(round_done)
                        if done:
                            record_game_result(j)
                            obs, info = reset_env(j)
                        obs_arr[j] = obs
                        mask_arr[j] = info.get("action_mask",
                                               np.zeros(act_dim, dtype=np.float32))

                    # Phase 2: Process all opponent actions until agent's turn
                    for _ in range(50):
                        pending = []
                        for j in range(env_count):
                            if envs[j].needs_opponent_action():
                                pending.append(j)
                        if not pending:
                            break

                        # Separate bot/forced seats from NN seats
                        nn_by_net = {}  # sp_networks idx -> [(env_idx, player)]
                        nn_temp = w_params["nn_temperature"]
                        for j in pending:
                            player = envs[j].game.current_player

                            if j in forced_bots:
                                # Forced homogeneous table — use forced bot
                                bot = forced_bot_inst[j]
                                action = bot.act(envs[j].game, player)
                                if envs[j].game.phase == Phase.BIDDING:
                                    act_idx = action + 52
                                else:
                                    act_idx = card_index(action)
                                rw, rd = envs[j].step_opponent(act_idx)
                                rew_arr[j] += rw
                                if rd:
                                    done_arr[j] = 1.0
                                if envs[j].game.is_game_over:
                                    done_arr[j] = 1.0
                                    record_game_result(j)
                                    obs, info = reset_env(j)
                                    obs_arr[j] = obs
                                    mask_arr[j] = info.get(
                                        "action_mask",
                                        np.zeros(act_dim, dtype=np.float32))
                                continue

                            cfg_idx = env_seat_configs[j].get(player, 0)
                            cfg_type = opp_configs[cfg_idx][0]

                            if cfg_type != "nn":
                                # Bot seat — use PFSP-selected bot type
                                bot = config_bots[cfg_idx]
                                action = bot.act(envs[j].game, player)
                                if envs[j].game.phase == Phase.BIDDING:
                                    act_idx = action + 52
                                else:
                                    act_idx = card_index(action)
                                rw, rd = envs[j].step_opponent(act_idx)
                                rew_arr[j] += rw
                                if rd:
                                    done_arr[j] = 1.0
                                if envs[j].game.is_game_over:
                                    done_arr[j] = 1.0
                                    record_game_result(j)
                                    obs, info = reset_env(j)
                                    obs_arr[j] = obs
                                    mask_arr[j] = info.get(
                                        "action_mask",
                                        np.zeros(act_dim, dtype=np.float32))
                            else:
                                # NN seat — batch by network
                                sp_idx = nn_idx_map[cfg_idx]
                                if sp_idx not in nn_by_net:
                                    nn_by_net[sp_idx] = []
                                nn_by_net[sp_idx].append((j, player))

                        # Batched forward pass per NN network
                        for sp_idx, items in nn_by_net.items():
                            batch_obs = np.empty(
                                (len(items), obs_dim), dtype=np.float32)
                            batch_mask = np.empty(
                                (len(items), act_dim), dtype=np.float32)
                            for idx, (j, player) in enumerate(items):
                                o, m = get_obs_and_mask_for_bot(
                                    envs[j].game, player)
                                batch_obs[idx] = o
                                batch_mask[idx] = m

                            with torch.no_grad():
                                obs_t = torch.from_numpy(batch_obs)
                                mask_t = torch.from_numpy(batch_mask)
                                logits, _ = sp_networks[sp_idx](obs_t, mask_t)
                                if nn_temp != 1.0:
                                    logits = logits / nn_temp
                                opp_actions = (
                                    torch.distributions.Categorical(
                                        logits=logits).sample().numpy())

                            for idx, (j, player) in enumerate(items):
                                rw, rd = envs[j].step_opponent(
                                    int(opp_actions[idx]))
                                rew_arr[j] += rw
                                if rd:
                                    done_arr[j] = 1.0
                                if envs[j].game.is_game_over:
                                    done_arr[j] = 1.0
                                    record_game_result(j)
                                    obs, info = reset_env(j)
                                    obs_arr[j] = obs
                                    mask_arr[j] = info.get(
                                        "action_mask",
                                        np.zeros(act_dim,
                                                 dtype=np.float32))

                    # Final obs/mask for all envs (after opponent processing)
                    for j in range(env_count):
                        if not envs[j].game.is_game_over:
                            obs_arr[j] = envs[j]._get_obs()
                            mask_arr[j] = envs[j]._get_action_mask()

                    r_rew[step] = rew_arr
                    r_done[step] = done_arr

                # Compute per-config and per-(config, player_count) rewards
                env_total_rew = r_rew.sum(axis=0)  # shape: (env_count,)
                config_rewards = {}      # cfg_idx -> avg reward
                config_pc_rewards = {}   # (cfg_idx, player_count) -> avg reward
                for cfg_idx in configs_used:
                    rew_list = []
                    pc_buckets = {}  # player_count -> [rewards]
                    for j in range(env_count):
                        if j in forced_bots:
                            continue  # skip forced envs
                        if cfg_idx in env_seat_configs[j].values():
                            r = float(env_total_rew[j])
                            rew_list.append(r)
                            pc = envs[j].game.num_players
                            pc_buckets.setdefault(pc, []).append(r)
                    if rew_list:
                        config_rewards[cfg_idx] = sum(rew_list) / len(rew_list)
                    for pc, vals in pc_buckets.items():
                        config_pc_rewards[(cfg_idx, pc)] = (
                            sum(vals) / len(vals))

                # Send complete rollout + per-config reward/win data
                remote.send((r_obs, r_mask, r_act, r_logp, r_val,
                             r_rew, r_done, obs_arr.copy(), mask_arr.copy(),
                             set(configs_used), config_rewards,
                             config_pc_rewards,
                             dict(config_win_data), dict(pc_wins),
                             dict(forced_cell_wins)))

    except (EOFError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        remote.close()


_shutdown_requested = False


def _handle_shutdown(signum, frame):
    """Signal handler for graceful shutdown (used by dashboard)."""
    global _shutdown_requested
    _shutdown_requested = True


def main():
    global _shutdown_requested
    _shutdown_requested = False
    signal.signal(signal.SIGTERM, _handle_shutdown)

    args = parse_args()
    apply_config(args)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Training across player counts: {PLAYER_COUNTS}")

    run_name = args.run_name or f"PPO_{_to_base36(int(time.time()))}"
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"Logging to runs/{run_name}", flush=True)

    # CSV log for evaluation
    csv_path = f"runs/{run_name}/eval_log.csv"
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    csv_mode = "a" if (args.resume and os.path.exists(csv_path)) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    csv_writer = csv.writer(csv_file)

    if csv_mode == "w":
        header = ["step"]
        for np_ in PLAYER_COUNTS:
            for opp in ["random", "heuristic", "smart"]:
                header += [f"{np_}p_vs_{opp}_score", f"{np_}p_vs_{opp}_win", f"{np_}p_vs_{opp}_bid_acc", f"{np_}p_vs_{opp}_win_score"]
        for opp in ["random", "heuristic", "smart"]:
            header += [f"agg_vs_{opp}_score", f"agg_vs_{opp}_win", f"agg_vs_{opp}_bid_acc", f"agg_vs_{opp}_win_score"]
        csv_writer.writerow(header)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine worker count
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 8, 24)

    # Ensure num_envs is divisible by num_workers
    envs_per_worker = args.num_envs // num_workers
    N = envs_per_worker * num_workers
    if N != args.num_envs:
        print(f"Adjusted num_envs from {args.num_envs} to {N} "
              f"(divisible by {num_workers} workers)")

    # Get obs/action dims from a temporary env
    tmp_env = make_env_with_random_players(0, args.seed)
    obs_dim = tmp_env.obs_size
    action_dim = tmp_env.action_size
    del tmp_env

    # Initialize network
    network = OhHellNetwork(obs_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr, eps=1e-5)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"Network: hidden_dim={args.hidden_dim}, params={param_count:,}")

    # Resume from checkpoint if requested
    global_step = 0
    _resumed_checkpoint = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint.get("global_step", 0)
        _resumed_checkpoint = checkpoint
        print(f"  Resumed at step {global_step}")
    elif getattr(args, "init_weights", None):
        print(f"Initializing weights from: {args.init_weights}")
        wt_data = torch.load(args.init_weights, map_location=device, weights_only=False)
        # Support snapshot format, checkpoint format, or raw state_dict
        if "state_dict" in wt_data:
            network.load_state_dict(wt_data["state_dict"])
        elif "network" in wt_data:
            network.load_state_dict(wt_data["network"])
        elif "model_state_dict" in wt_data:
            network.load_state_dict(wt_data["model_state_dict"])
        else:
            network.load_state_dict(wt_data)
        print("  Weights loaded (optimizer and step counter start fresh)")

    # Launch worker processes
    print(f"Launching {num_workers} env workers ({envs_per_worker} envs each, {N} total)...")
    workers = []
    parent_remotes = []
    for w in range(num_workers):
        parent_remote, worker_remote = mp.Pipe()
        p = mp.Process(
            target=_env_worker,
            args=(worker_remote, parent_remote, envs_per_worker,
                  args.seed + w * envs_per_worker, args.seed, args.hidden_dim),
            daemon=True,
        )
        p.start()
        worker_remote.close()
        workers.append(p)
        parent_remotes.append(parent_remote)

    # Wait for all workers to be ready
    for remote in parent_remotes:
        remote.recv()  # "ready"
    print(f"All {num_workers} workers ready.")

    # PFSP opponent pool
    if args.exploiter_mode:
        opponent_pool = []
        sp_active = True
        args.self_play_start = 0  # save snapshots immediately
        print("Exploiter mode: no fixed bots, pool from loaded snapshots only")
    else:
        opponent_pool = create_opponent_pool(args.pfsp_reward_window)
        sp_active = False

    # Load snapshots from all directories
    snapshot_dir = args.snapshot_dir
    os.makedirs(snapshot_dir, exist_ok=True)
    for ld in args.load_dirs:
        os.makedirs(ld, exist_ok=True)
    all_startup_dirs = [snapshot_dir] + list(args.load_dirs)
    loaded_snapshots = load_snapshot_dirs(
        all_startup_dirs, args.hidden_dim, args.pfsp_pool_size,
        args.pfsp_reward_window)
    if loaded_snapshots:
        opponent_pool.extend(loaded_snapshots)
        if not args.exploiter_mode:
            sp_active = True
        print(f"  Loaded {len(loaded_snapshots)} snapshots from "
              f"{len(all_startup_dirs)} dir(s)")
    else:
        print(f"  No compatible snapshots found in "
              f"{len(all_startup_dirs)} dir(s)")

    # Exploiter: wait for opponents if pool is empty
    if args.exploiter_mode and len(opponent_pool) == 0:
        print("  Waiting for opponent snapshots to appear...")
        while len(opponent_pool) == 0:
            time.sleep(10)
            loaded = load_snapshot_dirs(
                all_startup_dirs, args.hidden_dim,
                args.pfsp_pool_size, args.pfsp_reward_window)
            if loaded:
                opponent_pool.extend(loaded)
                print(f"  Found {len(loaded)} snapshots, starting training")

    # Worker params for opponent composition
    worker_params = {
        "forced_rates": {
            "smart": args.forced_smart,
            "heuristic": args.forced_heuristic,
            "random": args.forced_random,
        },
        "homogeneous_rate": args.opp_homogeneous_rate,
        "primary_bias": args.opp_primary_bias,
        "nn_temperature": args.nn_temperature,
    }
    if args.exploiter_mode:
        worker_params["forced_rates"] = {}

    # Adaptive player count distribution
    pc_win_tracker = {pc: deque(maxlen=args.pc_win_window)
                      for pc in PLAYER_COUNTS}
    pc_base_weights = list(args.pc_base_weights)

    BOT_TYPES = ["smart", "heuristic", "random"]
    cell_win_tracker = {
        (bot, pc): deque(maxlen=args.forced_win_window)
        for bot in BOT_TYPES for pc in PLAYER_COUNTS
    }

    # Restore win tracking data from checkpoint
    if _resumed_checkpoint is not None:
        saved_pc = _resumed_checkpoint.get("pc_win_tracker", {})
        for pc in PLAYER_COUNTS:
            if pc in saved_pc:
                pc_win_tracker[pc].extend(saved_pc[pc])
        saved_pool_wins = _resumed_checkpoint.get("pool_win_data", {})
        for opp in opponent_pool:
            if opp["id"] in saved_pool_wins:
                for entry in saved_pool_wins[opp["id"]]:
                    opp["win_data"].append(tuple(entry))
        saved_cells = _resumed_checkpoint.get("cell_win_tracker", {})
        for key, data in saved_cells.items():
            if isinstance(key, tuple) and key in cell_win_tracker:
                cell_win_tracker[key].extend(data)
        _resumed_checkpoint = None  # free memory

    # Storage buffers (GPU)
    T = args.steps_per_rollout
    obs_buf = torch.zeros((T, N, obs_dim), dtype=torch.float32, device=device)
    mask_buf = torch.zeros((T, N, action_dim), dtype=torch.float32, device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    done_buf = torch.zeros((T, N), dtype=torch.float32, device=device)
    val_buf = torch.zeros((T, N), dtype=torch.float32, device=device)

    num_updates = args.total_timesteps // (T * N)
    start_time = time.time()
    _sps_history = deque(maxlen=20)  # (timestamp, step) for rolling SPS

    start_update = global_step // (T * N) + 1
    _priority_opponents = []

    def _select_and_send(update_num):
        """Select opponents via PFSP and send collect command to workers.

        Encapsulates: agent weight snapshot, snapshot-dir rescan, PFSP
        opponent selection, opp_configs building, and sending the collect
        command to all workers.

        Returns all_selected (list of opponent entries) on success,
        or None if no opponents are available (exploiter-mode skip).
        """
        nonlocal sp_active

        agent_sd = {k: v.cpu().clone() for k, v in network.state_dict().items()}

        # Pre-selection rescan: check for new snapshots before choosing opponents
        if (args.rescan_interval > 0
                and update_num % args.rescan_interval == 0):
            rescan_dirs = list(args.load_dirs)
            if args.snapshot_dir and not args.exploiter_mode:
                rescan_dirs.append(args.snapshot_dir)
            if rescan_dirs:
                existing_keys = {(e.get("run_tag", ""), e["step"])
                                 for e in opponent_pool
                                 if e["type"] == "snapshot"}
                new_entries = load_snapshot_dirs(
                    rescan_dirs, args.hidden_dim, args.pfsp_pool_size,
                    args.pfsp_reward_window, existing_keys)
                if new_entries:
                    snapshot_entries = [e for e in opponent_pool
                                       if e["type"] == "snapshot"]
                    total_after = len(snapshot_entries) + len(new_entries)
                    while total_after > args.pfsp_pool_size and snapshot_entries:
                        # Prefer evicting non-EXP entries so exploiter
                        # snapshots aren't crowded out by high-step MAIN entries
                        non_exp = [e for e in snapshot_entries
                                   if not e["id"].startswith("EXP")]
                        if non_exp:
                            oldest = min(non_exp, key=lambda e: e["step"])
                        else:
                            oldest = min(snapshot_entries,
                                         key=lambda e: e["step"])
                        opponent_pool.remove(oldest)
                        snapshot_entries.remove(oldest)
                        total_after -= 1
                    opponent_pool.extend(new_entries)
                    if not sp_active:
                        sp_active = True
                    # Queue exploiter snapshots for immediate play
                    for ne in new_entries:
                        if ne["id"].startswith("EXP"):
                            _priority_opponents.append(ne)
                    event_print(f"  -> Rescan: +{len(new_entries)} snapshots "
                               f"(pool: {len(opponent_pool)} total)")

        # PFSP opponent selection
        all_selected = []
        if args.exploiter_mode:
            # Exploiter: always play the latest snapshot only
            snapshots = [e for e in opponent_pool if e["type"] == "snapshot"]
            if snapshots:
                latest = max(snapshots, key=lambda e: e["step"])
                all_selected = [latest]
                latest["last_used"] = update_num
        else:
            # Force-play new exploiter snapshots first
            while _priority_opponents and len(all_selected) < 3:
                prio = _priority_opponents.pop(0)
                if any(e is prio for e in opponent_pool):
                    all_selected.append(prio)
                    prio["last_used"] = update_num

            # Reserve one slot for the latest exploiter snapshot.
            # Pick from the newest run (highest run_tag, which is a
            # base36 timestamp), then the highest step within that run.
            # This ensures a freshly started exploiter always gets the
            # dedicated slot over older runs with more accumulated steps.
            if not all_selected:
                exploiters = [e for e in opponent_pool
                              if e["id"].startswith("EXP")]
                if exploiters:
                    best_exp = max(exploiters,
                                   key=lambda e: (e.get("run_tag", ""),
                                                  e["step"]))
                    all_selected.append(best_exp)
                    best_exp["last_used"] = update_num

            # Fill remaining slots with PFSP-weighted selection
            remaining_pool = [e for e in opponent_pool if e not in all_selected]
            for _ in range(min(3 - len(all_selected), len(remaining_pool))):
                opp = pfsp_select(remaining_pool, update_num,
                                  args.pfsp_exploration_bonus,
                                  args.pfsp_staleness_divisor,
                                  args.pfsp_max_staleness_multiplier)
                if opp is None:
                    break
                all_selected.append(opp)
                opp["last_used"] = update_num
                remaining_pool = [e for e in remaining_pool if e is not opp]

        # Build opponent configs (index 0 = primary/hardest)
        opp_configs = []
        for opp in all_selected:
            if opp["type"] in ("random", "heuristic", "smart"):
                opp_configs.append((opp["type"], None))
            else:
                opp_configs.append(("nn", opp["state_dict"]))
                if not sp_active:
                    sp_active = True
                    print(f"  Self-play activated at step {global_step} (PFSP)")

        # Guard: empty pool (exploiter mode before rescan finds opponents)
        if not opp_configs:
            if args.exploiter_mode:
                event_print("  Warning: no opponents in pool, skipping update")
                return None
            else:
                opp_configs = [("heuristic", None)]

        # Send collect command to all workers
        for remote in parent_remotes:
            remote.send(("collect", agent_sd, opp_configs, T, worker_params))
        return all_selected

    # Async double-buffering: kick off first rollout collection
    _inflight_selected = _select_and_send(start_update)
    _collect_inflight = _inflight_selected is not None

    for update in range(start_update, num_updates + 1):
        if _shutdown_requested:
            event_print("[train] Shutdown requested, saving final checkpoint...")
            break

        # Handle exploiter mode skip (no opponents available)
        if _inflight_selected is None:
            global_step += T * N
            if update < num_updates and not _shutdown_requested:
                _inflight_selected = _select_and_send(update + 1)
                _collect_inflight = _inflight_selected is not None
            continue

        # Receive complete rollouts from all workers
        last_obs_parts = []
        last_mask_parts = []
        all_configs_used = set()
        all_config_rewards = {}     # cfg_idx -> [per-worker averages]
        all_config_pc_rewards = {}  # (cfg_idx, pc) -> [per-worker averages]
        all_config_win_data = {}    # cfg_idx_or_str -> [(adjusted, raw_won)]
        all_pc_wins = {}            # pc -> [raw_won]
        all_forced_cell_wins = {}   # (bot_type_str, pc) -> [raw_won]
        for w, remote in enumerate(parent_remotes):
            (r_obs, r_mask, r_act, r_logp, r_val,
             r_rew, r_done, last_obs, last_mask,
             w_configs_used, w_config_rewards,
             w_config_pc_rewards,
             w_config_win_data, w_pc_wins,
             w_forced_cell_wins) = remote.recv()

            s = w * envs_per_worker
            e = s + envs_per_worker
            obs_buf[:, s:e, :] = torch.from_numpy(r_obs).to(device)
            mask_buf[:, s:e, :] = torch.from_numpy(r_mask).to(device)
            act_buf[:, s:e] = torch.from_numpy(r_act).to(device)
            logp_buf[:, s:e] = torch.from_numpy(r_logp).to(device)
            val_buf[:, s:e] = torch.from_numpy(r_val).to(device)
            rew_buf[:, s:e] = torch.from_numpy(r_rew).to(device)
            done_buf[:, s:e] = torch.from_numpy(r_done).to(device)
            last_obs_parts.append(last_obs)
            last_mask_parts.append(last_mask)
            all_configs_used.update(w_configs_used)
            for cfg_idx, avg_r in w_config_rewards.items():
                all_config_rewards.setdefault(cfg_idx, []).append(avg_r)
            for key, avg_r in w_config_pc_rewards.items():
                all_config_pc_rewards.setdefault(key, []).append(avg_r)
            for key, win_vals in w_config_win_data.items():
                all_config_win_data.setdefault(key, []).extend(win_vals)
            for pc, win_vals in w_pc_wins.items():
                all_pc_wins.setdefault(pc, []).extend(win_vals)
            for key, win_vals in w_forced_cell_wins.items():
                all_forced_cell_wins.setdefault(key, []).extend(win_vals)

        # Aggregate per-config rewards across workers
        final_config_rewards = {
            k: sum(v) / len(v) for k, v in all_config_rewards.items()
        }

        current_obs = torch.from_numpy(
            np.concatenate(last_obs_parts)).to(device)
        current_masks = torch.from_numpy(
            np.concatenate(last_mask_parts)).to(device)

        rollout_selected = _inflight_selected
        _collect_inflight = False

        # Async double-buffering: send next collect immediately.
        # Workers start collecting the next rollout while GPU trains on this one.
        if update < num_updates and not _shutdown_requested:
            _inflight_selected = _select_and_send(update + 1)
            _collect_inflight = _inflight_selected is not None

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr = frac * args.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Compute GAE
        with torch.no_grad():
            next_value = network.get_value(current_obs, current_masks)

        advantages = torch.zeros_like(rew_buf)
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - done_buf[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - done_buf[t]
                next_val = val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_val * next_non_terminal - val_buf[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
        returns = advantages + val_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_masks = mask_buf.reshape(-1, action_dim)
        b_actions = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = val_buf.reshape(-1)

        # PPO update
        batch_size = T * N
        indices = np.arange(batch_size)

        clip_losses = []
        value_losses = []
        entropy_losses = []

        for epoch in range(args.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_idx = indices[start:end]

                _, new_logp, entropy, new_value = network.get_action_and_value(
                    b_obs[mb_idx], b_masks[mb_idx], b_actions[mb_idx]
                )

                logratio = new_logp - b_logp[mb_idx]
                ratio = logratio.exp()

                # Per-minibatch advantage normalization
                mb_adv = b_advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate objective
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss
                v_unclipped = (new_value - b_returns[mb_idx]) ** 2
                v_clipped = b_values[mb_idx] + torch.clamp(
                    new_value - b_values[mb_idx], -args.clip_eps, args.clip_eps)
                v_clipped_loss = (v_clipped - b_returns[mb_idx]) ** 2
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                optimizer.step()

                clip_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(ent_loss.item())

        global_step += T * N

        # Logging
        avg_reward = rew_buf.sum(dim=0).mean().item()
        writer.add_scalar("train/avg_episode_reward", avg_reward, global_step)
        writer.add_scalar("train/policy_loss", np.mean(clip_losses), global_step)
        writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
        writer.add_scalar("train/entropy", np.mean(entropy_losses), global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        # Update PFSP reward (kept for diagnostics)
        for i, opp in enumerate(rollout_selected):
            if i in final_config_rewards:
                opp["rewards"].append(final_config_rewards[i])

        # Update PFSP win data (used for opponent selection)
        for i, opp in enumerate(rollout_selected):
            if i in all_config_win_data:
                entries = all_config_win_data[i]
                sum_adj = sum(a for a, _ in entries)
                sum_raw = sum(w for _, w in entries)
                opp["win_data"].append((sum_adj, sum_raw, len(entries)))

        # Forced bot tables: attribute win data to fixed bot entries directly
        if not args.exploiter_mode:
            for opp in opponent_pool[:3]:  # first 3 are always fixed bots
                if opp["type"] in all_config_win_data:
                    entries = all_config_win_data[opp["type"]]
                    sum_adj = sum(a for a, _ in entries)
                    sum_raw = sum(w for _, w in entries)
                    opp["win_data"].append((sum_adj, sum_raw, len(entries)))

        # Update per-player-count win tracker and compute adaptive weights
        for pc, wins in all_pc_wins.items():
            pc_win_tracker[pc].extend(wins)

        min_pc_samples = min(10, args.pc_win_window)
        pc_win_rates = {}
        all_pc_have_data = True
        for pc in PLAYER_COUNTS:
            if len(pc_win_tracker[pc]) >= min_pc_samples:
                pc_win_rates[pc] = sum(pc_win_tracker[pc]) / len(pc_win_tracker[pc])
            else:
                all_pc_have_data = False

        if all_pc_have_data:
            weakness = [1.0 - pc_win_rates[pc] for pc in PLAYER_COUNTS]
            w_sum = sum(weakness) or 1.0
            weakness_norm = [w / w_sum for w in weakness]
            alpha = args.pc_adaptation_rate
            adaptive_weights = [
                (1 - alpha) * base + alpha * weak
                for base, weak in zip(pc_base_weights, weakness_norm)
            ]
            adaptive_weights = [max(w, args.pc_min_weight) for w in adaptive_weights]
            total_w = sum(adaptive_weights)
            adaptive_weights = [w / total_w for w in adaptive_weights]
            worker_params["player_count_weights"] = adaptive_weights
        else:
            worker_params["player_count_weights"] = None

        # Update per-(bot, pc) cell win tracker and compute adaptive forced rates
        for cell_key, wins in all_forced_cell_wins.items():
            if cell_key in cell_win_tracker:
                cell_win_tracker[cell_key].extend(wins)

        total_forced_budget = (args.forced_smart + args.forced_heuristic
                               + args.forced_random)
        min_cell_samples = min(10, args.forced_win_window)
        cell_win_rates = {}
        all_cells_have_data = True
        for key in cell_win_tracker:
            if len(cell_win_tracker[key]) >= min_cell_samples:
                cell_win_rates[key] = (sum(cell_win_tracker[key])
                                       / len(cell_win_tracker[key]))
            else:
                all_cells_have_data = False

        if all_cells_have_data and total_forced_budget > 0:
            # Weakness weights: lower win rate = higher weakness
            cell_weakness = {k: 1.0 - v for k, v in cell_win_rates.items()}
            total_weakness = sum(cell_weakness.values()) or 1.0
            adaptive_cell = {k: total_forced_budget * w / total_weakness
                             for k, w in cell_weakness.items()}

            # Base cell rates (mimics old flat system, spread by PC weights)
            base_bot = {"smart": args.forced_smart,
                        "heuristic": args.forced_heuristic,
                        "random": args.forced_random}
            current_pc_w = (worker_params.get("player_count_weights")
                            or pc_base_weights)
            base_cell = {(bot, pc): base_bot[bot] * current_pc_w[i]
                         for bot in BOT_TYPES
                         for i, pc in enumerate(PLAYER_COUNTS)}
            base_sum = sum(base_cell.values())
            if base_sum > 0:
                base_cell = {k: v * total_forced_budget / base_sum
                             for k, v in base_cell.items()}

            # Blend, enforce min, renormalize to budget
            alpha_f = args.forced_adaptation_rate
            blended = {k: (1 - alpha_f) * base_cell[k] + alpha_f * adaptive_cell[k]
                       for k in cell_win_tracker}
            blended = {k: max(v, args.forced_min_cell_rate)
                       for k, v in blended.items()}
            total_b = sum(blended.values())
            if total_b > 0:
                blended = {k: v * total_forced_budget / total_b
                           for k, v in blended.items()}

            # Convert to conditional rates: P(forced_as_bot | PC=pc) = cell_rate / pc_weight
            forced_pc_bot_rates = {}
            for (bot, pc), rate in blended.items():
                pw = current_pc_w[PLAYER_COUNTS.index(pc)]
                cond = min(rate / pw, 0.5) if pw > 1e-9 else 0.0
                forced_pc_bot_rates.setdefault(pc, {})[bot] = cond
            worker_params["forced_pc_bot_rates"] = forced_pc_bot_rates
        else:
            worker_params["forced_pc_bot_rates"] = None

        _sps_history.append((time.time(), global_step))
        if len(_sps_history) >= 2:
            dt = _sps_history[-1][0] - _sps_history[0][0]
            ds = _sps_history[-1][1] - _sps_history[0][1]
            sps = int(ds / dt) if dt > 0 else 0
        else:
            sps = int((T * N) / max(time.time() - start_time, 0.01))
        opp_str = "+".join(opp["id"] for opp in rollout_selected)
        dashboard_text = format_dashboard(
            global_step, sp_active, opp_str, avg_reward,
            np.mean(clip_losses), np.mean(value_losses), np.mean(entropy_losses), sps,
            opponent_pool, cell_win_rates, cell_win_tracker,
            pc_win_tracker, pc_win_rates, pc_base_weights,
            worker_params, min_pc_samples, writer,
            exploiter_mode=args.exploiter_mode)
        dashboard_print(dashboard_text)

        # Write status.json for dashboard GUI (atomic write)
        _pool_info = []
        for _e in opponent_pool:
            _wd = _e.get("win_data", [])
            _total_raw = sum(rw for _, rw, _ in _wd) if _wd else 0
            _total_games = sum(g for _, _, g in _wd) if _wd else 0
            _pool_info.append({
                "id": _e["id"], "type": _e["type"],
                "step": _e.get("step", 0),
                "win_rate": _total_raw / _total_games if _total_games > 0 else None,
                "games": _total_games,
            })
        _status = {
            "global_step": global_step, "sps": sps,
            "total_timesteps": args.total_timesteps,
            "avg_reward": float(avg_reward),
            "pg_loss": float(np.mean(clip_losses)),
            "v_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "lr": optimizer.param_groups[0]["lr"],
            "sp_active": sp_active,
            "pc_win_rates": {str(pc): pc_win_rates.get(pc) for pc in PLAYER_COUNTS},
            "cell_win_rates": {f"{bk}_{pc}": cell_win_rates.get((bk, pc))
                               for bk in ("smart", "heuristic", "random")
                               for pc in PLAYER_COUNTS},
            "opponent_pool": _pool_info,
            "run_name": run_name,
            "timestamp": time.time(),
        }
        _status_path = os.path.join(writer.log_dir, "status.json")
        try:
            with open(_status_path + ".tmp", "w") as _sf:
                json.dump(_status, _sf)
            os.replace(_status_path + ".tmp", _status_path)
        except OSError:
            pass

        # Snapshot — save to disk (always) and add to own pool (normal mode only)
        if (global_step >= args.self_play_start
                and global_step % args.snapshot_interval < T * N):
            snap_sd = {k: v.cpu().clone() for k, v in network.state_dict().items()}

            # Persist to disk with atomic rename (safe for cross-process reads)
            snap_path = os.path.join(snapshot_dir, f"{run_name}_{_step_tag(global_step)}.pt")
            snap_tmp = snap_path + ".tmp"
            torch.save({
                "state_dict": snap_sd,
                "hidden_dim": args.hidden_dim,
                "step": global_step,
            }, snap_tmp)
            os.replace(snap_tmp, snap_path)

            if not args.exploiter_mode:
                # Add to own pool (self-play)
                snap_entry = {
                    "id": f"snap_{_step_tag(global_step)}", "type": "snapshot",
                    "state_dict": snap_sd, "step": global_step,
                    "rewards": deque(maxlen=args.pfsp_reward_window),
                    "win_data": deque(maxlen=args.pfsp_reward_window),
                    "last_used": 0,
                }
                snapshot_entries = [e for e in opponent_pool
                                    if e["type"] == "snapshot"]
                if len(snapshot_entries) >= args.pfsp_pool_size:
                    non_exp = [e for e in snapshot_entries
                               if not e["id"].startswith("EXP")]
                    if non_exp:
                        oldest = min(non_exp, key=lambda e: e["step"])
                    else:
                        oldest = min(snapshot_entries,
                                     key=lambda e: e["step"])
                    opponent_pool.remove(oldest)
                opponent_pool.append(snap_entry)

            snap_count = len([e for e in opponent_pool if e["type"] == "snapshot"])
            event_print(f"  -> Saved snapshot {snap_path} "
                       f"(pool: {len(opponent_pool)} total, {snap_count} snapshots)")

        # Periodic evaluation across all player counts (skip for exploiter)
        if not args.exploiter_mode and global_step % args.eval_interval < T * N:
            network.eval()
            event_print(f"  Evaluating at step {global_step} (all player counts)...")

            csv_row = [global_step]
            csv_row_aggs = {}
            eval_results = {}
            for opp_type in ["random", "heuristic", "smart"]:
                all_metrics, agg = evaluate_all_configs(
                    network, device, args.eval_games, opp_type
                )
                eval_results[opp_type] = all_metrics
                for np_ in PLAYER_COUNTS:
                    m = all_metrics[np_]
                    csv_row += [m["avg_score"], m["win_rate"], m["bid_accuracy"], m["avg_winning_score"]]
                    writer.add_scalar(f"eval/{np_}p_vs_{opp_type}_score", m["avg_score"], global_step)
                    writer.add_scalar(f"eval/{np_}p_vs_{opp_type}_win", m["win_rate"], global_step)
                    writer.add_scalar(f"eval/{np_}p_vs_{opp_type}_bid_acc", m["bid_accuracy"], global_step)
                    writer.add_scalar(f"eval/{np_}p_vs_{opp_type}_win_score", m["avg_winning_score"], global_step)

                csv_row_aggs[opp_type] = [agg["avg_score"], agg["win_rate"], agg["bid_accuracy"], agg["avg_winning_score"]]

                writer.add_scalar(f"eval/agg_vs_{opp_type}_score", agg["avg_score"], global_step)
                writer.add_scalar(f"eval/agg_vs_{opp_type}_win", agg["win_rate"], global_step)
                writer.add_scalar(f"eval/agg_vs_{opp_type}_bid_acc", agg["bid_accuracy"], global_step)
                writer.add_scalar(f"eval/agg_vs_{opp_type}_win_score", agg["avg_winning_score"], global_step)

                event_print(f"    vs {opp_type.capitalize():10s} (agg): "
                            f"score={agg['avg_score']:.1f} win={agg['win_rate']:.2f} "
                            f"bid_acc={agg['bid_accuracy']:.2f} win_score={agg['avg_winning_score']:.1f}")
                for np_ in PLAYER_COUNTS:
                    m = all_metrics[np_]
                    event_print(f"      {np_}p: score={m['avg_score']:.1f} "
                                f"win={m['win_rate']:.2f} bid_acc={m['bid_accuracy']:.2f} "
                                f"win_score={m['avg_winning_score']:.1f} "
                                f"[{m['min_winning_score']}-{m['max_winning_score']}]")

            # Weakness ranking (all opponent x player_count combos by win rate)
            weakness = []
            for opp_r, all_m in eval_results.items():
                for np_ in PLAYER_COUNTS:
                    m = all_m[np_]
                    weakness.append((opp_r, np_, m["win_rate"],
                                     m["bid_accuracy"], m["avg_score"]))
            weakness.sort(key=lambda x: x[2])
            event_print("    Weakness Ranking (by win rate):")
            for rank, (opp_r, np_, wr, ba, sc) in enumerate(weakness[:5], 1):
                event_print(f"      {rank}. vs {opp_r.capitalize():<10s} {np_}p — "
                            f"win={wr:.2f} bid_acc={ba:.2f} score={sc:.1f}")

            for opp_type in ["random", "heuristic", "smart"]:
                csv_row += csv_row_aggs[opp_type]
            csv_writer.writerow(csv_row)
            csv_file.flush()
            network.train()

        # Save checkpoint
        if global_step % args.checkpoint_interval < T * N:
            path = os.path.join(args.save_dir, f"{run_name}_CHKPT_{_step_tag(global_step)}.pt")
            ckpt = {
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "hidden_dim": args.hidden_dim,
                "pc_win_tracker": {pc: list(tracker) for pc, tracker in pc_win_tracker.items()},
                "pool_win_data": ({opp["id"]: [list(e) for e in opp["win_data"]]
                                   for opp in opponent_pool[:3]}
                                  if not args.exploiter_mode else {}),
                "cell_win_tracker": {k: list(v) for k, v in cell_win_tracker.items()},
            }
            torch.save(ckpt, path)
            event_print(f"  -> Checkpoint saved: {path}")

    # Drain any inflight rollout that workers are still collecting
    # (happens when shutdown is requested while workers are mid-collect)
    if _collect_inflight:
        for remote in parent_remotes:
            try:
                remote.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass

    # Cleanup workers
    for remote in parent_remotes:
        try:
            remote.send(None)
        except (BrokenPipeError, ConnectionResetError):
            pass
    for p in workers:
        p.join(timeout=5)

    # Final save
    final_path = os.path.join(args.save_dir, f"{run_name}_CHKPT_FINAL.pt")
    final_ckpt = {
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "hidden_dim": args.hidden_dim,
        "pc_win_tracker": {pc: list(tracker) for pc, tracker in pc_win_tracker.items()},
        "pool_win_data": ({opp["id"]: [list(e) for e in opp["win_data"]]
                           for opp in opponent_pool[:3]}
                          if not args.exploiter_mode else {}),
        "cell_win_tracker": {k: list(v) for k, v in cell_win_tracker.items()},
    }
    torch.save(final_ckpt, final_path)
    print(f"Training complete. Final model saved to {final_path}")

    csv_file.close()
    writer.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down.")
