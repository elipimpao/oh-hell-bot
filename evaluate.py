"""Evaluation and plotting for Oh Hell bot training.

Supports cross-configuration evaluation across all player counts (2-5).
"""

import os
import csv
import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from game import OhHellGame, Phase
from env import OhHellEnv
from network import OhHellNetwork
from bots import RandomBot, HeuristicBot, SmartBot
from train import evaluate_all_configs, PLAYER_COUNTS


def evaluate_checkpoint(checkpoint_path, num_games=50, device="cpu"):
    """Evaluate a trained checkpoint across all player counts."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    obs_dim = OhHellEnv(num_players=4, opponents=[RandomBot()] * 3).obs_size
    hidden_dim = checkpoint.get("hidden_dim", 256)

    network = OhHellNetwork(obs_dim, hidden_dim=hidden_dim).to(device)
    network.load_state_dict(checkpoint["network"])
    network.eval()

    results = {}
    for opp_type in ["random", "heuristic", "smart"]:
        all_metrics, agg = evaluate_all_configs(network, device, num_games, opp_type)
        results[opp_type] = {"per_config": all_metrics, "aggregate": agg}

    return results


def plot_training(csv_path, output_path="training_progress.png"):
    """Generate training progress plots from CSV log.

    Creates a 3x3 grid: rows = (score, win_rate, bid_acc), cols = (vs_random, vs_heuristic, vs_smart).
    Each subplot has lines for 2p, 3p, 4p, 5p and the aggregate.
    """
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(val))

    steps = data["step"]

    colors = {2: "tab:red", 3: "tab:green", 4: "tab:blue", 5: "tab:purple"}
    metrics = ["score", "win", "bid_acc"]
    metric_labels = ["Average Score", "Win Rate", "Bid Accuracy"]
    opponents = ["random", "heuristic", "smart"]

    fig, axes = plt.subplots(3, 3, figsize=(22, 12), sharex=True)

    for col, opp in enumerate(opponents):
        for row, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]

            # Per player count
            for np_ in PLAYER_COUNTS:
                key = f"{np_}p_vs_{opp}_{metric}"
                if key in data:
                    ax.plot(steps, data[key], label=f"{np_}p",
                            color=colors[np_], linewidth=1.5, alpha=0.7)

            # Aggregate
            agg_key = f"agg_vs_{opp}_{metric}"
            if agg_key in data:
                ax.plot(steps, data[agg_key], label="Aggregate",
                        color="black", linewidth=2.5, linestyle="--")

            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"vs {opp.capitalize()}")
            if row == 2:
                ax.set_xlabel("Training Steps")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Oh Hell Bot Training Progress (All Player Counts)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Oh Hell bot")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--plot", type=str, default=None,
                        help="Path to eval_log.csv to generate plot")
    parser.add_argument("--output", type=str, default="training_progress.png")
    parser.add_argument("--num-games", type=int, default=50,
                        help="Games per player-count per opponent-type")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.plot:
        plot_training(args.plot, args.output)
    elif args.checkpoint:
        results = evaluate_checkpoint(args.checkpoint, args.num_games, args.device)

        print(f"\nEvaluation Results ({args.num_games} games per config per opponent):\n")
        for opp_type in ["random", "heuristic", "smart"]:
            r = results[opp_type]
            print(f"  vs {opp_type.capitalize()}:")
            print(f"  {'Config':<10} {'Avg Score':>10} {'Win Rate':>10} {'Bid Acc':>10} {'Win Score':>10} {'Range':>12}")
            print(f"  {'-'*67}")
            for np_ in PLAYER_COUNTS:
                m = r["per_config"][np_]
                print(f"  {f'{np_} players':<10} {m['avg_score']:>10.1f} "
                      f"{m['win_rate']:>10.2%} {m['bid_accuracy']:>10.2%} "
                      f"{m['avg_winning_score']:>10.1f} "
                      f"[{m['min_winning_score']}-{m['max_winning_score']}]")
            agg = r["aggregate"]
            print(f"  {'AGGREGATE':<10} {agg['avg_score']:>10.1f} "
                  f"{agg['win_rate']:>10.2%} {agg['bid_accuracy']:>10.2%} "
                  f"{agg['avg_winning_score']:>10.1f} "
                  f"[{agg['min_winning_score']}-{agg['max_winning_score']}]")
            print()

        # Weakness ranking across all opponent x player_count combos
        weakness = []
        for opp_type in ["random", "heuristic", "smart"]:
            r = results[opp_type]
            for np_ in PLAYER_COUNTS:
                m = r["per_config"][np_]
                weakness.append((opp_type, np_, m["win_rate"],
                                 m["bid_accuracy"], m["avg_score"]))
        weakness.sort(key=lambda x: x[2])
        print("Weakness Ranking (by win rate, ascending):\n")
        for rank, (opp, np_, wr, ba, sc) in enumerate(weakness, 1):
            print(f"  {rank:>2}. vs {opp.capitalize():<10s} {np_}p — "
                  f"win={wr:.2%}  bid_acc={ba:.2%}  score={sc:.1f}")
        print()
    else:
        print("Provide --checkpoint or --plot. Run with --help for usage.")


if __name__ == "__main__":
    main()
