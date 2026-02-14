"""Oh Hell Engine GUI — enter your real game state and get move recommendations.

Uses tkinter (built into Python). Load a trained checkpoint and use the
neural network to recommend bids and card plays for any game position.
"""

import sys
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import torch
import numpy as np

from game import OhHellGame, Phase, Card, Suit, card_str, card_index, index_to_card
from game import SUIT_NAMES, RANK_NAMES
from env import OhHellEnv, MAX_PLAYERS
from network import OhHellNetwork
from train import get_obs_and_mask_for_bot


SUIT_COLORS = {0: "#2E7D32", 1: "#1565C0", 2: "#C62828", 3: "#37474F"}
SUIT_SYMBOLS = ["♣", "♦", "♥", "♠"]


class OhHellGUI:
    def __init__(self, root, checkpoint_path=None, device="cpu"):
        self.root = root
        self.root.title("Oh Hell Engine")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        self.device = torch.device(device)
        self.network = None
        self.game = None
        self.my_seat = 0

        # State for manual game construction
        self.selected_hand = set()  # set of Card
        self.trump_suit = None
        self.num_players = tk.IntVar(value=4)
        self.dealer = tk.IntVar(value=0)
        self.hand_size = tk.IntVar(value=0)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0",
                        font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1a1a2e", foreground="#e0e0e0",
                        font=("Segoe UI", 13, "bold"))
        style.configure("Rec.TLabel", background="#1a1a2e", foreground="#00e676",
                        font=("Segoe UI", 12, "bold"))
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("Trump.TButton", font=("Segoe UI", 12, "bold"))
        style.configure("TLabelframe", background="#1a1a2e", foreground="#e0e0e0",
                        font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("TSpinbox", font=("Segoe UI", 10))

        self._build_ui()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            obs_dim = OhHellEnv(num_players=4, opponents=[]).obs_size
            hidden_dim = checkpoint.get("hidden_dim", 256)
            self.network = OhHellNetwork(obs_dim, hidden_dim=hidden_dim).to(self.device)
            self.network.load_state_dict(checkpoint["network"])
            self.network.eval()
            step = checkpoint.get("global_step", "?")
            self.status_var.set(f"Model loaded (step {step})")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load checkpoint:\n{e}")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Top bar: load model + game setup
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(top, text="Load Model...", command=self._browse_model).pack(side=tk.LEFT)
        ttk.Label(top, text="  Players:").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=2, to=5, width=3, textvariable=self.num_players,
                     command=self._on_settings_change).pack(side=tk.LEFT)
        ttk.Label(top, text="  Your seat:").pack(side=tk.LEFT)
        self.seat_spin = ttk.Spinbox(top, from_=0, to=4, width=3,
                                      command=self._on_settings_change)
        self.seat_spin.pack(side=tk.LEFT)
        self.seat_spin.set("0")
        ttk.Label(top, text="  Dealer:").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=0, to=4, width=3, textvariable=self.dealer,
                     command=self._on_settings_change).pack(side=tk.LEFT)

        # Trump selector
        trump_frame = ttk.LabelFrame(main, text="Trump Suit", padding=5)
        trump_frame.pack(fill=tk.X, pady=(0, 8))

        self.trump_buttons = []
        for s in range(4):
            btn = tk.Button(trump_frame, text=SUIT_SYMBOLS[s], font=("Segoe UI", 16, "bold"),
                            fg=SUIT_COLORS[s], bg="#2a2a4e", activebackground="#3a3a6e",
                            width=3, relief=tk.RAISED, bd=2,
                            command=lambda suit=s: self._set_trump(suit))
            btn.pack(side=tk.LEFT, padx=4)
            self.trump_buttons.append(btn)

        # Card picker
        card_frame = ttk.LabelFrame(main, text="Your Hand (click to toggle)", padding=5)
        card_frame.pack(fill=tk.X, pady=(0, 8))

        self.card_buttons = {}
        for s in range(4):
            row = ttk.Frame(card_frame)
            row.pack(fill=tk.X, pady=1)
            lbl = tk.Label(row, text=SUIT_SYMBOLS[s], font=("Segoe UI", 14, "bold"),
                          fg=SUIT_COLORS[s], bg="#1a1a2e", width=2)
            lbl.pack(side=tk.LEFT)
            for r in range(13):
                card = Card(suit=s, rank=r)
                btn = tk.Button(row, text=RANK_NAMES[r], font=("Consolas", 9),
                                width=3, fg=SUIT_COLORS[s], bg="#2a2a4e",
                                activebackground="#3a3a6e", relief=tk.RAISED, bd=1,
                                command=lambda c=card: self._toggle_card(c))
                btn.pack(side=tk.LEFT, padx=1)
                self.card_buttons[card] = btn

        self.hand_label = ttk.Label(main, text="Hand: (none selected)")
        self.hand_label.pack(fill=tk.X)

        # Bidding section
        bid_frame = ttk.LabelFrame(main, text="Bidding", padding=5)
        bid_frame.pack(fill=tk.X, pady=(8, 4))

        self.bid_entries = []
        bid_row = ttk.Frame(bid_frame)
        bid_row.pack(fill=tk.X)
        for i in range(5):
            lbl = ttk.Label(bid_row, text=f"P{i}:")
            lbl.pack(side=tk.LEFT, padx=(8, 0))
            entry = ttk.Entry(bid_row, width=4)
            entry.pack(side=tk.LEFT, padx=(0, 4))
            self.bid_entries.append((lbl, entry))

        # Current trick section
        trick_frame = ttk.LabelFrame(main, text="Current Trick (cards played)", padding=5)
        trick_frame.pack(fill=tk.X, pady=(4, 4))

        self.trick_entries = []
        trick_row = ttk.Frame(trick_frame)
        trick_row.pack(fill=tk.X)
        for i in range(5):
            lbl = ttk.Label(trick_row, text=f"P{i}:")
            lbl.pack(side=tk.LEFT, padx=(8, 0))
            entry = ttk.Entry(trick_row, width=6)
            entry.pack(side=tk.LEFT, padx=(0, 4))
            self.trick_entries.append((lbl, entry))

        ttk.Label(trick_frame,
                  text="Enter cards as rank+suit: e.g. AS (Ace Spades), 10H (10 Hearts), KC (King Clubs)").pack(
            fill=tk.X, pady=(4, 0))

        # Tricks won section
        tricks_won_frame = ttk.LabelFrame(main, text="Tricks Won This Round", padding=5)
        tricks_won_frame.pack(fill=tk.X, pady=(4, 4))

        self.tricks_won_entries = []
        tw_row = ttk.Frame(tricks_won_frame)
        tw_row.pack(fill=tk.X)
        for i in range(5):
            lbl = ttk.Label(tw_row, text=f"P{i}:")
            lbl.pack(side=tk.LEFT, padx=(8, 0))
            entry = ttk.Entry(tw_row, width=4)
            entry.insert(0, "0")
            entry.pack(side=tk.LEFT, padx=(0, 4))
            self.tricks_won_entries.append((lbl, entry))

        # Cards seen section
        seen_frame = ttk.LabelFrame(main, text="Cards Already Played This Round (optional)", padding=5)
        seen_frame.pack(fill=tk.X, pady=(4, 4))

        self.seen_text = tk.Text(seen_frame, height=2, font=("Consolas", 9),
                                 bg="#2a2a4e", fg="#e0e0e0", insertbackground="#e0e0e0")
        self.seen_text.pack(fill=tk.X)
        ttk.Label(seen_frame,
                  text="Space-separated: e.g. AS 10H KC 2D").pack(fill=tk.X, pady=(2, 0))

        # Trump broken checkbox
        self.trump_broken_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main, text="Trump has been broken (played in a previous trick)",
                        variable=self.trump_broken_var).pack(fill=tk.X, pady=(4, 4))

        # Action buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(8, 4))

        self.rec_bid_btn = ttk.Button(btn_frame, text="Get Bid Recommendation",
                                       command=self._recommend_bid)
        self.rec_bid_btn.pack(side=tk.LEFT, padx=4)

        self.rec_play_btn = ttk.Button(btn_frame, text="Get Play Recommendation",
                                        command=self._recommend_play)
        self.rec_play_btn.pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_frame, text="Clear All", command=self._clear_all).pack(side=tk.RIGHT, padx=4)

        # Recommendation display
        rec_frame = ttk.LabelFrame(main, text="Recommendation", padding=8)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.rec_label = ttk.Label(rec_frame, text="Load a model and enter your game state.",
                                    style="Rec.TLabel", wraplength=700)
        self.rec_label.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="No model loaded")
        status = ttk.Label(main, textvariable=self.status_var, foreground="#888888")
        status.pack(fill=tk.X, pady=(8, 0))

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select trained checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialdir="checkpoints"
        )
        if path:
            self._load_checkpoint(path)

    def _set_trump(self, suit):
        self.trump_suit = suit
        for i, btn in enumerate(self.trump_buttons):
            if i == suit:
                btn.config(relief=tk.SUNKEN, bg="#4a4a8e")
            else:
                btn.config(relief=tk.RAISED, bg="#2a2a4e")

    def _toggle_card(self, card):
        btn = self.card_buttons[card]
        if card in self.selected_hand:
            self.selected_hand.remove(card)
            btn.config(bg="#2a2a4e", relief=tk.RAISED)
        else:
            self.selected_hand.add(card)
            btn.config(bg="#4CAF50", relief=tk.SUNKEN)

        hand_str = " ".join(card_str(c) for c in sorted(self.selected_hand, key=lambda c: (c.suit, c.rank)))
        self.hand_label.config(text=f"Hand ({len(self.selected_hand)} cards): {hand_str or '(none)'}")

    def _on_settings_change(self):
        pass  # Could update visibility of player entries

    def _parse_card_str(self, s):
        """Parse a card string like 'AS', '10H', 'KC', '2D' into a Card."""
        s = s.strip().upper()
        if not s:
            return None

        suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
        rank_map = {n.upper(): i for i, n in enumerate(RANK_NAMES)}

        suit_char = s[-1]
        rank_str = s[:-1]

        if suit_char not in suit_map:
            raise ValueError(f"Unknown suit '{suit_char}' in '{s}'")
        if rank_str not in rank_map:
            raise ValueError(f"Unknown rank '{rank_str}' in '{s}'")

        return Card(suit=suit_map[suit_char], rank=rank_map[rank_str])

    def _build_game_state(self, phase):
        """Construct a game object from the GUI inputs."""
        np_ = self.num_players.get()
        seat = int(self.seat_spin.get())
        dealer = self.dealer.get()
        hand_size = len(self.selected_hand)

        if not self.selected_hand:
            raise ValueError("Select your hand cards first.")
        if self.trump_suit is None:
            raise ValueError("Select a trump suit.")
        if np_ < 2 or np_ > 5:
            raise ValueError("Player count must be 2-5.")

        # Create a game object and manually set its state
        import random as rng_mod
        game = OhHellGame(np_, rng=rng_mod.Random(0))

        # Override game state
        game.dealer = dealer
        game.hand_size = hand_size
        game.trump_suit = self.trump_suit
        game.trump_card = self.trump_suit * 13  # placeholder (rank 0 of trump suit)
        game.trump_broken = self.trump_broken_var.get()

        # Set hands — we only know our hand
        game.hands = [[] for _ in range(np_)]
        game.hands[seat] = sorted(card_index(c) for c in self.selected_hand)

        # Bids
        game.bids = [-1] * np_
        for i in range(np_):
            if i < len(self.bid_entries):
                _, entry = self.bid_entries[i]
                val = entry.get().strip()
                if val:
                    game.bids[i] = int(val)

        # Tricks won
        game.tricks_won = [0] * np_
        for i in range(np_):
            if i < len(self.tricks_won_entries):
                _, entry = self.tricks_won_entries[i]
                val = entry.get().strip()
                if val:
                    game.tricks_won[i] = int(val)

        # Cards seen
        game.cards_played_this_round = set()
        seen_text = self.seen_text.get("1.0", tk.END).strip()
        if seen_text:
            for cs in seen_text.split():
                card = self._parse_card_str(cs)
                if card:
                    game.cards_played_this_round.add(card_index(card))

        # Current trick
        game.current_trick = []
        for i in range(np_):
            if i < len(self.trick_entries):
                _, entry = self.trick_entries[i]
                val = entry.get().strip()
                if val:
                    card = self._parse_card_str(val)
                    if card:
                        ci = card_index(card)
                        game.current_trick.append((i, ci))
                        game.cards_played_this_round.add(ci)

        game.phase = Phase.BIDDING if phase == "bid" else Phase.PLAYING

        # Set trick leader and bidding state
        game.trick_leader = (dealer + 1) % np_
        game.bid_order = [(dealer + 1 + i) % np_ for i in range(np_)]
        bids_placed = sum(1 for b in game.bids if b >= 0)
        game.current_bidder_index = bids_placed
        game.trick_number = sum(game.tricks_won)

        # Round index (for game state, not used for inference)
        game.round_index = 0

        return game, seat

    def _get_recommendations(self, game, seat, phase):
        """Get the network's action probabilities."""
        if self.network is None:
            raise ValueError("No model loaded. Click 'Load Model...' first.")

        obs, mask = get_obs_and_mask_for_bot(game, seat)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.network(obs_t, mask_t)
            # Apply mask
            masked = logits.masked_fill(mask_t == 0, float("-inf"))
            probs = torch.softmax(masked, dim=-1).squeeze(0).cpu().numpy()

        return probs, value.item()

    def _recommend_bid(self):
        try:
            game, seat = self._build_game_state("bid")
            probs, value = self._get_recommendations(game, seat, "bid")

            # Extract bid probabilities (actions 52+)
            bid_probs = []
            legal = game.get_legal_bids(seat)
            for b in range(len(self.selected_hand) + 1):
                if b in legal:
                    p = probs[52 + b]
                    bid_probs.append((b, p))

            bid_probs.sort(key=lambda x: -x[1])

            lines = ["BID RECOMMENDATIONS:\n"]
            for bid, prob in bid_probs:
                bar = "█" * int(prob * 30)
                lines.append(f"  Bid {bid:>2d}:  {prob:>6.1%}  {bar}")

            best_bid = bid_probs[0][0]
            lines.append(f"\n→ Recommended bid: {best_bid}  (confidence: {bid_probs[0][1]:.0%})")
            lines.append(f"  State value: {value:.3f}")

            self.rec_label.config(text="\n".join(lines))

        except Exception as e:
            self.rec_label.config(text=f"Error: {e}")

    def _recommend_play(self):
        try:
            game, seat = self._build_game_state("play")
            probs, value = self._get_recommendations(game, seat, "play")

            # Extract play probabilities (actions 0-51)
            card_probs = []
            legal = game.get_legal_plays(seat)
            for card in legal:
                idx = card_index(card)
                p = probs[idx]
                card_probs.append((card, p))

            card_probs.sort(key=lambda x: -x[1])

            lines = ["PLAY RECOMMENDATIONS:\n"]
            for card, prob in card_probs:
                bar = "█" * int(prob * 30)
                lines.append(f"  {card_str(card):>4s}:  {prob:>6.1%}  {bar}")

            best = card_probs[0][0]
            lines.append(f"\n→ Recommended play: {card_str(best)}  (confidence: {card_probs[0][1]:.0%})")
            lines.append(f"  State value: {value:.3f}")

            self.rec_label.config(text="\n".join(lines))

        except Exception as e:
            self.rec_label.config(text=f"Error: {e}")

    def _clear_all(self):
        self.selected_hand.clear()
        for card, btn in self.card_buttons.items():
            btn.config(bg="#2a2a4e", relief=tk.RAISED)
        self.hand_label.config(text="Hand: (none selected)")

        self.trump_suit = None
        for btn in self.trump_buttons:
            btn.config(relief=tk.RAISED, bg="#2a2a4e")

        for _, entry in self.bid_entries:
            entry.delete(0, tk.END)
        for _, entry in self.trick_entries:
            entry.delete(0, tk.END)
        for _, entry in self.tricks_won_entries:
            entry.delete(0, tk.END)
            entry.insert(0, "0")
        self.seen_text.delete("1.0", tk.END)
        self.trump_broken_var.set(False)
        self.rec_label.config(text="Cleared. Enter your game state.")


def main():
    parser = argparse.ArgumentParser(description="Oh Hell Engine GUI")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (can also load via GUI)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("800x900")
    app = OhHellGUI(root, checkpoint_path=args.checkpoint, device=args.device)
    root.mainloop()


if __name__ == "__main__":
    main()
