"""Gymnasium environment for Oh Hell."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import OhHellGame, Phase, card_index, index_to_card

MAX_PLAYERS = 5
MAX_BID = 12  # max hand size (capped at 12 for 2-4 players)


class OhHellEnv(gym.Env):
    """Oh Hell environment for a single RL agent.

    The agent controls one seat. Opponents are played by bot policies passed in.
    The action space is Discrete(65): actions 0-51 for playing cards, 52-64 for bids 0-12.
    Action masking is provided via info["action_mask"].
    """

    metadata = {"render_modes": []}

    def __init__(self, num_players=4, opponents=None, agent_seat=0, seed=None,
                 auto_opponents=True):
        super().__init__()
        self.num_players = num_players
        self.agent_seat = agent_seat
        self.opponents = opponents or []
        self._seed = seed
        self.auto_opponents = auto_opponents

        # Action space: 0-51 = card plays, 52-64 = bids (0-12)
        self.action_size = 52 + MAX_BID + 1  # 65
        self.action_space = spaces.Discrete(self.action_size)

        # Observation space: flat float vector
        self.obs_size = self._calc_obs_size()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )

        # Preallocated buffers (reused every step to avoid allocations)
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)
        self._mask_buffer = np.zeros(self.action_size, dtype=np.float32)

        self.game = None

    def _calc_obs_size(self):
        """Calculate observation vector size."""
        return (
            52          # hand
            + 4         # trump suit one-hot
            + 2         # phase one-hot
            + 4         # player count one-hot (2p/3p/4p/5p)
            + MAX_PLAYERS  # bids (padded)
            + MAX_PLAYERS  # tricks_won (padded)
            + 1         # trick deficit
            + 52        # cards in current trick
            + 4         # lead suit one-hot
            + MAX_PLAYERS * 52  # per-player cards played (rotated, padded)
            + 52        # cards unseen
            + 1         # trump broken
            + 1         # hand size normalized
            + MAX_PLAYERS  # position relative to dealer
            + MAX_PLAYERS  # position within current trick
        )  # = 52+4+2+4+5+5+1+52+4+260+52+1+1+5+5 = 453

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = seed if seed is not None else self._seed
        import random
        rng = random.Random(s)
        self.game = OhHellGame(self.num_players, rng=rng)

        # Play opponent actions until it's the agent's turn
        if self.auto_opponents:
            self._play_opponents()

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action):
        """Take an action for the agent."""
        assert self.game.current_player == self.agent_seat

        reward = 0.0
        round_complete = False

        # Decode and execute action
        if self.game.phase == Phase.BIDDING:
            bid = action - 52  # actions 52-64 map to bids 0-12
            self.game.place_bid(bid)
        else:
            card = index_to_card(action)  # actions 0-51 map to cards
            events = self.game.play_card(card)
            if events.get("round_complete"):
                reward += self._compute_reward(events)
                round_complete = True

        # Play opponents until agent's turn again or game over
        if self.auto_opponents:
            opp_reward, opp_round = self._play_opponents()
            reward += opp_reward
            round_complete = round_complete or opp_round

        terminated = self.game.is_game_over
        obs = self._get_obs() if not terminated else np.zeros(self.obs_size, dtype=np.float32)
        info = {"action_mask": self._get_action_mask()} if not terminated else {}
        info["round_complete"] = round_complete

        if terminated:
            info["final_scores"] = list(self.game.scores)

        return obs, reward, terminated, False, info

    def _play_opponents(self):
        """Play all opponent turns until it's the agent's turn or game ends.
        Returns (accumulated_reward, round_complete_flag)."""
        reward = 0.0
        round_complete = False

        while not self.game.is_game_over and self.game.current_player != self.agent_seat:
            player = self.game.current_player
            opp_idx = self._opponent_index(player)
            bot = self.opponents[opp_idx]

            if self.game.phase == Phase.BIDDING:
                bid = bot.act(self.game, player)
                self.game.place_bid(bid)
            else:
                card = bot.act(self.game, player)
                events = self.game.play_card(card)
                if events.get("round_complete"):
                    reward += self._compute_reward(events)
                    round_complete = True

        return reward, round_complete

    def needs_opponent_action(self):
        """Check if an opponent (not the agent) needs to act next."""
        return (not self.game.is_game_over and
                self.game.current_player != self.agent_seat)

    def step_opponent(self, action):
        """Apply one opponent action. Returns (reward, round_complete)."""
        reward = 0.0
        round_complete = False
        if self.game.phase == Phase.BIDDING:
            self.game.place_bid(action - 52)
        else:
            events = self.game.play_card(index_to_card(action))
            if events.get("round_complete"):
                reward += self._compute_reward(events)
                round_complete = True
        return reward, round_complete

    def _opponent_index(self, player):
        """Map player seat to opponent list index."""
        seats = [i for i in range(self.num_players) if i != self.agent_seat]
        return seats.index(player)

    def _compute_reward(self, events):
        """Compute shaped reward for the agent from round events.

        Hit: (10 + bid) / max_score  (range ~0.45 to 1.0)
        Miss: -0.2 * |bid - tricks_won| / hand_size  (range -0.2 to 0)

        Uses bids/tricks/hand_size from events dict (captured before
        _start_round() resets them for the next round).
        """
        agent_score = events["round_scores"][self.agent_seat]
        max_score = 10 + self.game.max_hand_size
        if agent_score > 0:
            # Hit the bid exactly
            return agent_score / max_score
        else:
            # Missed — penalty proportional to distance
            bid = events["round_bids"][self.agent_seat]
            tricks = events["round_tricks_won"][self.agent_seat]
            hand_size = events["round_hand_size"]
            return -0.2 * abs(bid - tricks) / max(hand_size, 1)

    def _get_obs(self):
        """Build observation vector for the agent.

        Writes directly into a preallocated buffer to avoid allocations.
        Accesses game fields directly instead of copying via get_state().
        Cards are ints 0-51, used directly as array indices.
        """
        if self.game.is_game_over:
            return np.zeros(self.obs_size, dtype=np.float32)

        game = self.game
        seat = self.agent_seat
        np_ = self.num_players
        obs = self._obs_buffer
        obs[:] = 0.0

        # Hand (52) — offset 0
        for card in game.hands[seat]:
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
            p = (seat + i) % np_
            b = game.bids[p]
            if b >= 0:
                obs[62 + i] = b * inv_hs

        # Tricks won (5) — offset 67
        inv_hand = 1.0 / max(game.hand_size, 1)
        for i in range(np_):
            p = (seat + i) % np_
            obs[67 + i] = game.tricks_won[p] * inv_hand

        # Trick deficit (1) — offset 72
        my_bid = game.bids[seat]
        obs[72] = (my_bid - game.tricks_won[seat]) * inv_hs if my_bid >= 0 else 0.0

        # Current trick cards (52) — offset 73
        ct = game.current_trick
        for _, card in ct:
            obs[73 + card] = 1.0

        # Lead suit (4) — offset 125
        if game.phase == Phase.PLAYING and ct:
            obs[125 + ct[0][1] // 13] = 1.0

        # Per-player cards played (5*52) — offset 129
        for i in range(np_):
            p = (seat + i) % np_
            base = 129 + i * 52
            for card in game.cards_played_by[p]:
                obs[base + card] = 1.0

        # Cards unseen (52) — offset 389
        obs[389:441] = 1.0
        for card in game.hands[seat]:
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
        obs[443 + (seat - game.dealer) % np_] = 1.0

        # Position within trick (5) — offset 448
        if game.phase == Phase.PLAYING:
            trick_pos = len(ct)
            if trick_pos < MAX_PLAYERS:
                obs[448 + trick_pos] = 1.0

        return obs.copy()

    def _get_action_mask(self):
        """Build action mask: 1 for legal actions, 0 for illegal."""
        mask = self._mask_buffer
        mask[:] = 0.0

        if self.game.is_game_over:
            return mask.copy()

        if self.game.phase == Phase.BIDDING:
            for bid in self.game.get_legal_bids():
                mask[52 + bid] = 1.0
        else:
            for card in self.game.get_legal_plays():
                mask[card] = 1.0

        return mask.copy()
