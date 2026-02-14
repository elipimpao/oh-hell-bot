"""Neural network for Oh Hell RL agent."""

import math
import torch
import torch.nn as nn
import numpy as np
from env import MAX_BID


class OhHellNetwork(nn.Module):
    """Two-headed network: shared encoder with separate bid, play, and value heads.

    - Bid head: outputs logits over bids 0..MAX_BID
    - Play head: outputs logits over 52 cards
    - Value head: single scalar state value

    Uses layer normalization in the encoder and orthogonal weight initialization.
    """

    def __init__(self, obs_dim, hidden_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.bid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, MAX_BID + 1),
        )

        self.play_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 52),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization with appropriate gains."""
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Policy heads: small gain for near-uniform initial policy
        for head in [self.bid_head, self.play_head]:
            for i, module in enumerate(head):
                if isinstance(module, nn.Linear):
                    if i == len(head) - 1:  # final layer
                        nn.init.orthogonal_(module.weight, gain=0.01)
                    else:
                        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                    nn.init.zeros_(module.bias)

        # Value head: gain=1.0 for final layer
        for i, module in enumerate(self.value_head):
            if isinstance(module, nn.Linear):
                if i == len(self.value_head) - 1:  # final layer
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, obs, action_mask):
        """Forward pass.

        Args:
            obs: (batch, obs_dim) observation tensor
            action_mask: (batch, 65) binary mask — 1 for legal actions

        Returns:
            logits: (batch, 65) masked logits for action selection
            value: (batch, 1) state value estimate
        """
        features = self.encoder(obs)

        bid_logits = self.bid_head(features)    # (batch, 13)
        play_logits = self.play_head(features)  # (batch, 52)

        # Combine into full action logits: [play_0..play_51, bid_0..bid_12]
        logits = torch.cat([play_logits, bid_logits], dim=-1)

        # Mask illegal actions to -inf
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        value = self.value_head(features)

        return logits, value

    def get_action_and_value(self, obs, action_mask, action=None):
        """Sample action or evaluate given action.

        Args:
            obs: (batch, obs_dim)
            action_mask: (batch, 65)
            action: (batch,) optional — if provided, evaluate this action

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs, action_mask):
        """Get state value only (for GAE computation)."""
        _, value = self.forward(obs, action_mask)
        return value.squeeze(-1)
