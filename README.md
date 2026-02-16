# Oh Hell Bot

A reinforcement learning agent for the card game [Oh Hell](https://en.wikipedia.org/wiki/Oh_hell), with the goal of creating the strongest possible agent. Trained with PPO (Proximal Policy Optimization) and Prioritized Fictitious Self-Play (PFSP), the agent learns to both bid and play cards across all player counts (2-5) using a single neural network.

## Files

### Core Game

- **`game.py`** — Pure game engine. Implements all Oh Hell rules: dealing, bidding (with the dealer restriction that prevents total bids from equaling hand size), trick-taking with lead-suit following, trump breaking, scoring (10 + bid for exact match, 0 otherwise), and round progression (hand sizes count down from max to 1 then back up). Supports 2-5 players. No ML dependencies. Cards are plain integers 0-51 (`suit = card // 13`, `rank = card % 13`). Automatically imports the Cython-accelerated engine (`game_fast`) if available, falling back to pure Python otherwise.

- **`game_fast.pyx`** — Cython-accelerated drop-in replacement for the game engine. Uses typed `cdef` class attributes and C-level internal methods (`_start_round`, `_resolve_trick`, `_score_round`) to eliminate Python object overhead in hot paths. Build with `python setup_cython.py build_ext --inplace` (requires a C compiler).

- **`setup_cython.py`** — Build script for the Cython game engine. Compiles `game_fast.pyx` with bounds-check and wraparound disabled for maximum performance.

- **`bots.py`** — Three tiers of rule-based opponents:
  - `RandomBot` — Uniform random from legal actions. Provides baseline diversity.
  - `HeuristicBot` — Bids based on high cards and trump count, plays with simple trick-winning logic (play high when needing tricks, duck when satisfied).
  - `SmartBot` — Card-counting heuristic bot. Tracks cards played to identify guaranteed winners, detects opponent voids from off-suit plays, uses finesse ducking (shedding dangerous middle cards), evaluates hand strength with suit length/protection/void awareness, and adjusts bids based on position and player count.

### ML Components

- **`env.py`** — Gymnasium environment wrapping the game engine for a single RL agent. Defines the 453-dimensional observation space and 65-dimensional action space (52 card plays + 13 bid values). Provides action masking for legal moves and shaped rewards. Supports both auto-opponent mode (bots play automatically) and manual mode (caller controls all players, used by the multiprocessing workers for NN opponents).

- **`network.py`** — PyTorch neural network with a shared encoder (3 hidden layers, each with layer normalization and ReLU) and three heads: bid head (13 outputs), play head (52 outputs), and value head (scalar). Uses orthogonal weight initialization with small gain (0.01) on policy output layers for near-uniform initial policy. Supports action masking and sampling via `get_action_and_value()`.

- **`train.py`** — PPO training loop (CleanRL-style). Trains across all player counts simultaneously using multiprocessing workers that collect complete rollouts independently with CPU-side agent inference. Uses PFSP for opponent selection with a pool of self-play snapshots and rule-based bots. Includes periodic evaluation against all opponent types plus snapshot evaluation, checkpointing, TensorBoard logging, and CSV eval logs.

### Dashboard & GUIs

- **`dashboard.py`** — Gradio-based training dashboard that runs in any browser. Provides a complete web UI for configuring, launching, monitoring, and managing training runs without touching config files or a terminal. Features 6 tabs: Training (config form + start/pause/resume/stop), Dashboard (live metrics, reward/loss curves, win rate heatmap, per-player-count win rates), Evaluation (historical charts + on-demand eval), Opponent Pool (PFSP pool viewer + snapshot browser), League (independent main + exploiter agent controls), and Settings (config presets, run history, resource monitor, log viewer). Communicates with `train.py` via stdout parsing and `status.json` for real-time metrics.

- **`gui/`** — Web-based game interface using FastAPI + WebSockets. Two modes:
  - **Play Mode**: Full game simulation where a human plays one seat against configurable AI bots (random, heuristic, smart, or neural network). Bot turns auto-advance with configurable animation speed. Supports custom hand sizes, trump selection, and dev mode (all cards face-up).
  - **Advisor Mode**: Decision advisor for real-world games. Mirror an ongoing game (e.g., on Trickster Cards), input the game state as it happens, and receive AI recommendations with probability distributions from a trained neural network.
  - Key files: `server.py` (FastAPI app + WebSocket handler), `session.py` (game simulation), `advisor.py` (NN-based recommendations), `bot_manager.py` (bot creation + model loading), `static/` (HTML/CSS/JS frontend).

### Chrome Extension (Trickster Cards Integration)

- **`extension/`** — Chrome Extension (Manifest V3) that integrates with [Trickster Cards](https://www.trickstercards.com) to automatically read game state and display AI recommendations as an overlay on the game page. Eliminates manual data entry — the extension intercepts Trickster's console.log events to detect hands, bids, plays, and trick results in real time. Communicates with the existing FastAPI backend via WebSocket through a background service worker. Key files:
  - `manifest.json` — MV3 manifest targeting `trickstercards.com/game/*`
  - `inject.js` — Runs in the page's main world. Hooks `console.log` to intercept structured game events (player joins, cards dealt, bids, card plays, trick results) and forwards them via `window.postMessage`. Also handles autoplay DOM interaction (clicking bid buttons and card elements).
  - `content.js` — Content script: game state machine, seat mapping (handles non-contiguous Trickster seats), trump card detection from DOM (`aria-label` parsing), backend communication via background service worker, recommendation overlay rendering, autoplay scheduling, and settings UI (gear menu with autoplay toggle and delay slider).
  - `background.js` — MV3 service worker. Handles HTTP/WebSocket communication on behalf of the content script (bypasses mixed-content/CORS restrictions).
  - `overlay.css` — Styles for the floating recommendation panel, settings menu, toggle switch, delay slider, and AUTO badge.
  - `popup.html/js/css` — Extension popup for backend URL and snapshot path configuration.
  - `icons/` — Extension icons (16, 48, 128px).

### Utilities

- **`evaluate.py`** — Standalone evaluation and plotting. Evaluates any checkpoint across all player counts against random, heuristic, and smart opponents. Generates a 3x3 training progress plot (score/win rate/bid accuracy vs each opponent type) from CSV logs, with per-player-count lines and aggregate. Includes weakness ranking across all opponent/player-count combinations.

- **`play.py`** — Tkinter GUI engine for getting move recommendations from a trained agent. Enter your real game state (hand, trump, bids, trick, cards played) and the neural network recommends bids or card plays with probability distributions and confidence levels.

- **`league.py`** — League training orchestrator. Reads `league.toml`, launches multiple `train.py` subprocesses (main agent + exploiter), monitors health, auto-restarts crashed processes, and handles graceful shutdown via Ctrl+C. Agents communicate through the filesystem by writing snapshots to their own directory and reading from each other's.

- **`league.toml`** — League configuration file. Defines agent roles (main vs exploiter), snapshot directories, rescan intervals, load directories, and per-agent config overrides.

- **`requirements.txt`** — Python dependencies: torch, gymnasium, numpy, tqdm, tensorboard, matplotlib, tomli, gradio, plotly, psutil, fastapi, uvicorn, websockets. Optional: cython (for building the accelerated game engine).

## Training Flow & Strategy

### The Big Picture

The goal is to train a single neural network that plays Oh Hell as strongly as possible across all player counts (2-5). The strategy draws from [AlphaStar](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)'s league training: maintain a diverse pool of opponents, bias training toward the hardest ones, and never stop facing weaker opponents to prevent [catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference).

The training loop repeats a simple cycle:

```
┌─────────────────────────────────────────────────────────────┐
│  1. SELECT OPPONENTS                                        │
│     PFSP picks 3 opponents from the pool, biased toward     │
│     harder ones (lower reward = higher selection weight)     │
│                                                             │
│  2. COLLECT EXPERIENCE                                      │
│     Workers run 512 steps across 256 parallel tables,       │
│     each table with 2-5 players. Opponents are the PFSP     │
│     picks — either rule-based bots or NN snapshots.         │
│     This produces 131,072 step-experiences per cycle.       │
│                                                             │
│  3. LEARN                                                   │
│     PPO runs 4 epochs over the collected data in mini-      │
│     batches of 4096, producing 128 gradient updates.        │
│     The agent's weights are updated. No learning happens    │
│     during step 2 — the agent plays with frozen weights.    │
│                                                             │
│  4. MEASURE & SNAPSHOT                                      │
│     The global reward from step 2 is attributed to the      │
│     selected opponents (for future PFSP weighting).         │
│     Periodically, the agent's current weights are saved     │
│     as a new snapshot in the opponent pool.                  │
│                                                             │
│  Repeat until total timesteps reached.                      │
└─────────────────────────────────────────────────────────────┘
```

### How Opponents Are Chosen

**The opponent pool** contains 3 fixed bot types (RandomBot, HeuristicBot, SmartBot) plus up to 100 self-play snapshots (frozen copies of the agent from earlier in training) and, during league training, exploiter snapshots loaded from external directories. All opponents live in a single pool managed by Prioritized Fictitious Self-Play (PFSP).

**PFSP selection** picks 3 opponents per cycle. Each pick is stochastic but weighted toward harder opponents — the weight formula `(1 - normalized_reward + 0.1)^2` gives higher probability to opponents the agent has lower reward against. A staleness boost ensures opponents that haven't been picked recently still get periodic play, and new opponents with no data get an exploration bonus.

**Why 3 opponents?** Multi-player tables (3-5 players) have multiple opponent seats. Rather than filling all seats with the same opponent, 3 distinct PFSP picks provide table diversity — the agent might face SmartBot at seat 1, a snapshot at seat 2, and HeuristicBot at seat 3 within the same game.

### How Tables Are Composed

Each of the 256 parallel environments has a randomly chosen player count (15% 2p, 20% 3p, 30% 4p, 35% 5p). The 3 PFSP-selected opponents are distributed across seats:

| Table type | Probability | Seat assignment |
|------------|-------------|-----------------|
| **Homogeneous** | 25% | All seats get config 0 (hardest PFSP pick) |
| **Mixed** | 75% | Seat 1 = config 0; remaining seats = 50% config 0, 50% random across all 3 configs |

This means the agent regularly faces both pure-opposition tables (e.g., all SmartBots) and mixed-difficulty tables. The homogeneous tables are critical because evaluation tests against all-identical opponents — without them, the agent would never train in conditions matching evaluation.

**For different table sizes:**
- **2-player**: Only 1 opponent seat, always gets config 0 (the PFSP-hardest). Configs 1-2 are unused and don't receive reward measurements.
- **3-player**: 2 opponent seats. In homogeneous mode, both get config 0. In mixed mode, seat 1 = config 0, seat 2 = 50% config 0 / 50% random.
- **4-player**: 3 opponent seats. Same logic, more diversity possible.
- **5-player**: 4 opponent seats. Maximum table diversity — mixed tables can have all 3 configs represented, sometimes with duplicates.

### Why Keep Weak Opponents?

Training only against the strongest opponents causes **strategy collapse** — the agent over-specializes against one play style and forgets how to handle others. [Research on AlphaStar's league training](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) showed that "playing to win is insufficient" and that diverse opponents (including weak ones) are essential for robust play.

Weak opponents serve several roles:
- **Regression tests**: If win rate against RandomBot drops, something fundamental is broken
- **Basic skill maintenance**: Bidding accuracy, legal play patterns, trick-counting
- **Strategy diversity**: Random opponents create chaotic games the agent must handle
- **Preventing catastrophic forgetting**: Skills not exercised in training can decay

PFSP naturally handles the balance — as the agent improves, weak opponents get lower selection weight but are never removed entirely. The staleness mechanism ensures they still get periodic play.

## Implementation Details

### Architecture

The training system uses a main process for PPO updates and multiple worker subprocesses for environment stepping. Each worker owns a slice of the environments and a CPU copy of the agent network, collecting complete rollouts of T steps independently. This eliminates per-step IPC between the main process and workers — only full rollout buffers are transferred between updates.

Each worker also maintains one or more self-play opponent networks (lazy-allocated) and config-driven bot instances. When the main process sends a "collect" command, it includes the current agent weights and a list of opponent configs (bot types and/or NN snapshot state dicts). The worker loads NN weights and creates bot instances matching each config's type, assigns each opponent seat in each environment to a config, and runs T steps of interaction. Bot seats use the PFSP-selected bot type (not the env's stored bots), so PFSP difficulty targeting works for all opponent types. NN seats are batched per-network for efficient inference.

### Opponent Pool and PFSP

All opponents — the three fixed bot types (RandomBot, HeuristicBot, SmartBot), self-play snapshots, and (in league mode) exploiter snapshots — live in a single opponent pool managed by Prioritized Fictitious Self-Play (PFSP).

**PFSP selection** biases toward harder opponents. Each opponent's difficulty is estimated from a rolling window of recent training rewards (default: 5 measurements). The selection weight is:

```
base_weight = (1 - normalized_reward + 0.1)^2
```

Lower reward (harder opponent) means higher weight. Opponents with no measurements get an exploration bonus (default: 1.0). A staleness multiplier (`1 + min(staleness / divisor, max_mult)`) boosts opponents that haven't been selected recently, preventing any opponent from becoming permanently unmeasured. All PFSP parameters are configurable in `[pfsp]`.

Each update selects **3 opponents** from the pool via PFSP without replacement. All 3 selections are PFSP-weighted (biased toward harder opponents), but since PFSP is stochastic, this naturally produces variety — sometimes all 3 are hard snapshots, sometimes it's a mix of snapshots and bots.

**Reward tracking** updates selected opponents' reward histories after every update — but only for configs that were actually seated. Workers report which config indices appeared in any seat assignment during the rollout. In 2-player games only config 0 (primary) is used since there's just one opponent seat, so configs 1-2 don't receive noisy reward signal. In 3-5 player games, all 3 configs are typically used, tripling measurement frequency compared to tracking only the primary.

### Seat Assignment

Each environment in a worker has opponent seats assigned to one of the 3 selected configs:

- **25% of tables are homogeneous** — all seats get config 0 (the PFSP-hardest). This ensures the agent regularly faces tables where every opponent is the same type (e.g., all SmartBots or all the same snapshot), matching evaluation conditions.
- **75% of tables are mixed** — Seat 1 always gets config 0, remaining seats have a 50% chance of config 0, otherwise uniform random across all 3 configs.

This creates a healthy mix: homogeneous tables train the agent for pure-opposition scenarios (like evaluation), while mixed tables provide diversity and prevent overfitting to a single opponent style.

### Training Phases

#### Phase 1: Bot Warmup

For the first 500K steps (`--self-play-start`), no snapshots exist in the pool. The opponent pool contains only the 3 fixed bot types, and PFSP selects among them. PFSP controls which bot types actually play at each seat — when SmartBot is selected as a config, all seats assigned to that config play as SmartBots. This phase teaches fundamental skills: legal play, bidding-to-trick correlation, and basic card sense.

#### Phase 2: PFSP Self-Play

After the warmup threshold, snapshots of the agent are saved to the pool every 1M steps (`--snapshot-interval`). The pool holds the 100 most recent snapshots (oldest evicted when full) plus the 3 fixed bots. PFSP naturally transitions from pure bot training to mixed self-play as snapshots accumulate — no explicit fraction parameter needed.

Self-play opponents are diverse by construction: the pool contains snapshots from many training stages, and PFSP's stochastic weighting ensures the agent doesn't fixate on any single opponent. This prevents co-adaptation and encourages robust, general-purpose play.

#### Phase 3: League Training (Exploiter)

Once the main agent is reasonably strong, `league.py` launches an exploiter agent alongside it. The exploiter is a second network that trains exclusively against snapshots of the main agent (no fixed bots, no self-play). Its sole objective is to discover and exploit weaknesses in the main agent's play.

The two agents communicate through the filesystem:
1. Main agent saves snapshots to `league_snapshots/main/`
2. Exploiter periodically rescans that directory, discovers new snapshots, and adds them to its pool
3. Exploiter trains against them and saves its own snapshots to `league_snapshots/exploiter/`
4. Main agent rescans the exploiter directory and adds exploiter snapshots to its PFSP pool
5. PFSP weights exploiter snapshots heavily (since the main agent loses to them)
6. Main agent trains against exploiter snapshots, patching the exposed weaknesses

This creates an arms race that systematically eliminates blind spots. Based on [AlphaStar's league training](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) and [Minimax Exploiter](https://www.ubisoft.com/en-us/studio/laforge/news/H2DwtSgIZtFiXL55l4N6n/minimax-exploiter-a-data-efficient-approach-for-competitive-self-play) research.

### Reward Shaping

The agent receives a shaped reward at the end of each round (not per trick):

- **Hit** (exact bid): `(10 + bid) / max_possible_score` — ranges from ~0.45 to 1.0
- **Miss**: `-0.2 * |bid - tricks_won| / hand_size` — small penalty proportional to miss distance

The asymmetry is intentional: hits are strongly rewarded while misses are lightly penalized. The proportional miss penalty provides gradient signal for near-misses, helping the agent learn to bid more accurately rather than treating all misses equally.

### Per-Round Episodes

Each round is treated as an independent episode for value bootstrapping. The done signal fires at round boundaries, resetting the value estimate. A full Oh Hell game can span ~300 steps across many rounds, and without this reset the discount factor would blur reward attribution across rounds, making it hard for the agent to learn which round-level decisions led to which outcomes.

### Multi-Player-Count Training

Every environment selects a player count (2, 3, 4, or 5) with weighted sampling. Base weights are 15% for 2p, 20% for 3p, 30% for 4p, 35% for 5p — higher player counts are weighted more because they have more strategic depth and are more common in real play. The agent trains across all configurations simultaneously with a single network — the player count one-hot in the observation tells it how to adapt.

**Adaptive distribution**: The system tracks per-player-count win rates during training and shifts the distribution toward weaker player counts. Controlled by `[player_counts]` in config.toml — `adaptation_rate` (0.0 = static, 1.0 = fully adaptive) blends base weights with weakness-based weights, and `min_weight` prevents any count from being starved entirely.

### Evaluation

Periodic evaluation (every `--eval-interval` steps) runs the agent across all 4 player counts against all 3 bot types (random, heuristic, smart), using greedy (argmax) action selection for deterministic results. Metrics tracked:

- **Average score** — total points across all rounds
- **Win rate** — fraction of games where the agent has the highest score
- **Bid accuracy** — fraction of rounds where the agent hits their bid exactly
- **Winning score** — average/min/max score of game winners (context for how strong the competition is)

A weakness ranking sorts all 12 opponent/player-count combinations by win rate to highlight where the agent struggles most.

## Observation Space (453 dimensions)

All observations are relative to the agent's perspective. Player-indexed features are rotated so the agent is always at index 0.

| # | Feature | Size | Encoding | Description |
|---|---------|------|----------|-------------|
| 1 | Hand | 52 | Binary (0/1) | Which cards the agent currently holds. Each position maps to a specific card (suit x 13 + rank). |
| 2 | Trump suit | 4 | One-hot | Which of the four suits is trump this round. |
| 3 | Phase | 2 | One-hot | Whether the current phase is bidding (index 0) or playing (index 1). |
| 4 | Player count | 4 | One-hot | Number of players in the game: index 0 = 2p, index 1 = 3p, index 2 = 4p, index 3 = 5p. |
| 5 | Bids | 5 | Float [-1, 1] | Each player's bid divided by hand size. -1.0 = hasn't bid yet or unused slot. Agent is index 0, then clockwise. |
| 6 | Tricks won | 5 | Float [0, 1] | Each player's tricks won divided by hand size. 0.0 for unused slots. Agent is index 0, then clockwise. |
| 7 | Trick deficit | 1 | Float [-1, 1] | (Agent's bid - tricks won) / hand size. Positive = still needs tricks, negative = over-tricked. 0.0 during bidding. |
| 8 | Current trick | 52 | Binary (0/1) | Cards on the table in the current trick. Cleared each new trick. |
| 9 | Lead suit | 4 | One-hot | Suit of the first card played in the current trick. All zeros during bidding or at trick start. |
| 10 | Per-player cards played | 260 | Binary (0/1) | Five 52-dim vectors (one per player slot), marking every card each player has played this round. Agent is index 0, then clockwise. Unused slots are zeros. |
| 11 | Cards unseen | 52 | Binary (0/1) | Cards not in the agent's hand, not yet played, and not the trump card. Helps reason about what opponents might hold. |
| 12 | Trump broken | 1 | Binary (0/1) | 1.0 if trump has been played off-lead this round (unlocks leading with trump). |
| 13 | Hand size | 1 | Float [0, 1] | Current round's hand size / max hand size. Tells the agent which round it is. |
| 14 | Dealer position | 5 | One-hot | Agent's position relative to the dealer (0 = agent is dealer, 1 = one seat left, etc.). |
| 15 | Trick position | 5 | One-hot | How many cards played before the agent in the current trick (0 = agent leads). All zeros during bidding. |

## Action Space (65 dimensions)

| Range | Action | Description |
|-------|--------|-------------|
| 0-51 | Card play | Play a specific card (suit x 13 + rank). Only valid during playing phase. |
| 52-64 | Bid (0-12) | Place a bid of 0 to 12. Only valid during bidding phase. |

Illegal actions are masked to `-inf` logits so the agent can only select legal moves.

## Usage Guide

### Setup

```bash
pip install -r requirements.txt

# Optional: build Cython-accelerated game engine (requires C compiler)
pip install cython
python setup_cython.py build_ext --inplace
```

All directories (`checkpoints/`, `snapshots/`, `runs/`, etc.) are created automatically on first run. No manual setup needed beyond installing dependencies.

### Training from Scratch

```bash
python train.py
```

This starts a full training run with default settings from `config.toml`. Here's what happens:

1. **Phase 1 (first ~500K steps)**: The agent trains against rule-based bots only (Random, Heuristic, Smart). It learns basic card play, bidding, and trick-taking.
2. **Phase 2 (500K+ steps)**: Self-play snapshots start accumulating. PFSP gradually shifts focus from bots to the agent's own frozen copies, building robust strategy.

**What gets created:**

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `checkpoints/` | `PPO_ABC123_1M.pt`, `..._2M.pt`, ... | Full training state for resuming |
| `snapshots/` | `PPO_ABC123_1M.pt`, `..._2M.pt`, ... | Lightweight weights for opponent pool |
| `runs/<run_name>/` | TensorBoard logs + `eval_log.csv` | Training metrics and evaluation history |

The run name is auto-generated (e.g., `PPO_ABC123`) but can be set with `--run-name` or in config.toml.

### Resuming Training

There are two ways to load a previous model, and they do very different things:

**`--resume` (continue the same run)**
```bash
python train.py --resume checkpoints/PPO_ABC123_50M.pt
```
Restores everything: network weights, optimizer state (momentum, learning rate schedule), and step counter. Training continues exactly where it left off. Use this after stopping and restarting.

**`--init-weights` (start fresh with pretrained weights)**
```bash
python train.py --init-weights checkpoints/PPO_ABC123_50M.pt
```
Loads network weights only. Optimizer starts fresh, step counter resets to 0. Use this for transfer learning or when starting a new exploiter agent from the main agent's weights.

Both can be set via CLI or in config.toml under `[checkpoints]`:
```toml
[checkpoints]
resume = "checkpoints/PPO_ABC123_50M.pt"
```

**Tip**: To find the latest checkpoint, look in `checkpoints/` for the file with the highest step number (e.g., `_215M.pt` is newer than `_50M.pt`).

### Checkpoints vs Snapshots

The system saves two types of files. Understanding the difference is important:

| | Checkpoints | Snapshots |
|---|---|---|
| **Location** | `checkpoints/` | `snapshots/` (or `league_snapshots/`) |
| **Contains** | Weights + optimizer + step counter | Weights only |
| **Size** | Larger (~2x) | Smaller |
| **Purpose** | Resume training | PFSP opponent pool + league communication |
| **Save interval** | `checkpoint_interval` (default: 1M steps) | `snapshot_interval` (default: 1M steps) |
| **Safe to delete?** | Only if you won't resume from them | Yes, pool reloads from disk on startup |

Checkpoints follow the naming pattern `{run_name}_CHKPT_{step}.pt` (e.g., `PPO_ABC123_CHKPT_127M.pt`). A `_CHKPT_FINAL.pt` file is saved when training completes. Snapshots follow `{run_name}_{step}.pt` (e.g., `PPO_ABC123_127M.pt`).

### League Training (Main + Exploiter)

League training runs two agents simultaneously — a main agent and an exploiter — that communicate through shared snapshot directories.

**Prerequisites**: A trained main agent (checkpoint + at least one snapshot).

**Step 1: Configure `league.toml`**

```toml
[agents.main.overrides]
# Point to the main agent's latest checkpoint
resume = "checkpoints/PPO_ABC123_200M.pt"

[agents.exploiter.overrides]
# Point to the main agent's latest snapshot (exploiter starts from these weights)
init_weights = "league_snapshots/main/PPO_ABC123_200M.pt"
```

Also adjust resource allocation — both agents share your CPU/GPU:
```toml
[agents.main.overrides]
num_envs = 256
num_workers = 16

[agents.exploiter.overrides]
num_envs = 128
num_workers = 8
```

**Step 2: Run**

```bash
python league.py
```

The orchestrator launches both agents as subprocesses, prefixes their output with `[main]` and `[exploiter]`, and auto-restarts either if it crashes (up to 3 times).

**Step 3: Stop and resume**

Press `Ctrl+C` for graceful shutdown. To resume later:

1. Find the latest checkpoint for each agent in `checkpoints/` (they'll have `MAIN_` or `EXP_` prefixes)
2. Update `league.toml`:
   ```toml
   [agents.main.overrides]
   resume = "checkpoints/MAIN_XYZ789_250M.pt"

   [agents.exploiter.overrides]
   resume = "checkpoints/EXP_XYZ789_30M.pt"
   ```
3. Run `python league.py` again

**Starting a fresh exploiter** (keeping the main agent): Use `init_weights` instead of `resume` for the exploiter. This gives it the main agent's weights but resets its training.

### Training Dashboard (Browser GUI)

The Gradio dashboard provides full training control from a browser — no terminal or config file editing needed.

```bash
python dashboard.py                  # opens at http://localhost:7860
python dashboard.py --port 8080      # custom port
python dashboard.py --share          # create a public Gradio link
```

**Tab overview:**

| Tab | What it does |
|-----|-------------|
| **Training** | Configure all parameters (timesteps, envs, learning rate, opponent composition, player count weights, etc.) via form controls. Start/Pause/Resume/Stop training with one click. Resume from any checkpoint via dropdown. |
| **Dashboard** | Live metrics updated every 3 seconds: reward & loss curves, entropy & throughput, win rate heatmap (3 bots x 4 player counts), per-player-count win rate bars. Metric cards show current reward, losses, entropy, and SPS. |
| **Evaluation** | View historical evaluation charts from CSV logs. Run on-demand evaluation of any checkpoint against all bot types and player counts. |
| **Opponent Pool** | View the current PFSP pool (opponent ID, type, step, win rate, games played). Browse all snapshot files on disk across directories. |
| **League** | Independent controls for main and exploiter agents. Configure each agent's envs, workers, snapshot directory, load directories, resume checkpoint, and init weights. Start/stop agents independently. |
| **Settings** | Save/load named config presets. View run history. System resource monitor (CPU, memory, GPU). Scrollable training log output. |

The dashboard communicates with `train.py` through two channels:
1. **stdout parsing** — regex extracts step, reward, loss, entropy, and SPS from the live training output (powers the top two charts immediately)
2. **`status.json`** — train.py writes a structured JSON file atomically every update with win rates, opponent pool info, and detailed metrics (powers the win rate heatmap and per-player-count charts after ~10 updates of data accumulation)

### Web Game Interface

A browser-based interface for playing Oh Hell against AI opponents or getting move advice for real games.

```bash
python gui/run.py                    # opens at http://localhost:8000
python gui/run.py --port 9000        # custom port
```

**Play Mode**: Start a game with 2-5 players. You play one seat; the rest are AI bots (configurable: random, heuristic, smart, or neural network from any snapshot). Bot turns auto-advance. Supports custom hand sizes, trump card selection, and a dev mode that shows all cards face-up.

**Advisor Mode**: Mirror a real-world game in progress. Input the game state step by step — your hand, trump card, bids, cards played — and the neural network recommends optimal bids or card plays with full probability distributions and confidence levels. For a hands-free experience on [Trickster Cards](https://www.trickstercards.com), use the Chrome Extension (`extension/`) which feeds game state automatically.

### Chrome Extension (Trickster Cards Advisor)

A Chrome Extension that overlays real-time AI recommendations directly on the [Trickster Cards](https://www.trickstercards.com) game page. No manual data entry — the extension automatically reads all game events from the page.

**Setup:**

1. Start the backend:
   ```bash
   python gui/run.py
   ```
2. Load the extension in Chrome: `chrome://extensions` → Enable Developer Mode → Load Unpacked → select the `extension/` folder
3. (Optional) Click the extension icon to configure the backend URL and snapshot path

**How it works:**

The extension intercepts Trickster Cards' structured console.log output to detect game events (player joins, cards dealt, bids, plays, trick results). It maps Trickster's player names and seat numbers to the advisor's internal representation, detects the trump card from the DOM, and feeds everything to the backend's AdvisorSession via WebSocket. When it's your turn, the overlay shows bid or play recommendations with probability distributions.

**Autoplay mode:**

Click the gear icon (⚙) on the overlay to open settings. Enable "Autoplay" to have the extension automatically submit bids and play cards based on the top recommendation. The delay slider (50ms–2000ms) controls how long the recommendation is visible before the extension acts.

**Architecture:**

```
[Trickster Cards page]
    │  console.log events
    ▼
[inject.js — page world]
    │  window.postMessage
    ▼
[content.js — content script]
    │  chrome.runtime.connect (Port)
    ▼
[background.js — service worker]
    │  WebSocket to localhost:8000
    ▼
[FastAPI backend — gui/server.py]
    │  AdvisorSession + NN inference
    ▼
[Recommendations → content.js overlay]
```

**Features:**
- Auto-detects game start, rejoin, and player count changes
- Handles non-contiguous Trickster seat numbers
- Recovers from backend restarts (auto-reconnects)
- Recovers from page refresh mid-game (replays round state from bid order)
- Trump card auto-detected from DOM with retry logic for dealing animations
- Draggable, minimizable overlay panel
- Settings persist across sessions via `chrome.storage.local`

### Evaluation and Monitoring

**Terminal dashboard**: Training prints a live dashboard to stdout showing reward, loss, PFSP pool status, win rates by player count, and opponent selection.

**TensorBoard**: Launch it to see loss curves, reward trends, and evaluation metrics over time:
```bash
tensorboard --logdir runs/
```

**Evaluation**: Run a checkpoint against all bot types across all player counts:
```bash
python evaluate.py --checkpoint checkpoints/PPO_ABC123_200M.pt
```

**Plot training progress** from the CSV eval log:
```bash
python evaluate.py --plot runs/PPO_ABC123/eval_log.csv
```

**Play GUI (Tkinter)**: Get move recommendations from a trained agent for real games:
```bash
python play.py --checkpoint checkpoints/PPO_ABC123_200M.pt
```

### Key Parameters

All parameters live in `config.toml` with descriptive comments. CLI arguments override config file values:

```
Precedence: built-in defaults < config.toml < CLI arguments
```

**Essential** (adjust for your hardware and goals):

| Parameter | Section | Default | Description |
|-----------|---------|---------|-------------|
| `total_timesteps` | `[training]` | 1B | How long to train. 100M+ for decent play. |
| `num_envs` | `[training]` | 256 | Parallel environments. Scale to CPU cores. |
| `num_workers` | `[training]` | 0 (auto) | Worker processes. 0 = auto-detect. |
| `hidden_dim` | `[training]` | 512 | Network width. Must match when resuming. |
| `resume` | `[checkpoints]` | "" | Checkpoint path to resume from. |

**Tuning** (adjust if training stalls or diverges):

| Parameter | Section | Default | Description |
|-----------|---------|---------|-------------|
| `lr` | `[training]` | 2.5e-4 | Learning rate. Lower if training is unstable. |
| `ent_coef` | `[training]` | 0.01 | Entropy bonus. Higher = more exploration. |
| `snapshot_interval` | `[selfplay]` | 1M | How often to save opponent snapshots. Must be >= 1M (filename uses `127M` format). |
| `eval_interval` | `[evaluation]` | 10M | How often to run full evaluation. |
| `pfsp_pool_size` | `[pfsp]` | 100 | Max snapshots in opponent pool. |

See [config.toml](config.toml) for the full list organized by section: `[training]`, `[pfsp]`, `[opponents]`, `[selfplay]`, `[player_counts]`, `[evaluation]`, `[checkpoints]`, `[league]`.

## Known Limitations & TODO

### Resolved

- [x] **Per-config reward tracking**: Each opponent now gets its own reward measurement based on the envs where it was actually seated, instead of all 3 opponents getting the same global average.
- [x] **Per-player-count reward tracking**: Workers compute per-(config, player_count) reward breakdowns.
- [x] **Forced homogeneous bot tables**: Configurable rates (`[opponents.forced_homogeneous]`) guarantee the agent faces all-SmartBot, all-HeuristicBot, and all-RandomBot tables regardless of PFSP ranking.
- [x] **NN temperature control**: `nn_temperature` setting controls opponent randomness (1.0 = standard, <1.0 = greedy, >1.0 = random).
- [x] **PFSP parameters configurable**: Exploration bonus, staleness, reward window, pool size, and log interval are all configurable via `[pfsp]` section.
- [x] **TOML config file**: All settings in one place with descriptions.
- [x] **Exploiter agents**: League training with a main agent and exploiter. The exploiter trains exclusively against main agent snapshots to discover weaknesses, and its snapshots are injected into the main agent's PFSP pool. Managed by `league.py` orchestrator.
- [x] **Cython game engine**: `game_fast.pyx` provides a C-accelerated drop-in replacement for the game engine, reducing per-step overhead in the opponent simulation loop.
- [x] **Adaptive player count distribution**: Training automatically shifts toward weaker player counts based on rolling win rates.
- [x] **Training dashboard**: Gradio-based browser UI (`dashboard.py`) for configuring, launching, monitoring, and managing training runs. Includes live metrics, win rate heatmaps, opponent pool viewer, league controls, config presets, and resource monitoring.
- [x] **Web game interface**: FastAPI + WebSocket browser UI (`gui/`) for playing Oh Hell against AI opponents (Play Mode) and getting real-time move recommendations for live games (Advisor Mode).
- [x] **Chrome Extension for Trickster Cards**: Manifest V3 extension (`extension/`) that automatically reads game state from the Trickster Cards page via console.log interception and displays AI recommendations as an in-page overlay. Includes autoplay mode with configurable delay.

### Future Work

- [ ] **Use per-(config, player_count) rewards for PFSP**: Currently per-config rewards are used for PFSP weighting. The per-player-count breakdown is computed but only logged, not used in opponent selection. A future improvement could weight opponents differently by player count.
