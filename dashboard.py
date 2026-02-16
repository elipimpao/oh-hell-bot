"""Gradio-based training dashboard for Oh Hell bot.

Provides a browser UI for configuring, starting/stopping, and monitoring
training runs. Works on Windows, macOS, and Linux.

Usage:
    python dashboard.py                  # opens in browser at localhost:7860
    python dashboard.py --port 8080      # custom port
"""

import os
import sys
import glob
import json
import time
import platform
import subprocess
import threading
import re
import csv
import argparse
from collections import deque
from datetime import datetime, timedelta

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_COUNTS = [2, 3, 4, 5]

_B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _to_base36(n):
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(_B36[n % 36])
        n //= 36
    return "".join(reversed(digits))


# Default config values matching config.toml
DEFAULTS = {
    "total_timesteps": 1_000_000_000,
    "num_envs": 256,
    "num_workers": 0,
    "steps_per_rollout": 512,
    "lr": 2.5e-4,
    "hidden_dim": 512,
    "seed": 42,
    "homogeneous_rate": 0.25,
    "forced_smart": 0.05,
    "forced_heuristic": 0.03,
    "forced_random": 0.02,
    "pc_2p": 0.15,
    "pc_3p": 0.20,
    "pc_4p": 0.30,
    "pc_5p": 0.35,
    "self_play_start": 500_000,
    "snapshot_interval": 1_000_000,
    "checkpoint_interval": 1_000_000,
    "eval_interval": 10_000_000,
    "eval_games": 50,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "epochs": 4,
    "minibatch_size": 4096,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "anneal_lr": True,
}


# ============================================================
#  Training Manager — subprocess lifecycle
# ============================================================

class TrainingManager:
    """Manages a single train.py subprocess."""

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.process = None
        self.stdout_lines = deque(maxlen=5000)
        self.timestamped_lines = deque(maxlen=5000)  # (timestamp, line)
        self.status = "idle"  # idle | running | paused | stopping
        self.current_run_name = None
        self.paused_checkpoint = None
        self._reader_thread = None
        self._lock = threading.Lock()
        self._start_time = None
        self._metrics_history = deque(maxlen=2000)
        self._latest_metrics = {}

    def start(self, config_dict, resume_path=None):
        """Start a training subprocess."""
        with self._lock:
            if self.process and self.process.poll() is None:
                return "Training already running."

            cmd = [sys.executable, "train.py", "--config", "config.toml"]

            # Apply config overrides as CLI args
            cli_map = {
                "total_timesteps": "--total-timesteps",
                "num_envs": "--num-envs",
                "num_workers": "--num-workers",
                "steps_per_rollout": "--steps-per-rollout",
                "lr": "--lr",
                "hidden_dim": "--hidden-dim",
                "seed": "--seed",
                "homogeneous_rate": "--opp-homogeneous-rate",
                "forced_smart": "--forced-smart",
                "forced_heuristic": "--forced-heuristic",
                "forced_random": "--forced-random",
                "self_play_start": "--self-play-start",
                "snapshot_interval": "--snapshot-interval",
                "checkpoint_interval": "--checkpoint-interval",
                "eval_interval": "--eval-interval",
                "eval_games": "--eval-games",
                "ent_coef": "--ent-coef",
                "gamma": "--gamma",
                "gae_lambda": "--gae-lambda",
                "clip_eps": "--clip-eps",
                "epochs": "--epochs",
                "minibatch_size": "--minibatch-size",
                "vf_coef": "--vf-coef",
                "max_grad_norm": "--max-grad-norm",
            }

            for key, flag in cli_map.items():
                if key in config_dict and config_dict[key] is not None:
                    cmd += [flag, str(config_dict[key])]

            # Player count weights
            pc_weights = [
                config_dict.get("pc_2p", DEFAULTS["pc_2p"]),
                config_dict.get("pc_3p", DEFAULTS["pc_3p"]),
                config_dict.get("pc_4p", DEFAULTS["pc_4p"]),
                config_dict.get("pc_5p", DEFAULTS["pc_5p"]),
            ]
            cmd += ["--pc-base-weights"] + [str(w) for w in pc_weights]

            # Anneal LR
            if config_dict.get("anneal_lr", True):
                cmd.append("--anneal-lr")

            # Resume
            if resume_path:
                cmd += ["--resume", resume_path]

            # Spawn subprocess
            kwargs = {}
            if platform.system() == "Windows":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            # Generate run name so we can locate status.json
            run_name = f"PPO_{_to_base36(int(time.time()))}"
            cmd += ["--run-name", run_name]

            self.process = subprocess.Popen(
                cmd, cwd=self.project_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, **kwargs,
            )
            self.status = "running"
            self._start_time = time.time()
            self.stdout_lines.clear()
            self._metrics_history.clear()
            self._latest_metrics = {}
            self.paused_checkpoint = None
            self.current_run_name = run_name

            self._reader_thread = threading.Thread(
                target=self._read_stdout, daemon=True)
            self._reader_thread.start()

            return f"Training started (pid {self.process.pid}). Command:\n{' '.join(cmd)}"

    def stop(self):
        """Stop training gracefully."""
        with self._lock:
            if not self.process or self.process.poll() is not None:
                self.status = "idle"
                return "No training running."
            self.status = "stopping"

        try:
            self.process.terminate()
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

        with self._lock:
            self.status = "idle"
        return f"Training stopped (exit code {self.process.returncode})."

    def pause(self):
        """Pause = stop + remember state for resume."""
        if self.status != "running":
            return "Not running."
        result = self.stop()
        with self._lock:
            self.paused_checkpoint = self._find_latest_checkpoint()
            self.status = "paused"
        return result + f"\nWill resume from: {self.paused_checkpoint or 'N/A'}"

    def resume(self, config_dict):
        """Resume from the latest checkpoint."""
        ckpt = self.paused_checkpoint or self._find_latest_checkpoint()
        if not ckpt:
            return "No checkpoint found to resume from."
        return self.start(config_dict, resume_path=ckpt)

    def get_status(self):
        """Return current status dict."""
        with self._lock:
            running = (self.process is not None
                       and self.process.poll() is None) if self.process else False
            if self.status == "running" and not running:
                self.status = "idle"

            elapsed = ""
            if self._start_time and running:
                dt = timedelta(seconds=int(time.time() - self._start_time))
                elapsed = str(dt)

            return {
                "status": self.status,
                "run_name": self.current_run_name,
                "elapsed": elapsed,
                "pid": self.process.pid if self.process and running else None,
                "metrics": dict(self._latest_metrics),
                "log_tail": list(self.stdout_lines)[-100:],
            }

    def get_status_json(self):
        """Read the status.json file written by train.py."""
        if not self.current_run_name:
            return None
        path = os.path.join(self.project_dir, "runs",
                            self.current_run_name, "status.json")
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _read_stdout(self):
        """Background thread: read stdout and parse metrics."""
        step_re = re.compile(
            r'Step\s+([\d,]+)\s+\[(\w+):([^\]]+)\]\s+\|\s+'
            r'rew\s+([-\d.]+)\s+\|\s+pg\s+([-\d.]+)\s+\|\s+'
            r'vf\s+([-\d.]+)\s+\|\s+ent\s+([-\d.]+)\s+\|\s+SPS\s+(\d+)')
        run_re = re.compile(r'runs[/\\](\S+)')

        try:
            for raw_line in self.process.stdout:
                line = raw_line.rstrip("\n\r")
                now = time.time()
                with self._lock:
                    self.stdout_lines.append(line)
                    self.timestamped_lines.append((now, line))

                    # Detect run name
                    if self.current_run_name is None:
                        m = run_re.search(line)
                        if m:
                            # Extract run name (path after runs/)
                            rn = m.group(1).rstrip("/\\")
                            if "/" in rn:
                                rn = rn.split("/")[0]
                            if "\\" in rn:
                                rn = rn.split("\\")[0]
                            self.current_run_name = rn

                    # Parse dashboard line
                    m = step_re.match(line)
                    if m:
                        metrics = {
                            "step": int(m.group(1).replace(",", "")),
                            "mode": m.group(2),
                            "opponents": m.group(3),
                            "reward": float(m.group(4)),
                            "pg_loss": float(m.group(5)),
                            "v_loss": float(m.group(6)),
                            "entropy": float(m.group(7)),
                            "sps": int(m.group(8)),
                            "timestamp": time.time(),
                        }
                        self._latest_metrics = metrics
                        self._metrics_history.append(metrics)
        except (ValueError, OSError):
            pass

    def _find_latest_checkpoint(self):
        """Find the most recent checkpoint file across all subdirectories."""
        base = os.path.join(self.project_dir, "checkpoints")
        dirs = [base]
        if os.path.isdir(base):
            for entry in os.listdir(base):
                full = os.path.join(base, entry)
                if os.path.isdir(full):
                    dirs.append(full)
        files = []
        for d in dirs:
            for pat in ["*_CHKPT_*.pt", "*_FINAL.pt"]:
                files.extend(glob.glob(os.path.join(d, pat)))
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def get_metrics_history(self):
        """Return metrics history as a list."""
        with self._lock:
            return list(self._metrics_history)


# ============================================================
#  League Manager — multi-agent orchestration
# ============================================================

class LeagueManager:
    """Manages main + exploiter agents as separate TrainingManagers."""

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.agents = {}

    def start_agent(self, name, role, config_overrides,
                    resume_path=None, init_weights=None,
                    snapshot_dir=None, load_dirs=None):
        """Start a league agent."""
        if name in self.agents and self.agents[name].status == "running":
            return f"Agent {name} already running."

        mgr = TrainingManager(self.project_dir)
        cmd = [sys.executable, "train.py", "--config", "config.toml"]

        # Run name with proper prefix (MAIN_ or EXP_)
        role_prefix = "EXP" if role == "exploiter" else "MAIN"
        run_name = f"{role_prefix}_{_to_base36(int(time.time()))}"
        cmd += ["--run-name", run_name]

        # Checkpoint dir: each role saves to its own subdirectory
        save_dir = os.path.join("checkpoints", name)
        cmd += ["--save-dir", save_dir]

        # Role-specific flags
        if role == "exploiter":
            cmd.append("--exploiter-mode")

        if snapshot_dir:
            cmd += ["--snapshot-dir", snapshot_dir]
        if load_dirs:
            cmd += ["--load-dirs", load_dirs]
        cmd += ["--rescan-interval", "1"]

        if resume_path:
            cmd += ["--resume", resume_path]
        elif init_weights:
            cmd += ["--init-weights", init_weights]

        # Apply overrides
        for key, val in config_overrides.items():
            cli_key = "--" + key.replace("_", "-")
            cmd += [cli_key, str(val)]

        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        mgr.process = subprocess.Popen(
            cmd, cwd=self.project_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, **kwargs,
        )
        mgr.status = "running"
        mgr.current_run_name = run_name
        mgr._start_time = time.time()
        mgr._reader_thread = threading.Thread(
            target=mgr._read_stdout, daemon=True)
        mgr._reader_thread.start()

        self.agents[name] = mgr
        return f"Agent {name} ({role}) started (pid {mgr.process.pid})."

    def stop_agent(self, name):
        if name not in self.agents:
            return f"Agent {name} not found."
        return self.agents[name].stop()

    def stop_all(self):
        results = []
        for name in list(self.agents):
            results.append(self.stop_agent(name))
        return "\n".join(results)

    def get_agent_status(self, name):
        if name not in self.agents:
            return {"status": "idle"}
        return self.agents[name].get_status()


# ============================================================
#  Utility functions
# ============================================================

def discover_checkpoints(subdirs=None):
    """Find all checkpoint files, sorted newest first.

    Args:
        subdirs: List of subdirectory names under checkpoints/ to search.
                 None searches checkpoints/ root and all subdirectories.
    """
    base = os.path.join(PROJECT_DIR, "checkpoints")
    if subdirs is not None:
        dirs = [os.path.join(base, s) for s in subdirs]
    else:
        # Search root + all subdirectories
        dirs = [base]
        if os.path.isdir(base):
            for entry in os.listdir(base):
                full = os.path.join(base, entry)
                if os.path.isdir(full):
                    dirs.append(full)
    files = []
    for d in dirs:
        for pat in ["*_CHKPT_*.pt", "*_FINAL.pt"]:
            files.extend(glob.glob(os.path.join(d, pat)))
    result = []
    for f in files:
        result.append({
            "path": f,
            "name": os.path.basename(f),
            "mtime": os.path.getmtime(f),
            "size_mb": os.path.getsize(f) / (1024 * 1024),
        })
    return sorted(result, key=lambda x: x["mtime"], reverse=True)


def discover_snapshots(dirs=None):
    """Find all snapshot files across directories."""
    if dirs is None:
        dirs = ["snapshots", "league_snapshots/main", "league_snapshots/exploiter"]
    result = []
    for d in dirs:
        full = os.path.join(PROJECT_DIR, d)
        if not os.path.isdir(full):
            continue
        for f in glob.glob(os.path.join(full, "*.pt")):
            result.append({
                "path": f,
                "name": os.path.basename(f),
                "dir": d,
                "mtime": os.path.getmtime(f),
                "size_mb": os.path.getsize(f) / (1024 * 1024),
            })
    return sorted(result, key=lambda x: x["mtime"], reverse=True)


def discover_runs():
    """Scan runs/ for past training runs."""
    runs_dir = os.path.join(PROJECT_DIR, "runs")
    if not os.path.isdir(runs_dir):
        return []
    result = []
    for name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, name)
        if not os.path.isdir(run_path):
            continue
        csv_path = os.path.join(run_path, "eval_log.csv")
        status_path = os.path.join(run_path, "status.json")
        latest_step = 0
        if os.path.exists(status_path):
            try:
                with open(status_path, "r") as f:
                    s = json.load(f)
                    latest_step = s.get("global_step", 0)
            except (json.JSONDecodeError, OSError):
                pass
        elif os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        latest_step = int(lines[-1].split(",")[0])
            except (ValueError, OSError):
                pass
        result.append({
            "name": name,
            "latest_step": latest_step,
            "created": os.path.getctime(run_path),
            "has_eval": os.path.exists(csv_path),
        })
    return sorted(result, key=lambda x: x["created"], reverse=True)


def read_eval_csv(run_name):
    """Read eval_log.csv and return as dict of lists."""
    csv_path = os.path.join(PROJECT_DIR, "runs", run_name, "eval_log.csv")
    if not os.path.exists(csv_path):
        return None
    data = {}
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    if key not in data:
                        data[key] = []
                    try:
                        data[key].append(float(val))
                    except ValueError:
                        data[key].append(0.0)
    except OSError:
        return None
    return data if data else None


def checkpoint_choices(role=None):
    """Return list of checkpoint names for dropdown.

    Args:
        role: "main" or "exploiter" to filter by subdirectory.
              None returns all checkpoints.
    """
    subdirs = [role] if role else None
    ckpts = discover_checkpoints(subdirs=subdirs)
    return [c["name"] for c in ckpts]


def get_resource_info():
    """Get CPU/memory usage."""
    if not HAS_PSUTIL:
        return "Install psutil for resource monitoring: pip install psutil"
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    info = f"CPU: {cpu:.1f}%\nMemory: {mem.percent:.1f}% ({mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB)"
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem_alloc = torch.cuda.memory_allocated(i) / 1e9
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                info += f"\nGPU {i} ({name}): {mem_alloc:.1f} / {mem_total:.1f} GB"
    except ImportError:
        pass
    return info


# ============================================================
#  Chart builders
# ============================================================

def build_reward_loss_chart(metrics_history):
    """Build reward and loss curves from stdout-parsed metrics."""
    if not metrics_history:
        fig = go.Figure()
        fig.update_layout(title="Reward & Loss (waiting for data...)",
                          height=350)
        return fig

    steps = [m["step"] for m in metrics_history]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=steps, y=[m["reward"] for m in metrics_history],
        name="Avg Reward", line=dict(color="#2196F3", width=2),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=steps, y=[m["pg_loss"] for m in metrics_history],
        name="Policy Loss", line=dict(color="#FF5722", width=1.5, dash="dot"),
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=steps, y=[m["v_loss"] for m in metrics_history],
        name="Value Loss", line=dict(color="#FF9800", width=1.5, dash="dash"),
    ), secondary_y=True)

    fig.update_layout(
        title="Reward & Loss", height=350,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Reward", secondary_y=False)
    fig.update_yaxes(title_text="Loss", secondary_y=True)
    return fig


def build_winrate_heatmap(status_json):
    """Build a 3x4 win rate heatmap from status.json cell_win_rates."""
    bots = ["smart", "heuristic", "random"]
    bot_labels = ["Smart", "Heuristic", "Random"]
    pc_labels = ["2p", "3p", "4p", "5p"]

    if not status_json or "cell_win_rates" not in status_json:
        fig = go.Figure()
        fig.update_layout(
            title="Win Rate Grid (waiting for status.json...)", height=300,
            margin=dict(l=80, r=30, t=40, b=30),
        )
        return fig

    cwr = status_json["cell_win_rates"]
    has_any = any(cwr.get(f"{b}_{pc}") is not None
                  for b in bots for pc in PLAYER_COUNTS)

    z = []
    for bot in bots:
        row = []
        for pc in PLAYER_COUNTS:
            val = cwr.get(f"{bot}_{pc}")
            row.append(val * 100 if val is not None else None)
        z.append(row)

    title = "Win Rate Grid (Training)"
    if not has_any:
        title = "Win Rate Grid (accumulating samples...)"

    fig = go.Figure(data=go.Heatmap(
        z=z, x=pc_labels, y=bot_labels,
        colorscale="RdYlGn", zmin=0, zmax=100,
        text=[[f"{v:.0f}%" if v is not None else "--" for v in row] for row in z],
        texttemplate="%{text}", textfont={"size": 14},
        hovertemplate="vs %{y} %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=300,
        margin=dict(l=80, r=30, t=40, b=30),
    )
    return fig


def build_entropy_sps_chart(metrics_history):
    """Build entropy and SPS chart."""
    if not metrics_history:
        fig = go.Figure()
        fig.update_layout(title="Entropy & SPS (waiting for data...)",
                          height=350)
        return fig

    steps = [m["step"] for m in metrics_history]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=steps, y=[m["entropy"] for m in metrics_history],
        name="Entropy", line=dict(color="#9C27B0", width=2),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=steps, y=[m["sps"] for m in metrics_history],
        name="SPS", line=dict(color="#4CAF50", width=1.5),
    ), secondary_y=True)

    fig.update_layout(
        title="Entropy & Throughput", height=350,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Entropy", secondary_y=False)
    fig.update_yaxes(title_text="Steps/sec", secondary_y=True)
    return fig


def build_eval_charts(run_name):
    """Build 3x3 eval chart grid from CSV data."""
    data = read_eval_csv(run_name) if run_name else None
    if not data or "step" not in data:
        fig = go.Figure()
        fig.update_layout(title="No evaluation data yet", height=600)
        return fig

    steps = data["step"]
    metrics = ["score", "win", "bid_acc"]
    metric_labels = ["Average Score", "Win Rate", "Bid Accuracy"]
    opponents = ["random", "heuristic", "smart"]
    opp_labels = ["vs Random", "vs Heuristic", "vs Smart"]
    colors = {2: "#F44336", 3: "#4CAF50", 4: "#2196F3", 5: "#9C27B0"}

    fig = make_subplots(
        rows=3, cols=3, subplot_titles=[
            f"{ml} {ol}" for ml in metric_labels for ol in opp_labels
        ],
        vertical_spacing=0.08, horizontal_spacing=0.06,
    )

    for row, metric in enumerate(metrics, 1):
        for col, opp in enumerate(opponents, 1):
            for pc in PLAYER_COUNTS:
                key = f"{pc}p_vs_{opp}_{metric}"
                if key in data:
                    fig.add_trace(go.Scatter(
                        x=steps, y=data[key],
                        name=f"{pc}p", line=dict(color=colors[pc], width=1.5),
                        legendgroup=f"{pc}p",
                        showlegend=(row == 1 and col == 1),
                    ), row=row, col=col)
            agg_key = f"agg_vs_{opp}_{metric}"
            if agg_key in data:
                fig.add_trace(go.Scatter(
                    x=steps, y=data[agg_key],
                    name="Aggregate", line=dict(color="black", width=2.5, dash="dash"),
                    legendgroup="agg",
                    showlegend=(row == 1 and col == 1),
                ), row=row, col=col)

    fig.update_layout(height=700, margin=dict(t=50, b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def build_pc_winrate_chart(status_json):
    """Build per-player-count win rate bar chart."""
    if not status_json or "pc_win_rates" not in status_json:
        fig = go.Figure()
        fig.update_layout(title="PC Win Rates (waiting for status.json...)",
                          height=300)
        return fig

    pwr = status_json["pc_win_rates"]
    has_any = any(pwr.get(str(pc)) is not None for pc in PLAYER_COUNTS)

    labels = []
    values = []
    colors = ["#F44336", "#4CAF50", "#2196F3", "#9C27B0"]
    for pc in PLAYER_COUNTS:
        labels.append(f"{pc}p")
        val = pwr.get(str(pc))
        values.append(val * 100 if val is not None else 0)

    title = "Win Rate by Player Count"
    if not has_any:
        title = "Win Rate by Player Count (accumulating samples...)"

    fig = go.Figure(data=go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.0f}%" if v > 0 else "--" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title=title, height=300,
        yaxis=dict(range=[0, 100], title="Win Rate %"),
        margin=dict(l=50, r=30, t=40, b=30),
    )
    return fig


# ============================================================
#  Config presets
# ============================================================

PRESETS_DIR = os.path.join(PROJECT_DIR, "presets")


def list_presets():
    """List saved config preset names."""
    if not os.path.isdir(PRESETS_DIR):
        return []
    return [f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]


def save_preset(name, config_dict):
    """Save a config preset."""
    os.makedirs(PRESETS_DIR, exist_ok=True)
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
    return f"Preset '{name}' saved."


def load_preset(name):
    """Load a config preset."""
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
#  Gradio UI
# ============================================================

manager = TrainingManager(PROJECT_DIR)
league_mgr = LeagueManager(PROJECT_DIR)


def build_ui():
    with gr.Blocks(title="Oh Hell Bot - Training Dashboard") as app:

        gr.Markdown("# Oh Hell Bot — Training Dashboard")

        # ---- Tab 1: Training Control ----
        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Configuration")

                    with gr.Accordion("Training Parameters", open=True):
                        with gr.Row():
                            total_ts = gr.Number(
                                label="Total Timesteps",
                                value=DEFAULTS["total_timesteps"], precision=0)
                            num_envs = gr.Number(
                                label="Num Envs",
                                value=DEFAULTS["num_envs"], precision=0)
                        with gr.Row():
                            num_workers = gr.Number(
                                label="Num Workers (0=auto)",
                                value=DEFAULTS["num_workers"], precision=0)
                            hidden_dim = gr.Dropdown(
                                label="Hidden Dim",
                                choices=[256, 512, 1024],
                                value=DEFAULTS["hidden_dim"])
                        with gr.Row():
                            lr_input = gr.Number(
                                label="Learning Rate",
                                value=DEFAULTS["lr"])
                            seed_input = gr.Number(
                                label="Seed",
                                value=DEFAULTS["seed"], precision=0)
                        with gr.Row():
                            ent_coef = gr.Number(
                                label="Entropy Coef",
                                value=DEFAULTS["ent_coef"])
                            anneal_lr = gr.Checkbox(
                                label="Anneal LR",
                                value=DEFAULTS["anneal_lr"])

                    with gr.Accordion("Opponent Composition", open=True):
                        homog_rate = gr.Slider(
                            label="Homogeneous Table Rate",
                            minimum=0, maximum=1.0, step=0.05,
                            value=DEFAULTS["homogeneous_rate"])
                        with gr.Row():
                            f_smart = gr.Slider(
                                label="Forced Smart %",
                                minimum=0, maximum=0.2, step=0.01,
                                value=DEFAULTS["forced_smart"])
                            f_heur = gr.Slider(
                                label="Forced Heuristic %",
                                minimum=0, maximum=0.2, step=0.01,
                                value=DEFAULTS["forced_heuristic"])
                            f_rand = gr.Slider(
                                label="Forced Random %",
                                minimum=0, maximum=0.2, step=0.01,
                                value=DEFAULTS["forced_random"])

                    with gr.Accordion("Player Count Distribution", open=True):
                        with gr.Row():
                            pc_2p = gr.Slider(label="2p Weight", minimum=0,
                                              maximum=1.0, step=0.05,
                                              value=DEFAULTS["pc_2p"])
                            pc_3p = gr.Slider(label="3p Weight", minimum=0,
                                              maximum=1.0, step=0.05,
                                              value=DEFAULTS["pc_3p"])
                        with gr.Row():
                            pc_4p = gr.Slider(label="4p Weight", minimum=0,
                                              maximum=1.0, step=0.05,
                                              value=DEFAULTS["pc_4p"])
                            pc_5p = gr.Slider(label="5p Weight", minimum=0,
                                              maximum=1.0, step=0.05,
                                              value=DEFAULTS["pc_5p"])

                    with gr.Accordion("Intervals & Saving", open=False):
                        with gr.Row():
                            sp_start = gr.Number(
                                label="Self-Play Start Step",
                                value=DEFAULTS["self_play_start"], precision=0)
                            snap_int = gr.Number(
                                label="Snapshot Interval (>= 1M)",
                                value=DEFAULTS["snapshot_interval"], precision=0)
                        with gr.Row():
                            ckpt_int = gr.Number(
                                label="Checkpoint Interval (>= 1M)",
                                value=DEFAULTS["checkpoint_interval"], precision=0)
                            eval_int = gr.Number(
                                label="Eval Interval",
                                value=DEFAULTS["eval_interval"], precision=0)
                        eval_games_input = gr.Number(
                            label="Eval Games",
                            value=DEFAULTS["eval_games"], precision=0)

                    with gr.Accordion("Resume from Checkpoint", open=False):
                        resume_dd = gr.Dropdown(
                            label="Resume From",
                            choices=checkpoint_choices(),
                            value=None, allow_custom_value=True)
                        refresh_ckpt_btn = gr.Button("Refresh Checkpoints",
                                                     size="sm")

                with gr.Column(scale=1):
                    gr.Markdown("### Controls")
                    with gr.Row():
                        start_btn = gr.Button("Start", variant="primary",
                                              size="lg")
                    with gr.Row():
                        pause_btn = gr.Button("Pause", size="lg")
                        resume_btn = gr.Button("Resume", size="lg")
                    with gr.Row():
                        stop_btn = gr.Button("Stop", variant="stop", size="lg")

                    gr.Markdown("### Status")
                    status_text = gr.Textbox(
                        label="Status", lines=6, interactive=False,
                        value="Idle")
                    output_msg = gr.Textbox(
                        label="Last Action", lines=3, interactive=False)

            # Config inputs list for gathering values
            config_inputs = [
                total_ts, num_envs, num_workers, hidden_dim,
                lr_input, seed_input, ent_coef, anneal_lr,
                homog_rate, f_smart, f_heur, f_rand,
                pc_2p, pc_3p, pc_4p, pc_5p,
                sp_start, snap_int, ckpt_int, eval_int,
                eval_games_input, resume_dd,
            ]

            def gather_config(*values):
                keys = [
                    "total_timesteps", "num_envs", "num_workers", "hidden_dim",
                    "lr", "seed", "ent_coef", "anneal_lr",
                    "homogeneous_rate", "forced_smart", "forced_heuristic",
                    "forced_random",
                    "pc_2p", "pc_3p", "pc_4p", "pc_5p",
                    "self_play_start", "snapshot_interval",
                    "checkpoint_interval", "eval_interval",
                    "eval_games", "_resume",
                ]
                return dict(zip(keys, values))

            def on_start(*values):
                cfg = gather_config(*values)
                resume = cfg.pop("_resume", None)
                if resume:
                    # Resolve to full path: check subdirs then root
                    full = None
                    for sub in ["main", "exploiter", ""]:
                        p = os.path.join(PROJECT_DIR, "checkpoints", sub, resume)
                        if os.path.exists(p):
                            full = p
                            break
                    if not full:
                        full = resume  # user may have typed full path
                    return manager.start(cfg, resume_path=full)
                return manager.start(cfg)

            def on_pause():
                return manager.pause()

            def on_resume(*values):
                cfg = gather_config(*values)
                cfg.pop("_resume", None)
                return manager.resume(cfg)

            def on_stop():
                return manager.stop()

            def refresh_ckpts():
                return gr.update(choices=checkpoint_choices())

            start_btn.click(on_start, inputs=config_inputs,
                            outputs=output_msg)
            pause_btn.click(on_pause, outputs=output_msg)
            resume_btn.click(on_resume, inputs=config_inputs,
                             outputs=output_msg)
            stop_btn.click(on_stop, outputs=output_msg)
            refresh_ckpt_btn.click(refresh_ckpts, outputs=resume_dd)

        # ---- Tab 2: Dashboard ----
        with gr.Tab("Dashboard"):
            dash_status = gr.Markdown("**Status:** Idle")
            with gr.Row():
                dash_reward = gr.Textbox(label="Avg Reward", value="--",
                                         interactive=False)
                dash_pg = gr.Textbox(label="Policy Loss", value="--",
                                      interactive=False)
                dash_vf = gr.Textbox(label="Value Loss", value="--",
                                      interactive=False)
                dash_ent = gr.Textbox(label="Entropy", value="--",
                                      interactive=False)
                dash_sps = gr.Textbox(label="SPS", value="--",
                                      interactive=False)

            with gr.Row():
                reward_loss_plot = gr.Plot(label="Reward & Loss")
                entropy_sps_plot = gr.Plot(label="Entropy & SPS")

            with gr.Row():
                winrate_heatmap = gr.Plot(label="Win Rate Grid")
                pc_winrate_plot = gr.Plot(label="Win Rate by Player Count")

        # ---- Tab 3: Evaluation ----
        with gr.Tab("Evaluation"):
            with gr.Row():
                eval_run_dd = gr.Dropdown(
                    label="Select Run",
                    choices=[r["name"] for r in discover_runs()],
                    value=None)
                eval_refresh_btn = gr.Button("Refresh", size="sm")
            eval_plot = gr.Plot(label="Evaluation History")

            gr.Markdown("### On-Demand Evaluation")
            with gr.Row():
                eval_ckpt_dd = gr.Dropdown(
                    label="Checkpoint", choices=checkpoint_choices())
                eval_ngames = gr.Number(label="Games per config",
                                        value=50, precision=0)
                eval_device = gr.Dropdown(
                    label="Device", choices=["cpu", "cuda"], value="cpu")
                eval_run_btn = gr.Button("Run Evaluation", variant="primary")
            eval_output = gr.Textbox(label="Evaluation Results", lines=20,
                                     interactive=False)

            def on_eval_refresh():
                runs = [r["name"] for r in discover_runs()]
                return gr.update(choices=runs)

            def on_eval_run(ckpt_name, n_games, device):
                if not ckpt_name:
                    return "Select a checkpoint first."
                # Check subdirs then root
                ckpt_path = None
                for sub in ["main", "exploiter", ""]:
                    p = os.path.join(PROJECT_DIR, "checkpoints", sub, ckpt_name)
                    if os.path.exists(p):
                        ckpt_path = p
                        break
                if not ckpt_path:
                    return f"Checkpoint not found: {ckpt_name}"
                try:
                    result = subprocess.run(
                        [sys.executable, "evaluate.py",
                         "--checkpoint", ckpt_path,
                         "--num-games", str(int(n_games)),
                         "--device", device],
                        cwd=PROJECT_DIR, capture_output=True, text=True,
                        timeout=600,
                    )
                    return result.stdout + result.stderr
                except subprocess.TimeoutExpired:
                    return "Evaluation timed out (10 min limit)."
                except Exception as e:
                    return f"Error: {e}"

            def on_eval_chart(run_name):
                return build_eval_charts(run_name)

            eval_refresh_btn.click(on_eval_refresh, outputs=eval_run_dd)
            eval_run_btn.click(on_eval_run,
                               inputs=[eval_ckpt_dd, eval_ngames, eval_device],
                               outputs=eval_output)
            eval_run_dd.change(on_eval_chart, inputs=eval_run_dd,
                               outputs=eval_plot)

        # ---- Tab 4: Opponent Pool ----
        with gr.Tab("Opponent Pool"):
            gr.Markdown("### Current Pool")
            pool_table = gr.Dataframe(
                headers=["ID", "Type", "Step", "Win Rate", "Games"],
                label="Opponent Pool",
                interactive=False)
            pool_refresh_btn = gr.Button("Refresh Pool", size="sm")

            gr.Markdown("### Snapshots on Disk")
            snap_table = gr.Dataframe(
                headers=["Name", "Directory", "Size (MB)", "Date"],
                label="Snapshot Files",
                interactive=False)
            snap_refresh_btn = gr.Button("Refresh Snapshots", size="sm")

            def refresh_pool():
                sj = manager.get_status_json()
                if not sj or "opponent_pool" not in sj:
                    return []
                rows = []
                for e in sj["opponent_pool"]:
                    wr = f"{e['win_rate']*100:.1f}%" if e.get("win_rate") is not None else "--"
                    rows.append([
                        e["id"], e["type"], str(e.get("step", "")),
                        wr, str(e.get("games", 0)),
                    ])
                return rows

            def refresh_snaps():
                snaps = discover_snapshots()
                rows = []
                for s in snaps:
                    dt = datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M")
                    rows.append([s["name"], s["dir"],
                                 f"{s['size_mb']:.1f}", dt])
                return rows

            pool_refresh_btn.click(refresh_pool, outputs=pool_table)
            snap_refresh_btn.click(refresh_snaps, outputs=snap_table)

        # ---- Tab 5: League ----
        with gr.Tab("League"):
            gr.Markdown("### League Training (Main + Exploiter)")

            with gr.Row():
                # Main agent
                with gr.Column():
                    gr.Markdown("#### Main Agent")
                    _main_ckpts = checkpoint_choices("main")
                    main_resume = gr.Dropdown(
                        label="Resume Checkpoint",
                        choices=_main_ckpts,
                        value=_main_ckpts[0] if _main_ckpts else None,
                        allow_custom_value=True)
                    main_envs = gr.Number(label="Num Envs", value=256,
                                          precision=0)
                    main_workers = gr.Number(label="Num Workers", value=16,
                                             precision=0)
                    main_snap_dir = gr.Textbox(
                        label="Snapshot Dir",
                        value="league_snapshots/main")
                    main_load_dirs = gr.Textbox(
                        label="Load Dirs (comma-separated)",
                        value="league_snapshots/exploiter,snapshots")
                    with gr.Row():
                        main_start_btn = gr.Button("Start Main",
                                                   variant="primary")
                        main_stop_btn = gr.Button("Stop Main",
                                                  variant="stop")
                    main_status = gr.Textbox(label="Main Status",
                                             interactive=False, lines=3)

                # Exploiter agent
                with gr.Column():
                    gr.Markdown("#### Exploiter Agent")
                    _exp_ckpts = checkpoint_choices("exploiter")
                    _exp_snaps = [s["name"] for s in discover_snapshots()]
                    with gr.Row():
                        exp_no_resume = gr.Checkbox(
                            label="Do not resume from checkpoint",
                            value=False)
                        exp_resume = gr.Dropdown(
                            label="Resume Checkpoint",
                            choices=_exp_ckpts,
                            value=_exp_ckpts[0] if _exp_ckpts else None,
                            allow_custom_value=True)
                    with gr.Row(visible=False) as exp_init_row:
                        exp_no_init = gr.Checkbox(
                            label="Start from scratch",
                            value=False)
                        exp_init = gr.Dropdown(
                            label="Init Weights (snapshot)",
                            choices=_exp_snaps,
                            value=_exp_snaps[0] if _exp_snaps else None,
                            allow_custom_value=True)
                    exp_envs = gr.Number(label="Num Envs", value=128,
                                         precision=0)
                    exp_workers = gr.Number(label="Num Workers", value=8,
                                            precision=0)
                    exp_snap_dir = gr.Textbox(
                        label="Snapshot Dir",
                        value="league_snapshots/exploiter")
                    exp_load_dirs = gr.Textbox(
                        label="Load Dirs",
                        value="league_snapshots/main")
                    exp_total_ts = gr.Number(
                        label="Total Timesteps", value=1_000_000_000,
                        precision=0)
                    with gr.Row():
                        exp_start_btn = gr.Button("Start Exploiter",
                                                  variant="primary")
                        exp_stop_btn = gr.Button("Stop Exploiter",
                                                 variant="stop")
                    exp_status = gr.Textbox(label="Exploiter Status",
                                            interactive=False, lines=3)

                    def on_no_resume_toggle(checked):
                        # When "do not resume" is checked, disable dropdown
                        # and show the init weights row
                        return (gr.update(interactive=not checked,
                                          value=None if checked else gr.update()),
                                gr.update(visible=checked))

                    def on_no_init_toggle(checked):
                        return gr.update(interactive=not checked,
                                         value=None if checked else gr.update())

                    exp_no_resume.change(on_no_resume_toggle,
                                         inputs=exp_no_resume,
                                         outputs=[exp_resume, exp_init_row])
                    exp_no_init.change(on_no_init_toggle,
                                       inputs=exp_no_init,
                                       outputs=exp_init)

            league_refresh_btn = gr.Button("Refresh League Checkpoints",
                                           size="sm")
            league_stop_all_btn = gr.Button("Stop All Agents",
                                            variant="stop")
            league_output = gr.Textbox(label="League Output",
                                       interactive=False, lines=3)

            def _resolve_checkpoint(name, role):
                """Resolve a checkpoint name to full path, checking role subdir first."""
                if not name:
                    return None
                # Check role-specific subdir first
                p = os.path.join(PROJECT_DIR, "checkpoints", role, name)
                if os.path.exists(p):
                    return p
                # Fall back to root checkpoints/ (legacy)
                p = os.path.join(PROJECT_DIR, "checkpoints", name)
                if os.path.exists(p):
                    return p
                return name  # user may have typed full path

            def start_main(resume, envs, workers, snap_dir, load_dirs):
                resume_path = _resolve_checkpoint(resume, "main")
                return league_mgr.start_agent(
                    "main", "main",
                    {"num_envs": int(envs), "num_workers": int(workers)},
                    resume_path=resume_path,
                    snapshot_dir=snap_dir,
                    load_dirs=load_dirs,
                )

            def start_exploiter(no_resume, resume, no_init, init_w, envs,
                                workers, snap_dir, load_dirs, total):
                resume_path = None
                if not no_resume and resume:
                    resume_path = _resolve_checkpoint(resume, "exploiter")
                init_path = None
                if not no_init and not resume_path and init_w:
                    # Try multiple directories
                    for d in ["league_snapshots/main", "snapshots"]:
                        p = os.path.join(PROJECT_DIR, d, init_w)
                        if os.path.exists(p):
                            init_path = p
                            break
                    if not init_path:
                        init_path = init_w
                return league_mgr.start_agent(
                    "exploiter", "exploiter",
                    {"num_envs": int(envs), "num_workers": int(workers),
                     "total_timesteps": int(total)},
                    resume_path=resume_path,
                    init_weights=init_path,
                    snapshot_dir=snap_dir,
                    load_dirs=load_dirs,
                )

            def stop_main():
                return league_mgr.stop_agent("main")

            def stop_exploiter():
                return league_mgr.stop_agent("exploiter")

            def stop_all_league():
                return league_mgr.stop_all()

            def refresh_league():
                main_ckpts = checkpoint_choices("main")
                exp_ckpts = checkpoint_choices("exploiter")
                snaps = [s["name"] for s in discover_snapshots()]
                return (gr.update(choices=main_ckpts),
                        gr.update(choices=exp_ckpts),
                        gr.update(choices=snaps))

            main_start_btn.click(
                start_main,
                inputs=[main_resume, main_envs, main_workers,
                        main_snap_dir, main_load_dirs],
                outputs=league_output)
            exp_start_btn.click(
                start_exploiter,
                inputs=[exp_no_resume, exp_resume, exp_no_init, exp_init,
                        exp_envs, exp_workers, exp_snap_dir, exp_load_dirs,
                        exp_total_ts],
                outputs=league_output)
            main_stop_btn.click(stop_main, outputs=league_output)
            exp_stop_btn.click(stop_exploiter, outputs=league_output)
            league_stop_all_btn.click(stop_all_league, outputs=league_output)
            league_refresh_btn.click(refresh_league,
                                     outputs=[main_resume, exp_resume,
                                              exp_init])

        # ---- Tab 6: Settings ----
        with gr.Tab("Settings"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Config Presets")
                    preset_name = gr.Textbox(label="Preset Name",
                                             placeholder="my_config")
                    preset_dd = gr.Dropdown(label="Load Preset",
                                            choices=list_presets())
                    with gr.Row():
                        save_preset_btn = gr.Button("Save Current Config")
                        load_preset_btn = gr.Button("Load Preset")
                    preset_msg = gr.Textbox(label="Preset Status",
                                            interactive=False)

                with gr.Column():
                    gr.Markdown("### Run History")
                    run_table = gr.Dataframe(
                        headers=["Run Name", "Latest Step", "Has Eval",
                                 "Created"],
                        label="Past Runs", interactive=False)
                    run_refresh_btn = gr.Button("Refresh", size="sm")

            gr.Markdown("### Resource Monitor")
            resource_text = gr.Textbox(label="System Resources", lines=4,
                                       interactive=False,
                                       value=get_resource_info())

            gr.Markdown("### Training Log")
            log_text = gr.Textbox(label="Training Log (scrollable)",
                                  lines=25, interactive=False,
                                  autoscroll=True)

            # Preset handlers — need all config inputs
            # We'll use the same config_inputs from Training tab
            # but Gradio requires them to be in the same Blocks scope
            # So we handle presets differently — save from current form state

            def on_save_preset(name, *values):
                if not name:
                    return "Enter a preset name.", gr.update()
                cfg = gather_config(*values)
                cfg.pop("_resume", None)
                result = save_preset(name, cfg)
                return result, gr.update(choices=list_presets())

            def on_load_preset(name):
                if not name:
                    return ("Select a preset.",) + tuple(
                        gr.update() for _ in config_inputs)
                cfg = load_preset(name)
                if not cfg:
                    return (f"Preset '{name}' not found.",) + tuple(
                        gr.update() for _ in config_inputs)
                # Map config keys back to UI components
                vals = [
                    cfg.get("total_timesteps", DEFAULTS["total_timesteps"]),
                    cfg.get("num_envs", DEFAULTS["num_envs"]),
                    cfg.get("num_workers", DEFAULTS["num_workers"]),
                    cfg.get("hidden_dim", DEFAULTS["hidden_dim"]),
                    cfg.get("lr", DEFAULTS["lr"]),
                    cfg.get("seed", DEFAULTS["seed"]),
                    cfg.get("ent_coef", DEFAULTS["ent_coef"]),
                    cfg.get("anneal_lr", DEFAULTS["anneal_lr"]),
                    cfg.get("homogeneous_rate", DEFAULTS["homogeneous_rate"]),
                    cfg.get("forced_smart", DEFAULTS["forced_smart"]),
                    cfg.get("forced_heuristic", DEFAULTS["forced_heuristic"]),
                    cfg.get("forced_random", DEFAULTS["forced_random"]),
                    cfg.get("pc_2p", DEFAULTS["pc_2p"]),
                    cfg.get("pc_3p", DEFAULTS["pc_3p"]),
                    cfg.get("pc_4p", DEFAULTS["pc_4p"]),
                    cfg.get("pc_5p", DEFAULTS["pc_5p"]),
                    cfg.get("self_play_start", DEFAULTS["self_play_start"]),
                    cfg.get("snapshot_interval", DEFAULTS["snapshot_interval"]),
                    cfg.get("checkpoint_interval",
                            DEFAULTS["checkpoint_interval"]),
                    cfg.get("eval_interval", DEFAULTS["eval_interval"]),
                    cfg.get("eval_games", DEFAULTS["eval_games"]),
                    None,  # resume dropdown
                ]
                return (f"Loaded '{name}'.",) + tuple(vals)

            save_preset_btn.click(
                on_save_preset,
                inputs=[preset_name] + config_inputs,
                outputs=[preset_msg, preset_dd])
            load_preset_btn.click(
                on_load_preset,
                inputs=preset_dd,
                outputs=[preset_msg] + config_inputs)

            def refresh_runs():
                runs = discover_runs()
                rows = []
                for r in runs:
                    dt = datetime.fromtimestamp(r["created"]).strftime(
                        "%Y-%m-%d %H:%M")
                    rows.append([r["name"], f"{r['latest_step']:,}",
                                 "Yes" if r["has_eval"] else "No", dt])
                return rows

            run_refresh_btn.click(refresh_runs, outputs=run_table)

        # ---- Timer for live updates ----
        timer = gr.Timer(value=3)

        def tick():
            # Pick the active source: standalone manager OR league main agent
            # (league main's metrics are shown on Dashboard when league is active)
            main_agent = league_mgr.agents.get("main")
            exp_agent = league_mgr.agents.get("exploiter")
            league_active = (main_agent and main_agent.status == "running")

            if league_active:
                st = main_agent.get_status()
                sj = main_agent.get_status_json()
                mh = main_agent.get_metrics_history()
            else:
                st = manager.get_status()
                sj = manager.get_status_json()
                mh = manager.get_metrics_history()

            # Status text
            status_parts = [f"Status: {st['status'].upper()}"]
            if league_active:
                status_parts[0] = "Status: LEAGUE RUNNING"
            if st["run_name"]:
                status_parts.append(f"Run: {st['run_name']}")
            if st["pid"]:
                status_parts.append(f"PID: {st['pid']}")
            if st["elapsed"]:
                status_parts.append(f"Elapsed: {st['elapsed']}")
            m = st["metrics"]
            if m:
                status_parts.append(f"Step: {m.get('step', 0):,}")
                status_parts.append(f"SPS: {m.get('sps', 0):,}")
            status_str = "\n".join(status_parts)

            # Dashboard markdown
            if st["status"] == "running" or league_active:
                label = "League" if league_active else "Running"
                dash_md = (f"**Status:** {label} | "
                           f"**Run:** {st['run_name'] or '?'} | "
                           f"**Step:** {m.get('step', 0):,} | "
                           f"**Elapsed:** {st['elapsed']}")
            else:
                dash_md = f"**Status:** {st['status'].capitalize()}"

            # Metric cards
            rew = f"{m['reward']:.3f}" if m.get("reward") is not None else "--"
            pg = f"{m['pg_loss']:.4f}" if m.get("pg_loss") is not None else "--"
            vf = f"{m['v_loss']:.4f}" if m.get("v_loss") is not None else "--"
            ent = f"{m['entropy']:.4f}" if m.get("entropy") is not None else "--"
            sps_str = f"{m['sps']:,}" if m.get("sps") is not None else "--"

            # Charts
            rl_chart = build_reward_loss_chart(mh)
            es_chart = build_entropy_sps_chart(mh)
            wr_chart = build_winrate_heatmap(sj)
            pc_chart = build_pc_winrate_chart(sj)

            # Log — merge league agent logs chronologically
            if league_active:
                merged = []
                if main_agent:
                    for ts, ln in list(main_agent.timestamped_lines)[-300:]:
                        merged.append((ts, f"[main] {ln}"))
                if exp_agent and exp_agent.status == "running":
                    for ts, ln in list(exp_agent.timestamped_lines)[-300:]:
                        merged.append((ts, f"[exp] {ln}"))
                merged.sort(key=lambda x: x[0])
                log = "\n".join(ln for _, ln in merged[-500:])
            else:
                log = "\n".join(st["log_tail"][-500:]) if st["log_tail"] else ""

            # Resource
            res = get_resource_info()

            # League status — show metrics when available
            main_st = league_mgr.get_agent_status("main")
            exp_st = league_mgr.get_agent_status("exploiter")

            def _agent_status_str(agent_st, agent_mgr):
                parts = [f"Status: {agent_st['status'].upper()}"]
                if agent_st.get("pid"):
                    parts.append(f"PID: {agent_st['pid']}")
                if agent_st.get("elapsed"):
                    parts.append(f"Elapsed: {agent_st['elapsed']}")
                am = agent_st.get("metrics", {})
                if am:
                    parts.append(f"Step: {am.get('step', 0):,}  |  "
                                 f"SPS: {am.get('sps', 0):,}  |  "
                                 f"Reward: {am.get('reward', 0):.3f}")
                    parts.append(f"Opponents: {am.get('opponents', '?')}")
                return "\n".join(parts)

            main_str = _agent_status_str(
                main_st, league_mgr.agents.get("main"))
            exp_str = _agent_status_str(
                exp_st, league_mgr.agents.get("exploiter"))

            return (
                status_str,          # status_text
                dash_md,             # dash_status
                rew, pg, vf,         # metric cards
                ent, sps_str,
                rl_chart,            # reward_loss_plot
                es_chart,            # entropy_sps_plot
                wr_chart,            # winrate_heatmap
                pc_chart,            # pc_winrate_plot
                log,                 # log_text
                res,                 # resource_text
                main_str,            # main_status
                exp_str,             # exp_status
            )

        timer.tick(
            tick,
            outputs=[
                status_text,
                dash_status,
                dash_reward, dash_pg, dash_vf, dash_ent, dash_sps,
                reward_loss_plot, entropy_sps_plot,
                winrate_heatmap, pc_winrate_plot,
                log_text, resource_text,
                main_status, exp_status,
            ],
        )

    return app


# ============================================================
#  Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Oh Hell Bot Training Dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    app = build_ui()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
