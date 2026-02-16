"""League orchestrator for Oh Hell multi-agent training.

Reads league.toml, launches main + exploiter agents as subprocesses
of train.py with appropriate flags, monitors health, handles Ctrl+C.

Usage:
    python league.py                     # uses league.toml
    python league.py --config my.toml    # custom config
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import threading

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

_B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _to_base36(n):
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(_B36[n % 36])
        n //= 36
    return "".join(reversed(digits))


def parse_league_config(config_path):
    """Parse league.toml and validate structure."""
    if tomllib is None:
        print("Error: tomllib/tomli required. Install tomli: pip install tomli")
        sys.exit(1)
    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    if "league" not in config:
        print("Error: [league] section required in config")
        sys.exit(1)
    if "agents" not in config:
        print("Error: [agents] section required in config")
        sys.exit(1)
    for name, agent in config["agents"].items():
        if "role" not in agent:
            print(f"Error: agent '{name}' missing 'role' field")
            sys.exit(1)
        if agent["role"] not in ("main", "exploiter"):
            print(f"Error: agent '{name}' has invalid role '{agent['role']}' "
                  f"(must be 'main' or 'exploiter')")
            sys.exit(1)
    return config


def build_agent_command(agent_name, agent_cfg, league_cfg):
    """Build the subprocess command for an agent.

    Maps league.toml agent config to train.py CLI arguments.
    """
    snapshot_base = league_cfg.get("snapshot_base", "league_snapshots")
    role = agent_cfg["role"]

    cmd = [sys.executable, "train.py"]

    # Base config file
    if "config" in agent_cfg:
        cmd += ["--config", agent_cfg["config"]]

    # Snapshot dir: each agent writes to its own subdirectory
    snap_dir = agent_cfg.get("snapshot_dir",
                              os.path.join(snapshot_base, agent_name))
    cmd += ["--snapshot-dir", snap_dir]

    # Checkpoint dir: each agent saves checkpoints to its own subdirectory
    checkpoint_base = league_cfg.get("checkpoint_base", "checkpoints")
    checkpoint_dir = agent_cfg.get("checkpoint_dir",
                                    os.path.join(checkpoint_base, agent_name))
    cmd += ["--save-dir", checkpoint_dir]

    # Load dirs: read other agents' snapshots
    load_dirs = agent_cfg.get("load_dirs", [])
    if load_dirs:
        cmd += ["--load-dirs", ",".join(load_dirs)]

    # Rescan interval
    if "rescan_interval" in agent_cfg:
        cmd += ["--rescan-interval", str(agent_cfg["rescan_interval"])]

    # Exploiter mode
    if role == "exploiter":
        cmd.append("--exploiter-mode")

    # Run name: short prefix + base36 timestamp
    role_prefix = "EXP" if role == "exploiter" else "MAIN"
    run_name = agent_cfg.get("run_name",
                              f"{role_prefix}_{_to_base36(int(time.time()))}")
    cmd += ["--run-name", run_name]

    # Apply overrides (maps to train.py CLI args)
    overrides = agent_cfg.get("overrides", {})
    for key, value in overrides.items():
        cli_key = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(cli_key)
        else:
            cmd += [cli_key, str(value)]

    return cmd, snap_dir


# --- Output prefixing ---
_print_lock = threading.Lock()


def _prefix_reader(stream, prefix):
    """Read lines from a subprocess stream and print with a prefix."""
    try:
        for raw_line in stream:
            line = raw_line.rstrip("\n\r")
            with _print_lock:
                print(f"[{prefix}] {line}", flush=True)
    except ValueError:
        pass  # stream closed
    finally:
        stream.close()


def launch_agent(agent_name, cmd, script_dir):
    """Launch a subprocess with stdout/stderr prefixed by agent name."""
    proc = subprocess.Popen(
        cmd, cwd=script_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    t = threading.Thread(target=_prefix_reader, args=(proc.stdout, agent_name),
                         daemon=True)
    t.start()
    return proc


def main():
    parser = argparse.ArgumentParser(
        description="Oh Hell League Orchestrator — launches main + exploiter agents")
    parser.add_argument("--config", type=str, default="league.toml",
                        help="Path to league TOML config")
    parser.add_argument("--max-restarts", type=int, default=3,
                        help="Max automatic restarts per agent on crash")
    parser.add_argument("--restart-cooldown", type=float, default=30.0,
                        help="Seconds to wait before restarting a crashed agent")
    cli_args = parser.parse_args()

    config = parse_league_config(cli_args.config)
    league_cfg = config["league"]
    agents_cfg = config["agents"]

    # Create snapshot directories
    snapshot_base = league_cfg.get("snapshot_base", "league_snapshots")
    os.makedirs(snapshot_base, exist_ok=True)
    for agent_name, agent_cfg in agents_cfg.items():
        snap_dir = agent_cfg.get("snapshot_dir",
                                  os.path.join(snapshot_base, agent_name))
        os.makedirs(snap_dir, exist_ok=True)
        for ld in agent_cfg.get("load_dirs", []):
            os.makedirs(ld, exist_ok=True)

    # Build commands and launch
    processes = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for agent_name, agent_cfg in agents_cfg.items():
        cmd, snap_dir = build_agent_command(agent_name, agent_cfg, league_cfg)
        print(f"[league] Launching {agent_name} ({agent_cfg['role']})")
        print(f"         snapshot_dir: {snap_dir}")
        print(f"         cmd: {' '.join(cmd)}")
        proc = launch_agent(agent_name, cmd, script_dir)
        processes[agent_name] = {
            "proc": proc,
            "cmd": cmd,
            "restarts": 0,
            "role": agent_cfg["role"],
        }
        # Stagger launches to avoid GPU contention at startup
        time.sleep(2)

    print(f"\n[league] All {len(processes)} agents launched. "
          f"Press Ctrl+C to stop.\n")

    # Monitor loop
    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not shutdown:
            time.sleep(5)

            for agent_name, info in list(processes.items()):
                proc = info["proc"]
                ret = proc.poll()

                if ret is not None and not shutdown:
                    if ret == 0:
                        print(f"[league] {agent_name} exited normally (code 0)")
                        continue

                    print(f"[league] {agent_name} crashed (exit code {ret})")

                    if info["restarts"] < cli_args.max_restarts:
                        info["restarts"] += 1
                        print(f"[league] Restarting {agent_name} in "
                              f"{cli_args.restart_cooldown}s "
                              f"(restart {info['restarts']}/{cli_args.max_restarts})")
                        time.sleep(cli_args.restart_cooldown)
                        if shutdown:
                            break
                        new_proc = launch_agent(agent_name, info["cmd"],
                                                script_dir)
                        info["proc"] = new_proc
                        print(f"[league] {agent_name} restarted "
                              f"(pid {new_proc.pid})")
                    else:
                        print(f"[league] {agent_name} exceeded max restarts, "
                              f"giving up")

            # Check if all agents have finished
            all_done = all(info["proc"].poll() is not None
                           for info in processes.values())
            if all_done:
                print("[league] All agents have exited.")
                break

    finally:
        # Graceful shutdown: terminate all children
        print("\n[league] Shutting down all agents...")
        for agent_name, info in processes.items():
            proc = info["proc"]
            if proc.poll() is None:
                print(f"[league] Terminating {agent_name} (pid {proc.pid})")
                try:
                    proc.terminate()
                except OSError:
                    pass

        # Wait for graceful exit
        for agent_name, info in processes.items():
            proc = info["proc"]
            try:
                proc.wait(timeout=15)
                print(f"[league] {agent_name} exited "
                      f"(code {proc.returncode})")
            except subprocess.TimeoutExpired:
                print(f"[league] {agent_name} did not exit, killing")
                proc.kill()


if __name__ == "__main__":
    main()
