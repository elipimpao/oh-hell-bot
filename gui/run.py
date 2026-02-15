"""Launch the Oh Hell GUI server."""

import sys
import os
import argparse

# Add project root to path so we can import game, bots, network, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Oh Hell GUI Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    from gui.server import app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
