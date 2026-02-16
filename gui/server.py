"""FastAPI server for the Oh Hell GUI."""

import os
import json
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from gui.session import GameSession
from gui.advisor import AdvisorSession
from gui.bot_manager import list_snapshots

# Project root (parent of gui/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Oh Hell GUI")

# CORS: allow the Chrome extension (running on trickstercards.com) to reach this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory session store
sessions: dict = {}


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/snapshots")
async def get_snapshots():
    return list_snapshots(PROJECT_ROOT)


class SessionCreateRequest(BaseModel):
    mode: str  # "play" or "advisor"
    num_players: int
    seats: Optional[list] = None
    human_seat: int = 0
    dev_mode: bool = False
    auto_play_speed: float = 1.0
    custom_hand_size: Optional[int] = None
    custom_hands: Optional[dict] = None
    custom_trump_card: Optional[int] = None
    advisor_snapshot: Optional[str] = None


@app.post("/api/sessions")
async def create_session(req: SessionCreateRequest):
    config = req.model_dump()

    # Resolve snapshot paths to absolute paths
    if config.get("seats"):
        for seat in config["seats"]:
            if seat.get("snapshot_path"):
                seat["snapshot_path"] = os.path.join(PROJECT_ROOT, seat["snapshot_path"])
    if config.get("advisor_snapshot"):
        config["advisor_snapshot"] = os.path.join(PROJECT_ROOT, config["advisor_snapshot"])

    try:
        if req.mode == "play":
            session = GameSession(config)
        elif req.mode == "advisor":
            session = AdvisorSession(config)
        else:
            return {"error": f"Unknown mode: {req.mode}"}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    sessions[session.id] = session
    return {"session_id": session.id, "mode": req.mode}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted"}
    return {"error": "Session not found"}


@app.websocket("/ws/{session_id}")
async def game_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    try:
        # Send initial state
        if isinstance(session, GameSession):
            # Auto-advance bots if they go first
            initial_events = session._auto_advance_bots()
            await websocket.send_json({
                "type": "state_update",
                "events": initial_events,
                "state": session.get_full_state()
            })
        else:
            await websocket.send_json({
                "type": "advisor_state",
                "state": session._get_state_for_frontend()
            })

        # Message loop
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            try:
                response = session.handle_message(msg)
                await websocket.send_json(response)
            except Exception as e:
                traceback.print_exc()
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
