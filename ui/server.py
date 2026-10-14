"""Local FastAPI server for the play and replay page. One play session at a time; pacing is done by the page."""
import importlib
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bots import registry
from ui import replay
from ui.game import PlaySession

STATIC = Path(__file__).resolve().parent / "static"
settings = {"jit": True, "bot_delay_ms": 600}
current: dict[str, PlaySession | None] = {"session": None}
lock = threading.Lock()   # handlers run in a thread pool; the session is not safe to step concurrently


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if current["session"] is not None:
        current["session"].close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"bot_delay_ms": settings["bot_delay_ms"], "jit": settings["jit"]}


@app.get("/api/bots")
def bots():
    """Finals first, then the builtins, then every individual version. The registry is rediscovered on
    every call, so bots added while the server runs show up when the page is reloaded."""
    with lock:   # a table being built must not see the registry mid-reload
        importlib.invalidate_caches()   # let the import system see bot files created since startup
        importlib.reload(registry)
    finals = list(registry.FINAL)
    others = [name for name in registry.list_bots() if name not in registry.FINAL]
    builtins = [name for name in others if name.startswith("builtin/")]
    return {"finals": finals + builtins, "versions": [name for name in others if name not in builtins]}


class NewSession(BaseModel):
    seat_bots: list[str]
    small_blind: int = 1
    big_blind: int = 2
    starting_bb: int = 100
    seed: int | None = None


class HumanAction(BaseModel):
    type: str
    value: int = 0


def session() -> PlaySession:
    if current["session"] is None:
        raise HTTPException(404, "no session is running")
    return current["session"]


@app.post("/api/play/new")
def new_session(body: NewSession, show_bot_cards: bool = False):
    if not 1 <= len(body.seat_bots) <= 8:
        raise HTTPException(400, "a table has 2 to 9 seats")
    with lock:
        if current["session"] is not None:
            current["session"].close()
            current["session"] = None
        try:
            current["session"] = PlaySession(body.seat_bots, body.small_blind, body.big_blind, body.starting_bb,
                                             body.seed, jit=settings["jit"])
        except Exception as error:   # a half-written bot in the registry should not take the server down
            raise HTTPException(400, f"could not start the table: {type(error).__name__}: {error}")
        return {"event": "new", "state": current["session"].snapshot(show_bot_cards)}


@app.get("/api/play/state")
def state(show_bot_cards: bool = False):
    with lock:
        return {"event": "state", "state": session().snapshot(show_bot_cards)}


@app.post("/api/play/advance")
def advance(show_bot_cards: bool = False):
    with lock:
        play = session()
        event = play.advance()
        return {"event": event, "state": play.snapshot(show_bot_cards)}


@app.post("/api/play/act")
def act(body: HumanAction, show_bot_cards: bool = False):
    with lock:
        play = session()
        if body.type not in ("fold", "check_call", "raise"):
            raise HTTPException(400, f"unknown action {body.type}")
        try:
            play.act(body.type, body.value)
        except ValueError as error:
            raise HTTPException(409, str(error))
        return {"event": "acted", "state": play.snapshot(show_bot_cards)}


@app.post("/api/play/end")
def end():
    with lock:
        play = session()
        play.close()
        current["session"] = None
        return {"session": play.session_id, "hands": play.writer.hands}


@app.get("/api/replay/sessions")
def sessions():
    return replay.list_sessions()


@app.get("/api/replay/session/{session_id}")
def session_hands(session_id: str):
    if "/" in session_id or session_id.startswith("."):
        raise HTTPException(400, "bad session id")
    if not (replay.SESSIONS / session_id / "manifest.json").exists():
        raise HTTPException(404, "no such session")
    return replay.load_session(session_id)


@app.get("/api/replay/tournaments")
def tournaments():
    return replay.list_tournaments()


@app.get("/api/replay/tournament/{experiment}/{pair}/{seed}/{table}")
def tournament_table(experiment: str, pair: str, seed: int, table: int):
    blocks = {(b["experiment"], b["pair"], b["seed"]): b for b in replay.list_tournaments()}
    block = blocks.get((experiment, pair, seed))
    if block is None:
        raise HTTPException(404, "no such block")
    if not 0 <= table < block["tables"]:
        raise HTTPException(400, f"table must be 0 to {block['tables'] - 1}")
    try:
        return replay.tournament_table(experiment, pair, seed, table)
    except ValueError as error:
        raise HTTPException(409, str(error))
