"""FastAPI server for the play and replay page. One play table per browser (the page sends its id in an
X-Client header); pacing is done by the page."""
import importlib
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bots import registry
from ui import replay
from ui.game import PlaySession

STATIC = Path(__file__).resolve().parent / "static"
settings = {"jit": True, "bot_delay_ms": 600, "max_tables": 8, "idle_minutes": 30, "sessions_dir": None}


@dataclass
class Table:
    play: PlaySession
    lock: threading.Lock = field(default_factory=threading.Lock)   # a table is not safe to step concurrently
    last_used: float = field(default_factory=time.monotonic)


tables: dict[str, Table] = {}
tables_lock = threading.Lock()   # guards `tables` and registry reloads; handlers run in a thread pool


def close_idle() -> None:
    """Close tables nobody has touched for `idle_minutes`, skipping any that are mid-step."""
    cutoff = time.monotonic() - settings["idle_minutes"] * 60
    with tables_lock:
        for client, table in list(tables.items()):
            if table.last_used < cutoff and table.lock.acquire(blocking=False):
                try:
                    table.play.close()
                    del tables[client]
                finally:
                    table.lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for table in tables.values():
        table.play.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def client(x_client: str = Header("local", max_length=64)) -> str:
    return x_client


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
    with tables_lock:   # a table being built must not see the registry mid-reload
        importlib.invalidate_caches()   # let the import system see bot files created since startup
        importlib.reload(registry)
    finals = list(registry.FINAL)
    others = [name for name in registry.list_bots() if name not in registry.FINAL]
    builtins = [name for name in others if name.startswith("builtin/")]
    return {"finals": finals + builtins, "versions": [name for name in others if name not in builtins]}


class NewSession(BaseModel):
    player_name: str = Field(max_length=40)
    seat_bots: list[str]
    small_blind: int = 1
    big_blind: int = 2
    starting_bb: int = 100
    seed: int | None = None


class HumanAction(BaseModel):
    type: str
    value: int = 0


def table(client: str = Depends(client)) -> Table:
    found = tables.get(client)
    if found is None:
        raise HTTPException(404, "no table is running (an idle table is closed after a while): press End session")
    found.last_used = time.monotonic()
    return found


@app.post("/api/play/new")
def new_session(body: NewSession, show_bot_cards: bool = False, client: str = Depends(client)):
    if not 1 <= len(body.seat_bots) <= 8:
        raise HTTPException(400, "a table has 2 to 9 seats")
    name = body.player_name.strip()
    if not name:
        raise HTTPException(400, "enter a name to sit down")
    close_idle()
    with tables_lock:
        old = tables.pop(client, None)
        if old is not None:
            with old.lock:
                old.play.close()
        if len(tables) >= settings["max_tables"]:
            raise HTTPException(503, f"all {settings['max_tables']} tables are in use, try again in a few minutes")
        try:
            new = Table(PlaySession(body.seat_bots, body.small_blind, body.big_blind, body.starting_bb,
                                    body.seed, sessions_dir=settings["sessions_dir"], jit=settings["jit"], player_name=name))
        except Exception as error:   # a half-written bot in the registry should not take the server down
            raise HTTPException(400, f"could not start the table: {type(error).__name__}: {error}")
        tables[client] = new
    with new.lock:   # the first snapshot compiles the bots; other tables keep playing meanwhile
        return {"event": "new", "state": new.play.snapshot(show_bot_cards)}


@app.get("/api/play/state")
def state(show_bot_cards: bool = False, table: Table = Depends(table)):
    with table.lock:
        return {"event": "state", "state": table.play.snapshot(show_bot_cards)}


@app.post("/api/play/advance")
def advance(show_bot_cards: bool = False, table: Table = Depends(table)):
    with table.lock:
        event = table.play.advance()
        return {"event": event, "state": table.play.snapshot(show_bot_cards)}


@app.post("/api/play/act")
def act(body: HumanAction, show_bot_cards: bool = False, table: Table = Depends(table)):
    if body.type not in ("fold", "check_call", "raise"):
        raise HTTPException(400, f"unknown action {body.type}")
    with table.lock:
        try:
            table.play.act(body.type, body.value)
        except ValueError as error:
            raise HTTPException(409, str(error))
        return {"event": "acted", "state": table.play.snapshot(show_bot_cards)}


@app.post("/api/play/end")
def end(client: str = Depends(client)):
    with tables_lock:
        found = tables.pop(client, None)
    if found is None:
        raise HTTPException(404, "no session is running")
    with found.lock:
        found.play.close()
    return {"session": found.play.session_id, "hands": found.play.writer.hands}


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
    if not block["replayable"]:
        raise HTTPException(409, "this block's hands are not available here")
    if not 0 <= table < block["tables"]:
        raise HTTPException(400, f"table must be 0 to {block['tables'] - 1}")
    try:
        return replay.tournament_table(experiment, pair, seed, table)
    except ValueError as error:
        raise HTTPException(409, str(error))
