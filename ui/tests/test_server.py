import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from ui import server  # noqa: E402


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setitem(server.settings, "sessions_dir", tmp_path)
    monkeypatch.setitem(server.settings, "max_tables", 2)
    yield TestClient(server.app)
    for table in server.tables.values():
        table.play.close()
    server.tables.clear()


def call(api, client, method, path, json=None):
    return api.request(method, path, json=json, headers={"X-Client": client})


def test_each_browser_gets_its_own_table(api, monkeypatch):
    new = {"player_name": "Ann", "seat_bots": ["builtin/random"]}
    a = call(api, "a", "POST", "/api/play/new", {**new, "seed": 1}).json()["state"]
    b = call(api, "b", "POST", "/api/play/new", {**new, "seed": 2}).json()["state"]
    assert a["session"] != b["session"]
    assert call(api, "c", "POST", "/api/play/new", {**new, "player_name": "  "}).status_code == 400
    assert call(api, "c", "POST", "/api/play/new", new).status_code == 503   # max_tables is 2
    assert call(api, "c", "GET", "/api/play/state").status_code == 404

    for _ in range(20):   # stepping a does not move b
        call(api, "a", "POST", "/api/play/advance")
    assert call(api, "b", "GET", "/api/play/state").json()["state"]["actions"] == b["actions"]
    assert call(api, "a", "POST", "/api/play/new", new).status_code == 200   # replacing your own table is not capped
    assert server.tables["a"].play.writer.manifest["player_name"] == "Ann"

    monkeypatch.setitem(server.settings, "idle_minutes", 0)   # every table is now idle
    server.close_idle()
    assert server.tables == {}
    assert call(api, "b", "POST", "/api/play/end").status_code == 404
