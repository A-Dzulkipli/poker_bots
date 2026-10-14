"""Registry of bots for the plain-Python engine (poker_bots.pure_game).

Same layout and names as bots/registry.py, which holds the JAX bots:
    "<folder>/<lineage>"       a frozen pure_bots/<folder>/<lineage>_final.py
    "<folder>/<lineage>/vN"    pure_bots/<folder>/<lineage>/vN.py
    "builtin/random"           pure_game.RandomBot
Each file defines a class named Bot with act(obs) -> ActionData (see the README's bot contract).
Any folder with an `__init__.py` is scanned, and nothing is imported until `get_bot` is called.

    from pure_bots.registry import get_bot
    table = Table([get_bot("builtin/random")(seed=0), get_bot("<folder>/<lineage>")()])
"""
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILTIN = {"builtin/random": ("poker_bots.pure_game", "RandomBot")}


def _discover() -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    finals, versions = {}, {}
    for folder in sorted(p for p in ROOT.iterdir() if (p / "__init__.py").exists()):
        for final in sorted(folder.glob("*_final.py")):
            lineage = final.stem.removesuffix("_final")
            finals[f"{folder.name}/{lineage}"] = (f"pure_bots.{folder.name}.{final.stem}", "Bot")
        versions_found = [p for p in folder.glob("*/v*.py") if re.fullmatch(r"v\d+", p.stem)]
        for version in sorted(versions_found, key=lambda p: (p.parent.name, int(p.stem[1:]))):
            versions[f"{folder.name}/{version.parent.name}/{version.stem}"] = (
                f"pure_bots.{folder.name}.{version.parent.name}.{version.stem}", "Bot")
    return finals, versions


FINAL, VERSIONS = _discover()
REGISTRY = {**BUILTIN, **FINAL, **VERSIONS}


def get_bot(name: str) -> type:
    """Return the bot class registered under `name`."""
    module, attribute = REGISTRY[name]
    return getattr(importlib.import_module(module), attribute)


def list_bots(folder: str | None = None, finals_only: bool = False) -> list[str]:
    names = FINAL if finals_only else REGISTRY
    return [name for name in names if folder is None or name.startswith(f"{folder}/")]
