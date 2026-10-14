"""Central registry of every bot kept in this repository.

Names are paths:
    "phase_1_remedy/alan_wake_claude"      final (deployed) version of a lineage
    "phase_1_remedy/alan_wake_claude/v2"   any individual version of that lineage
    "probes/aggressor"                     probe bots that measure one leak (bots/probes/)
    "rubick/rubick", "rubick/rubick_strict" meta-bots that switch between finals by hole cards (bots/rubick/)
    "builtin/random"                       bots that ship with the engine

Folders: phase_1_remedy (6-max, 20 lines), phase_2_suneater (heads-up, 50 lines),
phase_3_dota (heads-up, 100 lines), phase_4_sea_monsters (heads-up, 100 lines, opponent
tracking), probes, rubick. Any folder with an `__init__.py` is scanned.

    from bots.registry import get_bot, list_bots, FINAL
    bot = get_bot("phase_1_remedy/alan_wake_claude")()     # registry entries are classes
    list_bots(phase="phase_1_remedy", finals_only=True)

Finals are the frozen `<lineage>_final.py` files at the root of each phase folder. Versions are
the `vN.py` files inside each lineage folder. Discovery is by file layout, and nothing is
imported until `get_bot` is called.
"""
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILTIN = {"builtin/random": ("poker_bots.poker_game", "RandomBot")}


def _discover() -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    finals, versions = {}, {}
    for phase in sorted(p for p in ROOT.iterdir() if (p / "__init__.py").exists()):
        for final in sorted(phase.glob("*_final.py")):
            lineage = final.stem.removesuffix("_final")
            finals[f"{phase.name}/{lineage}"] = (f"bots.{phase.name}.{final.stem}", "Bot")
        versions_found = [p for p in phase.glob("*/v*.py") if re.fullmatch(r"v\d+", p.stem)]
        for version in sorted(versions_found, key=lambda p: (p.parent.name, int(p.stem[1:]))):
            versions[f"{phase.name}/{version.parent.name}/{version.stem}"] = (
                f"bots.{phase.name}.{version.parent.name}.{version.stem}", "Bot")
    return finals, versions


FINAL, VERSIONS = _discover()
REGISTRY = {**BUILTIN, **FINAL, **VERSIONS}


def get_bot(name: str) -> type:
    """Return the bot class registered under `name`."""
    module, attribute = REGISTRY[name]
    return getattr(importlib.import_module(module), attribute)


def list_bots(phase: str | None = None, finals_only: bool = False) -> list[str]:
    names = FINAL if finals_only else REGISTRY
    return [name for name in names if phase is None or name.startswith(f"{phase}/")]
