"""Freeze each phase-6 lineage's deployed version into `bots/phase_6_trails/<lineage>_final.py`.

    python experiments/phase_6_trails/build_finals.py

The deployed version is read from the lineage's `__init__.py`. The shared equity.py helpers are
inlined in place of the version's `from .equity import ...` line, so each final stands alone.
Run verify_finals.py afterwards.
"""
import hashlib
import re
from pathlib import Path

PHASE = Path(__file__).resolve().parents[2] / "bots" / "phase_6_trails"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


for lineage in sorted(p for p in PHASE.iterdir() if (p / "__init__.py").exists() and (p / "equity.py").exists()):
    version = re.search(r"from \.(v\d+) import Bot as DeployedBot", (lineage / "__init__.py").read_text()).group(1)
    source, helpers = lineage / f"{version}.py", lineage / "equity.py"
    lines = source.read_text().splitlines(keepends=True)
    imports = [i for i, line in enumerate(lines) if line.startswith("from .equity import")]
    assert len(imports) == 1, f"{source}: expected one single-line `from .equity import`"
    assert not any(re.match(r"\s*from \.", line) for i, line in enumerate(lines) if i != imports[0]), source
    equity_body = re.sub(r'\A""".*?"""\n', "", helpers.read_text(), flags=re.S)
    out = (f"# Frozen copy of {lineage.name}/{version}.py, the version deployed in that lineage's README.\n"
           "# The shared equity.py helpers are inlined so this file stands alone.\n"
           f"# Source sha256 (first 16): {version}.py {sha(source)}, equity.py {sha(helpers)}\n"
           + "".join(lines[:imports[0]])
           + "\n# ---- shared helpers, verbatim from equity.py ----\n" + equity_body
           + f"\n\n# ---- bot, verbatim from {version}.py ----\n" + "".join(lines[imports[0] + 1:]))
    target = PHASE / f"{lineage.name}_final.py"
    target.write_text(out)
    print(f"{target.name}: {version}, {len(out.splitlines())} lines")
