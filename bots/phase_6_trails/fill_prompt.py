"""Fill a bot prompt from a TOML file and write it to the path the TOML names.

    python fill_prompt.py alan_wake_claude.toml

TOML layout (see example.toml):
    output    where to write the filled prompt, relative to the TOML file
    template  optional; defaults to botgen_v1_prompt.md next to this script
    [slots]   BOT_NAME, LINEAGE, BENCHMARK_SET, CONSTRAINTS, optional ROOT_DIR
              (defaults to the TOML file's directory, or its parent if that
              directory is named BOT_NAME), and METADATA_BLOCKS as
              [[slots.METADATA_BLOCKS]] tables, each rendered as one JSON object.
Exits without writing if any {{SLOT}} in the template is left unfilled.
"""
import json
import re
import sys
import tomllib
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
config = tomllib.loads(config_path.read_text())
template_path = config_path.parent / config.get("template", Path(__file__).resolve().parent / "botgen_v1_prompt.md")
default_root = config_path.parent
if default_root.name == config["slots"]["BOT_NAME"]:  # TOML lives inside the bot's own folder
    default_root = default_root.parent
slots = {"ROOT_DIR": str(default_root), **config["slots"]}
slots["METADATA_BLOCKS"] = ",\n            ".join(json.dumps(block) for block in slots.get("METADATA_BLOCKS", []))

template = re.sub(r"\A<!--.*?-->\n+", "", template_path.read_text(), flags=re.S)
missing = sorted(set(re.findall(r"\{\{([A-Z_]+)\}\}", template)) - set(slots))
if missing:
    sys.exit(f"unfilled slots: {missing}")
output_path = config_path.parent / config["output"]
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: str(slots[m.group(1)]), template))
print(f"wrote {output_path}")
