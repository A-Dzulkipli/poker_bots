# poker_bots

## Quickstart: play against the bots

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.13 for you if it's missing.

    git clone <this repo> && cd poker_bots
    uv run --group ui python -m ui

The first run installs the dependencies, then opens http://127.0.0.1:8000. Pick opponents and seats, then
press Sit down. More options and the replay viewer are in [ui/README.md](ui/README.md).
