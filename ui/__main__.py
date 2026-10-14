"""Run the poker UI:  uv run --group ui python -m ui  [--port 8000] [--bot-delay 600] [--eager]"""
import argparse
import webbrowser

import jax
import uvicorn

from ui import server


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bot-delay", type=int, default=600, help="milliseconds between bot actions (adjustable in the page)")
    parser.add_argument("--eager", action="store_true", help="run without jit so print() inside bots works")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server.settings.update(jit=not args.eager, bot_delay_ms=args.bot_delay)
    if args.eager:
        jax.config.update("jax_disable_jit", True)
    url = f"http://{args.host}:{args.port}"
    print(f"Poker UI at {url}  (Ctrl+C to stop; play sessions are saved under ui/sessions/)")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
