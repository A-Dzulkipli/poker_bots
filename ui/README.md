# Poker UI

Play against registry bots and replay hands. Spec: the project's `notes/ui_contract.md`.

    uv run --group ui python -m ui                  # opens http://127.0.0.1:8000
    uv run --group ui python -m ui --eager          # no jit, so print() inside a bot works
    uv run --group ui python -m ui --bot-delay 300 --port 8001 --no-browser

- **Play.** Heads-up by default; 2–9 seats, you are always seat 0. Opponents come from
  `bots.registry` (finals first; tick "list every version" for the rest). Bot actions are paced
  (delay adjustable in the page), "show bot cards" is off by default, and raise amounts are
  raise-to totals. Keys: `f` fold, `c` check/call, `r` bet/raise, space or `n` deals the next hand.
- **Replay.** Your saved sessions, or any tournament block under `experiments/*/results/`. Format-2
  blocks (phase 2 onwards) are read straight from their stored actions. Format-1 blocks (phase 1,
  outcomes only) are replayed with `experiments/tournament.regenerate_block`, which is refused if the
  engine has changed since the block ran. Any hand whose `net_chips` differs from the stored rows is
  flagged. Arrow keys step (←/→) and change hand (↑/↓).

## Files

The UI only reads the engine, the registry and `experiments/` (it imports `hand_records` and
`tournament` from there). It writes only `ui/sessions/<session>/` (gitignored): the three
format-2 tables (`seat_hands`, `hands`, `actions`) built by `hand_records.records_to_frames` with
`session` as the key column, plus `manifest.json`. The column reference is `experiments/README.md`.
The human seat's `bot` and `by` are `"human"`; bots use roster-style names (`alan_wake_claude`,
`random`, `alan_wake_claude/v2` for a version). Files are rewritten every 10 hands and when the
session ends (End session, or stopping the server).

Tests: `uv run pytest ui/tests` (no UI dependencies needed; the phase-1 replay test regenerates a whole block, so it takes a while).
