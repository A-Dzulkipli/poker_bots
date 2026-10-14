# Phase 1 (remedy) round robin

Every pair from the four phase-1 finals plus RandomBot plays 6-max with three seats each
(10 pairs). Run from the repo root:

    uv run python experiments/phase_1_remedy/run_tournament.py              # seeds 0 1: 2 x 64 tables x 250 hands = 32,000 per pair
    uv run python experiments/phase_1_remedy/run_tournament.py --seeds 2 3  # extend; existing blocks are skipped
    uv run python experiments/phase_1_remedy/run_tournament.py --summarise  # rebuild summaries only

Bots come from `bots.registry` (`phase_1_remedy/<lineage>` finals and `builtin/random`).

## Storage layout

```
results/
    manifest.json                      config, engine hashes, bot sources and metadata, per-block timings
    seat_hands/<bot_a>__<bot_b>/seed_<s>.parquet   raw data, one file per (pair, seed) block
    pairs.parquet                      per (bot, opponent): bb/100, SE, z, hands, tables
    standings.parquet                  per bot: mean bb/100 over opponents, mean vs finals only, clear wins and losses
```

- **Raw data is one row per seat per hand** (zstd parquet). Columns: `pair`, `seed`, `table`,
  `hand`, `seat`, `bot`, `button`, `starting_stack`, `net_chips` (ending minus starting stack,
  before rebuys and cash-outs), `rebuy`, `cash_out`, `showdown` (seat reached showdown).
  Read everything with `pl.scan_parquet("results/seat_hands/**/*.parquet")`.
- **Actions and cards are not stored.** A hand is reproducible from its block's seed, the
  engine hashes and the bot sources in the manifest. That keeps a 16,000-hand block around 0.5 MB.
  Each block also records `engine_sha256`, so blocks run on different engine code are detectable.
- **Blocks are immutable.** More hands for a pair means a new seed, never rewriting a block.
  The summaries are derived and can always be rebuilt with `--summarise`.
- **Pair names follow roster order** (finals alphabetically, then random), and one file holds both sides, e.g. `alan_wake_claude__random`.

## Statistics

`bb_per_100` is a bot's mean net per seat-hand, in big blinds, times 100. Within a pair it is
zero-sum: one side's number is the negation of the other's. The SE is clustered by table: the
standard deviation of per-table bb/100 divided by √tables. Tables are independent, but seats and
hands within a table are not, so a naive per-seat-hand SE (what `bench.py` prints) misstates it.
`z = bb_per_100 / se`; |z| > 2 counts as a clear result.
