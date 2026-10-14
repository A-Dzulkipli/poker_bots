# rubick_counter

A heads-up bot built to beat `rubick/rubick` and `rubick/rubick_strict`. Registry name: `counter/rubick_counter`.

## Watch it play in the browser

    uv run --group ui python -m ui

On http://127.0.0.1:8000: set **Seat 0 played by** to `counter/rubick_counter`, set **Opponent** to
`rubick/rubick_strict` (or `rubick/rubick`), press **Sit down**. Auto-deal switches on, so it plays by
itself; the toolbar shows its running chips and bb/100. Lower **Bot delay** to speed it up.

## Results (heads-up, 64 tables x 250 hands, seed 9, bb/100 for the row bot, table-clustered SE)

| bot | vs rubick_strict | vs rubick |
|---|---|---|
| **rubick_counter** | **+182.4 ± 8.0** | +46.4 ± 11.2 |
| kraken_gpt (its base) | +52.7 ± 6.9 | +50.0 ± 7.7 |

Against rubick_strict it wins about 3.5x what the best existing bot does. Against rubick (argmax) it is
level with kraken_gpt: the pressure plays mostly switch themselves off there.

## How it works

`rubick_strict` plays windrunner_glm on ~90% of hands (rubick: ~30%). windrunner multiplies its equity
by 0.58 per unit of our betting pressure regardless of bet size, then calls only if that beats the pot
odds. Three plays exploit that:

1. **Big-blind 3-bet to 20bb** over its open: it folds even aces.
2. **4-bet all-in** over its 3-bet: it folds everything, and an all-in removes its re-raise option.
3. **Postflop pressure bets** (0.8 / 0.45 / 0.35 pot by street).

Each play is gated. The bot tracks, per spot, how often the opponent folds, re-raises or calls after
our raise (pseudo-count priors, state persists across hands), and presses only when the estimated EV
beats checking/calling by 10% of the pot. Everything else is kraken_gpt unchanged.

## Development notes

- v1 (always jam): +102 vs rubick_strict, -535 vs rubick. Ungated aggression is a disaster vs the mix.
- Postflop pressure with an optimistic prior lost ~80 bb/100 to invoker_gpt (it check-raises); a 0.5
  postflop prior and the EV margin fixed it.
- Modelling equity-when-called preflop as a mild drop (preflop equities are compressed) instead of
  "top of a random range" lifted rubick_strict from +80 to +190.
- Tracking re-raises (not just folds) stopped big 3-bets bleeding to 4-bets.
- The 4-bet jam uses a bigger equity drop against opponents who don't fold to our 3-bets (rubick's mix),
  and the small one against windrunner-like folders: kept rubick_strict at +182 while rubick went +18 -> +46.
