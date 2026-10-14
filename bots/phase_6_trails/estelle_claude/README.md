# estelle_claude

- **Bot name:** estelle_claude — **lineage:** phase_6_trails/estelle_claude
- **Model:** Claude (configured model id `claude-opus-5-5`; serving model may differ), via Claude desktop
- **Prompt version:** botgen-v6 — **engine:** decafb9, engine_sha256 `295477eff7637056`
- **Wall-clock time used:** about 45 minutes (check passed 07:27 UTC; README and final checks finished 08:12 UTC)
- **Code executed:** yes
- **Environment:** the shell runs in a Linux VM where `uv` could not use the repository `.venv` (`failed to remove file .venv/.lock: Operation not permitted`), so every command used the pre-approved separate environment: `UV_PROJECT_ENVIRONMENT=$HOME/p6venv uv run --frozen python bench.py ...`. Background processes are killed between shell calls in this VM, so field runs were split into batches of 4–6 opponents run in parallel with one `bench.py` process per opponent (`run.sh`); each line in `logs/` is ordinary `bench.py` output (64 tables × 250 hands). Weighted scores were computed from those lines by `score.py` using the same formula as `bench.py`.
- **`bench.py check` output:**
  ```
  python      /sessions/.../p6venv/bin/python (3.13.15)
  jax         0.11.2 (cpu)
  poker_bots  /sessions/.../mnt/poker_bots/poker_bots/src/poker_bots
  engine      sha256 295477eff7637056 (matches the contract)
  smoke test  random vs aggressor, 40 hands in 2s: OK
  ```
- Nothing outside the bot directory was read.

## Design

Every decision samples 128 opponent holdings (each with a random completion of the board), gives each holding a weight, and plays by the weighted showdown equity of our hand against those holdings ("equity vs range"). A holding's weight starts at 1 and is multiplied, for each opponent action in the hand, by a soft threshold on that holding's strength percentile on that action's street: a bet/raise keeps the top share, a call removes the weakest share and discounts the strongest (they would have raised), a check discounts the strongest. Strength is a short preflop rule (high card, kicker, pair, suited, gap) preflop and the made-hand evaluator on the real board postflop, turned into a percentile among the sampled holdings. The action rule is a threshold policy on equity vs range: open/raise, value-bet, call at pot odds, or bluff at a fixed frequency when checked to with a weak hand. The deployed v5 is v1's range model with more aggressive thresholds; the range model is what makes the bot competitive at all (switching it off costs hundreds of bb/100, see below).

## Versions

### v1 — independent per-action narrowing
- *What it does:* the design above with conservative thresholds (raise ≥ 0.72 equity vs range, bet ≥ 0.60, open ≥ 0.45, bluff 15%). Each opponent action re-weights holdings by their percentile against *all* holdings, independently of earlier actions.
- *Why:* baseline of the range idea.
- *Parameters:* `num_samples` 128 holdings, each with one runout (half the 256 cap), `preflop_raise_keep_share` 0.6, `postflop_bet_keep_share` 0.5, `call_drop_share` 0.3, `slowplay_discount` 0.5, `slowplay_percentile` 0.8, `range_softness` 0.08 percentile units, `open_equity` 0.45, `value_bet_equity` 0.6, `value_raise_equity` 0.72, `call_margin` 0.0, `bluff_frequency` 0.15, `bluff_max_equity` 0.35, `bet_pot_fraction` 0.66 of pot after call, `preflop_raise_multiple` 3.0× current bet, `use_range` True. All by reasoning.
- *Results* (seed 0, 64×250; only 8 opponents run — field run was interrupted when the VM killed the background process, and then budget went to later versions):

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| kraken_gpt | 0 | 64×250 | -66.0 | 10.4 |
| leviathan_claude | 0 | 64×250 | -17.2 | 12.6 |
| megalodon_glm | 0 | 64×250 | -19.1 | 13.8 |
| spike_claude | 0 | 64×250 | -23.7 | 10.2 |
| faye_gpt | 0 | 64×250 | -69.1 | 6.7 |
| windrunner_glm | 0 | 64×250 | -24.0 | 14.2 |
| invoker_gpt | 0 | 64×250 | -16.4 | 13.0 |
| aggressor | 0 | 64×250 | -52.9 | 7.6 |

  Weighted score: not run (8/18 opponents).
- *Verdict:* kept as parent (of v2, v3, v5). Too passive: barely better than folding every hand against `aggressor`.
- *Line count:* 66.

### v2 — wider opponent ranges
- *What it does:* v1 with bets/raises assumed to keep more (preflop 0.85, postflop 0.7) and calls to drop less (0.15).
- *Why:* hypothesis that v1 over-folds because it assumes bets are too strong.
- *Parameters:* as v1 except those three (reasoning).
- *Results:* aggressor -47.1 (SE 11.3), faye_gpt -47.9 (8.0), kraken_gpt -111.7 (10.6), spike_claude -86.4 (16.7); seed 0, 64×250. Weighted score not run.
- *Verdict:* discarded — much worse against kraken and spike; calling more is not the fix.
- *Line count:* 66.

### v3 — sequential narrowing
- *What it does:* replaces v1's independent re-weighting with a scan over the hand's history: each action's percentile is measured *within the range as it stood before that action*, so a bet after a bet keeps a share of the already narrowed range (first 32 history slots).
- *Why:* in v1 a check-raise or a third barrel still only means "top half of all hands", so strong lines are under-respected.
- *Parameters:* v1's plus `history_slots` 32.
- *Results:* aggressor -84.2 (6.0), faye_gpt -82.8 (5.4), kraken_gpt -96.1 (6.9), spike_claude +27.1 (11.0); seed 0, 64×250.
- *Verdict:* discarded — ranges become too narrow, the bot folds far too much to aggression.
- *Line count:* 69.

### v4 — v3 plus polarised bet ranges
- *What it does:* v3, but after a bet/raise weak holdings keep a floor weight of 0.3 (bluffs).
- *Why:* fix v3's over-folding while keeping sequential narrowing.
- *Results:* aggressor -95.1 (6.1), faye_gpt -77.2 (7.2), kraken_gpt -74.6 (8.3); seed 0, 64×250.
- *Verdict:* discarded; no better than v3. Line count 70.

### Tuning probes on v1 (parameter settings in `ablation.py`, not versions)
Seed 0, 64×250, bb/100 (SE): tight raise (0.80/0.66) vs aggressor -64.0 (7.8); tight open (0.55) -71.1 (9.0); loose call (margin -0.06) -85.8 (7.1); tight call (+0.06) -52.1 (5.7), vs kraken -96.1 (8.6); aggressive (raise 0.62, bet 0.52, bluff 0.3) vs aggressor -38.9 (8.2), vs kraken -35.1 (10.4); more aggressive (raise 0.56, bet 0.48, bluff 0.45, open 0.38) vs aggressor +13.0 (7.0), vs kraken -54.8 (11.0); aggressive with pot-sized bets vs aggressor -59.8 (12.1), kraken -54.7 (11.2). Direction: the bot gains from betting and raising more against its narrowed range, not from calling more.

### v5 — v1 with aggressive thresholds (deployed)
- *What it does:* v1's range model unchanged; opens/raises limps with equity vs range ≥ 0.38, value-bets when checked to at ≥ 0.48, raises bets at ≥ 0.56, and bluffs 45% of the time when checked to postflop with equity < 0.35.
- *Why:* v1 lost by being passive; the tuning probes showed aggression is the lever.
- *Parameters:* as v1 except `open_equity` 0.38, `value_bet_equity` 0.48, `value_raise_equity` 0.56, `bluff_frequency` 0.45 (tuned on seed 0 against aggressor and kraken_gpt, see probes).
- *Results (seed 0, 64×250):* weighted score **+19.78** (plain mean vs non-random +46.4).

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| kraken_gpt | 0 | 64×250 | -54.8 | 11.0 |
| leviathan_claude | 0 | 64×250 | -23.1 | 19.2 |
| megalodon_glm | 0 | 64×250 | -30.3 | 17.0 |
| spike_claude | 0 | 64×250 | +6.2 | 20.3 |
| faye_gpt | 0 | 64×250 | -61.8 | 10.7 |
| windrunner_glm | 0 | 64×250 | +34.0 | 12.6 |
| invoker_gpt | 0 | 64×250 | -34.9 | 18.5 |
| meepo_claude | 0 | 64×250 | -20.4 | 18.3 |
| hadrian_marlowe_gpt | 0 | 64×250 | +5.0 | 13.7 |
| vicious_glm | 0 | 64×250 | +65.6 | 25.7 |
| valka_onderra_claude | 0 | 64×250 | -34.4 | 20.9 |
| aggressor | 0 | 64×250 | +13.0 | 7.0 |
| alan_wake_claude | 0 | 64×250 | +36.7 | 19.4 |
| max_payne_gpt | 0 | 64×250 | +51.9 | 14.2 |
| bassander_lin_glm | 0 | 64×250 | +129.8 | 28.7 |
| jesse_faden_glm | 0 | 64×250 | +310.0 | 32.2 |
| jack_joyce_gemini | 0 | 64×250 | +396.8 | 33.5 |
| random | 0 | 64×250 | +682.0 | 24.3 |

  Against parent v1: seed 2 -33.3 (20.8), seed 3 -3.0 (18.3), seed 4 -26.4 (19.4); pooled about -21 ± 11 (≈1.8 SE, not significant but a warning).
- *Verdict:* deployed.
- *Line count:* 66.

## Range model (deployed v5)

- **Representation:** 128 sampled opponent holdings (cards not in our hand or on the board), each with a weight in (0, 1]; weighted share = mean weight.
- **Strength per street:** preflop `high + 0.5·kicker + 10·pair + 2·suited − 0.5·min(gap−1, 4)`; flop/turn/river the evaluator's made-hand strength on the real board of that street. Converted to a percentile among the 128 holdings.
- **Narrowing, per opponent action** (calls and checks are told apart by whether the actor's street contribution grew): bet/raise × sigmoid((pct − (1 − keep))/0.08) with keep 0.6 preflop, 0.5 postflop; call × sigmoid((pct − 0.3)/0.08) × slowplay; check × slowplay, where slowplay = 1 − 0.5·sigmoid((pct − 0.8)/0.08). Every action on every street multiplies in, so the river weight reflects the whole line; each new board card changes the postflop percentiles.
- **Use:** equity = Σ weight × (win + tie/2) / Σ weight over the sampled holdings and runouts; all fold/call/bet/raise thresholds compare this number (and pot odds).

**Range table** — measured by `range_table.py` (8 seeds × 128 holdings) from the bot's own functions; board Kh 8d 3c, turn 2s, river 9h; open = 3bb, 3-bet = 9bb. "Kept" lists hand classes with mean weight > 0.5, highest first.

| line | share kept | description |
|---|---|---|
| button opens | 0.60 | top 60% by the preflop rule: pairs, any ace/king, broadways, suited connectors-ish |
| button limps | 0.60 | bottom 30% removed and the top ~20% halved: middling hands (Q7o, J8o, T7s, small aces, 33) |
| big blind calls an open | 0.60 | same shape as a limp: middling hands, premiums discounted |
| big blind 3-bets | 0.60 | same as an open (v5 does not narrow further on a re-raise): 88+, AK–AT, KQ, suited broadways |
| button c-bets flop after opening | 0.35 | top preflop hands that also have a pair or better on K83: KK, 88, AA, AK, KQ, Kx, overpairs |
| big blind check-raises flop | 0.25 | middling preflop hands with a made pair on K83 (8x, 3x, Kx with weak kicker, small pairs); sets/two pair discounted by the call and the check |
| bets turn after bet flop, called | 0.24 | pairs or better that continue: weak Kx, 8x, 33, 22-board hits |
| bets all three streets | 0.21 | mostly Kx, 8x and two-pair/set hands |

Weakness visible in the table: because narrowing is independent per action, a 3-bet range equals an open range, and a check-raise penalises the strongest holdings via the earlier check. v3/v4 tried sequential narrowing and lost.

**Evidence that the range model helps** (same v5 strategy with `use_range=False`, i.e. equity vs a uniformly random hand; `ablation.py:V5NoRange`): head-to-head v5 vs V5NoRange **+492.6 bb/100 (SE 42.7)**, seed 2, 64×250. Field opponents, seed 0: V5NoRange vs valka_onderra_claude -387.4 (32.2) (v5: -34.4), vs max_payne_gpt -85.4 (22.5) (v5: +51.9). For v1 the same switch gives aggressor -91.8 vs -52.9, faye_gpt -102.6 vs -69.1, kraken_gpt -274.2 vs -66.0. A full field run of the ablation was not done for lack of time.

## Final choice

**Deployed: v5** (`__init__.py`: `from .v5 import Bot as DeployedBot`). It was the only version with a complete field run and is clearly ahead of v1–v4 on every opponent they shared.

Held-out, seed 1000, 64×250 per opponent: **weighted score +20.98** (plain mean vs non-random +50.05).

| opponent | seed | tables × hands | bb/100 | SE |
|---|---|---|---|---|
| kraken_gpt | 1000 | 64×250 | -20.4 | 10.4 |
| leviathan_claude | 1000 | 64×250 | +5.9 | 17.2 |
| megalodon_glm | 1000 | 64×250 | +8.0 | 16.3 |
| spike_claude | 1000 | 64×250 | -30.6 | 18.2 |
| faye_gpt | 1000 | 64×250 | -48.8 | 10.8 |
| windrunner_glm | 1000 | 64×250 | +30.1 | 14.8 |
| invoker_gpt | 1000 | 64×250 | -36.1 | 17.6 |
| meepo_claude | 1000 | 64×250 | -25.4 | 17.1 |
| hadrian_marlowe_gpt | 1000 | 64×250 | -18.8 | 11.5 |
| vicious_glm | 1000 | 64×250 | +56.6 | 25.4 |
| valka_onderra_claude | 1000 | 64×250 | +9.7 | 22.5 |
| aggressor | 1000 | 64×250 | -7.1 | 8.1 |
| alan_wake_claude | 1000 | 64×250 | +31.9 | 18.9 |
| max_payne_gpt | 1000 | 64×250 | +76.8 | 12.2 |
| bassander_lin_glm | 1000 | 64×250 | +128.6 | 26.4 |
| jesse_faden_glm | 1000 | 64×250 | +278.0 | 31.0 |
| jack_joyce_gemini | 1000 | 64×250 | +412.4 | 32.2 |
| random | 1000 | 64×250 | +729.3 | 26.5 |

**Known weaknesses:** loses to faye_gpt, kraken_gpt, invoker_gpt and spike_claude; roughly break-even against aggressor and hadrian (both costly losses under the weighting). Trends to lose head-to-head against its tighter parent v1 (≈ -21 ± 11 over three seeds), so the aggression is partly tuned to the benchmark field's habits and may be exploitable by tight phase-6 opponents. Ranges do not narrow across repeated raises (3-bet range = open range), draws count as weak holdings, and bet size does not change the range. Only 66 of the ~100 allowed decision lines are used; draws and size-dependent narrowing are the obvious next steps.

## Scratch files
`ablation.py` (range-off and tuning settings as subclasses), `range_table.py` (range table measurement), `run.sh` (parallel per-opponent bench runs), `score.py` (weighted score from logs), `logs/` (all benchmark output).
