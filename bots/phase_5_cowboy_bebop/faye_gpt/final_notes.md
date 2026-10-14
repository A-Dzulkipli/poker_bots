
## Board texture and draws in v10

Card ranks and suits use the contract's modulo-13 / integer-division encoding; invalid board slots are masked. These are explicit heuristics on top of Monte Carlo showdown equity, which already accounts for possible runouts.

| Feature | Computation | Decision effect at deployed defaults |
|---|---|---|
| Paired board | Any dealt board rank occurs at least twice | Adds 1/3 to normalized danger |
| Suit concentration | Maximum dealt board count in any suit | At least two contributes 1/2 to wetness; at least three contributes 1/3 to danger |
| Connected ranks | Maximum number of distinct board ranks in any five-rank straight window, including the wheel | Three contributes 1/2 to wetness; four contributes 1/3 to danger |
| High board | Highest dealt rank is queen or higher | A dry high board qualifies for an occasional in-position stab |
| Flush improvement cards | Four visible cards of a suit, at least one in our hand; count unseen cards of that suit | Included in next-card draw probability and semibluff support |
| Straight improvement cards | Four distinct ranks in a five-rank window, with at least one rank supplied uniquely by a hole card | Count missing-rank cards; exclude board-only completions and already-made straights |
| Overcard improvement cards | Unpaired hole ranks higher than all board ranks, when no rank is paired | Count unseen cards pairing an overcard |
| Street transition | Use only dealt board cards and denominator 50 minus board count | Draw probability is set to zero on the river; suit, connection and pairing features are recomputed |

Unseen improvement cards are counted as a union over all 52 card IDs, excluding known cards, so overlapping straight/flush/overcard possibilities are not double-counted. They are **candidate improvement cards, not guaranteed winning outs**: opponent holdings can make them dirty. Next-card probability uses one card even on the flop; the Monte Carlo estimate separately includes the full runout.

Let E be uniform-opponent equity, W normalized wetness, G normalized danger, D next-card improvement probability, F estimated fold rate, and b the base pot fraction. Postflop adjusted equity loses `0.06 × G × (1 − E)` and gains `0.12 × D`. Bet sizing grows from `b = 0.40` to `0.40 + 0.06 × W`, i.e. 0.40, 0.43 or 0.46 of the pot after calling, before legal clipping. A draw with D ≥ 0.15 and E ≥ 0.40 qualifies for an occasional semibluff; dry queen-high-or-higher boards qualify in position.

V10 also adds `(1 − F) × D × (1 + 2b) / (1 + b)` to the fold-equivalent steal estimate. This is the algebraic contribution of called-bet improvement equity to a simplified one-street bet EV, not a clean-outs guarantee. Crossing `b / (1 + b) + 0.04` raises bluff frequency from 0.12 on eligible hands to 0.80 on a free postflop action. Thus drawing potential can change betting frequency directly, rather than merely adding a small call margin.

### Feature ablation evidence

All rows below are matched on seed, 128 tables per opponent, and 250 hands per table. Disabling `features_enabled` removes explicit texture discounts, draw bonuses, wet-board sizing, texture/draw eligibility and draw-supported steal credit. Uniform-equity Monte Carlo still sees the board; otherwise the comparison would change more than the requested feature layer.

| Comparison | Features enabled | Features disabled | Enabled minus disabled |
|---|---:|---:|---:|
| v7 field, training seed 2 | +49.35 | +49.26 | +0.09 weighted |
| v10 field, held-out seed 1000 | +51.86 | +49.18 | +2.68 weighted |

Direct play, v7 versus its disabled counterpart on seed 2: **+0.80 bb/100, SE 3.80**. Direct play, deployed v10 versus its disabled counterpart on seed 1000: **-6.10 bb/100, SE 3.99**.

The v7 field difference and head-to-head result do not establish a benefit. The v10 direct-match point estimate favors the disabled control and is within two standard errors of zero, so it also does not establish a benefit. The deployed comparison is the measured effect for v10; modest differences must be interpreted with the reported uncertainty, not as proof that each feature helps. There is no per-feature causal isolation, multiple experiments were tried, and no formal aggregate-score significance test was performed. The mandatory features demonstrably alter decisions, but their performance value is not assumed from that fact.

Disabled v10 field results:

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 1000 | 128 × 250 | +4.99 | 5.47 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| leviathan_claude | 1000 | 128 × 250 | +44.25 | 5.50 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| megalodon_glm | 1000 | 128 × 250 | +50.23 | 7.04 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| windrunner_glm | 1000 | 128 × 250 | +70.39 | 5.03 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| invoker_gpt | 1000 | 128 × 250 | +38.54 | 4.29 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| meepo_claude | 1000 | 128 × 250 | +38.73 | 4.13 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +55.93 | 3.32 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| valka_onderra_claude | 1000 | 128 × 250 | +52.95 | 4.99 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| aggressor | 1000 | 128 × 250 | +25.87 | 4.12 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| alan_wake_claude | 1000 | 128 × 250 | +75.41 | 3.82 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| bassander_lin_glm | 1000 | 128 × 250 | +77.31 | 6.37 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| max_payne_gpt | 1000 | 128 × 250 | +58.63 | 4.09 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| jesse_faden_glm | 1000 | 128 × 250 | +120.76 | 7.79 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| jack_joyce_gemini | 1000 | 128 × 250 | +120.89 | 7.96 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |
| random | 1000 | 128 × 250 | +112.26 | 4.87 | [field_ablation_v10_seed1000.log](field_ablation_v10_seed1000.log) |

## Final choice

Deploy **v10**, using the one-line export in `__init__.py`. Its held-out weighted score is **+51.86**, versus **+50.06** for v7 and **+45.46** for v8, each evaluated once on seed 1000 at 128 × 250 per opponent. The deployed version has the higher held-out score of the three feature-enabled finalists. V10's training score was +54.45, compared with v7's +53.35; that small training difference was not treated as a reliable improvement by itself.

The direct held-out parent comparison is **+3.00 bb/100 (SE 4.46)** against v7. All ten other written versions, including later experiment v11, were checked, rather than only selecting favorable ancestors. The disabled-feature strategy is a required control and is not eligible for deployment under this task's explicit board/draw requirement.

### Held-out field, every benchmark

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| kraken_gpt | 1000 | 128 × 250 | -2.47 | 5.63 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| leviathan_claude | 1000 | 128 × 250 | +54.71 | 5.54 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| megalodon_glm | 1000 | 128 × 250 | +56.79 | 7.22 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| windrunner_glm | 1000 | 128 × 250 | +68.72 | 5.30 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| invoker_gpt | 1000 | 128 × 250 | +38.27 | 4.47 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| meepo_claude | 1000 | 128 × 250 | +37.92 | 4.39 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| hadrian_marlowe_gpt | 1000 | 128 × 250 | +55.83 | 3.27 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| valka_onderra_claude | 1000 | 128 × 250 | +56.99 | 5.53 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| aggressor | 1000 | 128 × 250 | +21.93 | 3.97 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| alan_wake_claude | 1000 | 128 × 250 | +80.82 | 4.14 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| bassander_lin_glm | 1000 | 128 × 250 | +102.43 | 7.11 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| max_payne_gpt | 1000 | 128 × 250 | +56.65 | 4.03 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| jesse_faden_glm | 1000 | 128 × 250 | +134.14 | 7.59 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| jack_joyce_gemini | 1000 | 128 × 250 | +124.75 | 9.11 | [field_v10_seed1000.log](field_v10_seed1000.log) |
| random | 1000 | 128 × 250 | +108.64 | 5.17 | [field_v10_seed1000.log](field_v10_seed1000.log) |

### Held-out own-version and ablation matches

| Opponent | Seed | Tables × hands | bb/100 | SE | Log |
|---|---:|---:|---:|---:|---|
| ablation_v10:Bot | 1000 | 128 × 250 | -6.10 | 3.99 | [ablation_match_v10_seed1000.log](ablation_match_v10_seed1000.log) |
| v11:Bot | 1000 | 128 × 250 | +8.79 | 4.89 | [heldout_self_v10_v11_seed1000.log](heldout_self_v10_v11_seed1000.log) |
| v1:Bot | 1000 | 128 × 250 | +55.94 | 4.77 | [heldout_self_v10_v1_seed1000.log](heldout_self_v10_v1_seed1000.log) |
| v2:Bot | 1000 | 128 × 250 | +64.18 | 2.65 | [heldout_self_v10_v2_seed1000.log](heldout_self_v10_v2_seed1000.log) |
| v3:Bot | 1000 | 128 × 250 | +15.85 | 4.66 | [heldout_self_v10_v3_seed1000.log](heldout_self_v10_v3_seed1000.log) |
| v4:Bot | 1000 | 128 × 250 | +11.56 | 5.17 | [heldout_self_v10_v4_seed1000.log](heldout_self_v10_v4_seed1000.log) |
| v5:Bot | 1000 | 128 × 250 | -1.58 | 4.57 | [heldout_self_v10_v5_seed1000.log](heldout_self_v10_v5_seed1000.log) |
| v6:Bot | 1000 | 128 × 250 | +2.98 | 4.67 | [heldout_self_v10_v6_seed1000.log](heldout_self_v10_v6_seed1000.log) |
| v7:Bot | 1000 | 128 × 250 | +3.00 | 4.46 | [heldout_self_v10_v7_seed1000.log](heldout_self_v10_v7_seed1000.log) |
| v8:Bot | 1000 | 128 × 250 | +0.86 | 3.72 | [heldout_self_v10_v8_seed1000.log](heldout_self_v10_v8_seed1000.log) |
| v9:Bot | 1000 | 128 × 250 | +6.91 | 4.95 | [heldout_self_v10_v9_seed1000.log](heldout_self_v10_v9_seed1000.log) |

### Known weaknesses and limits

- The strongest field opponent remains close to break-even or slightly losing in held-out play; this is not a claim to beat every opponent reliably.
- The main range model is a linear strength-percentile approximation to random-hand equity, not a solved strategy or a calibrated posterior over hands. Sixty-four samples introduce decision noise.
- Aggregate raise/fold rates mix situations and are affected by our own changing strategy. The twelve-action prior stabilizes early play but does not remove that selection effect.
- Draw outs can be dirty, and the improvement bonus overlaps information already present in Monte Carlo equity. The ablation therefore matters; feature usefulness is not guaranteed.
- The low bet sizes and frequent steals can be exploited by opponents that call or reraise appropriately. Validation against the fixed field cannot establish general superiority against unseen phase-5 lineages.
- v9's additional reraise adaptation earned no clear gain. v11's sampled range model lost to v7 by 19.72 bb/100 (SE 5.30) and was discarded without a full field run.
- No source version was revised after benchmarking. The initial out-of-directory filename search and the early parameter-exposure shortfall are disclosed above; they are not silently erased.

### Verification

```text
v10:Bot metadata, sample cap, single class: PASS
JIT/vmap shapes, int32 actions, legal raise totals, no free folds: PASS
Features enabled versus disabled: 13 action-type changes; 133 total action changes / 768 probes
```

All eleven versions passed smoke execution. The final version uses **64 samples per decision**, fixed-shape scalar float32 summary counts, no stored random keys, and **92 decision-code lines**. Static checks verified imports, JSON metadata and provenance, sample caps, one class per version, and fingerprints of frozen sources. The exact parameter-only ablation difference was checked. `bench.py` and `equity.py` retain their recorded hashes.
