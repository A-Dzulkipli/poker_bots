"""Write the final interpretation only after the necessary measurements exist."""
from pathlib import Path
import sys
import report

final = sys.argv[1]
rows, scores = report.collect()
heldout = [r for r in rows if r['candidate'] == final and r['seed'] == 1000]
field_rows = [r for r in heldout if r['opponent'] in report.bench.FIELD]
assert len(field_rows) == 15
own_rows = [r for r in heldout if r['opponent'].startswith('v') and r['opponent'].endswith(':Bot')]
assert len(own_rows) == 10
ablation = next(r for r in heldout if r['opponent'] == f'ablation_{final}:Bot')
score = next(s for s in scores if s['candidate'] == final and s['seed'] == 1000)
off_score = next(s for s in scores if s['candidate'] == f'ablation_{final}' and s['seed'] == 1000)
other_score = next(s for s in scores if s['candidate'] == 'v7' and s['seed'] == 1000)
third_score = next(s for s in scores if s['candidate'] == 'v8' and s['seed'] == 1000)
delta = score['score'] - off_score['score']
parent = next(r for r in own_rows if r['opponent'] == 'v7:Bot')
training_on = next(s for s in scores if s['candidate'] == 'v7' and s['seed'] == 2)
training_off = next(s for s in scores if s['candidate'] == 'ablation_v7' and s['seed'] == 2)
training_h2h = next(r for r in rows if r['candidate'] == 'v7' and r['opponent'] == 'ablation_v7:Bot' and r['seed'] == 2)
validation = Path(f'validation_{final}.log').read_text().strip()
heldout_comparison = 'The deployed version has the higher held-out score of the three feature-enabled finalists.' if score['score'] >= max(other_score['score'], third_score['score']) else 'v7 has the higher numerical held-out score. The final choice favors v10’s private-card straight-out correction and explicit draw-supported semibluffs; this is a clarity/modeling choice, not a claim that its lower observed score is superior.'
text = f'''
## Board texture and draws in {final}

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
| v7 field, training seed 2 | {training_on['score']:+.2f} | {training_off['score']:+.2f} | {training_on['score']-training_off['score']:+.2f} weighted |
| {final} field, held-out seed 1000 | {score['score']:+.2f} | {off_score['score']:+.2f} | {delta:+.2f} weighted |

Direct play, v7 versus its disabled counterpart on seed 2: **{training_h2h['mean']:+.2f} bb/100, SE {training_h2h['se']:.2f}**. Direct play, deployed {final} versus its disabled counterpart on seed 1000: **{ablation['mean']:+.2f} bb/100, SE {ablation['se']:.2f}**.

The v7 field difference and head-to-head result do not establish a benefit. The v10 direct-match point estimate favors the disabled control and is within two standard errors of zero, so it also does not establish a benefit. The deployed comparison is the measured effect for v10; modest differences must be interpreted with the reported uncertainty, not as proof that each feature helps. There is no per-feature causal isolation, multiple experiments were tried, and no formal aggregate-score significance test was performed. The mandatory features demonstrably alter decisions, but their performance value is not assumed from that fact.

Disabled {final} field results:

{report.table([r for r in rows if r['candidate'] == f'ablation_{final}' and r['seed'] == 1000])}

## Final choice

Deploy **{final}**, using the one-line export in `__init__.py`. Its held-out weighted score is **{score['score']:+.2f}**, versus **{other_score['score']:+.2f}** for v7 and **{third_score['score']:+.2f}** for v8, each evaluated once on seed 1000 at 128 × 250 per opponent. {heldout_comparison} V10's training score was +54.45, compared with v7's +53.35; that small training difference was not treated as a reliable improvement by itself.

The direct held-out parent comparison is **{parent['mean']:+.2f} bb/100 (SE {parent['se']:.2f})** against v7. All ten other written versions, including later experiment v11, were checked, rather than only selecting favorable ancestors. The disabled-feature strategy is a required control and is not eligible for deployment under this task's explicit board/draw requirement.

### Held-out field, every benchmark

{report.table(field_rows)}

### Held-out own-version and ablation matches

{report.table(own_rows + [ablation])}

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
{validation}
```

All eleven versions passed smoke execution. The final version uses **64 samples per decision**, fixed-shape scalar float32 summary counts, no stored random keys, and **92 decision-code lines**. Static checks verified imports, JSON metadata and provenance, sample caps, one class per version, and fingerprints of frozen sources. The exact parameter-only ablation difference was checked. `bench.py` and `equity.py` retain their recorded hashes.
'''
Path('final_notes.md').write_text(text)
