"""Scratch: calibrate preflop_score thresholds to hand percentiles (all 1326 combos)."""
import itertools
import jax.numpy as jnp

def preflop_score(c0, c1):
    r = jnp.array([c0 % 13, c1 % 13])
    hi, lo = jnp.max(r), jnp.min(r)
    suited = (c0 // 13) == (c1 // 13)
    gap = hi - lo
    pair_bonus = jnp.where(hi == lo, 0.45 + hi / 13 * 0.35, 0.0)
    gap_penalty = jnp.where(hi == lo, 0.0, jnp.minimum(gap, 4) * 0.022)
    return float(hi / 12 * 0.42 + lo / 12 * 0.18 + pair_bonus + suited * 0.08 - gap_penalty)

scores = sorted((preflop_score(a, b), f"{a%13}-{b%13}{'s' if a//13==b//13 else 'o'}")
                for a, b in itertools.combinations(range(52), 2))
n = len(scores)
for pct in [5, 10, 15, 25, 35, 45, 50, 55, 65, 75, 85, 88, 92, 95, 98]:
    s = scores[int(n * pct / 100)]
    print(f"top {100-pct:>2}% kept: score > {s[0]:.3f}  (worst kept e.g. {s[1]})")
print("min", scores[0], "max", scores[-1])
