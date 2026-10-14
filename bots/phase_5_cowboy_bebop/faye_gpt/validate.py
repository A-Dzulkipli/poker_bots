"""Contract checks on constructed, legal observations; no engine source access."""
import ast
import inspect
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import bench
from poker_bots.poker_game import ActionRecord, PlayerTableState


def observation(board, cost, hole=(8, 7)):
    integer = lambda value: jnp.asarray(value, jnp.int32)
    valid = jnp.arange(100) == 0
    street = {3: 1, 4: 2, 5: 3}[sum(card >= 0 for card in board)]
    history = ActionRecord(
        player_id=jnp.where(valid, 1, -1).astype(jnp.int32),
        action_type=jnp.where(valid, 2, 0).astype(jnp.int32),
        amount=jnp.where(valid, cost, 0).astype(jnp.int32),
        requested_type=jnp.where(valid, 2, 0).astype(jnp.int32),
        requested=jnp.where(valid, cost, 0).astype(jnp.int32),
        is_all_in=jnp.zeros(100, jnp.bool_),
        round=jnp.full(100, street, jnp.int32), is_valid=valid & (cost > 0))
    return PlayerTableState(
        hole_cards=integer(hole), board=integer(board), history=history,
        stacks=integer([200, 200 - cost]), seated=jnp.ones(2, jnp.bool_),
        in_hand=jnp.ones(2, jnp.bool_), button=integer(0), sb_seat=integer(0),
        bb_seat=integer(1), contributed=integer([0, cost]), current_round=integer(street),
        current_bet=integer(cost), min_raise=integer(max(cost, 2)), pot=integer(20 + cost),
        can_raise=jnp.asarray(True), raise_lower_bound=integer(cost + max(cost, 2)),
        raise_upper_bound=integer(200), big_blind=integer(2), hand_count=integer(0))


def check_version(spec):
    bot = bench.load(spec)
    params = bot.metadata()['params']
    assert set(params) == set(inspect.signature(type(bot)).parameters)
    json.dumps(bot.metadata(), allow_nan=False)
    assert 0 < bot.samples <= 256
    source = Path(spec.split(':')[0] + '.py').read_text()
    tree = ast.parse(source)
    assert len([node for node in tree.body if isinstance(node, ast.ClassDef)]) == 1
    statements = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    assert len(statements) == len(set(statements))
    print(spec, 'metadata, sample cap, single class: PASS')
    return bot


if __name__ == '__main__':
    import sys
    spec = sys.argv[1]
    bot = check_version(spec)
    disabled = type(bot)(**(bot.metadata()['params'] | {'features_enabled': False}))
    boards = [(6, 5, 25, -1, -1), (6, 5, 25, 31, -1), (6, 5, 25, 31, 40),
              (12, 24, 39, -1, -1), (12, 24, 39, 4, -1), (12, 24, 39, 4, 17)]
    observations = [observation(board, cost) for board in boards for cost in (0, 4, 12, 30)]
    batch = jax.tree.map(lambda *values: jnp.stack(values), *observations)
    keys = jax.random.split(jax.random.PRNGKey(9), 32)
    def actions(strategy, key):
        return jax.vmap(lambda obs: strategy.act(obs, strategy.init_state(key), key)[0])(batch)
    on = jax.jit(jax.vmap(lambda key: actions(bot, key)))(keys)
    off = jax.jit(jax.vmap(lambda key: actions(disabled, key)))(keys)
    assert on.type.dtype == jnp.int32 and on.value.dtype == jnp.int32
    assert on.type.shape == (32, 24) and on.value.shape == (32, 24)
    assert bool(jnp.all((on.type >= 0) & (on.type <= 2)))
    free = batch.current_bet == 0
    assert bool(jnp.all(jnp.where(free, on.type != 0, True)))
    assert bool(jnp.all(jnp.where(on.type == 2, on.value >= batch.raise_lower_bound, True)))
    assert bool(jnp.all(jnp.where(on.type == 2, on.value <= batch.raise_upper_bound, True)))
    type_changes = int(jnp.sum(on.type != off.type))
    action_changes = int(jnp.sum((on.type != off.type) | (on.value != off.value)))
    print('JIT/vmap shapes, int32 actions, legal raise totals, no free folds: PASS')
    print('Features enabled versus disabled:', type_changes, 'action-type changes;', action_changes, 'total action changes / 768 probes')
    assert action_changes > 0
