"""Local contract checks built only from the observation contract in PROMPT.md."""
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp

# bench supplies the documented local package loader; no engine source is inspected.
from bench import load


def observation(round_number, cost, can_raise, seats=2, raises=0, nuts=False):
    history = SimpleNamespace(
        is_valid=jnp.arange(100) < raises,
        round=jnp.full(100, round_number, dtype=jnp.int32),
        player_id=jnp.ones(100, dtype=jnp.int32),
        action_type=jnp.full(100, 2, dtype=jnp.int32),
    )
    board = [8, 9, 10, 11, 12] if nuts else [0, 14, 28, 42, 7]
    dealt = [0, 3, 4, 5][round_number]
    board = [card if index < dealt else -1 for index, card in enumerate(board)]
    return SimpleNamespace(
        hole_cards=jnp.array([25, 38], dtype=jnp.int32),
        board=jnp.array(board, dtype=jnp.int32), history=history,
        stacks=jnp.full(seats, 200, dtype=jnp.int32),
        seated=jnp.ones(seats, dtype=jnp.bool_), in_hand=jnp.ones(seats, dtype=jnp.bool_),
        button=jnp.asarray(0, dtype=jnp.int32),
        contributed=jnp.zeros(seats, dtype=jnp.int32),
        current_round=jnp.asarray(round_number, dtype=jnp.int32),
        current_bet=jnp.asarray(cost, dtype=jnp.int32),
        pot=jnp.asarray(20 + cost, dtype=jnp.int32),
        big_blind=jnp.asarray(2, dtype=jnp.int32),
        can_raise=jnp.asarray(can_raise),
        raise_lower_bound=jnp.asarray(cost + 2, dtype=jnp.int32),
        raise_upper_bound=jnp.asarray(200, dtype=jnp.int32),
    )


def main():
    protected = json.loads(Path('reference_hashes.json').read_text())
    for file, expected in protected.items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
    for name in sys.argv[1:]:
        bot = load(name + ':Bot')
        assert bot.init_state(jax.random.key(0)) is None
        assert not hasattr(bot, 'observe')
        assert set(inspect.signature(bot.__init__).parameters) == set(bot.metadata()['params'])
        json.dumps(bot.metadata())
        for round_number in range(4):
            for cost, can_raise in [(0, True), (8, True), (200, False)]:
                obs = observation(round_number, cost, can_raise, raises=int(cost > 0))
                actions, state = jax.jit(jax.vmap(lambda key: bot.act(obs, None, key)))(jax.random.split(jax.random.key(7), 4))
                assert state is None
                assert actions.type.shape == (4,) and actions.value.shape == (4,)
                assert actions.type.dtype == jnp.int32 and actions.value.dtype == jnp.int32
                assert bool(jnp.all((actions.type >= 0) & (actions.type <= 2)))
                if cost == 0:
                    assert bool(jnp.all(actions.type != 0)), (name, round_number, 'free fold')
                if not can_raise:
                    assert bool(jnp.all(actions.type != 2)), (name, 'illegal raise')
                raised = actions.type == 2
                assert bool(jnp.all(~raised | (actions.value >= obs.raise_lower_bound)))
                assert bool(jnp.all(~raised | (actions.value <= obs.raise_upper_bound)))
        if int(name[1:]) >= 4:
            # Same effective call, with or without an opponent's unmatched excess.
            for round_number in range(4):
                short = observation(round_number, 40, False, raises=1)
                overbet = observation(round_number, 100, False, raises=1)
                short.stacks = short.stacks.at[0].set(40)
                overbet.stacks = overbet.stacks.at[0].set(40)
                keys = jax.random.split(jax.random.key(9), 32)
                left, _ = jax.jit(jax.vmap(lambda key: bot.act(short, None, key)))(keys)
                right, _ = jax.jit(jax.vmap(lambda key: bot.act(overbet, None, key)))(keys)
                assert bool(jnp.all(left.type == right.type)), (name, 'uncallable excess changed call')
        obs = observation(3, 0, True, seats=3)
        _, state = jax.jit(lambda key: bot.act(obs, jnp.asarray(7), key))(jax.random.key(8))
        assert int(state) == 7
        source = Path(name + '.py').read_text()
        tree = ast.parse(source)
        counts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name not in {'__init__', 'init_state', 'metadata'}:
                lines = source.splitlines()[node.lineno - 1:node.end_lineno]
                counts.append(sum(bool(line.strip()) and not line.strip().startswith('#') for line in lines))
        print(name, 'passed jit/vmap, state, action, metadata and protected-file checks;', sum(counts), 'decision lines', flush=True)


if __name__ == '__main__':
    main()
