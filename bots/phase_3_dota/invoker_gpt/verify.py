"""Local contract checks; not imported by any deployed bot."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from bench import load


def observation(round_id, can_raise, current_bet, stack, board):
    history = SimpleNamespace(
        is_valid=jnp.zeros(100, dtype=jnp.bool_),
        action_type=jnp.zeros(100, dtype=jnp.int32),
        player_id=jnp.full(100, -1, dtype=jnp.int32),
        round=jnp.zeros(100, dtype=jnp.int32),
    )
    return SimpleNamespace(
        hole_cards=jnp.array([0, 14], dtype=jnp.int32),
        board=jnp.array(board, dtype=jnp.int32),
        stacks=jnp.array([stack, 100], dtype=jnp.int32),
        contributed=jnp.array([0, current_bet], dtype=jnp.int32),
        seated=jnp.ones(2, dtype=jnp.bool_),
        in_hand=jnp.ones(2, dtype=jnp.bool_),
        button=jnp.array(1, dtype=jnp.int32),
        current_round=jnp.array(round_id, dtype=jnp.int32),
        current_bet=jnp.array(current_bet, dtype=jnp.int32),
        pot=jnp.array(10 + current_bet, dtype=jnp.int32),
        big_blind=jnp.array(2, dtype=jnp.int32),
        can_raise=jnp.array(can_raise),
        raise_lower_bound=jnp.array(min(stack, max(2, 2 * current_bet)), dtype=jnp.int32),
        raise_upper_bound=jnp.array(stack, dtype=jnp.int32),
        history=history,
    )


def main():
    prior = json.loads(Path('source_audit.json').read_text())
    reports = {}
    cases = [
        observation(3, True, 0, 100, [12, 24, 36, 7, 19]),
        observation(1, False, 0, 100, [12, 24, 36, -1, -1]),
        observation(3, False, 20, 3, [12, 24, 36, 7, 19]),
    ]
    for path in sorted(Path('.').glob('v[0-9]*.py'), key=lambda p: int(p.stem[1:])):
        source = path.read_text()
        digest = hashlib.sha256(source.encode()).hexdigest()
        if path.name in prior:
            assert digest == prior[path.name]['sha256'], 'Benchmarked source changed'
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        assert len(classes) == 1 and classes[0].name == 'Bot'
        methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)]
        assert not any(node.name == 'observe' for node in methods)
        decisions = [node for node in methods if node.name not in ('__init__', 'init_state', 'metadata')]
        lines = source.splitlines()
        count = sum(bool(line.strip()) and not line.lstrip().startswith('#') for node in decisions for line in lines[node.lineno-1:node.end_lineno])
        assert count <= 110
        bot = load(path.stem + ':Bot')
        metadata = bot.metadata()
        json.dumps(metadata)
        assert metadata['version'] == int(path.stem[1:])
        expected_params = set(inspect.signature(type(bot)).parameters)
        assert set(metadata['params']) == expected_params
        assert bot.init_state(jax.random.PRNGKey(0)) is None
        assert 1 <= bot.num_samples <= 256
        try:
            type(bot)(num_samples=257)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing sample cap')
        for case in cases:
            run = jax.jit(jax.vmap(lambda key: bot.act(case, None, key)))
            actions, state = run(jax.random.split(jax.random.PRNGKey(9), 8))
            assert state is None
            assert actions.type.shape == (8,) and actions.value.shape == (8,)
            assert actions.type.dtype == jnp.int32 and actions.value.dtype == jnp.int32
            assert bool(jnp.all((actions.type >= 0) & (actions.type <= 2)))
            if int(case.current_bet) == 0:
                assert bool(jnp.all(actions.type != 0)), 'Folded a free check'
            if not bool(case.can_raise):
                assert bool(jnp.all(actions.type != 2)), 'Raised when not legal'
            raised = actions.type == 2
            legal_size = (actions.value >= case.raise_lower_bound) & (actions.value <= case.raise_upper_bound)
            assert bool(jnp.all(~raised | legal_size))
        reports[path.name] = {'decision_lines': count, 'sha256': digest, 'checks': 'PASS'}
        print(path.name, count, 'decision lines: PASS', flush=True)
    Path('source_audit.json').write_text(json.dumps(reports, indent=2) + '\n')
    print('All source, metadata, JIT/vmap, dtype, free-check and short-stack checks passed.')


if __name__ == '__main__':
    main()
