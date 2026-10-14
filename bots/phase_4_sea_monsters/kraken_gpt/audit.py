"""Local validation utility; not part of any deployed decision code."""
import ast
import hashlib
import inspect
import json
from collections import namedtuple
from pathlib import Path
import jax
import jax.numpy as jnp
from bench import load

History = namedtuple('History', 'is_valid player_id round action_type')
Record = namedtuple('Record', 'history')


def decision_lines(path):
    source = path.read_text()
    lines = source.splitlines()
    included = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name not in {'__init__', 'init_state', 'metadata'}:
            included.update(range(node.lineno, node.end_lineno + 1))
    return sum(bool(lines[i - 1].strip()) and not lines[i - 1].lstrip().startswith('#') for i in included)


def check_tracking(bot):
    # Three valid heads-up action sequences exercise check/bet, raise/fold, raise/raise,
    # and raise/call. Invalid padding contains a fake opponent bet and must not count.
    cases = [
        ([(0, 0, 4), (1, 0, 5), (0, 1, 1), (1, 1, 2), (0, 1, 2), (1, 1, 0)], (1, 1, 1, 1, 0)),
        ([(1, 0, 4), (0, 0, 5), (1, 0, 1), (0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 2), (1, 2, 2), (0, 2, 1)], (1, 0, 1, 0, 1)),
        ([(0, 0, 4), (1, 0, 5), (0, 0, 2), (1, 0, 1), (1, 1, 1), (0, 1, 2), (1, 1, 1)], (1, 0, 1, 0, 0)),
    ]
    names = ('opponent_free_actions', 'opponent_bets_when_free', 'opponent_facing_bets',
             'opponent_folds_to_bets', 'opponent_raises_facing_bets')
    state = bot.init_state(jax.random.PRNGKey(0))
    for rows, increments in cases:
        valid_count = len(rows)
        padded = rows + [(1, 3, 2)] * (12 - valid_count)
        history = History(jnp.arange(12) < valid_count,
                          jnp.array([row[0] for row in padded], jnp.int32),
                          jnp.array([row[1] for row in padded], jnp.int32),
                          jnp.array([row[2] for row in padded], jnp.int32))
        updated = jax.jit(bot.observe)(Record(history), state, jax.random.PRNGKey(1))
        for name, count in zip(names, increments):
            if name in state:
                assert bool(jnp.isclose(updated[name] - state[name], count)), name
        assert jax.tree.structure(state) == jax.tree.structure(updated)
        for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(updated)):
            assert before.shape == after.shape == ()
            assert before.dtype == after.dtype == jnp.float32
        state = updated


prior_audit_path = Path('audit_results.json')
prior_audit = json.loads(prior_audit_path.read_text()) if prior_audit_path.exists() else []
for previous in prior_audit:
    version_path = Path(previous['version'] + '.py')
    assert hashlib.sha256(version_path.read_bytes()).hexdigest() == previous['sha256'], 'Benchmarked version changed'
results = []
for path in sorted(Path('.').glob('v[0-9]*.py'), key=lambda path: int(path.stem[1:])):
    tree = ast.parse(path.read_text())
    assert [node.name for node in tree.body if isinstance(node, ast.ClassDef)] == ['Bot']
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(item.name in {'jax', 'jax.numpy'} for item in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.level == 1 and node.module == 'equity') or node.module.startswith('poker_bots')
    bot = load(path.stem + ':Bot')
    metadata = bot.metadata()
    assert set(metadata['params']) == set(inspect.signature(bot.__init__).parameters)
    assert metadata['contributions'] == [{'role': 'requester', 'who': 'a'}, {'role': 'generator', 'interface': 'GPT (Codex)'}]
    assert 1 <= bot.num_samples <= 256
    count = decision_lines(path)
    assert count <= 110
    if hasattr(bot, 'observe'):
        check_tracking(bot)
    results.append({'version': path.stem, 'decision_lines': count,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'metadata': metadata})
Path('audit_results.json').write_text(json.dumps(results, indent=2) + '\n')
print('PASS: imports, provenance, line budget, sample cap, and JIT tracking counts/shapes/dtypes')
for result in results:
    print(result['version'], result['decision_lines'], 'decision lines')

# The stateless controls must make exactly the same decisions as their parents
# before any evidence has been learned, with the same observations and keys.
Observation = namedtuple('Observation', 'hole_cards board stacks seated in_hand contributed current_round current_bet pot button can_raise raise_lower_bound raise_upper_bound big_blind history')
for adaptive_name, control_name in [('v6', 'v8'), ('v7', 'v9'), ('v10', 'v11')]:
    if not Path(control_name + '.py').exists():
        continue
    adaptive = load(adaptive_name + ':Bot')
    control = load(control_name + ':Bot')
    prior_state = adaptive.init_state(jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(5), 8)
    for street in range(4):
        board_count = [0, 3, 4, 5][street]
        board = jnp.where(jnp.arange(5) < board_count, jnp.array([0, 14, 28, 42, 8]), -1)
        history = History(jnp.array([True, True, True, True, False, False]),
                          jnp.array([0, 1, 0, 1, -1, -1], jnp.int32),
                          jnp.array([0, 0, street, street, 0, 0], jnp.int32),
                          jnp.array([4, 5, 2, 2, 3, 3], jnp.int32))
        obs = Observation(jnp.array([7, 23], jnp.int32), board, jnp.array([190, 180], jnp.int32),
                          jnp.array([True, True]), jnp.array([True, True]), jnp.array([10, 20], jnp.int32),
                          jnp.asarray(street, jnp.int32), jnp.asarray(20, jnp.int32), jnp.asarray(50, jnp.int32),
                          jnp.asarray(0, jnp.int32), jnp.asarray(True), jnp.asarray(30, jnp.int32),
                          jnp.asarray(200, jnp.int32), jnp.asarray(2, jnp.int32), history)
        adaptive_actions = jax.jit(jax.vmap(lambda key: adaptive.act(obs, prior_state, key)[0]))(keys)
        control_actions = jax.jit(jax.vmap(lambda key: control.act(obs, None, key)[0]))(keys)
        assert bool(jnp.all(adaptive_actions.type == control_actions.type))
        assert bool(jnp.all(adaptive_actions.value == control_actions.value))
    print('PASS: exact prior-state action equality', adaptive_name, control_name)
