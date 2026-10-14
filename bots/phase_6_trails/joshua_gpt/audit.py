"""Audit only this directory's source files and benchmark logs."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import jax
import bench
from report_tools import decision_lines

hashes=json.loads(Path('source_hashes.json').read_text())
checks=[]
for name, digest in hashes.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name + ' changed after freezing'
checks.append('Frozen versions and shared helpers match saved hashes.')
for path in sorted(Path('.').glob('v[0-9]*.py')):
    module=importlib.import_module('joshua_gpt.' + path.stem)
    bot=module.Bot()
    tree=ast.parse(path.read_text())
    classes=[node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert len(classes) == 1 and classes[0].name == 'Bot'
    assert bot.init_state(jax.random.PRNGKey(0)) is None
    assert not hasattr(bot, 'observe')
    assert 1 <= bot.samples <= 256
    assert decision_lines(path.stem) <= 110
    assert 'hand_count' not in path.read_text()
    json.dumps(bot.metadata())
    for method in classes[0].body:
        if isinstance(method, ast.FunctionDef) and method.name != '__init__':
            for node in ast.walk(method):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    assert not isinstance(node.value, ast.Name) or node.value.id != 'self'
    act=next(node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name=='act')
    returned=act.body[-1].value
    assert isinstance(returned, ast.Tuple) and isinstance(returned.elts[1], ast.Name) and returned.elts[1].id=='state'
    checks.append(f'{path.name}: {decision_lines(path.stem)} decision lines, {bot.samples} samples, stateless, JSON metadata, unchanged state return.')
Path('audit.log').write_text('\n'.join(checks) + '\n')
print('\n'.join(checks))
