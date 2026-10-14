"""Verify the requested deliverables using only files in this bot directory."""
import importlib
import json
from pathlib import Path
import bench
from report_tools import records, RESULT, SCORE

selection=json.loads(Path('selection.json').read_text())
final=selection['final']
assert Path('__init__.py').read_text() == f'from .{final} import Bot as DeployedBot\n'
for version in range(1,9):
    name='v'+str(version)
    smoke=Path(name+'_smoke.log').read_text()
    assert '2x20' in smoke and 'Traceback' not in smoke
    field=Path(name+'_field_s0.log').read_text()
    assert len(RESULT.findall(field)) == 18 and SCORE.search(field)
for name in ['v3','v5','v7']:
    field=Path(name+'_field_s1000.log').read_text()
    assert len(RESULT.findall(field)) == 18 and SCORE.search(field)
    assert {row[1].strip() for row in RESULT.findall(field)} == set(bench.FIELD)
rows=records()
for earlier in range(1,int(final[1:])):
    assert any(row['version']==final and row['opponent']==f'v{earlier}:Bot' and row['seed']==1000 for row in rows)
control='v8' if final=='v3' else 'v6'
assert any(row['version']==final and row['opponent']==control+':Bot' for row in rows)
final_params=importlib.import_module('joshua_gpt.'+final).Bot().metadata()['params']
control_params=importlib.import_module('joshua_gpt.'+control).Bot().metadata()['params']
for key,value in control_params.items():
    if key!='range_power':
        assert final_params[key] == value, key
assert control_params['range_power'] == 0.0 and final_params['range_power'] > 0.0
assert len(json.loads(Path(final+'_ranges.json').read_text())) >= 8
report=Path('README.md').read_text()
assert 'Sandbox deviation' in report and 'Wall-clock' in report
assert all('## v'+str(version) in report for version in range(1,9))
print('PASS: eight smoke tests, eight development fields, three one-time held-out fields.')
print('PASS: deployed version versus every earlier version and matching uniform-hand control.')
print('PASS: deployment export, measured range table, complete version sections and provenance.')
print('Unique 250-hand benchmark rows:',len(rows))
