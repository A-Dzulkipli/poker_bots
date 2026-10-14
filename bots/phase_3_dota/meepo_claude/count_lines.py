"""Scratch tool: counts decision-code lines (act + helpers; excludes imports, docstrings, __init__, init_state, metadata)."""
import ast, sys
src = open(sys.argv[1]).read()
lines = src.splitlines()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
total = 0
for f in cls.body:
    if isinstance(f, ast.FunctionDef) and f.name not in ("__init__", "init_state", "metadata"):
        body = f.body[1:] if isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant) else f.body
        seg = lines[f.lineno - 1:f.end_lineno]
        n = sum(1 for l in seg if l.strip() and not l.strip().startswith("#"))
        if body is not f.body:
            d = f.body[0]; n -= (d.end_lineno - d.lineno + 1)
        total += n
print(sys.argv[1], total)
