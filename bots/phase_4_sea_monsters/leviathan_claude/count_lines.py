"""Scratch: count decision-code lines (non-blank, non-comment, excluding docstrings) in act/observe/helpers."""
import ast, sys
EXCLUDE = {"__init__", "init_state", "metadata"}
for path in sys.argv[1:]:
    src = open(path).read(); lines = src.splitlines(); tree = ast.parse(src); total = 0
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name not in EXCLUDE]:
            doc = ast.get_docstring(fn); body_start = fn.body[0].end_lineno if doc else fn.lineno - 1
            seg = lines[fn.lineno - 1:fn.end_lineno]
            skip = set(range(fn.body[0].lineno, fn.body[0].end_lineno + 1)) if doc else set()
            total += sum(1 for i, l in enumerate(seg, fn.lineno) if l.strip() and not l.strip().startswith("#") and i not in skip)
    print(path, total)
