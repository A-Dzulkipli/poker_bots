"""Scratch: count decision-code lines (act, observe, helpers) of a version file."""
import ast, sys
src = open(sys.argv[1]).read()
tree = ast.parse(src)
lines = src.splitlines()
skip = {"__init__", "init_state", "metadata"}
total = 0
def count(fn):
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    n = 1  # def line
    for ln in range(body[0].lineno, fn.end_lineno + 1):
        s = lines[ln - 1].strip()
        if s and not s.startswith("#"):
            n += 1
    return n
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        total += count(node)
    if isinstance(node, ast.ClassDef):
        for f in node.body:
            if isinstance(f, ast.FunctionDef) and f.name not in skip:
                total += count(f)
print(sys.argv[1], total)
