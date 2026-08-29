"""
Minimal undefined-name checker.

`python -m py_compile` only proves a file PARSES. It cannot catch a name that is
read but never bound — that is a runtime NameError, which is how
"name 'p_te' is not defined" reached production. This walks each function's
scope and reports every Name that is loaded but never bound in that function,
any enclosing function, module scope, or builtins.

Not a full type checker. It catches exactly the class of bug that has been
biting: renamed variables, unpacking changes, and stale references left behind
by a search-and-replace edit.

Usage:  python3 lint_names.py file.py [file.py ...]
"""
from __future__ import annotations

import ast
import builtins
import sys
from typing import Dict, List, Set, Tuple

BUILTINS: Set[str] = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__"}


def _bound_names(node: ast.AST) -> Set[str]:
    """Every name bound anywhere inside `node`, not descending into nested defs."""
    out: Set[str] = set()

    class Collector(ast.NodeVisitor):
        def __init__(self, root: bool = False):
            self.root = root

        def visit_Name(self, n: ast.Name):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                out.add(n.id)

        def visit_arg(self, n: ast.arg):
            out.add(n.arg)

        def visit_alias(self, n: ast.alias):
            name = n.asname or n.name.split(".")[0]
            out.add(name)

        def visit_ExceptHandler(self, n: ast.ExceptHandler):
            if n.name:
                out.add(n.name)
            self.generic_visit(n)

        def visit_Global(self, n: ast.Global):
            out.update(n.names)

        def visit_Nonlocal(self, n: ast.Nonlocal):
            out.update(n.names)

        def _nested(self, n):
            # The def's own name is bound here; its body is a separate scope.
            out.add(n.name)

        def visit_FunctionDef(self, n):
            if self.root:
                for a in _all_args(n.args):
                    out.add(a)
                self.generic_visit(n)
            else:
                self._nested(n)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, n):
            if self.root:
                self.generic_visit(n)
            else:
                self._nested(n)

        def visit_Lambda(self, n):
            if self.root:
                for a in _all_args(n.args):
                    out.add(a)
                self.generic_visit(n)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        c = Collector(root=True)
        for a in _all_args(node.args):
            out.add(a)
        body = node.body if isinstance(node.body, list) else [node.body]
        for stmt in body:
            c.root = False
            Collector(root=False).visit(stmt)
            _walk_non_scope(stmt, out)
    else:
        for stmt in getattr(node, "body", []):
            Collector(root=False).visit(stmt)
            _walk_non_scope(stmt, out)
    return out


def _all_args(a: ast.arguments) -> List[str]:
    names = [x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _walk_non_scope(node: ast.AST, out: Set[str]) -> None:
    """Collect bindings, stopping at nested function/class boundaries."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(child.name)
            continue
        if isinstance(child, ast.Lambda):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            out.add(child.id)
        elif isinstance(child, ast.alias):
            out.add(child.asname or child.name.split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            out.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            out.update(child.names)
        _walk_non_scope(child, out)


def _loaded_names(node: ast.AST) -> List[Tuple[str, int]]:
    """Names read inside this scope, not descending into nested defs."""
    found: List[Tuple[str, int]] = []

    def walk(n: ast.AST, top: bool = False):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                # Default values and decorators evaluate in THIS scope.
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    for d in list(child.args.defaults) + [x for x in child.args.kw_defaults if x]:
                        walk_expr(d)
                for d in getattr(child, "decorator_list", []):
                    walk_expr(d)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                found.append((child.id, child.lineno))
            walk(child)

    def walk_expr(n: ast.AST):
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                found.append((sub.id, sub.lineno))

    walk(node, top=True)
    return found


def check(path: str) -> List[str]:
    tree = ast.parse(open(path).read(), filename=path)
    module_names: Set[str] = set()
    _walk_non_scope(tree, module_names)

    problems: List[str] = []

    def visit_scope(node: ast.AST, enclosing: Set[str]):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local = _bound_names(node)
            visible = enclosing | local | BUILTINS
            for name, line in _loaded_names(node):
                if name not in visible:
                    problems.append(f"{path}:{line}: undefined name '{name}' in {node.name}()")
            child_scope = visible
        else:
            child_scope = enclosing | BUILTINS

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_scope(child, child_scope)
            elif isinstance(child, ast.ClassDef):
                for gc in ast.iter_child_nodes(child):
                    if isinstance(gc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_scope(gc, child_scope)
            else:
                for gc in ast.walk(child):
                    if isinstance(gc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        visit_scope(gc, child_scope)
                break

    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_scope(child, module_names)
        elif isinstance(child, ast.ClassDef):
            for gc in ast.iter_child_nodes(child):
                if isinstance(gc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit_scope(gc, module_names)
    return problems


if __name__ == "__main__":
    all_problems: List[str] = []
    for p in sys.argv[1:]:
        all_problems.extend(check(p))
    if all_problems:
        for line in sorted(set(all_problems)):
            print(line)
        print(f"\n{len(set(all_problems))} problem(s)")
        sys.exit(1)
    print("no undefined names found")
