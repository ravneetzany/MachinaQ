"""Static ingestion of OpenSCAD (.scad) parametric part definitions.

Parses the primitive-call vocabulary actually used by the target parts
library (cyl/cylinder, cube/cuboid, translate/rotate + a handful of BOSL2
single-axis translate shorthands, union/difference, module definitions and
invocation, if/else, and the two list-comprehension `for` forms plus a
statement-level range `for`) into `SurfacePrimitive`-shaped records with an
absolute axis, without invoking the OpenSCAD binary. Anything outside that
vocabulary is reported as unparsed with a reason rather than guessed.

Not a general OpenSCAD interpreter — see design.md decision 3 and its
"Vocabulary addendum" for what is and is not supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .geometry import Axis, Vector3, axis_to_details
from .primitive import SurfacePrimitive

UNDEF = object()  # OpenSCAD's `undef` sentinel


class ScadUnsupported(Exception):
    """Raised when a construct falls outside the supported vocabulary."""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"""
      (?P<WS>\s+)
    | (?P<LCOMMENT>//[^\n]*)
    | (?P<BCOMMENT>/\*.*?\*/)
    | (?P<NUMBER>\d+\.\d+|\.\d+|\d+\.?)
    | (?P<STRING>"(?:[^"\\]|\\.)*")
    | (?P<IDENT>\$?[A-Za-z_][A-Za-z0-9_]*)
    | (?P<OP><=|>=|==|!=|&&|\|\||[-+*/%()\[\]{},;=<>!?:])
""", re.VERBOSE | re.DOTALL)


@dataclass
class Token:
    kind: str
    value: str


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise ScadUnsupported(f"unrecognized character at position {pos}: {source[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind in ("WS", "LCOMMENT", "BCOMMENT"):
            continue
        tokens.append(Token(kind, m.group()))
    return tokens


# ---------------------------------------------------------------------------
# AST nodes (plain tuples/dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class Num:
    value: float


@dataclass
class Str:
    value: str


@dataclass
class Bool:
    value: bool


@dataclass
class Undef:
    pass


@dataclass
class VarRef:
    name: str


@dataclass
class ListLit:
    items: List[Any]


@dataclass
class RangeLit:
    start: Any
    step: Optional[Any]
    end: Any


@dataclass
class ForEachComp:
    var: str
    iterable: Any
    body: Any


@dataclass
class ForCComp:
    inits: List[Tuple[str, Any]]
    cond: Any
    updates: List[Tuple[str, Any]]
    body: Any


@dataclass
class Index:
    target: Any
    index: Any


@dataclass
class FuncCall:
    name: str
    args: List[Any]


@dataclass
class UnaryOp:
    op: str
    operand: Any


@dataclass
class BinOp:
    op: str
    left: Any
    right: Any


@dataclass
class Ternary:
    cond: Any
    then: Any
    otherwise: Any


@dataclass
class Assign:
    name: str
    expr: Any


@dataclass
class If:
    cond: Any
    then_block: List[Any]
    else_block: List[Any]


@dataclass
class ForStmt:
    var: str
    iterable: Any
    body: List[Any]


@dataclass
class ModuleDef:
    name: str
    params: List[Tuple[str, Optional[Any]]]
    body: List[Any]


@dataclass
class CallStmt:
    name: str
    args: List[Tuple[Optional[str], Any]]
    block: Optional[List[Any]]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek_val(self) -> Optional[str]:
        tok = self.peek()
        return tok.value if tok else None

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, value: str) -> Token:
        tok = self.peek()
        if tok is None or tok.value != value:
            raise ScadUnsupported(f"expected {value!r}, got {tok.value if tok else 'EOF'!r}")
        return self.advance()

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    # -- program / statements -------------------------------------------------

    def parse_program(self) -> List[Any]:
        stmts = []
        while not self.at_end():
            stmts.append(self.parse_statement())
        return stmts

    def parse_block(self) -> List[Any]:
        if self.peek_val() == "{":
            self.advance()
            stmts = []
            while self.peek_val() != "}":
                if self.at_end():
                    raise ScadUnsupported("unterminated block")
                stmts.append(self.parse_statement())
            self.advance()
            return stmts
        return [self.parse_statement()]

    def parse_statement(self) -> Any:
        val = self.peek_val()

        if val in ("include", "use"):
            # include <path>; / use <path>; — skip to ';'
            while self.peek_val() != ";":
                if self.at_end():
                    raise ScadUnsupported("unterminated include/use directive")
                self.advance()
            self.advance()
            return CallStmt("__noop__", [], None)

        if val == "module":
            self.advance()
            name = self.advance().value
            self.expect("(")
            params = self.parse_params()
            self.expect(")")
            body = self.parse_block()
            return ModuleDef(name, params, body)

        if val == "if":
            self.advance()
            self.expect("(")
            cond = self.parse_expr()
            self.expect(")")
            then_block = self.parse_block()
            else_block: List[Any] = []
            if self.peek_val() == "else":
                self.advance()
                else_block = self.parse_block()
            return If(cond, then_block, else_block)

        if val == "for":
            self.advance()
            self.expect("(")
            var = self.advance().value
            self.expect("=")
            iterable = self.parse_expr()
            self.expect(")")
            body = self.parse_block()
            return ForStmt(var, iterable, body)

        if val is not None and re.match(r"^\$?[A-Za-z_]", val):
            # IDENT '=' expr ';'   OR   IDENT '(' args ')' (block|';')
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].value == "=":
                name = self.advance().value
                self.advance()  # '='
                expr = self.parse_expr()
                self.expect(";")
                return Assign(name, expr)

            name = self.advance().value
            self.expect("(")
            args = self.parse_args()
            self.expect(")")
            if self.peek_val() == "{":
                block = self.parse_block()
                return CallStmt(name, args, block)
            if self.peek_val() == ";":
                self.advance()
                return CallStmt(name, args, None)
            # A bare call immediately followed by another statement (e.g.
            # `up(z[i]) cyl(...);` — BOSL2 shorthand chaining) is treated as
            # a call whose "block" is the single following statement.
            block = [self.parse_statement()]
            return CallStmt(name, args, block)

        raise ScadUnsupported(f"unsupported statement starting with {val!r}")

    def parse_params(self) -> List[Tuple[str, Optional[Any]]]:
        params: List[Tuple[str, Optional[Any]]] = []
        if self.peek_val() == ")":
            return params
        while True:
            name = self.advance().value
            default = None
            if self.peek_val() == "=":
                self.advance()
                default = self.parse_expr()
            params.append((name, default))
            if self.peek_val() == ",":
                self.advance()
                continue
            break
        return params

    def parse_args(self) -> List[Tuple[Optional[str], Any]]:
        args: List[Tuple[Optional[str], Any]] = []
        if self.peek_val() == ")":
            return args
        while True:
            name = None
            if (
                self.pos + 1 < len(self.tokens)
                and re.match(r"^\$?[A-Za-z_]", self.peek_val() or "")
                and self.tokens[self.pos + 1].value == "="
            ):
                name = self.advance().value
                self.advance()
            expr = self.parse_expr()
            args.append((name, expr))
            if self.peek_val() == ",":
                self.advance()
                continue
            break
        return args

    # -- expressions (precedence climbing) ------------------------------------

    def parse_expr(self) -> Any:
        return self.parse_ternary()

    def parse_ternary(self) -> Any:
        cond = self.parse_or()
        if self.peek_val() == "?":
            self.advance()
            then = self.parse_expr()
            self.expect(":")
            otherwise = self.parse_expr()
            return Ternary(cond, then, otherwise)
        return cond

    def parse_or(self) -> Any:
        left = self.parse_and()
        while self.peek_val() == "||":
            self.advance()
            left = BinOp("||", left, self.parse_and())
        return left

    def parse_and(self) -> Any:
        left = self.parse_not()
        while self.peek_val() == "&&":
            self.advance()
            left = BinOp("&&", left, self.parse_not())
        return left

    def parse_not(self) -> Any:
        if self.peek_val() == "!":
            self.advance()
            return UnaryOp("!", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> Any:
        left = self.parse_additive()
        if self.peek_val() in ("<", ">", "<=", ">=", "==", "!="):
            op = self.advance().value
            right = self.parse_additive()
            return BinOp(op, left, right)
        return left

    def parse_additive(self) -> Any:
        left = self.parse_multiplicative()
        while self.peek_val() in ("+", "-"):
            op = self.advance().value
            left = BinOp(op, left, self.parse_multiplicative())
        return left

    def parse_multiplicative(self) -> Any:
        left = self.parse_unary()
        while self.peek_val() in ("*", "/", "%"):
            op = self.advance().value
            left = BinOp(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> Any:
        if self.peek_val() == "-":
            self.advance()
            return UnaryOp("-", self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self) -> Any:
        node = self.parse_primary()
        while True:
            if self.peek_val() == "[":
                self.advance()
                index = self.parse_expr()
                self.expect("]")
                node = Index(node, index)
            elif self.peek_val() == "(" and isinstance(node, VarRef):
                self.advance()
                args = []
                if self.peek_val() != ")":
                    args.append(self.parse_expr())
                    while self.peek_val() == ",":
                        self.advance()
                        args.append(self.parse_expr())
                self.expect(")")
                node = FuncCall(node.name, args)
            else:
                break
        return node

    def parse_primary(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise ScadUnsupported("unexpected end of expression")

        if tok.kind == "NUMBER":
            self.advance()
            return Num(float(tok.value))
        if tok.kind == "STRING":
            self.advance()
            return Str(tok.value[1:-1])
        if tok.value == "true":
            self.advance()
            return Bool(True)
        if tok.value == "false":
            self.advance()
            return Bool(False)
        if tok.value == "undef":
            self.advance()
            return Undef()
        if tok.value == "(":
            self.advance()
            expr = self.parse_expr()
            self.expect(")")
            return expr
        if tok.value == "[":
            return self.parse_list_or_range_or_comprehension()
        if tok.kind == "IDENT":
            self.advance()
            return VarRef(tok.value)

        raise ScadUnsupported(f"unsupported expression token {tok.value!r}")

    def parse_list_or_range_or_comprehension(self) -> Any:
        self.expect("[")
        if self.peek_val() == "]":
            self.advance()
            return ListLit([])

        if self.peek_val() == "for":
            self.advance()
            self.expect("(")
            inits = self.parse_assign_list()
            if self.peek_val() == ";":
                self.advance()
                cond = self.parse_expr()
                self.expect(";")
                updates = self.parse_assign_list()
                self.expect(")")
                body = self.parse_expr()
                self.expect("]")
                return ForCComp(inits, cond, updates, body)
            self.expect(")")
            body = self.parse_expr()
            self.expect("]")
            if len(inits) != 1:
                raise ScadUnsupported("for-each list comprehension expects exactly one binding")
            var, iterable = inits[0]
            return ForEachComp(var, iterable, body)

        first = self.parse_expr()
        if self.peek_val() == ":":
            self.advance()
            second = self.parse_expr()
            if self.peek_val() == ":":
                self.advance()
                third = self.parse_expr()
                self.expect("]")
                return RangeLit(first, second, third)
            self.expect("]")
            return RangeLit(first, None, second)

        items = [first]
        while self.peek_val() == ",":
            self.advance()
            items.append(self.parse_expr())
        self.expect("]")
        return ListLit(items)

    def parse_assign_list(self) -> List[Tuple[str, Any]]:
        assigns = []
        while True:
            name = self.advance().value
            self.expect("=")
            expr = self.parse_expr()
            assigns.append((name, expr))
            if self.peek_val() == ",":
                self.advance()
                continue
            break
        return assigns


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

Env = Dict[str, Any]

_TRANSLATE_SHORTHANDS = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "fwd": (0.0, -1.0, 0.0),
}

# Argument names we actually evaluate for geometry-relevant calls — every
# other kwarg (anchor, chamfer1/2, rounding, edges, spin, $fn, ...) is left
# unevaluated so unresolved BOSL2 constants (BOTTOM, TOP, ...) never need
# to be looked up.
_CYL_DIAMETER_KEYS = ("d",)
_CYL_RADIUS_KEYS = ("r",)
_CYL_HEIGHT_KEYS = ("h", "height", "l", "length")


def is_truthy(value: Any) -> bool:
    if value is UNDEF or value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return len(value) > 0
    if isinstance(value, list):
        return len(value) > 0
    return True


def _expand_range(rng: "_RangeValue") -> List[float]:
    step = rng.step if rng.step is not None else 1.0
    if step == 0:
        raise ScadUnsupported("range step of 0")
    values = []
    v = rng.start
    if step > 0:
        while v <= rng.end + 1e-9:
            values.append(v)
            v += step
    else:
        while v >= rng.end - 1e-9:
            values.append(v)
            v += step
    return values


@dataclass
class _RangeValue:
    start: float
    step: Optional[float]
    end: float


class Transform:
    __slots__ = ("rotation", "translation")

    def __init__(self, rotation: Tuple[Vector3, Vector3, Vector3], translation: Vector3) -> None:
        self.rotation = rotation
        self.translation = translation

    @staticmethod
    def identity() -> "Transform":
        return Transform(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0))

    def apply_direction(self, v: Vector3) -> Vector3:
        r = self.rotation
        return (
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        )

    def translated(self, delta: Vector3) -> "Transform":
        d = self.apply_direction(delta)
        t = self.translation
        return Transform(self.rotation, (t[0] + d[0], t[1] + d[1], t[2] + d[2]))

    def rotated(self, degrees: Vector3) -> "Transform":
        rx, ry, rz = (_deg_to_rad(a) for a in degrees)
        import math

        Rx = ((1, 0, 0), (0, math.cos(rx), -math.sin(rx)), (0, math.sin(rx), math.cos(rx)))
        Ry = ((math.cos(ry), 0, math.sin(ry)), (0, 1, 0), (-math.sin(ry), 0, math.cos(ry)))
        Rz = ((math.cos(rz), -math.sin(rz), 0), (math.sin(rz), math.cos(rz), 0), (0, 0, 1))

        def matmul(a, b):
            return tuple(
                tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                for i in range(3)
            )

        child_rotation = matmul(matmul(Rz, Ry), Rx)
        new_rotation = matmul(self.rotation, child_rotation)
        return Transform(new_rotation, self.translation)


def _deg_to_rad(deg: float) -> float:
    import math
    return deg * math.pi / 180.0


@dataclass
class _KnownWrapperGeometry:
    """A user-defined module we recognize well enough to short-circuit
    (not needed today — reserved for parity with py_source_ingest's
    wrapper-template approach; currently unused)."""


class ScadInterpreter:
    def __init__(self) -> None:
        # name -> (module_def, closure_env) — nested modules (e.g. a module
        # defined inside another module's body) lexically capture the
        # enclosing scope's variables, per OpenSCAD semantics.
        self.modules: Dict[str, Tuple[ModuleDef, Env]] = {}
        self.primitives: List[SurfacePrimitive] = []
        self._next_face_id = 1

    # -- expression evaluation -------------------------------------------------

    def eval_expr(self, node: Any, env: Env) -> Any:
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Bool):
            return node.value
        if isinstance(node, Undef):
            return UNDEF
        if isinstance(node, VarRef):
            if node.name not in env:
                raise ScadUnsupported(f"unresolved identifier '{node.name}'")
            return env[node.name]
        if isinstance(node, ListLit):
            return [self.eval_expr(i, env) for i in node.items]
        if isinstance(node, RangeLit):
            start = self.eval_expr(node.start, env)
            step = self.eval_expr(node.step, env) if node.step is not None else None
            end = self.eval_expr(node.end, env)
            return _RangeValue(start, step, end)
        if isinstance(node, Index):
            target = self.eval_expr(node.target, env)
            idx = self.eval_expr(node.index, env)
            if not isinstance(target, list):
                raise ScadUnsupported("indexing a non-list value")
            return target[int(idx)]
        if isinstance(node, UnaryOp):
            val = self.eval_expr(node.operand, env)
            if node.op == "-":
                return -val
            if node.op == "!":
                return not is_truthy(val)
            raise ScadUnsupported(f"unsupported unary operator {node.op}")
        if isinstance(node, BinOp):
            return self._eval_binop(node, env)
        if isinstance(node, Ternary):
            cond = self.eval_expr(node.cond, env)
            return self.eval_expr(node.then if is_truthy(cond) else node.otherwise, env)
        if isinstance(node, FuncCall):
            return self._eval_func(node, env)
        if isinstance(node, ForEachComp):
            iterable = self.eval_expr(node.iterable, env)
            if isinstance(iterable, _RangeValue):
                iterable = _expand_range(iterable)
            if not isinstance(iterable, list):
                raise ScadUnsupported("for-each comprehension over a non-list value")
            results = []
            for item in iterable:
                child = dict(env)
                child[node.var] = item
                results.append(self.eval_expr(node.body, child))
            return results
        if isinstance(node, ForCComp):
            child = dict(env)
            for name, expr in node.inits:
                child[name] = self.eval_expr(expr, child)
            results = []
            while is_truthy(self.eval_expr(node.cond, child)):
                results.append(self.eval_expr(node.body, child))
                for name, expr in node.updates:
                    child[name] = self.eval_expr(expr, child)
            return results
        raise ScadUnsupported(f"unsupported expression node {type(node).__name__}")

    def _eval_binop(self, node: BinOp, env: Env) -> Any:
        if node.op == "&&":
            return is_truthy(self.eval_expr(node.left, env)) and is_truthy(self.eval_expr(node.right, env))
        if node.op == "||":
            return is_truthy(self.eval_expr(node.left, env)) or is_truthy(self.eval_expr(node.right, env))
        left = self.eval_expr(node.left, env)
        right = self.eval_expr(node.right, env)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        if node.op == "%":
            return left % right
        if node.op == "<":
            return left < right
        if node.op == ">":
            return left > right
        if node.op == "<=":
            return left <= right
        if node.op == ">=":
            return left >= right
        if node.op == "==":
            return left == right
        if node.op == "!=":
            return left != right
        raise ScadUnsupported(f"unsupported binary operator {node.op}")

    def _eval_func(self, node: FuncCall, env: Env) -> Any:
        if node.name == "len":
            val = self.eval_expr(node.args[0], env)
            if isinstance(val, _RangeValue):
                val = _expand_range(val)
            return len(val)
        if node.name == "sum":
            val = self.eval_expr(node.args[0], env)
            return sum(val)
        if node.name == "is_undef":
            val = self.eval_expr(node.args[0], env)
            return val is UNDEF
        raise ScadUnsupported(f"unsupported function call '{node.name}'")

    # -- statement execution -----------------------------------------------

    def exec_stmts(self, stmts: List[Any], env: Env, transform: Transform) -> None:
        for stmt in stmts:
            self.exec_stmt(stmt, env, transform)

    def exec_stmt(self, stmt: Any, env: Env, transform: Transform) -> None:
        if isinstance(stmt, ModuleDef):
            self.modules[stmt.name] = (stmt, dict(env))
            return
        if isinstance(stmt, Assign):
            env[stmt.name] = self.eval_expr(stmt.expr, env)
            return
        if isinstance(stmt, If):
            if is_truthy(self.eval_expr(stmt.cond, env)):
                self.exec_stmts(stmt.then_block, dict(env), transform)
            else:
                self.exec_stmts(stmt.else_block, dict(env), transform)
            return
        if isinstance(stmt, ForStmt):
            iterable = self.eval_expr(stmt.iterable, env)
            if isinstance(iterable, _RangeValue):
                iterable = _expand_range(iterable)
            if not isinstance(iterable, list):
                raise ScadUnsupported("statement-level for over a non-list value")
            for item in iterable:
                child = dict(env)
                child[stmt.var] = item
                self.exec_stmts(stmt.body, child, transform)
            return
        if isinstance(stmt, CallStmt):
            self._exec_call(stmt, env, transform)
            return
        raise ScadUnsupported(f"unsupported statement node {type(stmt).__name__}")

    def _exec_call(self, stmt: CallStmt, env: Env, transform: Transform) -> None:
        name = stmt.name

        if name == "__noop__":
            return

        if name == "translate":
            delta = self._eval_vec3(stmt.args[0][1], env)
            child_transform = transform.translated(delta)
            self.exec_stmts(stmt.block or [], dict(env), child_transform)
            return

        if name == "rotate":
            degrees = self._eval_vec3(stmt.args[0][1], env)
            child_transform = transform.rotated(degrees)
            self.exec_stmts(stmt.block or [], dict(env), child_transform)
            return

        if name in _TRANSLATE_SHORTHANDS:
            amount = self.eval_expr(stmt.args[0][1], env)
            axis = _TRANSLATE_SHORTHANDS[name]
            delta = (axis[0] * amount, axis[1] * amount, axis[2] * amount)
            child_transform = transform.translated(delta)
            self.exec_stmts(stmt.block or [], dict(env), child_transform)
            return

        if name in ("union", "difference", "intersection"):
            # No real boolean geometry — children (including a difference's
            # subtracted tool) all contribute primitives, per design.md
            # decision 3's stated non-goal (no CAD kernel).
            self.exec_stmts(stmt.block or [], dict(env), transform)
            return

        if name in ("cyl", "cylinder"):
            self._emit_cylinder(stmt.args, env, transform)
            return

        if name in ("cube", "cuboid"):
            self._emit_planar(stmt.args, env, transform)
            return

        if name in self.modules:
            module, closure_env = self.modules[name]
            child_env = self._bind_params(module.params, stmt.args, env, closure_env)
            self.exec_stmts(module.body, child_env, transform)
            return

        raise ScadUnsupported(f"unsupported/unrecognized call '{name}'")

    def _bind_params(
        self,
        params: List[Tuple[str, Optional[Any]]],
        args: List[Tuple[Optional[str], Any]],
        caller_env: Env,
        closure_env: Optional[Env] = None,
    ) -> Env:
        child_env: Env = dict(closure_env) if closure_env else {}
        positional = [a for a in args if a[0] is None]
        keyword = {a[0]: a[1] for a in args if a[0] is not None}

        for i, (pname, default) in enumerate(params):
            if pname in keyword:
                child_env[pname] = self.eval_expr(keyword[pname], caller_env)
            elif i < len(positional):
                child_env[pname] = self.eval_expr(positional[i][1], caller_env)
            elif default is not None:
                child_env[pname] = self.eval_expr(default, child_env)
            else:
                child_env[pname] = UNDEF
        return child_env

    def _eval_vec3(self, node: Any, env: Env) -> Vector3:
        val = self.eval_expr(node, env)
        if not isinstance(val, list) or len(val) != 3:
            raise ScadUnsupported("expected a 3-element vector")
        return (float(val[0]), float(val[1]), float(val[2]))

    def _lookup_arg(self, args: List[Tuple[Optional[str], Any]], keys: Tuple[str, ...], env: Env) -> Optional[float]:
        by_name = {a[0]: a[1] for a in args if a[0] is not None}
        for key in keys:
            if key in by_name:
                return float(self.eval_expr(by_name[key], env))
        return None

    def _emit_cylinder(self, args: List[Tuple[Optional[str], Any]], env: Env, transform: Transform) -> None:
        diameter = self._lookup_arg(args, _CYL_DIAMETER_KEYS, env)
        radius = self._lookup_arg(args, _CYL_RADIUS_KEYS, env)
        if radius is None and diameter is not None:
            radius = diameter / 2.0
        if radius is None:
            raise ScadUnsupported("cyl/cylinder call without a resolvable d= or r=")

        axis_direction = transform.apply_direction((0.0, 0.0, 1.0))
        axis = Axis(direction=axis_direction, point=transform.translation)

        details: Dict[str, float] = {"radius": radius}
        details.update(axis_to_details(axis))
        self.primitives.append(SurfacePrimitive(
            face_id=self._next_face_id, type="cylindrical", details=details,
        ))
        self._next_face_id += 1

    def _emit_planar(self, args: List[Tuple[Optional[str], Any]], env: Env, transform: Transform) -> None:
        # cube/cuboid's dominant face normal, per design.md: the local Z
        # axis (thickness direction) transformed by the accumulated rotation.
        normal = transform.apply_direction((0.0, 0.0, 1.0))
        axis = Axis(direction=normal, point=transform.translation)
        details: Dict[str, float] = {}
        details.update(axis_to_details(axis))
        self.primitives.append(SurfacePrimitive(
            face_id=self._next_face_id, type="planar", details=details,
        ))
        self._next_face_id += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ScadIngestResult:
    primitives: List[SurfacePrimitive]
    principal_axis: Optional[Axis]


_INCLUDE_RE = re.compile(r"(include|use)\s*<[^>]*>\s*;?")


def ingest_source(source: str) -> ScadIngestResult:
    """Parse `.scad` source text and return its extracted primitives plus a
    reduced principal axis (or None when the cylindrical primitives don't
    all share one axis)."""
    # `include <path/with.dots>;` / `use <...>;` use a raw <...> syntax that
    # isn't valid within the rest of this module's token grammar — strip it
    # before tokenizing, same treatment as comments.
    source = _INCLUDE_RE.sub("", source)
    tokens = tokenize(source)
    program = Parser(tokens).parse_program()

    interpreter = ScadInterpreter()
    env: Env = {}
    interpreter.exec_stmts(program, env, Transform.identity())

    from .geometry import reduce_principal_axis

    principal_axis = reduce_principal_axis(interpreter.primitives)
    return ScadIngestResult(primitives=interpreter.primitives, principal_axis=principal_axis)


def ingest_file(path: Union[str, Path]) -> ScadIngestResult:
    source = Path(path).read_text(encoding="utf-8")
    return ingest_source(source)


_EXCLUDED_DIR_NAMES = {"lib", "docs", "tests"}


def discover_scad_files(root: Union[str, Path]) -> List[Path]:
    """Find `.scad` part files under `root`, excluding shared/support
    directories (`lib/`, `docs/`, `tests/`)."""
    root = Path(root)
    results: List[Path] = []
    for path in sorted(root.rglob("*.scad")):
        if any(part in _EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts[:-1]):
            continue
        results.append(path)
    return results


@dataclass
class ScadDiscoveryEntry:
    path: Path
    result: Optional[ScadIngestResult]
    error: Optional[str]


def ingest_directory(root: Union[str, Path]) -> List[ScadDiscoveryEntry]:
    """Ingest every discovered `.scad` part file under `root`. A file that
    fails to parse is reported with an error reason rather than raising."""
    entries: List[ScadDiscoveryEntry] = []
    for path in discover_scad_files(root):
        try:
            result = ingest_file(path)
            entries.append(ScadDiscoveryEntry(path=path, result=result, error=None))
        except ScadUnsupported as exc:
            entries.append(ScadDiscoveryEntry(path=path, result=None, error=str(exc)))
    return entries
