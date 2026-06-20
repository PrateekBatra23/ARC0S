"""
helper_tools.py — local grounding tools for the Agent Arena harness.

These run on YOUR machine, not the Arena's server — the Arena can only see what
ends up in `submit_task`'s content, so every tool's output must be reflected in
the final submission (exact expression + result, exact test output, what a
search actually found) for it to count as "grounded" rather than asserted.

SECURITY NOTE: run_python executes model-generated code in a subprocess with a
timeout, a stripped environment (no API keys/tokens visible to it), and
CPU/memory caps on POSIX systems. It is still NOT a hardened sandbox — it has
no network or filesystem isolation, so model-generated code can still make
outbound requests or read/write files this OS user can access. Only run this
harness on a disposable machine/VM/container, never one with sensitive data
beyond what this script itself needs. For real isolation, run the whole
harness inside a network-restricted Docker container.
"""

import ast
import asyncio
import math
import operator as op
import os
import sys
import tempfile

import httpx

try:
    import resource  # POSIX only
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

# Keys that must never be visible to model-generated code, even by accident
# (e.g. a script that prints os.environ).
_SENSITIVE_ENV_KEYS = {
    "ID_TOKEN", "GEMINI_API_KEY", "TRACELOOP_API_KEY",
    "OPENCODE_GO_API_KEY", "GOOGLE_API_KEY",
}

_CPU_SECONDS_LIMIT = 5
_MEMORY_BYTES_LIMIT = 512 * 1024 * 1024  # 512 MB
_NPROC_LIMIT = 32


def _filtered_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_KEYS}


def _limit_resources() -> None:
    """preexec_fn target — runs in the child right after fork, POSIX only."""
    if not _HAS_RESOURCE:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS_LIMIT, _CPU_SECONDS_LIMIT))
        resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_BYTES_LIMIT, _MEMORY_BYTES_LIMIT))
        resource.setrlimit(resource.RLIMIT_NPROC, (_NPROC_LIMIT, _NPROC_LIMIT))
    except (ValueError, OSError):
        pass  # best-effort — some limits are rejected under containers/CI


# ── web_search ─────────────────────────────────────────────────────────────

async def web_search(query: str) -> str:
    """Search the web for current facts, definitions, or background info.
    Use this before stating ANY fact, current event, or specific real-world
    detail you are not certain of. If nothing reliable is found, say so in your
    submission rather than presenting a guess as a verified fact."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            data = r.json()
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:5]:
            text = topic.get("Text") if isinstance(topic, dict) else None
            if text:
                parts.append(text)
        if not parts:
            return f"No instant-answer results for '{query}'. State this uncertainty explicitly if relevant."
        return "\n".join(parts[:5])
    except Exception as e:
        return f"Search failed for '{query}': {e}"


# ── calculate ─────────────────────────────────────────────────────────────

_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil,
}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


async def calculate(expression: str) -> str:
    """Evaluate a numeric math expression safely (+ - * / // % ** and
    sqrt/sin/cos/tan/log/exp/floor/ceil/abs/round/pi/e). Use for ANY arithmetic.
    Quote the exact expression and result in your submission as proof of work —
    a bare final number with no shown calculation reads as a guess."""
    try:
        tree = ast.parse(expression, mode="eval")
        return f"Result: {_eval_node(tree.body)}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ── run_python ──────────────────────────────────────────────────────────────

async def run_python(code: str, timeout_seconds: int = 10) -> str:
    """Execute Python code in a subprocess and return its real stdout/stderr/exit
    status. Use this to actually TEST a solution — run it against example
    inputs and check the output — before claiming it works. Paste the real
    output into your submission; 'I tested this and it works' with no shown
    output reads as an unverified claim, not a verified one."""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)

        kwargs = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": _filtered_env(),
        }
        if os.name == "posix":
            kwargs["preexec_fn"] = _limit_resources

        proc = await asyncio.create_subprocess_exec(sys.executable, path, **kwargs)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"TIMEOUT: execution exceeded {timeout_seconds}s and was killed."
        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        status = "OK" if proc.returncode == 0 else f"EXIT CODE {proc.returncode}"
        return f"[{status}]\nstdout:\n{out}\nstderr:\n{err}".strip()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
