"""
Dynamic prompt engine for the Agent Arena.
Provides task-type detection and a single strong composite prompt
that guides the agent to analyze, solve, and submit in one turn —
grounding every checkable claim in real tool output along the way.
"""

# ── Task type detection ──────────────────────────────────────────────────────
TASK_PATTERNS = {
    "code": [
        "code", "function", "implement", "write a", "program", "script",
        "class", "algorithm", "api", "method", "library", "module", "package",
        "build", "create a", "develop", "application", "service", "endpoint",
    ],
    "debug": [
        "debug", "fix", "error", "bug", "issue", "broken", "fails",
        "exception", "traceback", "crash", "wrong", "incorrect", "not working",
        "repair", "resolve", "troubleshoot",
    ],
    "explain": [
        "explain", "describe", "what is", "how does", "why", "difference between",
        "concept", "theory", "overview", "introduction", "compare", "contrast",
        "elaborate", "clarify", "discuss",
    ],
    "optimize": [
        "optimize", "performance", "efficient", "slow", "bottleneck", "memory",
        "speed", "complexity", "scale", "improve", "faster", "latency",
        "throughput", "resource", "cache", "compress", "reduce",
    ],
    "design": [
        "design", "architecture", "system", "database schema", "pattern",
        "structure", "model", "diagram", "plan", "blueprint", "component",
        "microservice", "flow", "sequence", "entity relationship",
    ],
    "test": [
        "test", "unit test", "pytest", "assert", "coverage", "mock", "testing",
        "tdd", "spec", "validate", "verify", "bdd", "integration test",
        "regression", "benchmark",
    ],
    "data": [
        "data", "csv", "json", "sql", "query", "database", "etl", "pipeline",
        "transform", "clean", "analyze", "visualization", "chart", "pandas",
        "dataframe", "dataset",
    ],
    "security": [
        "security", "auth", "authentication", "authorization", "jwt", "oauth",
        "encrypt", "hash", "vulnerability", "sanitize", "xss", "csrf", "sql injection",
        "penetration", "secure",
    ],
}


def detect_task_type(title: str = "", description: str = "") -> str:
    text = f"{title} {description}".lower()
    scores = {}
    for task_type, keywords in TASK_PATTERNS.items():
        scores[task_type] = sum(1 for kw in keywords if kw in text)
    if not scores or max(scores.values(), default=0) == 0:
        return "general"
    return max(scores, key=scores.get)


# ── Prompt templates ─────────────────────────────────────────────────────────

def _format_task(task: dict) -> str:
    lines = [
        f"Title: {task.get('title', 'N/A')}",
        f"Level: {task.get('level', 'N/A')}",
        f"Points: {task.get('points', 'N/A')}",
        f"Difficulty: {task.get('difficulty', 'N/A')}",
        f"Description:\n{task.get('description', 'N/A')}",
    ]
    return "\n".join(lines)


# task_type -> (solving guidance, which grounding tool is most relevant)
_TYPE_GUIDANCE = {
    "code": (
        "Write clean, well-commented code with docstrings, type hints, error handling, "
        "and a usage example. Then actually run it with run_python against at least one "
        "example input and include the real output as proof it works.",
        "run_python",
    ),
    "debug": (
        "Identify the root cause, provide the fixed code, and use run_python to actually "
        "reproduce the bug and confirm the fix resolves it — include the before/after output.",
        "run_python",
    ),
    "explain": (
        "Use clear analogies, step-by-step breakdowns, and concrete examples. If you state any "
        "specific fact, figure, or current detail, verify it with web_search first and note "
        "what was found, rather than asserting it from memory.",
        "web_search",
    ),
    "optimize": (
        "Show before/after reasoning, run_python to actually time or compare both versions "
        "where feasible, and report the real measured difference rather than an estimate.",
        "run_python",
    ),
    "design": (
        "Provide architecture, component breakdown, data flow, and technology choices with "
        "justification. Verify any cited fact (e.g. a tool's real-world limits) with web_search "
        "rather than asserting it from memory.",
        "web_search",
    ),
    "test": (
        "Provide a complete test suite, then use run_python to actually execute it and include "
        "the real pass/fail output — a test suite you haven't run is not verified.",
        "run_python",
    ),
    "data": (
        "Provide the pipeline/transformation logic and use run_python to actually run it against "
        "sample data, including the real output, not a hypothetical one.",
        "run_python",
    ),
    "security": (
        "Summarize the threat model and provide secure code. Use calculate for any numeric "
        "estimate (e.g. entropy, brute-force time) instead of approximating it.",
        "calculate",
    ),
    "general": (
        "Provide a complete, well-structured, thorough solution with examples and reasoning. "
        "Use calculate/run_python/web_search for anything checkable.",
        "any relevant tool",
    ),
}


def build_task_prompt(
    task: dict, agent_id: str, task_id: str, recent_submissions: list | None = None,
) -> str:
    """
    Single composite prompt that instructs the agent to:
      1. Analyze the task deeply
      2. Produce a complete, high-quality, GROUNDED solution
      3. Call submit_task with the full solution

    `recent_submissions` (optional): the last few {title, content} pairs this
    agent already submitted. Each task runs in its own fresh session with no
    memory of previous tasks, so this is the only way the model can fulfil a
    task that says things like "remember the X task, now do it in Y" — without
    it, such a task is unsolvable rather than just harder.
    """
    task_type = detect_task_type(task.get("title", ""), task.get("description", ""))
    guidance, primary_tool = _TYPE_GUIDANCE.get(task_type, _TYPE_GUIDANCE["general"])

    recall_block = ""
    if recent_submissions:
        entries = "\n\n".join(
            f"--- \"{s['title']}\" ---\n{s['content']}" for s in recent_submissions
        )
        recall_block = f"""

RECENT SUBMISSIONS (most recent {len(recent_submissions)} — this task may
reference one of these by name, e.g. "do the X task again in Y"; if so, use
the actual prior content below rather than guessing what it might have been):
{entries}
"""

    return f"""
You have been assigned a new task. Solve it completely in this turn.

TASK ({task_type.upper()}):
{_format_task(task)}
{recall_block}
REASONING & SOLVING INSTRUCTIONS:
1. ANALYZE — Restate the problem, extract requirements and edge cases, outline your approach.
   If this task references a previous one, use the actual content above as your starting point.
2. SOLVE — Produce a complete answer. {guidance}
3. VERIFY — Before submitting, mentally check correctness, completeness, and edge cases.

GROUNDING REQUIREMENT (this is scored — the evaluator can tell a verified answer
from a confident guess):
- Any specific number, "this works," or real-world fact in your final answer MUST be
  backed by an actual tool call this turn, and the tool's real output must be visible
  in your submitted content (the exact expression and result, the exact test/run output,
  or what the search actually returned) — not just your restated conclusion.
- For this task type, {primary_tool} is the most relevant grounding tool — use it.
- Skip a tool only for content that is genuinely not checkable (e.g. pure stylistic
  or subjective reasoning) — don't call tools that add nothing.
- If a tool returns no useful result or an error, say so plainly in your submission
  rather than filling the gap with an unverified guess.

SUBMISSION:
After your analysis, solution, and verification, call submit_task(content=<your complete,
self-contained final answer, including the tool evidence above>) in this same turn.
Do NOT stop after analysis. Do NOT ask for confirmation.
After submit_task returns, you are DONE with this task — it cannot be
submitted again. Do not call any more tools "to double check" or search
further; that wastes time and tokens for zero benefit. Stop immediately.

(Internal reference — not needed in your tools since the harness already has them:
agent_id={agent_id}, task_id={task_id})
""".strip()
