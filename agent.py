"""
agent.py — Agent Arena Multi-Turn Google ADK Agent
====================================================

A fully autonomous agent that navigates the Agent Arena:
  - Registers once and fetches each task DETERMINISTICALLY in Python
    (no LLM call spent on mechanical bookkeeping)
  - Hands the LLM a fresh, small session per task — no compounding history,
    so token/time cost doesn't grow across the run
  - The LLM's only tools are submit_task, skip_task, and grounding tools
    (web_search / calculate / run_python) — agent_id/task_id are always
    supplied by the harness, never typed by the model
  - On a missed submission: one recovery turn, then a deterministic forced
    skip — the run can never get stuck on one task
  - Transient network errors are retried with backoff before the LLM ever
    sees them; real application errors are not retried
  - Prints a running scoreboard after each task attempt

Dependencies
------------
    pip install google-adk fastmcp traceloop-sdk google-genai litellm \
                python-dotenv httpx \
                opentelemetry-api opentelemetry-sdk \
                opentelemetry-exporter-otlp-proto-http

Usage
-----
    python agent.py
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv
load_dotenv()

# ── Google ADK ────────────────────────────────────────────────────────────────
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# ── FastMCP ───────────────────────────────────────────────────────────────────
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError

# ── Traceloop ─────────────────────────────────────────────────────────────────
from traceloop.sdk import Traceloop, set_association_properties
from traceloop.sdk.decorators import workflow
from traceloop.sdk.tracing import set_conversation_id

# ── OTel logging ──────────────────────────────────────────────────────────────
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.resources import Resource

# ── Dynamic prompts ───────────────────────────────────────────────────────────
from prompts import build_task_prompt, detect_task_type

# ── Local grounding tools ─────────────────────────────────────────────────────
from helper_tools import web_search, calculate, run_python

# ── LiteLLM (optional — enables OpenCode Go and other providers) ─────────────
try:
    from google.adk.models.lite_llm import LiteLlm
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MCP_ENDPOINT = "https://agent-arena.dev/mcp"

ID_TOKEN = os.environ.get("ID_TOKEN", "")

AGENT_NAME    = "xpribot-v2"
LINKEDIN_URL  = "https://www.linkedin.com/in/xprilion"
GITHUB_URL    = "https://github.com/xprilion/agent-arena-bot"
GEMINI_MODEL   = "gemini-3-flash-preview"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TRACELOOP_API_KEY = os.environ.get("TRACELOOP_API_KEY", "")

OPENCODE_GO_API_KEY = os.environ.get("OPENCODE_GO_API_KEY", "")
OPENCODE_GO_MODEL   = os.environ.get("OPENCODE_GO_MODEL", "kimi-k2.6")
OPENCODE_GO_BASE    = "https://opencode.ai/zen/go/v1"

MAX_TASKS    = 20
MCP_RETRIES  = 3              # retry attempts for transient network errors
MCP_BACKOFF  = 0.5             # base seconds; doubles each retry (0.5, 1, 2)

APP_NAME = "arena-adk-agent"
USER_ID  = "arena-user"


def _active_model():
    if OPENCODE_GO_API_KEY and _LITELLM_AVAILABLE:
        return LiteLlm(
            model=f"openai/{OPENCODE_GO_MODEL}",
            api_base=OPENCODE_GO_BASE,
            api_key=OPENCODE_GO_API_KEY,
        )
    return GEMINI_MODEL


def _active_model_name() -> str:
    if OPENCODE_GO_API_KEY and _LITELLM_AVAILABLE:
        return f"opencode-go/{OPENCODE_GO_MODEL}"
    return GEMINI_MODEL


AGENT_STACK = f"Python / Google ADK / {_active_model_name()} / Traceloop"


def _check_credentials() -> None:
    missing = []
    if not ID_TOKEN:
        missing.append("ID_TOKEN")
    if not GEMINI_API_KEY and not (OPENCODE_GO_API_KEY and _LITELLM_AVAILABLE):
        missing.append("GEMINI_API_KEY")
    if missing:
        raise SystemExit(
            f"Missing required env var(s): {', '.join(missing)}.\n"
            "Set them in your environment or a .env file before running.\n"
            "  ID_TOKEN       — sign in to the Arena web app, DevTools -> Application -> "
            "Storage -> copy the Firebase id token (expires ~1hr, grab it right before running).\n"
            "  GEMINI_API_KEY — https://aistudio.google.com"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(tag: str, msg: str, level: str = "INFO") -> None:
    emoji = {
        "REGISTER": "📝", "FETCH": "📥", "SUBMIT": "📤",
        "SCORE": "🏆", "LEVEL": "🚀", "SKIP": "⏭️",
        "ERROR": "❌", "WARN": "⚠️", "DONE": "✅",
        "TASK": "📋", "LOOP": "🔄", "AGENT": "🤖",
        "TRACE": "📡", "RECOVER": "🔧", "RETRY": "🔁",
    }.get(tag, "•")
    print(f"[{_ts()}] {emoji} [{tag}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Run-scoped state
# ─────────────────────────────────────────────────────────────────────────────

class RunState:
    def __init__(self) -> None:
        self.run_id       = str(uuid.uuid4())
        self.execution_id = str(uuid.uuid4())
        self.agent_id     = ""
        self.task_id      = ""
        self.conversation_id = ""

        self.current_level = 1
        self.total_score   = 0
        self.tasks_attempted = 0
        self.tasks_passed    = 0
        self.level_history: list[dict] = []

        self.current_task: Optional[dict] = None

    def record(self, level: int, task_title: str, score: int, levelled_up: bool) -> None:
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up or score >= 70:
            self.tasks_passed += 1
        if levelled_up:
            self.current_level = level + 1
        self.level_history.append({
            "level": level, "task": task_title,
            "score": score, "levelled_up": levelled_up,
        })

    def scoreboard(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  SCOREBOARD  (run {self.run_id[:8]})  model: {_active_model_name()}",
            f"{'─'*60}",
            f"  Current Level : {self.current_level}",
            f"  Total Score   : {self.total_score}",
            f"  Tasks Done    : {self.tasks_attempted}  (passed: {self.tasks_passed})",
            f"{'─'*60}",
        ]
        for entry in self.level_history:
            icon = "✅" if entry["levelled_up"] else ("🟡" if entry["score"] >= 70 else "❌")
            lines.append(
                f"  {icon} L{entry['level']}  {entry['task'][:40]:<40}  {entry['score']:>3}/100"
            )
        lines.append(f"{'─'*60}\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OTel / Traceloop logging
# ─────────────────────────────────────────────────────────────────────────────

class _OtelOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        tid = getattr(record, "otelTraceID", "0")
        return tid not in ("0", "00000000000000000000000000000000", None, "")


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(h)
    return logger


agent_logger = _make_logger("arena.agent")
task_logger  = _make_logger("arena.task")


def init_tracing() -> None:
    Traceloop.init(
        app_name=APP_NAME,
        api_key=TRACELOOP_API_KEY or None,
        disable_batch=True,
        telemetry_enabled=False,
    )
    log_provider = LoggerProvider(resource=Resource.create({"service.name": APP_NAME}))
    exporter = ConsoleLogExporter()
    if TRACELOOP_API_KEY:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        exporter = OTLPLogExporter(
            endpoint="https://api.traceloop.com/v1/logs",
            headers={"Authorization": f"Bearer {TRACELOOP_API_KEY}", "x-traceloop-sdk-version": "traceloop-sdk"},
        )
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    for logger in (agent_logger, task_logger):
        h = LoggingHandler(logger_provider=log_provider)
        h.setLevel(logging.INFO)
        h.addFilter(_OtelOnlyFilter())
        logger.addHandler(h)
    _log("TRACE", "Traceloop initialised.")


# ─────────────────────────────────────────────────────────────────────────────
# MCP helper — with retry-with-backoff for transient errors only
# ─────────────────────────────────────────────────────────────────────────────

_TRANSIENT_EXCEPTIONS = (
    httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
    httpx.RemoteProtocolError, httpx.PoolTimeout, ConnectionError, OSError,
)


async def _mcp_call(tool_name: str, arguments: dict, state: "RunState") -> str:
    transport = StreamableHttpTransport(url=MCP_ENDPOINT)

    for attempt in range(1, MCP_RETRIES + 1):
        try:
            async with Client(transport=transport, name="arena-adk-agent") as client:
                set_association_properties({
                    "execution.id": state.execution_id,
                    "run.id":       state.run_id,
                    "agent.id":     state.agent_id,
                    "task.id":      state.task_id,
                    "agent.name":   AGENT_NAME,
                    "agent.stack":  AGENT_STACK,
                })
                if state.conversation_id:
                    set_conversation_id(state.conversation_id)

                result = await client.call_tool(tool_name, arguments)
                if result is None:
                    return f"ERROR: {tool_name} returned no response"
                return "\n".join(
                    getattr(b, "text", "") for b in result.content if getattr(b, "text", None)
                )

        except ToolError as e:
            # Real application-level error — don't retry, surface immediately.
            _log("ERROR", f"{tool_name}: {e}")
            return f"ERROR: {e}"

        except _TRANSIENT_EXCEPTIONS as e:
            if attempt < MCP_RETRIES:
                wait = MCP_BACKOFF * (2 ** (attempt - 1))
                _log("RETRY", f"{tool_name} transient error (attempt {attempt}/{MCP_RETRIES}): {e} — retrying in {wait}s")
                await asyncio.sleep(wait)
                continue
            _log("ERROR", f"{tool_name}: exhausted {MCP_RETRIES} retries — {e}")
            return f"ERROR: {tool_name} failed after {MCP_RETRIES} attempts: {e}"

        except Exception as e:
            _log("ERROR", f"{tool_name}: {e}")
            return f"ERROR: {e}"

    return f"ERROR: {tool_name} failed after {MCP_RETRIES} attempts"


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic harness calls — NOT exposed to the LLM.
# Registration and task-fetching need no judgment, so they cost zero LLM
# tokens/latency and can never be mistyped by the model.
# ─────────────────────────────────────────────────────────────────────────────

async def register_agent_call(state: RunState) -> str:
    result = await _mcp_call("register_agent", {
        "idToken":     ID_TOKEN,
        "name":        AGENT_NAME,
        "stack":       AGENT_STACK,
        "linkedinUrl": LINKEDIN_URL,
        "githubUrl":   GITHUB_URL,
    }, state)

    match = re.search(r"AGENT_ID:\s*(\S+?)\.?(\s|$)", result)
    if match:
        state.agent_id = match.group(1)
        state.conversation_id = state.agent_id
        set_association_properties({"agent.id": state.agent_id, "run.id": state.run_id})
        set_conversation_id(state.agent_id)

    level_match = re.search(r"Level[:\s]+(\d+)", result)
    if level_match:
        state.current_level = int(level_match.group(1))

    agent_logger.info("Registered", extra={"agent_id": state.agent_id, "run_id": state.run_id})
    _log("REGISTER", f"agent_id={state.agent_id}  level={state.current_level}")
    return result


async def get_tasks_call(state: RunState) -> str:
    result = await _mcp_call("get_tasks", {
        "idToken": ID_TOKEN, "agentId": state.agent_id,
    }, state)

    state.current_task = None
    try:
        data = json.loads(result)
        task_obj = None
        if isinstance(data, dict) and "id" in data:
            task_obj = data
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "id" in data[0]:
            task_obj = data[0]

        if task_obj:
            state.task_id         = task_obj["id"]
            state.current_task    = task_obj
            state.conversation_id = f"{state.agent_id}-{state.task_id}"
            set_association_properties({"task.id": state.task_id, "execution.id": state.execution_id})
            set_conversation_id(state.conversation_id)
            _log("FETCH", f"task={state.task_id}  '{task_obj.get('title')}'  L{task_obj.get('level')}")
    except json.JSONDecodeError:
        pass

    return result


async def skip_task_call(state: RunState, reason: str) -> str:
    _log("SKIP", f"skipping {state.task_id[:8] if state.task_id else '?'}  reason={reason[:50]}")
    return await _mcp_call("skip_task", {
        "idToken": ID_TOKEN, "agentId": state.agent_id,
        "taskId": state.task_id, "reason": reason,
    }, state)


async def submit_task_call(state: RunState, content: str) -> str:
    new_exec = str(uuid.uuid4())
    state.execution_id = new_exec
    set_association_properties({
        "execution.id": new_exec, "task.id": state.task_id, "agent.id": state.agent_id,
    })

    task_logger.info("Submitting", extra={
        "agent_id": state.agent_id, "task_id": state.task_id, "execution_id": new_exec,
    })

    result = await _mcp_call("submit_task", {
        "idToken":     ID_TOKEN,
        "agentId":     state.agent_id,
        "taskId":      state.task_id,
        "executionId": new_exec,
        "content":     content,
        "metadata": {
            "agent_name": AGENT_NAME, "agent_stack": AGENT_STACK,
            "run_id": state.run_id, "execution_id": new_exec, "model": _active_model_name(),
        },
    }, state)

    score_match = re.search(r"Score:\s*(\d+)/100", result, re.IGNORECASE)
    score       = int(score_match.group(1)) if score_match else -1
    levelled_up = "LEVEL_UP" in result.upper()

    task_title = state.current_task.get("title", state.task_id) if state.current_task else state.task_id
    state.record(state.current_level, task_title, score, levelled_up)

    lu_emoji = "🚀 LEVEL_UP!" if levelled_up else ""
    _log("SCORE", f"{score}/100  {lu_emoji}")
    print(state.scoreboard())

    task_logger.info("Submitted", extra={
        "agent_id": state.agent_id, "task_id": state.task_id,
        "score": score, "levelled_up": levelled_up,
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LLM-exposed tools — only the steps that need judgment.
# agent_id / task_id are NEVER LLM-supplied — the harness always uses the
# authoritative values from state, so the model can't pass a stale/wrong one.
# ─────────────────────────────────────────────────────────────────────────────

def make_llm_tools(state: RunState) -> list:

    async def submit_task(content: str) -> str:
        """Submit your complete, grounded final answer for the CURRENT task for AI
        evaluation. Scored 0-100; score >= 70 means LEVEL_UP. Can only be called
        once per task — don't call this until your answer is complete, verified,
        and every checkable claim is backed by visible tool output."""
        return await submit_task_call(state, content)

    async def skip_task(reason: str = "") -> str:
        """Abandon the CURRENT task without penalty — use only when it's
        genuinely impossible or already submitted. A fresh task follows."""
        return await skip_task_call(state, reason)

    return [submit_task, skip_task, web_search, calculate, run_python]


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are an expert autonomous agent competing in the Agent Arena evaluation system.
The current task's full details are always given to you directly in the user
message — you never need to fetch anything yourself.

AVAILABLE TOOLS:
- submit_task(content): Submit your final grounded answer for the current task.
- skip_task(reason): Skip an impossible/already-submitted task.
- web_search(query), calculate(expression), run_python(code): grounding tools —
  use them for any claim, number, or "this works" assertion you want credited
  as verified rather than guessed. Show their real output in your submission.

RULES:
- Never call submit_task twice for the same task.
- Do not ask for confirmation — act autonomously.
- When instructed to analyze, solve, and submit, do all three in this turn.
- Don't call a grounding tool for content that has nothing to verify — only use
  what actually helps.

IDENTITY: Agent Name: {AGENT_NAME} | Stack: {AGENT_STACK}
""".strip()


def build_agent(state: RunState) -> LlmAgent:
    return LlmAgent(
        name="arena_agent",
        model=_active_model(),
        instruction=SYSTEM_PROMPT,
        tools=make_llm_tools(state),
        generate_content_config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-turn runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_turn(
    runner:          Runner,
    session_service: InMemorySessionService,
    session_id:      str,
    message:         str,
) -> str:
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])

    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                args_str = str(dict(fc.args))
                preview  = args_str[:120]
                _log("AGENT", f"→ {fc.name}  {preview}{'...' if len(args_str) > 120 else ''}")

            elif getattr(part, "function_response", None):
                fr = part.function_response
                resp_str = str(fr.response)[:150].replace("\n", " ")
                _log("AGENT", f"← {fr.name}  {resp_str}{'...' if len(str(fr.response)) > 150 else ''}")

        if event.is_final_response() and event.content.parts:
            text = event.content.parts[0].text
            if text:
                final_text = text

    return final_text


# ─────────────────────────────────────────────────────────────────────────────
# Main workflow
# ─────────────────────────────────────────────────────────────────────────────

@workflow(name="arena_adk_run")
async def run() -> None:
    state = RunState()

    print(f"\n{'═'*60}")
    print(f"  AGENT ARENA  —  {_active_model_name()}")
    print(f"{'═'*60}")
    _log("REGISTER", f"Agent: {AGENT_NAME}")
    _log("REGISTER", f"Run ID: {state.run_id}")
    _log("REGISTER", f"Max tasks: {MAX_TASKS}")
    print(f"{'═'*60}\n")

    set_association_properties({
        "run.id":       state.run_id,
        "execution.id": state.execution_id,
        "agent.name":   AGENT_NAME,
        "agent.stack":  AGENT_STACK,
    })

    session_service = InMemorySessionService()
    agent  = build_agent(state)
    runner = Runner(agent=agent, session_service=session_service, app_name=APP_NAME)

    # ── Bootstrap: register deterministically — zero LLM calls ────────────────
    _log("REGISTER", "Registering...")
    await register_agent_call(state)
    if not state.agent_id:
        _log("ERROR", "Registration failed — no AGENT_ID returned. Aborting.")
        return

    # ── Main task loop ────────────────────────────────────────────────────────
    for task_num in range(1, MAX_TASKS + 1):
        _log("LOOP", f"Fetching task #{task_num}...")
        await get_tasks_call(state)

        if not state.current_task or not state.task_id:
            _log("DONE", "No more tasks available.")
            break

        task = state.current_task
        task_title = task.get("title", "Unknown")
        task_type  = detect_task_type(task_title, task.get("description", ""))
        desc       = task.get("description", "")[:600]

        print(f"\n{'━'*60}")
        _log("TASK", f"#{task_num} | {task_title}")
        _log("TASK", f"Type: {task_type.upper()} | Level: {task.get('level', '?')} | ID: {state.task_id[:8]}")
        _log("TASK", f"Desc: {desc}{'...' if len(task.get('description', '')) > 600 else ''}")
        print(f"{'━'*60}")

        # Fresh, small session per task — no compounding history across tasks.
        session_id = f"{state.run_id}-task-{task_num}"
        await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

        prompt = build_task_prompt(task, state.agent_id, state.task_id)
        prev_attempted = state.tasks_attempted

        _log("AGENT", "Solving (analysis + grounding + solution + submit in one turn)...")
        await run_turn(runner, session_service, session_id, prompt)

        if state.tasks_attempted > prev_attempted:
            _log("SCORE", f"Task #{task_num} submitted successfully.")
        else:
            _log("WARN", f"Task #{task_num} was NOT submitted. Recovering...")
            await run_turn(
                runner, session_service, session_id,
                "You have NOT submitted yet. Call submit_task(content=<your complete, "
                "grounded final answer>) NOW, or skip_task(reason=...) if the task is "
                "genuinely impossible to solve.",
            )
            if state.tasks_attempted == prev_attempted:
                _log("ERROR", f"Recovery failed for task #{task_num}. Forcing a deterministic skip.")
                await skip_task_call(state, reason="Agent failed to submit after recovery prompt.")

        state.current_task = None
        state.task_id = ""

    # ── Final report — pure Python, no LLM call needed ─────────────────────────
    print(f"\n{'═'*60}")
    _log("DONE", "Run complete.")
    print(f"{'═'*60}")
    print(state.scoreboard())
    agent_logger.info("Run complete", extra={
        "run_id":          state.run_id,
        "total_score":     state.total_score,
        "tasks_attempted": state.tasks_attempted,
        "final_level":     state.current_level,
    })


if __name__ == "__main__":
    _check_credentials()
    init_tracing()
    asyncio.run(run())
