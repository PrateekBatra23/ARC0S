# ARC0S — Agent Arena Competitor

A fully autonomous agent harness built on **Google ADK** to compete in
**Agent Arena** - the live, gamified tournament that capped off **Agent
Dev-Sprint: Build, Deploy, and Battle**, hosted by Amadeus Software Labs
India in Bengaluru. ARC0S registers itself, pulls tasks one at a time, 
solves each with an LLM, and
submits - with deterministic bookkeeping, fresh per-task sessions, and
built-in recovery so the run never gets permanently stuck on one task.

## How it works

- **Deterministic bookkeeping, zero LLM cost.** Registration and task-fetching
  are plain Python calls to the Arena's MCP tools — no LLM call is spent on
  mechanical steps the model has no judgment to add to.
- **Fresh session per task.** Each task gets its own small ADK session with no
  compounding history, so token and time cost don't grow across a run. A
  bounded recall log (the last 3 submissions, content capped at 600 chars) is
  injected into each new task's prompt, so a task that references "the
  previous task" still has real material to work from.
- **A narrow tool surface for the LLM.** The model only ever sees
  `submit_task`, `skip_task`, and three grounding tools — `web_search`,
  `calculate`, `run_python`. `agent_id` and `task_id` are always supplied by
  the harness from authoritative state, never typed by the model, so they
  can't be mistyped or hallucinated.
- **Hard stop after submission.** The instant `submit_task` returns a result,
  the harness closes the turn immediately — the model cannot keep calling
  tools after the task is already done (each task can only be submitted
  once, so anything after that point is wasted time and tokens).
- **One recovery attempt, then a forced skip.** If a turn ends without a
  submission, the harness sends one explicit "submit or skip now" prompt. If
  that still doesn't produce a submission, it deterministically calls
  `skip_task` itself — the run can never hang on a single task.
- **Error-aware retries.**
  - MCP network calls (register/fetch/submit/skip) retry with backoff only on
    transient errors (connection/timeout issues) — real application errors
    are surfaced immediately, not retried.
  - LLM calls that fail with a rate-limit/quota signal retry with backoff
    (currently 6 attempts, 20s base doubling each time).
  - LLM calls that fail with an auth-type signal (invalid key, permission
    denied) are treated as fatal — the whole run aborts cleanly with a clear
    message instead of silently force-skipping every remaining task one by
    one.
- **Live scoreboard.** After every submission the harness parses the Arena's
  JSON response (`{"status":"EVALUATED","score":N,...}`) and prints a running
  scoreboard — level, total score, tasks attempted/passed.
- **Full tracing.** Every register/fetch/submit/skip call and every LLM turn
  is wrapped in Traceloop/OpenTelemetry spans, tagged with run ID, agent ID,
  task ID, and execution ID.

## Model providers

ARC0S picks a model provider at startup, in priority order:

1. **OpenRouter**, if `OPENROUTER_API_KEY` is set — routed through
   [LiteLLM](https://docs.litellm.ai/docs/providers). Defaults to Gemini 3
   Flash Preview (`google/gemini-3-flash-preview`), billed through OpenRouter
   instead of Google AI Studio — a separate quota pool from direct Gemini.
   Override the model with `OPENROUTER_MODEL` in `.env` (e.g. a different
   model from [openrouter.ai/models](https://openrouter.ai/models)).
2. **OpenCode Go** (`kimi-k2.6` by default, override with
   `OPENCODE_GO_MODEL`), if `OPENCODE_GO_API_KEY` is set.
3. **Direct Gemini** (`gemini-3-flash-preview`) via `GEMINI_API_KEY`, as the
   fallback if neither of the above is configured.

## Setup

```bash
pip install google-adk fastmcp traceloop-sdk google-genai litellm \
            python-dotenv httpx \
            opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-http
```

Create a `.env` file in the project root:

```env
# Required
ID_TOKEN=...              # Firebase ID token from agent-arena.dev — sign in,
                           # then DevTools -> Application -> Storage. Expires
                           # ~1hr, so grab it right before running.

# At least one of the following is required
GEMINI_API_KEY=...        # https://aistudio.google.com
OPENROUTER_API_KEY=...    # https://openrouter.ai/keys
OPENCODE_GO_API_KEY=...

# Optional
TRACELOOP_API_KEY=...     # Arena-issued tracing key
OPENCODE_GO_MODEL=...     # defaults to kimi-k2.6
OPENROUTER_MODEL=...      # defaults to google/gemini-3-flash-preview
```

## Usage

```bash
python arc0s.py
```

The harness will register, then loop through tasks (up to 20 per run) until
the Arena reports no more tasks are available or a fatal error aborts the
run. A full scoreboard prints after the run completes.

## Project structure

| File | Responsibility |
|---|---|
| `arc0s.py` | Main harness — registration, task loop, error handling, scoring |
| `prompts.py` | Task-type detection and per-task prompt construction |
| `helper_tools.py` | Grounding tools exposed to the LLM: `web_search`, `calculate`, `run_python` |

## Identity

- **Agent name:** ARC0S
- **Stack:** Python / Google ADK / *(active model)* / Traceloop
- **LinkedIn:** https://www.linkedin.com/in/prateekbatradel/