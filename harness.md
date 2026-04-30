# Benchmark Harness

The benchmark harness evaluates LLM puzzle-solving across **model x strategy x puzzle** combinations. It handles execution, retries, token tracking, caching, and result persistence.

Built on [pydantic-ai](https://ai.pydantic.dev/) for agent construction. Located in `ppbench/benchmarks/`.

Core components:

- **`run()`** — orchestrator that builds the task matrix and runs everything concurrently
- **`run_strategy()`** — execution engine that runs a single strategy against a single puzzle
- **`Strategy`** — abstract base class defining what the agent does
- **`StorageManager`** — handles caching, persistence, and result storage

---

## Quick Start

### Python API

```python
import asyncio
from ppbench.benchmarks import run, DirectAskStrategy, BasicAgenticSolve

results = asyncio.run(run(
    models=["anthropic/claude-sonnet-4-6"],
    strategies=[DirectAskStrategy, BasicAgenticSolve],
    dataset="golden_30",
    n_puzzles=5,
))
```

### CLI

```bash
# Default: auto-detects all providers with API keys, runs ALL in parallel
uv run python -u -m ppbench.benchmarks.run_benchmark

# Multiple models explicitly
uv run python -u -m ppbench.benchmarks.run_benchmark --model openai/gpt-4o anthropic/claude-sonnet-4-6 glm/glm-4-plus

# Single model
uv run python -u -m ppbench.benchmarks.run_benchmark --model openai/gpt-4o --dataset golden_30

# Direct ask only, 10 puzzles
uv run python -u -m ppbench.benchmarks.run_benchmark --model openai/gpt-4o --strategy direct_ask --n-puzzles 10

# Custom dataset (registered in dataset.py)
uv run python -u -m ppbench.benchmarks.run_benchmark --model anthropic/claude-sonnet-4-6 --dataset my_dataset

# Specific puzzles by ID
uv run python -u -m ppbench.benchmarks.run_benchmark --model openai/gpt-4o --puzzles tapa_p0n0zzzx lits_00004030

# Local model
uv run python -u -m ppbench.benchmarks.run_benchmark --model local/qwen3.5-35b --strategy direct_ask

# Filter by puzzle type
uv run python -u -m ppbench.benchmarks.run_benchmark --model openai/gpt-4o --puzzle-types tapa slither

uv run python -u -m ppbench.benchmarks.run_benchmark --model bedrock/zhipu.glm-4 --dataset my_dataset --puzzles game1 game2 --strategy basic_agentic

uv run python -u -m ppbench.benchmarks.run_benchmark --model bedrock/zhipu.glm-4 --dataset my_dataset --dry-run

# To filter by puzzle type, use --puzzle-types:
uv run python -u -m ppbench.benchmarks.run_benchmark --model bedrock/zhipu.glm-4 --dataset my_dataset --puzzle-types pairloop --strategy basic_agentic
# Or use a specific puzzle ID from the dry-run output:
uv run python -u -m ppbench.benchmarks.run_benchmark --model bedrock/zhipu.glm-4 --dataset my_dataset --puzzles sudoku2_4g2l638p --strategy basic_agentic


# Auto-timestamped
uv run python -u -m ppbench.benchmarks.run_benchmark --model bedrock/zhipu.glm-4 --dataset my_dataset
# Custom path
uv run python -u -m ppbench.benchmarks.run_benchmark --model bedrock/zhipu.glm-4 --dataset my_dataset -o output/my_run


# 100 steps max, 3 runs per puzzle
uv run python -u -m ppbench.benchmarks.run_benchmark -m glm/glm-4-plus --max-steps 100 --runs 3
# Combine with other filters
uv run python -u -m ppbench.benchmarks.run_benchmark -m glm/glm-4-plus --dataset my_dataset --puzzle-types sudoku --max-steps 50 --runs 2 --strategy basic_agentic
```

#### CLI Arguments

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | auto-detect | Model(s) in `provider/model@variant` format. Multiple allowed. If omitted, runs all providers with API keys set. |
| `--strategy` | `-s` | `both` | `direct_ask`, `basic_agentic`, or `both` |
| `--dataset` | `-d` | `golden_30` | Dataset name (any name registered in `dataset.py`) |
| `--n-puzzles` | `-n` | all | Limit number of puzzles |
| `--puzzles` | | | Specific puzzle IDs or URLs (overrides `--dataset`) |
| `--puzzle-types` | | | Filter by puzzle type (e.g., `tapa slither`) |
| `--concurrency` | `-c` | `10` | Max concurrent tasks |
| `--output-dir` | `-o` | `output/<timestamp>` | Output directory |
| `--seed` | | | Random seed for puzzle sampling |
| `--max-steps` | | unlimited | Max model request steps (agent iterations) per puzzle |
| `--runs` | | `1` | Repeat each (model × strategy × puzzle) combo N times |
| `--dry-run` | | | Create output folder structure without making API calls |

Run `uv run python -u -m ppbench.benchmarks.run_benchmark --help` for full usage.

#### Multi-Model Auto-Detection

When `--model` is omitted, the CLI scans environment variables and runs **all providers that have API keys set**, in parallel. Default models per provider:

| Provider | Default Model | Key Required |
|----------|--------------|--------------|
| openai | `openai/gpt-4o` | `OPENAI_API_KEY` |
| anthropic | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| google | `google/gemini-2.5-flash` | `GOOGLE_API_KEY` |
| xai | `xai/grok-3-mini` | `XAI_API_KEY` |
| openrouter | `openrouter/deepseek/deepseek-chat-v3-0324` | `OPENROUTER_API_KEY` |
| glm | `glm/glm-4-plus` | `GLM_API_KEY` |
| bedrock | `bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |

To override defaults, pass `--model` explicitly with one or more models:

```bash
uv run python -u -m ppbench.benchmarks.run_benchmark --model openai/gpt-4o anthropic/claude-sonnet-4-6 glm/glm-4-plus
```

---

## The `run()` API

```python
async def run(
    models: list[str],
    strategies: list[Type[Strategy]],
    dataset: str = "golden_30",
    puzzles: list[str] | None = None,
    puzzle_types: list[str] | None = None,
    n_puzzles: int | None = None,
    concurrency: int = 10,
    output_dir: str = "output/runs",
    seed: int | None = None,
) -> list[DetailedRunResult]:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[str]` | required | Model identifiers in `provider/model@variant` format |
| `strategies` | `list[Type[Strategy]]` | required | Strategy **classes** (not instances) |
| `dataset` | `str` | `"golden_30"` | Which puzzle dataset to load |
| `puzzles` | `list[str] \| None` | `None` | Specific puzzle URLs or IDs — overrides `dataset`/`n_puzzles` |
| `puzzle_types` | `list[str] \| None` | `None` | Filter to specific types, e.g. `["tapa", "lits"]` |
| `n_puzzles` | `int \| None` | `None` | Limit puzzle count (applied after `puzzle_types` filter) |
| `concurrency` | `int` | `10` | Max concurrent tasks via `asyncio.Semaphore` |
| `output_dir` | `str` | `"output/runs"` | Where results are saved |
| `seed` | `int \| None` | `None` | Random seed for puzzle sampling (applied before `n_puzzles` slicing) |

### Execution Flow

1. **Load puzzles** — from `dataset` or resolve explicit `puzzles` (URLs or IDs like `tapa_p0n0zzzx`)
2. **Build task matrix** — every combination of model x strategy x puzzle
3. **Filter incompatible combos** — strategies with `requires_tools=True` are skipped for models that don't support tool calling
4. **Run concurrently** — all tasks run via `asyncio.gather` with a semaphore controlling concurrency
5. **Cache check** — completed runs (from prior executions) are skipped automatically
6. **Returns** `list[DetailedRunResult]`

The `puzzles` parameter accepts both full URLs (`https://puzz.link/p?sudoku/9/9/...`) and puzzle IDs (`tapa_p0n0zzzx`). IDs are resolved by scanning the specified dataset and fallback datasets.

---

## Models

Models use `provider/model-name@variant` syntax, parsed by `ppbench/benchmarks/model_list.py`.

### Supported Providers

| Provider | Example | Notes |
|----------|---------|-------|
| `openai` | `openai/gpt-4o` | Direct OpenAI API |
| `openai` | `openai/gpt-5.2@medium` | Responses API with reasoning effort |
| `anthropic` | `anthropic/claude-sonnet-4-6` | Direct Anthropic API |
| `anthropic` | `anthropic/claude-opus-4-6@thinking` | Extended thinking (budget_tokens=32000) |
| `google` | `google/gemini-3-pro` | Gemini API (auto-adds `-preview` suffix) |
| `bedrock` | `bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0` | AWS Bedrock Converse API |
| `glm` | `glm/glm-4-plus` | Zhipu AI GLM API (OpenAI-compatible) |
| `xai` | `xai/grok-4-1-fast` | xAI via OpenAI-compatible API |
| `openrouter` | `openrouter/deepseek/deepseek-v3.2` | OpenRouter via OpenAI-compatible API |
| `local` | `local/my-model` | Local server (LM Studio, ollama, vLLM) |

### Variants

| Provider | Variants | Effect |
|----------|----------|--------|
| `openai` | `@low`, `@medium`, `@high`, `@xhigh` | Reasoning effort (uses Responses API) |
| `anthropic` | `@thinking` | Extended thinking (budget_tokens=32000, max_tokens=64K or 128K for opus) |
| `anthropic` | `@1m` | 1M context window (beta header) |
| `anthropic` | `@low`, `@medium`, `@high`, `@max` | Effort level (opus models only) |
| `google` | `@minimal`, `@low`, `@medium`, `@high` | Thinking level |

### Environment Variables

API keys can be set via environment variables or a `.env` file in the project root (auto-loaded via `python-dotenv`).

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | openai |
| `ANTHROPIC_API_KEY` | anthropic |
| `GOOGLE_API_KEY` | google |
| `XAI_API_KEY` | xai |
| `OPENROUTER_API_KEY` | openrouter |
| `GLM_API_KEY` | glm (Zhipu AI) |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | bedrock (standard AWS credentials) |
| `BEDROCK_REGION` | bedrock (default: `us-east-1`, falls back to `AWS_DEFAULT_REGION`) |
| `BEDROCK_INFERENCE_PROFILE` | bedrock (optional ARN for inference profile routing) |
| `LOCAL_API_BASE` | local (default: `http://127.0.0.1:1234/v1`) |
| `LOCAL_API_KEY` | local (optional, default: `"not-needed"`) |

### Adding a New Model

Add a `_build_*` function in `ppbench/benchmarks/model_list.py` following the existing pattern. The function receives `(model_name: str, variant: str | None)` and returns a pydantic-ai model object.

---

## Strategies

A strategy defines **what** the agent does. The harness handles execution, retries, usage tracking, and caching.

### Strategy ABC

```python
class Strategy(ABC):
    requires_tools: bool = False
    cache_aliases: list[str] = []   # Legacy strategy IDs for cache compatibility

    @property
    def strategy_id(self) -> str:
        """MD5 hash of this subclass's method sources.
        Harness changes don't invalidate the cache — only strategy logic changes do."""

    @abstractmethod
    def build_agent(self, puzzle, model_obj, model_name) -> AgentConfig:
        """Create the agent, prompt, and deps. Pure setup — no execution."""

    @abstractmethod
    def extract_result(self, puzzle, deps, output) -> StrategyResult:
        """Interpret the agent's output. Replay moves on a fresh puzzle, check success."""

    def on_node(self, node, ctx) -> None:
        """Optional per-step hook. Override for compactification, progress tracking, etc."""

    def extract_logs(self, deps) -> list:
        """Optional. Return structured logs for move-by-move tracing."""
```

### AgentConfig

Returned by `build_agent()`. Everything the harness needs to run an agent.

```python
@dataclass
class AgentConfig:
    agent: Any              # pydantic-ai Agent instance
    prompt: str             # User prompt to send
    deps: Any = None        # Strategy-specific context (mutable during run)
    usage_limits: Any = None  # pydantic-ai UsageLimits
```

### StrategyResult

Returned by `extract_result()`. Domain-specific output from a completed run.

```python
@dataclass
class StrategyResult:
    is_success: bool
    parsed_moves: list[str] = field(default_factory=list)
    raw_output: str = ""
    detail_data: dict = field(default_factory=dict)
```

### Built-in Strategies

#### DirectAskStrategy (`ppbench/benchmarks/strategies/direct_ask.py`)

Single-shot, no tools. The simplest strategy.

- `requires_tools = False`
- Asks the model to solve the puzzle and return moves as a JSON array in a markdown code block
- Provides puzzle rules, example inputs, coordinate system examples, and the board state
- `extract_result()` parses JSON from the output, replays moves on a fresh puzzle, checks `isComplete()`

#### BasicAgenticSolve (`ppbench/benchmarks/strategies/basic_agentic.py`)

Tool-calling agent that iteratively solves puzzles.

- `requires_tools = True`
- Available tools:
  - `make_move(movestring)` — apply a single move, returns new board state
  - `make_multi_move(movelist)` — apply multiple moves at once
  - `check_board_for_completeness()` — check current state against puzzle rules
  - `render_board_as_svg()` — get full SVG rendering of the board
  - `get_rules()` — get puzzle rules and failure examples
  - `reset_puzzle()` — erase all moves, start fresh
  - `give_up()` — forfeit (counts as failure)
- Output validator keeps retrying until `puzzle.isComplete()` or `give_up` (up to 5000 moves)
- Uses `AgenticContext` dataclass to hold mutable state: `puzzle`, `log`, `list_of_moves`, `gave_up`
- `extract_result()` replays ALL recorded moves on a fresh puzzle to verify success
- `extract_logs()` returns move-by-move logs from `StrategyLogger`

### Writing a Custom Strategy

#### Minimal example (no tools)

```python
import json
import re
from pydantic_ai import Agent
from ppbench import Puzzle
from ppbench.benchmarks import Strategy, AgentConfig, StrategyResult

class MyDirectStrategy(Strategy):
    requires_tools = False

    def build_agent(self, puzzle, model_obj, model_name):
        agent = Agent(
            model_obj,
            system_prompt="Solve this puzzle. Return moves as a JSON array in a ```json code block.",
        )
        prompt = f"Puzzle type: {puzzle.pid}\nBoard:\n{puzzle.get_string_repr()}"
        return AgentConfig(agent=agent, prompt=prompt)

    def extract_result(self, puzzle, deps, output):
        # Parse JSON from output
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", output, re.DOTALL)
        moves = json.loads(match.group(1)) if match else []

        # Replay on fresh puzzle
        fresh = Puzzle.from_url(puzzle.url)
        for move in moves:
            fresh.send_move(move)

        return StrategyResult(
            is_success=fresh.isComplete(),
            parsed_moves=moves,
            raw_output=output,
        )
```

#### Tool-calling example

```python
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from ppbench import Puzzle
from ppbench.benchmarks import Strategy, AgentConfig, StrategyResult

@dataclass
class MyContext:
    puzzle: Puzzle
    moves: list[str] = field(default_factory=list)

class MyAgenticStrategy(Strategy):
    requires_tools = True

    def build_agent(self, puzzle, model_obj, model_name):
        agent = Agent(model_obj, deps_type=MyContext, system_prompt="Solve this puzzle using the tools.")

        @agent.tool
        async def make_move(ctx: RunContext[MyContext], move: str) -> str:
            ctx.deps.puzzle.send_move(move)
            ctx.deps.moves.append(move)
            return f"Board after move:\n{ctx.deps.puzzle.get_string_repr()}"

        @agent.output_validator
        async def check_done(ctx: RunContext[MyContext], output: str) -> str:
            if not ctx.deps.puzzle.isComplete():
                raise ModelRetry("Puzzle not solved yet. Keep going!")
            return output

        deps = MyContext(puzzle=puzzle)
        prompt = f"Puzzle type: {puzzle.pid}\nBoard:\n{puzzle.get_string_repr()}"
        return AgentConfig(agent=agent, prompt=prompt, deps=deps)

    def extract_result(self, puzzle, deps, output):
        fresh = Puzzle.from_url(puzzle.url)
        for move in deps.moves:
            fresh.send_move(move)
        return StrategyResult(
            is_success=fresh.isComplete(),
            parsed_moves=deps.moves,
            raw_output=output,
        )
```

Use your custom strategy like any built-in:

```python
results = asyncio.run(run(
    models=["openai/gpt-4o"],
    strategies=[MyDirectStrategy, MyAgenticStrategy],
    dataset="golden_30",
))
```

### Strategy ID and Caching

`strategy_id` is an MD5 hash of the subclass's own method sources (class name + all non-inherited methods). This means:

- Changing harness code (retries, storage, etc.) does **not** invalidate the cache
- Changing your strategy's `build_agent()`, `extract_result()`, or tools **does** invalidate it
- `cache_aliases` lets you list old strategy IDs that are functionally equivalent (e.g., after a harness-only refactor)

---

## The Harness (`run_strategy()`)

`run_strategy()` is the execution engine that runs a single strategy against a single puzzle.

```python
async def run_strategy(
    strategy: Strategy,
    puzzle: Any,
    model_obj: Any,
    model_name: str,
    storage: Optional[StorageManager] = None,
    max_retries: int = 3,
    request_timeout: float = 12 * 60 * 60,  # 12 hours
) -> DetailedRunResult:
```

### Execution Steps

1. Call `strategy.build_agent()` to get `AgentConfig`
2. Run agent via `agent.iter()` streaming — calls `strategy.on_node()` per step
3. For each `ModelRequestNode`, stream the response
4. On success: extract messages and output
5. Extract token usage from all `ModelResponse` messages (always, even on error)
6. Call `strategy.extract_result()` to interpret the output
7. Call `strategy.extract_logs()` for move-by-move tracing
8. Build `DetailedRunResult` and save via `StorageManager`

### Retry Logic

- **Max retries:** 3 (configurable)
- **Retryable errors:** 429 rate limits, API connection errors, HTTP timeouts, `UnexpectedModelBehavior`, remote protocol errors
- **Non-retryable:** Context too long errors (detected via string patterns: `"maximum context length"`, `"maximum prompt length"`, `"prompt is too long"`, `"too many tokens"`)
- **Backoff for 429:** Parses `retry-after` from response body, caps at 300s. Fallback: `10 * 2^attempt + jitter`, caps at 120s
- **General backoff:** `2^attempt + jitter`, caps at 60s

On non-retryable errors, the harness captures partial messages and stops immediately. On retry exhaustion, it captures the last error's traceback.

Token usage is extracted in **all** cases — success, failure, and error — so you always have cost data.

---

## Output and Storage

### Directory Structure

```
output/<timestamp>/
├── runs.jsonl                              # Index: one RunResult JSON per line
├── logs/
│   ├── all.log                             # Unified real-time log (all models)
│   ├── openai_gpt-4o.log                   # Per-model log
│   ├── glm_glm-4-plus.log                  # Per-model log
│   └── ...
├── traces/                                 # Arc-explainer style trajectory JSONL
│   └── {safe_model_name}/
│       └── {puzzle_id}.jsonl               # Trace: header → steps → summary
├── artifacts/
│   └── {strategy_id}/
│       └── {safe_model_name}/
│           └── {puzzle_id}.json            # DetailedRunResult (full artifact)
└── .live/                                  # Ephemeral live progress files (cleaned up after completion)
```

Model names are sanitized: `/` and `@` replaced with `_`.

### Real-Time Logs

Every run creates `logs/` with live event streams:

- **`all.log`** — unified log across all models, prefixed with `[model_name]`
- **`{model_name}.log`** — per-model log (e.g., `openai_gpt-4o.log`)

Log event types:

| Event | Format |
|-------|--------|
| Task start | `START {strategy} \| {puzzle_id}` |
| Task end | `{PASS\|FAIL\|ERR} {strategy} \| {puzzle_id} \| {duration} \| {reqs} reqs \| {moves} moves` |
| Cached skip | `SKIP {strategy} \| {puzzle_id} \| cached` |
| Model response | `RESPONSE {puzzle_id} \| in={input_tokens} out={output_tokens}` |
| Retry | `RETRY({attempt}) {strategy} \| {puzzle_id} \| {error} \| backoff={seconds}s` |
| Error | `ERROR {strategy} \| {puzzle_id} \| {error_message}` |
| System | `[SYSTEM] {message}` (unified log only) |

Example `all.log` output:
```
[2026-04-30 12:07:01.123] [SYSTEM] Benchmark started | models=['openai/gpt-4o', 'glm/glm-4-plus'] | strategies=['BasicAgenticSolve'] | puzzles=3
[2026-04-30 12:07:01.456] [openai/gpt-4o] START BasicAgenticSolve | sudoku2_4g2l638p
[2026-04-30 12:07:01.460] [glm/glm-4-plus] START BasicAgenticSolve | sudoku2_4g2l638p
[2026-04-30 12:07:05.789] [openai/gpt-4o] RESPONSE sudoku2_4g2l638p | in=1234 out=567
[2026-04-30 12:07:12.345] [openai/gpt-4o] PASS BasicAgenticSolve | sudoku2_4g2l638p | 10.9s | 3 reqs | 42 moves
[2026-04-30 12:07:15.678] [glm/glm-4-plus] RETRY(1) BasicAgenticSolve | sudoku2_4g2l638p | ModelHTTPError | backoff=2.5s
```

### RunResult (Index Entry)

Each line in `runs.jsonl`:

| Field | Type | Description |
|-------|------|-------------|
| `strategy_id` | `str` | Hash identifying the strategy version |
| `model_name` | `str` | Model identifier as passed to `run()` |
| `puzzle_id` | `str` | Format: `{pid}_{url_hash}` |
| `puzzle_url` | `str` | Original puzz.link URL |
| `is_success` | `bool` | Whether the puzzle was solved |
| `duration_seconds` | `float` | Wall-clock time for this run |
| `total_requests` | `int` | Number of LLM API requests |
| `parsed_moves` | `list[str]` | Moves extracted by the strategy |
| `timestamp` | `str` | UTC ISO 8601 timestamp |
| `error_type` | `str \| null` | Exception type if errored |
| `exception_traceback` | `str \| null` | Full traceback if errored |
| `run_number` | `int` | Which run (0-indexed) when using `--runs` |
| `total_steps` | `int` | Model request cycles completed |
| `max_steps` | `int \| null` | Step limit that was set (if any) |
| `final_score` | `float` | 0.0 (unsolved) or 1.0 (solved) |
| `final_score_pct` | `int` | 0 or 100 |
| `solved` | `bool` | Same as `is_success` (arc-explainer compat) |
| `cost_usd` | `float \| null` | Total cost (null if pricing unavailable) |
| `total_input_tokens` | `int` | Sum of input tokens across all requests |
| `total_output_tokens` | `int` | Sum of output tokens across all requests |
| `total_reasoning_tokens` | `int` | Sum of reasoning/thinking tokens |
| `reset_count` | `int` | Number of puzzle resets during the run |

### DetailedRunResult (Artifact)

Each `{puzzle_id}.json` file:

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `RunResult` | The index entry (same data) |
| `request_usages` | `list[TokenUsage]` | Per-request `{input_tokens, output_tokens, details}` |
| `full_history` | `list[dict]` | Serialized pydantic-ai messages (full conversation) |
| `logs` | `list[dict]` | Strategy-specific logs (e.g., move-by-move from `StrategyLogger`) |
| `detail_data` | `dict` | `raw_output` + any strategy-specific data |

### Caching Behavior

- On initialization, `StorageManager` hydrates an in-memory cache from `runs.jsonl`
- Each run key is `(strategy_id, model_name, puzzle_id)`
- Cache states: `"completed"` (is_success=True), `"failed"` (is_success=False, no error), `"error"` (has error_type), `"missing"` (not in cache)
- **Completed runs are always skipped on re-run**
- Failed and errored runs are NOT skipped — they will be re-attempted

To force re-run of a completed puzzle: delete its line from `runs.jsonl` AND the corresponding artifact file.

### Trajectory Traces (Arc-Explainer Style)

Every run produces a `.jsonl` trace file in `traces/{model}/`. Each file contains:

1. **Header line** — run metadata (model, puzzle, strategy, max_steps, timestamp)
2. **Step lines** — one per model request cycle (action, score, tokens, board state)
3. **Summary line** — final stats (total_steps, solved, tokens, elapsed, error)

Example trace file (`traces/openai_gpt-4o/sudoku_4g2l638p.jsonl`):

```jsonl
{"type": "header", "schema_version": "1.0", "run_id": "openai/gpt-4o_sudoku_4g2l638p_run0", "model": "openai/gpt-4o", "puzzle_id": "sudoku_4g2l638p", "puzzle_type": "sudoku", "run_number": 0, "max_steps": 100, "strategy_id": "abc123", "timestamp": "2026-04-30T12:07:01Z"}
{"type": "step", "step": 1, "action": "make_move({\"move\": \"R1C1=5\"})", "score": 0.0, "score_pct": 0, "done": false, "state": "NOT_FINISHED", "input_tokens": 1234, "output_tokens": 89, "board_state": "...", "moves_so_far": ["R1C1=5"]}
{"type": "step", "step": 2, "action": "make_multi_move({\"moves\": [\"R1C2=3\", \"R2C1=7\"]})", "score": 0.0, "score_pct": 0, "done": false, "state": "NOT_FINISHED", "input_tokens": 1456, "output_tokens": 102, "board_state": "...", "moves_so_far": ["R1C1=5", "R1C2=3", "R2C1=7"]}
{"type": "step", "step": 15, "action": "check_board_for_completeness()", "score": 1.0, "score_pct": 100, "done": true, "state": "WIN", "input_tokens": 2100, "output_tokens": 45, "board_state": "...", "moves_so_far": ["..."]}
{"type": "summary", "total_steps": 15, "final_score": 1.0, "final_score_pct": 100, "solved": true, "total_input_tokens": 28500, "total_output_tokens": 1200, "elapsed_seconds": 45.2, "total_moves": 81}
```

#### TraceStep Fields

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Step number (1-indexed) |
| `action` | `str` | Tool call(s) or model text (summarized) |
| `score` | `float` | 0.0 or 1.0 (binary puzzle completion) |
| `score_pct` | `int` | 0 or 100 |
| `done` | `bool` | Whether puzzle is complete at this step |
| `state` | `str` | `NOT_FINISHED` or `WIN` |
| `input_tokens` | `int` | Tokens consumed this step |
| `output_tokens` | `int` | Tokens generated this step |
| `reasoning_tokens` | `int` | Thinking/reasoning tokens this step |
| `cached_input_tokens` | `int` | Cached input tokens this step |
| `step_cost_usd` | `float \| null` | Cost for this step (if pricing available) |
| `cumulative_cost_usd` | `float \| null` | Running total cost |
| `board_state` | `str` | Puzzle board string representation |
| `moves_so_far` | `list[str]` | All moves applied up to this point |

#### Loading Traces in Python

```python
import json
from pathlib import Path

trace_file = Path("output/20260430_120701/traces/openai_gpt-4o/sudoku_4g2l638p.jsonl")
lines = [json.loads(line) for line in trace_file.read_text().strip().split("\n")]

header = lines[0]   # type == "header"
steps = [l for l in lines if l["type"] == "step"]
summary = lines[-1]  # type == "summary"

print(f"Solved: {summary['solved']} in {summary['total_steps']} steps")
print(f"Tokens: {summary['total_input_tokens']} in / {summary['total_output_tokens']} out")
```

---

## Datasets

### Built-in Datasets

| Name | Size | Location | Description |
|------|------|----------|-------------|
| `golden` / `golden_300` | 300 | `ppbench/bundled/golden_300.jsonl` | Curated benchmark: 20 types x 15 puzzles each |
| `golden_30` | 30 | `ppbench/bundled/golden_30.jsonl` | Small subset for expensive agentic strategies |
| `full` | 62,231 | `ppbench/data/full_dataset.jsonl` | All 94 puzzle types (download from HuggingFace) |

### Record Format

Each line in the JSONL file:

```json
{
  "puzzlink_url": "https://puzz.link/p?sudoku/9/9/...",
  "pid": "sudoku",
  "number_required_moves": 42,
  "solution_enc": "..."
}
```

- `puzzlink_url` — canonical puzzle URL (encodes the full puzzle state)
- `pid` — puzzle type identifier
- `number_required_moves` — minimum moves needed to solve
- `solution_enc` — XOR+base64 encoded solution (decrypted automatically by `load_dataset()`)

### Downloading the Full Dataset

```bash
pip install huggingface-hub
huggingface-cli download bluecoconut/pencil-puzzle-bench \
    full_dataset.jsonl \
    --repo-type dataset \
    --local-dir ppbench/data
```

---

## Loading Your Own Dataset

There are three approaches depending on your needs.

### Approach 1: Pass Puzzle URLs Directly

The simplest way. Pass a list of [puzz.link](https://puzz.link) URLs to the `puzzles` parameter:

```python
import asyncio
from ppbench.benchmarks import run, DirectAskStrategy

results = asyncio.run(run(
    models=["openai/gpt-4o"],
    strategies=[DirectAskStrategy],
    puzzles=[
        "https://puzz.link/p?sudoku/9/9/g3j1k4i6i2j5h8l4h5j6k1h8h2n7k2h8h5l8h6i4i1j2k3i5",
        "https://puzz.link/p?slither/5/5/dgag2i1c2b",
    ],
))
```

This bypasses the dataset system entirely. Each URL is self-contained — it encodes the full puzzle state, no server needed.

### Approach 2: Create a JSONL File and Load Manually

Create your own `.jsonl` file with one JSON object per line. Minimum required fields are `puzzlink_url` and `pid`:

```jsonl
{"puzzlink_url": "https://puzz.link/p?sudoku/9/9/...", "pid": "sudoku"}
{"puzzlink_url": "https://puzz.link/p?slither/5/5/...", "pid": "slither"}
{"puzzlink_url": "https://puzz.link/p?tapa/6/6/...", "pid": "tapa"}
```

Then load it and pass the URLs:

```python
import json
import asyncio
from ppbench.benchmarks import run, DirectAskStrategy

with open("my_puzzles.jsonl") as f:
    records = [json.loads(line) for line in f if line.strip()]

results = asyncio.run(run(
    models=["openai/gpt-4o"],
    strategies=[DirectAskStrategy],
    puzzles=[r["puzzlink_url"] for r in records],
))
```

### Approach 3: Register in `load_dataset()`

For repeated use, add your dataset to `ppbench/dataset.py`:

```python
# In ppbench/dataset.py, add to the dataset_files dict:
dataset_files = {
    "golden": "golden_300.jsonl",
    "golden_300": "golden_300.jsonl",
    "golden_30": "golden_30.jsonl",
    "my_dataset": "my_dataset.jsonl",  # <-- add this
}
```

Place `my_dataset.jsonl` in `ppbench/bundled/`, then use it like any built-in:

```python
results = asyncio.run(run(
    models=["openai/gpt-4o"],
    strategies=[DirectAskStrategy],
    dataset="my_dataset",
    n_puzzles=10,
))
```

### Where to Get Puzzle URLs

- Browse puzzles at [puzz.link](https://puzz.link) — click any puzzle and copy the URL from your browser
- The URL encodes the entire puzzle state (board size, givens, constraints) — no external server or database needed
- Puzzle types include: sudoku, slither, tapa, nurikabe, lits, masyu, heyawake, akari, and 86+ others

---

## Analyzing Results

### Load the Index

```python
import json
from pathlib import Path

RUNS_DIR = Path("output/runs")

with open(RUNS_DIR / "runs.jsonl") as f:
    runs = [json.loads(line) for line in f]

# Summary
passing = [r for r in runs if r["is_success"]]
failing = [r for r in runs if not r["is_success"] and not r.get("error_type")]
errored = [r for r in runs if r.get("error_type")]
print(f"Pass: {len(passing)}, Fail: {len(failing)}, Error: {len(errored)}")
```

### Load a Detailed Artifact

```python
def load_artifact(strategy_id, model_name, puzzle_id):
    safe_model = model_name.replace("/", "_").replace("@", "_")
    path = RUNS_DIR / "artifacts" / strategy_id / safe_model / f"{puzzle_id}.json"
    with open(path) as f:
        return json.load(f)

# Example: analyze token usage
r = passing[0]
artifact = load_artifact(r["strategy_id"], r["model_name"], r["puzzle_id"])

usages = artifact["request_usages"]
total_input = sum(u["input_tokens"] for u in usages)
total_output = sum(u["output_tokens"] for u in usages)
print(f"Tokens: {total_input:,} in + {total_output:,} out = {total_input + total_output:,} total")
print(f"Requests: {len(usages)}")
```

### Inspect Move Traces

```python
# Strategy logs (move-by-move for agentic strategies)
logs = artifact.get("logs", [])
for entry in logs:
    print(f"  [{entry['timestamp']:6.1f}s] {entry['message']}")

# Parsed moves
moves = r["parsed_moves"]
for i, move in enumerate(moves, 1):
    print(f"  {i:3d}. {move}")
```

### Full Conversation History

```python
# The full pydantic-ai message history
history = artifact["full_history"]
for msg in history:
    print(f"  [{msg.get('role', msg.get('kind', '?'))}] {str(msg.get('content', ''))[:100]}...")
```

See `examples/analyze_results.py` for a complete working example.

---

## Example Scripts

| Script | Description | Command |
|--------|-------------|---------|
| `examples/quick_test.py` | 1 puzzle (`tapa_p0n0zzzx`), both strategies, 1 model | `uv run python -u examples/quick_test.py` |
| `examples/dataset_sweep.py` | golden_30 dataset, DirectAsk only, 1 model, concurrency=10 | `uv run python -u examples/dataset_sweep.py` |
| `examples/multi_model.py` | 3 models (Gemini, GPT, Claude), both strategies, 1 puzzle | `uv run python -u examples/multi_model.py` |
| `examples/local_model_test.py` | Local model via LM Studio/ollama, DirectAsk, 1 puzzle | `uv run python -u examples/local_model_test.py` |
| `examples/analyze_results.py` | Load `runs.jsonl` + artifacts, print summary table + detailed analysis | `uv run python examples/analyze_results.py` |
