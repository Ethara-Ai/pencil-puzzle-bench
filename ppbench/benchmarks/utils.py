import asyncio
import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Literal, Optional


class BenchmarkFileLogger:
    """Real-time file logger that writes to both a unified log and per-model logs."""

    def __init__(self, base_dir: str):
        self._base = Path(base_dir)
        self._logs_dir = self._base / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._unified_path = self._logs_dir / "all.log"
        self._lock = threading.Lock()
        self._model_files: Dict[str, Path] = {}

    def _safe_model_filename(self, model_name: str) -> str:
        return model_name.replace("/", "_").replace("@", "_") + ".log"

    def _model_path(self, model_name: str) -> Path:
        if model_name not in self._model_files:
            self._model_files[model_name] = self._logs_dir / self._safe_model_filename(model_name)
        return self._model_files[model_name]

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _write(self, model_name: str, line: str):
        ts = self._timestamp()
        formatted = f"[{ts}] [{model_name}] {line}\n"
        with self._lock:
            with open(self._unified_path, "a") as f:
                f.write(formatted)
            with open(self._model_path(model_name), "a") as f:
                f.write(f"[{ts}] {line}\n")

    def task_start(self, model_name: str, strategy: str, puzzle_id: str):
        self._write(model_name, f"START {strategy} | {puzzle_id}")

    def task_end(self, model_name: str, strategy: str, puzzle_id: str, status: str, duration: float, requests: int, moves: int):
        self._write(model_name, f"{status} {strategy} | {puzzle_id} | {duration:.1f}s | {requests} reqs | {moves} moves")

    def task_skip(self, model_name: str, strategy: str, puzzle_id: str):
        self._write(model_name, f"SKIP {strategy} | {puzzle_id} | cached")

    def task_error(self, model_name: str, strategy: str, puzzle_id: str, error: str):
        self._write(model_name, f"ERROR {strategy} | {puzzle_id} | {error}")

    def task_retry(self, model_name: str, strategy: str, puzzle_id: str, attempt: int, error: str, backoff: float):
        self._write(model_name, f"RETRY({attempt}) {strategy} | {puzzle_id} | {error} | backoff={backoff:.1f}s")

    def model_response(self, model_name: str, puzzle_id: str, input_tokens: int, output_tokens: int):
        self._write(model_name, f"RESPONSE {puzzle_id} | in={input_tokens} out={output_tokens}")

    def move(self, model_name: str, puzzle_id: str, move_str: str):
        self._write(model_name, f"MOVE {puzzle_id} | {move_str}")

    def info(self, model_name: str, message: str):
        self._write(model_name, message)

    def global_info(self, message: str):
        ts = self._timestamp()
        formatted = f"[{ts}] [SYSTEM] {message}\n"
        with self._lock:
            with open(self._unified_path, "a") as f:
                f.write(formatted)


@dataclass
class LogEntry:
    """A single log message with timestamp."""

    timestamp: float
    message: str
    level: str = "info"


class StrategyLogger:
    """
    Thin logging wrapper that collects messages for storage in detail_data.
    """

    def __init__(self):
        self.entries: List[LogEntry] = []
        self._start_time = time.time()

    def _log(self, message: str, level: str = "info"):
        elapsed = time.time() - self._start_time
        entry = LogEntry(timestamp=elapsed, message=message, level=level)
        self.entries.append(entry)

    def info(self, message: str):
        self._log(message, "info")

    def debug(self, message: str):
        self._log(message, "debug")

    def warn(self, message: str):
        self._log(message, "warn")

    def error(self, message: str):
        self._log(message, "error")

    def to_list(self) -> List[Dict[str, Any]]:
        """Export logs as list of dicts for storage."""
        return [asdict(e) for e in self.entries]


@dataclass
class TokenUsage:
    """Standardized usage container."""

    input_tokens: int = 0
    output_tokens: int = 0
    details: Dict[str, int] = field(default_factory=dict)


@dataclass
class RunResult:
    """Lightweight Index Entry"""

    strategy_id: str
    model_name: str
    puzzle_id: str
    puzzle_url: str

    is_success: bool
    duration_seconds: float
    total_requests: int

    parsed_moves: List[str] = field(default_factory=list)

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error_type: Optional[str] = None
    exception_traceback: Optional[str] = None

    # Arc-explainer style enrichment
    run_number: int = 0
    total_steps: int = 0
    max_steps: Optional[int] = None
    final_score: float = 0.0  # 0.0 or 1.0
    final_score_pct: int = 0  # 0 or 100
    solved: bool = False
    cost_usd: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    reset_count: int = 0


@dataclass
class DetailedRunResult:
    """Heavy Artifact"""

    summary: RunResult

    request_usages: List[TokenUsage] = field(default_factory=list)
    full_history: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    detail_data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# ─────────────────────────────────────────────────────────────────────
# Trajectory types (arc-explainer style)
# ─────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """Per-step trajectory record — one entry per model request cycle."""

    run_id: str
    model: str
    puzzle_id: str
    puzzle_type: str
    run_number: int
    step: int

    action: str  # tool call or model output text (summarized)
    score: float  # 0.0–1.0 (binary for pencil puzzles: 0 or 1)
    score_pct: int  # 0–100
    done: bool
    state: str  # NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER

    cumulative_cost_usd: Optional[float] = None
    step_cost_usd: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0

    observation: str = ""  # board state or tool result after action
    reasoning: str = ""  # model reasoning/thinking text if available
    board_state: str = ""  # string repr of puzzle at this step
    moves_so_far: List[str] = field(default_factory=list)


@dataclass
class RunRecord:
    """Per-run summary — one entry per (model × puzzle × run_number)."""

    run_id: str
    model: str
    puzzle_id: str
    puzzle_type: str
    run_number: int

    total_steps: int
    max_steps: Optional[int]
    final_score: float  # 0.0 or 1.0 for pencil puzzles
    final_score_pct: int  # 0 or 100
    solved: bool

    cost_usd: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    elapsed_seconds: float = 0.0

    error: Optional[str] = None
    strategy_id: str = ""
    seed: Optional[int] = None
    reset_count: int = 0
    total_moves: int = 0
    parsed_moves: List[str] = field(default_factory=list)


@dataclass
class TraceHeader:
    """First line of a trace JSONL file."""

    type: str = "header"
    schema_version: str = "1.0"
    run_id: str = ""
    model: str = ""
    puzzle_id: str = ""
    puzzle_type: str = ""
    run_number: int = 0
    seed: Optional[int] = None
    max_steps: Optional[int] = None
    system_prompt: str = ""
    timestamp: str = ""
    strategy_id: str = ""


@dataclass
class TraceStep:
    """Per-step line in trace JSONL."""

    type: str = "step"
    step: int = 0
    action: str = ""
    score: float = 0.0
    score_pct: int = 0
    done: bool = False
    state: str = "NOT_FINISHED"
    observation: str = ""
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    step_cost_usd: Optional[float] = None
    cumulative_cost_usd: Optional[float] = None
    board_state: str = ""
    moves_so_far: List[str] = field(default_factory=list)


@dataclass
class TraceSummary:
    """Final line of a trace JSONL file."""

    type: str = "summary"
    total_steps: int = 0
    final_score: float = 0.0
    final_score_pct: int = 0
    solved: bool = False
    cost_usd: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    total_moves: int = 0
    reset_count: int = 0


class TraceWriter:
    """Writes arc-explainer style trace JSONL files (header → steps → summary)."""

    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("")

    def _append(self, data: dict):
        with open(self._path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def write_header(self, header: TraceHeader):
        self._append(asdict(header))

    def write_step(self, step: TraceStep):
        self._append(asdict(step))

    def write_summary(self, summary: TraceSummary):
        self._append(asdict(summary))


class TaskPool:
    """
    Simple async task pool with per-group stats tracking.
    Handles semaphore, flight/pass/fail/done counts, and completion callbacks.
    """

    def __init__(self, concurrency: int = 50, on_complete: Optional[Callable] = None):
        self.sem = asyncio.Semaphore(concurrency)
        self.on_complete = on_complete
        self.stats: Dict[Hashable, Dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "fail": 0, "flight": 0, "done": 0}
        )
        self._tasks: List[tuple] = []  # (group, coro_fn, args)

    def submit(self, group: Hashable, coro_fn: Callable, *args):
        """Queue a task. coro_fn(*args) should return bool (success) or raise."""
        self._tasks.append((group, coro_fn, args))

    async def run(self):
        """Run all submitted tasks concurrently."""

        async def wrapped(group, coro_fn, args):
            s = self.stats[group]
            async with self.sem:
                s["flight"] += 1
                try:
                    success = await coro_fn(*args)
                    s["pass" if success else "fail"] += 1
                except Exception:
                    s["fail"] += 1
                    raise
                finally:
                    s["flight"] -= 1
                    s["done"] += 1
                    if self.on_complete:
                        self.on_complete()

        await asyncio.gather(
            *(wrapped(g, fn, a) for g, fn, a in self._tasks),
            return_exceptions=True,
        )


class StorageManager:
    def __init__(self, base_dir: str = "run_output"):
        self.base_dir = Path(base_dir)
        self.index_path = self.base_dir / "runs.jsonl"
        self.artifacts_dir = self.base_dir / "artifacts"

        self._lock = asyncio.Lock()

        self._cache: Dict[tuple, str] = {}
        self._error_types: Dict[tuple, str] = {}
        self._tracebacks: Dict[tuple, str] = {}

        self._initialize_storage()

    def _initialize_storage(self):
        """Creates dirs and hydrates memory cache from existing jsonl."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            print(f"Hydrating cache from {self.index_path}...")
            with open(self.index_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        key = (
                            data["strategy_id"],
                            data["model_name"],
                            data["puzzle_id"],
                        )

                        if data.get("is_success"):
                            status = "completed"
                        elif data.get("error_type"):
                            status = "error"
                            self._error_types[key] = data.get("error_type", "")
                            self._tracebacks[key] = data.get("exception_traceback", "")
                        else:
                            status = "failed"

                        self._cache[key] = status
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"Cache hydrated with {len(self._cache)} runs.")

    def lookup(
        self, strategy_id: str, model_name: str, puzzle_id: str
    ) -> Literal["completed", "failed", "error", "missing"]:
        key = (strategy_id, model_name, puzzle_id)
        return self._cache.get(key, "missing")

    def get_error_type(
        self, strategy_id: str, model_name: str, puzzle_id: str
    ) -> str | None:
        """Get the error type for a run, if it was an error."""
        key = (strategy_id, model_name, puzzle_id)
        return self._error_types.get(key)

    def is_retriable_error(
        self, strategy_id: str, model_name: str, puzzle_id: str
    ) -> bool:
        """Check if a run's error is retriable (network-level failure)."""
        key = (strategy_id, model_name, puzzle_id)
        tb = self._tracebacks.get(key, "")
        return "httpcore.ConnectError" in tb

    async def save(self, result: DetailedRunResult):
        """
        Writes the heavy artifact to disk and appends the summary to the index.
        Thread-safe execution using asyncio.Lock.
        """
        summary = result.summary

        safe_model = summary.model_name.replace("/", "_").replace("@", "_")
        target_dir = self.artifacts_dir / summary.strategy_id / safe_model

        key = (summary.strategy_id, summary.model_name, summary.puzzle_id)

        if summary.is_success:
            self._cache[key] = "completed"
        elif summary.error_type:
            self._cache[key] = "error"
        else:
            self._cache[key] = "failed"

        async with self._lock:
            target_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{summary.puzzle_id}.json"
            file_path = target_dir / filename
            temp_path = file_path.with_suffix(".tmp")

            with open(temp_path, "w") as f:
                f.write(result.to_json())
            os.replace(temp_path, file_path)

            with open(self.index_path, "a") as f:
                f.write(json.dumps(asdict(summary), default=str) + "\n")

            # Clean up any partial file for this run
            self._cleanup_partial(summary.strategy_id, summary.model_name, summary.puzzle_id)

    def save_partial(
        self,
        strategy_id: str,
        model_name: str,
        puzzle_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 0,
        moves: List[str] = None,
        elapsed_seconds: float = 0,
        status: str = "running",
    ):
        """
        Write live progress to a separate .live directory.
        Called during runs to show tokens used, moves made, etc.
        Non-async for simplicity (small writes).
        """
        live_dir = self.base_dir / ".live"
        live_dir.mkdir(parents=True, exist_ok=True)

        safe_model = model_name.replace("/", "_").replace("@", "_")
        filename = f"{strategy_id}_{safe_model}_{puzzle_id}.json"
        file_path = live_dir / filename

        data = {
            "strategy_id": strategy_id,
            "model_name": model_name,
            "puzzle_id": puzzle_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests": requests,
            "moves": moves or [],
            "elapsed_seconds": round(elapsed_seconds, 1),
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Atomic write
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f)
        os.replace(temp_path, file_path)

    def _cleanup_partial(self, strategy_id: str, model_name: str, puzzle_id: str):
        """Remove partial file after run completes."""
        live_dir = self.base_dir / ".live"
        safe_model = model_name.replace("/", "_").replace("@", "_")
        filename = f"{strategy_id}_{safe_model}_{puzzle_id}.json"
        file_path = live_dir / filename
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
