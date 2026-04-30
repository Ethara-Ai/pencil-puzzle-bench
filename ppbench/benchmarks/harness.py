"""Harness: runs strategies and owns all operational concerns.

The harness is responsible for:
  - Running the agent (streaming, retries, timeouts)
  - Extracting usage from messages (always, including errors)
  - Error classification and traceback capture
  - Building DetailedRunResult
  - Saving to storage

The harness NEVER touches puzzle logic, prompts, or tool definitions.
"""

import asyncio
import json
import time
import traceback
from typing import Any, Optional

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelResponse

from .strategy import Strategy, StrategyResult
from .utils import (
    BenchmarkFileLogger,
    DetailedRunResult,
    RunResult,
    StorageManager,
    TokenUsage,
    TraceHeader,
    TraceStep,
    TraceSummary,
    TraceWriter,
)


def _extract_usages(messages: list) -> list[TokenUsage]:
    """Extract token usage from a list of pydantic-ai messages."""
    usages = []
    for msg in messages:
        if isinstance(msg, ModelResponse) and msg.usage:
            usages.append(TokenUsage(
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                details=msg.usage.details or {},
            ))
    return usages


def _serialize_messages(messages: list) -> list[dict]:
    """Serialize pydantic-ai messages to JSON-safe dicts."""
    try:
        from pydantic_ai.messages import ModelMessagesTypeAdapter
        raw = ModelMessagesTypeAdapter.dump_json(messages).decode()
        return json.loads(raw)
    except Exception:
        return []


def _extract_action_from_node(node: Any, run: Any) -> str:
    """Extract a human-readable action summary from the latest agent node."""
    try:
        from pydantic_ai.messages import ToolCallPart, TextPart
        msgs = run.result.all_messages() if run.result else []
        if not msgs:
            return ""
        last_msg = msgs[-1]
        parts = getattr(last_msg, "parts", [])
        actions = []
        for part in parts:
            if isinstance(part, ToolCallPart):
                args_str = json.dumps(part.args)[:200] if part.args else ""
                actions.append(f"{part.tool_name}({args_str})")
            elif isinstance(part, TextPart):
                text = part.content[:200] if part.content else ""
                if text:
                    actions.append(text)
        return " | ".join(actions) if actions else ""
    except Exception:
        return ""


async def run_strategy(
    strategy: Strategy,
    puzzle: Any,
    model_obj: Any,
    model_name: str,
    storage: Optional[StorageManager] = None,
    logger: Optional[BenchmarkFileLogger] = None,
    max_steps: int | None = None,
    puzzle_id_override: str | None = None,
    run_number: int = 0,
    seed: int | None = None,
    trace_dir: Optional[str] = None,
    max_retries: int = 3,
    request_timeout: float = 12 * 60 * 60,
) -> DetailedRunResult:
    """Run a strategy against a puzzle. Handles all operational concerns.

    Returns a DetailedRunResult with usage data populated in ALL cases
    (success, failure, and error).
    """
    import httpx
    import openai
    from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

    # Optional timeout error types
    retryable_types = [
        openai.APIError,
        openai.APIConnectionError,
        httpx.RemoteProtocolError,
        httpx.ReadTimeout,
        TimeoutError,
    ]
    try:
        from ppbench.benchmarks.connection_logger import ChunkTimeoutError, TTFBTimeoutError
        retryable_types.extend([TTFBTimeoutError, ChunkTimeoutError])
    except ImportError:
        pass
    retryable = tuple(retryable_types)

    start_time = time.time()
    config = strategy.build_agent(puzzle, model_obj, model_name)
    puzzle_id = puzzle_id_override or f"{puzzle.pid}_{puzzle.id}"

    if max_steps is not None:
        from pydantic_ai import UsageLimits
        if config.usage_limits:
            existing_limit = getattr(config.usage_limits, 'request_limit', None)
            if existing_limit is None or max_steps < existing_limit:
                config.usage_limits.request_limit = max_steps
        else:
            config.usage_limits = UsageLimits(request_limit=max_steps)

    context_too_long_patterns = (
        "maximum context length",
        "maximum prompt length",
        "prompt is too long",
        "too many tokens",
    )

    def is_retryable(e: Exception) -> bool:
        if isinstance(e, retryable):
            return True
        if isinstance(e, ModelHTTPError):
            body_str = str(e.body).lower() if e.body else ""
            if any(p in body_str for p in context_too_long_patterns):
                return False
            if e.status_code == 429:
                return True
            if e.status_code == 400:
                if "reasoning" in body_str and "without its required" in body_str:
                    return True
        if isinstance(e, UnexpectedModelBehavior):
            return True
        return False

    def get_backoff(e: Exception, attempt: int) -> float:
        import random
        import re
        base = 2 ** attempt
        jitter = random.uniform(0, 1)
        if isinstance(e, ModelHTTPError) and e.status_code == 429:
            body_str = str(e.body) if e.body else ""
            match = re.search(r'retry.*?(\d+(?:\.\d+)?)\s*s', body_str, re.IGNORECASE)
            if match:
                return min(float(match.group(1)) + jitter, 300)
            return min(10 * (2 ** attempt) + jitter, 120)
        return min(base + jitter, 60)

    # --- Execute ---

    last_run = [None]
    messages = []       # Accumulated messages (partial or full)
    raw_output = ""
    error_type = None
    trace = None
    step_count = [0]
    reset_count = [0]

    agent_kwargs = dict(deps=config.deps)
    if config.usage_limits:
        agent_kwargs["usage_limits"] = config.usage_limits

    # --- Trajectory setup ---
    from pathlib import Path
    trace_writer = None
    if trace_dir:
        safe_model = model_name.replace("/", "_").replace("@", "_")
        trace_path = Path(trace_dir) / safe_model / f"{puzzle_id}.jsonl"
        trace_writer = TraceWriter(trace_path)
        trace_writer.write_header(TraceHeader(
            run_id=f"{model_name}_{puzzle_id}_run{run_number}",
            model=model_name,
            puzzle_id=puzzle_id,
            puzzle_type=puzzle.pid,
            run_number=run_number,
            seed=seed,
            max_steps=max_steps,
            system_prompt=config.prompt[:2000] if isinstance(config.prompt, str) else "",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            strategy_id=strategy.strategy_id,
        ))

    cumulative_input = [0]
    cumulative_output = [0]
    cumulative_reasoning = [0]

    async def do_run():
        async with config.agent.iter(config.prompt, **agent_kwargs) as run:
            last_run[0] = run
            async for node in run:
                strategy.on_node(node, run.ctx)

                if PydanticAgent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:
                            pass

                    step_count[0] += 1

                    if trace_writer:
                        step_messages = run.result.all_messages() if run.result else []
                        step_usages = _extract_usages(step_messages[-(2):]) if step_messages else []
                        step_input = step_usages[-1].input_tokens if step_usages else 0
                        step_output = step_usages[-1].output_tokens if step_usages else 0
                        step_reasoning = step_usages[-1].details.get("reasoning_tokens", 0) if step_usages else 0
                        step_cached = step_usages[-1].details.get("cached_tokens", 0) if step_usages else 0

                        cumulative_input[0] += step_input
                        cumulative_output[0] += step_output
                        cumulative_reasoning[0] += step_reasoning

                        action_summary = _extract_action_from_node(node, run)
                        board_state = ""
                        moves_so_far: list[str] = []
                        try:
                            board_state = puzzle.get_string_repr()
                        except Exception:
                            pass
                        try:
                            if hasattr(config.deps, "list_of_moves"):
                                moves_so_far = list(config.deps.list_of_moves)
                        except Exception:
                            pass

                        is_done = puzzle.isComplete()
                        trace_writer.write_step(TraceStep(
                            step=step_count[0],
                            action=action_summary,
                            score=1.0 if is_done else 0.0,
                            score_pct=100 if is_done else 0,
                            done=is_done,
                            state="WIN" if is_done else "NOT_FINISHED",
                            input_tokens=step_input,
                            output_tokens=step_output,
                            reasoning_tokens=step_reasoning,
                            cached_input_tokens=step_cached,
                            board_state=board_state,
                            moves_so_far=moves_so_far,
                        ))

        return run.result

    last_error = None
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(request_timeout):
                result = await do_run()

            # Success — extract everything
            messages = result.all_messages()
            raw_output = result.output
            break

        except Exception as e:
            if not is_retryable(e):
                # Non-retryable: capture what we can and stop
                error_type = str(type(e))
                trace = traceback.format_exc()
                if logger:
                    logger.task_error(model_name, strategy.__class__.__name__, puzzle_id, f"{type(e).__name__}: {e}")
                if last_run[0]:
                    try:
                        messages = last_run[0].all_messages()
                    except Exception:
                        pass
                break

            last_error = e
            if attempt < max_retries - 1:
                backoff = get_backoff(e, attempt)
                if logger:
                    logger.task_retry(model_name, strategy.__class__.__name__, puzzle_id, attempt + 1, str(type(e).__name__), backoff)
                await asyncio.sleep(backoff)
    else:
        # All retries exhausted
        error_type = str(type(last_error))
        trace = traceback.format_exc()
        if last_run[0]:
            try:
                messages = last_run[0].all_messages()
            except Exception:
                pass

    # --- Extract usage (harness concern — always, regardless of outcome) ---

    request_usages = _extract_usages(messages)
    full_history = _serialize_messages(messages)

    if logger and request_usages:
        for usage in request_usages:
            logger.model_response(model_name, puzzle_id, usage.input_tokens, usage.output_tokens)

    # --- Extract domain result (strategy concern) ---

    strategy_result = StrategyResult(is_success=False)
    if error_type is None:
        try:
            strategy_result = strategy.extract_result(puzzle, config.deps, raw_output)
        except Exception as e:
            error_type = str(type(e))
            trace = traceback.format_exc()

    # --- Extract logs (strategy concern — move-by-move tracing) ---

    try:
        logs = strategy.extract_logs(config.deps)
    except Exception:
        logs = []

    # --- Build and save ---

    duration = time.time() - start_time

    total_input = sum(u.input_tokens for u in request_usages)
    total_output = sum(u.output_tokens for u in request_usages)
    total_reasoning = sum(u.details.get("reasoning_tokens", 0) for u in request_usages)

    detailed = DetailedRunResult(
        summary=RunResult(
            strategy_id=strategy.strategy_id,
            model_name=model_name,
            puzzle_id=puzzle_id,
            puzzle_url=puzzle.url,
            is_success=strategy_result.is_success,
            duration_seconds=duration,
            total_requests=len(request_usages),
            parsed_moves=strategy_result.parsed_moves,
            error_type=error_type,
            exception_traceback=trace,
            run_number=run_number,
            total_steps=step_count[0],
            max_steps=max_steps,
            final_score=1.0 if strategy_result.is_success else 0.0,
            final_score_pct=100 if strategy_result.is_success else 0,
            solved=strategy_result.is_success,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_reasoning_tokens=total_reasoning,
            reset_count=reset_count[0],
        ),
        request_usages=request_usages,
        full_history=full_history,
        logs=logs,
        detail_data={
            "raw_output": strategy_result.raw_output,
            **strategy_result.detail_data,
        },
    )

    if trace_writer:
        trace_writer.write_summary(TraceSummary(
            total_steps=step_count[0],
            final_score=1.0 if strategy_result.is_success else 0.0,
            final_score_pct=100 if strategy_result.is_success else 0,
            solved=strategy_result.is_success,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_reasoning_tokens=total_reasoning,
            elapsed_seconds=duration,
            error=error_type,
            total_moves=len(strategy_result.parsed_moves),
            reset_count=reset_count[0],
        ))

    if storage:
        await storage.save(detailed)

    return detailed
