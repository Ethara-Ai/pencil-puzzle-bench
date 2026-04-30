"""
Run the benchmark from the command line.

Usage:
    uv run python -u -m ppbench.benchmarks.run_benchmark
    uv run python -u -m ppbench.benchmarks.run_benchmark --model anthropic/claude-sonnet-4-6 --dataset golden_30
    uv run python -u -m ppbench.benchmarks.run_benchmark --model openai/gpt-4o --strategy direct_ask --n-puzzles 5
    uv run python -u -m ppbench.benchmarks.run_benchmark --model local/my-model --dataset my_dataset --concurrency 5
"""

import argparse
import asyncio
import json
import os
import random
from datetime import datetime

from ppbench.benchmarks import BasicAgenticSolve, DirectAskStrategy, run
from ppbench.benchmarks.model_list import get_model, supports_tools, API_KEY_ENV
from ppbench import load_dataset

STRATEGY_MAP = {
    "direct_ask": DirectAskStrategy,
    "basic_agentic": BasicAgenticSolve,
    "both": None,  # sentinel for both
}

DEFAULT_MODELS = {
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-sonnet-4-6",
    "google": "google/gemini-2.5-flash",
    "xai": "xai/grok-3-mini",
    "openrouter": "openrouter/deepseek/deepseek-chat-v3-0324",
    "glm": "glm/glm-4-plus",
    "bedrock": "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
}


def _detect_available_models() -> list[str]:
    available = []
    for provider, env_var in API_KEY_ENV.items():
        if os.environ.get(env_var):
            if provider in DEFAULT_MODELS:
                available.append(DEFAULT_MODELS[provider])
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        available.append(DEFAULT_MODELS["bedrock"])
    return available


def _write_sample_artifact(path, strategy, model, puzzle_id, puzzle_url):
    sample = {
        "summary": {
            "strategy_id": strategy.strategy_id,
            "model_name": model,
            "puzzle_id": puzzle_id,
            "puzzle_url": puzzle_url,
            "is_success": False,
            "duration_seconds": 0.0,
            "total_requests": 0,
            "parsed_moves": [],
            "timestamp": "<filled at runtime>",
            "error_type": None,
            "exception_traceback": None,
            "run_number": 0,
            "total_steps": 0,
            "max_steps": None,
            "final_score": 0.0,
            "final_score_pct": 0,
            "solved": False,
            "cost_usd": None,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
            "reset_count": 0,
        },
        "request_usages": [
            {"input_tokens": 0, "output_tokens": 0, "details": {}}
        ],
        "full_history": ["<serialized pydantic-ai messages>"],
        "logs": [
            {"timestamp": 0.0, "message": "<strategy log entry>", "level": "info"}
        ],
        "detail_data": {"raw_output": "<model output text>"},
    }
    with open(path, "w") as f:
        json.dump(sample, f, indent=2)


def _write_sample_trace(path, model, puzzle_id, puzzle_type, strategy):
    lines = [
        {
            "type": "header",
            "schema_version": "1.0",
            "run_id": f"{model}_{puzzle_id}_run0",
            "model": model,
            "puzzle_id": puzzle_id,
            "puzzle_type": puzzle_type,
            "run_number": 0,
            "seed": None,
            "max_steps": None,
            "system_prompt": "<system prompt truncated>",
            "timestamp": "<filled at runtime>",
            "strategy_id": strategy.strategy_id,
        },
        {
            "type": "step",
            "step": 1,
            "action": "make_move({\"move\": \"R1C1=5\"})",
            "score": 0.0,
            "score_pct": 0,
            "done": False,
            "state": "NOT_FINISHED",
            "observation": "",
            "reasoning": "",
            "input_tokens": 1234,
            "output_tokens": 89,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "step_cost_usd": None,
            "cumulative_cost_usd": None,
            "board_state": "<puzzle board string>",
            "moves_so_far": ["R1C1=5"],
        },
        {
            "type": "summary",
            "total_steps": 1,
            "final_score": 0.0,
            "final_score_pct": 0,
            "solved": False,
            "cost_usd": None,
            "total_input_tokens": 1234,
            "total_output_tokens": 89,
            "total_reasoning_tokens": 0,
            "elapsed_seconds": 0.0,
            "error": None,
            "total_moves": 0,
            "reset_count": 0,
        },
    ]
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _dry_run(args, strategies, models):
    """Validate setup without calling the API."""
    from pathlib import Path
    from ppbench import Puzzle

    output_dir = args.output_dir or f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = Path(output_dir)

    print("=" * 60)
    print("DRY RUN — creating output structure (no API calls)")
    print("=" * 60)
    print()

    # 1. Show models
    print(f"Models ({len(models)}):")
    for model in models:
        provider = model.split("/")[0]
        env_key = API_KEY_ENV.get(provider)
        key_status = ""
        if env_key:
            key_status = " ✓" if os.environ.get(env_key) else " ✗ KEY MISSING"
        elif provider == "bedrock":
            key_status = " ✓" if os.environ.get("AWS_ACCESS_KEY_ID") else " ✗ AWS CREDS MISSING"
        print(f"  {model} (tools={supports_tools(model)}{key_status})")
    print()

    # 2. Load puzzles
    print(f"Dataset: {args.dataset}")
    if args.puzzles:
        print(f"  Explicit puzzles: {args.puzzles}")
        records = []
        all_records = None
        for p in args.puzzles:
            if "://" in p:
                records.append({"puzzlink_url": p, "pid": p.split("?")[1].split("/")[0] if "?" in p else "unknown"})
            else:
                if all_records is None:
                    all_records = {}
                    for ds in dict.fromkeys([args.dataset, "golden_300", "golden"]):
                        try:
                            for r in load_dataset(ds):
                                rid = f"{r['pid']}_{r['puzzlink_url'][-8:]}"
                                all_records[rid] = r
                        except Exception:
                            continue
                if p in all_records:
                    records.append(all_records[p])
                else:
                    print(f"  ⚠ puzzle '{p}' not found in any dataset")
    else:
        records = load_dataset(args.dataset)
        if args.puzzle_types:
            records = [r for r in records if r["pid"] in args.puzzle_types]
            print(f"  Puzzle types filter: {args.puzzle_types}")
        if args.seed is not None:
            rng = random.Random(args.seed)
            records = list(records)
            rng.shuffle(records)
        if args.n_puzzles is not None:
            records = records[:args.n_puzzles]

    print(f"  Puzzles loaded: {len(records)}")
    print()

    # 3. Determine valid strategies per model
    print(f"Strategies: {[s.__name__ for s in strategies]}")
    for s in strategies:
        inst = s()
        if inst.requires_tools:
            no_tool_models = [m for m in models if not supports_tools(m)]
            if no_tool_models:
                print(f"  ⚠ {s.__name__} requires tools — skipped for: {no_tool_models}")
    print()

    # 4. Create output directory structure
    base_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = base_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    (logs_dir / "all.log").touch()
    for model in models:
        safe = model.replace("/", "_").replace("@", "_")
        (logs_dir / f"{safe}.log").touch()

    total_tasks = 0

    for strat_cls in strategies:
        strategy = strat_cls()
        for model in models:
            if strategy.requires_tools and not supports_tools(model):
                continue
            safe_model = model.replace("/", "_").replace("@", "_")
            strategy_dir = artifacts_dir / strategy.strategy_id / safe_model
            strategy_dir.mkdir(parents=True, exist_ok=True)

            trace_model_dir = traces_dir / safe_model
            trace_model_dir.mkdir(parents=True, exist_ok=True)

            for record in records:
                url = record["puzzlink_url"]
                puzzle = Puzzle.from_url(url)
                puzzle_id = f"{puzzle.pid}_{puzzle.id}"
                total_tasks += 1

                _write_sample_artifact(strategy_dir / f"{puzzle_id}.json", strategy, model, puzzle_id, url)
                _write_sample_trace(trace_model_dir / f"{puzzle_id}.jsonl", model, puzzle_id, puzzle.pid, strategy)

    print(f"Output dir: {output_dir}")
    print(f"  logs/")
    print(f"    all.log")
    for model in models:
        safe = model.replace("/", "_").replace("@", "_")
        print(f"    {safe}.log")
    print(f"  traces/")
    for model in models:
        safe = model.replace("/", "_").replace("@", "_")
        print(f"    {safe}/  ({len(records)} .jsonl traces)")
    print(f"  artifacts/")
    for strat_cls in strategies:
        strategy = strat_cls()
        for model in models:
            if strategy.requires_tools and not supports_tools(model):
                continue
            safe_model = model.replace("/", "_").replace("@", "_")
            print(f"    {strategy.strategy_id}/{safe_model}/  ({len(records)} puzzles)")
    print()
    print(f"Total tasks: {total_tasks} ({len(records)} puzzles × {len(models)} models × strategies)")
    print(f"Concurrency: {args.concurrency}")
    print()
    print(f"✓ Dry run complete. Output structure created at: {output_dir}")
    print(f"  Remove --dry-run to execute and populate with results.")



def main():
    parser = argparse.ArgumentParser(
        description="Run the pencil puzzle benchmark harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Claude Sonnet, both strategies, golden_30, 3 puzzles
  uv run python -u -m ppbench.benchmarks.run_benchmark

  # Specific model and dataset
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
        """,
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        default=None,
        help="Model identifier(s) in provider/model@variant format. Multiple allowed. "
             "If omitted, auto-detects all providers with API keys set.",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGY_MAP.keys()),
        default="both",
        help="Strategy to use (default: both)",
    )
    parser.add_argument(
        "--dataset", "-d",
        default="golden_30",
        help="Dataset name (default: golden_30). Use any name registered in dataset.py.",
    )
    parser.add_argument(
        "--n-puzzles", "-n",
        type=int,
        default=None,
        help="Number of puzzles to run (default: all in dataset)",
    )
    parser.add_argument(
        "--puzzles",
        nargs="+",
        default=None,
        help="Specific puzzle IDs or URLs (overrides --dataset and --n-puzzles)",
    )
    parser.add_argument(
        "--puzzle-types",
        nargs="+",
        default=None,
        help="Filter to specific puzzle types (e.g., tapa slither sudoku)",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Max concurrent tasks (default: 10)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: output/<timestamp>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for puzzle sampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate setup (load dataset, resolve model, list puzzles) without calling the API",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max model request steps per puzzle (caps agent iterations). "
             "E.g., --max-steps 100 stops the agent after 100 model calls.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat each (model × strategy × puzzle) combo (default: 1). "
             "Results are saved with _runN suffix.",
    )

    args = parser.parse_args()

    if args.strategy == "both":
        strategies = [DirectAskStrategy, BasicAgenticSolve]
    else:
        strategies = [STRATEGY_MAP[args.strategy]]

    if args.model:
        models = args.model
    else:
        models = _detect_available_models()
        if not models and not args.dry_run:
            print("No API keys found in environment. Set at least one in .env:")
            for provider, env_var in API_KEY_ENV.items():
                print(f"  {env_var}  →  {DEFAULT_MODELS.get(provider, provider + '/...')}")
            print("  AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY  →  bedrock/...")
            return
        if not models:
            models = list(DEFAULT_MODELS.values())
        else:
            print(f"Auto-detected {len(models)} model(s): {models}")

    if args.dry_run:
        _dry_run(args, strategies, models)
        return

    output_dir = args.output_dir or f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = asyncio.run(run(
        models=models,
        strategies=strategies,
        dataset=args.dataset,
        puzzles=args.puzzles,
        puzzle_types=args.puzzle_types,
        n_puzzles=args.n_puzzles,
        concurrency=args.concurrency,
        output_dir=output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
        runs=args.runs,
    ))


if __name__ == "__main__":
    main()
