from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import (
    ROOT,
    latest_results_path,
    parse_results,
    run_heuristic_baselines,
    run_model_eval,
    summarize_results_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--rollouts-per-example", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--output-json", default=str(ROOT / "configs" / "base_model_eval_summary.json"))
    args = parser.parse_args()

    output = {
        "heuristics": run_heuristic_baselines(
            split=args.split,
            max_examples=args.num_examples,
            seed=0,
            include_lexical=True,
        )
    }

    if not args.skip_model:
        results_path = run_model_eval(
            split=args.split,
            model=args.model,
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            max_concurrent=args.max_concurrent,
            sampling_args={"temperature": 0.3, "max_tokens": 1024},
            state_columns=["episode_summary", "trajectory_log", "posterior_trace"],
        )
        metadata, rows = parse_results(results_path)
        output["base_model"] = {
            "results_path": str(results_path),
            "metadata": metadata,
            "summary": summarize_results_rows(rows),
        }
    else:
        existing = latest_results_path("research-hypothesis-analysis", args.model)
        if existing is not None:
            _, rows = parse_results(existing)
            output["existing_base_model_results"] = {
                "results_path": str(existing),
                "summary": summarize_results_rows(rows),
            }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
