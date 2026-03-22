from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, parse_results, run_model_eval, summarize_results_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--rollouts-per-example", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--output-json", default=str(ROOT / "configs" / "heldout_eval_summary.json"))
    args = parser.parse_args()

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
    output = {
        "results_path": str(results_path),
        "metadata": metadata,
        "summary": summarize_results_rows(rows),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
