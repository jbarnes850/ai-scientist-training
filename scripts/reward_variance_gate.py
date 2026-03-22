from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_AXES = [
    "reward",
    "mean_experiment_reward",
    "mean_calibration_reward",
    "final_map_correct",
]


def load_rows(results_path: Path) -> list[dict]:
    rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl")
    parser.add_argument("--axes", default=",".join(DEFAULT_AXES))
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--max-flat-fraction", type=float, default=0.5)
    args = parser.parse_args()

    axes = [axis for axis in args.axes.split(",") if axis]
    rows = load_rows(Path(args.results_jsonl))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["example_id"])].append(row)

    report = {"groups": len(grouped), "axes": {}}
    failed = False
    for axis in axes:
        near_zero = 0
        total = 0
        group_stds = []
        for group_rows in grouped.values():
            values = [
                float(row[axis])
                for row in group_rows
                if axis in row and isinstance(row[axis], (int, float))
            ]
            if len(values) < 2:
                continue
            total += 1
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = variance ** 0.5
            group_stds.append(std)
            if std < args.threshold:
                near_zero += 1
        flat_fraction = (near_zero / total) if total else 1.0
        report["axes"][axis] = {
            "groups_with_variance": total,
            "near_zero_groups": near_zero,
            "flat_fraction": flat_fraction,
            "mean_group_std": (sum(group_stds) / len(group_stds)) if group_stds else 0.0,
        }
        if flat_fraction > args.max_flat_fraction:
            failed = True

    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
