from __future__ import annotations

import argparse
import json
from pathlib import Path

from _epistemic_common import ROOT, _load_rows, simulate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--policy", default="oracle")
    parser.add_argument("--output-jsonl", default=str(ROOT / "configs" / "example_trajectories.jsonl"))
    args = parser.parse_args()

    rows = _load_rows(split=args.split, max_examples=-1, seed=0)
    picks = []
    seen = set()
    for row in rows:
        key = (row["info"]["mode"], bool(row["info"]["contradiction_metadata"]["is_contradiction_focused"]))
        if key in seen:
            continue
        seen.add(key)
        picks.append(row["info"])
        if len(seen) == 4:
            break

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, episode in enumerate(picks):
            result = simulate_policy(episode, args.policy, seed=index)
            payload = {
                "episode_id": episode["episode_id"],
                "mode": episode["mode"],
                "contradiction": bool(episode["contradiction_metadata"]["is_contradiction_focused"]),
                "policy": args.policy,
                "summary": result["summary"],
                "trajectory_log": result["trajectory_log"],
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
