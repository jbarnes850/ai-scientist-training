from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "research_hypothesis_analysis"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from research_hypothesis_analysis.dataset import generate_frozen_dataset, split_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-seed", type=int, default=20260322)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = generate_frozen_dataset(base_seed=args.base_seed, force=args.force)
    payload = {
        "manifest": manifest,
        "paths": {
            split: str(split_path(split))
            for split in manifest["splits"]
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
