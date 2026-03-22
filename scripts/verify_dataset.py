from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "epistemic_taste"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from epistemic_taste.dataset import MANIFEST_PATH, SPLIT_SPECS, ensure_frozen_dataset, split_path  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    ensure_frozen_dataset()
    report = {"manifest_path": str(MANIFEST_PATH), "splits": {}, "overlap_checks": {}}

    domain_sets = {}
    template_sets = {}
    prior_ranges = {}
    likelihood_ranges = {}
    failed = False

    for split_name, split_spec in SPLIT_SPECS.items():
        rows = _load_rows(split_path(split_name))
        mode_counts = Counter(row["info"]["mode"] for row in rows)
        contradiction_count = sum(
            bool(row["info"]["contradiction_metadata"]["is_contradiction_focused"])
            for row in rows
        )
        prompt_domains = {row["info"]["domain_key"] for row in rows}
        prompt_templates = {int(row["info"]["template_id"]) for row in rows}
        prior_maps = [max(row["info"]["prior"].values()) for row in rows]
        likelihood_multipliers = [
            float(row["info"]["hidden"]["likelihood_multiplier"]) for row in rows
        ]

        report["splits"][split_name] = {
            "row_count": len(rows),
            "mode_counts": dict(mode_counts),
            "contradiction_fraction": contradiction_count / len(rows),
            "domain_keys": sorted(prompt_domains),
            "template_ids": sorted(prompt_templates),
            "prior_map_range_observed": [min(prior_maps), max(prior_maps)],
            "likelihood_multiplier_range_observed": [
                min(likelihood_multipliers),
                max(likelihood_multipliers),
            ],
        }

        domain_sets[split_name] = prompt_domains
        template_sets[split_name] = prompt_templates
        prior_ranges[split_name] = (min(prior_maps), max(prior_maps))
        likelihood_ranges[split_name] = (
            min(likelihood_multipliers),
            max(likelihood_multipliers),
        )

        if len(rows) != split_spec.count:
            failed = True

    report["overlap_checks"]["domain_overlap"] = {
        "train_dev": sorted(domain_sets["train"] & domain_sets["dev"]),
        "train_test": sorted(domain_sets["train"] & domain_sets["test"]),
        "dev_test": sorted(domain_sets["dev"] & domain_sets["test"]),
    }
    report["overlap_checks"]["template_overlap"] = {
        "train_dev": sorted(template_sets["train"] & template_sets["dev"]),
        "train_test": sorted(template_sets["train"] & template_sets["test"]),
        "dev_test": sorted(template_sets["dev"] & template_sets["test"]),
    }
    report["overlap_checks"]["prior_range_overlap"] = prior_ranges
    report["overlap_checks"]["likelihood_range_overlap"] = likelihood_ranges

    if any(report["overlap_checks"]["domain_overlap"].values()):
        failed = True
    if any(report["overlap_checks"]["template_overlap"].values()):
        failed = True

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
