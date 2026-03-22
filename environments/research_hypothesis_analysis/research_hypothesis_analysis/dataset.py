from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_dataset

from .bayes import ACTIVE_MODE, PASSIVE_MODE, SplitSpec, build_episode

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
DEFAULT_BASE_SEED = 20260322

SPLIT_SPECS = {
    "train": SplitSpec(
        name="train",
        count=4000,
        domains=("cell_signaling", "ecology", "materials"),
        template_ids=(0, 1, 2, 3, 4, 5),
        prior_map_range=(0.45, 0.65),
        likelihood_multiplier_range=(0.95, 1.05),
        contradiction_rate=0.30,
    ),
    "dev": SplitSpec(
        name="dev",
        count=500,
        domains=("public_health",),
        template_ids=(6, 7),
        prior_map_range=(0.66, 0.80),
        likelihood_multiplier_range=(1.10, 1.20),
        contradiction_rate=0.20,
    ),
    "test": SplitSpec(
        name="test",
        count=500,
        domains=("econ_policy",),
        template_ids=(8, 9),
        prior_map_range=(0.34, 0.44),
        likelihood_multiplier_range=(0.80, 0.90),
        contradiction_rate=0.40,
    ),
}


def split_path(split: str) -> Path:
    return DATA_DIR / f"{split}.jsonl"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _strip_non_serializable(info: dict) -> dict:
    stripped = dict(info)
    stripped.pop("bayes_reference", None)
    return stripped


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            serializable_row = dict(row)
            serializable_row["info"] = _strip_non_serializable(serializable_row["info"])
            handle.write(json.dumps(serializable_row, sort_keys=True) + "\n")


def _build_schedule(
    *,
    count: int,
    active_ratio: float,
    contradiction_rate: float,
    domains: tuple[str, ...],
    template_ids: tuple[int, ...],
    seed: int,
) -> list[dict]:
    import numpy as np

    rng = np.random.default_rng(seed)
    active_count = int(round(count * active_ratio))
    contradiction_count = int(round(count * contradiction_rate))

    modes = [ACTIVE_MODE] * active_count + [PASSIVE_MODE] * (count - active_count)
    contradiction_flags = [True] * contradiction_count + [False] * (count - contradiction_count)
    domain_schedule = [domains[index % len(domains)] for index in range(count)]
    template_schedule = [template_ids[index % len(template_ids)] for index in range(count)]

    rng.shuffle(modes)
    rng.shuffle(contradiction_flags)
    rng.shuffle(domain_schedule)
    rng.shuffle(template_schedule)

    return [
        {
            "mode": modes[index],
            "contradiction_target": contradiction_flags[index],
            "domain_key": domain_schedule[index],
            "template_id": template_schedule[index],
        }
        for index in range(count)
    ]


def generate_frozen_dataset(
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    force: bool = False,
) -> dict:
    ensure_data_dir()
    if MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    manifest = {
        "version": "research-hypothesis-analysis-v1",
        "base_seed": base_seed,
        "splits": {},
    }

    for split_index, (split_name, split_spec) in enumerate(SPLIT_SPECS.items()):
        rows = []
        schedule = _build_schedule(
            count=split_spec.count,
            active_ratio=0.70,
            contradiction_rate=split_spec.contradiction_rate,
            domains=split_spec.domains,
            template_ids=split_spec.template_ids,
            seed=base_seed + split_index * 997,
        )
        for episode_index, plan in enumerate(schedule):
            row = build_episode(
                split_spec=split_spec,
                episode_index=episode_index,
                episode_seed=base_seed + split_index * 100_000 + episode_index,
                mode=plan["mode"],
                contradiction_target=plan["contradiction_target"],
                domain_key=plan["domain_key"],
                template_id=plan["template_id"],
            )
            rows.append(row)
        _write_jsonl(split_path(split_name), rows)
        manifest["splits"][split_name] = {
            "count": split_spec.count,
            "domains": list(split_spec.domains),
            "template_ids": list(split_spec.template_ids),
            "prior_map_range": list(split_spec.prior_map_range),
            "likelihood_multiplier_range": list(split_spec.likelihood_multiplier_range),
            "contradiction_rate": split_spec.contradiction_rate,
            "mode_counts": {
                ACTIVE_MODE: sum(
                    1 for row in rows if row["info"]["mode"] == ACTIVE_MODE
                ),
                PASSIVE_MODE: sum(
                    1 for row in rows if row["info"]["mode"] == PASSIVE_MODE
                ),
            },
        }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def ensure_frozen_dataset() -> dict:
    if MANIFEST_PATH.exists() and all(split_path(split).exists() for split in SPLIT_SPECS):
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return generate_frozen_dataset()


def build_dataset(split: str, max_examples: int = -1, seed: int = 0) -> Dataset:
    ensure_frozen_dataset()
    path = split_path(split)
    dataset = load_dataset("json", data_files=str(path), split="train")
    if seed:
        dataset = dataset.shuffle(seed=seed)
    if max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    return dataset
