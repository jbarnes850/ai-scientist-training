from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "research_hypothesis_analysis"
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from research_hypothesis_analysis.bayes import (  # noqa: E402
    ACTIVE_MODE,
    EXPERIMENT_COSTS,
    EXPERIMENT_IDS,
    EXTRA_TURN_PENALTY,
    TURN_BUDGET,
    apply_observation,
    compute_episode_summary,
    utility_map_for_state,
)
from research_hypothesis_analysis.dataset import build_dataset, ensure_frozen_dataset  # noqa: E402


def _record_event(
    state: dict,
    *,
    kind: str,
    value: float,
    reason: str,
    metadata: dict | None = None,
) -> None:
    state["reward_events"].append(
        {
            "kind": kind,
            "value": float(value),
            "reason": reason,
            "metadata": metadata or {},
        }
    )


def _load_rows(split: str, max_examples: int, seed: int) -> list[dict]:
    ensure_frozen_dataset()
    dataset = build_dataset(split=split, max_examples=max_examples, seed=seed)
    return dataset.to_list()


def _lexical_score(description: str) -> int:
    positive_tokens = [
        "high",
        "fine-grained",
        "single-cell",
        "matched",
        "high-resolution",
        "linked",
        "panel",
        "tagged",
    ]
    negative_tokens = ["cheap", "proxy", "repeat", "compressed", "routine"]
    lower = description.lower()
    return sum(token in lower for token in positive_tokens) - sum(
        token in lower for token in negative_tokens
    )


def _choose_experiment(
    episode: dict,
    available: list[str],
    current_posterior: dict[str, float],
    policy: str,
    rng: random.Random,
) -> str:
    if policy == "cheap_first":
        return min(
            available,
            key=lambda experiment_id: (EXPERIMENT_COSTS[experiment_id], EXPERIMENT_IDS.index(experiment_id)),
        )
    if policy == "lexical":
        visible = {item["experiment_id"]: item["description"] for item in episode["visible_experiments"]}
        return max(
            available,
            key=lambda experiment_id: (_lexical_score(visible[experiment_id]), -EXPERIMENT_IDS.index(experiment_id)),
        )
    if policy == "oracle":
        utilities = utility_map_for_state(
            current_posterior,
            episode["hidden"]["likelihoods"],
            available,
        )
        return max(
            available,
            key=lambda experiment_id: (utilities[experiment_id], -EXPERIMENT_IDS.index(experiment_id)),
        )
    return rng.choice(available)


def simulate_policy(episode: dict, policy: str, seed: int = 0) -> dict:
    rng = random.Random(seed)
    canonical_prior = dict(episode["hidden"].get("canonical_prior", episode["prior"]))
    state = {
        "mode": episode["mode"],
        "episode_spec": episode,
        "current_posterior": canonical_prior,
        "reward_events": [],
        "belief_reports": [],
        "used_experiments": [],
        "available_experiments": list(EXPERIMENT_IDS),
        "trajectory_log": [],
        "posterior_trace": [
            {"step_index": 0, "source": "prior", "posterior": dict(canonical_prior)}
        ],
        "observation_history": [],
        "last_valid_belief": None,
        "stop_condition": f"simulated_{policy}",
    }

    def record_exact_belief(stop: bool) -> None:
        belief = dict(state["current_posterior"])
        state["belief_reports"].append(
            {
                "belief": belief,
                "exact_posterior": belief,
                "stop": stop,
                "brier": 0.0,
                "calibration_reward": 0.0,
            }
        )
        state["last_valid_belief"] = belief
        _record_event(
            state,
            kind="calibration_reward",
            value=0.0,
            reason="negative_brier_loss",
            metadata={"brier": 0.0},
        )

    if episode["mode"] != ACTIVE_MODE:
        first_step = episode["passive_plan"][0]
        apply_observation(
            state=state,
            experiment_id=first_step["experiment_id"],
            outcome_id=first_step["outcome_id"],
            observation_text=first_step["observation_text"],
            track_experiment=False,
        )
        for passive_index, passive_step in enumerate(episode["passive_plan"], start=1):
            if passive_index > 1:
                apply_observation(
                    state=state,
                    experiment_id=passive_step["experiment_id"],
                    outcome_id=passive_step["outcome_id"],
                    observation_text=passive_step["observation_text"],
                    track_experiment=False,
                )
            record_exact_belief(stop=passive_index == len(episode["passive_plan"]))
        return {
            "policy": policy,
            "summary": compute_episode_summary(state),
            "trajectory_log": state["trajectory_log"],
        }

    while len(state["used_experiments"]) < TURN_BUDGET:
        if state["used_experiments"] and max(state["current_posterior"].values()) >= 0.85:
            _record_event(
                state,
                kind="extra_turn_penalty",
                value=EXTRA_TURN_PENALTY,
                reason="posterior_already_confident",
                metadata={"posterior_max": max(state["current_posterior"].values())},
            )

        chosen = _choose_experiment(
            episode,
            state["available_experiments"],
            state["current_posterior"],
            policy,
            rng,
        )
        utilities = utility_map_for_state(
            state["current_posterior"],
            episode["hidden"]["likelihoods"],
            state["available_experiments"],
        )
        best = max(
            state["available_experiments"],
            key=lambda experiment_id: (utilities[experiment_id], -EXPERIMENT_IDS.index(experiment_id)),
        )
        regret = float(utilities[best] - utilities[chosen])
        _record_event(
            state,
            kind="experiment_reward",
            value=-regret,
            reason="expected_information_gain_minus_cost",
            metadata={
                "experiment_id": chosen,
                "chosen_utility": float(utilities[chosen]),
                "best_experiment": best,
                "best_utility": float(utilities[best]),
                "regret": regret,
            },
        )
        outcome_id = episode["active_outcomes"][chosen]
        apply_observation(
            state=state,
            experiment_id=chosen,
            outcome_id=outcome_id,
            observation_text=episode["observation_bank"][chosen][outcome_id],
            track_experiment=True,
        )
        state["used_experiments"].append(chosen)
        state["available_experiments"].remove(chosen)
        record_exact_belief(stop=len(state["used_experiments"]) == TURN_BUDGET)

    return {
        "policy": policy,
        "summary": compute_episode_summary(state),
        "trajectory_log": state["trajectory_log"],
    }


def run_heuristic_baselines(
    *,
    split: str,
    max_examples: int,
    seed: int,
    include_lexical: bool = True,
) -> dict:
    rows = _load_rows(split=split, max_examples=max_examples, seed=seed)
    baseline_names = ["random", "cheap_first", "oracle"]
    if include_lexical:
        baseline_names.append("lexical")

    results = {}
    for baseline_name in baseline_names:
        summaries = [
            simulate_policy(row["info"], baseline_name, seed=seed + index)["summary"]
            for index, row in enumerate(rows)
        ]
        results[baseline_name] = aggregate_summaries(summaries)
    return results


def aggregate_summaries(summaries: list[dict]) -> dict:
    if not summaries:
        return {}
    numeric_keys = [
        key
        for key, value in summaries[0].items()
        if isinstance(value, (int, float))
    ]
    aggregate = {
        key: sum(float(summary[key]) for summary in summaries) / len(summaries)
        for key in numeric_keys
    }
    aggregate["num_examples"] = len(summaries)
    return aggregate


def latest_results_path(env_id: str, model: str) -> Path | None:
    runs_dir = ROOT / "environments" / env_id.replace("-", "_") / "outputs" / "evals" / (
        f"{env_id}--{model.replace('/', '--')}"
    )
    if not runs_dir.exists():
        return None
    candidates = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "results.jsonl").exists() and (candidate / "metadata.json").exists():
            return candidate
    return None


def parse_results(results_path: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads((results_path / "metadata.json").read_text(encoding="utf-8"))
    rows = []
    with (results_path / "results.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return metadata, rows


def summarize_results_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    numeric_keys = [
        "reward",
        "mean_experiment_reward",
        "mean_calibration_reward",
        "mean_regret",
        "mean_brier",
        "final_map_correct",
        "extra_turn_penalty_total",
        "invalid_belief_penalty_total",
        "malformed_action_penalty_total",
        "num_turns",
    ]
    summary = {"num_rows": len(rows)}
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and isinstance(row[key], (int, float))
        ]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def run_model_eval(
    *,
    split: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    sampling_args: dict,
    state_columns: list[str],
) -> Path:
    env_args = {"split": split, "max_examples": num_examples, "seed": 0}
    command = [
        "prime",
        "eval",
        "run",
        "research-hypothesis-analysis",
        "--env-dir-path",
        str(ROOT / "environments"),
        "--env-args",
        json.dumps(env_args),
        "--provider",
        "prime",
        "--model",
        model,
        "--num-examples",
        str(num_examples),
        "--rollouts-per-example",
        str(rollouts_per_example),
        "--max-concurrent",
        str(max_concurrent),
        "--sampling-args",
        json.dumps(sampling_args),
        "--state-columns",
        ",".join(state_columns),
        "--save-results",
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    results_path = latest_results_path("research-hypothesis-analysis", model)
    if results_path is None:
        raise FileNotFoundError("prime eval completed but no saved results directory was found")
    return results_path
