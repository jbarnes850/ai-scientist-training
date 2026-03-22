import copy
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel
import verifiers as vf
from verifiers.types import State, ToolMessage

from .bayes import (
    ACTIVE_MODE,
    BELIEF_KEYS,
    EPISODE_VERSION,
    EXPERIMENT_IDS,
    MALFORMED_ACTION_PENALTY,
    OPAQUE_LABELS,
    TURN_BUDGET,
    apply_observation,
    build_alias_map,
    compute_episode_summary,
    invert_alias_map,
    rewrite_prompt_with_aliases,
    utility_map_for_state,
)
from .dataset import build_dataset, ensure_frozen_dataset


SYSTEM_PROMPT = """You are operating inside a synthetic Bayesian research environment.

Your goal is to identify the true hypothesis as efficiently as possible.
Experiments differ in how much they reveal and what they cost.
Choose experiments that resolve uncertainty, not just ones that are cheap or familiar.
After each observation, revise your beliefs to reflect the strength of the evidence.
A small shift in data can sometimes demand a large shift in belief.

Use exactly one tool call per assistant turn.
Do not rely on hidden chain-of-thought. Report beliefs explicitly with the provided tool.
When you see new evidence, update your stated belief proportionally to that evidence."""


class BeliefVector(BaseModel):
    H1: float
    H2: float
    H3: float


def _rewrite_tool_defs_with_aliases(
    tool_defs: list,
    alias_map: dict[str, str],
) -> list:
    """Deep-copy tool_defs and replace run_experiment enum with opaque labels."""
    new_defs = copy.deepcopy(tool_defs)
    for td in new_defs:
        if td.get("name") == "run_experiment":
            props = td.get("parameters", {}).get("properties", {})
            if "experiment_id" in props:
                props["experiment_id"]["enum"] = list(OPAQUE_LABELS)
    return new_defs


class EpistemicTasteEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        split: str = "train",
        eval_split: str | None = None,
        max_examples: int = -1,
        eval_max_examples: int | None = None,
        seed: int = 0,
        trajectory_dump_path: str | None = None,
        ensure_dataset: bool = True,
        **kwargs,
    ):
        if ensure_dataset:
            ensure_frozen_dataset()

        self.split = split
        self.eval_split = eval_split or split
        self.max_examples = max_examples
        self.eval_max_examples = eval_max_examples
        self.seed = seed
        self.trajectory_dump_path = (
            Path(trajectory_dump_path).expanduser() if trajectory_dump_path else None
        )

        rubric = vf.Rubric()
        rubric.add_reward_func(mean_experiment_reward)
        rubric.add_reward_func(mean_calibration_reward)
        rubric.add_reward_func(final_map_bonus)
        rubric.add_reward_func(extra_turn_penalty_total)
        rubric.add_reward_func(invalid_belief_penalty_total)
        rubric.add_reward_func(malformed_action_penalty_total)
        rubric.add_metric(mean_regret)
        rubric.add_metric(mean_brier)
        rubric.add_metric(final_map_correct)
        rubric.add_metric(experiment_turns)
        rubric.add_metric(valid_belief_reports)
        rubric.add_metric(unused_budget)
        rubric.add_metric(contradiction_episode)
        rubric.add_metric(total_reward_reconstructed)

        super().__init__(
            tools=[],
            max_turns=7,
            dataset=lambda: build_dataset(split=split, max_examples=max_examples, seed=seed),
            eval_dataset=lambda: build_dataset(
                split=self.eval_split,
                max_examples=eval_max_examples if eval_max_examples is not None else max_examples,
                seed=seed,
            ),
            system_prompt=SYSTEM_PROMPT,
            rubric=rubric,
            env_id="epistemic-taste",
            env_args={
                "split": split,
                "eval_split": self.eval_split,
                "max_examples": max_examples,
                "eval_max_examples": eval_max_examples,
                "seed": seed,
                "trajectory_dump_path": str(self.trajectory_dump_path)
                if self.trajectory_dump_path
                else None,
            },
            **kwargs,
        )
        self.add_tool(self.run_experiment, args_to_skip=["state"])
        self.add_tool(self.report_belief, args_to_skip=["state"])

    async def setup_state(self, state: State) -> State:
        episode = cast(dict, state["info"])
        state["episode_version"] = EPISODE_VERSION
        state["episode_spec"] = episode
        state["mode"] = episode["mode"]
        state["available_experiments"] = list(EXPERIMENT_IDS)
        state["current_posterior"] = dict(episode["prior"])
        state["posterior_trace"] = [
            {
                "step_index": 0,
                "source": "prior",
                "posterior": dict(episode["prior"]),
            }
        ]
        state["observation_history"] = []
        state["belief_reports"] = []
        state["reward_events"] = []
        state["trajectory_log"] = []
        state["used_experiments"] = []
        state["awaiting_belief"] = False
        state["passive_reveal_index"] = 0
        state["last_valid_belief"] = None
        state["episode_summary"] = {}

        if episode["mode"] != ACTIVE_MODE:
            first_step = cast(dict, episode["passive_plan"][0])
            apply_observation(
                state=state,
                experiment_id=first_step["experiment_id"],
                outcome_id=first_step["outcome_id"],
                observation_text=first_step["observation_text"],
                track_experiment=False,
            )
            state["passive_reveal_index"] = 1
            state["awaiting_belief"] = True
            state["trajectory_log"].append(
                {
                    "event": "passive_initial_observation",
                    "experiment_id": first_step["experiment_id"],
                    "outcome_id": first_step["outcome_id"],
                    "posterior_after": dict(state["current_posterior"]),
                }
            )
        # -- Presentation perturbation: opaque labels + order shuffle --
        trajectory_seed = int(state["trajectory_id"][:16], 16)
        perturb_rng = np.random.default_rng(trajectory_seed)
        alias_map = build_alias_map(perturb_rng)
        state["alias_map"] = alias_map
        state["reverse_alias_map"] = invert_alias_map(alias_map)
        state["prompt"] = rewrite_prompt_with_aliases(state["prompt"], alias_map)
        state["tool_defs"] = _rewrite_tool_defs_with_aliases(
            state["tool_defs"], alias_map
        )

        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: State,
        **kwargs,
    ) -> dict:
        tool_args["state"] = state
        if tool_name == "run_experiment" and "experiment_id" in tool_args:
            alias_map = state.get("alias_map", {})
            raw_id = tool_args["experiment_id"]
            if raw_id in alias_map:
                tool_args["experiment_id"] = alias_map[raw_id]
        return tool_args

    def _append_reward_event(
        self,
        state: State,
        *,
        kind: str,
        value: float,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        event = {
            "kind": kind,
            "value": float(value),
            "reason": reason,
            "metadata": metadata or {},
        }
        state["reward_events"].append(event)
        state["trajectory_log"].append({"event": "reward", **event})

    def _append_tool_extras(self, state: State, extras: dict) -> None:
        if state["trajectory"]:
            state["trajectory"][-1]["extras"].update(extras)

    def _register_malformed_action(
        self,
        state: State,
        *,
        reason: str,
        details: dict | None = None,
    ) -> None:
        self._append_reward_event(
            state,
            kind="malformed_action_penalty",
            value=MALFORMED_ACTION_PENALTY,
            reason=reason,
            metadata=details,
        )

    async def env_response(
        self, messages: vf.Messages, state: State, **kwargs
    ) -> vf.Messages:
        last_msg = cast(vf.AssistantMessage, messages[-1])
        tool_calls = list(last_msg.tool_calls or [])
        tool_messages: list[ToolMessage] = []
        step_extras: dict[str, object] = {"tool_calls_seen": len(tool_calls)}

        if len(tool_calls) > 1:
            for extra_call in tool_calls[1:]:
                self._register_malformed_action(
                    state,
                    reason="multiple_tool_calls",
                    details={"tool_name": extra_call.name},
                )
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content="Only one tool action is allowed per assistant turn.",
                        tool_call_id=extra_call.id,
                    )
                )
            tool_calls = tool_calls[:1]

        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            try:
                tool_name = tool_call.name
                if tool_name not in self.tool_map:
                    raise KeyError(f"Unsupported tool '{tool_name}'")
                parsed_args = json.loads(tool_call.arguments)
                if not isinstance(parsed_args, dict):
                    raise ValueError(
                        "Tool arguments must be a JSON object."
                    )
                tool_args = self.update_tool_args(
                    tool_name, parsed_args, messages, state, **kwargs
                )
                tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
                tool_messages.append(tool_message)
                step_extras["tool_name"] = tool_name
            except Exception as exc:
                self._register_malformed_action(
                    state,
                    reason="tool_parse_or_dispatch_error",
                    details={"error": repr(exc)},
                )
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=f"Invalid action: {exc}",
                        tool_call_id=tool_call_id,
                    )
                )
                step_extras["tool_error"] = repr(exc)

        self._append_tool_extras(state, step_extras)
        return tool_messages

    async def run_experiment(
        self,
        experiment_id: Literal[
            "informative",
            "medium",
            "cheap_weak",
            "redundant",
            "high_cost_informative",
        ],
        state: State,
    ) -> str:
        episode = cast(dict, state["episode_spec"])
        current_posterior = cast(dict[str, float], state["current_posterior"])

        if state["mode"] != ACTIVE_MODE:
            self._register_malformed_action(state, reason="run_experiment_in_passive_mode")
            return "This is a passive episode. `run_experiment` is disabled. Use `report_belief` only."
        if state["awaiting_belief"]:
            self._register_malformed_action(state, reason="experiment_before_belief_report")
            return "You must report your belief after the latest observation before running another experiment."
        if experiment_id not in state["available_experiments"]:
            self._register_malformed_action(
                state,
                reason="experiment_unavailable",
                details={"experiment_id": experiment_id},
            )
            display_id = state.get("reverse_alias_map", {}).get(experiment_id, experiment_id)
            return f"Experiment `{display_id}` is unavailable or already used."

        if state["used_experiments"] and max(current_posterior.values()) >= 0.85:
            self._append_reward_event(
                state,
                kind="extra_turn_penalty",
                value=-0.05,
                reason="posterior_already_confident",
                metadata={"posterior_max": max(current_posterior.values())},
            )

        utility_map = utility_map_for_state(
            current_posterior,
            episode["hidden"]["likelihoods"],
            state["available_experiments"],
        )
        best_experiment = max(
            state["available_experiments"],
            key=lambda exp_id: (utility_map[exp_id], -EXPERIMENT_IDS.index(exp_id)),
        )
        chosen_utility = float(utility_map[experiment_id])
        best_utility = float(utility_map[best_experiment])
        regret = best_utility - chosen_utility
        experiment_reward = -regret

        self._append_reward_event(
            state,
            kind="experiment_reward",
            value=experiment_reward,
            reason="expected_information_gain_minus_cost",
            metadata={
                "experiment_id": experiment_id,
                "chosen_utility": chosen_utility,
                "best_experiment": best_experiment,
                "best_utility": best_utility,
                "regret": regret,
            },
        )

        outcome_id = cast(dict, episode["active_outcomes"])[experiment_id]
        observation_text = cast(dict, episode["observation_bank"])[experiment_id][outcome_id]
        apply_observation(
            state=state,
            experiment_id=experiment_id,
            outcome_id=outcome_id,
            observation_text=observation_text,
            track_experiment=True,
        )
        state["available_experiments"].remove(experiment_id)
        state["used_experiments"].append(experiment_id)
        state["awaiting_belief"] = True

        self._append_tool_extras(
            state,
            {
                "latest_experiment_id": experiment_id,
                "latest_outcome_id": outcome_id,
                "latest_regret": regret,
                "latest_experiment_reward": experiment_reward,
            },
        )
        display_id = state.get("reverse_alias_map", {}).get(experiment_id, experiment_id)
        return (
            f"Observation from `{display_id}`: {observation_text}\n"
            f"Current evidence turns used: {len(state['used_experiments'])}/{TURN_BUDGET}.\n"
            "Now call `report_belief` with your updated probabilities."
        )

    async def report_belief(
        self,
        belief: BeliefVector,
        stop: bool,
        state: State,
    ) -> str:
        if state["mode"] == ACTIVE_MODE and not state["awaiting_belief"]:
            self._register_malformed_action(state, reason="belief_before_observation")
            return "You must run an experiment before reporting a new belief in an active episode."

        belief_payload = (
            belief.model_dump() if isinstance(belief, BeliefVector) else dict(belief)
        )
        validation_error = validate_belief_payload(belief_payload)

        if validation_error is not None:
            self._append_reward_event(
                state,
                kind="invalid_belief_penalty",
                value=-0.5,
                reason=validation_error,
            )
            if stop:
                state["final_env_response"] = [
                    {
                        "role": "user",
                        "content": "Episode complete. No further actions are available.",
                    }
                ]
            return f"Invalid belief vector: {validation_error}"

        exact_posterior = cast(dict[str, float], state["current_posterior"])
        brier = sum(
            (float(belief_payload[key]) - float(exact_posterior[key])) ** 2
            for key in BELIEF_KEYS
        )
        calibration_reward = -brier

        state["belief_reports"].append(
            {
                "belief": {key: float(belief_payload[key]) for key in BELIEF_KEYS},
                "exact_posterior": dict(exact_posterior),
                "stop": bool(stop),
                "brier": float(brier),
                "calibration_reward": float(calibration_reward),
            }
        )
        state["last_valid_belief"] = {
            key: float(belief_payload[key]) for key in BELIEF_KEYS
        }
        self._append_reward_event(
            state,
            kind="calibration_reward",
            value=calibration_reward,
            reason="negative_brier_loss",
            metadata={"brier": brier},
        )
        self._append_tool_extras(
            state,
            {
                "latest_brier": brier,
                "latest_calibration_reward": calibration_reward,
                "stop_requested": bool(stop),
            },
        )

        if stop:
            state["awaiting_belief"] = False
            state["final_env_response"] = [
                {
                    "role": "user",
                    "content": "Episode complete. No further actions are available.",
                }
            ]
            return "Belief recorded. Episode closed."

        if state["mode"] == ACTIVE_MODE:
            state["awaiting_belief"] = False
            if len(state["used_experiments"]) >= TURN_BUDGET:
                state["final_env_response"] = [
                    {
                        "role": "user",
                        "content": "Evidence budget exhausted. Episode complete.",
                    }
                ]
                return "Belief recorded. Evidence budget exhausted."
            reverse_map = state.get("reverse_alias_map", {})
            remaining = ", ".join(
                reverse_map.get(eid, eid) for eid in state["available_experiments"]
            )
            return (
                "Belief recorded. Choose one remaining experiment next.\n"
                f"Remaining experiments: {remaining}."
            )

        passive_plan = cast(list[dict], state["episode_spec"]["passive_plan"])
        if state["passive_reveal_index"] >= len(passive_plan):
            state["final_env_response"] = [
                {
                    "role": "user",
                    "content": "Passive evidence sequence complete. Episode closed.",
                }
            ]
            return "Belief recorded. No further observations remain."

        next_step = passive_plan[state["passive_reveal_index"]]
        apply_observation(
            state=state,
            experiment_id=next_step["experiment_id"],
            outcome_id=next_step["outcome_id"],
            observation_text=next_step["observation_text"],
            track_experiment=False,
        )
        state["passive_reveal_index"] += 1
        state["awaiting_belief"] = True
        return (
            f"Next observation: {next_step['observation_text']}\n"
            "Update your belief with another `report_belief` call."
        )

    @vf.cleanup
    async def finalize_episode(self, state: State) -> None:
        if (
            state.get("stop_condition") == "no_tools_called"
            and state.get("final_env_response") is None
        ):
            self._register_malformed_action(state, reason="assistant_turn_without_tool_call")

        summary = compute_episode_summary(state)
        state["episode_summary"] = summary
        if self.trajectory_dump_path:
            self.trajectory_dump_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "episode_id": state["episode_spec"]["episode_id"],
                "split": state["episode_spec"]["split"],
                "mode": state["mode"],
                "summary": summary,
                "trajectory_log": state["trajectory_log"],
                "posterior_trace": state["posterior_trace"],
            }
            with self.trajectory_dump_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


def validate_belief_payload(
    belief: dict[str, float],
) -> str | None:
    if sorted(belief.keys()) != list(BELIEF_KEYS):
        return "belief keys must be exactly H1, H2, H3"
    try:
        numeric = {key: float(value) for key, value in belief.items()}
    except (TypeError, ValueError):
        return "belief values must be numeric"
    if any(value < 0.0 or value > 1.0 for value in numeric.values()):
        return "belief values must lie in [0, 1]"
    if abs(sum(numeric.values()) - 1.0) > 1e-3:
        return "belief values must sum to 1 within 1e-3"
    return None


def _episode_summary_value(state: State, key: str) -> float:
    summary = state.get("episode_summary") or compute_episode_summary(state)
    return float(summary[key])


async def mean_experiment_reward(state: State) -> float:
    return _episode_summary_value(state, "mean_experiment_reward")


async def mean_calibration_reward(state: State) -> float:
    return _episode_summary_value(state, "mean_calibration_reward")


async def final_map_bonus(state: State) -> float:
    return _episode_summary_value(state, "final_map_bonus")


async def extra_turn_penalty_total(state: State) -> float:
    return _episode_summary_value(state, "extra_turn_penalty_total")


async def invalid_belief_penalty_total(state: State) -> float:
    return _episode_summary_value(state, "invalid_belief_penalty_total")


async def malformed_action_penalty_total(state: State) -> float:
    return _episode_summary_value(state, "malformed_action_penalty_total")


async def mean_regret(state: State) -> float:
    return _episode_summary_value(state, "mean_regret")


async def mean_brier(state: State) -> float:
    return _episode_summary_value(state, "mean_brier")


async def final_map_correct(state: State) -> float:
    return _episode_summary_value(state, "final_map_correct")


async def experiment_turns(state: State) -> float:
    return _episode_summary_value(state, "experiment_turns")


async def valid_belief_reports(state: State) -> float:
    return _episode_summary_value(state, "valid_belief_reports")


async def unused_budget(state: State) -> float:
    return _episode_summary_value(state, "unused_budget")


async def contradiction_episode(state: State) -> float:
    return _episode_summary_value(state, "contradiction_episode")


async def total_reward_reconstructed(state: State) -> float:
    return _episode_summary_value(state, "total_reward_reconstructed")


def load_environment(
    split: str = "train",
    eval_split: str | None = None,
    max_examples: int = -1,
    eval_max_examples: int | None = None,
    seed: int = 0,
    trajectory_dump_path: str | None = None,
    ensure_dataset: bool = True,
    **kwargs,
) -> vf.Environment:
    return EpistemicTasteEnv(
        split=split,
        eval_split=eval_split,
        max_examples=max_examples,
        eval_max_examples=eval_max_examples,
        seed=seed,
        trajectory_dump_path=trajectory_dump_path,
        ensure_dataset=ensure_dataset,
        **kwargs,
    )
