from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from verifiers.types import State

BELIEF_KEYS = ("H1", "H2", "H3")
OUTCOME_IDS = ("O1", "O2", "O3")
EXPERIMENT_IDS = (
    "informative",
    "medium",
    "cheap_weak",
    "redundant",
    "high_cost_informative",
)
ACTIVE_MODE = "active"
PASSIVE_MODE = "passive"
TURN_BUDGET = 3
EPISODE_VERSION = "epistemic-taste-v1"
MALFORMED_ACTION_PENALTY = -0.25
INVALID_BELIEF_PENALTY = -0.5
FINAL_MAP_BONUS = 0.25
EXTRA_TURN_PENALTY = -0.05
PLAUSIBLE_OBSERVATION_THRESHOLD = 0.20

EXPERIMENT_COSTS = {
    "informative": 1.0,
    "medium": 1.0,
    "cheap_weak": 0.4,
    "redundant": 0.8,
    "high_cost_informative": 1.6,
}

EXPERIMENT_TYPES = {
    "informative": "informative",
    "medium": "medium",
    "cheap_weak": "cheap_weak",
    "redundant": "redundant",
    "high_cost_informative": "high_cost_informative",
}

DOMAIN_LIBRARY = {
    "cell_signaling": {
        "display": "cell signaling",
        "question": "Which mechanism best explains why receptor X produces a transient response after perturbation P?",
        "hypotheses": {
            "H1": "a thresholded feed-forward pulse model",
            "H2": "rapid receptor desensitization after activation",
            "H3": "a delayed negative-feedback loop that shuts the pathway down",
        },
        "experiments": {
            "informative": "Run a high-time-resolution phospho-signaling assay after perturbation P.",
            "medium": "Measure paired phospho and transcript snapshots at an intermediate cadence.",
            "cheap_weak": "Read out one low-cost proxy marker from the pathway's downstream target panel.",
            "redundant": "Repeat a compressed version of the main pathway readout with a slightly different antibody panel.",
            "high_cost_informative": "Use a high-cost single-cell time-course with perturbation tracking.",
        },
        "outcomes": {
            "informative": [
                "The assay shows a sharp early spike that rapidly resets.",
                "The response rises strongly and then collapses as receptor availability drops.",
                "The response peaks later and is followed by a broad suppression phase.",
            ],
            "medium": [
                "The paired snapshot suggests an early phospho pulse with weak downstream persistence.",
                "The paired snapshot suggests activation followed by fast receptor exhaustion.",
                "The paired snapshot suggests delayed shutdown after an initially sustained response.",
            ],
            "cheap_weak": [
                "The proxy marker is only slightly elevated above baseline.",
                "The proxy marker rises modestly and decays quickly.",
                "The proxy marker remains mildly elevated into the late window.",
            ],
            "redundant": [
                "The compressed assay again favors an early pulse-like profile.",
                "The compressed assay again favors a desensitization-like decline.",
                "The compressed assay again favors delayed suppression.",
            ],
            "high_cost_informative": [
                "Single-cell traces separate into a brief synchronized pulse before reset.",
                "Single-cell traces show activation followed by fast receptor-state depletion.",
                "Single-cell traces show delayed feedback-mediated shutdown after initial activation.",
            ],
        },
    },
    "ecology": {
        "display": "ecology",
        "question": "Which mechanism best explains the observed collapse-and-recovery dynamics in the marsh food web?",
        "hypotheses": {
            "H1": "a threshold predator pulse that briefly suppresses grazers",
            "H2": "rapid resource depletion after an initial bloom",
            "H3": "delayed density-dependent feedback from the nursery habitat",
        },
        "experiments": {
            "informative": "Run a fine-grained predator exclusion and resource sampling campaign across the disturbance window.",
            "medium": "Collect paired grazer-count and nutrient snapshots at moderate cadence.",
            "cheap_weak": "Use one inexpensive proxy survey from the historical monitoring station.",
            "redundant": "Repeat the standard exclusion assay with nearly identical marsh transects.",
            "high_cost_informative": "Deploy a tagged high-resolution field array across the entire estuary segment.",
        },
        "outcomes": {
            "informative": [
                "Counts show a brief synchronized predator pulse followed by rapid normalization.",
                "Resource measures spike and then crash as the bloom exhausts the local base.",
                "The late window reveals a lagged nursery feedback that suppresses recovery.",
            ],
            "medium": [
                "The moderate survey favors a short predator pulse with limited persistence.",
                "The moderate survey favors bloom-driven depletion after the initial rise.",
                "The moderate survey favors delayed habitat feedback.",
            ],
            "cheap_weak": [
                "The proxy station shows only a faint anomaly.",
                "The proxy station shows a mild bump that fades quickly.",
                "The proxy station shows a mild late-stage drag on recovery.",
            ],
            "redundant": [
                "The repeated assay again points to a short predator pulse.",
                "The repeated assay again points to rapid resource depletion.",
                "The repeated assay again points to delayed habitat feedback.",
            ],
            "high_cost_informative": [
                "The field array isolates a coherent predator pulse before recovery resumes.",
                "The field array isolates depletion-driven collapse after the bloom front passes.",
                "The field array isolates delayed nursery feedback as the late constraint.",
            ],
        },
    },
    "materials": {
        "display": "materials",
        "question": "Which mechanism best explains the unstable conductivity profile in the thin-film stack?",
        "hypotheses": {
            "H1": "a thresholded transient alignment of conductive domains",
            "H2": "contact degradation after the initial current surge",
            "H3": "delayed defect-mediated feedback from the interface layer",
        },
        "experiments": {
            "informative": "Run a high-time-resolution bias-sweep with in situ structural readout.",
            "medium": "Measure paired conductivity and interface signatures at moderate cadence.",
            "cheap_weak": "Use a low-cost proxy read from the standard wafer monitor.",
            "redundant": "Repeat the baseline conductivity scan with a nearly identical probe geometry.",
            "high_cost_informative": "Run a high-cost microscopy-plus-bias sequence on the same stack.",
        },
        "outcomes": {
            "informative": [
                "The stack shows a brief conductive alignment that quickly relaxes.",
                "Conductivity rises early and then falls as contacts degrade.",
                "The interface remains stable early but collapses later as defects accumulate.",
            ],
            "medium": [
                "The moderate readout favors a short-lived alignment event.",
                "The moderate readout favors contact degradation after the initial surge.",
                "The moderate readout favors delayed defect feedback.",
            ],
            "cheap_weak": [
                "The wafer proxy barely shifts from baseline.",
                "The wafer proxy shows a modest early rise and quick fade.",
                "The wafer proxy shows a mild late-stage degradation signature.",
            ],
            "redundant": [
                "The repeated scan again favors a transient alignment story.",
                "The repeated scan again favors contact degradation.",
                "The repeated scan again favors delayed defect feedback.",
            ],
            "high_cost_informative": [
                "Microscopy resolves a synchronized but brief domain alignment.",
                "Microscopy resolves early conduction followed by contact-state collapse.",
                "Microscopy resolves delayed interface damage consistent with defect feedback.",
            ],
        },
    },
    "public_health": {
        "display": "public health",
        "question": "Which mechanism best explains the short-lived uptake of the community intervention?",
        "hypotheses": {
            "H1": "a threshold outreach pulse that briefly changes behavior",
            "H2": "initial uptake followed by fast operational attrition",
            "H3": "delayed trust backlash after the first intervention wave",
        },
        "experiments": {
            "informative": "Run a fine-grained follow-up survey linked to delivery and attendance records.",
            "medium": "Collect paired attendance and sentiment snapshots at moderate cadence.",
            "cheap_weak": "Use one inexpensive proxy check from a routine administrative dashboard.",
            "redundant": "Repeat the standard attendance audit with almost the same outreach sites.",
            "high_cost_informative": "Run a high-cost mixed-methods panel with participant-level follow-up.",
        },
        "outcomes": {
            "informative": [
                "Follow-up records show a brief outreach pulse with quick reversion.",
                "Follow-up records show early uptake followed by fast delivery attrition.",
                "Follow-up records show delayed trust erosion after the initial wave.",
            ],
            "medium": [
                "The moderate snapshot favors a short-lived outreach pulse.",
                "The moderate snapshot favors operational attrition after early uptake.",
                "The moderate snapshot favors delayed trust backlash.",
            ],
            "cheap_weak": [
                "The dashboard proxy is barely above baseline.",
                "The dashboard proxy rises modestly and then decays.",
                "The dashboard proxy stays slightly depressed in the late period.",
            ],
            "redundant": [
                "The repeated audit again favors a pulse-like intervention effect.",
                "The repeated audit again favors operational attrition.",
                "The repeated audit again favors delayed trust backlash.",
            ],
            "high_cost_informative": [
                "The panel isolates a brief behavior pulse before reversion.",
                "The panel isolates early uptake followed by fast operational failure.",
                "The panel isolates delayed trust backlash as the late driver.",
            ],
        },
    },
    "econ_policy": {
        "display": "econ policy",
        "question": "Which mechanism best explains the brief improvement and later fadeout in the policy response?",
        "hypotheses": {
            "H1": "a threshold expectation pulse that briefly changes firm behavior",
            "H2": "initial policy uptake followed by rapid implementation friction",
            "H3": "delayed general-equilibrium feedback that reverses the early gain",
        },
        "experiments": {
            "informative": "Run a high-frequency linked firm-and-price panel across the intervention window.",
            "medium": "Collect paired compliance and price snapshots at moderate cadence.",
            "cheap_weak": "Use one inexpensive proxy indicator from the routine macro dashboard.",
            "redundant": "Repeat the standard compliance audit with nearly identical reporting units.",
            "high_cost_informative": "Run a high-cost matched microdata panel with firm-level tracking.",
        },
        "outcomes": {
            "informative": [
                "The linked panel shows a brief expectation pulse before normalization.",
                "The linked panel shows early uptake followed by fast implementation friction.",
                "The linked panel shows delayed equilibrium feedback that erodes the initial gain.",
            ],
            "medium": [
                "The moderate panel favors a short-lived expectation pulse.",
                "The moderate panel favors implementation friction after early uptake.",
                "The moderate panel favors delayed equilibrium feedback.",
            ],
            "cheap_weak": [
                "The dashboard proxy barely moves.",
                "The dashboard proxy rises modestly and then snaps back.",
                "The dashboard proxy drifts down only in the late window.",
            ],
            "redundant": [
                "The repeated audit again favors a brief expectation pulse.",
                "The repeated audit again favors implementation friction.",
                "The repeated audit again favors delayed equilibrium feedback.",
            ],
            "high_cost_informative": [
                "The matched microdata isolate a brief expectation-driven shift before normalization.",
                "The matched microdata isolate implementation friction after the initial response.",
                "The matched microdata isolate delayed equilibrium feedback as the late driver.",
            ],
        },
    },
}

WORDING_BANK = {
    0: {
        "header": "Research question",
        "hypotheses": "Candidate hypotheses",
        "prior": "Current prior",
        "experiments": "Candidate experiments",
        "budget": "Evidence budget",
        "experiment_line": "- {exp_id}: {description} Cost={cost:.1f}.",
    },
    1: {
        "header": "Open question",
        "hypotheses": "Hypothesis set",
        "prior": "Prior belief state",
        "experiments": "Available studies",
        "budget": "Remaining experiment turns",
        "experiment_line": "- {exp_id}: {description} Estimated cost {cost:.1f}.",
    },
    2: {
        "header": "Decision target",
        "hypotheses": "Working hypotheses",
        "prior": "Bayesian prior",
        "experiments": "Allowed next experiments",
        "budget": "Turn budget",
        "experiment_line": "- {exp_id}: {description} Budget weight {cost:.1f}.",
    },
    3: {
        "header": "Question under study",
        "hypotheses": "Competing explanations",
        "prior": "Starting belief",
        "experiments": "Evidence options",
        "budget": "Maximum experiment turns",
        "experiment_line": "- {exp_id}: {description} Cost score {cost:.1f}.",
    },
    4: {
        "header": "Mechanism question",
        "hypotheses": "Mechanistic candidates",
        "prior": "Initial posterior guess",
        "experiments": "Permitted experiments",
        "budget": "Experiment budget",
        "experiment_line": "- {exp_id}: {description} Cost index {cost:.1f}.",
    },
    5: {
        "header": "Investigative target",
        "hypotheses": "Named hypotheses",
        "prior": "Belief prior",
        "experiments": "Study menu",
        "budget": "Budgeted turns",
        "experiment_line": "- {exp_id}: {description} Cost unit {cost:.1f}.",
    },
    6: {
        "header": "Target question",
        "hypotheses": "Possible explanations",
        "prior": "Prior over explanations",
        "experiments": "Experiment menu",
        "budget": "Evidence-turn budget",
        "experiment_line": "- {exp_id}: {description} Estimated burden {cost:.1f}.",
    },
    7: {
        "header": "Mechanism to resolve",
        "hypotheses": "Explanation candidates",
        "prior": "Current hypothesis weights",
        "experiments": "Candidate probes",
        "budget": "Probe budget",
        "experiment_line": "- {exp_id}: {description} Cost burden {cost:.1f}.",
    },
    8: {
        "header": "Question to disambiguate",
        "hypotheses": "Alternative accounts",
        "prior": "Prior mass by hypothesis",
        "experiments": "Experimental options",
        "budget": "Maximum evidence turns",
        "experiment_line": "- {exp_id}: {description} Estimated cost weight {cost:.1f}.",
    },
    9: {
        "header": "Unresolved question",
        "hypotheses": "Competing accounts",
        "prior": "Starting hypothesis distribution",
        "experiments": "Available evidence actions",
        "budget": "Total experiment turns",
        "experiment_line": "- {exp_id}: {description} Cost coefficient {cost:.1f}.",
    },
}


@dataclass(frozen=True)
class SplitSpec:
    name: str
    count: int
    domains: tuple[str, ...]
    template_ids: tuple[int, ...]
    prior_map_range: tuple[float, float]
    likelihood_multiplier_range: tuple[float, float]
    contradiction_rate: float


def format_distribution(dist: dict[str, float]) -> str:
    return ", ".join(f"{key}={dist[key]:.3f}" for key in BELIEF_KEYS)


def normalize(values: Iterable[float]) -> list[float]:
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("distribution has non-positive mass")
    return [float(value) / total for value in values]


def entropy(dist: dict[str, float]) -> float:
    total = 0.0
    for value in dist.values():
        if value > 0.0:
            total -= float(value) * math.log(float(value))
    return total


def posterior_update(
    prior: dict[str, float],
    likelihood_table: dict[str, list[float]],
    outcome_id: str,
) -> dict[str, float]:
    outcome_index = OUTCOME_IDS.index(outcome_id)
    posterior_unnormalized = {
        hypothesis_id: float(prior[hypothesis_id]) * float(likelihood_table[hypothesis_id][outcome_index])
        for hypothesis_id in BELIEF_KEYS
    }
    normalized = normalize(posterior_unnormalized.values())
    return {
        hypothesis_id: normalized[index] for index, hypothesis_id in enumerate(BELIEF_KEYS)
    }


def exact_brier(belief: dict[str, float], posterior: dict[str, float]) -> float:
    return sum(
        (float(belief[hypothesis_id]) - float(posterior[hypothesis_id])) ** 2
        for hypothesis_id in BELIEF_KEYS
    )


def _kl_divergence(left: list[float], right: list[float]) -> float:
    total = 0.0
    epsilon = 1e-12
    for left_value, right_value in zip(left, right, strict=True):
        left_adj = max(float(left_value), epsilon)
        right_adj = max(float(right_value), epsilon)
        total += left_adj * math.log(left_adj / right_adj)
    return total


def symmetric_kl(left: list[float], right: list[float]) -> float:
    return 0.5 * (_kl_divergence(left, right) + _kl_divergence(right, left))


def table_to_dict(table: list[list[float]]) -> dict[str, list[float]]:
    return {
        hypothesis_id: [float(value) for value in table[index]]
        for index, hypothesis_id in enumerate(BELIEF_KEYS)
    }


def utility_map_for_state(
    prior: dict[str, float],
    likelihoods: dict[str, dict[str, list[float]]],
    available_experiments: Iterable[str],
) -> dict[str, float]:
    current_entropy = entropy(prior)
    utilities: dict[str, float] = {}
    for experiment_id in available_experiments:
        likelihood_table = likelihoods[experiment_id]
        predictive = []
        for outcome_index, outcome_id in enumerate(OUTCOME_IDS):
            predictive_mass = sum(
                float(prior[hypothesis_id]) * float(likelihood_table[hypothesis_id][outcome_index])
                for hypothesis_id in BELIEF_KEYS
            )
            predictive.append((outcome_id, predictive_mass))
        expected_gain = 0.0
        for outcome_id, probability in predictive:
            if probability <= 0.0:
                continue
            posterior = posterior_update(prior, likelihood_table, outcome_id)
            expected_gain += probability * (current_entropy - entropy(posterior))
        utilities[experiment_id] = expected_gain - 0.15 * EXPERIMENT_COSTS[experiment_id]
    return utilities


def posterior_argmax(dist: dict[str, float]) -> str:
    return max(BELIEF_KEYS, key=lambda key: (float(dist[key]), -BELIEF_KEYS.index(key)))


def _sample_prior(
    rng: np.random.Generator,
    prior_map_range: tuple[float, float],
) -> dict[str, float]:
    low, high = prior_map_range
    for _ in range(2048):
        preferred = int(rng.integers(0, len(BELIEF_KEYS)))
        dominant_mass = float(rng.uniform(low, high))
        split = rng.dirichlet(np.array([2.0, 2.0]))
        probs = [0.0, 0.0, 0.0]
        probs[preferred] = dominant_mass
        others = [index for index in range(len(BELIEF_KEYS)) if index != preferred]
        remainder = 1.0 - dominant_mass
        probs[others[0]] = remainder * float(split[0])
        probs[others[1]] = remainder * float(split[1])
        if max(probs) != probs[preferred]:
            continue
        if min(probs) < 0.05:
            continue
        return {
            hypothesis_id: float(probs[index])
            for index, hypothesis_id in enumerate(BELIEF_KEYS)
        }
    raise RuntimeError("failed to sample prior")


def _sample_rows_from_prototypes(
    rng: np.random.Generator,
    prototypes: list[list[float]],
    concentration: float,
) -> list[list[float]]:
    return [
        [float(value) for value in rng.dirichlet(concentration * np.array(prototype))]
        for prototype in prototypes
    ]


def _pairwise_skl_stats(rows: list[list[float]]) -> tuple[float, float]:
    scores = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            scores.append(symmetric_kl(rows[left], rows[right]))
    return min(scores), max(scores)


def _sample_likelihoods(
    rng: np.random.Generator,
    likelihood_multiplier: float,
) -> tuple[dict[str, dict[str, list[float]]], str]:
    informative_conc = 25.0 * likelihood_multiplier
    medium_conc = 30.0 * likelihood_multiplier
    weak_conc = 120.0 * likelihood_multiplier
    redundant_conc = 150.0 * likelihood_multiplier
    high_cost_conc = 28.0 * likelihood_multiplier

    informative_prototypes = [
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.10, 0.80],
    ]
    medium_prototypes = [
        [0.65, 0.25, 0.10],
        [0.40, 0.45, 0.15],
        [0.20, 0.60, 0.20],
    ]
    high_cost_prototypes = [
        [0.72, 0.18, 0.10],
        [0.10, 0.72, 0.18],
        [0.18, 0.10, 0.72],
    ]

    informative_rows = None
    for _ in range(2048):
        candidate = _sample_rows_from_prototypes(rng, informative_prototypes, informative_conc)
        min_skl, _ = _pairwise_skl_stats(candidate)
        if min_skl >= 0.35:
            informative_rows = candidate
            break
    if informative_rows is None:
        raise RuntimeError("failed to sample informative likelihoods")

    medium_rows = None
    for _ in range(2048):
        candidate = _sample_rows_from_prototypes(rng, medium_prototypes, medium_conc)
        permutation = list(rng.permutation(len(OUTCOME_IDS)))
        candidate = [[row[index] for index in permutation] for row in candidate]
        min_skl, max_skl = _pairwise_skl_stats(candidate)
        if 0.12 <= min_skl <= 0.30 and max_skl <= 0.45:
            medium_rows = candidate
            break
    if medium_rows is None:
        raise RuntimeError("failed to sample medium likelihoods")

    weak_rows = None
    for _ in range(2048):
        shared = rng.dirichlet(np.array([6.0, 6.0, 6.0]))
        candidate = [
            [float(value) for value in rng.dirichlet(weak_conc * shared)]
            for _ in BELIEF_KEYS
        ]
        _, max_skl = _pairwise_skl_stats(candidate)
        if max_skl <= 0.03:
            weak_rows = candidate
            break
    if weak_rows is None:
        raise RuntimeError("failed to sample cheap weak likelihoods")

    parent_id = "informative" if bool(rng.integers(0, 2)) else "medium"
    parent_rows = informative_rows if parent_id == "informative" else medium_rows
    redundant_rows = None
    for _ in range(2048):
        candidate = [
            [
                float(value)
                for value in rng.dirichlet(
                    redundant_conc * np.array(parent_rows[hypothesis_index])
                )
            ]
            for hypothesis_index in range(len(BELIEF_KEYS))
        ]
        mean_skl = float(
            np.mean(
                [
                    symmetric_kl(candidate[hypothesis_index], parent_rows[hypothesis_index])
                    for hypothesis_index in range(len(BELIEF_KEYS))
                ]
            )
        )
        if mean_skl <= 0.02:
            redundant_rows = candidate
            break
    if redundant_rows is None:
        raise RuntimeError("failed to sample redundant likelihoods")

    high_cost_rows = None
    for _ in range(2048):
        candidate = _sample_rows_from_prototypes(
            rng, high_cost_prototypes, high_cost_conc
        )
        min_skl, _ = _pairwise_skl_stats(candidate)
        if 0.20 <= min_skl <= 0.35:
            high_cost_rows = candidate
            break
    if high_cost_rows is None:
        raise RuntimeError("failed to sample high cost informative likelihoods")

    return (
        {
            "informative": table_to_dict(informative_rows),
            "medium": table_to_dict(medium_rows),
            "cheap_weak": table_to_dict(weak_rows),
            "redundant": table_to_dict(redundant_rows),
            "high_cost_informative": table_to_dict(high_cost_rows),
        },
        parent_id,
    )


def _contradiction_metadata(
    prior: dict[str, float],
    true_hypothesis: str,
    likelihoods: dict[str, dict[str, list[float]]],
) -> dict:
    prior_map = posterior_argmax(prior)
    witness = None
    prior_map_mass = float(prior[prior_map])
    for experiment_id in EXPERIMENT_IDS:
        likelihood_table = likelihoods[experiment_id]
        true_row = likelihood_table[true_hypothesis]
        for outcome_id, true_prob in zip(OUTCOME_IDS, true_row, strict=True):
            if float(true_prob) < PLAUSIBLE_OBSERVATION_THRESHOLD:
                continue
            posterior = posterior_update(prior, likelihood_table, outcome_id)
            drop = prior_map_mass - float(posterior[prior_map])
            if drop >= 0.20:
                witness = {
                    "experiment_id": experiment_id,
                    "outcome_id": outcome_id,
                    "prior_map": prior_map,
                    "prior_map_mass_before": prior_map_mass,
                    "prior_map_mass_after": float(posterior[prior_map]),
                    "mass_drop": drop,
                    "plausible_under_true": float(true_prob),
                }
                break
        if witness is not None:
            break
    return {
        "is_contradiction_focused": witness is not None,
        "witness": witness,
    }


def _sample_outcome(
    rng: np.random.Generator,
    likelihood_table: dict[str, list[float]],
    true_hypothesis: str,
) -> str:
    probabilities = np.array(likelihood_table[true_hypothesis], dtype=float)
    outcome_index = int(rng.choice(len(OUTCOME_IDS), p=probabilities))
    return OUTCOME_IDS[outcome_index]


def _render_prompt(
    *,
    domain_key: str,
    template_id: int,
    mode: str,
    hypotheses: dict[str, str],
    prior: dict[str, float],
    experiment_descriptions: dict[str, str],
    passive_initial_observation: str | None,
) -> list[dict[str, str]]:
    wording = WORDING_BANK[template_id]
    domain = DOMAIN_LIBRARY[domain_key]

    hypothesis_lines = "\n".join(
        f"- {hypothesis_id}: {hypotheses[hypothesis_id]}"
        for hypothesis_id in BELIEF_KEYS
    )
    experiment_lines = "\n".join(
        wording["experiment_line"].format(
            exp_id=experiment_id,
            description=experiment_descriptions[experiment_id],
            cost=EXPERIMENT_COSTS[experiment_id],
        )
        for experiment_id in EXPERIMENT_IDS
    )
    protocol = (
        "Active episode. Alternate exactly one `run_experiment` call with one `report_belief` call. "
        "You may stop early by setting `stop=true` in `report_belief`."
        if mode == ACTIVE_MODE
        else "Passive episode. No experiments can be run. Use only `report_belief`; each non-final report reveals the next fixed observation."
    )
    observation_block = ""
    if passive_initial_observation is not None:
        observation_block = f"\nInitial revealed observation:\n- {passive_initial_observation}\n"

    prompt = (
        f"{wording['header']}: {domain['question']}\n\n"
        f"{wording['hypotheses']}:\n{hypothesis_lines}\n\n"
        f"{wording['prior']}: {format_distribution(prior)}\n\n"
        f"{wording['experiments']}:\n{experiment_lines}\n\n"
        f"{wording['budget']}: {TURN_BUDGET}\n"
        f"Protocol: {protocol}\n"
        "Belief reports must use probabilities over H1, H2, H3 that sum to 1."
        f"{observation_block}"
    )
    return [{"role": "user", "content": prompt}]


def build_episode(
    *,
    split_spec: SplitSpec,
    episode_index: int,
    episode_seed: int,
    mode: str,
    contradiction_target: bool,
    domain_key: str,
    template_id: int,
) -> dict:
    rng = np.random.default_rng(episode_seed)
    domain = DOMAIN_LIBRARY[domain_key]
    for _ in range(4096):
        prior = _sample_prior(rng, split_spec.prior_map_range)
        likelihood_multiplier = float(
            rng.uniform(*split_spec.likelihood_multiplier_range)
        )
        likelihoods, redundant_parent = _sample_likelihoods(rng, likelihood_multiplier)
        true_hypothesis = BELIEF_KEYS[int(rng.integers(0, len(BELIEF_KEYS)))]
        contradiction_metadata = _contradiction_metadata(
            prior=prior,
            true_hypothesis=true_hypothesis,
            likelihoods=likelihoods,
        )
        if contradiction_metadata["is_contradiction_focused"] != contradiction_target:
            continue

        experiment_descriptions = {
            experiment_id: domain["experiments"][experiment_id]
            for experiment_id in EXPERIMENT_IDS
        }
        observation_bank = {
            experiment_id: {
                outcome_id: domain["outcomes"][experiment_id][outcome_index]
                for outcome_index, outcome_id in enumerate(OUTCOME_IDS)
            }
            for experiment_id in EXPERIMENT_IDS
        }
        active_outcomes = {
            experiment_id: _sample_outcome(
                rng, likelihoods[experiment_id], true_hypothesis
            )
            for experiment_id in EXPERIMENT_IDS
        }

        reference_plan = []
        current_posterior = dict(prior)
        remaining = list(EXPERIMENT_IDS)
        reference_trace = [{"step_index": 0, "source": "prior", "posterior": dict(prior)}]
        for step_index in range(1, TURN_BUDGET + 1):
            utilities = utility_map_for_state(current_posterior, likelihoods, remaining)
            chosen_experiment = max(
                remaining,
                key=lambda experiment_id: (
                    utilities[experiment_id],
                    -EXPERIMENT_IDS.index(experiment_id),
                ),
            )
            chosen_outcome = active_outcomes[chosen_experiment]
            observation_text = observation_bank[chosen_experiment][chosen_outcome]
            posterior_after = posterior_update(
                current_posterior, likelihoods[chosen_experiment], chosen_outcome
            )
            reference_plan.append(
                {
                    "step_index": step_index,
                    "experiment_id": chosen_experiment,
                    "outcome_id": chosen_outcome,
                    "observation_text": observation_text,
                    "utility": float(utilities[chosen_experiment]),
                    "posterior_after": dict(posterior_after),
                }
            )
            reference_trace.append(
                {
                    "step_index": step_index,
                    "source": chosen_experiment,
                    "posterior": dict(posterior_after),
                }
            )
            current_posterior = posterior_after
            remaining.remove(chosen_experiment)

        passive_initial_observation = (
            reference_plan[0]["observation_text"] if mode == PASSIVE_MODE else None
        )
        prompt = _render_prompt(
            domain_key=domain_key,
            template_id=template_id,
            mode=mode,
            hypotheses=domain["hypotheses"],
            prior=prior,
            experiment_descriptions=experiment_descriptions,
            passive_initial_observation=passive_initial_observation,
        )
        info = {
            "episode_version": EPISODE_VERSION,
            "episode_id": f"{split_spec.name}-{episode_index:05d}",
            "split": split_spec.name,
            "mode": mode,
            "domain": domain["display"],
            "domain_key": domain_key,
            "template_id": template_id,
            "question": domain["question"],
            "hypotheses": domain["hypotheses"],
            "prior": prior,
            "visible_experiments": [
                {
                    "experiment_id": experiment_id,
                    "type": EXPERIMENT_TYPES[experiment_id],
                    "description": experiment_descriptions[experiment_id],
                    "cost": EXPERIMENT_COSTS[experiment_id],
                }
                for experiment_id in EXPERIMENT_IDS
            ],
            "passive_plan": reference_plan,
            "passive_initial_observation": passive_initial_observation,
            "active_outcomes": active_outcomes,
            "observation_bank": observation_bank,
            "reference_trace": reference_trace,
            "contradiction_metadata": contradiction_metadata,
            "hidden": {
                "true_hypothesis": true_hypothesis,
                "likelihoods": likelihoods,
                "likelihood_multiplier": likelihood_multiplier,
                "redundant_parent": redundant_parent,
                "exact_posterior_trace": reference_trace,
            },
        }
        return {
            "prompt": prompt,
            "info": info,
        }
    raise RuntimeError("failed to build episode with requested contradiction target")


def apply_observation(
    *,
    state: State,
    experiment_id: str,
    outcome_id: str,
    observation_text: str,
    track_experiment: bool,
) -> None:
    episode = state["episode_spec"]
    likelihoods = episode["hidden"]["likelihoods"]
    prior_before = dict(state["current_posterior"])
    posterior_after = posterior_update(prior_before, likelihoods[experiment_id], outcome_id)
    state["current_posterior"] = posterior_after
    step_index = len(state["posterior_trace"])
    state["posterior_trace"].append(
        {
            "step_index": step_index,
            "source": experiment_id,
            "posterior": dict(posterior_after),
        }
    )
    state["observation_history"].append(
        {
            "step_index": step_index,
            "experiment_id": experiment_id,
            "outcome_id": outcome_id,
            "observation_text": observation_text,
            "posterior_before": prior_before,
            "posterior_after": dict(posterior_after),
        }
    )
    state["trajectory_log"].append(
        {
            "event": "observation",
            "step_index": step_index,
            "experiment_id": experiment_id,
            "outcome_id": outcome_id,
            "observation_text": observation_text,
            "posterior_before": prior_before,
            "posterior_after": dict(posterior_after),
            "track_experiment": track_experiment,
        }
    )


def compute_episode_summary(state: State) -> dict[str, float | int | str]:
    reward_events = list(state.get("reward_events", []))
    experiment_rewards = [
        float(event["value"])
        for event in reward_events
        if event["kind"] == "experiment_reward"
    ]
    regrets = [
        float(event["metadata"]["regret"])
        for event in reward_events
        if event["kind"] == "experiment_reward"
    ]
    calibration_rewards = [
        float(event["value"])
        for event in reward_events
        if event["kind"] == "calibration_reward"
    ]
    briers = [
        float(event["metadata"]["brier"])
        for event in reward_events
        if event["kind"] == "calibration_reward"
    ]
    extra_turn_penalty_total = sum(
        float(event["value"])
        for event in reward_events
        if event["kind"] == "extra_turn_penalty"
    )
    invalid_belief_penalty_total = sum(
        float(event["value"])
        for event in reward_events
        if event["kind"] == "invalid_belief_penalty"
    )
    malformed_action_penalty_total = sum(
        float(event["value"])
        for event in reward_events
        if event["kind"] == "malformed_action_penalty"
    )
    last_valid_belief = state.get("last_valid_belief")
    true_hypothesis = state["episode_spec"]["hidden"]["true_hypothesis"]
    final_map_correct = 0.0
    if last_valid_belief is not None and posterior_argmax(last_valid_belief) == true_hypothesis:
        final_map_correct = 1.0
    final_map_bonus = FINAL_MAP_BONUS * final_map_correct
    mean_experiment_reward = (
        float(np.mean(experiment_rewards)) if experiment_rewards else 0.0
    )
    mean_calibration_reward = (
        float(np.mean(calibration_rewards)) if calibration_rewards else 0.0
    )
    summary = {
        "mode": state["mode"],
        "termination_reason": state.get("stop_condition"),
        "true_hypothesis": true_hypothesis,
        "final_reported_map": posterior_argmax(last_valid_belief)
        if last_valid_belief is not None
        else "NONE",
        "mean_experiment_reward": mean_experiment_reward,
        "mean_calibration_reward": mean_calibration_reward,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mean_brier": float(np.mean(briers)) if briers else 0.0,
        "final_map_bonus": final_map_bonus,
        "extra_turn_penalty_total": extra_turn_penalty_total,
        "invalid_belief_penalty_total": invalid_belief_penalty_total,
        "malformed_action_penalty_total": malformed_action_penalty_total,
        "final_map_correct": final_map_correct,
        "experiment_turns": len(state.get("used_experiments", [])),
        "valid_belief_reports": len(state.get("belief_reports", [])),
        "unused_budget": max(0, TURN_BUDGET - len(state.get("used_experiments", []))),
        "contradiction_episode": float(
            bool(state["episode_spec"]["contradiction_metadata"]["is_contradiction_focused"])
        ),
    }
    summary["total_reward_reconstructed"] = (
        summary["mean_experiment_reward"]
        + summary["mean_calibration_reward"]
        + summary["final_map_bonus"]
        + summary["extra_turn_penalty_total"]
        + summary["invalid_belief_penalty_total"]
        + summary["malformed_action_penalty_total"]
    )
    return summary
