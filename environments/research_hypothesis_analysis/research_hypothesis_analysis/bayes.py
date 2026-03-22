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
EPISODE_VERSION = "research-hypothesis-analysis-v1"
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

OPAQUE_LABELS = (
    "Experiment_A",
    "Experiment_B",
    "Experiment_C",
    "Experiment_D",
    "Experiment_E",
)

OBSERVATION_PREFIXES = {
    "informative": (
        "Primary readout",
        "Main assay result",
        "High-signal finding",
        "Most diagnostic observation",
    ),
    "medium": (
        "Intermediate readout",
        "Mid-resolution finding",
        "Moderate evidence summary",
        "Paired snapshot result",
    ),
    "cheap_weak": (
        "Low-cost proxy",
        "Quick proxy check",
        "Cheap signal read",
        "Weak indicator",
    ),
    "redundant": (
        "Repeat assay",
        "Redundant check",
        "Replication readout",
        "Follow-up repeat",
    ),
    "high_cost_informative": (
        "High-cost panel",
        "Deep profiling result",
        "Comprehensive readout",
        "Premium assay finding",
    ),
}

OBSERVATION_TEMPLATES = (
    "{prefix}: {core}",
    "{prefix} for the {domain} setting: {core}",
    "{prefix}. {core}",
    "Observed in the {domain} setting, {core_lc}",
)


def build_alias_map(rng: np.random.Generator) -> dict[str, str]:
    """Return {opaque_label: canonical_id} with shuffled assignment."""
    canonical = list(EXPERIMENT_IDS)
    rng.shuffle(canonical)
    return dict(zip(OPAQUE_LABELS, canonical))


def invert_alias_map(alias_map: dict[str, str]) -> dict[str, str]:
    """Return {canonical_id: opaque_label}."""
    return {v: k for k, v in alias_map.items()}


def build_hypothesis_display_map(rng: np.random.Generator) -> dict[str, str]:
    """Return {visible_label: canonical_hypothesis_id} with shuffled assignment."""
    canonical = list(BELIEF_KEYS)
    rng.shuffle(canonical)
    return {
        visible_label: canonical[index]
        for index, visible_label in enumerate(BELIEF_KEYS)
    }


def invert_hypothesis_display_map(
    display_to_canonical: dict[str, str],
) -> dict[str, str]:
    """Return {canonical_hypothesis_id: visible_label}."""
    return {canonical: visible for visible, canonical in display_to_canonical.items()}


def build_display_order(
    rng: np.random.Generator,
    labels: Iterable[str] = BELIEF_KEYS,
) -> tuple[str, ...]:
    ordered = list(labels)
    rng.shuffle(ordered)
    return tuple(ordered)


def remap_hypotheses_to_visible(
    canonical_hypotheses: dict[str, str],
    display_to_canonical: dict[str, str],
) -> dict[str, str]:
    return {
        visible_label: canonical_hypotheses[canonical_label]
        for visible_label, canonical_label in display_to_canonical.items()
    }


def remap_distribution_to_visible(
    canonical_dist: dict[str, float],
    display_to_canonical: dict[str, str],
) -> dict[str, float]:
    return {
        visible_label: float(canonical_dist[canonical_label])
        for visible_label, canonical_label in display_to_canonical.items()
    }


def remap_distribution_to_canonical(
    visible_dist: dict[str, float],
    display_to_canonical: dict[str, str],
) -> dict[str, float]:
    return {
        canonical_label: float(visible_dist[visible_label])
        for visible_label, canonical_label in display_to_canonical.items()
    }


def rewrite_prompt_with_aliases(
    prompt_messages: list,
    alias_map: dict[str, str],
) -> list:
    """Replace canonical experiment IDs with opaque labels and reorder.

    Works with both plain dicts and Pydantic message objects (UserMessage).
    """
    import copy

    new_messages = copy.deepcopy(prompt_messages)

    for msg in new_messages:
        if msg.get("role") != "user":
            continue
        content = msg["content"] if isinstance(msg, dict) else msg.content
        if not isinstance(content, str):
            continue
        lines = content.split("\n")
        experiment_indices: list[int] = []
        experiment_lines_by_canonical: dict[str, str] = {}
        for i, line in enumerate(lines):
            for cid in EXPERIMENT_IDS:
                if line.startswith(f"- {cid}:"):
                    experiment_indices.append(i)
                    experiment_lines_by_canonical[cid] = line
                    break

        if len(experiment_lines_by_canonical) != len(EXPERIMENT_IDS):
            continue

        reordered: list[str] = []
        for opaque_label in OPAQUE_LABELS:
            canonical_id = alias_map[opaque_label]
            original_line = experiment_lines_by_canonical[canonical_id]
            new_line = original_line.replace(f"- {canonical_id}:", f"- {opaque_label}:", 1)
            reordered.append(new_line)

        for idx, new_line in zip(experiment_indices, reordered):
            lines[idx] = new_line
        new_content = "\n".join(lines)
        if isinstance(msg, dict):
            msg["content"] = new_content
        else:
            msg.content = new_content

    return new_messages


def render_observation_text(
    rng: np.random.Generator,
    *,
    domain_display: str,
    experiment_id: str,
    core_text: str,
) -> str:
    prefix = str(rng.choice(OBSERVATION_PREFIXES[experiment_id]))
    template = str(rng.choice(OBSERVATION_TEMPLATES))
    core_lc = core_text[0].lower() + core_text[1:] if core_text else core_text
    return template.format(
        prefix=prefix,
        core=core_text,
        core_lc=core_lc,
        domain=domain_display,
    )


def build_observation_bank(
    rng: np.random.Generator,
    *,
    domain_key: str,
) -> dict[str, dict[str, str]]:
    domain = DOMAIN_LIBRARY[domain_key]
    observation_bank: dict[str, dict[str, str]] = {}
    for experiment_id in EXPERIMENT_IDS:
        observation_bank[experiment_id] = {}
        for outcome_index, outcome_id in enumerate(OUTCOME_IDS):
            core_text = domain["outcomes"][experiment_id][outcome_index]
            observation_bank[experiment_id][outcome_id] = render_observation_text(
                rng,
                domain_display=domain["display"],
                experiment_id=experiment_id,
                core_text=core_text,
            )
    return observation_bank


DOMAIN_LIBRARY = {
    "cell_signaling": {
        "display": "cell signaling",
        "frames": (
            {
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
            },
            {
                "question": "Which explanation best accounts for the short kinase burst after the ligand challenge?",
                "hypotheses": {
                    "H1": "an upstream pulse that briefly aligns the cascade before it relaxes",
                    "H2": "fast receptor-state depletion once signaling begins",
                    "H3": "a lagged inhibitory circuit that closes the pathway after a short delay",
                },
                "experiments": {
                    "informative": "Run a dense kinase time-course immediately after the ligand challenge.",
                    "medium": "Collect paired kinase and transcript snapshots across the early and middle windows.",
                    "cheap_weak": "Check one inexpensive downstream phospho-proxy from the routine pathway panel.",
                    "redundant": "Repeat the standard signaling assay with almost the same panel and cadence.",
                    "high_cost_informative": "Run a single-cell ligand-response time-course with perturbation tracking.",
                },
            },
            {
                "question": "What most plausibly drives the brief pathway activation seen after inhibitor washout?",
                "hypotheses": {
                    "H1": "a transient feed-forward alignment that self-resolves quickly",
                    "H2": "an initial rebound followed by rapid desensitization of the receptor complex",
                    "H3": "a delayed feedback suppressor that turns the pathway off after the rebound",
                },
                "experiments": {
                    "informative": "Measure pathway phosphorylation at high temporal resolution during washout.",
                    "medium": "Sample paired phospho and downstream transcription signatures at moderate cadence.",
                    "cheap_weak": "Use a low-cost downstream reporter as a quick proxy for pathway activity.",
                    "redundant": "Repeat the baseline washout assay with nearly identical readout settings.",
                    "high_cost_informative": "Profile the washout response with a high-cost single-cell live readout.",
                },
            },
        ),
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
        "frames": (
            {
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
            },
            {
                "question": "What most likely explains the sudden dip and rebound in the estuary grazer network?",
                "hypotheses": {
                    "H1": "a brief predator surge that knocks the grazer layer down before easing",
                    "H2": "a bloom that quickly exhausts the local resource base",
                    "H3": "a delayed habitat feedback that suppresses recovery only after the first shock",
                },
                "experiments": {
                    "informative": "Run a high-frequency predator-exclusion and nutrient sampling campaign across the shock window.",
                    "medium": "Measure grazers and nutrient levels together at an intermediate cadence.",
                    "cheap_weak": "Use one low-cost proxy read from the standing estuary monitor.",
                    "redundant": "Repeat the usual exclusion assay on nearly the same shoreline transects.",
                    "high_cost_informative": "Deploy a high-resolution tagged field array across the estuary reach.",
                },
            },
            {
                "question": "Which account best fits the lagoon ecosystem's brief crash followed by partial recovery?",
                "hypotheses": {
                    "H1": "a threshold predator pulse that briefly depresses the consumer layer",
                    "H2": "resource depletion immediately after the early bloom pulse",
                    "H3": "a lagged nursery-habitat feedback that binds only in the late phase",
                },
                "experiments": {
                    "informative": "Run a dense exclusion-and-resource survey throughout the disturbance interval.",
                    "medium": "Collect paired consumer and nutrient snapshots over the same period.",
                    "cheap_weak": "Read one inexpensive ecosystem proxy from the routine lagoon station.",
                    "redundant": "Repeat the baseline field assay with almost unchanged transects.",
                    "high_cost_informative": "Install a high-cost tagged sensor array across the lagoon segment.",
                },
            },
        ),
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
        "frames": (
            {
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
            },
            {
                "question": "What best explains the brief conductivity gain and later collapse in the printed device?",
                "hypotheses": {
                    "H1": "a short-lived alignment of conductive domains under the initial bias",
                    "H2": "rapid contact wear after the first current surge",
                    "H3": "a delayed interface-feedback process driven by defect build-up",
                },
                "experiments": {
                    "informative": "Run a dense bias sweep with simultaneous structural monitoring on the device.",
                    "medium": "Collect paired conductivity and interface diagnostics at intermediate cadence.",
                    "cheap_weak": "Use a quick low-cost proxy from the routine fabrication monitor.",
                    "redundant": "Repeat the standard conductivity scan with almost the same probe setup.",
                    "high_cost_informative": "Run a high-cost microscopy-and-bias session on the same printed device.",
                },
            },
            {
                "question": "Which mechanism most plausibly drives the unstable transport pattern in the interface stack?",
                "hypotheses": {
                    "H1": "a transient domain-ordering event that briefly improves transport",
                    "H2": "contact degradation that sets in immediately after the early surge",
                    "H3": "a delayed defect-feedback process centered on the buried interface",
                },
                "experiments": {
                    "informative": "Measure the transport response with a high-resolution bias-time protocol and structural readout.",
                    "medium": "Sample transport and interface signatures together across early and mid windows.",
                    "cheap_weak": "Read one inexpensive proxy signal from the routine wafer monitor.",
                    "redundant": "Repeat the baseline transport scan with nearly unchanged probe geometry.",
                    "high_cost_informative": "Profile the stack with a microscopy-plus-bias workflow on the same sample.",
                },
            },
        ),
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
        "frames": (
            {
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
            },
            {
                "question": "What most likely explains the vaccine outreach program's early surge and fast fadeout?",
                "hypotheses": {
                    "H1": "a brief outreach pulse that changes behavior only in the first wave",
                    "H2": "strong initial uptake followed by rapid delivery-system attrition",
                    "H3": "a delayed trust backlash that appears after early visibility peaks",
                },
                "experiments": {
                    "informative": "Run a dense follow-up study linking outreach exposure, delivery logs, and attendance.",
                    "medium": "Measure participation and sentiment together at an intermediate cadence.",
                    "cheap_weak": "Use one low-cost proxy indicator from the routine public-health dashboard.",
                    "redundant": "Repeat the standard uptake audit on nearly the same outreach locations.",
                    "high_cost_informative": "Run a high-cost participant panel with linked follow-up interviews.",
                },
            },
            {
                "question": "Which mechanism best fits the brief adoption bump in the text-reminder program?",
                "hypotheses": {
                    "H1": "a threshold reminder pulse that produces only a short-lived behavior shift",
                    "H2": "early adoption followed by rapid operational drop-off in delivery",
                    "H3": "a lagged backlash in trust or sentiment after the first reminder wave",
                },
                "experiments": {
                    "informative": "Run a high-resolution follow-up linking reminder exposure, delivery success, and attendance.",
                    "medium": "Collect paired participation and sentiment snapshots over the rollout window.",
                    "cheap_weak": "Read one inexpensive proxy from the standing intervention dashboard.",
                    "redundant": "Repeat the usual attendance review with almost the same participant groups.",
                    "high_cost_informative": "Run a costly mixed-methods follow-up panel on the same reminder cohort.",
                },
            },
        ),
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
        "frames": (
            {
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
            },
            {
                "question": "What most plausibly explains the tax-credit rollout's early gains and later fadeout?",
                "hypotheses": {
                    "H1": "a brief expectations pulse that shifts firm behavior only in the first phase",
                    "H2": "initial uptake followed by fast implementation frictions inside the program",
                    "H3": "a delayed equilibrium adjustment that unwinds the early improvement",
                },
                "experiments": {
                    "informative": "Run a high-frequency linked panel of firms, compliance, and prices over the rollout window.",
                    "medium": "Measure compliance and prices together at intermediate cadence.",
                    "cheap_weak": "Use one inexpensive proxy indicator from the regular macro dashboard.",
                    "redundant": "Repeat the baseline compliance audit on nearly identical reporting units.",
                    "high_cost_informative": "Run a matched microdata panel with high-cost firm-level tracking.",
                },
            },
            {
                "question": "Which account best fits the export-support policy's short bump and later reversal?",
                "hypotheses": {
                    "H1": "a threshold expectations pulse that briefly changes planning behavior",
                    "H2": "initial take-up followed by rapid operational friction in implementation",
                    "H3": "a lagged general-equilibrium feedback that offsets the early gains",
                },
                "experiments": {
                    "informative": "Measure firms, prices, and compliance in a dense linked panel across the intervention period.",
                    "medium": "Collect paired compliance and price snapshots through the same window.",
                    "cheap_weak": "Read one low-cost proxy from the standing macro-policy dashboard.",
                    "redundant": "Repeat the standard audit with almost unchanged reporting units.",
                    "high_cost_informative": "Run a high-cost matched panel with firm-level outcome tracking.",
                },
            },
        ),
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
        "layout": "sections",
        "header": "Research question",
        "hypotheses": "Candidate hypotheses",
        "prior": "Current prior",
        "experiments": "Candidate experiments",
        "budget": "Evidence budget",
        "experiment_line": "- {exp_id}: {description} Cost={cost:.1f}.",
    },
    1: {
        "layout": "brief",
        "header": "Open question",
        "hypotheses": "Hypothesis set",
        "prior": "Prior belief state",
        "experiments": "Available studies",
        "budget": "Remaining experiment turns",
        "experiment_line": "- {exp_id}: {description} Estimated cost {cost:.1f}.",
    },
    2: {
        "layout": "casefile",
        "header": "Decision target",
        "hypotheses": "Working hypotheses",
        "prior": "Bayesian prior",
        "experiments": "Allowed next experiments",
        "budget": "Turn budget",
        "experiment_line": "- {exp_id}: {description} Budget weight {cost:.1f}.",
    },
    3: {
        "layout": "checklist",
        "header": "Question under study",
        "hypotheses": "Competing explanations",
        "prior": "Starting belief",
        "experiments": "Evidence options",
        "budget": "Maximum experiment turns",
        "experiment_line": "- {exp_id}: {description} Cost score {cost:.1f}.",
    },
    4: {
        "layout": "memo",
        "header": "Mechanism question",
        "hypotheses": "Mechanistic candidates",
        "prior": "Initial posterior guess",
        "experiments": "Permitted experiments",
        "budget": "Experiment budget",
        "experiment_line": "- {exp_id}: {description} Cost index {cost:.1f}.",
    },
    5: {
        "layout": "dossier",
        "header": "Investigative target",
        "hypotheses": "Named hypotheses",
        "prior": "Belief prior",
        "experiments": "Study menu",
        "budget": "Budgeted turns",
        "experiment_line": "- {exp_id}: {description} Cost unit {cost:.1f}.",
    },
    6: {
        "layout": "sections",
        "header": "Target question",
        "hypotheses": "Possible explanations",
        "prior": "Prior over explanations",
        "experiments": "Experiment menu",
        "budget": "Evidence-turn budget",
        "experiment_line": "- {exp_id}: {description} Estimated burden {cost:.1f}.",
    },
    7: {
        "layout": "brief",
        "header": "Mechanism to resolve",
        "hypotheses": "Explanation candidates",
        "prior": "Current hypothesis weights",
        "experiments": "Candidate probes",
        "budget": "Probe budget",
        "experiment_line": "- {exp_id}: {description} Cost burden {cost:.1f}.",
    },
    8: {
        "layout": "casefile",
        "header": "Question to disambiguate",
        "hypotheses": "Alternative accounts",
        "prior": "Prior mass by hypothesis",
        "experiments": "Experimental options",
        "budget": "Maximum evidence turns",
        "experiment_line": "- {exp_id}: {description} Estimated cost weight {cost:.1f}.",
    },
    9: {
        "layout": "memo",
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


def format_distribution(
    dist: dict[str, float],
    order: Iterable[str] = BELIEF_KEYS,
) -> str:
    return ", ".join(f"{key}={dist[key]:.3f}" for key in order)


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
    question: str,
    template_id: int,
    mode: str,
    hypotheses: dict[str, str],
    prior: dict[str, float],
    hypothesis_display_order: tuple[str, ...],
    experiment_descriptions: dict[str, str],
    passive_initial_observation: str | None,
) -> list[dict[str, str]]:
    wording = WORDING_BANK[template_id]

    hypothesis_lines = "\n".join(
        f"- {hypothesis_id}: {hypotheses[hypothesis_id]}"
        for hypothesis_id in hypothesis_display_order
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
    label_rule = (
        "Treat H1, H2, and H3 as episode-local labels. "
        "Belief reports must use probabilities over H1, H2, H3 that sum to 1."
    )
    observation_block = ""
    if passive_initial_observation is not None:
        observation_block = f"\nInitial revealed observation:\n- {passive_initial_observation}\n"
    prior_line = format_distribution(prior, order=hypothesis_display_order)

    if wording["layout"] == "sections":
        prompt = (
            f"{wording['header']}: {question}\n\n"
            f"{wording['hypotheses']}:\n{hypothesis_lines}\n\n"
            f"{wording['prior']}: {prior_line}\n\n"
            f"{wording['experiments']}:\n{experiment_lines}\n\n"
            f"{wording['budget']}: {TURN_BUDGET}\n"
            f"Protocol: {protocol}\n"
            f"{label_rule}{observation_block}"
        )
    elif wording["layout"] == "brief":
        prompt = (
            f"{wording['header']}\n"
            f"Question under review: {question}\n\n"
            f"{wording['prior']}: {prior_line}\n"
            f"{wording['budget']}: {TURN_BUDGET}\n\n"
            f"{wording['hypotheses']}:\n{hypothesis_lines}\n\n"
            f"{wording['experiments']}:\n{experiment_lines}\n\n"
            f"Operating rule: {protocol}\n"
            f"{label_rule}{observation_block}"
        )
    elif wording["layout"] == "casefile":
        prompt = (
            f"{wording['header']}\n\n"
            f"Question to resolve\n- {question}\n\n"
            f"{wording['hypotheses']}\n{hypothesis_lines}\n\n"
            f"{wording['experiments']}\n{experiment_lines}\n\n"
            f"{wording['prior']}\n- {prior_line}\n- {wording['budget']}: {TURN_BUDGET}\n\n"
            f"Protocol\n- {protocol}\n- {label_rule}{observation_block}"
        )
    elif wording["layout"] == "checklist":
        prompt = (
            f"{wording['header']}: {question}\n\n"
            "Checklist\n"
            f"1. Review the candidate labels below.\n{hypothesis_lines}\n\n"
            f"2. Start from this belief state: {prior_line}\n\n"
            f"3. Choose from these evidence options.\n{experiment_lines}\n\n"
            f"4. Respect the {wording['budget'].lower()}: {TURN_BUDGET}\n"
            f"5. Follow the protocol: {protocol}\n"
            f"6. {label_rule}{observation_block}"
        )
    elif wording["layout"] == "memo":
        prompt = (
            f"{wording['header']}\n"
            f"Focus: {question}\n\n"
            f"{wording['hypotheses']}:\n{hypothesis_lines}\n\n"
            f"{wording['experiments']}:\n{experiment_lines}\n\n"
            "Decision frame\n"
            f"- {wording['prior']}: {prior_line}\n"
            f"- {wording['budget']}: {TURN_BUDGET}\n"
            f"- Protocol: {protocol}\n"
            f"- {label_rule}{observation_block}"
        )
    else:
        prompt = (
            f"{wording['header']}\n\n"
            f"Question\n{question}\n\n"
            f"{wording['hypotheses']}\n{hypothesis_lines}\n\n"
            f"{wording['prior']}\n{prior_line}\n\n"
            f"{wording['experiments']}\n{experiment_lines}\n\n"
            f"Constraints\n- {wording['budget']}: {TURN_BUDGET}\n- {protocol}\n- {label_rule}{observation_block}"
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
    frame_bank = domain["frames"]
    frame_index = episode_seed % len(frame_bank)
    frame = frame_bank[frame_index]
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
            experiment_id: frame["experiments"][experiment_id]
            for experiment_id in EXPERIMENT_IDS
        }
        observation_bank = build_observation_bank(rng, domain_key=domain_key)
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
        hypothesis_display_to_canonical = build_hypothesis_display_map(rng)
        hypothesis_canonical_to_display = invert_hypothesis_display_map(
            hypothesis_display_to_canonical
        )
        hypothesis_display_order = build_display_order(rng)
        visible_hypotheses = remap_hypotheses_to_visible(
            frame["hypotheses"],
            hypothesis_display_to_canonical,
        )
        visible_prior = remap_distribution_to_visible(
            prior,
            hypothesis_display_to_canonical,
        )
        visible_reference_trace = [
            {
                "step_index": step["step_index"],
                "source": step["source"],
                "posterior": remap_distribution_to_visible(
                    step["posterior"],
                    hypothesis_display_to_canonical,
                ),
            }
            for step in reference_trace
        ]
        visible_passive_plan = []
        for step in reference_plan:
            visible_step = dict(step)
            visible_step["posterior_after"] = remap_distribution_to_visible(
                step["posterior_after"],
                hypothesis_display_to_canonical,
            )
            visible_passive_plan.append(visible_step)
        prompt = _render_prompt(
            question=frame["question"],
            template_id=template_id,
            mode=mode,
            hypotheses=visible_hypotheses,
            prior=visible_prior,
            hypothesis_display_order=hypothesis_display_order,
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
            "frame_id": frame_index,
            "template_id": template_id,
            "question": frame["question"],
            "hypotheses": visible_hypotheses,
            "prior": visible_prior,
            "hypothesis_display_order": list(hypothesis_display_order),
            "visible_experiments": [
                {
                    "experiment_id": experiment_id,
                    "type": EXPERIMENT_TYPES[experiment_id],
                    "description": experiment_descriptions[experiment_id],
                    "cost": EXPERIMENT_COSTS[experiment_id],
                }
                for experiment_id in EXPERIMENT_IDS
            ],
            "passive_plan": visible_passive_plan,
            "passive_initial_observation": passive_initial_observation,
            "active_outcomes": active_outcomes,
            "observation_bank": observation_bank,
            "reference_trace": visible_reference_trace,
            "contradiction_metadata": contradiction_metadata,
            "hidden": {
                "canonical_prior": prior,
                "canonical_hypotheses": frame["hypotheses"],
                "hypothesis_display_to_canonical": hypothesis_display_to_canonical,
                "hypothesis_canonical_to_display": hypothesis_canonical_to_display,
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
