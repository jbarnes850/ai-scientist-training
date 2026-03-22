from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from research_hypothesis_analysis.bayes import (  # noqa: E402
    ACTIVE_MODE,
    EXPERIMENT_IDS,
    OPAQUE_LABELS,
    SplitSpec,
    build_alias_map,
    build_hypothesis_display_map,
    build_observation_bank,
    build_episode,
    build_display_order,
    invert_alias_map,
    invert_hypothesis_display_map,
    posterior_update,
    remap_distribution_to_canonical,
    remap_distribution_to_visible,
    rewrite_prompt_with_aliases,
    utility_map_for_state,
)
from research_hypothesis_analysis.environment import (  # noqa: E402
    _rewrite_tool_defs_with_aliases,
    validate_belief_payload,
)


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split_spec = SplitSpec(
            name="unit",
            count=1,
            domains=("cell_signaling",),
            template_ids=(0,),
            prior_map_range=(0.45, 0.65),
            likelihood_multiplier_range=(0.95, 1.05),
            contradiction_rate=0.30,
        )

    def test_build_episode_has_expected_structure(self) -> None:
        episode = build_episode(
            split_spec=self.split_spec,
            episode_index=0,
            episode_seed=7,
            mode=ACTIVE_MODE,
            contradiction_target=False,
            domain_key="cell_signaling",
            template_id=0,
        )
        info = episode["info"]
        self.assertEqual(info["split"], "unit")
        self.assertEqual(len(info["visible_experiments"]), 5)
        self.assertEqual(len(info["passive_plan"]), 3)
        self.assertEqual(info["episode_version"], "research-hypothesis-analysis-v1")

    def test_posterior_update_stays_normalized(self) -> None:
        likelihoods = {
            "H1": [0.8, 0.1, 0.1],
            "H2": [0.1, 0.8, 0.1],
            "H3": [0.1, 0.1, 0.8],
        }
        posterior = posterior_update(
            {"H1": 0.4, "H2": 0.35, "H3": 0.25},
            likelihoods,
            "O1",
        )
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=6)
        self.assertGreater(posterior["H1"], posterior["H2"])

    def test_utility_map_returns_all_available_experiments(self) -> None:
        episode = build_episode(
            split_spec=self.split_spec,
            episode_index=0,
            episode_seed=19,
            mode=ACTIVE_MODE,
            contradiction_target=True,
            domain_key="cell_signaling",
            template_id=0,
        )
        utilities = utility_map_for_state(
            episode["info"]["hidden"]["canonical_prior"],
            episode["info"]["hidden"]["likelihoods"],
            ["informative", "cheap_weak"],
        )
        self.assertEqual(set(utilities.keys()), {"informative", "cheap_weak"})
        self.assertTrue(all(isinstance(value, float) for value in utilities.values()))

    def test_belief_validation(self) -> None:
        self.assertIsNone(
            validate_belief_payload({"H1": 0.2, "H2": 0.3, "H3": 0.5})
        )
        self.assertIsNotNone(
            validate_belief_payload({"H1": 0.2, "H2": 0.3, "H4": 0.5})
        )
        self.assertIsNotNone(
            validate_belief_payload({"H1": 0.2, "H2": 0.3, "H3": 0.7})
        )


class PresentationPerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split_spec = SplitSpec(
            name="unit",
            count=1,
            domains=("cell_signaling",),
            template_ids=(0,),
            prior_map_range=(0.45, 0.65),
            likelihood_multiplier_range=(0.95, 1.05),
            contradiction_rate=0.30,
        )
        self.episode = build_episode(
            split_spec=self.split_spec,
            episode_index=0,
            episode_seed=42,
            mode=ACTIVE_MODE,
            contradiction_target=False,
            domain_key="cell_signaling",
            template_id=0,
        )

    def test_build_alias_map_produces_valid_mapping(self) -> None:
        rng = np.random.default_rng(123)
        alias_map = build_alias_map(rng)
        self.assertEqual(set(alias_map.keys()), set(OPAQUE_LABELS))
        self.assertEqual(sorted(alias_map.values()), sorted(EXPERIMENT_IDS))

        rng2 = np.random.default_rng(456)
        alias_map2 = build_alias_map(rng2)
        self.assertNotEqual(alias_map, alias_map2)

    def test_invert_alias_map(self) -> None:
        rng = np.random.default_rng(789)
        alias_map = build_alias_map(rng)
        reverse = invert_alias_map(alias_map)
        for opaque, canonical in alias_map.items():
            self.assertEqual(reverse[canonical], opaque)

    def test_hypothesis_display_map_round_trip(self) -> None:
        rng = np.random.default_rng(111)
        display_to_canonical = build_hypothesis_display_map(rng)
        canonical_to_display = invert_hypothesis_display_map(display_to_canonical)
        canonical_prior = {"H1": 0.2, "H2": 0.3, "H3": 0.5}
        visible_prior = remap_distribution_to_visible(
            canonical_prior,
            display_to_canonical,
        )
        round_trip = remap_distribution_to_canonical(
            visible_prior,
            display_to_canonical,
        )
        self.assertEqual(set(display_to_canonical.keys()), {"H1", "H2", "H3"})
        self.assertEqual(set(canonical_to_display.keys()), {"H1", "H2", "H3"})
        self.assertEqual(round_trip, canonical_prior)

    def test_build_display_order_is_permutation(self) -> None:
        rng = np.random.default_rng(222)
        order = build_display_order(rng)
        self.assertEqual(set(order), {"H1", "H2", "H3"})
        self.assertEqual(len(order), 3)

    def test_rewrite_prompt_replaces_ids(self) -> None:
        rng = np.random.default_rng(101)
        alias_map = build_alias_map(rng)
        original_prompt = self.episode["prompt"]
        rewritten = rewrite_prompt_with_aliases(original_prompt, alias_map)

        rewritten_text = rewritten[0]["content"]
        for canonical_id in EXPERIMENT_IDS:
            self.assertNotIn(f"- {canonical_id}:", rewritten_text)
        for opaque_label in OPAQUE_LABELS:
            self.assertIn(f"- {opaque_label}:", rewritten_text)

        # Original prompt unchanged
        original_text = original_prompt[0]["content"]
        for canonical_id in EXPERIMENT_IDS:
            self.assertIn(f"- {canonical_id}:", original_text)

    def test_rewrite_tool_defs(self) -> None:
        rng = np.random.default_rng(202)
        alias_map = build_alias_map(rng)
        tool_defs = [
            {
                "name": "run_experiment",
                "parameters": {
                    "properties": {
                        "experiment_id": {
                            "enum": list(EXPERIMENT_IDS),
                            "type": "string",
                        }
                    }
                },
            },
            {
                "name": "report_belief",
                "parameters": {"properties": {"belief": {}, "stop": {}}},
            },
        ]
        rewritten = _rewrite_tool_defs_with_aliases(tool_defs, alias_map)
        run_exp = [td for td in rewritten if td["name"] == "run_experiment"][0]
        self.assertEqual(
            run_exp["parameters"]["properties"]["experiment_id"]["enum"],
            list(OPAQUE_LABELS),
        )
        # Original unchanged
        orig_run = [td for td in tool_defs if td["name"] == "run_experiment"][0]
        self.assertEqual(
            orig_run["parameters"]["properties"]["experiment_id"]["enum"],
            list(EXPERIMENT_IDS),
        )

    def test_different_seeds_produce_different_aliases(self) -> None:
        maps = [build_alias_map(np.random.default_rng(i)) for i in range(10)]
        unique = {tuple(sorted(m.items())) for m in maps}
        self.assertGreater(len(unique), 1)

    def test_rewrite_prompt_on_pydantic_messages(self) -> None:
        """Verify rewrite works on normalized UserMessage objects, not just dicts."""
        from verifiers.types import UserMessage

        raw_prompt = self.episode["prompt"]
        pydantic_prompt = [UserMessage(**msg) for msg in raw_prompt]

        rng = np.random.default_rng(303)
        alias_map = build_alias_map(rng)
        rewritten = rewrite_prompt_with_aliases(pydantic_prompt, alias_map)

        content = rewritten[0].content
        for canonical_id in EXPERIMENT_IDS:
            self.assertNotIn(f"- {canonical_id}:", content)
        for opaque_label in OPAQUE_LABELS:
            self.assertIn(f"- {opaque_label}:", content)

    def test_rewrite_tool_defs_on_pydantic_tools(self) -> None:
        """Verify rewrite works on normalized Tool objects, not just dicts."""
        from verifiers.types import Tool

        pydantic_defs = [
            Tool(
                name="run_experiment",
                description="",
                parameters={
                    "properties": {
                        "experiment_id": {
                            "enum": list(EXPERIMENT_IDS),
                            "type": "string",
                        }
                    }
                },
            ),
            Tool(
                name="report_belief",
                description="",
                parameters={"properties": {"belief": {}, "stop": {}}},
            ),
        ]
        rng = np.random.default_rng(404)
        alias_map = build_alias_map(rng)
        rewritten = _rewrite_tool_defs_with_aliases(pydantic_defs, alias_map)
        run_exp = [td for td in rewritten if td.get("name") == "run_experiment"][0]
        self.assertEqual(
            run_exp.parameters["properties"]["experiment_id"]["enum"],
            list(OPAQUE_LABELS),
        )

    def test_surface_frames_add_prompt_diversity(self) -> None:
        questions = set()
        prompt_bodies = set()
        for seed in range(12):
            episode = build_episode(
                split_spec=self.split_spec,
                episode_index=seed,
                episode_seed=seed,
                mode=ACTIVE_MODE,
                contradiction_target=False,
                domain_key="cell_signaling",
                template_id=seed % 6,
            )
            questions.add(episode["info"]["question"])
            prompt_bodies.add(episode["prompt"][0]["content"])
        self.assertGreater(len(questions), 1)
        self.assertGreater(len(prompt_bodies), 1)

    def test_observation_bank_varies_across_episode_seeds(self) -> None:
        texts = set()
        for seed in range(20, 28):
            rng = np.random.default_rng(seed)
            observation_bank = build_observation_bank(rng, domain_key="cell_signaling")
            texts.add(observation_bank["informative"]["O1"])
        self.assertGreater(len(texts), 1)


if __name__ == "__main__":
    unittest.main()
