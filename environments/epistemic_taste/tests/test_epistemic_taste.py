from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epistemic_taste.bayes import (  # noqa: E402
    ACTIVE_MODE,
    SplitSpec,
    build_episode,
    posterior_update,
    utility_map_for_state,
)
from epistemic_taste.epistemic_taste import validate_belief_payload  # noqa: E402


class EpistemicTasteTests(unittest.TestCase):
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
        self.assertEqual(info["episode_version"], "epistemic-taste-v1")

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
            episode["info"]["prior"],
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


if __name__ == "__main__":
    unittest.main()
