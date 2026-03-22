# Pre-Training Gate Report: 2026-03-22

## Decision

No-go for hosted GRPO yet. The task is not dead and not saturated, but the GRPO-aligned reward-variance gate fails because the experiment-selection signal is effectively flat within rollout groups.

## Concrete Artifacts

- Updated dev base-model summary: local artifact at `configs/base_model_eval_summary.json`
- GRPO-aligned base-model summary: local artifact at `configs/base_model_eval_summary_grpo8.json`
- Saved eval results used for gating:
  - `environments/epistemic_taste/outputs/evals/epistemic-taste--Qwen--Qwen3-30B-A3B-Instruct-2507/6689ca4e/results.jsonl`
  - `environments/epistemic_taste/outputs/evals/epistemic-taste--Qwen--Qwen3-30B-A3B-Instruct-2507/6ae3ab9e/results.jsonl`

Note: the `outputs/` eval artifacts are local run products and are not committed to the public repo.

## Validation Evidence

- Preflight passed:
  - `uv run ruff check environments/epistemic_taste scripts`
  - `.venv/bin/python -m py_compile environments/epistemic_taste/epistemic_taste/*.py scripts/*.py environments/epistemic_taste/tests/test_epistemic_taste.py`
- Eval parity checked:
  - model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
  - sampling: `temperature=0.3`, `max_tokens=1024`
  - same environment, system prompt, and tool schema as training
  - first ran the script-default `64 x 4` proxy, then corrected to a GRPO-aligned `32 x 8` run because training uses `rollouts_per_example = 8`

## Dev Base-Model Results

### `64 x 4` proxy

- `final_map_correct = 0.6914`
- `mean_regret = 0.0355`
- `mean_brier = 0.1621`
- `reward = -0.0830`

### `32 x 8` GRPO-aligned

- `final_map_correct = 0.6055`
- `mean_regret = 0.0381`
- `mean_brier = 0.1870`
- `reward = -0.1441`

## Dead vs Saturated Check

- Not dead: base model is well above `0%`
- Not saturated: `final_map_correct` is below the `80%` saturation threshold in both runs
- Conclusion: task difficulty is usable

## Reward-Variance Gate

### `64 x 4` proxy

- `reward flat_fraction = 0.40625` pass
- `mean_calibration_reward flat_fraction = 0.40625` pass
- `mean_experiment_reward flat_fraction = 0.96875` fail
- `final_map_correct flat_fraction = 0.90625` fail

### `32 x 8` aligned run

- `reward flat_fraction = 0.4375` pass
- `mean_calibration_reward flat_fraction = 0.4375` pass
- `mean_experiment_reward flat_fraction = 1.0` fail
- `final_map_correct flat_fraction = 0.78125` fail

## Interpretation

- Calibration has usable within-group variance.
- Total reward also has usable within-group variance.
- RQ1's explicit experiment-selection component does not.
- In the `32 x 8` run, the model is making essentially the same experiment choices across all `8` rollouts for a given example, so `mean_experiment_reward` is flat and group-normalized gradients will collapse on that axis.
- Under the current guardrail, training should stop before GPU spend.

## Current Blocker

- The gating blocker is reward variance on the experiment-selection axis, not task saturation.

## Recommended Next Step

1. Fix RQ1 variance before training. The most defensible options are increasing rollout diversity for active episodes or making experiment-choice surfaces less deterministic per example.
2. Rerun the `32 x 8` gate after that change.
3. Launch hosted GRPO only after `mean_experiment_reward` drops below the `0.5` flat-fraction threshold.

If needed, the next implementation pass should target the minimum change set that increases experiment-selection variance without weakening the calibration objective.
