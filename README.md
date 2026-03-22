## Epistemic Taste

Synthetic Prime/verifiers environment for GRPO training on epistemic behavior under uncertainty:

- `RQ1`: choose experiments by expected information gain rather than convenience
- `RQ2`: revise beliefs proportionally when evidence conflicts with the current hypothesis

This repo contains one frozen synthetic environment family, a reproducible 5,000-episode dataset, local eval utilities, and a hosted Prime RL config.

## Repo Layout

- `environments/epistemic_taste/`: environment package, generator, frozen data, tests
- `scripts/`: dataset QA, base-model eval, held-out eval, reward-variance gate, trajectory dump
- `configs/`: Prime RL config and reproducible logged example trajectories

## Requirements

- Python `3.11+`
- `uv`
- Prime CLI authenticated via `prime login`
- GitHub CLI authenticated via `gh auth login` if you want to publish a remote

## Local Setup

```bash
cd /Users/jarrodbarnes/ai-scientist-training
uv sync
prime env install epistemic-taste -p ./environments
```

## Reproducible Dataset

The frozen split files are committed under `environments/epistemic_taste/epistemic_taste/data/`.

To regenerate them deterministically:

```bash
cd /Users/jarrodbarnes/ai-scientist-training
.venv/bin/python scripts/generate_epistemic_taste_dataset.py --force
.venv/bin/python scripts/verify_dataset.py --strict
```

Expected split spec:

- train: `4000` episodes, `70%` active / `30%` passive, contradiction rate `0.30`
- dev: `500` episodes, `70%` active / `30%` passive, contradiction rate `0.20`
- test: `500` episodes, `70%` active / `30%` passive, contradiction rate `0.40`

## Local Validation

```bash
cd /Users/jarrodbarnes/ai-scientist-training
uv run ruff check environments/epistemic_taste scripts
.venv/bin/python -m py_compile environments/epistemic_taste/epistemic_taste/*.py scripts/*.py environments/epistemic_taste/tests/test_epistemic_taste.py
.venv/bin/python -m unittest environments/epistemic_taste/tests/test_epistemic_taste.py
```

## Evaluation

Heuristic baselines plus base-model eval on dev:

```bash
cd /Users/jarrodbarnes/ai-scientist-training
.venv/bin/python scripts/base_model_eval.py --model 'Qwen/Qwen3-30B-A3B-Instruct-2507' --split dev --num-examples 64 --rollouts-per-example 4 --max-concurrent 8
```

Held-out eval on test:

```bash
cd /Users/jarrodbarnes/ai-scientist-training
.venv/bin/python scripts/heldout_eval.py --model 'Qwen/Qwen3-30B-A3B-Instruct-2507' --split test --num-examples 200 --rollouts-per-example 4 --max-concurrent 8
```

Reward variance gate for GRPO groups:

```bash
cd /Users/jarrodbarnes/ai-scientist-training
.venv/bin/python scripts/reward_variance_gate.py /absolute/path/to/results.jsonl --max-flat-fraction 0.5
```

Example trajectories:

```bash
cd /Users/jarrodbarnes/ai-scientist-training
.venv/bin/python scripts/dump_example_trajectories.py --split dev --policy oracle --output-jsonl /Users/jarrodbarnes/ai-scientist-training/configs/example_trajectories.jsonl
```

## Hosted Prime RL

The hosted config lives at `configs/epistemic_taste_grpo.toml` and now points at the published environment ID `jbarnes850/epistemic-taste`.

Publish a new version and run hosted training:

```bash
cd /Users/jarrodbarnes/ai-scientist-training
prime env push epistemic-taste -p ./environments -o jbarnes850
prime rl run configs/epistemic_taste_grpo.toml
```

## Notes

- The environment exposes two explicit tool actions: `run_experiment(experiment_id)` and `report_belief(belief, stop)`.
- The latent Bayesian world is synthetic. Scientific domain language is surface rendering only.
- The source of truth for reward is exact hidden posterior state, not lexical surface cues.
