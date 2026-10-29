# Configurations

YAML configurations for:
- `datasets/`: source schema and split plan
- `generation/`: model eligibility, prompt families and sample budgets
- `training/`: classifier and optimization recipes
- `experiments/`: matched controlled comparisons, seeds and test protocol

Only lightweight *template* configs are committed at initialization. Validated runnable configs follow per phase.

## Evaluation configuration

`experiments/evaluation.yaml` is a **not-executed** research contract: fixed real-test endpoint, the six class labels, bootstrap policy and prespecified contrasts. The evaluation notebooks expose equivalent settings in editable config cells and refuse placeholder paths. The preceding `experiments/main.yaml` is not proof of an executed experiment. Keep actual model revision, split SHA, training budgets and checkpoint digests with each Kaggle run.
