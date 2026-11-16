"""Every folder and output location of the project, in one place.

  configs/envs/<env>.yaml          environment, physics ranges, dataset policies (datagen/generate.py)
  configs/models/<model>_<env>.yaml model + training settings (models/<model>/train.py)
  datasets/<data>/                 generated data (git-ignored)
  runs/<model>_<data>/             checkpoints, config.yaml, history.json, extract/ caches (git-ignored);
                                   smoke-test runs live in runs/_smoke/
  results/                         every output (git-ignored):
    expert_graphs/                 expert_graph*.json, checks*.json             (checks/)
    nulls/                         null_calibration_<data>[_<split>]/           (discovery/null_calibration.py)
    data_baselines/                data_baseline<tag>_<rule>.json               (run_all.py --data-baseline)
    jepa/, dreamer/                <run><tag>_diff_<rule>/epoch####/ and curves (run_all.py)
    studies/                       pruning_comparison_*, physics_conditioning_*, counterfactual_<data>_<run>/ (studies/)
    _smoke/                        results of runs in runs/_smoke/
  logs/                            console logs (git-ignored)
"""
from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / 'configs'
ENV_CONFIGS = CONFIGS / 'envs'
MODEL_CONFIGS = CONFIGS / 'models'
DATASETS = HERE / 'datasets'
RUNS = HERE / 'runs'
SMOKE_RUNS = RUNS / '_smoke'
LOGS = HERE / 'logs'

RESULTS = HERE / 'results'
EXPERT = RESULTS / 'expert_graphs'
NULLS = RESULTS / 'nulls'
BASELINES = RESULTS / 'data_baselines'
STUDIES = RESULTS / 'studies'
SMOKE_RESULTS = RESULTS / '_smoke'

MODEL_TYPES = {'jepa': 'jepa', 'dreamerv3': 'dreamer'}   # checkpoint model_type -> results folder


def model_type_of_run(run_dir):
    """'jepa' or 'dreamerv3', from the run's config.yaml (written by the training scripts;
    JEPA runs made before this key existed have none and are JEPA)."""
    p = Path(run_dir) / 'config.yaml'
    if p.exists():
        return (yaml.safe_load(p.read_text()) or {}).get('model_type', 'jepa')
    return 'jepa'


def model_results(run_dir):
    """Results folder for a run: results/_smoke/ for smoke runs, else results/<model>/."""
    run_dir = Path(run_dir).resolve()
    if run_dir.parent == SMOKE_RUNS.resolve():
        return SMOKE_RESULTS
    return RESULTS / MODEL_TYPES[model_type_of_run(run_dir)]
