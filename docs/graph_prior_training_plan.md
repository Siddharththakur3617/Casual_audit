# Plan: training the cart-pole JEPA with the expert graph as a prior

Status 2026-11-16: code written, nothing run yet. Run order: README, "Graph prior during training".

## Goal

The main study audits whether a pixels-only JEPA learns cart-pole's causal graph. This study reverses the question. It gives the expert graph to the model during training and compares the training run with a standard JEPA run. The comparison covers:

- epochs and time needed to reach a given val loss or ridge R²
- the generalisation gap (the current 200-epoch run overfits: train rollout 0.024, val 0.138)
- long open-loop rollouts
- how fast the graph becomes recoverable with the existing probe pipeline

The encoder, the latent rollout loss, the inverse model and SIGReg stay unchanged in every arm (this JEPA has no EMA target). The graph enters in one of two ways.

| Arm | Config | What changes |
|---|---|---|
| standard | `jepa_cartpole.yaml`, `--epochs 100 --probe-every 5` | nothing |
| masked | `jepa_cartpole_graphmask.yaml` | The predictor runs on (time, slot) tokens. Attention is masked so slot j reads only its parents, itself and ctx. Projections and the BatchNorm projector are per slot. |
| jacobian | `jepa_cartpole_graphjac.yaml` | The predictor is unchanged. A finite-difference penalty on ∂ẑ_j/∂z_i applies to every absent edge i→j. |

Both graph arms also use `lambda_ctx`, a loss that keeps ctx constant within a window. Hidden physics are constant within an episode, so ctx cannot carry x or θ past the mask, and the loss needs no labels.

## What the graph says (`models/jepa/graph.py`)

- **Slots:** z (192) = [x 40 | v̄ 40 | θ 40 | ω̄ 40 | ctx 32]. ctx is free and is meant for M, m, l, g. These are true parents of every change but have no node in the graph.
- **Parents:** taken from `checks/expert_graph.EXPERT`, strong and weak edges both, because a missing true edge hurts far more than an extra one.
  - x ← v̄, θ, ω̄
  - v̄ ← θ, ω̄
  - θ ← ω̄
  - ω̄ ← θ
  - Self is always allowed, because the predictor outputs z_{t+1}, which equals z_t plus the change.
  - Every slot reads ctx; ctx reads only itself.
- **Absent state edges:** x→v̄, x→θ, x→ω̄, v̄→θ and v̄→ω̄. In words, the pole subsystem never sees the cart. There are only 5 such edges, so a near-null in-distribution result is plausible and should be reported as such.
- **No leakage through depth:** the mask is transitively closed, so stacking 6 masked layers adds no non-parent path within one step. Paths across time steps are physical.
- **Actions:** u_t and u_{t−1} are parents of all four changes, so the existing global AdaLN action conditioning already matches the graph. u_{t−2} is not in the action context.

## Implementation

- `models/jepa/graph.py`:
  - slot layout
  - `slot_mask`, with edge sets `strong_weak` (default), `strong`, `shuffled` (matched count, a control) and `full` (dense slots, a control)
  - `non_edges`, `parent_edges`, `graph_spec`
- `models/jepa/model.py`:
  - `GraphMaskedPredictor`
  - `ConditionalBlock` takes an optional boolean `attn_mask`
  - `JEPA` picks the predictor from `predictor_type` and exposes `graph_spec`
  - `encode` and `predict` keep their shapes, so `probing/`, `run_all.py` and the curve code run unchanged (`model_type` stays `jepa`)
- `models/jepa/train.py`:
  - `edge_sensitivity`: noise of 0.1 × the batch std is added to block i at every history position, and the result is ‖Δẑ_j‖² / ‖δ‖². Inputs are detached, so the penalty shapes the predictor and not the encoder. The predictor runs in eval mode for the penalty passes, because train-mode BatchNorm would spread the perturbation to every slot.
  - New losses `lambda_jac` and `lambda_ctx`; seven loss parts are now logged.
  - Per-epoch `train_s` and `wall_s`, and a `--probe-every` flag.
  - `graph_sensitivity_val` on probe epochs, for slot-layout runs only. The standard JEPA has no slots, so its graph learning is measured by the `run_all.py --curve` graph recovery, which reads true-state edges and needs no slots.
- `evaluate/compare_training.py`: plots and a table:
  - val rollout loss against epoch and against optimisation time
  - epochs and hours to 1.1 × the standard run's final val rollout loss
  - epochs to R² ≥ 0.9 per variable
  - generalisation gap
  - open-loop rollout to 20 steps, as latent relative MSE and as state nRMSE through a ridge readout fitted on each model's own real latents
  - graph-recovery F1 and AUROC across checkpoints, if `curve.json` exists
- `checks/verify_graph_mask.py`: a data-free CPU check of the mask, its exact invariance, shapes and the penalty estimator.

## Run order and budget

1. `python -m checks.verify_graph_mask`
2. Smoke run: one epoch of each arm on `datasets/cc_small`.
3. Weight pick: 10-epoch runs with `lambda_jac` ∈ {0.1, 1.0} and `lambda_ctx` ∈ {0.1, 1.0}. Keep the largest weight whose val rollout loss at epoch 10 stays within 5% of the standard run's. The configs ship with `lambda_jac` 1.0 and `lambda_ctx` 0.1 as placeholders.
4. Three 100-epoch runs, seed 42, about 1.75 h each. The masked arm is somewhat slower, because it has 25 predictor tokens instead of 5.
5. `run_all.py --curve --splits 20 --prune null_cal_fwer` on each run. The 20-split null already exists.
6. `python -m evaluate.compare_training ...`
7. Optional: a second seed (`--seed 43`) for the standard run and the better graph arm.

## Caveats for the write-up

- The slots are not tied to named variables, because no state labels are used and the model stays pixels-only. The mask constrains information flow between slots, but the encoder decides what each slot holds. The probe pipeline, which reads true-state inputs, measures the outcome independently of the slots.
- With one seed per arm, differences are indicative only.
- The 100-epoch standard run is new. The existing 200-epoch run has a different cosine schedule and is not a fair baseline.
- The open-loop windows (24 frames) drop episodes shorter than that. The same windows are used for every arm.
