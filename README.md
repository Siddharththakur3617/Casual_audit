# Do JEPA world models learn causal structure?

A JEPA world model is trained on rendered frames only (no state input). We read a cause-and-effect graph out of its *predicted* state changes and compare it with the graph derived from the exact physics equations. The study covers two environments: **cart-pole** and **Acrobot** (a double pendulum with an unrelated distractor object).

**Results, discussion and related work are in [BTP_REPORT.md](BTP_REPORT.md).** This README covers what the code does, how to run it, and where each output goes.

## Status (2026-11-12)

| | Cart-pole (`datasets/cc_phys`) | Acrobot (`datasets/ac_phys`) |
|---|---|---|
| Equation checks and expert graph | done, 6/6 pass | done, 6/6 pass |
| JEPA training (200 epochs) | done (`runs/jepa_cc_phys`) | done (`runs/jepa_ac_phys`) |
| Differential probe, epochs 200 and 0 | done | done |
| Pruning-rule choice | done: `null_cal_fwer` | fixed in advance: `null_cal_fwer` |
| Graphs from the true data and the model, epochs 200 and 0 | done (inside the pruning comparison) | done |
| Graph on unseen (validation) episodes | — | done, epoch 200 |
| Rule check on fixed physics (`datasets/cc`) | **to do** | — |
| 21-checkpoint training curve | **to do** | postponed |
| Interchange interventions | **to do** | **to do** |

## Method in one paragraph

1. **Simulate** each environment with hidden physics drawn per episode, and render 128×128 frames.
2. **Derive the expert graph** from the exact equations: the Jacobian of one transition, at each row's own physics, cut into strong / weak / absent tiers at a normalised effect of 0.1.
3. **Train the JEPA** on pixels only: a ViT encoder, a transformer predictor that sees the last 5 latents and the actions [u(t−1), u(t)], an inverse model, and SIGReg.
4. **Read the predicted change** with the *differential probe*. It is an MLP fitted on the encoder's real latent change z(t+1) − z(t), then frozen and applied to the predictor's change ẑ(t+1) − z(t). A target counts as testable only if the encoder's own change is readable (R² ≥ 0.5).
5. **Score every edge**, from the true simulator inputs to the read-out change, with leave-one-covariate-out (LOCO) refits and shuffled decoys over 50 random episode half-splits.
6. **Prune** with `null_cal_fwer`: an edge is kept only if it beats the largest edge statistic in every one of 20 shuffled-target repeats (max-T, exact permutation p ≤ 0.05).
7. **Compare** with the expert graph: strong-graph F1, AUROC and false positives. The untrained (epoch-0) model gives the floor.

Fixed before any result: frame skip, the 5-latent history, the [u(t−1), u(t)] action context, true-state edge inputs, the testable cut R² ≥ 0.5, the tier cut 0.1 and the 50 splits. For Acrobot the pruning rule (`null_cal_fwer`) and the differential probe were also fixed in advance, with no rule comparison.

## Folder layout

| Path | What it is |
|---|---|
| `BTP_REPORT.md` | The report: approach, environments, answer key, results, related work, limitations |
| `cc_common.py` | Cart-pole constants, Euler step, episode → (inputs, targets) stacking, dataset loading |
| `envs/` | Environment registry (`cartpole`, `acrobot`). Shared code picks the environment from the dataset's `metadata.json` (`"env"`; no key = cart-pole) |
| `sim/` | Simulators: `cartpole.py`, `acrobot.py` (Spong arm + spring-driven distractor block) |
| `configs/` | `causal_cartpole.yaml`, `causal_acrobot.yaml`: environment, physics ranges, episode mix, model and training settings |
| `checks/` | Step 1: `verify_equations.py` (cart-pole; `--env acrobot` hands over to Acrobot), `verify_acrobot.py`. Writes the expert graph |
| `expert_graph.py` | Hand-derived cart-pole graph (adjacency, initial tiers); used by the checks |
| `datagen/` | Step 2: `generate.py` (datasets), `dataset.py` (training windows) |
| `model/` | JEPA (`jepa.py`) and the SIGReg loss (`sigreg.py`) |
| `train_cc.py` | Step 3: trains the JEPA for either environment |
| `probing/` | Step 4: `extract.py` (latents, predictor output, true variables; cached), `fit_probes.py` (differential probe) |
| `discovery/` | Steps 5–6: `edge_scores.py` (LOCO + decoys), `prune.py` (all pruning rules), `null_calibration.py` (shuffled-target repeats) |
| `evaluate/compare.py` | Step 7: comparison with the expert graph, markdown tables, plots |
| `run_all.py` | Steps 4–7 end to end: data baseline, one checkpoint, probe only, or the training curve |
| `compare_pruning.py` | Cart-pole only: compares all pruning rules on the same scores (report §6) |
| `diagnose_physics.py` | Cart-pole only: does the hidden physics confound the true-data graph? (report §7) |
| `compare_pruning_phys.py` | Cart-pole only: every graph scored conditional on the episode physics (sensitivity analysis) |
| `reports/`, `research_notes/` | Literature review behind report §9 (with source tags) and its four research notes |
| `datasets/`, `runs/`, `results/`, `logs/` | Data, checkpoints and extraction caches, outputs, logs (all git-ignored) |

## Setup

Python 3.12 on Windows, PowerShell, from this folder. `requirements.txt` pins PyTorch 2.11 with CUDA 12.8, which the RTX 5090 needs.

```powershell
uv venv .venv --python 3.12            # or: python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$PY = ".\.venv\Scripts\python.exe"; $env:PYTHONUTF8 = "1"   # the scripts print Unicode
```

- **CPU only:** set `$env:CUDA_VISIBLE_DEVICES = "-1"`. Never set it to `""`: PowerShell then deletes the variable, and the GPU is used anyway.
- **GPU loss:** the RTX 5090 on this machine sometimes drops off the driver ("GPU is lost"). Kill the frozen Python process, reboot, and rerun the same command. Every long step is resumable.
- The code is self-contained. The model architecture follows the `JEPA_control` cart-pole model, but nothing is imported from it.

## Run order

Each command runs from this folder. Times are on the RTX 5090.

### Cart-pole, varied physics (main study)

```powershell
& $PY checks\verify_equations.py --vary-physics                     # ~5 min CPU -> expert graph + checks
& $PY datagen\generate.py --vary-physics                            # -> datasets/cc_phys
& $PY -m discovery.null_calibration --data datasets/cc_phys --reps 20   # ~4-5 min per repeat
& $PY train_cc.py --data datasets/cc_phys --save-dir runs/jepa_cc_phys  # ~3.5 h; --resume <ckpt> to continue
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0200.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0000.pt --probe-only --seeds 3
& $PY compare_pruning.py --data datasets/cc_phys --run-dir runs/jepa_cc_phys   # all rules; data, noise, null and model graphs
& $PY diagnose_physics.py --data datasets/cc_phys --run-dir runs/jepa_cc_phys
& $PY compare_pruning_phys.py --data datasets/cc_phys --run-dir runs/jepa_cc_phys
```

The cart-pole graphs in the report (true data, trained model, untrained model) are the `null_cal_fwer` rows of `compare_pruning.py`. The same graphs can be produced as stand-alone runs:

```powershell
& $PY run_all.py --data datasets/cc_phys --data-baseline --prune null_cal_fwer
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0200.pt --prune null_cal_fwer
```

Still to do:

```powershell
# rule check on fixed physics: needs its own null first
& $PY -m discovery.null_calibration --data datasets/cc --reps 20
& $PY compare_pruning.py --data datasets/cc --run-dir runs/jepa_cc
# 21-checkpoint curve: the null must use the same split count as the curve
& $PY -m discovery.null_calibration --data datasets/cc_phys --reps 20 --splits 20
& $PY run_all.py --data datasets/cc_phys --run-dir runs/jepa_cc_phys --curve --splits 20 --prune null_cal_fwer
```

### Acrobot

```powershell
& $PY checks\verify_equations.py --env acrobot --vary-physics       # -> expert graph + checks
& $PY datagen\generate.py --env acrobot --vary-physics              # ~35 min -> datasets/ac_phys
& $PY -m discovery.null_calibration --data datasets/ac_phys --reps 20
& $PY run_all.py --data datasets/ac_phys --data-baseline --prune null_cal_fwer
& $PY train_cc.py --data datasets/ac_phys --config configs/causal_acrobot.yaml --save-dir runs/jepa_ac_phys   # ~3.5 h
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0200.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0000.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0200.pt --prune null_cal_fwer
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0000.pt --prune null_cal_fwer
# unseen episodes: the validation split needs its own null (fewer rows)
& $PY -m discovery.null_calibration --data datasets/ac_phys --reps 20 --split val
& $PY run_all.py --data datasets/ac_phys --data-baseline --prune null_cal_fwer --split val
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0200.pt --prune null_cal_fwer --split val
```

### Cart-pole, fixed physics (first model)

`datasets/cc` and `runs/jepa_cc` are the original fixed-physics data and model: `datagen\generate.py` without `--vary-physics`, and `checks\verify_equations.py` without the flag. Report §7 uses this model's probe ("trained on fixed physics"). Every command above also works with `--data datasets/cc`. Without `--vary-physics` the generator reproduces `datasets/cc` exactly.

### Smoke test

`datagen\generate.py --small` writes a tiny cart-pole dataset (`datasets/cc_small`); `datagen\generate.py --env acrobot --vary-physics --small` writes `datasets/ac_phys_small`. Train it for one epoch with `train_cc.py ... --epochs 1`, then run `run_all.py` on it. This exercises the whole pipeline in minutes.

## Important flags and gotchas

- **Always pass `--prune null_cal_fwer`.** `run_all.py` still defaults to `margin`, the original rule, and the rule name is part of every output path.
- **The null must match the rows and the split count.** `null_cal*` rules read `results/null_calibration_<data>[_<split>]/rep###_s<splits>.json`. A run with `--splits 20` or `--split val` needs nulls made with the same option.
- **Batch-norm recalibration (Acrobot).** The Acrobot encoder's batch-norm running statistics go stale, which made eval-mode losses meaningless (val 15 instead of 0.45). For Acrobot (`RECALIBRATE_BN` in `envs/acrobot.py`), training and extraction recompute them from 200 training batches before every evaluation. Cart-pole is unaffected. Extraction cache names do not record this. Delete any Acrobot cache made before the fix (2026-11-11 15:09); the one known stale cache is already renamed `*.stale_pre_bnfix`.
- **Caches.** Extractions are cached in `runs/<run>/extract/<data>[_<split>]_<checkpoint>.npz` (~100 MB each), null repeats and comparison scores in `results/`. Finished checkpoints are skipped on a rerun.
- **Training-time probe.** `train_cc.py` logs a ridge probe (`probe_r2_x_xdot_theta_thetadot`) as a rough progress signal. For Acrobot it reads the raw 6-d state with unwrapped angles, so it is not a result.
- **Acrobot angles** enter edge scoring as a (sin θ, cos θ) pair. LOCO drops the pair, and the decoys shuffle it, as one node.

## Outputs

`<tag>` is `_<dataset name>` (empty for `datasets/cc`) plus `_<split>` when the split is not `train`.

| File | Contents | Report |
|---|---|---|
| `results/checks<_cc_phys or _ac_phys>.json` | The six equation checks | §3 |
| `results/expert_graph_final<_cc_phys or _ac_phys>.json` | Expert graph with tiers and effects (no suffix = `datasets/cc`) | §3 |
| `datasets/<data>/summary.json` | Episode and transition counts, physics ranges | §2 |
| `runs/<run>/history.json`, `config.yaml`, `checkpoints/` | Training losses per epoch, 21 checkpoints (every 10 epochs) | §4 |
| `results/<run><tag>_diff_<rule>/epoch####/probe.json` | Differential-probe R² (encoder and predictor, seed spread) | §7, §8 |
| `results/<run><tag>_diff_<rule>/epoch####/{summary.md, edges.md, results.json}` | Graph from one checkpoint: metrics, per-edge table, full scores | §8 |
| `results/data_baseline<tag>_<rule>.json` | Graph from the true targets | §8 |
| `results/null_calibration_<data>[_<split>]/` | Shuffled-target null repeats | §6 |
| `results/pruning_comparison_cc_phys_jepa_cc_phys/report.md` | All pruning rules on all validation and model graphs | §6, §7 |
| `results/physics_conditioning_cc_phys/report.md` | Hidden-physics confounding of the data graph | §7 |
| `results/pruning_comparison_phys_cc_phys_jepa_cc_phys/report.md` | Physics-conditioned rule comparison | §6 |

The report's cart-pole probe table reads `results/jepa_cc_arms/epoch0200/differential/probe.json` (fixed physics; this folder was named under the earlier three-readout design), `results/jepa_cc_cc_phys_diff_margin/epoch0200/probe.json` (fixed-physics model on varied physics) and `results/jepa_cc_phys_cc_phys_diff_margin/epoch0200/probe.json`. The probe does not depend on the pruning rule, so `_margin` in these names does not matter.

## Pruning rules (`discovery/prune.py`)

Every rule reads the same per-split scores, and none sees the expert graph.

| Rule | Keeps an edge if | Source |
|---|---|---|
| `margin` | LOCO beats the strongest shuffled decoy by log 1.5 in ≥ 70% of splits | Meinshausen & Bühlmann 2010; Barber & Candès 2015 |
| `fdr` | Episode-clustered z-test of LOCO vs the strongest decoy per split, p-values combined across splits (2 × median), BH at q = 0.05 | Lei et al. 2018; Meinshausen, Meier & Bühlmann 2009 |
| `min_effect` | As `fdr`, with H0 "explains ≤ 1% more target variance than a decoy" | Minimum-relevant-effect test |
| `effect` | Normalised Jacobian of the full fit ≥ 0.1 and above every decoy's, in ≥ 70% of splits | The expert's own tier rule |
| `effect_fdr` | `effect` and `fdr` | — |
| `knockoff` | Derandomised Gaussian model-X knockoffs, knockoff+ at q = 0.1, selected in ≥ 50% of splits | Candès et al. 2018; Ren, Wei & Candès 2023 |
| `null_cal` | The `min_effect` statistic beats the same edge's statistic in all 20 shuffled-target repeats (p = 1/21 ≤ 0.05) | Phipson & Smyth 2010 |
| **`null_cal_fwer`** (chosen) | As `null_cal`, but against the largest statistic over all edges in each repeat (max-T): 5% false positives per graph | Westfall & Young 1993 |
| `null_cal_vif_fwer` | `null_cal_fwer` with the 1% margin scaled by (1 − R²) of the input on the others (collinearity) | Sensitivity analysis only |

The rule was chosen on graphs that do not depend on the model's quality: the true data, the true data with noise matched to the model, shuffled targets, and the untrained model. Report §6 has the comparison.
