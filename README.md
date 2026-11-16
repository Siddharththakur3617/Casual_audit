# Do world models learn causal structure?

A world model is trained on rendered frames only (no state input). We read a cause-and-effect graph out of its *predicted* state changes and compare it with the graph derived from the exact physics equations. Two environments: **cart-pole** and **Acrobot** (a double pendulum with an unrelated distractor object). Two world models: our **JEPA** and **DreamerV3**, both trained on the same data and scored against the same expert graphs.

**Results, discussion and related work are in [BTP_REPORT.md](BTP_REPORT.md).** This README covers what the code does, how to run it, and where each output goes.

## Status (2026-11-13)

| | Cart-pole (`datasets/cc_phys`) | Acrobot (`datasets/ac_phys`) |
|---|---|---|
| Equation checks and expert graph | done, 6/6 pass | done, 6/6 pass |
| JEPA training (200 epochs) | done (`runs/jepa_cc_phys`) | done (`runs/jepa_ac_phys`) |
| JEPA probe and graphs, epochs 200 and 0 | done | done |
| Pruning-rule choice | done: `null_cal_fwer` | fixed in advance: `null_cal_fwer` |
| JEPA graph on unseen (validation) episodes | — | done, epoch 200 |
| JEPA 21-checkpoint training curve | done (`curve_s20/curve_summary.md`; not yet in the report) | postponed |
| DreamerV3 port, smoke test | done | done |
| DreamerV3 training (500 epochs), probe and graphs | **to do** | **to do** |
| Rule check on fixed physics (`datasets/cc`) | **to do** | — |
| Counterfactual queries (JEPA, epoch 200; `docs/counterfactual_study.md`) | done | done |
| Interchange interventions | **to do** | **to do** |
| Graph prior during training (masked attention, Jacobian penalty; see below) | code written, **not run** | — |

## Method in one paragraph

1. **Simulate** each environment with hidden physics drawn per episode, and render 128×128 frames.
2. **Derive the expert graph** from the exact equations: the Jacobian of one transition, at each row's own physics, cut into strong / weak / absent tiers at a normalised effect of 0.1.
3. **Train a world model** on pixels only: the JEPA (ViT encoder, transformer predictor over the last 5 latents and the actions [u(t−1), u(t)], inverse model, SIGReg) or DreamerV3 (recurrent state-space model with image reconstruction).
4. **Read the predicted change** with the *differential probe*. It is an MLP fitted on the model's real latent change z(t+1) − z(t) (encoder side), then frozen and applied to the predicted change ẑ(t+1) − z(t). A target counts as testable only if the encoder-side change is readable (R² ≥ 0.5).
5. **Score every edge**, from the true simulator inputs to the read-out change, with leave-one-covariate-out (LOCO) refits and shuffled decoys over 50 random episode half-splits.
6. **Prune** with `null_cal_fwer`: an edge is kept only if it beats the largest edge statistic in every one of 20 shuffled-target repeats (max-T, exact permutation p ≤ 0.05).
7. **Compare** with the expert graph: strong-graph F1, AUROC and false positives. The untrained (epoch-0) model gives the floor.

Fixed before any result: frame skip, the 5-latent history, the [u(t−1), u(t)] action context, true-state edge inputs, the testable cut R² ≥ 0.5, the tier cut 0.1 and the 50 splits. For Acrobot the pruning rule (`null_cal_fwer`) and the differential probe were also fixed in advance, with no rule comparison. DreamerV3's settings are fixed in `configs/models/dreamer_*.yaml` (see "DreamerV3" below).

## Folder layout

```
causal_structure_cartpole/
├── README.md  BTP_REPORT.md  requirements.txt
├── run_all.py          Steps 4-7 for any model: probe -> edge scores -> pruning -> comparison
├── paths.py            every folder and output path, in one place
├── common.py           generic helpers: load a dataset split, stack episodes into edge inputs/targets
├── configs/
│   ├── envs/           cartpole.yaml, acrobot.yaml: environment, physics ranges, dataset policies
│   └── models/         jepa_<env>.yaml, dreamer_<env>.yaml: model + training settings
├── envs/               environment registry: variables, names, graph layout (cartpole.py, acrobot.py)
├── sim/                simulators and renderers (cartpole.py, acrobot.py)
├── checks/             Step 1: equation checks and the expert graph (verify_equations.py, verify_acrobot.py,
│                       expert_graph.py = hand-derived cart-pole graph); verify_graph_mask.py (graph-prior wiring)
├── datagen/            Step 2: generate.py (datasets), windows.py (JEPA training windows),
│                       episodes.py (whole-episode batches for DreamerV3)
├── models/
│   ├── jepa/           model.py, sigreg.py, train.py, graph.py (slot layout + expert-graph mask)
│   └── dreamer/        model.py (DreamerV3 world model + LaProp optimizer), train.py
├── probing/            Step 4: extract.py (picks the extractor by model type), extract_jepa.py,
│                       extract_dreamer.py, fit_probes.py (differential probe)
├── discovery/          Steps 5-6: edge_scores.py (LOCO + decoys), prune.py (pruning rules),
│                       null_calibration.py (shuffled-target repeats)
├── evaluate/           Step 7: compare.py (comparison with the expert graph, tables, plots);
│                       compare_training.py (standard vs graph-prior JEPA training runs)
├── counterfactual/     counterfactual queries: queries.py (windows, re-solved interventions, simulation,
│                       GPU renderer), model_jepa.py, scm.py (graph-restricted SCMs), metrics.py
├── studies/            side studies: cartpole_pruning_rules.py (rule comparison, report §6),
│                       cartpole_physics_confounding.py (report §7), cartpole_pruning_rules_phys.py,
│                       counterfactual.py (counterfactual study, both envs, report §12)
├── docs/               counterfactual_study.md, literature_review.md, research_notes/, curve_viz.html
├── datasets/           generated data                                   (git-ignored)
├── runs/               <model>_<data>/: checkpoints, config.yaml, history.json, extract/ caches;
│                       _smoke/ holds smoke-test runs                     (git-ignored)
├── results/            every output, see "Outputs"                       (git-ignored)
└── logs/               console logs                                      (git-ignored)
```

Shared code picks the environment from the dataset's `metadata.json` (`"env"`; no key = cart-pole), and the model from the checkpoint's `model_type` (no key = JEPA).

## Setup

Python 3.12 on Windows, PowerShell, from this folder. `requirements.txt` pins PyTorch 2.11 with CUDA 12.8, which the RTX 5090 needs.

```powershell
uv venv .venv --python 3.12            # or: python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$PY = ".\.venv\Scripts\python.exe"; $env:PYTHONUTF8 = "1"   # the scripts print Unicode
```

- **Run everything from this folder** as modules (`& $PY -m package.script`); only `run_all.py` is a plain script.
- **CPU only:** set `$env:CUDA_VISIBLE_DEVICES = "-1"`. Never set it to `""`: PowerShell then deletes the variable, and the GPU is used anyway.
- **GPU loss and crashes:** the RTX 5090 on this machine sometimes drops off the driver ("GPU is lost"), and on 2026-11-13 three runs died with native crashes ("Illegal instruction", "access violation") in plain NumPy and PyTorch code. Kill a frozen Python process, reboot if the GPU is lost, and rerun the same command. Every long step is resumable (training with `--resume <run>/checkpoints/checkpoint_last.pt`).
- The code is self-contained. The JEPA follows the `JEPA_control` cart-pole model and DreamerV3 follows the official implementation, but nothing is imported from either.

## Run order

Each command runs from this folder. Times are on the RTX 5090.

### Cart-pole, varied physics (main study)

```powershell
& $PY -m checks.verify_equations --vary-physics                     # ~5 min CPU -> expert graph + checks
& $PY -m datagen.generate --vary-physics                            # -> datasets/cc_phys
& $PY -m discovery.null_calibration --data datasets/cc_phys --reps 20   # ~4-5 min per repeat
& $PY -m models.jepa.train --data datasets/cc_phys --config configs/models/jepa_cartpole.yaml --save-dir runs/jepa_cc_phys  # ~3.5 h
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0200.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0000.pt --probe-only --seeds 3
& $PY -m studies.cartpole_pruning_rules --data datasets/cc_phys --run-dir runs/jepa_cc_phys   # all rules; data, noise, null and model graphs
& $PY -m studies.cartpole_physics_confounding --data datasets/cc_phys --run-dir runs/jepa_cc_phys
& $PY -m studies.cartpole_pruning_rules_phys --data datasets/cc_phys --run-dir runs/jepa_cc_phys
# 21-checkpoint curve: the null must use the same split count as the curve
& $PY -m discovery.null_calibration --data datasets/cc_phys --reps 20 --splits 20
& $PY run_all.py --data datasets/cc_phys --run-dir runs/jepa_cc_phys --curve --splits 20 --prune null_cal_fwer
```

The cart-pole graphs in the report (true data, trained model, untrained model) are the `null_cal_fwer` rows of `studies.cartpole_pruning_rules`. The same graphs can be produced as stand-alone runs:

```powershell
& $PY run_all.py --data datasets/cc_phys --data-baseline --prune null_cal_fwer
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0200.pt --prune null_cal_fwer
```

Still to do: the rule check on fixed physics, which needs its own null first.

```powershell
& $PY -m discovery.null_calibration --data datasets/cc --reps 20
& $PY -m studies.cartpole_pruning_rules --data datasets/cc --run-dir runs/jepa_cc
```

### Acrobot

```powershell
& $PY -m checks.verify_equations --env acrobot --vary-physics       # -> expert graph + checks
& $PY -m datagen.generate --env acrobot --vary-physics              # ~35 min -> datasets/ac_phys
& $PY -m discovery.null_calibration --data datasets/ac_phys --reps 20
& $PY run_all.py --data datasets/ac_phys --data-baseline --prune null_cal_fwer
& $PY -m models.jepa.train --data datasets/ac_phys --config configs/models/jepa_acrobot.yaml --save-dir runs/jepa_ac_phys   # ~3.5 h
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0200.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0000.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0200.pt --prune null_cal_fwer
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0000.pt --prune null_cal_fwer
# unseen episodes: the validation split needs its own null (fewer rows)
& $PY -m discovery.null_calibration --data datasets/ac_phys --reps 20 --split val
& $PY run_all.py --data datasets/ac_phys --data-baseline --prune null_cal_fwer --split val
& $PY run_all.py --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0200.pt --prune null_cal_fwer --split val
```

### DreamerV3

DreamerV3 is trained offline on the same datasets as the JEPA, so the expert graphs, nulls and data baselines above are reused unchanged. The extractor produces exactly the JEPA's rows (checked).

```powershell
# cart-pole (~25 s per epoch -> ~3.5 h)
& $PY -m models.dreamer.train --data datasets/cc_phys --config configs/models/dreamer_cartpole.yaml --save-dir runs/dreamer_cc_phys
& $PY run_all.py --data datasets/cc_phys --ckpt runs/dreamer_cc_phys/checkpoints/checkpoint_epoch0500.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/cc_phys --ckpt runs/dreamer_cc_phys/checkpoints/checkpoint_epoch0000.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/cc_phys --ckpt runs/dreamer_cc_phys/checkpoints/checkpoint_epoch0500.pt --prune null_cal_fwer
& $PY run_all.py --data datasets/cc_phys --ckpt runs/dreamer_cc_phys/checkpoints/checkpoint_epoch0000.pt --prune null_cal_fwer
# Acrobot (~6 h): the same with datasets/ac_phys, configs/models/dreamer_acrobot.yaml, runs/dreamer_ac_phys
# curve (cart-pole): 21 checkpoints, nulls with --splits 20 already exist
& $PY run_all.py --data datasets/cc_phys --run-dir runs/dreamer_cc_phys --curve --splits 20 --prune null_cal_fwer
```

Fixed before any result (2026-11-13):

- **Model.** The official world model at preset `size12m` (deter 2048, hidden 256, 32 latents × 16 classes, 8 blocks, 1% unimix, CNN depth 16), 7.7M parameters, 128×128 frames (one extra encoder/decoder stage), no reward, continue or policy heads. Losses: reconstruction 1.0, dyn 1.0, rep 0.1, free nats 1. Optimizer: LaProp, lr 4e-5, eps 1e-20, AGC 0.3, 1000 warmup steps. Deviations from the official code are listed in `models/dreamer/model.py` (`DEVIATIONS`).
- **Training.** 500 epochs (one epoch = every training episode once, as whole episodes from t = 0; batch 16, ~95 steps), checkpoints every 25 epochs (21 in all, epoch 0 = untrained floor). The trained result is the last checkpoint, epoch 500. Seed 42.
- **Readout.** Episodes are filtered from t = 0 with the posterior *mode* fed back, and features are [h, z-probabilities] (2560 dims). z(t) = posterior at t; z(t+1) = posterior at t+1 (sees frame t+1, the encoder control); ẑ(t+1) = prior at t+1 (no frame t+1, the prediction). Prior and posterior share h(t+1), so their difference is in z only. The same probe on the z part alone is stored as `probe.diag_z_only` (a diagnostic; no edges are scored on it).
- Everything else is as for the JEPA: differential probe, testable cut 0.5, 50 splits (20 for the curve), `null_cal_fwer`, tier cut 0.1, true-state edge inputs, train split.
- `runs/<run>/reports/epoch####.png` shows real frames, posterior reconstructions and a 10-step open-loop prior rollout every 25 epochs.

### Counterfactual queries (JEPA)

Recovered graph vs expert graph vs the model itself, on interventions with a physically consistent history (method and results: `docs/counterfactual_study.md`). It needs the 50-split `null_cal_fwer` graph of the checkpoint (the `run_all.py --prune null_cal_fwer` lines above). It takes a few minutes per run on the GPU and is resumable (caches in `runs/<run>/counterfactual/`).

```powershell
foreach ($d in @(@("cc_phys", "jepa_cc_phys"), @("ac_phys", "jepa_ac_phys"))) {
  $C = "runs/$($d[1])/checkpoints/checkpoint_epoch0200.pt"
  & $PY -m studies.counterfactual --data datasets/$($d[0]) --ckpt $C               # main, kappa 0.25
  & $PY -m studies.counterfactual --data datasets/$($d[0]) --ckpt $C --kappa 0.5   # sensitivity
  & $PY -m studies.counterfactual --data datasets/$($d[0]) --ckpt $C --rollout     # multi-step (k = 1..3)
}
```

Smoke test: `--data datasets/cc_small --ckpt runs/_smoke/jepa_cc_small/checkpoints/checkpoint_epoch0001.pt --graph expert --windows 60` (`--graph expert` stands in for a recovered graph the smoke run does not have).

### Graph prior during training (cart-pole JEPA)

Does giving the JEPA the expert graph while it trains make training more efficient? Plan and design reasons: `docs/graph_prior_training_plan.md`. Both graph arms keep the encoder, rollout loss, inverse model and SIGReg. They split z into slots `[x | v̄ | θ | ω̄ | ctx]` (40/40/40/40/32; ctx is meant for hidden physics), with parents from `checks/expert_graph.py` (strong + weak edges, `models/jepa/graph.py`):

- **masked** (`jepa_cartpole_graphmask.yaml`): tokens are (time, slot), and attention is masked so a slot reads only its parents, itself and ctx (`GraphMaskedPredictor` in `models/jepa/model.py`).
- **jacobian** (`jepa_cartpole_graphjac.yaml`): the standard predictor, plus a finite-difference penalty on the sensitivity of predicted slot j to every non-parent slot i (`lambda_jac`, in `models/jepa/train.py`).

Both arms also use `lambda_ctx`, which keeps ctx constant within a window. The only absent state edges are x→v̄, x→θ, x→ω̄, v̄→θ and v̄→ω̄, so a small or null in-distribution gain is a possible outcome.

```powershell
& $PY -m checks.verify_graph_mask                       # CPU, no data: mask, shapes, penalty estimator
# smoke (1 epoch on datasets/cc_small)
& $PY -m models.jepa.train --data datasets/cc_small --config configs/models/jepa_cartpole_graphmask.yaml --save-dir runs/_smoke/jepa_gmask --epochs 1
& $PY -m models.jepa.train --data datasets/cc_small --config configs/models/jepa_cartpole_graphjac.yaml  --save-dir runs/_smoke/jepa_gjac  --epochs 1
# main comparison, 100 epochs, seed 42 (~1.75 h each)
& $PY -m models.jepa.train --data datasets/cc_phys --config configs/models/jepa_cartpole.yaml           --save-dir runs/jepa_cc_phys_e100       --epochs 100 --probe-every 5
& $PY -m models.jepa.train --data datasets/cc_phys --config configs/models/jepa_cartpole_graphmask.yaml --save-dir runs/jepa_cc_phys_gmask_e100
& $PY -m models.jepa.train --data datasets/cc_phys --config configs/models/jepa_cartpole_graphjac.yaml  --save-dir runs/jepa_cc_phys_gjac_e100
foreach ($r in "jepa_cc_phys_e100", "jepa_cc_phys_gmask_e100", "jepa_cc_phys_gjac_e100") {
  & $PY run_all.py --data datasets/cc_phys --run-dir runs/$r --curve --splits 20 --prune null_cal_fwer
}
& $PY -m evaluate.compare_training --data datasets/cc_phys --runs runs/jepa_cc_phys_e100 runs/jepa_cc_phys_gmask_e100 runs/jepa_cc_phys_gjac_e100 --labels standard masked jacobian
```

Outputs: `results/studies/graph_prior_training/` (`training.png`, `probe_r2.png`, `rollout.png`, `graph.png`, `summary.md`, `summary.json`). Each `history.json` row now also has `train_s` and `wall_s`. Slot-layout runs also log `graph_sensitivity_val` on probe epochs. Optional second seed: rerun the baseline and the better arm with `--seed 43` into new run dirs.

### Cart-pole, fixed physics (first model)

`datasets/cc` and `runs/jepa_cc` are the original fixed-physics data and model: `-m datagen.generate` without `--vary-physics`, and `-m checks.verify_equations` without the flag. Report §7 uses this model's probe ("trained on fixed physics"). Every command above also works with `--data datasets/cc`. Without `--vary-physics` the generator reproduces `datasets/cc` exactly.

### Smoke test

`-m datagen.generate --small` writes a tiny cart-pole dataset (`datasets/cc_small`); `-m datagen.generate --env acrobot --vary-physics --small` writes `datasets/ac_phys_small`. Train either model for one epoch into `runs/_smoke/` (its results then go to `results/_smoke/`), then run `run_all.py` on it. This exercises the whole pipeline in minutes:

```powershell
& $PY -m models.dreamer.train --data datasets/cc_small --config configs/models/dreamer_cartpole.yaml --save-dir runs/_smoke/dreamer_cc_small --epochs 1
& $PY run_all.py --data datasets/cc_small --ckpt runs/_smoke/dreamer_cc_small/checkpoints/checkpoint_epoch0001.pt --splits 2 --prune margin
```

## Important flags and gotchas

- **Always pass `--prune null_cal_fwer`.** `run_all.py` still defaults to `margin`, the original rule, and the rule name is part of every output path.
- **The null must match the rows and the split count.** `null_cal*` rules read `results/nulls/null_calibration_<data>[_<split>]/rep###_s<splits>.json`. A run with `--splits 20` or `--split val` needs nulls made with the same option. Nulls depend only on the dataset, so every model shares them.
- **Batch-norm recalibration (JEPA on Acrobot).** The Acrobot JEPA encoder's batch-norm running statistics go stale, which made eval-mode losses meaningless (val 15 instead of 0.45). For Acrobot (`RECALIBRATE_BN` in `envs/acrobot.py`), JEPA training and extraction recompute them from 200 training batches before every evaluation. Cart-pole is unaffected, and DreamerV3 has no batch norm. Extraction cache names do not record this. Delete any Acrobot cache made before the fix (2026-11-11 15:09); the one known stale cache is already renamed `*.stale_pre_bnfix`.
- **Caches.** Extractions are cached in `runs/<run>/extract/<data>[_<split>]_<checkpoint>.npz`: ~100 MB for the JEPA, ~1.5 GB (cart-pole) and ~3 GB (Acrobot) for DreamerV3, whose features are 2560-d. A DreamerV3 curve therefore needs ~30 GB of disk on cart-pole. Null repeats and comparison scores are cached in `results/`. Finished checkpoints are skipped on a rerun.
- **Training-time probe (JEPA).** `models.jepa.train` logs a ridge probe (`probe_r2_x_xdot_theta_thetadot`) as a rough progress signal. For Acrobot it reads the raw 6-d state with unwrapped angles, so it is not a result.
- **Acrobot angles** enter edge scoring as a (sin θ, cos θ) pair. LOCO drops the pair, and the decoys shuffle it, as one node.

## Outputs

`<tag>` is `_<dataset name>` (empty for `datasets/cc`) plus `_<split>` when the split is not `train`. `<model>` is `jepa` or `dreamer`.

| File | Contents | Report |
|---|---|---|
| `results/expert_graphs/checks<_cc_phys or _ac_phys>.json` | The six equation checks | §3 |
| `results/expert_graphs/expert_graph_final<_cc_phys or _ac_phys>.json` | Expert graph with tiers and effects (no suffix = `datasets/cc`) | §3 |
| `datasets/<data>/summary.json` | Episode and transition counts, physics ranges | §2 |
| `runs/<run>/history.json`, `config.yaml`, `checkpoints/` | Training losses per epoch, 21 checkpoints | §4 |
| `results/<model>/<run><tag>_diff_<rule>/epoch####/probe.json` | Differential-probe R² (encoder and predictor, seed spread) | §7, §8 |
| `results/<model>/<run><tag>_diff_<rule>/epoch####/{summary.md, edges.md, results.json}` | Graph from one checkpoint: metrics, per-edge table, full scores | §8 |
| `results/<model>/<run><tag>_diff_<rule>/curve.json`, `curve_s<splits>/` | Training curve | — |
| `results/data_baselines/data_baseline<tag>_<rule>.json` | Graph from the true targets | §8 |
| `results/nulls/null_calibration_<data>[_<split>]/` | Shuffled-target null repeats | §6 |
| `results/studies/pruning_comparison_cc_phys_jepa_cc_phys/report.md` | All pruning rules on all validation and model graphs | §6, §7 |
| `results/studies/physics_conditioning_cc_phys/report.md` | Hidden-physics confounding of the data graph | §7 |
| `results/studies/pruning_comparison_phys_cc_phys_jepa_cc_phys/report.md` | Physics-conditioned rule comparison | §6 |
| `results/studies/counterfactual_<data>_<run>/epoch####_k###[_rollout]/` | Counterfactual study: `report.md` (all sections), `discovery_vs_model.md`, `confounding.md`, `physics.md`, `rollout.md`, `ce_heatmaps.png`, `results.json` | §12 |
| `results/_smoke/` | Results of runs in `runs/_smoke/` | — |

The report's cart-pole probe table reads `results/jepa/jepa_cc_arms/epoch0200/differential/probe.json` (fixed physics; this folder was named under the earlier three-readout design), `results/jepa/jepa_cc_cc_phys_diff_margin/epoch0200/probe.json` (fixed-physics model on varied physics) and `results/jepa/jepa_cc_phys_cc_phys_diff_margin/epoch0200/probe.json`. The probe does not depend on the pruning rule, so `_margin` in these names does not matter.

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
