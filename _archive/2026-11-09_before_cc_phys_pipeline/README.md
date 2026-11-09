# Causal structure learned by a cart-pole JEPA

How much of cart-pole's causal structure has a pixels-only JEPA world model learned?

Method (current, 2026-11-09):
1. The **differential probe** Q_D (MLP 192 → 256 → 256 → 4) reads the change Δd = (Δx, Δv̄, Δθ, Δω̄) from a latent change. It is fitted on the encoder's change z_{t+1} − z_t and then reads the predictor's change ẑ_{t+1} − z_t (5-latent history). It is the only readout: the normal and hybrid arms were removed on 2026-11-09, and the old code is in `_archive/2026-11-09_three_arms_and_old_pruning/`.
2. Every edge from the true [x, v̄, θ, ω̄, u_t, u_{t−1}, u_{t−2}] to the read-out change is scored with LOCO over 50 episode half-splits, pruned with a rule from `discovery/prune.py` (`--prune`), and compared with an expert graph derived from the equations.
3. `compare_pruning.py` compares the pruning rules on the same scores (see "Pruning rules" below).

The plan is at `C:\Users\Administrator\.claude\plans\ok-then-plan-this-enchanted-cookie.md`. This folder imports `JEPA_control` but does not modify it.

## Run order

Use the repo's virtual env: `PY=../Evaluating-Causal-Metrics-on-World-Models/JEPA_control/.venv/Scripts/python.exe`. Set `PYTHONUTF8=1` on Windows, because the repo prints Unicode arrows.

| step | command | time |
|---|---|---|
| 1 checks | `$PY checks/verify_equations.py` | ~2 min (done: 6/6 pass) |
| 2 data | `$PY datagen/generate.py` | ~3 min (done: `datasets/cc`) |
| 3 train | `$PY train_cc.py --data datasets/cc --save-dir runs/jepa_cc` | ~3.5 h on the RTX 5090 (62 s/epoch) |
| 4–7 data baseline | `$PY run_all.py --data datasets/cc --data-baseline` | GPU minutes |
| 4–7 final model | `$PY run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt` | |
| 4–7 random-init model | `$PY run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0000.pt` | |
| curve | `$PY run_all.py --data datasets/cc --run-dir runs/jepa_cc --curve --splits 20` | |

Outputs:
- `results/data_baseline.json`
- `results/jepa_cc/epoch####/{results.json, edges.md, graphs.png}`
- `results/jepa_cc/curve.{json,png}`

## Varied-physics dataset `datasets/cc_phys` (2026-11-09)

`python datagen/generate.py --vary-physics` draws each episode's physics uniformly from `physics_ranges` in the config. The physics is constant within an episode.

| parameter | range | old fixed value |
|---|---|---|
| cart mass M | 0.5 – 2.0 kg | 1.0 |
| pole mass m | 0.05 – 0.5 kg | 0.1 |
| pole half-length l | 0.25 – 0.75 m (pole 0.5 – 1.5 m, drawn at its true length) | 0.5 |
| gravity g | 4.9 – 14.7 m/s² | 9.8 |

Same episode types, counts and seeds as `datasets/cc`. The LQR gain is recomputed per episode. Each episode stores `phys_M, phys_m, phys_l, phys_g` as HDF5 attrs, and `probing/extract.py` returns them per transition as `physics` (n, 4). Without the flag the generator reproduces `datasets/cc` exactly, which was verified on `cc_small`.

The JEPA (`runs/jepa_cc`) was trained on fixed physics only, so on `cc_phys` it sees unfamiliar dynamics and pole lengths. The probe report includes change R² per third of each parameter's range (`R2_by_physics_tercile`).
```powershell
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt --probe-only --seeds 3
```
Output: `results/jepa_cc_cc_phys_diff_margin/epoch0200/probe.json`. The expert graph (`checks/verify_equations.py`) still assumes the default physics.

## Differential probe only, and new pruning rules (2026-11-09)

Run order (PowerShell, from this folder):
```powershell
$PY = ".\.venv\Scripts\python.exe"; $env:PYTHONUTF8 = "1"
& $PY compare_pruning.py --data datasets/cc --run-dir runs/jepa_cc            # rule comparison, ~80 min GPU
& $PY run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt --prune <rule>
& $PY run_all.py --data datasets/cc --run-dir runs/jepa_cc --curve --splits 20 --prune <rule>
```
Outputs: `results/pruning_comparison/report.md`; `results/jepa_cc_diff_<rule>/epoch####/…`; `results/data_baseline_<rule>.json`.

Pruning rules (`discovery/prune.py`; all read the same per-split scores, all blind to the expert graph):

| rule | keeps an edge if | source |
|---|---|---|
| margin (original) | LOCO beats the strongest shuffled decoy by log 1.5 in ≥ 70% of splits | Meinshausen & Bühlmann 2010; Barber & Candès 2015 |
| fdr | per-split episode-clustered z-test of LOCO vs the strongest decoy; p-values combined across splits (2 × median); BH at q = 0.05 | Lei et al. 2018; Williamson et al. 2023; Meinshausen, Meier & Bühlmann 2009 |
| min_effect | as fdr, but H0 is "explains ≤ 1% more target variance than a decoy" | minimum-relevant-effect test |
| effect | normalised Jacobian of the full fit ≥ 0.1 (the expert's own "strong" unit and cut-off) and above every decoy's, in ≥ 70% of splits | expert tier rule (check e) |
| effect_fdr | effect AND fdr | — |
| knockoff | derandomised Gaussian model-X knockoffs, knockoff+ (q = 0.1), selected in ≥ 50% of splits | Candès et al. 2018; Ren, Wei & Candès 2023 |

Why the rule matters here: θ and ω̄ are 98–99% predictable from the other inputs (even with open-loop forcing), so LOCO's unique information for edges out of θ and ω̄ is under 1% of target variance. The original log-ratio margin passes them on noise-free data only because the residual there is ~0. At the model's noise floor (held-out R² 0.93–0.98) the same edges fall below the margin.

## Earlier design: three readout arms (2026-11-07, superseded)

The question is whether, and how, the JEPA learns cart-pole's causal structure **during training**. The model's predicted change is read in three separate ways. Each arm trains its own MLP probe(s) (192 → 256 → 256 → out, SiLU, 40 epochs), only on **encoder** latents, cross-fitted by episode with the same 5 folds for every arm. The frozen probes then read the **predictor's** output. Each arm's results are reported separately: there is no primary arm and no cross-arm vote.

| arm | probe(s) | predictor change Δd̂ |
|---|---|---|
| normal | S_N: z → (x, v̄, θ, ω̄) | S_N(ẑ_{t+1}) − S_N(z_t) |
| differential | Q_D: z_{t+1} − z_t → Δd | Q_D(ẑ_{t+1} − z_t) |
| hybrid | H_L: z → (v̄, ω̄); H_D: Δz → (Δv̄, Δω̄) | Δx = DT·v̄(ẑ_{t+1}), Δθ = DT·ω̄(ẑ_{t+1}); Δv̄, Δω̄ = H_D(ẑ_{t+1} − z_t) |

- The hybrid's kinematic rows are exact: v̄_{t+1} = (x_{t+1} − x_t)/DT, so Δx = DT·v̄_{t+1}. This was checked on the data (max error 1.5e-8).
- The edge-scoring inputs are the **true** simulator states plus u_t, u_{t−1}, u_{t−2} for every arm and checkpoint, so only the readout of the model's change varies. `--inputs linear` gives the old linear-probed inputs as a continuity row.
- Scoring, pruning (log 1.5 margin, 70% stability) and the expert comparison are unchanged.
- Action effects are a finite difference of each arm's own readout along d ẑ / d u.

Criteria fixed in advance:
- **Testable target:** encoder-change R² ≥ 0.5. Edges into untestable targets are untestable, not absent. F1 is also reported on testable edges only.
- **Learned (per arm):** the epoch-200 predictor strong-graph AUROC is above that arm's epoch-0 floor, with no false-positive edges.
- **Emergence epoch of an edge:** the first checkpoint from which it is kept at every later checkpoint.

Run order (PowerShell, from this folder):
```powershell
$PY = ".\.venv\Scripts\python.exe"; $env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = "-1"     # CPU while the GPU is unstable; never set it to ""
& $PY run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0000.pt --probe-only --seeds 3
& $PY run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt
& $PY run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0000.pt
& $PY run_all.py --data datasets/cc --run-dir runs/jepa_cc --curve --splits 20     # stable GPU only
```
Outputs:
- `results/jepa_cc_arms/epoch####/<arm>/{results.json, edges.md, graphs.png, probe.json}` and `epoch####/summary.md`
- curve: `results/jepa_cc_arms/curve_s20/…`, `curve_<arm>.json` (including emergence epochs) and `curve.png`

Every run is **resumable**. Extractions are cached in `runs/jepa_cc/extract/` (about 100 MB each), and a finished (checkpoint, arm) pair is skipped. After a GPU loss, kill the frozen Python process, reboot and rerun the same command. The old linear readouts are still available with `--readout state|delta`.

## Changes from the plan made during implementation

1. **Expert tiers are measured, not hand-assigned.** The edge structure from the hand derivation matches the exact Jacobian of the simulator (check d). The strong/weak split comes from measured effect sizes (check e): strong means a normalised effect ≥ 0.1. Six edges changed tier. For example, θ → Δv̄ is weak because the pole (0.1 kg) is much lighter than the cart (1 kg). Final graph: 10 strong, 7 weak, 11 absent (`results/expert_graph_final.json`).
2. **Edge weight = leave-one-covariate-out (LOCO), not permutation importance.** Permutation importance gave absent inputs that are correlated with a real cause (v̄ → Δθ, via ω̄) more weight than the decoys. LOCO refits without the input and measures log(held-out MSE without / with). All drop-one variants of a split are trained as one batched network from shared initial weights.
3. **Pruning margin.** An edge is kept if it beats the strongest decoy by log 1.5 (removing it makes held-out error 50% worse than removing a decoy) in at least 70% of 50 half-splits. The margin was fixed in advance, not tuned on the expert graph.
4. **Data patches.** Frames use area-averaged resizing (OpenCV is missing, and the repo's fallback picks single pixels), and the LQR gain is computed for 0.04 s steps.

## Status (2026-11-05)

- Steps 1–2 are done, and the full pipeline was smoke-tested on `datasets/cc_small`. There, the data baseline recovers the strong expert graph exactly (F1 = 1, AUROC = 1).
- Training stopped after epoch 1: the GPU dropped off the driver (`nvlddmkm` event 153, "GPU is lost"). A reboot is needed before step 3.
