# Per-output parent masks (report route 2): design, caveats and their fixes

Status 2026-11-16: code written, not run yet. Everything runs on the training machine, in the order set by `tools/run_graph_prior_study.ps1`.

## Goal

The question is whether the cart-pole JEPA trains more efficiently when the expert graph is built into it. The comparison is against the standard JEPA and the two arms already in the `gnn` commit (f03aadb): **M**, masked attention (report route 3), and **J**, the Jacobian penalty (route 5).

This arm is **route 2, per-output parent masks**, in the style of CDL. It is built as an independent package, `models/jepa_peroutput/`. The `gnn` commit's code is not modified.

## How the arm works

- The latent z is split into one slot per state variable plus ctx:
  - cart-pole: [x | v̄ | θ | ω̄ | ctx]
  - Acrobot: [θ1 | ω̄1 | θ2 | ω̄2 | p | v̄p | ctx]
- For every slot j, one MLP f_j predicts the **change** of slot j:
  ẑ_j(t+1) = z_j(t) + f_j( slots that are parents of Δj, at every lag ; actions that are parents of Δj )
- **What each network can see:**
  - Slots that are not parents are multiplied by zero before the first layer, so they have no effect whatever the weights are.
  - z_j(t) is added back as plain integration and is not an input unless the graph lists self as a parent. For cart-pole, x→Δx and v̄→Δv̄ are absent; θ→Δθ and ω̄→Δω̄ are present.
  - Every change reads ctx, which is meant for the hidden physics. The change of ctx reads only ctx.
- **What is shared with the standard JEPA:** the encoder and the inverse model, imported from `models/jepa/model.py` without changes.
- **Loading:** checkpoints store `model_class`, and `probing/extract_jepa.load_model` uses it. `run_all.py`, the curve, the counterfactual study and `compare_training` therefore read these runs without changes.

## Caveats from the report and how each is fixed

| # | Caveat | Fix | Where |
|---|---|---|---|
| 1 | Masks must be per output, not a single input gate | One network per output slot, each with its own input mask | `model.py` |
| 2 | A residual makes every output depend on its own past | Each network predicts the change from its parents only; the own value is added back afterwards | `model.py`, `graph.py` |
| 3 | Hidden physics are parents of every change but are not graph nodes | **Built in:** every change reads ctx; the `lambda_ctx` loss keeps ctx constant within a window; `ctx_pool: true` feeds ctx to every network as its running mean, so fast state cannot ride on it. **Measured:** `evaluate/slot_selectivity.py` reports R²(ctx → state), which should be low, and R²(ctx → physics), which should be high. | `model.py`, `train.py`, `slot_selectivity.py` |
| 4 | The graph must hold at the model's time step | The expert graph is defined at frame-skip 2 (0.04 s), the model's step. Lagged inputs carry only the same parents. Actions enter only through the current context [u_{t−1}, u_t]. | `graph.py` |
| 5 | A missing true edge hurts more than an extra one | Strong and weak edges are both used | configs |
| 6 | Slots may not hold their variables (entangled latent) | **Measured:** `slot_selectivity.py` fits linear readouts from each slot to each variable, on unseen episodes. The standard JEPA cut into the same layout gives the chance level. **Fixed when needed:** anchoring with `lambda_anchor`, on all episodes (`_anchor`) or on 5% of training episodes (`_anchor05`, `anchor_fraction`). | `slot_selectivity.py`, `train.py` |
| 7 | A gain could come from the architecture, not the graph | A **dense control** uses the same networks with every input allowed. `per_output_hidden: auto` matches the standard predictor's parameter count exactly; the check requires ≤ 5%. | `model.py` |
| 8 | Multi-layer attention leaks along graph paths | Does not apply: each network sees its inputs once, with no attention | `model.py` |
| 9 | Batch-coupled layers spread a perturbation | No BatchNorm in the per-output networks | `model.py` |
| 10 | Cart-pole is dense, so a near-null in-distribution result is expected | The study tests where the report says a graph pays off. **Less data:** `datagen/subset.py`, 25% of training episodes. **Held-out physics:** `datagen/make_ood.py`, masses and gravity above the training ranges. **Long rollouts:** up to 20 steps. **Acrobot:** 22 truly absent distractor edges, with action masks per output, because the distractor has no action parent. | `datagen/`, `graph.py`, configs |
| 11 | A probe used in both the loss and the metric is circular | No probe in the loss. The anchor is a separate linear head. The evaluation uses fresh `run_all.py` probes and linear selectivity readouts. | — |
| 12 | One seed per arm | Seeds 42, 43 and 44 for standard, per-output and dense. M and J get extra seeds with `-AllArmSeeds`. `evaluate/compare_seeds.py` pairs runs by seed and reports mean ± sd, wins per seed and a Wilcoxon p. With 3 seeds the smallest possible p is 0.25, so the wins count matters more. | `compare_seeds.py`, runner |
| 13 | Loss weights were guessed | `evaluate/pick_lambda.py` takes the largest weight whose val rollout loss after 10 epochs stays within 5% of the standard run's. Used for `lambda_ctx` (per-output arm, reused for M) and `lambda_jac` (arm J). | `pick_lambda.py`, runner |
| 14 | The expert graph could be wrong | Wrong-graph controls: a **shuffled** graph with the same edge count, **2 true edges deleted** and **2 absent edges added**, run separately, as the report recommends. Diagnostics always score against the uncorrupted graph. | `graph.py`, runner |
| 15 | The parameter count was only estimated | `auto` width solves for it; `checks/verify_peroutput.py` prints both counts | `model.py` |

**Bug found in `evaluate/compare_training.py`** (`gnn` commit, left unchanged): its rollout evaluation hard-codes cart-pole's 4 state columns. `evaluate/rollout_eval.py` is a general version, and the runner compares Acrobot runs with `compare_seeds` instead.

## What the study runs (`tools/run_graph_prior_study.ps1`)

| Stage | Runs | Approx. time (RTX 5090) |
|---|---|---|
| sweep | 10-epoch runs: standard, per-output with `lambda_ctx` ∈ {0.1, 1}, J with `lambda_jac` ∈ {0.1, 1}; then the picks | 1.5 h |
| main | Seed 42, 100 epochs: standard, M, J, per-output, dense | 9 h |
| seeds | Seeds 43 and 44: standard, per-output, dense | 10 h |
| controls | Per-output with a shuffled graph, 2 edges deleted, 2 edges added | 5.5 h |
| anchor | Per-output with anchoring on all episodes, and on 5% of episodes | 3.5 h |
| lowdata | 25% of the training episodes: standard, per-output, dense | 1.5 h |
| ood | Held-out-physics evaluation sets (cart-pole, Acrobot) | 15 min |
| acrobot | 100 epochs: standard, per-output, dense | 6 h |
| analysis | See below | — |

**Analysis** covers:
- graph-recovery curves for every run (`run_all.py --curve`)
- training comparisons in distribution, on held-out physics, with low data and on Acrobot
- the seed comparison
- slot selectivity

Outputs go to `results/studies/graph_prior_training/`.

## How to read the results

1. **Selectivity first.** Per-output slots must clearly beat the null cut. Otherwise the masks do not constrain named variables, and the anchored runs are the meaningful ones.
2. **Graph versus architecture:** per-output against dense. Per-output against standard alone mixes the two.
3. **Wrong graphs:** per-output against shuffled, deleted edges and added edges. If the shuffled graph does as well as the expert graph, any gain comes from sparsity, not from the right graph.
4. **Where to look for an effect:** on cart-pole, expect it on low data, held-out physics and long rollouts. Acrobot should show it most clearly.
5. **Seeds:** trust only differences that hold for every seed.

## Verification

`python -m checks.verify_peroutput` runs on CPU with random weights and needs no data. It checks:
- the masks against the expert graph
- exact invariance to non-parents and to masked actions
- that self pairs move by exactly δ
- that every input reaches every output in the dense control
- parameter matching within 5%
- the wrong-graph controls
- anchor targets, episode ids and `--set` overrides
- that the loader hook keeps a saved graph

It needs PyTorch and has not been run.
