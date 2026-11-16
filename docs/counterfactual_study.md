# Counterfactual evaluation of the recovered causal graphs (JEPA, cart-pole and Acrobot)

Study date 2026-11-15. Code: `counterfactual/`, entry point `studies/counterfactual.py`. Outputs:
`results/studies/counterfactual_<data>_<run>/epoch0200_k###[_rollout]/`. Summary in BTP_REPORT.md, section 12 (not written yet; to follow the results).

## 1. Objective

The causal-discovery pipeline (differential probe → LOCO edge scores → `null_cal_fwer` pruning) produces a graph
for each world model. We have scored those graphs against the expert graphs. Edge agreement leaves three
questions open:

1. **Does the recovered graph describe the model?** A graph that is right about the physics but wrong about the
   model would look good against the expert and still be a bad audit.
2. **When the recovered graph and the expert disagree, whose fault is it?** Either the model computes the
   dynamics wrongly (a model error), or the discovery misreads a model that is right (a discovery error).
3. **Do the graphs give the right effects, not just the right edges?** An edge says only that an input matters.
   A counterfactual also asks how much, in which direction and in which state.

Counterfactual queries answer all three. A query asks, for one real window of a test episode: *had input i (a
state, an action, or the hidden physics) been different, with everything else at time t held fixed, what would
the next change have been?* The query is put to every source that can answer it, and the answers are compared
window by window.

## 2. Method

### 2.1 Query windows

A window is exactly what the JEPA sees at time t: the states s(t−5..t+1) (7 frames, giving 5 history latents)
and the actions u(t−5..t). The windows come from the **test split**, which no earlier analysis used: every usable
t, at most 3000 windows (seed 0), sampled from 3,446 usable cart-pole windows (190 episodes) and 6,350 Acrobot
windows (190 episodes).

### 2.2 Interventions with a consistent history

Each intervention has size δ = κ·sd(train input), with κ = 0.25 in the main run and 0.5 as a sensitivity run. It
is applied with both signs. Physics uses δ = κ·sd of the parameter's configured uniform range.

| Query | What is changed | What is held fixed | How |
|---|---|---|---|
| do(u(t)) | u(t) | all other inputs at t | change the last action; the history is unchanged |
| do(u(t−1)), do(u(t−2)) | that action | the frame states d(t) and the other actions | re-solve the start state s(t−5) |
| do(state node) | e.g. v̄(t) → v̄(t)+δ | the other frame states and all actions | re-solve s(t−5) |
| do(physics) | e.g. M → M+δ | d(t) and all actions | re-solve s(t−5) under the new physics |

**The re-solve.** The model cannot be given a state; it has to be given frames. Teleporting the state at t would
put a jump in the history. Instead, the start state s(t−5) is solved for, so that simulating the window forward
(with the episode physics and the counterfactual actions) reaches the intervened frame states exactly at t. This
is a square system: 4 unknowns for 4 frame states in cart-pole, 6 for 6 in Acrobot. It is solved with
vectorised damped Newton and a finite-difference Jacobian; angle residuals are wrapped. The whole 7-frame window
is then re-simulated and re-rendered, so the model sees a physically possible history.

**Rejected windows.** A window is rejected if the solve does not converge (|residual| > 1e-11 sd), if the window
leaves the dataset's state range (cart-pole |x| > 2.2 or |θ| > 1.2; Acrobot joint speed limits), or if an action
or physics value leaves its range. The effect uses whichever signs are valid.

**Factual reference.** The factual is the same pipeline with no intervention: re-simulated from the stored
s(t−5) and re-rendered. Every effect is a difference against this factual, so float32 storage drift cancels.

Variables come from `env.episode_variables`, the study's own definitions. Frames come from a GPU port of the sim
renderers, checked bit-identical to `sim/*.py` (section 2.6).

### 2.3 The sources that answer each query

| Source | What it is |
|---|---|
| **Truth** | The simulator at the episode physics |
| **Ceiling** | The encoder on the true counterfactual frames, read by the probe: Q_D(z′(t+1) − z′(t)). This is the best the readout can do |
| **Model** | The JEPA predictor on the counterfactual history and actions, read by the probe: Q_D(ẑ′(t+1) − z′(t)) |
| **Expert-graph SCM** | One masked MLP per target, seeing only its expert parents, fitted on true changes (train split) |
| **Recovered-graph SCM** | The same, with the recovered graph's parents, fitted on the model's read-out changes `dd_pred` (train split) |
| Diagnostic SCMs | recovered parents on true data (cost of graph errors alone); all inputs on model read-outs (cost of pruning); all inputs on true data (observational fit with physics hidden, i.e. confounded); and physics-conditioned versions of expert/true, recovered/model and all/true |

- **The probe.** The differential probe Q_D is fitted on the train extraction exactly as in the graph run (seed
  0, 5 episode folds). The test windows are read with the average of its 5 fold nets.
- **The SCMs.** These use the edge scorer's own masked network (`discovery.edge_scores.GroupMLP`, same training
  loop, Acrobot angles as sin/cos). There are 3 seeds, averaged. An SCM's counterfactual answer is f(o′) − f(o):
  with additive noise, the abducted noise cancels.
- **Physics conditioning.** Physics is a known input of the conditioned SCMs only. It is not a graph node, and
  the graphs are unchanged.

### 2.4 Effects and agreement

For node i and target j in one window, the effect is

  e = sign · (change′ − change_factual) / (κ · sd_train(target_j)),

averaged over the valid signs. Since δ = κ·sd(input), e estimates ∂target/∂input · sd(input)/sd(target). That is
the unit of the expert graph's effect sizes, so the numbers compare directly.

- **CE:** the mean |e| over windows, with a 95% episode-bootstrap interval (1000 resamples).
- **Counterfactual R² (CF-R²):** 1 − Σ(e_source − e_ref)² / Σ(e_ref − mean)², over windows. 1 means the source
  gives the reference's effect in every window. It is reported per cell and pooled.
- **Sign agreement:** the fraction of windows with |e_ref| > 0.05 where the signs agree.

### 2.5 The model's interventional graph and the discovery test

- **Readout floor:** the largest upper 95% bound of the ceiling CE over the expert-absent cells of testable
  targets. There the true effect is exactly zero, so anything the ceiling shows is readout artefact.
- **Model-interventional edge:** a cell whose model CE lower 95% bound is above the floor. These are the inputs
  the model demonstrably uses.
- **Discovery test:** the recovered graph is scored against the model-interventional graph. Every disagreement
  is labelled:
  - **discovery miss:** the model uses the input, but the graph lacks the edge.
  - **discovery false alarm:** the graph has the edge, but the model does not use the input.
- Crossing this with model-vs-expert separates model errors from discovery errors.

### 2.6 Rules fixed before any result

- Test split; at most 3000 windows (seed 0); κ = 0.25 main, 0.5 sensitivity.
- Testable target = train-probe encoder change R² ≥ 0.5, the existing rule. Acrobot Δp and Δv̄p are untestable
  and are left out of every model-side comparison.
- The readout floor and the model-edge rule above; 1000 bootstrap resamples; 3 SCM seeds; sign threshold 0.05.
- Recovered graph: the predictor graph, epoch 200, `null_cal_fwer`, 50 splits.
- **Pipeline sanity (a failure means a bug, not a finding):**
  - truth CE ≤ 1e-6 on every expert-absent cell;
  - Spearman ≥ 0.95 between the truth CE and the expert effect sizes on present cells;
  - the factual re-render equals the stored frames (planned tolerance: max pixel difference ≤ 2);
  - counterfactual frames equal the sim renderer's on a random sample;
  - each intervention moves only its own input, by exactly δ.

### 2.7 Multi-step rollouts

- **Queries:** do(u(t)) and do(u(t−1)), followed for k = 1, 2, 3 steps with the factual later actions. Windows
  need t+3 ≤ T.
- **Target:** the effect on the cumulative state change d(t+k) − d(t), normalised by the sd of the true factual
  change at that horizon.
- **Model:** autoregressive latent rollout (append ẑ, shift the 5-latent history), read by the probe step by
  step.
- **SCMs:** rolled forward on their own predictions.
- **What it tests:** chained paths that one-step edges cannot, e.g. u(t) → Δv̄ → x(t+2).

## 3. Results

**Status (2026-11-15): the runs are not finished, so no results are reported here yet.** Acrobot κ 0.25 is
complete; Acrobot κ 0.5 and the rollout, and all cart-pole runs, are pending (the GPU dropped and the machine
rebooted during the runs; the chains resume from their caches). BTP_REPORT.md section 12 does not exist yet.

**Deviation from a rule fixed in advance (Acrobot, found 2026-11-15).** The factual re-render differs from the
stored frames by at most 9 pixel levels on the first 200 windows, not ≤ 2 (mean difference 5.5e-7, so a handful of
edge pixels; the latents differ by at most 0.005 against an sd of 0.88). The cause is not yet tested. The likely
one is that the stored states are float32, so the re-simulated window drifts by up to about 1.6e-5 and flips
anti-aliased edge pixels. The counterfactual frames themselves are bit-identical to the sim renderer's. Every
effect is measured against the re-simulated factual, so this offset cancels in the effects, but the check as
declared did not pass and is reported as failed until the cause is confirmed.

## 4. Limitations

*(Filled in with the results.)*

## 5. Reproduce

```powershell
$PY = ".\.venv\Scripts\python.exe"; $env:PYTHONUTF8 = "1"
# prerequisite (cart-pole): the 50-split recovered graph at epoch 200
& $PY run_all.py --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0200.pt --prune null_cal_fwer
foreach ($d in @(@("cc_phys", "jepa_cc_phys"), @("ac_phys", "jepa_ac_phys"))) {
  $C = "runs/$($d[1])/checkpoints/checkpoint_epoch0200.pt"
  & $PY -m studies.counterfactual --data datasets/$($d[0]) --ckpt $C               # main, kappa 0.25
  & $PY -m studies.counterfactual --data datasets/$($d[0]) --ckpt $C --kappa 0.5   # sensitivity
  & $PY -m studies.counterfactual --data datasets/$($d[0]) --ckpt $C --rollout     # multi-step
}
```

Caches (probe, queries, per-chunk latents) are kept in `runs/<run>/counterfactual/`, so an interrupted run
resumes. With the GPU renderer, one run takes a few minutes.
