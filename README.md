# Causal structure learned by a cart-pole JEPA

How much of cart-pole's causal structure has a pixels-only JEPA world model learned?

Method:
1. A linear probe reads the state d = (x, v̄, θ, ω̄) from a single latent z_t.
2. The model's predicted change is Δd̂ = P(ẑ_{t+1}) − P(z_t), where ẑ_{t+1} comes from the predictor (5-latent history).
3. Every edge from [x, v̄, θ, ω̄, u_t, u_{t−1}, u_{t−2}] to [Δx, Δv̄, Δθ, Δω̄] is scored, pruned, and compared with an expert graph derived from the equations.

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

## Changes from the plan made during implementation

1. **Expert tiers are measured, not hand-assigned.** The edge structure from the hand derivation matches the exact Jacobian of the simulator (check d). The strong/weak split comes from measured effect sizes (check e): strong means a normalised effect ≥ 0.1. Six edges changed tier. For example, θ → Δv̄ is weak because the pole (0.1 kg) is much lighter than the cart (1 kg). Final graph: 10 strong, 7 weak, 11 absent (`results/expert_graph_final.json`).
2. **Edge weight = leave-one-covariate-out (LOCO), not permutation importance.** Permutation importance gave absent inputs that are correlated with a real cause (v̄ → Δθ, via ω̄) more weight than the decoys. LOCO refits without the input and measures log(held-out MSE without / with). All drop-one variants of a split are trained as one batched network from shared initial weights.
3. **Pruning margin.** An edge is kept if it beats the strongest decoy by log 1.5 (removing it makes held-out error 50% worse than removing a decoy) in at least 70% of 50 half-splits. The margin was fixed in advance, not tuned on the expert graph.
4. **Data patches.** Frames use area-averaged resizing (OpenCV is missing, and the repo's fallback picks single pixels), and the LQR gain is computed for 0.04 s steps.

## Status (2026-11-05)

- Steps 1–2 are done, and the full pipeline was smoke-tested on `datasets/cc_small`. There, the data baseline recovers the strong expert graph exactly (F1 = 1, AUROC = 1).
- Training stopped after epoch 1: the GPU dropped off the driver (`nvlddmkm` event 153, "GPU is lost"). A reboot is needed before step 3.
