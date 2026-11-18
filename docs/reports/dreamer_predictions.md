# DreamerV3: predicted causal-extraction results over training

**Status: predictions, not results.** Written on 2026-11-17, before any probe or graph was run on a DreamerV3 checkpoint. Every DreamerV3 number below is a single expected value reasoned from the measured JEPA results and from the differences between the two architectures. None was measured. Do not quote these numbers as findings. Once the real runs exist, add the measured values next to them and report where the predictions held or failed.

State of the DreamerV3 runs when this was written:

| Run | Checkpoints present | Planned | Probe / graph run |
|---|---|---|---|
| `runs/dreamer_cc_phys` | epochs 0–250 (every 25) | 500 | no |
| `runs/dreamer_ac_phys` | epochs 0–225 (every 25) | 500 | no |

Training stopped on 2026-11-13 and has not been resumed.

## 1. Setup being predicted

Everything matches the JEPA pipeline in [BTP_REPORT.md](../../BTP_REPORT.md) except the model:

- DreamerV3 world model, PyTorch port of the official `size12m` preset (7.7M parameters), 128×128 frames, no reward, continue or policy heads. Trained offline on `cc_phys` and `ac_phys`.
- Differential probe on the full state [h, z]. The prediction is the prior at t+1 and the encoder control is the posterior at t+1. Both share h(t+1), so a z-only probe is reported as a diagnostic.
- True-state edge inputs, testable cut R² ≥ 0.5, `null_cal_fwer` pruning, the same expert graphs, nulls and data baselines.
- The training curve uses 20 episode half-splits, as for the JEPA curve. The endpoint graphs use 50 splits.

## 2. What carries over unchanged

These depend only on the data and the scoring, not on the model, so DreamerV3 shares the JEPA's values.

- **Data baselines.** Cart-pole F1 .84, 10 edges kept, 1 FP. Acrobot data F1 1.00.
- **Edges the data cannot support.** The collinear theta edges (theta → dvbar, theta → dtheta) and 6 of the 7 weak cart-pole edges are missed on true data. They should be missed at every DreamerV3 checkpoint too.
- **Shuffled-target nulls.** 0 edges kept.

## 3. Cart-pole training curve (`cc_phys`, 21 checkpoints, 20 splits)

The DreamerV3 and JEPA checkpoints are matched by fraction of training: DreamerV3 epoch 25 ↔ JEPA epoch 10, …, DreamerV3 epoch 500 ↔ JEPA epoch 200. Metrics are for the **predictor** graph against the 10 strong expert edges. F1 = 2·TP / (kept + 10). The JEPA columns are measured (`results/jepa/jepa_cc_phys_cc_phys_diff_null_cal_fwer/curve_s20/curve_summary.md`).

| DreamerV3 epoch | Strong F1 | AUROC | Kept | FP | Predictor change R² dx / dvbar / dtheta / dwbar | JEPA epoch | JEPA F1 | JEPA kept / FP |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.308 | 0.62 | 3 | 1 | 0.31 / 0.42 / 0.18 / 0.12 | 0 | 0.000 | 0 / 0 |
| 25 | 0.545 | 0.57 | 12 | 6 | 0.55 / 0.48 / −0.60 / −0.90 | 10 | 0.522 | 14 / 7 |
| 50 | 0.609 | 0.60 | 13 | 6 | 0.62 / 0.60 / −0.20 / −0.40 | 20 | 0.522 | 14 / 7 |
| 75 | 0.636 | 0.64 | 12 | 5 | 0.70 / 0.66 / 0.10 / −0.05 | 30 | 0.609 | 13 / 6 |
| 100 | 0.636 | 0.69 | 12 | 5 | 0.74 / 0.72 / 0.35 / 0.24 | 40 | 0.455 | 13 / 7 |
| 125 | 0.667 | 0.71 | 11 | 4 | 0.76 / 0.75 / 0.41 / 0.33 | 50 | 0.583 | 15 / 7 |
| 150 | 0.667 | 0.75 | 11 | 4 | 0.80 / 0.80 / 0.55 / 0.47 | 60 | 0.609 | 14 / 6 |
| 175 | 0.700 | 0.79 | 10 | 3 | 0.82 / 0.83 / 0.63 / 0.56 | 70 | 0.522 | 14 / 7 |
| 200 | 0.737 | 0.83 | 9 | 2 | 0.84 / 0.85 / 0.70 / 0.62 | 80 | 0.609 | 14 / 6 |
| 225 | 0.737 | 0.85 | 9 | 2 | 0.86 / 0.87 / 0.75 / 0.68 | 90 | 0.583 | 14 / 7 |
| 250 | 0.778 | 0.86 | 8 | 1 | 0.87 / 0.88 / 0.79 / 0.72 | 100 | 0.696 | 14 / 5 |
| 275 | 0.778 | 0.88 | 8 | 1 | 0.88 / 0.89 / 0.82 / 0.75 | 110 | 0.636 | 13 / 5 |
| 300 | 0.824 | 0.89 | 7 | 0 | 0.88 / 0.90 / 0.84 / 0.78 | 120 | 0.600 | 11 / 4 |
| 325 | 0.824 | 0.90 | 7 | 0 | 0.89 / 0.90 / 0.86 / 0.80 | 130 | 0.667 | 12 / 4 |
| 350 | 0.750 | 0.90 | 6 | 0 | 0.89 / 0.91 / 0.87 / 0.81 | 140 | 0.632 | 10 / 3 |
| 375 | 0.824 | 0.91 | 7 | 0 | 0.89 / 0.91 / 0.88 / 0.83 | 150 | 0.706 | 8 / 1 |
| 400 | 0.750 | 0.91 | 6 | 0 | 0.90 / 0.92 / 0.89 / 0.84 | 160 | 0.667 | 5 / 0 |
| 425 | 0.750 | 0.92 | 6 | 0 | 0.90 / 0.92 / 0.90 / 0.85 | 170 | 0.750 | 7 / 0 |
| 450 | 0.824 | 0.92 | 7 | 0 | 0.90 / 0.92 / 0.90 / 0.85 | 180 | 0.750 | 6 / 0 |
| 475 | 0.750 | 0.92 | 6 | 0 | 0.90 / 0.92 / 0.91 / 0.86 | 190 | 0.667 | 5 / 0 |
| 500 | 0.750 | 0.92 | 6 | 0 | 0.90 / 0.92 / 0.91 / 0.86 | 200 | 0.667 | 5 / 0 |

### Expected shape and why

- **Epoch 0 is not empty (3 kept, 1 FP).** In DreamerV3 the prediction and the encoder control share h(t+1). Even a random GRU makes h a fixed function of past frames and actions, so the untrained prediction already carries some real information (Bao 2026, preprint: random Dreamers decode energy). The JEPA's untrained predictor output lies far outside the probe's range (R² around −10⁴), so its floor is 0.
- **Early phase (epochs 25–175): many false positives, but fewer than the JEPA.** As in the JEPA, the angle changes (dtheta, dwbar) start with negative R², and edges into them are noise. The shared h damps this, so FPs peak at 6 rather than 7 and start falling earlier.
- **Clean-up (epochs 200–300).** FPs fall to 0 by epoch 300, i.e. 60% of training. For the JEPA this happens at 80% (epoch 160).
- **Late phase (epochs 300–500): 6–7 edges, 0 FP, F1 alternating .75 and .82.** The alternation is one edge going in and out, as in the JEPA's late phase, not a trend. The missing edges are the collinear theta / wbar edges.
- **Final change R² matches the JEPA (.90 / .92 / .91 / .86 vs .85 / .94 / .90 / .86).** Both models infer physics from history. DreamerV3 should be a little better on dx and a little worse on dvbar, where the blurrier prior costs accuracy.

### Endpoints at 50 splits

| Checkpoint | Strong F1 | AUROC | Kept | FP | JEPA (measured, epoch 200 / 0) |
|---|---|---|---|---|---|
| DreamerV3 epoch 500 | 0.78 | 0.90 | 8 | 1 | .75 / 6 kept / 0 FP |
| DreamerV3 epoch 0 | 0.31 | 0.62 | 3 | 1 | 0 kept |
| z-only probe, epoch 500: predictor change R² | 0.55 / 0.60 / 0.48 / 0.42 | | | | n/a |

The single FP at epoch 500 is predicted to be vbar → dwbar, the physics-confounded edge that also appears on the true data.

## 4. Acrobot endpoints (`ac_phys`, 50 splits)

The JEPA Acrobot curve is postponed, so there is no measured curve to match. Only the endpoints are predicted here. The distractor block is where the two models should differ most: DreamerV3 reconstructs pixels and cannot discard the block, whereas the JEPA's objective lets it drop the block.

| Quantity | JEPA (measured, epoch 200) | DreamerV3 (predicted, epoch 500) |
|---|---|---|
| Arm targets: predictor change R² (4 targets) | .90 / .79 / .73 / .79 | .88 / .80 / .75 / .78 |
| Arm targets: readable-target F1 / FP | 1.00 / 0 | .97 / 0 (1 miss: u_t → dw2bar) |
| Distractor encoder change R², dp / dvpbar | −0.03 / −0.03 | .80 / .68 |
| Distractor predictor change R², dp / dvpbar | not testable | .66 / .52 |
| Distractor targets testable | 0 of 2 | 2 of 2 |
| All 54 edges: strong F1 / kept / FP | .72 / – / 10 | .87 / 19 / 2 |
| Untrained graph (epoch 0): kept / FP | 4 / 2 | 5 / 2 |
| Val-split strong F1 | .74 | .85 |

The 2 remaining FPs are predicted to be arm velocity → distractor targets: with a blurrier prior, a little of the arm's motion leaks into the block's read-out.

## 5. Outcomes that would contradict these predictions

- **Epoch 0 keeps 0 edges.** Then the shared h(t+1) is not giving the untrained prediction real information. The argument for a non-empty floor is wrong.
- **Epoch 0 keeps more than 5 edges.** Then the untrained floor is too high and the trained graphs cannot be interpreted without a stronger control, such as interchange interventions.
- **The distractor stays unreadable (encoder R² below 0.5).** Then losing the distractor is not specific to the JEPA objective. It comes from the data or the training setup.
- **False positives stay above 0 after epoch 300.** Then DreamerV3 does not clean up its graph earlier than the JEPA.
- **The z-only R² comes out close to the [h, z] R².** Then the shared h(t+1) is not inflating the readout.
- **Any trained cart-pole graph beats the data baseline (F1 > .84 at 50 splits).** That would point to a scoring or probe error, not a better model.

## 6. How sure each part is

- **High:** data baselines, data-limited misses and nulls are unchanged, because they do not depend on the model.
- **Medium–high:** the distractor becomes readable in DreamerV3.
- **Medium:** the curve's shape (noisy early graphs, clean-up, then a stable 6–7 edges with 0 FP) and the final F1.
- **Low:** the non-empty epoch-0 floor, the epoch at which FPs reach 0, and every individual number in the tables.

## 7. Comparing with the real runs

1. Finish training both runs to epoch 500.
2. Run the curve with `run_all.py --data datasets/cc_phys --run-dir runs/dreamer_cc_phys --curve --splits 20 --prune null_cal_fwer`, and the endpoint graphs with `--prune null_cal_fwer` at 50 splits for epochs 500 and 0 on both datasets.
3. Add a "measured" column next to each predicted column in §3 and §4. Mark each row in §5 as held or failed. Leave the predicted values unchanged.
