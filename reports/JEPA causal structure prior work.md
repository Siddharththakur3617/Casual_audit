# Five preprints come close; none run this audit

We found no published work that does what this study does. No paper takes an unconstrained, pixels-only, action-conditioned world model, reads its *predicted state change* through a probe fitted on the encoder and frozen, extracts an edge-level state/action → Δstate graph, and scores it against an exact simulator-Jacobian answer key under null-calibrated familywise error control with an untrained-model floor. The closest work is **Tan et al. 2026 ("What Can Latent World Models Know?", PokeWorld)**. It uses the same model family (action-conditioned JEPA, causal-transformer predictor, SIGReg) with hidden per-episode mass, drag and stiffness and an untrained reference, and it finds that what a probe can read and what the predictor uses come apart. Four other 2026 preprints each match one ingredient. Delta-JEPA has the encoder-Δz → Δstate probe. Bao has the argument that random-init decodability is uninformative, so evidence must come from the transition. Gonzalez Laiz & Schneider use the same CartPole + LeWM setting and ask whether the dynamics are faithful. Pandaram et al. build a simulator-Jacobian ground truth that includes cart-pole. The study's defensible novelty is therefore in **how the parts are assembled and where they are applied, not in any single test**. LOCO, shuffled decoys, response permutation, Westfall–Young max-T and exact permutation p-values are all textbook, and the premise that "a random encoder already decodes the state" is now independently established. The method's most exposed points are statistical and all have known fixes:
- max-T over row-shuffled targets controls per-edge FWER only under subset pivotality;
- 20 null repeats give a p-value floor of 1/21 ≈ 0.048;
- rows are shuffled across autocorrelated episodes;
- observational LOCO cannot separate θ from ω̄, which are about 95% predictable from the other inputs.

Almost every close neighbour is an **unreviewed 2026 arXiv preprint**, and the literature search was bounded, so "no precedent found" is not proof that none exists.

**How to read the source tags.** **[V]** means the research session opened the primary page (arXiv abstract or HTML, or the venue page). **[A]** means the session saw only an abstract, listing or search snippet. **[M]** means a well-known paper cited from the researchers' memory and not re-opened. Venue status is given where known. "Preprint" means no peer-reviewed venue was listed when checked. Most 2026 items were posted after this writer's own training data ends, so their content rests entirely on the research notes. Study numbers come from the project's own report ([BTP report](../BTP_REPORT.md)).

## PokeWorld is the nearest neighbour; four preprints each own one ingredient

**1. Tan et al. 2026, "What Can Latent World Models Know? Physical Information in Multimodal Predictive Representations" [V, preprint v4 Sep 2026] — very close.**

What it shares with our study:
- **Model family.** Action-conditioned JEPAs with a causal-transformer predictor, multi-horizon latent heads and SIGReg.
- **Hidden physics.** In the 2D PokeWorld pushing environment, objects look identical but carry **per-episode hidden mass, drag and stiffness**.
- **Untrained reference.** It shows that state-like readout comes free: on RH20T real-robot data, current-force readout is **0.91 for the untrained model vs 0.88 for the trained one** ([arXiv 2607.27017](https://arxiv.org/abs/2607.27017)).
- **Tests the forecasts, not the representation.** It regresses the predicted displacement or force on each true parameter.
  - Drag readout is weak at **0.12**, yet 16-step glide forecasts have a **slope ratio of 1.04** against the true dynamics.
  - A coordinate-target model reads drag at **0.41** but forecasts with a slope ratio of only **0.12**.
  - The authors conclude that "readability does not guarantee correct use" ([arXiv 2607.27017](https://arxiv.org/abs/2607.27017)).

This is the same epistemic move our study makes when it rejects state probes in favour of the predictor's change. It is the paper a reviewer will most likely hold up against ours.

What it does not do:
- It does not fit a Δ-probe on the encoder and transfer it to the predictor.
- It does not score a variable-to-variable graph.
- It has no simulator-Jacobian answer key and no formal false-positive control.
- Its object is dependence on physical *parameters*. Ours is the state/action → Δstate *structure*.

**2. Zhang, Wang et al. 2026, "Delta-JEPA" [V, preprint Jun 2026] — very close on probe construction and architecture.**
- **Architecture.** ViT-tiny encoder, 6-layer causal-transformer predictor with AdaLN actions, and an inverse-model decoder that recovers the action from Δz. This nearly reproduces our encoder–predictor–inverse-loss recipe, without SIGReg.
- **Probe.** It explicitly trains frozen-encoder probes "to predict state changes Δx_t = x_{t+1} − x_t from latent displacements Δz_t = z_{t+1} − z_t". Two-Room reaches linear **r 0.992 vs 0.765** for LeWM ([arXiv 2606.31232](https://arxiv.org/abs/2606.31232)).
- **Where it stops.** The probe is applied only to the encoder's Δz, never to the predictor's ẑ(t+1) − z(t). Predictor action-sensitivity is shown only qualitatively. There is no random-init control.
- **Consequence for us.** The *encoder half* of our differential probe has a direct precedent and must be cited as such. Our contribution is the *transfer* to the predictor and what we build on top of it.

**3. Bao 2026, "Correcting a learned physical invariant improves world-model rollouts" [V, preprint Aug 2026] — very close in logic.**
- **Setup.** DreamerV3 on 64×64 Pendulum video.
- **Random-init control.** **Six randomly initialised Dreamers** already give energy-correlated readouts of |ρ| 0.170–0.908, with 4 of 6 above 0.7. In the author's words: "Decodability alone does not explain this."
- **Moving the test to the transition.** Bao searches for a scalar conserved under the model's *own one-step transition*. Trained models reach |ρ| 0.967–0.975 with the true energy ([arXiv 2608.23526](https://arxiv.org/abs/2608.23526)).
- **Further controls.** Damped-pendulum "no invariant exists" models and 20 norm-matched random constraints.
- **Relation to our study.** It is the strongest independent support for our premise and our choice of floor. But it targets an invariant, not a graph. It has one system, no actions and three seeds.

**4. Gonzalez Laiz & Schneider 2026, "Understanding the Limits of Latent World Models" [V, preprint Oct 2026] — very close in setting.**
- **Setup.** DreamerV3, TD-MPC2 and LeWM, four seeds each, from states and from pixels, on DMC **CartPole-swingup**, Pendulum and Reacher.
- **Probes.** MLP probes on frozen encoder latents. Controls are a shuffled-target normalisation and a state-MLP topline.
- **Dynamics metric.** A "latent rollout consistency" score. Probes are applied to predicted latents only qualitatively.
- **Findings.** Pixel velocity error is about **15–22× the state-model median**. The authors conclude that "none solves the task by actually faithfully modeling the dynamics of the environment" ([arXiv 2610.37588](https://arxiv.org/abs/2610.37588)).
- **Tension with our result.** We find predictor-change R² of 0.85–0.94 on cart-pole ([BTP report](../BTP_REPORT.md)). The write-up should address this directly. Their test is multi-step and reward-driven. Ours is a one-step change on exploration data with frame-averaged velocities that are observable from two frames.

**5. Pandaram et al. 2025, "Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic RL Benchmarks" [V, preprint Nov 2025] — very close on the answer key, partial overall.**
- **Ground truth from simulator Jacobians.** They differentiate MJX (differentiable MuJoCo) over about 11M steps per environment, on eight DMC tasks including **CartpoleBalance**. "Global sparsity is rare."
- **Unconstrained networks miss the zeros.** A 2×512 MLP trained on true states misses the true sparsity "by multiple orders of magnitude" ([arXiv 2511.08086](https://arxiv.org/html/2511.08086v2)).
- **Pixels excluded.** They rule out pixels because observations are not invertible functions of state ([arXiv 2511.08086](https://arxiv.org/html/2511.08086v2)).
- **How our design answers that objection.** We never take a Jacobian through pixels. The predictor's change is read out into simulator variables and regressed on true simulator states. Their failure to recover exact zeros also supports our choice of a statistical test over raw Jacobian thresholding.

**6–7. The causal-representation-learning neighbours: TDRL and mechanism sparsity.**

TDRL (Yao, Chen & Zhang, NeurIPS 2022) [V]:
- It is the closest *environment* match. Its modified Cartpole is learned **from video under six hidden gravity domains (5–30)** with left/right actions, and two latents align with cart position and pole angle ([arXiv 2210.13647](https://www.arxiv.org/abs/2210.13647)).
- It builds identifiability into a bespoke model and scores variable recovery. We could not confirm that it scores a transition graph against ground truth.

Lachapelle et al. (CLeaR 2022; JMLR 2026) [V]:
- It learns binary **action→latent and latent→latent masks**, which is exactly the object of our graph.
- The JMLR version proves only *partial* disentanglement when sparsity or intervention diversity is insufficient ([arXiv 2107.10098](https://arxiv.org/abs/2107.10098v2); [JMLR](https://jmlr.org/papers/v27/24-0771.html)). That is a formal analogue of our θ/ω̄ collinearity problem.

Both learn structure *by construction*. Neither audits a generic world model after training.

**Partially related, useful for positioning.**

JEPA and video-model probing:
- **PSG-JEPA** [V, preprint] uses pairwise (z_t, z_{t+1}) encoder probes for velocity, with no random-init control ([arXiv 2608.06799](https://arxiv.org/abs/2608.06799)).
- **LeWorldModel** [V, preprint] is our architecture template. Its state-probe tables have no random-init control in the main text ([arXiv 2603.19312](https://arxiv.org/abs/2603.19312)).
- **Joseph et al., "Interpreting Physics in Video World Models"** [V, **ICML 2026, peer reviewed**] works on frozen V-JEPA 2. Motion direction emerges only at about 1/3 of network depth, as a ~40–50-dimensional ring-like code, and the authors conclude the model does not use factorised, physics-engine-style variables ([arXiv 2602.07050](https://arxiv.org/abs/2602.07050)).
- **RigidBench** [V, preprint] reports a random-init DiT reaching position R² **0.388 vs 0.709** trained ([arXiv 2608.15555](https://arxiv.org/abs/2608.15555)).

Testing whether the dynamics, not the state, were learned:
- **Vafa et al., "What Has a Foundation Model Found?"** [V via search, ICML 2025] shows an orbit-trained transformer predicts trajectories well while implying a "warped" force law ([arXiv 2507.06952](https://arxiv.org/abs/2507.06952)).
- **Schäfer et al.** [A, RSS 2026 workshop] find that imagined DreamerV3 rollouts stay flat across a friction sweep ([arXiv 2607.05966](https://arxiv.org/abs/2607.05966)).

Extracting dependencies from trained models:
- **"Transformer Is Inherently a Causal Learner"** [A, preprint] argues that gradient sensitivities of an *unconstrained* autoregressive transformer identify true lagged parents, with a recency bias toward recent lags ([arXiv 2601.05647](https://arxiv.org/html/2601.05647)). This is relevant to our u_{t−1}/u_{t−2} edges.
- **Liu & Chen 2026** [A, preprint] patch a rank-4 "velocity carrier" subspace in a collision world model. Their position edits fail specificity controls ([arXiv 2608.15156](https://arxiv.org/abs/2608.15156)).

The MBRL causal-dynamics line includes CDL (ICML 2022) [V], CAI (NeurIPS 2021) [V], CoDA [M], ELDEN [partly V] and Ke et al. 2021's pixel "Physics/Chemistry" benchmark [M] ([CDL](https://proceedings.mlr.press/v162/wang22ae.html); [CAI](https://arxiv.org/abs/2106.03443); [Ke et al.](https://arxiv.org/abs/2107.00848)). It mostly **builds the graph into training** and scores it on factored states with heuristic thresholds. Ke et al. vary the true graph and measure prediction quality; they never extract the learned model's graph.

## Only our study combines the transferred probe, the Jacobian key and FWER control

The table sets the study against the six closest papers. Cells come from the research notes. "Not in notes" means the notes did not report it, not that the paper lacks it. CDL details marked [M] are from memory.

| Feature | **This study** | Tan et al. 2026 (PokeWorld) | Delta-JEPA 2026 | Bao 2026 | Gonzalez Laiz & Schneider 2026 | Pandaram et al. 2025 | TDRL 2022 / CDL 2022 |
|---|---|---|---|---|---|---|---|
| Input | Pixels only | Pixels; some variants add proprioception or force | Pixels | Pixels (64×64) | Pixels and states | **States only** | TDRL: video. CDL: states [M] |
| Model | JEPA: patch-ViT, causal transformer, AdaLN actions, inverse loss, SIGReg | JEPA + causal transformer + SIGReg, multi-horizon | JEPA: ViT-tiny, 6-layer causal transformer, AdaLN, inverse decoder; no SIGReg | DreamerV3 | DreamerV3, TD-MPC2, LeWM | 2×512 MLP | Structured models with the graph or identifiability built in |
| Action-conditioned | Yes (continuous force/torque) | Yes | Yes | No (zero actions) | Yes | Yes | Yes |
| Hidden per-episode physics | Yes: masses, lengths, gravity | Yes: mass, drag, stiffness | No | No (damped vs conservative used as a control) | No | No | TDRL: gravity across 6 domains |
| Structure imposed in training | No | No | No | No | No | No | **Yes** |
| What is read | Predictor's ẑ(t+1)−z(t), through a **frozen probe fitted on encoder Δz** | Parameters from predictor state; slope ratios of forecasts | Encoder Δz → Δstate only | Scalar conserved under the model's one-step transition | Encoder state probes; rollout consistency; qualitative probes on predicted latents | MLP input Jacobian | Latents (MCC) / learned graph via CMI |
| Output | **28-edge (cart-pole) / 54-edge (Acrobot) state/action→Δstate graph** | Per-parameter dependence | Probe r | One invariant | Probe error, consistency score | Sparsity counts | Variables / graph |
| Answer key | **Exact simulator Jacobian**, strong/weak/absent tiers, 6 equation checks | True parameters; supervised-GRU ceiling | True Δstate | True energy | True states | **MJX simulator Jacobian** | True latents / graph known by construction |
| Formal false-positive control | **Null-calibrated max-T (Westfall–Young), decoys, 50 episode half-splits** | Not in notes | No | 20 random-constraint control | No | Exact-zero thresholds | Fixed CMI threshold [M] |
| Untrained / random-init floor | **Epoch-0 model** | Yes (untrained network) | No | **Six random-init models** | No (shuffled target only) | No | No |
| Known-absent nuisance edges | Distractor: 22 truly absent edges | No | No | No | No | Many structural zeros | CDL uses distractor variables [M] |
| Training-time curve | 21 checkpoints (cart-pole pending) | Not in notes | No | No | No | No | No |
| Seeds | 1 training run per environment; 3 probe seeds | Not in notes | 3 | 3 | 4 per model | Not in notes | Not in notes |
| Environments | Own cart-pole and Acrobot simulators | PokeWorld 2D pushing; RH20T real robot | Two-Room, OGB-Cube, Reacher, others | Pendulum-v1 | DMC Pendulum, CartPole, Reacher | 8 DMC tasks incl. CartpoleBalance | TDRL: modified Cartpole. CDL: Chemical, manipulation [M] |
| Status | B.Tech, unpublished | arXiv preprint | arXiv preprint | arXiv preprint | arXiv preprint | arXiv preprint | NeurIPS 2022 / ICML 2022 |

Reading across the rows, every closest paper shares at least one bolded feature with the study. None shares three or more of the transferred probe, the graph-level output, the Jacobian key and FWER control. The table also shows where the study is thin. It has one training run per environment, where Gonzalez Laiz & Schneider use four seeds per model. It uses two toy 2D systems, where Tan et al. add real-robot data. And it offers no identifiability theory, unlike the CRL line.

## Three moves have no precedent in the searched literature

**Moving the probe from encoder to predictor.**
- Delta-JEPA and PSG-JEPA fit Δ-probes on the encoder and stop there. Tan et al. test the predictor's forecasts against parameters through slope ratios. Bao fits an invariant on the model's predicted step.
- **None fits a Δstate probe on real encoder changes, freezes it, and uses it to read the predictor's change.** The world-model probing notes conclude that this step "appears to be a novel combination as of Nov 2026" ([arXiv 2606.31232](https://arxiv.org/abs/2606.31232); [arXiv 2607.27017](https://arxiv.org/abs/2607.27017)).
- The value is that the probe never sees the predictor's output during fitting. A good read-out on ẑ(t+1) − z(t) therefore requires the predictor to move the latent the way real dynamics do.
- The study's own numbers show why this matters:
  - The **untrained encoder already ranks edges with AUROC 0.96**.
  - The **untrained predictor sits at 0.48–0.55**, with strongly negative change R².
  - So the predictor floor is clean where the encoder floor is not ([BTP report](../BTP_REPORT.md)).
- This is our own version of Tan et al.'s "readability does not guarantee correct use". The study turns that observation into a control rather than only a finding.

**Auditing a graph, not a state, against an exact physics answer key.**
- The MBRL causal-dynamics and CRL literatures work the other way round. They **build the structure into the model** (masks, CMI tests, known intervention targets, sparsity penalties) and check that it was recovered ([CDL](https://proceedings.mlr.press/v162/wang22ae.html); [Lachapelle et al.](https://arxiv.org/abs/2107.10098v2)).
- Pandaram et al. are the only work found that asks whether an *unconstrained* trained network reproduces simulator-Jacobian zeros, and they do it on states while explicitly excluding pixels ([arXiv 2511.08086](https://arxiv.org/html/2511.08086v2)).
- The causal-dynamics notes found **no paper that does all of the following together** ([arXiv 2610.33894](https://arxiv.org/abs/2610.33894); [arXiv 2605.27589](https://arxiv.org/abs/2605.27589)):
  - uses an unconstrained pixels-only world model;
  - extracts an edge-level graph post hoc;
  - scores it against a simulator-Jacobian key;
  - applies statistical false-positive control with an untrained floor.
- The 2026 world-model "causal" benchmarks (What-If World, WMEval, the CausalVAE plug-in, ScratchWorld) are **behavioural or interventional**. They test predicted outcomes, not the structure of the dynamics. All of them were seen only as snippets [A].
- The study also verifies its answer key unusually carefully. It runs six equation checks, including symbolic Euler–Lagrange agreement to about 1e-14 and an exact-zero Jacobian pattern, and it differentiates each transition at its own episode's physics ([BTP report](../BTP_REPORT.md)).

**Error control and controls the JEPA probing literature lacks.**
- The 2026 JEPA probe tables report **no random-init control** in the text the researchers read: LeWM, Delta-JEPA and PSG-JEPA ([arXiv 2603.19312](https://arxiv.org/abs/2603.19312); [arXiv 2606.31232](https://arxiv.org/abs/2606.31232); [arXiv 2608.06799](https://arxiv.org/abs/2608.06799)).
- Neural Granger causality, GVAR and CDL-style work select edges by **penalty paths, stability heuristics or fixed CMI thresholds**, without formal FWER or FDR guarantees ([arXiv 1802.05842](https://arxiv.org/abs/1802.05842)).
- The study's pruning differs on several counts:
  - **Validation before results.** The rule was chosen on four validation graphs that do not depend on how good the model is (true data, noise-matched data, shuffled null, epoch-0 predictor). The trained model's score was read afterwards.
  - **Clean on the nulls.** `null_cal_fwer` was the only one of eight rules to keep **0 edges on both the shuffled null and the untrained predictor** while scoring F1 0.84 on data.
  - **Clean on the trained model.** The trained cart-pole predictor's graph kept 6 edges, **all true**, with zero false positives.
  - **Extra controls.** A distractor supplies 22 truly absent edges, and a 21-checkpoint curve is planned ([BTP report](../BTP_REPORT.md)).
- Two further design choices strengthen the audit:
  - **Inputs are true simulator states.** A missing edge then reflects the predictor, not a misread input.
  - **Drop-and-refit LOCO.** Plain permutation importance, which over-credits correlated inputs, is avoided, as the statistics literature recommends ([Hooker et al. 2021](https://arxiv.org/abs/1905.03151); [Chamma et al. 2023](https://arxiv.org/abs/2309.07593)).
- The edge-statistics notes found no paper applying LOCO + decoys + split repetition + max-T to a trained world model's predicted dynamics.

## Valid for the global null, not yet proven edge by edge

The honest framing for the write-up is the one the statistics notes propose: the study is **"a pre-registered, null-calibrated combination of established variable-importance tests, applied to a learned world model"**. It should not be described as a new test. Every statistical component has a primary reference:
- LOCO ([Lei et al. 2018](https://arxiv.org/abs/1604.04173); [Williamson et al. 2023](https://arxiv.org/abs/2004.03683));
- shuffled decoys ([Wu, Boos & Stefanski 2007](https://ideas.repec.org/a/bes/jnlasa/v102y2007p235-243.html), and Boruta's shadow features [M]);
- response-permutation importance nulls ([PIMP](https://pubmed.ncbi.nlm.nih.gov/20385727/));
- max-T FWER (Westfall & Young 1993 [M]);
- exact p = (1+b)/(K+1) ([Phipson & Smyth](https://arxiv.org/abs/1603.05766)).

Nor is the motivating premise new. Random-init baselines and control tasks have been standard in probing since 2018–2019 ([Zhang & Bowman](https://arxiv.org/abs/1809.10040); [Hewitt & Liang](https://arxiv.org/abs/1909.03368)) [M], and 2026 world-model papers reproduce the problem empirically ([Bao](https://arxiv.org/abs/2608.23526); [RigidBench](https://arxiv.org/abs/2608.15555)). The pixel cart-pole with hidden gravity has precedents too: TDRL, and AdaRL's change factors [M] ([arXiv 2210.13647](https://www.arxiv.org/abs/2210.13647); [arXiv 2107.02729](https://arxiv.org/abs/2107.02729)).

### The FWER guarantee is weaker than "5% for the whole graph"

The study states that `null_cal_fwer` gives "the whole graph a 5% false-positive rate". Row-shuffling the target creates the **complete null**: no input matters. Max-T over that null gives exact level-α control of the global null. Strong FWER for *individual* edges, when other edges are real, holds only under **subset pivotality** ([Meinshausen, Maathuis & Bühlmann 2011](https://arxiv.org/abs/1106.2068); [arXiv 1612.07010](https://arxiv.org/abs/1612.07010)). Refit-LOCO statistics plausibly violate it, for three reasons:
- residual MSE is far smaller when real causes are present;
- the log-ratio divides optimisation noise by that smaller MSE;
- collinear real causes change how the reduced model refits.

Neuroimaging resolves the same issue with **Freedman–Lane residual permutation** ([Winkler et al. 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4010955/)). The model-X route resamples each tested input conditionally ([HRT](https://arxiv.org/abs/1811.00645); [CPI](https://arxiv.org/abs/1901.09917)). The notes found no paper analysing subset pivotality for refit-LOCO specifically, so this is a reasoned concern, not a demonstrated failure.

The empirical record is reassuring but not decisive. The rule kept zero edges on the null and on the epoch-0 model, and zero false positives on the trained cart-pole model. But those are exactly the regimes closest to the complete null.

Three further statistical issues follow:
- **Coarse p-value floor.** With **K = 20 repeats the smallest attainable p is 1/21 ≈ 0.048**. An edge survives only by beating every one of 20 null maxima, so the decision hinges on the single most extreme repeat. Neuroimaging convention is K ≥ 99, often 1,000+ ([Nichols & Holmes](https://doi.org/10.1002/hbm.1058)) [M].
- **Exchangeability across time.** **Rows are shuffled across autocorrelated time steps**, which breaks exchangeability. The fix is exchangeability blocks: permute whole episodes, or apply circular shifts within episodes ([Winkler et al. 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4010955/)).
- **What the decoys calibrate.** The **shuffled decoys are independent of everything**. They calibrate pure fitting noise, but not a null input that is correlated with a true cause, which is what knockoff or CRT draws mimic ([Candès et al. 2018](https://arxiv.org/abs/1610.02351)) [M].

### Collinearity is an identifiability limit, and the estimand differs from the answer key

LOCO measures only an input's unique contribution, roughly effect² × (1 − R²). In cart-pole, θ and ω̄ are **94.5% and 95.3% predictable** from the other inputs. That is why the trained graph misses the four strong edges out of θ and ω̄ ([BTP report](../BTP_REPORT.md)). Verdinelli & Wasserman state that a perfectly predictable input's separate importance is not estimable, and regard that as "sensible" ([JMLR 2024](https://jmlr.org/papers/v25/22-0801.html); [arXiv 2303.05981](https://arxiv.org/abs/2303.05981)). The write-up should therefore do three things:
- **Report group LOCO for the {θ, ω̄} block** and mark the individual edges "not separately identifiable" rather than "missed".
- **Present the VIF-scaled rule** as a linear approximation to decorrelated LOCO. In a linear model, LOCO × VIF_j recovers β_j²·Var(x_j). This is the note-writer's derivation, not a published result.
- **Cite the partial-disentanglement result** of the JMLR mechanism-sparsity paper as the formal analogue ([JMLR](https://jmlr.org/papers/v27/24-0771.html)).

A subtler issue is that **the expert graph is conditional on physics, but the data and model graphs marginalise over hidden physics**. The data graph's one false positive, v̄→Δω̄, vanishes when the fits are told each episode's physics (effect 0.43 → 0.04), so it is a real marginal dependence, not a test error ([BTP report](../BTP_REPORT.md)). The model has no physics labels, so the **model-vs-data-graph agreement** is arguably the fairer primary comparison, with the expert graph as the physics-conditional reference. Amortised discovery methods handle per-episode variation explicitly and are the natural citations here ([ACD](https://arxiv.org/abs/2006.10833); [V-CDN](https://arxiv.org/abs/2007.00631v3)) [M for ACD].

### Observational edges are not yet proof of causal use

The edge test regresses the probe's read-out of the predictor's change on true simulator states, over the data distribution. It measures input–output dependence of the whole chain (renderer → encoder → predictor → frozen MLP probe). It does not intervene on anything inside the model. Three critiques apply:
- **Nonlinear alignment maps can be vacuous.** The probe is a nonlinear MLP. Sutter et al. show that with sufficiently powerful alignment maps, causal-abstraction alignments can be made to succeed on *randomly initialised* models ([NeurIPS 2025](https://arxiv.org/pdf/2507.08802)).
- **Expressive probes can learn the task themselves** ([Hewitt & Liang](https://arxiv.org/abs/1909.03368)) [M].
- **Readable is not the same as used.** The study itself finds that the model's own v̄→Δω̄ (0.26 vs 0.04 on data) and θ→Δθ (0.55 vs 0.18) differ from the data. It cannot tell model behaviour from probe artefact.

The epoch-0 predictor floor and the frozen-probe design blunt these critiques, but they do not remove them. The literature points to a clear upgrade: **interchange interventions with simulator counterfactuals**. Render frames with one variable changed, or swap a DAS-learned subspace, and score per edge whether the predictor's change follows the simulator ([Geiger et al., JMLR 2025](https://jmlr.org/papers/v26/23-0058.html); [DAS](https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf)). Linear alignment maps and a random-init model are then required controls. RAVEL-style "Cause/Isolate" scores directly target the θ/ω̄ separation ([arXiv 2402.17700](https://arxiv.org/abs/2402.17700)) [M]. No paper applying DAS/IIA to a physical world model was found, so this would also be a contribution.

### Scope limits and how to fix them

| Weakness | Why it matters | Remedy | Grounding |
|---|---|---|---|
| Strong per-edge FWER unproven | Max-T with response permutation needs subset pivotality | Freedman–Lane residual permutation, or a CRT/HRT conditional null; at minimum, a simulation with planted null inputs correlated with true causes that shows realised FWER ≤ 5% | [Winkler 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4010955/); [HRT](https://arxiv.org/abs/1811.00645) |
| K = 20 null repeats | p floor 1/21; large Monte Carlo error | K ≥ 99 (floor 0.01), with fewer splits per repeat if compute-bound; report Monte Carlo CIs | [Phipson & Smyth](https://arxiv.org/abs/1603.05766) |
| Row-level shuffling | Autocorrelated episodes are not exchangeable | Permute whole episodes, or circular shifts within episodes | [Winkler 2014](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4010955/) |
| Independent decoys | Do not mimic correlated nulls | Conditional or knockoff decoys drawn from p(x_j given x_−j) | [CPI](https://arxiv.org/abs/1901.09917) |
| Rule chosen among 8 on one dataset; null-calibrated rules added after the first six failed validation | Mild forking-paths risk | Treat `datasets/cc` and Acrobot (fixed rule in advance) as confirmatory; report all rules in an appendix | [Reisach et al. 2021](https://arxiv.org/abs/2102.13647) |
| θ/ω̄ collinearity | 4 strong cart-pole edges not identifiable | Group LOCO; randomised initial states to break collinearity; interchange interventions | [Verdinelli & Wasserman](https://jmlr.org/papers/v25/22-0801.html) |
| Physics-conditional key vs marginal estimand | "False positives" that are true marginal dependences | Report model-vs-data-graph agreement as primary | [BTP report](../BTP_REPORT.md) |
| Nonlinear probe, observational edges | Readability ≠ use; non-linear representation dilemma | Linear-probe robustness arm; DAS/IIA with simulator counterfactuals | [Sutter et al.](https://arxiv.org/pdf/2507.08802); [Tan et al.](https://arxiv.org/abs/2607.27017) |
| One training seed per environment | Graph may be a seed idiosyncrasy | ≥ 3 JEPA training seeds (Gonzalez Laiz & Schneider use 4) | [arXiv 2610.37588](https://arxiv.org/abs/2610.37588) |
| Acrobot memorisation | Inverse-model validation R² −0.38 vs 0.985 on train; graph may describe memorised episodes | Run the proposed validation-split probe before any Acrobot claim | [BTP report](../BTP_REPORT.md) |
| Distractor not encoded | Only 8 of 22 cross edges scorable, and those test the scorer, not the model | Report as a finding; consider a more salient distractor in a declared follow-up | [BTP report](../BTP_REPORT.md) |
| One-step readout only | Gonzalez Laiz & Schneider find multi-step dynamics unfaithful on CartPole | Apply the frozen probe to k-step rollout changes | [arXiv 2610.37588](https://arxiv.org/abs/2610.37588) |
| Weak edges out of reach (1/7 cart-pole, 0/8 Acrobot) | Recall claims limited to the strong graph | State this up front; the effect-size tier cut (0.1) is a convention | [BTP report](../BTP_REPORT.md) |

## Citations to use, by section of the write-up

Status codes: V = primary page fetched; A = abstract or snippet only; M = cited from memory. Spot-check every M and A item before submission, and confirm arXiv IDs flagged in the notes (ASR 2110.05721, Zeng et al. survey 2302.05209).

**Introduction and motivation**

| Citation | Use it to say | Status |
|---|---|---|
| Richens & Everitt, "Robust Agents Learn Causal World Models", ICLR 2024 ([arXiv 2402.10877](https://arxiv.org/abs/2402.10877)) | Theory says robust agents must be causal; we test empirically whether a predictive JEPA is | M, peer reviewed |
| Schölkopf et al., "Toward Causal Representation Learning", Proc. IEEE 2021 ([arXiv 2102.11107](https://arxiv.org/abs/2102.11107)) | Framing of causal variables from pixels | M, peer reviewed |
| Vafa et al., ICML 2025 ([arXiv 2507.06952](https://arxiv.org/abs/2507.06952)); Vafa et al., NeurIPS 2024 ([arXiv 2406.03689](https://arxiv.org/abs/2406.03689)) | Good prediction does not imply the right world model or force law | V / M, peer reviewed |
| Garrido et al. 2025 ([arXiv 2502.11831](https://arxiv.org/abs/2502.11831)) | JEPA predictors carry physical expectations (VoE) | V, preprint |

**Related work: JEPA and world-model probing (closest first)**

| Citation | Use it to say | Status |
|---|---|---|
| Tan et al. 2026 ([arXiv 2607.27017](https://arxiv.org/abs/2607.27017)) | Nearest work; readout ≠ use; untrained reference | V, preprint |
| Delta-JEPA 2026 ([arXiv 2606.31232](https://arxiv.org/abs/2606.31232)) | Precedent for the encoder-Δz probe; we add the transfer to the predictor | V, preprint |
| Bao 2026 ([arXiv 2608.23526](https://arxiv.org/abs/2608.23526)) | Random-init Dreamers decode energy; test the transition | V, preprint |
| Gonzalez Laiz & Schneider 2026 ([arXiv 2610.37588](https://arxiv.org/abs/2610.37588)) | CartPole + LeWM; latent dynamics unfaithful | V, preprint |
| LeWM ([arXiv 2603.19312](https://arxiv.org/abs/2603.19312)); LeJEPA/SIGReg ([arXiv 2511.08544](https://arxiv.org/abs/2511.08544)) | Architecture and regulariser sources | V / M, preprints |
| PSG-JEPA ([arXiv 2608.06799](https://arxiv.org/abs/2608.06799)) | Pairwise encoder transition probes | V, preprint |
| Joseph et al., ICML 2026 ([arXiv 2602.07050](https://arxiv.org/abs/2602.07050)) | Physics is distributed, not factorised, in V-JEPA 2 | V, peer reviewed |
| RigidBench ([arXiv 2608.15555](https://arxiv.org/abs/2608.15555)); Schäfer et al. ([arXiv 2607.05966](https://arxiv.org/abs/2607.05966)) | Random-init DiT floor; imagined rollouts ignore friction | V / A, preprint and workshop |
| Li et al., Othello-GPT, ICLR 2023 ([arXiv 2210.13382](https://arxiv.org/abs/2210.13382)) | Probe-plus-intervention template with a random-init baseline | M, peer reviewed |

**Related work: causal dynamics, causal representation learning and relational inference**

| Citation | Use it to say | Status |
|---|---|---|
| Pandaram et al. 2025 ([arXiv 2511.08086](https://arxiv.org/abs/2511.08086)) | Simulator-Jacobian ground truth incl. cart-pole; states only | V, preprint |
| CDL, ICML 2022 ([PMLR](https://proceedings.mlr.press/v162/wang22ae.html)); CAI, NeurIPS 2021 ([arXiv 2106.03443](https://arxiv.org/abs/2106.03443)); ELDEN, NeurIPS 2023 ([arXiv 2310.08702](https://arxiv.org/abs/2310.08702)); CoDA, NeurIPS 2020 ([arXiv 2007.02863](https://arxiv.org/abs/2007.02863)) | The structure-built-in paradigm; CMI edge scores as LOCO relatives | V / V / partly V / M |
| Ke et al., NeurIPS 2021 D&B ([arXiv 2107.00848](https://arxiv.org/abs/2107.00848)) | Pixel world models with ground-truth graphs (prediction, not extraction) | M |
| TDRL, NeurIPS 2022 ([arXiv 2210.13647](https://www.arxiv.org/abs/2210.13647)); AdaRL, ICLR 2022 ([arXiv 2107.02729](https://arxiv.org/abs/2107.02729)) | Pixel cart-pole under hidden gravity domains | V / M |
| Lachapelle et al., CLeaR 2022 / JMLR 2026 ([arXiv 2107.10098](https://arxiv.org/abs/2107.10098v2); [JMLR](https://jmlr.org/papers/v27/24-0771.html)) | Action/latent graph by construction; partial disentanglement | V, peer reviewed |
| CITRIS/iCITRIS ([arXiv 2206.06169](https://arxiv.org/abs/2206.06169)); V-CDN, NeurIPS 2020 ([arXiv 2007.00631](https://arxiv.org/abs/2007.00631v3)) | Graph from frames by construction; per-episode structure | V |
| NRI ([arXiv 1802.04687](https://arxiv.org/abs/1802.04687)); ACD ([arXiv 2006.10833](https://arxiv.org/abs/2006.10833)); Neural Granger ([arXiv 1802.05842](https://arxiv.org/abs/1802.05842)); JRNGC, ICML 2024 ([arXiv 2405.08779](https://arxiv.org/abs/2405.08779)) | Edge metrics vs physics ground truth; Jacobian Granger edges; no formal FWER | M / M / V / V |
| "Transformer Is Inherently a Causal Learner" ([arXiv 2601.05647](https://arxiv.org/html/2601.05647)) | Post hoc attribution recovers parents; recency bias on lags | A, preprint |
| Causal-JEPA ([arXiv 2602.11389](https://arxiv.org/abs/2602.11389)) | Nearest "causal" JEPA; no graph test | V; one note lists ICML 2026, the other treats it as a preprint |

**Method: probe design and controls**

| Citation | Use it to say | Status |
|---|---|---|
| Hewitt & Liang, EMNLP 2019 ([arXiv 1909.03368](https://arxiv.org/abs/1909.03368)); Zhang & Bowman 2018 ([arXiv 1809.10040](https://arxiv.org/abs/1809.10040)); Belinkov 2022 ([arXiv 2102.12452](https://arxiv.org/abs/2102.12452)) | Control tasks, random-init baselines; trained-minus-epoch-0 as selectivity | M, peer reviewed |
| Adebayo et al., NeurIPS 2018 ([arXiv 1810.03292](https://arxiv.org/abs/1810.03292)) | The untrained model as a parameter-randomisation sanity check | M, peer reviewed |

**Method: edge statistics and the pruning rule**

| Citation | Use it to say | Status |
|---|---|---|
| Lei et al., JASA 2018 ([arXiv 1604.04173](https://arxiv.org/abs/1604.04173)); Williamson et al., JASA 2023 ([arXiv 2004.03683](https://arxiv.org/abs/2004.03683)) | LOCO and its inference | M / V |
| Hooker et al. 2021 ([arXiv 1905.03151](https://arxiv.org/abs/1905.03151)); Verdinelli & Wasserman, JMLR 2024 ([JMLR](https://jmlr.org/papers/v25/22-0801.html)) | Why drop-and-refit over permutation; collinearity and decorrelated LOCO | V |
| Wu, Boos & Stefanski 2007 ([RePEc](https://ideas.repec.org/a/bes/jnlasa/v102y2007p235-243.html)); Boruta 2010 ([DOI](https://doi.org/10.18637/jss.v036.i11)); PIMP 2010 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/20385727/)) | Decoys and response-permutation nulls | V / M / V |
| Meinshausen, Meier & Bühlmann 2009 ([arXiv 0811.2177](https://arxiv.org/abs/0811.2177)); Meinshausen & Bühlmann 2010 ([arXiv 0809.2932](https://arxiv.org/abs/0809.2932)) | Split aggregation (2×median); stability selection | M |
| Westfall & Young 1993 (book); Meinshausen, Maathuis & Bühlmann 2011 ([arXiv 1106.2068](https://arxiv.org/abs/1106.2068)); Phipson & Smyth 2010 ([arXiv 1603.05766](https://arxiv.org/abs/1603.05766)) | Max-T, subset pivotality caveat, exact p-values | M / V via snippet / M |
| Barber & Candès 2015 ([arXiv 1404.5609](https://arxiv.org/abs/1404.5609)); Candès et al. 2018 ([arXiv 1610.02351](https://arxiv.org/abs/1610.02351)); Ren, Wei & Candès 2023 ([arXiv 2012.02717](https://arxiv.org/abs/2012.02717)) | Knockoff rule compared | M / M / V |
| Reisach et al., NeurIPS 2021 ([arXiv 2102.13647](https://arxiv.org/abs/2102.13647)) | Validating graph extraction on ground truth without gameable artefacts | V |

**Discussion, limitations and future work**

| Citation | Use it to say | Status |
|---|---|---|
| Winkler et al. 2014 ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4010955/)); HRT ([arXiv 1811.00645](https://arxiv.org/abs/1811.00645)); CPI ([arXiv 1901.09917](https://arxiv.org/abs/1901.09917)) | Remedies for pivotality and exchangeability | V |
| Geiger et al., JMLR 2025 ([JMLR](https://jmlr.org/papers/v26/23-0058.html)); DAS, CLeaR 2024 ([PMLR](https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf)); Sutter et al., NeurIPS 2025 ([arXiv 2507.08802](https://arxiv.org/pdf/2507.08802)) | Interchange-intervention follow-up and its triviality caveat | V |
| Elazar et al., amnesic probing, TACL 2021 ([arXiv 2006.00995](https://arxiv.org/abs/2006.00995)); LEACE ([arXiv 2306.03819](https://arxiv.org/abs/2306.03819)) | Cheaper "used vs decodable" test | V / M |
| Liu & Chen 2026 ([arXiv 2608.15156](https://arxiv.org/abs/2608.15156)) | Latent edits in a physics world model; position edits fail specificity | A, preprint |

## Conclusion

The literature has converged on this study's diagnosis from several directions in 2026. A random network already decodes physical state (Bao, RigidBench, Tan et al.). Probes on the encoder say little about what the predictor does (Tan et al., Gonzalez Laiz & Schneider). Encoder Δz is the right object to probe (Delta-JEPA). Nobody has yet joined these observations into a statistically controlled, graph-level audit of the predictor against physics, so the study's claim to a gap is real, but narrow. It should be phrased as an assembly with a specific new step: the frozen encoder-Δ probe transferred to the predictor. It should not be phrased as a new paradigm. Because most of the competing work is unreviewed and only months old, the paper should also say that the field is moving fast enough that a parallel audit could appear at any time. The obvious places to check first are the ICLR 2027 submissions, which the research notes did not search.

The cheapest way to make the paper reviewer-proof is to tighten what it claims rather than to add machinery:
- say "global-null FWER, empirically clean on the trained and untrained graphs" until a Freedman–Lane or planted-correlated-null check backs the per-edge claim;
- raise K and permute by episode;
- recast the θ/ω̄ misses as an identifiability result, reported with group LOCO.

Interchange interventions with simulator counterfactual renders would then do more than patch a weakness. With the simulator as the exact high-level model, they would make this the first causal-abstraction evaluation of a world model against a known physical causal model.
