# Probing trained world models / video & sequence models for physical state, dynamics and "world models"

Scope note: the comparison study is a pixels-only, action-conditioned JEPA (patch-ViT encoder on frame pairs, causal-transformer predictor over 5-latent history with AdaLN actions, inverse-model aux loss, SIGReg from LeJEPA, 3-step rollout loss) trained on cart-pole / Acrobot with hidden per-episode physics. It uses a "differential probe" (MLP fit on encoder Δz = z(t+1)−z(t) → true Δstate, frozen, then applied to the predictor's Δẑ = ẑ(t+1)−z(t)), rejects state probes because a random encoder decodes pole angle with R²≈0.95, uses an epoch-0 untrained floor and a 21-checkpoint training curve. Excluded per assignment: causal-dynamics MBRL (CDL, GRADER), causal representation learning (CITRIS), causal abstraction/interchange interventions, LOCO/knockoffs.

Verification legend: papers marked **[fetched]** had their arXiv page/HTML read in this session; papers marked **[memory]** are well-known works cited from prior knowledge with their primary link, and their details were not re-checked this session.

## Q1. Probing JEPA / V-JEPA / LeWM-style and other self-supervised video world models for physical variables or intuitive physics

### Takeaway
Yes. There is a fast-growing 2025–2026 literature here. Most of it probes **encoder** features for **states** (position, velocity, direction, plausibility) and compares against other *trained* models. Random-init or untrained controls are rare in the JEPA papers. The most similar paper in architecture and setup is Tan et al. 2026 (PokeWorld): an action-conditioned JEPA with a causal-transformer predictor and SIGReg, hidden per-episode mass/drag/stiffness, an untrained-network reference, and forecast-dependence tests. Its central finding is that what a probe can read out and what the model actually uses come apart.

### Cited Findings
**Garrido et al. 2025, "Intuitive physics understanding emerges from self-supervised pretraining on natural videos"** (Q. Garrido, N. Ballas, M. Assran, A. Bardes, L. Najman, M. Rabbat, E. Dupoux, Y. LeCun). arXiv preprint, Feb 2025 **[fetched]**.
- Uses the violation-of-expectation (VoE) paradigm on IntPhys, GRASP and InfLevel. "Surprise" is the gap between V-JEPA's predicted and observed representations. Models that predict in a learned representation space show object permanence and shape consistency. Pixel-space video predictors and multimodal LLMs score closer to chance. Even about one week of unique video gives above-chance results. — [arXiv 2502.11831](https://arxiv.org/abs/2502.11831)
- A secondary summary reports that randomly initialised models did worse than trained ones, and that the masking strategy mattered less than expected. I did not confirm this on the abstract page. — [bdtechtalks summary](https://bdtechtalks.com/2025/04/28/v-jepa-intuitive-physics/)
- This is a behavioural "surprise" test, not a probe of the predictor's change. Closeness: **loosely/partially related**. It establishes that JEPA predictors carry physical expectations, but it reads no variables and no dynamics.

**Joseph, Garrido, Balestriero, Kowal, Fel, Bakhtiari, Richards, Rabbat 2026, "Interpreting Physics in Video World Models"**. arXiv Feb 2026; ICML 2026, PMLR v306 **[fetched]**. — [arXiv 2602.07050](https://arxiv.org/abs/2602.07050); [PMLR](https://proceedings.mlr.press/v306/joseph26a.html)
- **Models:** frozen V-JEPA 2 (L/H/G) and VideoMAE-v2 G encoders.
- **Data:** IntPhys, Kubric toy balls with constant velocity or acceleration, and CLEVRER objects.
- **Probes:** linear probes on mean-pooled patches, attentive-MLP probes, per-patch probes, circular regression for direction, and orthogonal probe sequences to estimate dimensionality.
- **Controls:** control tasks (CLEVRER counting, ImageNet, SSv2, shuffled-vs-ordered video) and random-subspace baselines. No untrained-network baseline is described.
- **Findings:**
  - A "Physics Emergence Zone" appears at about 1/3 of network depth. Possible-vs-impossible probe accuracy rises from ~50% to ~85–95%, then degrades toward the output layers.
  - Speed and acceleration magnitude are decodable from early layers. Direction appears only at the PEZ, as a distributed, ring-like population code needing ~40–50 dimensions.
  - Direction and IntPhys subspaces are nearly orthogonal (principal angles 69–83°).
  - Ablating local attention in the PEZ drops direction R² from 0.97 to 0.14 and IntPhys accuracy from 78.3% to 61.7%.
  - Steering works only with about 20 orthogonal probe directions.
  - The authors conclude the models do not use factorised, physics-engine-style variables.
- Probes read **encoder** activations, not predictor outputs or frame differences.
- Closeness: **partially related**. The probing and intervention methods are similar, but the model is an action-free encoder and nothing is read from the predictor.

**Maes, Le Lidec, Scieur, LeCun, Balestriero 2026, "LeWorldModel (LeWM): Stable End-to-End JEPA from Pixels"**. arXiv:2603.19312 v3 (Jun 2026), no peer-reviewed venue listed **[fetched]**. — [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
- **Architecture:** ViT-tiny encoder, 6-layer causal-transformer predictor, AdaLN action conditioning (zero-initialised), and SIGReg (Epps–Pulley statistic on 1024 random projections, λ=0.1). This is essentially the same recipe the comparison study uses.
- **Probes on Push-T** (frozen encoder, linear and MLP):
  - Agent location: linear r 0.974, MLP r 0.998.
  - Block angle: linear r 0.902, MLP r 0.990.
  - DINO-WM is equal or better on several entries. LeWM beats PLDM on all rows.
- **Other analyses:**
  - Predicted latents are rolled out and decoded only for visual inspection. No quantitative probing of predicted latents is visible.
  - "Temporal straightening" of latent velocities v_t = z_{t+1}−z_t.
  - VoE: object teleportation causes a significant surprise spike (p<0.01). Colour change causes only a weak one.
- No random-init or untrained control appears in the main text (Appendix F not fully read).
- Closeness: **partially related**. It is the architecture template, and it illustrates the "state probe as evidence" practice the comparison study rejects.

**Balestriero & LeCun 2025, "LeJEPA"** introduces SIGReg. It is the regulariser source, not a probing paper **[memory]**. — [arXiv 2511.08544](https://arxiv.org/abs/2511.08544)

**Tan, Xu, Xu, Tao, Li, Hong, Feng, Du, Wang 2026, "What Can Latent World Models Know? Physical Information in Multimodal Predictive Representations"**. arXiv:2607.27017 v4 (Sep 2026), preprint **[fetched]**. — [arXiv 2607.27017](https://arxiv.org/abs/2607.27017)
- **Models:** action-conditioned JEPAs with a shared encoder, causal-transformer predictor, multi-horizon latent heads {1, 4, 16} and SIGReg. Variants differ in which input and target streams they use (vision, proprioception, touch/force).
- **Environment:** PokeWorld, a 2D pushing task. Objects look identical but have **per-episode hidden mass, drag and stiffness**. Also real RH20T robot data.
- **What is probed:**
  - Linear ridge readout of hidden parameters from the predictor state h_t.
  - Position from the frame embedding.
  - Velocity from h_t.
  - Forecast-dependence tests: the slope ratio of the model's *predicted* displacement or force against each true parameter, and errors of predicted action effects under scripted push/retreat sequences.
- **Controls:**
  - A supervised-GRU recovery ceiling.
  - An untrained network: on RH20T, current-force readout is 0.91 for the *untrained* model vs 0.88 for the trained VFX model.
  - No Hewitt-style control tasks.
- **Findings:**
  - Stiffness readout depends on prediction targets: −0.01 for VF vs 0.50 for VFX, with a 0.78 ceiling.
  - Drag readout is weak (0.12), yet 16-step glide forecasts have slope ratio 1.04, matching the true dynamics.
  - A coordinate-target model reads drag at 0.41 but its release forecasts have slope ratio only 0.12.
  - Quotes: "Readability does not guarantee correct use"; "The objective of the prediction head, more than the content of the representation, limits…".
- Closeness: **very close**. Same model family (JEPA + causal transformer + SIGReg + actions), hidden per-episode physics, an untrained reference showing state readout is free, and tests of the model's *predicted* changes against true parameter dependence. It does not freeze an encoder-Δz probe and transfer it to the predictor, and it does not score a variable-to-variable causal graph.

**Yan et al. 2026 (15 authors), "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models" (PSG-JEPA)**. arXiv:2608.06799 (Aug 2026) **[fetched]**. — [arXiv 2608.06799](https://arxiv.org/abs/2608.06799)
- LeWM backbone plus training-only grounding heads: a state head, and a transition head that maps latent pairs to multi-horizon joint-angle changes.
- **Probes:** linear and MLP, on frozen **encoder** outputs. Single-latent probes target state. Pairwise probes on (z_t, z_{t+1}) target joint velocity, gripper velocity and action.
- **Findings:**
  - Forward-only LeWM fails on some variables: end-effector yaw r=0.08, vs 0.94/0.98 with grounding.
  - Mean linear state r rises from 0.68 to 0.94, and transition r from 0.62 to 0.75.
- No random-init control. The predictor is evaluated only by latent MSE.
- Environments: OGBench-Cube/Scene, LIBERO, a real robot.
- Closeness: **partially related**. The pairwise "transition probe" resembles the differential probe, but it is applied only to the encoder and has no untrained floor.

**Zhang, Wang et al. 2026 (14 authors), "Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding"**. arXiv:2606.31232 (Jun 2026) **[fetched]**. — [arXiv 2606.31232](https://arxiv.org/abs/2606.31232)
- **Architecture:** ViT-tiny encoder, 6-layer causal-transformer predictor with AdaLN actions, and an inverse-model decoder that reconstructs the action from Δz_t = z_{t+1}−z_t. This is very close to the comparison study's inverse-model auxiliary loss. No SIGReg.
- **Probes:** they explicitly "train probes to predict state changes Δx_t = x_{t+1} − x_t from latent displacements Δz_t = z_{t+1} − z_t", using frozen-encoder linear and MLP probes with trajectory-level splits. Results:
  - Two-Room: linear r 0.992 vs LeWM 0.765.
  - OGB-Cube end-effector Δposition: r≈0.995 vs −0.01 to 0.76 for baselines.
  - Mixed results elsewhere, e.g. Reacher Δfinger r 0.359 vs PLDM 0.654.
- The probe is applied to the **encoder Δz only**, not to the predictor's predicted displacement.
- Predictor action sensitivity (fix z_t, vary the action) is shown only qualitatively.
- No random-init control. 3 seeds.
- Closeness: **very close on the probe construction**. It is the clearest published precedent for an encoder Δz → Δstate probe. It does not take the key step of transferring the frozen probe to ẑ(t+1)−z(t), and it has no untrained floor or causal-graph scoring.

**Other video-model physics probing (2026)** **[search snippets / fetched abstracts]**:
- "How Do Video Foundation Models Encode Intuitive Physics? Probing Across Pretraining Paradigms" (NeurIPS 2026 per the listing). It probes frozen V-JEPA, VideoMAE and LTX-Video features on IntPhys2 and Minimal Video Pairs. Temporal attentive probes beat linear ones. V-JEPA reaches 95.0% pair consistency on MVP. VideoMAE scores 73.9% on IntPhys2. Intuitive physics is "not a single, uniform capability". — [arXiv 2606.09646](https://arxiv.org/pdf/2606.09646); [NeurIPS listing](https://neurips.cc/virtual/2026/175855)
- "Probing Where Surprise Happens: Localized VoE Evaluation of Video World Models" (NeurIPS 2026 listing). It compares global vs spatiotemporally localised prediction error for V-JEPA models on Physical Concepts and IntPhys1. — [NeurIPS](https://neurips.cc/virtual/2026/165762)
- Esmati, Nath, Hofmann, Nowrouzezahrai, Kahou, Mirmehdi 2026, "The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show". Physical plausibility is linearly decodable from diffusion-transformer states (~81.27% average on IntPhys and InfLevel), beating V-JEPA and VideoMAE. The signal is absent in the VAE latent input and emerges inside the DiT. — [arXiv 2606.05328](https://arxiv.org/abs/2606.05328) **[fetched abstract]**
- Jain & Wu 2026, "RigidBench: Evaluating Rigid-Body Physics in Video Generation Models" **[fetched HTML]**. — [arXiv 2608.15555](https://arxiv.org/abs/2608.15555)
  - Probes the Wan 2.2 TI2V-5B DiT residual stream at every layer and noise level, with linear and MLP probes for position, contact, orientation, velocity and angular velocity.
  - Controls: shuffled labels (R²≈0, AUC≈0.5) and a **randomly initialised DiT** with the same VAE and architecture. The random DiT reaches position R² 0.388 vs 0.709 for base Wan, and contact AUC 0.696 vs 0.799. "The randomly initialized DiT still scores above chance."
  - Velocity and angular velocity "remain weak".
  - Intervention: INLP removal of the position subspace raises the flow-matching loss far more than removing a random subspace of matched rank.
  - Closeness: **partially related** (good control design: random init, shuffled labels, matched random-subspace ablation).

**Related earlier benchmarks** **[memory]**:
- Piloto et al. 2022, PLATO, Nature Human Behaviour. Object-centric VoE learner. — [Nature HB](https://www.nature.com/articles/s41562-022-01394-8)
- Bear et al. 2021, Physion, NeurIPS Datasets & Benchmarks. Readout of physical outcome prediction from many pretrained vision models. — [arXiv 2106.08261](https://arxiv.org/abs/2106.08261)
- V-JEPA 2: Assran et al. 2025. — [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- DINO-WM: Zhou et al. 2024. — [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- PLDM: Sobal et al. 2025. — [arXiv 2502.14819](https://arxiv.org/abs/2502.14819)
- In my searches these last three appear mainly as baselines in others' state-probe tables (e.g. the LeWM Table 1), not as dedicated probing studies.

### Inferences
- In 2026 the standard JEPA world-model evaluation is to probe frozen-encoder state (LeWM, QQWorld, Sub-JEPA, PSG-JEPA, Delta-JEPA), usually **without** a random-init control. The comparison study's critique of state probes therefore targets a live, common practice.
- Tan et al. 2026 and RigidBench independently show that untrained networks already read out "state-like" quantities well (current force 0.91 untrained; random-DiT position R² 0.39). This is consistent with the comparison study's random-encoder R²≈0.95 for pole angle.
- The comparison study's novel step is transferring a probe fit on encoder Δz to predictor Δẑ. I found no precedent for it. Delta-JEPA and PSG-JEPA stop at encoder Δz. Tan et al. test predicted forecasts against true parameters through slope ratios, not through a transferred probe.

### Gaps
- I did not read LeWM's Appendix F, so I cannot rule out a random-init control there.
- I did not read the full text of 2606.09646 or the "Probing Where Surprise Happens" paper, so their controls are unknown.
- Garrido et al.'s random-init comparison is reported only by a secondary source.
- I found no dedicated probing papers on DINO-WM or PLDM themselves.

## Q2. Probing RL world models (Dreamer, IRIS, DIAMOND, TD-MPC, PlaNet, MuZero) for ground-truth state or dynamics, and the controls used

### Takeaway
Two 2026 papers are the closest RL-world-model analogues.
- **Bao 2026** (DreamerV3 on pixel Pendulum): random-init Dreamers yield energy-correlated readouts up to |ρ|=0.908, so decodability is insufficient. Evidence must come from the *transition*.
- **Gonzalez Laiz & Schneider 2026**: probes DreamerV3, TD-MPC2 and LeWM on DMC CartPole and Pendulum, and shows that high reward coexists with unfaithful latent dynamics.

MuZero work mostly decodes latents to observations. I found no IRIS or DIAMOND state-probing studies.

### Cited Findings
**Bao 2026, "Correcting a learned physical invariant improves world-model rollouts"** (independent researcher). arXiv:2608.23526, Aug 2026 **[fetched]**. — [arXiv 2608.23526](https://arxiv.org/abs/2608.23526)
- **Setup:** DreamerV3 (13.5M parameters) trained on 64×64 Pendulum-v1 video with zero actions and no labels.
- **Method:** a label-free search over degree-4 polynomials of the top-12 latent PCs for a scalar that is conserved **under the model's own one-step transition**. That scalar is then compared with the true energy.
- **Controls:**
  - **Six randomly initialised Dreamers.** Energy correlation ranges 0.170–0.908, and 4 of 6 exceed a 0.7 threshold. "Decodability alone does not explain this: randomly initialized Dreamers also produce strongly energy-correlated readouts."
  - Damped-pendulum models as a "no invariant exists" control. Their drift is 0.203–0.266, vs about 1e-4 for conservative models.
  - 20 norm-matched random polynomial constraints.
- **Results:**
  - Trained models reach |ρ|E 0.967–0.975.
  - Projecting back onto the energy level set cuts 50-step rollout error by 2.9–3.5%. Random constraints have a median change of +53.9%.
  - "Dreamer has learned a physical constraint that its own rollout dynamics fail to preserve."
- **Limitations:** one system, three seeds, in-sample fitting.
- Closeness: **very close in logic**. It explicitly argues that random-init decodability is uninformative and moves the test onto the learned transition's predicted displacement. Differences: it looks for an invariant rather than a causal graph, and uses a single system with no actions.

**Gonzalez Laiz & Schneider 2026, "Understanding the Limits of Latent World Models"** (Helmholtz Munich / MCML). arXiv:2610.37588, Oct 2026 **[fetched]**. — [arXiv 2610.37588](https://arxiv.org/abs/2610.37588)
- **Models:** DreamerV3, TD-MPC2 and LeWM, 4 seeds each, trained on states and on pixels.
- **Environments:** DMC Pendulum-swingup, **CartPole-swingup** and Reacher-hard.
- **Probes:** a 2-hidden-layer MLP on frozen **encoder** latents, decoding positions, angles and velocities, with episode-level splits. Linear probes are in the appendix.
- **Controls:** a shuffled-target control used to normalise error, a state-MLP topline, and a persistence baseline.
- **Dynamics metric:** "latent rollout consistency" (LRC), comparing predicted latents with encoded ones. Probes are applied to predicted latents only qualitatively (CartPole rollouts, Fig. 5).
- **Findings:**
  - Pixel velocity error is about 15–22× the state-model median.
  - With 1 frame, velocity error is ~130%.
  - TD-MPC2 scores reward 832 with LRC₃₀ below 0. LeWM on pixels scores LRC₃₀ 1.2 with reward 151.
  - "none solves the task by actually faithfully modeling the dynamics of the environment".
- Closeness: **very close in setting** (same CartPole domain, LeWM-style JEPA, the question of whether the dynamics are faithful). It does not probe Δ-states or score causal dependencies, and has no random-init encoder control (shuffled-label only).

**Schäfer, Moller, Gao, Oefinger, Schmidt, Betz 2026, "Imagined Rollouts are Kinematic, Not Dynamic"**. RSS 2026 Robot World Model Workshop, arXiv:2607.05966 **[fetched abstract]**. — [arXiv 2607.05966](https://arxiv.org/abs/2607.05966)
- Uses a released DreamerV3 checkpoint on DMC walker-walk and introduces an imagined kinematic-consistency error (iKCE).
- Imagined iKCE is about 100× that of real rollouts, and it stays flat across a friction sweep while reward collapses. The authors read this as the model extrapolating kinematics without representing dynamics.
- Closeness: **partially related**. It is a test of whether imagined dynamics respond to a hidden physical parameter. That parallels asking whether the predictor's Δ depends on the hidden physics, but there is no probe or graph.

**MuZero:**
- Guei et al., "Demystifying MuZero Planning: Interpreting the Learned Model" (arXiv 2411.04580; reported as accepted to IEEE Trans. AI). Adds a decoder from hidden states to observations, on 9×9 Go, Gomoku and Atari. Dynamics accuracy degrades with unroll depth, and the model is better on board games than on Atari. — [arXiv 2411.04580](https://arxiv.org/abs/2411.04580)
- He et al., "What model does MuZero learn?" (arXiv 2306.00840). Finds the value-equivalent model is generally not accurate enough for policy evaluation. — [arXiv 2306.00840](https://arxiv.org/abs/2306.00840)
- Closeness: **loosely related** (decoders, not state probes; no controls).

**Other Dreamer probing:**
- A "World Models 2024" workshop paper fits linear-regression probes for agent xy position on DreamerV3 latents, reporting R² and MAE. — [PDF](https://hironona.github.io/assets/documents/papers/World_Models_2024.pdf)
- A Crafter study probes DreamerV3 states with ridge and MLP probes following Gurnee & Tegmark-style space/time probing (seen only in a search snippet; no random-init control seen).
- Zahorodnii 2025 (ICLR 2025 workshop), "Improving World Models using Supervision with Co-Evolving Linear Probes". Adds a linear probe on true variables as a training loss (Flappy Bird, Ha & Schmidhuber-style world model). — [arXiv 2504.03861](https://arxiv.org/abs/2504.03861)
- Closeness: **loosely related**.

### Inferences
- Bao 2026 is the strongest published independent support for the comparison study's premise that random-init decodability invalidates state-probe evidence. It is also the closest precedent for "evaluate the transition's predicted displacement, not the encoded state".
- No RL world-model paper I found trains a probe on encoder Δz and transfers it to imagined Δẑ, or scores which variables drive which predicted changes.

### Gaps
- I found no probing studies of IRIS, DIAMOND, PlaNet or TD-MPC(1) latents against ground-truth state beyond the above. This does not prove none exist.
- I did not read the full text of the Kinematic-not-Dynamic paper, so its probe protocol is unknown.

## Q3. Mechanistic interpretability of "world models" in sequence models (Othello-GPT, chess, Vafa et al., physics transformers, video generators)

### Takeaway
This line established the probe-plus-intervention template and random-init baselines for "world model" claims, and later critiques of probe-only evidence (Vafa et al. 2024/2025). It works on discrete board or graph states, and on video generators judged behaviourally. None read out a dynamics model's predicted *change* or score variable-to-variable dependencies. All are **loosely to partially related**.

### Cited Findings
- **Li, Hopkins, Bau, Viégas, Pfister, Wattenberg, "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task" (ICLR 2023)** **[memory]**. — [arXiv 2210.13382](https://arxiv.org/abs/2210.13382)
  - GPT trained on Othello move sequences.
  - Nonlinear MLP probes recover the board state, while linear probes do poorly.
  - Probes trained on a **randomly initialised** Othello-GPT have much higher error, used as a baseline.
  - Gradient-based interventions on the probed board state change the predicted legal moves, as causal evidence.
- **Nanda, Lee, Wattenberg, "Emergent Linear Representations in World Models of Self-Supervised Sequence Models" (BlackboxNLP 2023)** **[memory]**. — [arXiv 2309.00941](https://arxiv.org/abs/2309.00941)
  - The board is linearly decodable in a "mine vs theirs" basis.
  - Linear vector-addition interventions steer the model.
- **Karvonen, "Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models" (COLM 2024)** **[memory]**. — [arXiv 2403.15498](https://arxiv.org/abs/2403.15498)
  - Linear probes recover the board state and player skill in chess GPTs.
  - Compared against randomly initialised models.
  - Interventions on board and skill representations.
- **Vafa, Chen, Rambachan, Kleinberg, Mullainathan, "Evaluating the World Model Implicit in a Generative Model" (NeurIPS 2024)** **[memory]**. — [arXiv 2406.03689](https://arxiv.org/abs/2406.03689)
  - Myhill–Nerode-inspired metrics (sequence compression and distinction) for deterministic finite automata (DFA) world models: NYC taxi navigation, Othello, logic puzzles.
  - Models with high next-token accuracy can have incoherent world models.
  - A critique that probe or next-step accuracy alone overstates world-model recovery.
- **Vafa, Chang, Rambachan, Mullainathan, "What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models" (ICML 2025, PMLR v267)** **[search-verified]**. — [arXiv 2507.06952](https://arxiv.org/abs/2507.06952); [PMLR](https://proceedings.mlr.press/v267/vafa25a.html)
  - The "inductive-bias probe" adapts the model to small synthetic tasks generated from a postulated world model.
  - A transformer trained on planetary orbits predicts trajectories well but does not use Newtonian mechanics. Its implied force law is "warped", indicating task-specific heuristics.
  - Othello models reach near-100% legal-move accuracy but show weaker inductive bias toward the board state.
  - Closeness: **partially related**. It explicitly asks whether the *dynamics/force law* was learned rather than whether state is decodable. That is the same epistemic move as the comparison study, with a different method.
- **Kantamneni, Liu, Tegmark 2024, "How Do Transformers 'Do' Physics? Investigating the Simple Harmonic Oscillator"** **[memory; title and venue not re-verified]**. — [arXiv 2405.17209](https://arxiv.org/abs/2405.17209)
  - Probes transformers trained on SHO trajectories for intermediate quantities of known numerical integration methods, with correlational and causal (intervention) criteria.
  - Closeness: **partially related** (probes for the computation of the dynamics, not just the state).
- **Kang et al. 2024, "How Far is Video Generation from World Model: A Physical Law Perspective"** (ByteDance; ICML 2025) **[memory]**. — [arXiv 2411.02385](https://arxiv.org/abs/2411.02385)
  - Video diffusion models trained on 2D simulated motion (uniform motion, collision, parabola) generalise in distribution but fail out of distribution.
  - Generalisation looks "case-based", prioritising colour > size > velocity > shape.
  - A behavioural evaluation, not probing.
  - Closeness: **loosely related**.
- **Motamed, Culp, Swersky, Jaini, Geirhos 2025, "Do Generative Video Models Understand Physical Principles?" (Physics-IQ benchmark)** **[memory]**. — [arXiv 2501.09038](https://arxiv.org/abs/2501.09038)
  - Real-video continuation in fluid dynamics, optics, solid mechanics, magnetism and thermodynamics.
  - Physical understanding is limited and uncorrelated with visual realism.
  - Behavioural evaluation. Closeness: **loosely related**.

### Inferences
- The Othello and chess line already used random-init baselines and interventions. But its states are discrete and slowly varying, so random-init floors were far below trained accuracy. The comparison study's case, where the random floor is near ceiling for continuous pixel-rendered angles, is a stronger version of the problem these papers faced.
- Vafa et al. (2024, 2025) are the clearest published argument that good prediction or probe accuracy does not imply the correct underlying world model. They are useful framing for rejecting state probes.

### Gaps
- I did not re-verify the details of Kantamneni et al. 2024 this session, nor whether Li et al.'s random-init numbers are exactly as described.
- I found no work probing Sora-class video generators internally for continuous dynamics variables beyond RigidBench and "Invisible Hand" (Q1).

## Q4. Discovering physical concepts and state variables from neural-network latents (SciNet, AI Feynman, pixel-based HNN/LNN, Chen et al. 2022)

### Takeaway
This line *designs* models (bottlenecks, Hamiltonian structure, intrinsic-dimension estimation) so that latents correspond to physical variables, then checks the correspondence. Usually this is done by correlating latent units with true variables (SciNet) or by counting dimensions (Chen et al.). It is not post-hoc probing of a generic world model, and it does not test causal graphs. **Loosely related**.

### Cited Findings
- **Iten, Metger, Wilming, del Rio, Renner, "Discovering Physical Concepts with Neural Networks" (Phys. Rev. Lett. 124, 010508, 2020)** **[memory]**. — [arXiv 1807.10300](https://arxiv.org/abs/1807.10300)
  - SciNet is a β-VAE-like encoder → latent bottleneck → question-conditioned decoder.
  - Latent neurons correlate with physical parameters, e.g. oscillator damping and spring constant, and heliocentric angles from Copernicus data.
  - Disentanglement is enforced by architecture.
- **Udrescu & Tegmark, "AI Feynman" (Science Advances 2020)** **[memory]**. Symbolic regression from tabular data, not latents. — [arXiv 1905.11481](https://arxiv.org/abs/1905.11481)
- **Chen, Huang, Raghupathi, Chandratreya, Du, Lipson, "Automated discovery of fundamental variables hidden in experimental data" (Nature Computational Science 2022)** **[memory]**. — [arXiv 2112.10755](https://arxiv.org/abs/2112.10755)
  - Video-prediction autoencoders on pixel videos of physical systems (e.g. single and double pendulum, lava lamp, fire).
  - Estimates the intrinsic dimension of the latent manifold to infer the number of state variables. Discovered "neural state variables" are not identical to textbook variables.
  - Closeness: **loosely/partially related** (pixels → dynamics latents → physical variables; no causal graph).
- **Greydanus, Dzamba, Yosinski, "Hamiltonian Neural Networks" (NeurIPS 2019)**: includes a pixel-pendulum experiment with an autoencoder latent. — [arXiv 1906.01563](https://arxiv.org/abs/1906.01563)
- **Toth et al., "Hamiltonian Generative Networks" (ICLR 2020)**: learns Hamiltonian dynamics from pixel images of pendulum, mass-spring and n-body systems. — [arXiv 1909.13789](https://arxiv.org/abs/1909.13789)
- Both are **[memory]**. They impose conservation structure rather than probing a generic model.

### Inferences
- These works recover *state* variables (and sometimes their count or invariants). Bao 2026 (Q2) is effectively the post-hoc, generic-world-model descendant of the HNN/conservation idea.

### Gaps
- I did not search specifically for Lagrangian-from-pixels follow-ups or other "neural state variable" follow-ups (2023–2026). The tool budget was spent on the closer probing literature.

## Q5. Does any work read out the model's PREDICTED CHANGE (dynamics) rather than the encoded state, or score variable-to-variable dependencies of the predictor?

### Takeaway
Partially. No paper I found does the comparison study's exact move: fit a Δstate probe on encoder Δz, freeze it, and apply it to predictor Δẑ to score cause-to-effect dependencies. The nearest pieces are:
- Delta-JEPA and PSG-JEPA: encoder Δz → Δstate probes, but never transferred to the predictor.
- Tan et al. 2026: slope ratios of predicted displacement and force versus each hidden parameter, which is a per-variable dependence test on predictor outputs.
- Bao 2026: a conserved quantity under the model's own predicted transition, with random-init controls.
- Gonzalez Laiz & Schneider 2026: qualitative probe decoding of predicted CartPole rollouts, plus latent rollout consistency.
- Vafa et al. 2025: inferring the implicit force law.

### Cited Findings
- Delta-JEPA probes Δz_t → Δx_t on encoder embeddings only. The paper does not probe ẑ_{t+1}−z_t. Predictor action-sensitivity is qualitative (fix z_t, vary a_t). — [arXiv 2606.31232](https://arxiv.org/abs/2606.31232)
- PSG-JEPA pairwise probes on (z_t, z_{t+1}) target velocity and action. The predictor is judged only by latent MSE. — [arXiv 2608.06799](https://arxiv.org/abs/2608.06799)
- Tan et al. measure the slope of predicted displacement or peak force against true mass, drag and stiffness (e.g. glide slope ratio 1.04 for drag), and predicted action-effect differences. They find readout and use dissociate. — [arXiv 2607.27017](https://arxiv.org/abs/2607.27017)
- Bao fits the invariant using the model's predicted one-step displacement, and validates it with random-init, damped-system and random-constraint controls. — [arXiv 2608.23526](https://arxiv.org/abs/2608.23526)
- Gonzalez Laiz & Schneider apply state probes to predicted latents only for qualitative CartPole rollouts. Their quantitative dynamics metric is LRC. — [arXiv 2610.37588](https://arxiv.org/abs/2610.37588)
- Schäfer et al. test whether imagined rollouts respond to friction changes, and find they do not. — [arXiv 2607.05966](https://arxiv.org/abs/2607.05966)

### Inferences
- The differential probe transferred to the predictor, together with an epoch-0 floor and a training curve over checkpoints, appears to be a **novel combination** as of Nov 2026, within the literature searched here.
- The most defensible positioning: "Delta-JEPA/PSG-JEPA probe encoder Δz; Tan et al. and Bao test predicted dynamics against physical parameters or invariants; we transfer an encoder-Δ probe to the predictor and score the full state/action → Δstate dependency graph."

### Gaps
- Coverage is limited to ~20 searches and fetches. A dedicated search for "probe transfer" / "encoder-to-predictor probe" phrasing might find more.

## Q6. Probing-methodology controls relevant to "a random encoder already decodes the state"

### Takeaway
Standard NLP probing methodology anticipates this problem. Hewitt & Liang's control tasks and selectivity, random-init baselines (Zhang & Bowman 2018), MDL / information-theoretic probing, and Belinkov's survey all say that probe accuracy must be compared with what an untrained or trivial representation supports. Recent world-model papers re-discover it empirically: Bao (random Dreamers reach |ρ| up to 0.908), RigidBench (random DiT position R² 0.388), Tan et al. (untrained current-force readout 0.91).

### Cited Findings
- **Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019)** **[memory]**. — [arXiv 1909.03368](https://arxiv.org/abs/1909.03368)
  - Control tasks assign random but consistent labels to word types.
  - "Selectivity" is task accuracy minus control accuracy. Expressive MLP probes can show high accuracy with low selectivity.
- **Zhang & Bowman, "Language Modeling Teaches You More than Translation Does" (BlackboxNLP 2018)** **[memory]**. Random-init encoder baselines often reach surprisingly high probing accuracy. — [arXiv 1809.10040](https://arxiv.org/abs/1809.10040)
- **Voita & Titov, "Information-Theoretic Probing with Minimum Description Length" (EMNLP 2020)** **[memory]**. MDL distinguishes trained from random representations better than accuracy does. — [arXiv 2003.12298](https://arxiv.org/abs/2003.12298)
- **Pimentel et al., "Information-Theoretic Probing for Linguistic Structure" (ACL 2020)** **[memory]**. — [arXiv 2004.03061](https://arxiv.org/abs/2004.03061)
- **Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances" (Computational Linguistics 2022)** **[memory]**. Surveys baselines, control tasks, and the correlational-vs-causal gap. — [arXiv 2102.12452](https://arxiv.org/abs/2102.12452)
- **World-model instances of the problem:**
  - Random-init Dreamers: energy |ρ| 0.170–0.908. — [Bao 2026](https://arxiv.org/abs/2608.23526)
  - Random-init DiT: position R² 0.388 vs 0.709 trained, and still above chance. — [RigidBench](https://arxiv.org/abs/2608.15555)
  - Untrained fused-input encoder: current-force readout 0.91 vs 0.88 trained. — [Tan et al. 2026](https://arxiv.org/abs/2607.27017)
  - Shuffled-target control. — [Gonzalez Laiz & Schneider 2026](https://arxiv.org/abs/2610.37588)
  - Random-init Othello-GPT baseline. — [Li et al. 2023](https://arxiv.org/abs/2210.13382) **[memory]**
- **JEPA probe tables without a random-init control in the text I read:**
  - LeWM — [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
  - Delta-JEPA — [arXiv 2606.31232](https://arxiv.org/abs/2606.31232)
  - PSG-JEPA — [arXiv 2608.06799](https://arxiv.org/abs/2608.06799)

### Inferences
- The comparison study's epoch-0 floor is the world-model analogue of Zhang & Bowman's random-init baseline. Reporting trained-minus-epoch-0 is analogous to Hewitt & Liang selectivity.
- A pixel-rendered pole angle is close to a geometric (nearly linear) function of the image, so a random ViT preserving that information is unsurprising. This parallels RigidBench's finding that position is partly decodable from a random DiT while velocity stays weak even in trained models. I did not find a paper that states this for cart-pole explicitly.

### Gaps
- I found no paper reporting random-encoder state-probe R² specifically for rendered cart-pole or Acrobot.
- I found no paper applying Hewitt-style control tasks (as opposed to random-init or shuffled labels) to world-model probing.
