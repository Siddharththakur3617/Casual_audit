# Causal representation learning from pixels and causal abstraction / interchange interventions: prior work relevant to the JEPA cart-pole / Acrobot causal-graph study

Scope note: this file covers (a) causal representation learning (CRL) that recovers causal variables and/or graphs from pixels of physical or interactive systems, and (b) intervention-based interpretability (causal abstraction, interchange interventions, DAS, activation patching, mediation, amnesic probing, INLP, probing critiques), plus applications to world models or agents. Per the assignment it does not cover V-JEPA or Othello-GPT probing beyond interventions, CDL/GRADER-style causal MBRL, or LOCO/knockoff statistics.

Verification note: venue, year and core claims below come from searches and fetches made in this session, except where a bullet says "(primary link; details from the paper, not re-fetched this session)". For those, the arXiv/venue link is the primary source, and the report writer should treat finer details as lower-confidence.

Closeness scale relative to the study (a pixels-only, action-conditioned JEPA with hidden per-episode physics, tested post hoc against a simulator-Jacobian expert graph via a differential probe):
- **Very close**: same kind of system (cart-pole, pendulum, physics from video) and the object studied is the action/state→state-change graph.
- **Partially**: same question (causal variables/graph from pixels, or intervention tests of whether a network implements a causal model), but different systems or a different object.
- **Loosely related**: methodology or critique only.

## (a) Causal representation learning from pixels with temporal / interventional data

### Takeaway
CRL work learns causal variables, and sometimes their graph, **by construction**. Each method has a bespoke architecture and an identifiability assumption: known intervention targets (CITRIS/iCITRIS), binary agent interactions (BISCUIT), paired pre/post-intervention data (Brehmer et al.), sparse mechanisms with actions (Lachapelle et al.), or nonstationary domains (LEAP/TDRL). Methods are scored with MCC/R² against ground-truth latents and, where a graph is learned, against the true graph. The closest precedents are:
- **TDRL**: recovers cart position and pole angle from *cart-pole video* under six hidden gravity domains.
- **Mechanism-sparsity disentanglement (Lachapelle et al.)**: learns exactly an action→latent and latent→latent graph.
- **V-CDN**: causal graphs of physical systems from video.

None of these tests whether a *generic, already-trained* world model (such as a JEPA) implicitly implements the ground-truth graph. That post-hoc testing angle is what distinguishes the study.

### Cited Findings

**Position / framing papers**
- **Schölkopf, Locatello, Bauer, Ke, Kalchbrenner, Goyal, Bengio, "Toward Causal Representation Learning"**, Proceedings of the IEEE 2021. Position paper that defines CRL as discovering high-level causal variables from low-level observations such as pixels, and motivates interventions, modularity and sparse mechanism shift. [arXiv 2102.11107](https://arxiv.org/abs/2102.11107) (primary link; details from the paper, not re-fetched this session). Closeness: loosely related (framing).

**Temporal / interventional CRL from pixels (by construction)**
- **Lippe, Magliacane, Löwe, Asano, Cohen, Gavves, "CITRIS: Causal Identifiability from Temporal Intervened Sequences"**, ICML 2022 ([arXiv 2202.03169](https://arxiv.org/abs/2202.03169); primary link; details from the paper, not re-fetched this session).
  - What it does: VAE-style CRL on image sequences in which the intervention targets at each step are known. It identifies (possibly multidimensional) causal variables.
  - Environments: Temporal Causal3DIdent and Interventional Pong.
  - Ground truth: true latent factors.
  - Type: by construction. Closeness: partially.
- **Lippe et al., iCITRIS**, arXiv June 2022, published at ICLR 2023 as "Causal Representation Learning for Instantaneous and Temporal Effects in Interactive Systems" ([arXiv 2206.06169](https://arxiv.org/abs/2206.06169)).
  - What it does: handles *instantaneous* effects, which arise because "slower measurement or frame rate than many causal effects" breaks the earlier CITRIS-style identifiability. It needs perfect interventions with known targets. It identifies the causal variables while "simultaneously using a differentiable causal discovery method to learn their causal graph".
  - Results: reported to accurately identify the variables and the graph on three interactive-system datasets ([deepai abstract mirror](https://api.deepai.org/publication/icitris-causal-representation-learning-for-instantaneous-temporal-effects)).
  - Code and datasets: Temporal Causal3DIdent, Causal Pinball and a Voronoi benchmark ([GitHub phlippe/CITRIS](https://www.github.com/phlippe/CITRIS)).
  - Type: learns the graph by construction. Closeness: partially-to-very close; it is the closest CRL analogue of "learn the graph over latent causal variables from frames".
- **Lippe, Magliacane, Löwe, Asano, Cohen, Gavves, "BISCUIT: Causal Representation Learning from Binary Interactions"**, UAI 2023, PMLR 216 ([PMLR](https://proceedings.mlr.press/v216/lippe23a/lippe23a.pdf); [arXiv 2306.09643](https://arxiv.org/pdf/2306.09643.pdf); [anthology](https://mlanthology.org/uai/2023/lippe2023uai-biscuit)).
  - Identifiability: causal variables are identifiable if the agent's interactions with each variable can be described by an unknown binary variable, i.e. each variable has two mechanisms (for example, observational and interventional).
  - Method: a VAE that learns the variables and the binary interactions without supervision.
  - Environments: Voronoi, CausalWorld and iTHOR, all robotics-inspired.
  - Type: by construction. Closeness: partially, because the actions there are also the "interventions", as in the study's action-conditioning.
- **Brehmer, de Haan, Lippe, Cohen, "Weakly supervised causal representation learning"**, NeurIPS 2022 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa567e2b2c870f8f09a87b6e73370869-Abstract.html); [arXiv 2203.16437](https://arxiv.org/pdf/2203.16437)).
  - Setting: pairs of observations before and after an unknown, random, single-node intervention, with the noise held fixed.
  - Identifiability: the latent causal model (SCM plus decoder) is identifiable up to relabelling and elementwise reparameterisation, for arbitrary unknown graphs. This requires perfect, stochastic interventions with all interventions present.
  - Methods: implicit latent causal models (ILCM), plus an explicit variant (ELCM).
  - Environments: simple images and a simulated robotic manipulation dataset (CausalCircuit).
  - Type: by construction. Closeness: partially.
- **Yang, Liu, Chen, Shen, Hao, Wang, "CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models"**, CVPR 2021 ([arXiv 2004.08697](https://arxiv.org/abs/2004.08697); primary link; details from the paper, not re-fetched this session).
  - What it does: a VAE with a linear SCM layer. It uses concept labels as supervision.
  - Data: the widely reused synthetic **Pendulum** image dataset (pendulum angle and light position cause shadow length and position) and a water-flow dataset.
  - Ground truth: a known 4-variable graph.
  - Type: by construction (supervised). Closeness: partially. It is a pendulum, but the causal relations are static rendering relations, not dynamics.
- **Yao, Sun, Ho, Sun, Zhang, "Learning Temporally Causal Latent Processes from General Temporal Data" (LEAP)**, ICLR 2022 ([arXiv 2110.05428](https://arxiv.org/abs/2110.05428); primary link; details from the paper, not re-fetched this session).
  - Identifiability: for time-delayed latent causal processes under nonstationary noise.
  - TDRL later notes that LEAP's identifiability holds only for limited nonstationary cases, where the noise distribution must vary across all segments ([TDRL, arXiv 2210.13647](https://www.arxiv.org/abs/2210.13647)).
  - Closeness: partially.
- **Yao, Chen, Zhang, "Temporally Disentangled Representation Learning" (TDRL)**, NeurIPS 2022 ([proceedings](https://papers.neurips.cc/paper_files/paper/2022/hash/a938292feb86b94ebe3e6200ff7786ef-Abstract-Conference.html); [arXiv 2210.13647](https://www.arxiv.org/abs/2210.13647)).
  - Identifiability: for time-delayed latent variables with nonparametric transitions, under fixed or changing dynamics and modular distribution shifts.
  - **Modified Cartpole experiment** ([ar5iv full text](https://ar5iv.labs.arxiv.org/html/2210.13647)):
    - Input: video frames, encoded by a VAE.
    - Domains: six source domains with gravity ∈ {5, 10, 15, 20, 25, 30}. Two discrete actions (left/right) times six gravities give 12 segments.
    - Model: latent size 8, max lag 2, 2-D change factors.
    - Results: two latents align with cart position and pole angle (checked by MCC, scatterplots and traversals). TDRL is reported to beat the baselines (LEAP, SlowVAE, PCL, i-VAE, TCL, β-VAE, KVAE, DVBF), but the portion read did not give cart-pole MCC numbers.
    - Graph: I could not confirm that the transition graph was scored against a ground-truth graph for cart-pole. The LassoNet causal visualisation is described as optional post-processing.
  - Type: by construction (latents, with the graph only implicitly). Closeness: **very close**. It uses the same environment and the same hidden per-episode physics (gravity) idea, but it recovers variables rather than testing a trained world model's graph.
- **Lachapelle, Rodríguez López, Sharma, Everett, Le Priol, Lacoste, Lacoste-Julien, "Disentanglement via Mechanism Sparsity Regularization: A New Principle for Nonlinear ICA"**, CLeaR 2022 ([arXiv 2107.10098](https://arxiv.org/abs/2107.10098v2)).
  - Identifiability: latents are identifiable up to permutation when latent mechanisms are regularised to be sparse and a graph-connectivity criterion holds.
  - Method: a VAE with learned binary masks over the **latent→latent (temporal) and action→latent** graphs. Unknown-target interventions are linked to disentanglement.
  - Extended version, "Nonparametric Partial Disentanglement via Mechanism Sparsity: Sparse Actions, Interventions and Sparse Temporal Dependencies", JMLR vol. 27 (2026) ([JMLR](https://jmlr.org/papers/v27/24-0771.html); [arXiv 2401.04890](https://ar5iv.labs.arxiv.org/html/2401.04890)). It drops the graphical criterion, allows arbitrary graphs and gives "partial disentanglement" (the "consistency" equivalence), where some factors can stay entangled.
  - Type: by construction. Closeness: **very close in object**. The learned masks are precisely an action/state→next-state graph. Partial disentanglement is a formal analogue of the study's collinear θ/θ̇ difficulty.
- **Li, Torralba, Anandkumar, Fox, Garg, "Causal Discovery in Physical Systems from Videos" (V-CDN)**, NeurIPS 2020 ([proceedings](https://proceedings.neurips.cc/paper/2020/hash/6822951732be44edf818dc5a97d32ca6-Abstract.html); [arXiv 2007.00631](https://arxiv.org/abs/2007.00631v3)).
  - What it does: end-to-end causal discovery from video without ground-truth graph supervision. Unsupervised keypoints act as state, an inference module infers the graph, and a dynamics module predicts the future.
  - Data: episodes come from *different causal graphs that share a dynamics model*, which is analogous to hidden per-episode physics. Environments are planar multi-body interaction and fabrics.
  - Results: counterfactual prediction and extrapolation to unseen graphs.
  - Type: by construction. Closeness: **very close** (physics from pixels, per-episode structure).
- **Kipf, Fetaya, Wang, Welling, Zemel, "Neural Relational Inference for Interacting Systems"**, ICML 2018 ([arXiv 1802.04687](https://arxiv.org/abs/1802.04687); primary link; details from the paper, not re-fetched this session). Infers interaction graphs (springs, charged particles) from trajectories, not pixels. Closeness: loosely related; it is the precursor of V-CDN.

**Identifiability results from interventions (mostly non-pixel theory, relevant to action-conditioning)**
- **Ahuja, Mahajan, Wang, Bengio, "Interventional Causal Representation Learning"**, ICML 2023 ([arXiv 2209.11924](https://arxiv.org/abs/2209.11924); primary link; details from the paper, not re-fetched this session). Shows that do-interventions (hard interventions that fix a latent's support) give identifiability up to permutation and scaling under polynomial decoders. Closeness: loosely related (theory).
- **Squires, Seigal, Bhate, Uhler, "Linear Causal Disentanglement via Interventions"**, ICML 2023 ([arXiv 2211.16467](https://arxiv.org/abs/2211.16467)), and **von Kügelgen, Besserve, Wendong, Gresele, Kekić, Bareinboim, Blei, Schölkopf, "Nonparametric Identifiability of Causal Representations from Unknown Interventions"**, NeurIPS 2023 ([arXiv 2306.00542](https://arxiv.org/abs/2306.00542)). Both: primary links; details from the papers, not re-fetched this session. They give identifiability of the latent graph from single-node interventions in linear and nonparametric settings. Score-based CRL (Varici et al.) gives graph-recovery guarantees under soft interventions ([project page](https://bvarici.github.io/projects/CRL/)). Closeness: loosely related.
- **Huang, Feng, Lu, Magliacane, Zhang, "AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning"**, ICLR 2022 ([arXiv 2107.02729](https://arxiv.org/abs/2107.02729); primary link; details from the paper, not re-fetched this session). Learns a structured world model with domain-specific "change factors". The Cartpole experiments vary physical parameters (such as gravity and mass) across domains, including from pixels. Boundary note: this is causal-structure MBRL, which another researcher covers. It is listed only because its change factors are the closest analogue of "hidden per-episode physics". Closeness: partially.

**Benchmarks with physical / causal ground truth**
- **Liu, Alahi, Russell, Horn, Zietlow, Schölkopf, Locatello, "Causal Triplet: An Open Challenge for Intervention-centric Causal Representation Learning"**, CLeaR 2023, PMLR 213 pp. 553–573 ([arXiv 2301.05169](https://arxiv.org/abs/2301.05169v1); [anthology](https://mlanthology.org/clear/2023/liu2023clear-causal); [project](https://sites.google.com/view/causaltriplet)).
  - Task: infer which high-level action (intervention) happened between an image pair, in visually complex scenes with OOD tests.
  - Findings: object-centric and disentangled representations beat distributed ones, but a large ID/OOD gap remains.
  - Closeness: loosely related (action-as-intervention from pixels, but no dynamics graph).
- **Causal3DIdent**: introduced by von Kügelgen et al., "Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style", NeurIPS 2021 ([arXiv 2106.04619](https://arxiv.org/abs/2106.04619); primary link; details from the paper, not re-fetched this session). Temporal and instantaneous versions are used by CITRIS/iCITRIS ([GitHub](https://www.github.com/phlippe/CITRIS)). Other CRL datasets: the **CausalVAE Pendulum**, Interventional Pong, Causal Pinball, CausalWorld/iTHOR (BISCUIT), and TDRL's **modified Cartpole** (see above).
- **Gamella, Peters, Bühlmann, "Causal chambers as a real-world physical testbed for AI methodology"**, Nature Machine Intelligence 2025 ([arXiv 2404.11341](https://arxiv.org/abs/2404.11341); primary link; details from the paper, not re-fetched this session). Real physical devices (a light tunnel and a wind tunnel) with known causal graphs, including image outputs. Closeness: loosely related (a possible external benchmark).
- **Recent world-model variant: "Causal-JEPA: Learning World Models through Object-Level Latent Interventions"**, arXiv 2602.11389, 2026 preprint ([arXiv](https://arxiv.org/html/2602.11389v1)). Masks object states in latent space during training as a form of "latent intervention" to induce interaction-dependent prediction. It is reported to be agnostic about identifiability and does not recover a fixed graph ([ResearchGate summary](https://www.researchgate.net/publication/400741741_Causal-JEPA_Learning_World_Models_through_Object-Level_Latent_Interventions)). Type: training method. Closeness: partially (JEPA plus "interventions", but not a graph test).

### Inferences
- **Novelty.** None of the CRL papers found asks the study's question: did a standard, non-causal, pixels-only world model (a JEPA) learn the ground-truth action/state→state-change graph, tested post hoc against a simulator Jacobian? CRL papers build the graph into the model (masks, known targets, differentiable discovery) and evaluate latent recovery (MCC) and sometimes graph recovery. The study's post-hoc evaluation angle appears novel relative to this literature.
- **Suggested CRL baselines.** TDRL (cart-pole video, gravity domains) and Lachapelle et al.'s mechanism-sparsity model (an action→latent and latent→latent graph) are the most natural *by-construction* baselines. For example, train the mechanism-sparsity model or TDRL on the same rendered cart-pole/Acrobot data and compare its learned masks with the JEPA-derived graph and the expert graph.
- **Collinear θ/θ̇.** The partial-disentanglement and "consistency" results in the JMLR 2026 mechanism-sparsity paper are the formal statement that some latent factors cannot be separated without enough sparsity or intervention diversity. This could be cited to frame the collinearity problem as an identifiability limit rather than a methodological flaw.
- **Instantaneous effects.** iCITRIS's point about frame rate creating instantaneous effects is relevant to cart-pole/Acrobot at coarse frame skip: velocity→position effects within one frame step could appear "instantaneous".

### Gaps
- I did not confirm whether TDRL scored the learned cart-pole transition graph against a ground-truth graph, or what its cart-pole MCC values are. The last ~40k characters of the full text were not read.
- I did not verify whether iCITRIS includes a pendulum-like environment. The search found only Causal3DIdent, Causal Pinball and Voronoi.
- I found no CRL paper on Acrobot (double pendulum) from pixels with graph recovery.
- AdaRL details (exact pixel cart-pole setup) are from memory of the paper, not re-verified.

## (b) Causal abstraction and intervention-based interpretability: testing whether a network implements a causal model

### Takeaway
Causal abstraction gives the formal test the study lacks. Pick an alignment between high-level variables (x, ẋ, θ, θ̇, action) and subspaces of the network's representation. Run **interchange interventions**: swap that subspace from a "source" input into a "base" input. Then score **interchange intervention accuracy (IIA)**, the fraction of swaps where the network's output matches what the high-level causal model predicts. **DAS** learns the subspace (a rotation) by gradient descent.

These methods were developed almost entirely on language models and toy tasks. The important caveat is that, with non-linear alignment maps, causal abstraction becomes vacuous: any network can be aligned to any algorithm (Sutter et al., NeurIPS 2025). This bears directly on the study's MLP-based differential probe.

### Cited Findings

**Core causal-abstraction papers**
- **Geiger, Lu, Icard, Potts, "Causal Abstractions of Neural Networks"**, NeurIPS 2021 ([arXiv 2106.02997](https://arxiv.org/abs/2106.02997); primary link; details from the paper, not re-fetched this session). Introduces interchange interventions to test whether a network's internal representations realise a high-level causal model. Applied to NLI models. Type: tests an existing model. Closeness: partially (the method the study needs).
- **Geiger, Wu, Lu, Rozner, Kreiss, Icard, Goodman, Potts, "Inducing Causal Structure for Interpretable Neural Networks" (IIT)**, ICML 2022 ([PMLR](https://proceedings.mlr.press/v162/geiger22a/geiger22a.pdf)). Interchange intervention training trains a network so that it realises a given causal model's abstract structure ([search summary of the paper](https://arxiv.org/pdf/2301.04709)). Type: by construction. Closeness: partially. It is a possible training-time variant: train the JEPA predictor so that swapping the "ẋ subspace" produces the simulator's counterfactual.
- **Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, Icard, "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability"**, JMLR 26 (2025), pp. 1–64 ([JMLR](https://jmlr.org/papers/v26/23-0058.html); [arXiv 2301.04709](https://arxiv.org/abs/2301.04709)).
  - Generalises causal abstraction from hard and soft interventions to arbitrary transformations of mechanisms.
  - Formalises polysemanticity, the linear representation hypothesis, modular features and graded faithfulness.
  - Unifies activation and path patching, causal mediation, causal scrubbing, circuit analysis, SAEs, DAS and steering in one framework.
  - IIA is defined as the proportion of aligned interchange interventions on which the algorithm and the network give the same output.
  - Notes that successful interchange-intervention analysis implies controllability of the low-level model, but not the converse ([arXiv PDF](https://arxiv.org/pdf/2301.04709)).
  - Closeness: partially (the theoretical foundation).
- **Geiger, Wu, Potts, Icard, Goodman, "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations" (DAS)**, CLeaR 2024, PMLR 236 ([PMLR](https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf); [arXiv 2303.02536](https://arxiv.org/pdf/2303.02536)). Earlier alignment methods were "localist" (one variable per set of neurons). DAS learns a rotation and subspace per high-level variable by gradient descent, and IIA is the graded score. Closeness: partially. It is directly adoptable, because JEPA latents are distributed.
- **Wu, Geiger, Icard, Potts, Goodman, "Interpretability at Scale: Identifying Causal Mechanisms in Alpaca" (Boundless DAS)**, NeurIPS 2023 ([NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2023/file/f6a8b109d4d4fd64c75e94aaf85d9697-Paper-Conference.pdf); [arXiv 2305.08809](https://arxiv.org/pdf/2305.08809)). Makes the subspace size learnable through soft boundaries ([search summary](https://arxiv.org/pdf/2303.02536)). Closeness: loosely related (scaling).

**Benchmarks, critiques and refinements of DAS / patching**
- **Mueller et al., "MIB: A Mechanistic Interpretability Benchmark"**, ICML 2025 ([arXiv 2504.13151](https://arxiv.org/pdf/2504.13151)). Includes a causal-variable track that uses interchange interventions to check whether a hypothesised causal model abstracts an LM. Closeness: loosely related (evaluation protocol).
- **Sutter, Minder, Hofmann, Pimentel, "The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?"**, NeurIPS 2025 Spotlight ([arXiv 2507.08802](https://arxiv.org/pdf/2507.08802); [NeurIPS](https://neurips.cc/virtual/2025/poster/117348)).
  - Main result: with arbitrarily powerful alignment maps, any network can be mapped to any algorithm, so unrestricted causal abstraction is trivial.
  - Empirical evidence: non-linear maps on *randomly initialised* models reproduce DAS success, and IOI aligns to Pythia models that cannot solve IOI.
  - Closeness: partially. It is a critical caveat for the MLP probe and any non-linear alignment.
- **Makelov, Lange, Nanda, "Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching"**, ICLR 2024 ([arXiv 2311.17030](https://arxiv.org/abs/2311.17030); primary link; details from the paper, not re-fetched this session). Subspace patching can change outputs by activating a dormant parallel pathway, not the true variable. Closeness: loosely related (caveat).
- **Zhang & Nanda, "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods"**, ICLR 2024 ([arXiv 2309.16042](https://arxiv.org/pdf/2309.16042)). Best-practice guidance on corruption method and metric choice. Closeness: loosely related.
- **"Addressing divergent representations from causal interventions on neural networks"**, arXiv 2511.04638 (2025) ([arXiv](https://arxiv.org/pdf/2511.04638)). Concerns interventions pushing representations off the natural distribution. I saw only the title and search snippet, not the content. Closeness: loosely related (caveat for latent swaps).
- **"Combining Causal Models for More Accurate Abstractions of Neural Networks"**, arXiv 2503.11429 ([arXiv](https://arxiv.org/pdf/2503.11429)), and **"Bucketing the Good Apples"**, arXiv 2605.02234 ([arXiv](https://arxiv.org/html/2605.02234v1)). Recent refinements for partial or input-dependent abstraction; title and snippet only. Closeness: loosely related.
- **Huang, Wu, Potts, Geva, Geiger, "RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations"**, ACL 2024 ([arXiv 2402.17700](https://arxiv.org/abs/2402.17700); primary link; details from the paper, not re-fetched this session). Scores interventions on two axes: *Cause* (intervening on attribute A changes A) and *Isolate* (it leaves correlated attribute B unchanged). Closeness: partially. This is a ready-made metric pair for the collinear θ/θ̇ separation problem.

**Causal mediation, erasure and probing critiques**
- **Vig, Gehrmann, Belinkov, Qian, Nevo, Singer, Shieber, "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"**, NeurIPS 2020 ([arXiv 2004.12265](https://arxiv.org/abs/2004.12265); primary link; details from the paper, not re-fetched this session). Estimates total, direct and indirect effects of neurons and attention heads by intervening on inputs and activations. Closeness: loosely related. Direct/indirect effect decomposition maps onto "does ẋ affect Δθ̇ directly or only via θ̇ in the latent".
- **Elazar, Ravfogel, Jacovi, Goldberg, "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals"**, TACL 2021 ([arXiv 2006.00995](https://arxiv.org/abs/2006.00995)). Removes a property from representations with INLP and measures the change in the model's behaviour. This contrasts "decodable" with "used". Closeness: partially. It is a cheaper alternative to interchange interventions for the "used vs probe artefact" question.
- **Ravfogel, Elazar, Gonen, Twiton, Goldberg, "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection" (INLP)**, ACL 2020 ([arXiv 2004.07667](https://arxiv.org/abs/2004.07667)). Linear concept removal by iterated nullspace projection. Closed-form successor: **Belrose et al., "LEACE: Perfect Linear Concept Erasure in Closed Form"**, NeurIPS 2023 ([arXiv 2306.03819](https://arxiv.org/abs/2306.03819)). Both: primary links; details from the papers, not re-fetched this session. Closeness: loosely related (tools).
- **Ravfogel, Prasad, Linzen, Goldberg, "Counterfactual Interventions Reveal the Causal Effect of Relative Clause Representations on Agreement Prediction" (AlterRep)**, CoNLL 2021 ([arXiv 2105.06965](https://arxiv.org/abs/2105.06965); primary link; details from the paper, not re-fetched this session). Pushes representations to the other side of a probe's boundary and checks the downstream effect. Closeness: loosely related.
- **Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances"**, Computational Linguistics 48(1), 2022 ([arXiv 2102.12452](https://arxiv.org/abs/2102.12452)), and **Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks"**, EMNLP 2019 ([arXiv 1909.03368](https://arxiv.org/abs/1909.03368)). Both: primary links; details from the papers, not re-fetched this session. Two core points: probe accuracy shows information is *present*, not *used*; and expressive probes can learn the task themselves, so control tasks/selectivity and simple probes are needed. Closeness: partially (the exact critique the study faces).
- **Causal scrubbing**: Chan et al. (Redwood Research), 2022, [Alignment Forum post](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) (primary link; details from the post, not re-fetched this session). Tests a hypothesis by resampling activations that the hypothesis says are irrelevant, and measuring the preserved loss. The JMLR causal-abstraction paper unifies it as a special case ([arXiv 2301.04709](https://arxiv.org/abs/2301.04709)). Closeness: loosely related.

### Inferences (concrete methods the study could adopt and what they add)

1. **Use the simulator's counterfactuals as the high-level model.** This is a big advantage over the LLM setting. The simulator *is* the high-level causal model, so the exact counterfactual next state is available for any "base state with ẋ replaced by the source's ẋ". IIA can therefore be computed directly against ground truth rather than against a hand-written algorithm.
2. **DAS on the encoder latent, read out at the predictor.**
   - Learn a rotation R of the JEPA latent z_t (or of the predictor's hidden layer). Use one subspace per high-level variable (x, ẋ, θ, θ̇), trained so that swapping the ẋ-subspace from source to base makes the predictor's predicted latent change match the encoder latent change of the simulator's counterfactual next frame.
   - Report IIA, or an R²-style graded IIA for continuous outputs, per (input variable → output variable) edge.
   - What it adds: an edge such as ẋ→θ̈ is credited only if intervening on the model's ẋ-representation changes the predicted Δθ̇ the way physics says. That separates "model uses ẋ" from "probe exploits correlation".
3. **Interventions break collinearity.** Choose source/base pairs whose θ̇ is decorrelated from the other inputs (for example, same θ, different θ̇). The interchange then creates combinations that are rare in observational rollouts, which observational LOCO/decoy scoring cannot do. RAVEL-style *Cause* and *Isolate* scores quantify whether the θ̇ and θ subspaces are actually separable.
   - Caveats: off-manifold latents (arXiv 2511.04638) and Makelov et al.'s dormant-pathway illusion. Mitigate by restricting swaps to physically plausible combinations, which the simulator can render, so inputs can also be intervened on at the pixel level.
4. **Pixel-level ("input") interventions as a cheaper first step.** Re-render frame pairs from the simulator with one variable changed (for example, ẋ changed via the previous frame), feed them to the encoder, and measure the change in predicted Δ. This needs no alignment search, so it sidesteps the Sutter et al. non-linear-map critique for the *input* side. The output side is still read by the differential probe, so add a control (item 6).
5. **Amnesic / LEACE ablation.** Linearly erase ẋ information from z_t (INLP/LEACE) and measure the drop in the predictor's accuracy on Δθ̇. This gives a "used vs merely decodable" test in the spirit of Elazar et al. It is weaker than interchange (erasure can damage correlated θ̇ information), but cheap.
6. **Controls the study should add, given Sutter et al. and Hewitt & Liang.**
   - Restrict alignment maps to linear or orthogonal ones (as in DAS).
   - Run the same differential-probe and DAS pipeline on a **randomly initialised JEPA** and on an untrained or shuffled-action predictor. Edges recovered there are probe or alignment artefacts.
   - Report probe selectivity against a control task.
7. **IIT as an optional by-construction variant.** Interchange intervention training would turn the study into "can we induce the true graph", which is a different claim. It is useful as an upper-bound reference.
8. **Mediation decomposition (Vig et al.)** could separate the direct ẋ→Δθ̇ effect from the effect mediated by the latent θ̇ coordinate. This is relevant because the simulator Jacobian lists direct effects.

### Gaps
- I found no paper applying DAS, IIA or interchange interventions to a continuous-control dynamics model or learned physics simulator. The search for "distributed alignment search learned physics simulator / dynamics model" returned only LM-focused work ([DAS](https://arxiv.org/pdf/2303.02536); [MIB](https://arxiv.org/pdf/2504.13151)).
- How to define IIA for *continuous* regression outputs (latent change vectors) is not standardised in the sources read. The LM work uses discrete-output match. A tolerance-based or R²-based IIA would have to be defined by the study.
- I did not read the full text of arXiv 2511.04638 or 2605.02234.

## Has anyone applied causal-abstraction-style tests to check whether a world model implements the ground-truth physical causal graph?

### Takeaway
I found no published work that uses interchange interventions, DAS or IIA to test whether a learned world model (JEPA, Dreamer/RSSM, or similar) implements a ground-truth *physical* causal graph. The nearest neighbours are:
- interventions on board-state representations in Othello-GPT (a discrete game, not physics);
- intervention-verified planning representations in a Sokoban RL agent;
- 2026 preprints that edit hidden states of a recurrent world model in a collision environment and check rollout specificity, without scoring against a causal graph;
- interpretability tooling (WorldModelLens) that supports activation patching on Dreamer/IRIS/I-JEPA.

This looks like a genuine gap the study could fill. That conclusion is limited by search coverage: absence of evidence, not proof.

### Cited Findings
- **Li, Hopkins, Bau, Viégas, Pfister, Wattenberg, "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task"**, ICLR 2023 ([arXiv 2210.13382](https://arxiv.org/abs/2210.13382); primary link; details from the paper, not re-fetched this session). Beyond probes, it *intervenes* on Othello-GPT activations by editing the probed board state and shows that next-move predictions change consistently with the edited board. This makes it the canonical "intervene on the world-model latent and check consistency with the true world" design. The world is discrete (a board), not a physical causal graph. Closeness: partially.
- **Bush, Chung, Anwar, Garriga-Alonso, Krueger, "Interpreting Emergent Planning in Model-Free Reinforcement Learning"**, ICLR 2025 oral ([arXiv 2504.01871](https://arxiv.org/abs/2504.01871v1); [ICLR](https://iclr.cc/virtual/2025/oral/31895)). A three-step protocol on a DRC agent in Sokoban: probe for concepts, study how plans form, and verify through interventions that the plans causally affect behaviour. Closeness: loosely related (an agent, not a physics dynamics model; probe-then-intervene template).
- **"Mechanistic Interpretability of Reinforcement Learning Agents"**, arXiv 2411.00867 (2024 preprint) ([arXiv](https://arxiv.org/html/2411.00867v1)). Activation patching on a maze agent's neurons to steer behaviour. Closeness: loosely related.
- **Richens et al., "General agents contain world models"**, arXiv 2506.01622 (ICML 2025) ([arXiv](https://arxiv.org/pdf/2506.01622)). Argues the causal role of world-model features is established by intervening on their representations and checking that the policy changes as if the world changed. The paper's own method is theoretical and policy-based. Closeness: loosely related.
- **Liu & Chen, "Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models"**, arXiv 2608.15156 (Aug–Sep 2026 preprint) ([arXiv](https://arxiv.org/abs/2608.15156)).
  - Setup: a 192-D recurrent world model in a two-object collision environment, trained on factual and locally edited counterfactual trajectories.
  - Method: learns a rank-4 hidden-state "carrier" subspace for velocity edits, and checks that a one-shot patch launches the intended 12-step rollout.
  - Controls: no-op, random and wrong-target edits; a position-edit stress test fails its specificity controls.
  - Gap: no scoring against a ground-truth causal graph was described in the abstract.
  - The search snippet also stresses that decodability alone does not show the model relies on the information ([arXiv PDF](https://arxiv.org/pdf/2608.15156)).
  - Closeness: partially. It is the closest "interventions on a physics world model's latents" work found, but it is a preprint and not graph-level.
- **"Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models"**, arXiv 2609.21787 (2026 preprint) ([arXiv](https://arxiv.org/pdf/2609.21787)). Asks whether a small latent correction stays organised and effective as a recurrent world model rolls out. Title and snippet only. Closeness: partially or loosely.
- **Challagundla et al., "One Lens, Many Worlds: A Capability-Typed Interface for World-Model Interpretability" (WorldModelLens)**, arXiv 2606.09936 (June 2026 preprint) ([arXiv](https://arxiv.org/abs/2606.09936)). An open-source toolkit with hooks, caches, activation patching and intervention replay for PlaNet/Dreamer, IRIS and I-JEPA. From the abstract, it has no interchange-intervention or causal-abstraction tests and no ground-truth causal-graph checks. Closeness: loosely related (possible tooling).
- **Causal-JEPA**, arXiv 2602.11389 (2026). Uses latent "interventions" (object masking) as a *training* signal, not as a test of a learned graph ([arXiv](https://arxiv.org/html/2602.11389v1)). Closeness: partially.
- My targeted searches ("causal abstraction interchange intervention world model physics dynamics"; "distributed alignment search learned physics simulator / dynamics model") returned only LM-focused or theory papers. Neither returned an application to physical world models ([JMLR causal abstraction](https://arxiv.org/pdf/2301.04709); [causal abstraction survey, arXiv 2410.20161](https://arxiv.org/pdf/2410.20161)).

### Inferences
- **Novel contribution.** Combining (i) a pixels-only JEPA world model, (ii) a simulator that supplies *exact* counterfactuals, and (iii) DAS/IIA scored per edge against the simulator-Jacobian graph appears to be a novel contribution. It could be framed as "causal abstraction evaluation of a world model against a known physical SCM".
- **Strongest framing.** The observational LOCO/FWER graph is a *hypothesis*. Interchange-intervention IIA per edge is the *test*. Random-init and linear-only-alignment controls answer the Sutter et al. triviality critique.
- **Precedent for the design.** The Othello-GPT intervention design and the Liu & Chen 2026 carrier-subspace edits (velocity edits in a collision world model) are the closest design precedents to cite. Liu & Chen's finding that position edits fail specificity controls is a warning that some physical variables may not be cleanly addressable in recurrent or predictive latents. That parallels the study's θ/θ̇ collinearity.

### Gaps
- I could not access the full texts of the 2026 world-model intervention preprints (2608.15156, 2609.21787, 2606.09936). Their claims are from abstracts or snippets, and their peer-review status is unknown.
- I found no study applying interchange interventions to Dreamer/RSSM latents with respect to physical variables. I am not certain none exists; coverage of workshop papers (for example, NeurIPS/ICLR interpretability or world-model workshops) was not checked.
- I found no work defining IIA for continuous-valued physical counterfactuals, which the study would need to specify itself.
