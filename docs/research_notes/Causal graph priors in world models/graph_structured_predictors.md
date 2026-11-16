# Graph-structured dynamics predictors: a GIVEN graph as the architecture's wiring (GNN / message passing, graph-masked transformers)

Reading conventions used throughout:

- **GIVEN** = the graph is supplied to the model as input or fixed wiring. **LEARNED** = the graph is inferred from data. **IMPLICIT** = fully connected, no graph selection.
- Every paper carries year and venue; "preprint" means no peer-reviewed venue was found on the page read.
- "Page read" is the arXiv abstract page, the arXiv HTML, or the ar5iv HTML rendering. Several were truncated at 100k characters, which is noted where it matters.
- Numbers are as reported by the source. Nothing here was re-run.

---

## 1. How do Interaction Networks, Graph Networks, GNS, NRI and C-SWM use a graph, which take a known graph as input, and what does a true graph buy?

### Takeaway
Interaction Networks and "Graph networks as learnable physics engines" take the graph as a GIVEN input (relations / bodies and joints), NRI learns it but reports a GIVEN "true graph" decoder baseline, and C-SWM uses an IMPLICIT fully connected graph over slots. The only clean true-vs-full-vs-learned comparison found is NRI's Table 2: the true graph gives large gains over a fully connected graph on springs (many orders of magnitude at 1 step) but only modest gains on charged particles and Kuramoto, and the gain over a monolithic MLP shrinks to "negligible" on a system as small as Pendulum.

### Cited Findings

**Interaction Networks (Battaglia et al., 2016; arXiv 1612.00222; venue not stated on the page read, see Gaps) — graph GIVEN**
- The graph G = <O, R> is supplied to the model as input, and is an "attributed, directed multigraph", so several distinct relations may exist between two objects. — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222)
- Formulation: objects O are a D_S x N_O state matrix; relations are a triplet <Rr, Rs, Ra> with Rr and Rs binary N_O x N_R receiver and sender index matrices and Ra a D_R x N_R matrix of relation attributes; external effects X (D_X x N_O, e.g. gravity or control inputs) are applied to each object separately. — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222)
- A relational MLP fR is applied to each interaction term, effects are summed per receiver (aggregation must be commutative and associative), and an object MLP fO maps each object's combined input to its prediction. — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222)
- Graph density in the experiments: n-body and bouncing balls are all-to-all, n(n-1) relations; only the string domain is sparse (springs to immediate neighbours, 2(n-1), plus rigid relations from every mass to the static circle, 2n). — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222)
- Next-step velocity MSE (Table 2), constant velocity / MLP baseline / dynamics-only IN / IN: n-body 82 / 79 / 76 / 0.25; balls 0.074 / 0.072 / 0.074 / 0.0020; string 0.018 / 0.016 / 0.017 / 0.0011. — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222)
- Generalisation across object count: trained on 6 bodies and tested on 3 and 12; trained on 6 balls, tested on 3 and 9; trained on 15 string masses, tested on 5, 15 and 30. — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222)

**Graph networks as learnable physics engines (Sanchez-Gonzalez et al., ICML 2018; arXiv 1806.01242) — graph GIVEN (bodies and joints)**
- Bodies are nodes and joints are edges ("the bodies and joints correspond to the nodes and edges, respectively"); Appendix G indicates edges run parent-to-child or child-to-parent. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- A static graph holds global parameters (time step, gravity), per-body parameters (mass, inertia) and per-joint parameters (joint and motor type); a dynamic graph holds per-body position, orientation quaternion, linear and angular velocity, with action magnitudes placed on edges. Node and edge functions are shared across all bodies and joints. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- Systems: seven simulated MuJoCo systems (Pendulum, Cartpole, Acrobot, Swimmer, Cheetah, Walker2d, JACO arm) plus one real JACO arm. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- Against an MLP baseline (sweep of 20 hyperparameter combinations) the GN had lower error on 6 of 7 simulated systems; the gap was largest on Swimmer (much repeated structure) and negligible on Pendulum (little repeated structure). No error values are given in the text, only figures. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- The authors attribute the better train-to-test generalisation to parameter sharing acting as implicit regularisation, and state the models were "less successful when there was not much opportunity for sharing", with Cheetah as the example. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- Graph sizes: Pendulum has 2 bodies including the world, Cartpole has 3. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- The only architecture ablation visible in the text read is two stacked GNs versus a single GN/IN, with "large performance advantages" for two. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- Abstract-level claims: accurate forward prediction on eight physical systems, system identification, and use in gradient-based trajectory optimisation and offline policy optimisation. — [arXiv 1806.01242](https://arxiv.org/abs/1806.01242)

**Neural Relational Inference (Kipf, Fetaya et al., ICML 2018; arXiv 1802.04687) — graph LEARNED, with GIVEN true-graph and fully-connected baselines**
- NRI is a variational autoencoder whose latent code is the interaction graph and whose decoder is a GNN; it is trained unsupervised on observational trajectories. — [arXiv 1802.04687](https://arxiv.org/abs/1802.04687)
- Decoder: a separate message MLP per edge type, gated by the sampled edge-type variable; the message for pair (i, j) is sum_k z_ij,k * f_e^k([x_i, x_j]); node update is mu_j^{t+1} = x_j^t + f_v(sum_{i != j} h_(i,j)). Self-edges are excluded (the sum runs over i != j) and the self-term enters through the residual x_j^t. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- One edge type can be hard-coded as "non-edge" so that no message passes along it (used for Kuramoto and motion capture); separate MLPs per type are a deliberate choice so the edge type is harder for the model to ignore. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- Baselines: "NRI (full graph)" is the decoder on a fully connected graph with no edge inference; "NRI (true graph)" is the decoder alone given the ground-truth graph; "Supervised" trains the encoder on ground-truth edge labels. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- Future-state MSE, 5 objects (Table 2), at 1 / 10 / 20 steps: — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)

| Model | Springs 1 | Springs 10 | Springs 20 | Charged 1 | Charged 10 | Charged 20 | Kuramoto 1 | Kuramoto 10 | Kuramoto 20 |
|---|---|---|---|---|---|---|---|---|---|
| Static | 7.93e-5 | 7.59e-3 | 2.82e-2 | 5.09e-3 | 2.26e-2 | 5.42e-2 | 5.75e-2 | 3.79e-1 | 3.39e-1 |
| LSTM (single) | 2.27e-6 | 4.69e-4 | 4.90e-3 | 2.71e-3 | 7.05e-3 | 1.65e-2 | 7.81e-4 | 3.80e-2 | 8.08e-2 |
| LSTM (joint) | 4.13e-8 | 2.19e-5 | 7.02e-4 | 1.68e-3 | 6.45e-3 | 1.49e-2 | 3.44e-4 | 1.29e-2 | 4.74e-2 |
| NRI (full graph) | 1.66e-5 | 1.64e-3 | 6.31e-3 | 1.09e-3 | 3.78e-3 | 9.24e-3 | 2.15e-2 | 5.19e-2 | 8.96e-2 |
| NRI (learned) | 3.12e-8 | 3.29e-6 | 2.13e-5 | 1.05e-3 | 3.21e-3 | 7.06e-3 | 1.40e-2 | 2.01e-2 | 3.26e-2 |
| NRI (true graph) | 1.69e-11 | 1.32e-9 | 7.06e-6 | 1.04e-3 | 3.03e-3 | 5.71e-3 | 1.35e-2 | 1.54e-2 | 2.19e-2 |

- Edge recovery accuracy, 5 objects (Table 1): NRI (learned) 99.9% springs, 82.1% charged, 96.0% Kuramoto; supervised 99.9 / 95.0 / 99.7. With 10 objects NRI (learned) falls to 98.4 / 70.8 / 75.7. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- On charged particles the learned graph reaches nearly the true-graph MSE at 1 step (1.05e-3 vs 1.04e-3) although only about 82% of its edges are correct (the text says 82.6%, Table 1 says 82.1%). — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- A joint LSTM over the concatenated state beats NRI (learned) on Kuramoto at 1 step (3.44e-4 vs 1.40e-2) and loses at 20 steps (4.74e-2 vs 3.26e-2). — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)

**C-SWM (Kipf, van der Pol, Welling; arXiv 1911.12247; venue not stated on the page read, see Gaps) — graph IMPLICIT (fully connected)**
- The transition model is a GNN on a fully connected graph over K object slots, O(K^2) messages; the graph is not an input. The paper notes nearest-neighbour connectivity could give linear cost and leaves it to future work. — [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247)
- Each node receives its latent object state plus its one-hot action (zeros if no action applies to it); one round of node-to-edge and edge-to-node updates; the node MLP takes own state, own action and the sum of incoming messages; the output is a per-object delta added to the state. Node and edge MLPs are shared. — [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247)
- Ablation, Hits@1 / MRR at 1, 5, 10 steps. Replacing the GNN by a per-object MLP ("no relational model"): 2D Shapes 99.9/100, 97.4/98.4, 89.7/93.1; 3D Blocks 99.9/99.9, 96.3/97.7, 86.0/90.2, against full C-SWM 100/100, 100/100, 99.9/100 and 99.9/100, 99.9/100, 99.9/99.9. — [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247)
- Removing factored states (single MLP over the scene) costs far more: 2D Shapes 54.5/65.0, 34.4/47.4, 24.1/37.0. Removing the contrastive loss gives 49.9/55.2, 6.5/9.3, 1.4/2.6. — [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247)
- On 3-body physics C-SWM and an unstructured autoencoder world model are close: C-SWM 100/100, 97.2/98.5, 75.5/85.2 versus World Model AE 100/100, 97.7/98.8, 67.9/78.4. — [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247)
- Stated limitations: deterministic and Markov assumptions, no disambiguation of identical objects, variance across seeds and less interpretable slots on Atari. — [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247)

### Inferences
- The size of the "true graph" benefit in NRI tracks how sparse and how informative the graph is. On springs each edge switches an interaction on or off, and the full-graph decoder is about 6 orders of magnitude worse at 1 step. On charged particles (every pair interacts, the edge only sets the sign) full versus true differ by about 5% at 1 step and about 40% at 20 steps. This is the closest published analogue of a dense graph such as cart-pole.
- Two separate mechanisms are bundled in "GNN beats MLP" results: (a) parameter sharing across repeated nodes and edges, and (b) restricting information flow to graph parents. Sanchez-Gonzalez 2018 attribute their gain to (a). A 7-input to 4-output variable graph with heterogeneous variables has almost no repeated structure, so (a) is unavailable and only (b) remains.
- The NRI Kuramoto row is a caution for small systems: at 1 step a plain joint recurrent model was about 40x better than the graph decoder, and the graph model only won at long rollouts.
- In C-SWM the factorisation into per-object states carried much more of the gain than the pairwise message passing.

### Gaps
- Battaglia et al. 2018, "Relational inductive biases, deep learning, and graph networks" (arXiv 1806.01261) was not fetched. From background knowledge, not verified here: it defines the GN block (edge, node and global update functions with permutation-invariant aggregation) and is a position/review paper without new benchmarks.
- Sanchez-Gonzalez et al. 2020 "Learning to Simulate Complex Physics with Graph Networks" (GNS, arXiv 2002.09405, ICML 2020 from background knowledge) was not fetched. From background knowledge, not verified: the graph there is CONSTRUCTED from particle proximity (radius connectivity) each step rather than given or learned, with encode-process-decode and multiple message-passing steps.
- Venues for Interaction Networks (NeurIPS 2016 from background knowledge) and C-SWM (ICLR 2020 from background knowledge) were not stated on the pages read.
- Sanchez-Gonzalez 2018 numeric errors for Cartpole and Acrobot specifically exist only in figures; I could not extract them. Whether the paper ablates graph structure (fully connected, removed edges) could not be confirmed; the last 3.5k characters of the page were not read.
- No paper in this group corrupts the given graph.

---

## 2. Turning a directed causal graph with action nodes and lagged t -> t+1 edges into message passing

### Takeaway
Three concrete recipes exist in the sources: (i) directed sender/receiver relations with typed edges and per-receiver sum aggregation (IN, NRI), (ii) one predictor per output variable with a binary parent mask over per-input features and a permutation-invariant pooling (CDL), and (iii) MADE-style masked fully connected layers derived from the DAG (CFCN). Actions enter either as node-wise "external effects", as node inputs, as features on edges, or as extra input variables that have their own mask column; the self-dependence is usually carried by a residual connection rather than a self-loop edge.

### Cited Findings
- Directed aggregation from parents only: IN encodes each relation with a sender one-hot column in Rs and a receiver one-hot column in Rr, computes one effect per relation and sums effects at the receiver, so a node only receives from nodes that have a relation pointing at it. — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222)
- Edge-typed messages: NRI uses one message MLP per edge type and a hard-coded "non-edge" type that passes nothing. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- Self-dependence by residual instead of self-loop: NRI sums messages over i != j and adds the result to x_j^t; C-SWM likewise predicts a per-object delta added to the current state. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687); [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247)
- Where actions go: IN applies external effects X (including control inputs) to each object separately; C-SWM concatenates a per-object one-hot action to the node input; Sanchez-Gonzalez 2018 place action magnitudes on the joint edges. — [ar5iv 1612.00222](https://ar5iv.labs.arxiv.org/html/1612.00222); [ar5iv 1911.12247](https://ar5iv.labs.arxiv.org/html/1911.12247); [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- Bipartite lagged graph as a per-output mask (Causal Dynamics Learning, Wang et al., ICML 2022, arXiv 2206.13452): one predictive model per next-state variable; the action and every current state variable pass through separate feature extractors; a binary mask M_j sets non-parent features to -infinity; masked features are combined by element-wise max into h_j; a predictive network q_j outputs the distribution of variable j at t+1. — [ar5iv 2206.13452](https://ar5iv.labs.arxiv.org/html/2206.13452)
- In CDL the action is simply one more input feature with its own mask entry, so an "action -> variable j" edge is handled exactly like a state edge. — [ar5iv 2206.13452](https://ar5iv.labs.arxiv.org/html/2206.13452)
- MADE-style masking (CFCN in Vowels et al., "CaTs and DAGs", arXiv 2410.14485, preprint): masks derived from the DAG are adapted to each layer's shape so each output depends only on permitted predecessors; CFCN modifies the adjacency to include an identity diagonal after the first layer. — [arXiv 2410.14485v4](https://arxiv.org/html/2410.14485v4)
- A vector input gate is NOT a graph: in the ICWM technical report the causal mask is applied as a Hadamard product on the input, s_{t+1} = Phi(x_t ⊙ C), and the authors note that reducing the true adjacency to an input-vector gate "yields all ones" on their two-door task because the sparsity sits in the input-by-output transition matrix, which a vector gate cannot express. — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- Lagged inputs: ICWM feeds a history tensor of state, action and reward up to lag tau and masks it with a PCMCI+ derived tensor, i.e. lagged variables are additional masked input entries. — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- Multi-layer leakage: with several masked layers information can hop, so SPARTAN multiplies per-layer (A + I) matrices into a path matrix A_bar and calls token j a parent of token i if A_bar_ij >= 1. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)

### Inferences
- For a 7-input to 4-output bipartite graph the direct translation is recipe (ii): 4 output heads, each with a 7-entry binary row mask (28 mask bits = the 28 possible edges). This is exact for one layer of mixing; any further layers must stay per-output, otherwise the SPARTAN path-matrix argument shows the effective graph becomes the transitive closure of the per-layer masks.
- If the predictor is a multi-layer GNN or transformer over variable tokens, a bipartite t -> t+1 graph is only respected with a single message-passing round from input tokens to output tokens (cross-attention style, as in CaT) or with per-layer masks whose path product equals the intended adjacency.
- Self-loops need an explicit decision: with residual "delta" prediction the output always depends on its own previous value even when the expert graph has no self-edge. For state-change (delta) targets the residual can simply be dropped so the mask alone defines the parents.
- Lagged action nodes fit all three recipes as additional sender nodes/input features with their own mask column; no source treats them as special.

### Gaps
- No source found gives a worked example for a low-dimensional control system with one node per scalar state variable (as opposed to one node per body or object).
- I did not find a paper comparing aggregation choices (sum vs max vs concatenation) under a fixed given graph.
- The CaT appendix proof that DAG constraints are preserved through depth was not in the content read.

---

## 3. Transformer world models with one token per variable/object/slot: how graph-derived attention masks are applied

### Takeaway
Three mechanisms appear: hard masks (adjacency applied to attention scores before softmax, with a KNOWN DAG in the tabular-causal literature: DAG-aware Transformer, CaT), sampled hard masks with a sparsity penalty (SPARTAN, graph LEARNED), and multiplicative reweighting by a hand-specified relation matrix (STICA, policy/value only). The object-centric JEPA-family models named in the brief (C-JEPA, IA-JEPA, HCLSM) do not take a given graph: C-JEPA uses fully bidirectional attention and explicitly estimates no graph.

### Cited Findings

**Hard mask from a KNOWN DAG (tabular causal inference, not world models)**
- DAG-aware Transformer for Causal Effect Estimation (Liu et al., arXiv 2410.10044, preprint as far as the search result shows): the DAG is encoded as a binary adjacency matrix and converted to an attention mask that permits attention only along causal edges plus self-attention; the mask is added to the scaled scores as large negative values before the softmax. The search summary flags an index-convention detail (Eq. 6 masks on the transposed entry) that should be checked against the PDF. — [arXiv 2410.10044](https://arxiv.org/pdf/2410.10044)
- CaTs and DAGs (Vowels et al., arXiv 2410.14485 v4 dated 26 Jan 2026, preprint): the Causal Transformer is a cross-attention model; queries come from a learned embedding, keys and values from the embedded inputs; the transposed, topologically sorted adjacency is applied as a Hadamard mask on the QK^T scores before softmax; the embedded input is re-fed to the heads at every layer. — [arXiv 2410.14485v4](https://arxiv.org/html/2410.14485v4)
- CaT adds no self-connections or identity diagonal after the first layer ("self-connections are not needed"), unlike its masked-MLP sibling CFCN. — [arXiv 2410.14485v4](https://arxiv.org/html/2410.14485v4)
- CaT omits layer normalisation because it would rescale interventions, and reports that an embedding dimension of 1 struggles to disentangle variables. — [arXiv 2410.14485v4](https://arxiv.org/html/2410.14485v4)
- CaT states its robustness depends on "the strength of the constraints imposed via the DAG, and the coherence between this DAG and the true, underlying DGP", and assumes the DAG is known a priori. — [arXiv 2410.14485v4](https://arxiv.org/html/2410.14485v4)

**Sampled hard attention with sparsity penalty — graph LEARNED (SPARTAN)**
- SPARTAN (Lei, Schölkopf, Posner; arXiv 2411.06890, v1 Nov 2024 titled "A Sparse Transformer Learning Local Causation", v3 5 Dec 2025 retitled "A Sparse Transformer World Model Attending to What Matters"; no venue named on the page read). — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- Mechanism: for each query-key pair an edge A_ij ~ Bernoulli(sigmoid(q_i^T k_j)) is sampled with the Gumbel-softmax trick; scaled dot-product attention is computed with A as a hard mask; update s_i^{t+1} = MLP(h_i + s_i). — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- Objective: E[L + lambda * |A_bar|] where A_bar is the product over layers of (A + I); lambda is set by a Lagrangian scheme that minimises |A_bar| subject to MSE <= tau, with tau the error of a fully connected model (described as analogous to GECO). — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- One token per object (Interventional Pong: 4 objects, 32-dim; CREATE: 64-dim tokens; Waymo traffic: about 800 map lines and 50 vehicles); a learnable codebook of intervention tokens is appended instead of action tokens. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- Table 1, prediction error; structural Hamming distance (lower better): SPARTAN 8.60; 1.51 (Pong), 246.6; 1.17 (CREATE), 0.52; 6.84 (Traffic). Dense SlotFormer-like Transformer 8.83; 6.37, 265.5; 6.29, 0.51; 10.6. Global-graph masked-MLP ensemble (VCD/AdaRL style) 11.36; 5.42 and 277.4; 8.77. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- Table 2, percent change in error when non-causal entities are removed: SPARTAN 24.5 / 28.6 / 13.9; Transformer 1140.2 / 138.2 / 32.8; Global Graph 22.2 / 73.6 / not run. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- With learned slots on Pong the sparse model predicts slightly worse than the dense one: 7.21 +/- 0.69 vs 6.90 +/- 0.83, with SHD 2.69 vs 8.52. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- The paper concludes "attention mechanisms alone are not sufficient for discovering causal relationships": thresholded soft attention may "catch spurious edges or miss necessary edges" (e.g. a spurious Score -> Ball edge in Pong). — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)
- Stated limitations: needs pre-disentangled object tokens (ground-truth masks used), no identifiability theory for varying scene composition, sensitivity to lambda (high lambda causes mode collapse). There is no oracle-graph model; true graphs are used only for evaluation. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)

**Multiplicative reweighting by a hand-specified matrix (STICA)**
- STICA (Nishimoto and Matsubara, arXiv 2511.14262, Nov 2025, preprint): Slot Attention gives 5 object tokens; object, action and reward tokens feed a Transformer-XL dynamics model (10 layers, 4 heads, embedding 256) with temporal causal masking; positional encoding depends on time only. — [arXiv 2511.14262](https://arxiv.org/html/2511.14262v1)
- A hand-specified 3x3 matrix G over {policy/value target, causal objects, non-causal objects} and learned per-object causality scores W_t reweight attention as Norm(softmax(QK^T/sqrt(d)) ⊙ W_t G W_t^T) V with row-wise L1 renormalisation; this is applied only in the policy and value networks, not in the world model. — [arXiv 2511.14262](https://arxiv.org/html/2511.14262v1)
- Safety Gym, 8 tasks at 100k steps, mean return: STICA 11.90, DreamerV3 10.20, TWM 8.39, TD-MPC2 3.73, PPO 3.15. The causal-attention ablation is reported only in a figure and described as the largest single effect. — [arXiv 2511.14262](https://arxiv.org/html/2511.14262v1)

**JEPA-family object-centric models named in the brief — no given graph**
- Causal-JEPA / C-JEPA (Nam, Le Lidec, Maes, LeCun, Balestriero; arXiv 2602.11389, Feb 2026; arXiv comments say accepted to ICML 2026): predictor is a ViT-style masked transformer with bidirectional attention over entity tokens (object slots plus separate action and proprioception tokens); no positional encoding along the entity axis, temporal positional encoding only. — [arXiv 2602.11389 abs](https://arxiv.org/abs/2602.11389); [arXiv 2602.11389 HTML](https://arxiv.org/html/2602.11389)
- C-JEPA assumes, parameterises and estimates no causal graph; "causal" means temporally directed predictive dependence, and attention is interpreted as soft local relational structure. Its "influence neighbourhood" is a minimal predictively sufficient context set, explicitly not the causal parents. — [arXiv 2602.11389 HTML](https://arxiv.org/html/2602.11389)
- The structure enters through object-level masking: selected objects' latents are masked across the history window, keeping the earliest step as an identity anchor; future tokens are always masked. — [arXiv 2602.11389 HTML](https://arxiv.org/html/2602.11389)
- CLEVRER (VideoSAUR encoder), average / counterfactual accuracy per question: OC-JEPA without history masking 82.79 / 47.68; C-JEPA with 4 masked objects 89.40 / 68.81. With a SAVi encoder masking 4 objects degrades to 73.28 / 34.06 (best is 2 masked: 83.88 / 60.19). SlotFormer 79.44 / 47.29; OCVP-Seq 83.11 / 56.06. — [arXiv 2602.11389 HTML](https://arxiv.org/html/2602.11389)
- Push-T MPC success: DINO-WM (196 patch tokens) 91.33%, OC-JEPA 76.00%, C-JEPA 88.67% with about 1% of the latent tokens; planning 673 s vs 5,763 s. — [arXiv 2602.11389 HTML](https://arxiv.org/html/2602.11389)
- IA-JEPA (Paidi, "Entity-Centric World Models: Interaction-Aware Masking for Causal Video Prediction", arXiv 2605.15466, May 2026, single-author preprint): a motion-centric masking strategy targeting entities in collisions or momentum exchange; the abstract mentions no graph. Reported CLEVRER causal reasoning accuracy 14.26% vs 3.22% for patch masking. — [arXiv 2605.15466](https://arxiv.org/abs/2605.15466)
- HCLSM (Jaber and Jaber, arXiv 2603.29090, Mar 2026, preprint): slot attention, a three-level temporal engine and "causal structure learning through graph neural network interaction patterns"; the graph is LEARNED. 68M parameters, PushT next-state MSE 0.008. — [arXiv 2603.29090](https://arxiv.org/abs/2603.29090)

### Inferences
- A hard mask from a known graph on a transformer predictor is a well-defined, already-published construction, but only outside world models (causal effect estimation on tabular data). In the world-model literature read here, sparse attention is always learned or absent.
- CaT's cross-attention layout (queries are output slots, keys/values are always the raw embedded inputs, re-fed at each layer) is the transformer analogue of a bipartite t -> t+1 graph: it avoids multi-hop leakage by never letting output tokens attend to each other's processed states. A standard self-attention stack with the same mask in every layer does not have this property (SPARTAN's path matrix).
- SPARTAN's evidence says sparsity bought structure recovery and robustness to removed entities, not lower in-distribution prediction error (8.60 vs 8.83, 0.52 vs 0.51, and worse with learned slots). The same should be expected of a given-graph mask.
- An additive Graphormer-style bias leaves every edge usable; nothing in the sources read compares hard masks against soft biases under a fixed graph.

### Gaps
- Not fetched, so unverified here (background knowledge only): SlotFormer (arXiv 2210.05861, ICLR 2023) and OCVP (arXiv 2302.11850) use learned attention over slots, OCVP with separate object and temporal attention blocks; TWM, IRIS, STORM and DIAMOND are unstructured token/latent or diffusion world models with no per-variable graph variant that I know of; Graphormer (arXiv 2106.05234, NeurIPS 2021) adds learnable shortest-path-distance and edge biases to attention scores. Only the C-JEPA baseline numbers for SlotFormer and OCVP above are sourced.
- I found no world-model paper that applies a Graphormer-style additive bias derived from a known causal graph.
- The SPARTAN venue is not stated on the arXiv page read.
- IA-JEPA and HCLSM were read at abstract level only; predictor internals are unverified. Both are non-peer-reviewed preprints with small-scale evidence, and should be weighted accordingly.
- The DAG-aware Transformer was read only through a search summary; its experiments and numbers were not extracted.

---

## 4. 2024-2026 work that injects a known DAG or expert graph into a JEPA, Dreamer/RSSM or other latent world model

### Takeaway
I found no paper that hard-wires a full expert variable-level DAG into a JEPA predictor or the DreamerV3 RSSM. The closest items are coarse, hand-specified block structures (an exogenous/agent split of the RSSM state; a 3x3 relation matrix in STICA's policy) and a July 2026 technical report that masks a state-space transition network with ground-truth, discovered and corrupted graphs.

### Cited Findings
- "Spending the Dynamics Budget with Causally Factorized World Models" (Zu, Gao, Zhang; NeurIPS 2026 Workshop on Resource-Aware Agentic AI, poster): a causally factorised RSSM that splits the DreamerV3 deterministic state into an exogenous branch and an agent branch, the exogenous branch receiving neither actions nor agent-branch state. Claims: the recurrent core is 53.4% of DreamerV3 agent parameters, the factorised core has 17.6% fewer dynamics parameters at the same width, and it outperforms the monolithic baseline on Atari100k and Procgen (no scores on the page). — [NeurIPS 2026 virtual page](https://neurips.cc/virtual/2026/170698)
- STICA: hand-specified 3x3 relation matrix used to reweight attention in policy and value transformers on top of an object-centric transformer world model (see section 3). — [arXiv 2511.14262](https://arxiv.org/html/2511.14262v1)
- ICWM, "Learning Implicit Causal World Models from Multi-Agent Demonstrations" (Ghosh, Purdue; arXiv 2607.26336, 28 July 2026, single-author technical report, preprint): a neural transition network on low-dimensional state/action/reward history with a PCMCI+ derived input mask, compared against a graph-free baseline (AWM) and ablated with ground-truth and corrupted masks (details in section 5). — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- Search-level leads only (snippets, not opened): a "CsDreamer" built on a collider-structure RSSM for denoising distractors (NeurIPS 2025, "Leveraging Conditional Dependence for Efficient World Model Denoising"); a Dreamer-based model that sorts latents into four causal categories by their links to action and reward; a "CausalDreamer" that re-encodes a frozen Dreamer 4 tokenizer latent into four groups with actions routed only to controllable groups. — [NeurIPS 2025 proceedings page](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6f01b2a921c0a4dc34609613f4f74b8e-Abstract-Conference.html)
- C-JEPA explicitly avoids graph estimation and notes its influence neighbourhoods were not validated on datasets with explicit temporal causal graphs. — [arXiv 2602.11389 HTML](https://arxiv.org/html/2602.11389)

### Inferences
- Published graph priors in Dreamer-style models are block-level (which latent group sees the action or the reward), not variable-level adjacency. A 28-edge expert graph wired into a JEPA predictor or RSSM prior appears to be unoccupied ground as of November 2026, to the extent these searches cover it.
- A structural obstacle explains part of the gap: a known graph is defined over named physical variables, while RSSM and JEPA latents are entangled. Every method that uses a mask either works on ground-truth low-dimensional state (CDL, ICWM) or first imposes a one-token-per-entity latent (SPARTAN, C-JEPA, STICA).

### Gaps
- The search for this question was shallow (two queries). Absence of a variable-level known-graph JEPA/RSSM paper is a negative result from limited search, not a proof of absence.
- The CsDreamer, four-category Dreamer and CausalDreamer items are from search snippets only; authors, exact titles, venues and numbers are unverified. The search tool itself flagged date inconsistencies among those results.
- Whether the exogenous/agent split in "Spending the Dynamics Budget" is specified by hand or learned is not stated on the page read.
- No "physics-informed world model graph prior" paper was found or checked.

---

## 5. What happens when the given graph is wrong (missing or extra edges)?

### Takeaway
Direct evidence is thin and comes mostly from one non-peer-reviewed 2026 technical report: a corrupted mask inflated error by 1.16x to 17.9x and, per the search summary of that report, did worse than no mask at all in every environment group. Indirect evidence from NRI shows a graph with about 18% wrong edges can still match the true graph at short horizon when the graph is dense and edges carry little on/off information.

### Cited Findings
- ICWM (preprint, 2026): error inflation from corrupting the mask, 50 paired runs each, paired Wilcoxon tests: Two-Door full observation 1.79x, Two-Door partial observation 1.32x, Navigation full observation 1.47x, Navigation partial observation 17.90x, Giveway full observation 1.16x (not significant), Giveway noisy observation 1.27x. Values were read from figure labels by the fetch tool and should be checked against the PDF. — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- The same report trains one model under ground-truth, discovered (PCMCI) and corrupted masks plus a dense no-prior baseline; the corrupted mask underperformed the dense baseline in all six environment groups, with the largest harm in the interventional regime; where discovery recovered nearly the full graph, ground-truth and discovered masks performed the same. This sentence comes from a search-result summary of the PDF, which I could not open directly (file too large). — [arXiv 2607.26336 PDF](https://arxiv.org/pdf/2607.26336)
- Where the discovered mask is nearly dense (Giveway, full observability) "there is little structure for corruption to damage", which the authors give as the reason for the marginal effect there. — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- A correct mask can also hurt: under full observation the same mask that helped under partial observation hurt, "because gating inputs removes signal the model could have used"; a C-VAE given the same soft-gated graph was 2.66x worse than the graph-free baseline there. — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- Receiving a graph gave no significant improvement in the adversarial OOD comparison against the graph-free model: FOCUS x0.99, BC-WM x1.03, ICWM x1.00 (all n.s.), R-ICWM x1.52 and C-VAE x5.98 (both significantly worse). — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- NRI: on charged particles a learned graph with about 82% edge accuracy gives 1-step MSE 1.05e-3 against 1.04e-3 for the true graph; the gap widens with horizon (7.06e-3 vs 5.71e-3 at 20 steps). On Kuramoto (96.0% edge accuracy) learned vs true is 3.26e-2 vs 2.19e-2 at 20 steps. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- Extra edges (the fully connected limit): NRI full graph vs true graph is 1.66e-5 vs 1.69e-11 on springs at 1 step, but 1.09e-3 vs 1.04e-3 on charged particles. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- CDL with a learned mask misses infrequent dependencies and adds one or two false-positive edges per run in manipulation; a regularisation-learned mask is worse than CDL both in and out of distribution. — [ar5iv 2206.13452](https://ar5iv.labs.arxiv.org/html/2206.13452)
- Hard-masked DAG transformers assume the DAG is correct; CaT ties its robustness to the coherence between DAG and true data-generating process and reports no misspecification ablation in the text read. — [arXiv 2410.14485v4](https://arxiv.org/html/2410.14485v4)
- Soft attention as a graph proxy yields both spurious and missing edges under thresholding. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)

### Inferences
- The two error types are asymmetric under a hard mask. A missing true edge removes information the predictor cannot recover (irreducible bias, compounding over rollouts). An extra edge only returns the model toward the dense baseline, costing sample efficiency and OOD robustness but not in-distribution fit. The NRI full-graph rows and the ICWM "corrupted worse than dense" result are both consistent with this, though neither isolates missing from extra edges.
- For a study comparing an expert graph against a recovered graph, the informative ablation is therefore a four-way one (true/expert, recovered, corrupted, dense) with missing-edge and extra-edge corruption reported separately; no source read does the latter.
- Short-horizon prediction error is a weak detector of a wrong graph when the graph is dense (charged particles); rollout length and interventional or OOD evaluation are where the differences show.

### Gaps
- The ICWM corruption procedure (how many edges are flipped, whether removed or added) and its four-way table are in Appendix V, which I could not read. Treat every ICWM number as provisional: single-author technical report, not peer reviewed, extracted from truncated text.
- I found no ablation with systematically corrupted graphs in IN, GN-physics, C-SWM, SPARTAN, C-JEPA or CDL.
- CaT's results table did not render; whether a wrong-DAG experiment exists in its appendix is unknown.
- No source reports the effect of a wrong graph on downstream control or planning return.

---

## 6. Does a graph prior help when the true graph is dense, and is there evidence on small low-dimensional control systems?

### Takeaway
The available evidence points to little or no in-distribution prediction benefit when the graph is dense or the system is small: GN vs MLP was negligible on Pendulum, CDL's authors say causal learning offers no advantage where the ground-truth graph is effectively dense, and a 2025 AAAI paper measuring simulator Jacobians found zero globally-zero state-Jacobian entries on DMC CartpoleBalance. Reported benefits of structure concentrate in OOD states, partial observation, distractor robustness and long rollouts.

### Cited Findings
- "Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks" (Pandaram et al., arXiv 2511.08086, Nov 2025, accepted at AAAI-26 per the page): sparsity is measured by zero entries of the state and action Jacobians of the MuJoCo (MJX) step function on BallInCup, CartpoleBalance, CheetahRun, ReacherHard, FingerSpin, FingerTurnEasy, WalkerRun and Swimmer6 (state dimension 4 to 18, action dimension 1 to 6), with about 11 million PPO-expert steps per environment. — [arXiv 2511.08086](https://arxiv.org/html/2511.08086v1)
- Globally zero state-Jacobian entries were rare: CheetahRun 17 (5.25%), ReacherHard 3 (18.75%), WalkerRun 17 (5.25%), Swimmer6 30 (11.72%); the other four tasks, CartpoleBalance among them, had none. No action-Jacobian entry was globally zero in any task. The counts are upper bounds on true global sparsity. — [arXiv 2511.08086](https://arxiv.org/html/2511.08086v1)
- The paper concludes strong global sparsity priors may not be broadly beneficial and could block important interactions; the sparsity that exists is state- and time-dependent (contact, string slack). It does not test whether a prior improves model-based RL. — [arXiv 2511.08086](https://arxiv.org/html/2511.08086v1)
- Supervising a 2-layer 512-unit MLP with ground-truth Jacobian losses barely changed test prediction error (differences not significant), while an L1 penalty on predicted Jacobians raised sparsity but reduced accuracy. — [arXiv 2511.08086](https://arxiv.org/html/2511.08086v1)
- Sanchez-Gonzalez 2018 include Pendulum, Cartpole and Acrobot; the GN beat the MLP on 6 of 7 systems, with a negligible difference on Pendulum, attributed to little repeated structure. — [ar5iv 1806.01242](https://ar5iv.labs.arxiv.org/html/1806.01242)
- CDL: in the "physical" 5x5 grid environment the ground-truth graph is effectively dense and the authors note causal learning offers no advantage there; on the chemical "full" graph (most edges, long dependencies) CDL makes more prediction errors than on collider and chain graphs. — [ar5iv 2206.13452](https://ar5iv.labs.arxiv.org/html/2206.13452)
- CDL: in-distribution accuracy is similar across masked, modular, GNN and monolithic models; the difference appears on OOD states, where dense models degrade severely (some by "60∼90%"). On chemical collider, CDL 97.3% in and out of distribution versus NPS 88.0% ID and 35.7% OOD. — [ar5iv 2206.13452](https://ar5iv.labs.arxiv.org/html/2206.13452)
- NRI dense case: charged particles, where all pairs interact, show full-graph vs true-graph MSE of 1.09e-3 vs 1.04e-3 at 1 step and 9.24e-3 vs 5.71e-3 at 20 steps. — [ar5iv 1802.04687](https://ar5iv.labs.arxiv.org/html/1802.04687)
- ICWM: a discovered two-door input gate keeps 21 to 22 of 29 inputs open (mean gate weight about 0.74 to 0.76), described as "essentially a dense model in disguise"; observational discovery in dense settings yields fully connected graphs. — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- ICWM: the mask helps mainly under partial observation (0.53x the graph-free error on the stretched two-door layout) and a graph-free model matches the masked model at matched data; the pooled correlation between graph quality and OOD accuracy (about +0.62) falls to about 0 after conditioning on intervention strength. — [alphaXiv 2607.26336](https://www.alphaxiv.org/abs/2607.26336.md)
- SPARTAN: sparse attention gave equal or slightly worse prediction error than a dense transformer in some settings (Traffic 0.52 vs 0.51; learned-slot Pong 7.21 vs 6.90) while improving graph recovery and robustness to removed entities. — [arXiv 2411.06890v3](https://arxiv.org/html/2411.06890v3)

### Inferences
- For cart-pole, where most of the 7 inputs are parents of most of the 4 state-change outputs, a hard mask from the true graph removes few of the 28 edges, so the hypothesis class is nearly that of the dense predictor. The sources predict a near-null effect on in-distribution one-step error and locate any benefit, if present, in OOD states, distractor variables, low-data regimes or long rollouts.
- The edges a dense physical graph does exclude tend to be the scientifically interesting ones (e.g. position not affecting accelerations, or a distractor variable having no children). An evaluation built around exactly those excluded parents (perturb a non-parent and check the prediction is invariant, as in SPARTAN's Table 2) is where a graph prior can show a measurable effect even when MSE does not move.
- The Dynamic Sparsity result is for DMC CartpoleBalance with Jacobians of the full one-step map in generalised coordinates. Whether it transfers to a specific 7-input/4-output variable set depends on the parameterisation (for instance cos/sin angle features and state-change targets), so the expert graph for that specific parameterisation should be checked against simulator Jacobians directly.
- State-dependent sparsity (contacts) is largely absent in cart-pole and Acrobot, so the "local causal graph" line of work (SPARTAN and its precursors) has little to exploit there.

### Gaps
- No paper found tests a known-graph mask on cart-pole or Acrobot dynamics prediction with one node per scalar variable. Sanchez-Gonzalez 2018 is the nearest, but uses body/joint graphs and reports the Cartpole and Acrobot results only in figures.
- No source quantifies sample-efficiency gains from a correct dense graph in low-dimensional control.
- No evidence was found on whether a graph prior in the predictor changes what a probe recovers from a JEPA or RSSM latent (as opposed to prediction error); this appears untested.
- Evidence for this question rests on one AAAI paper, one ICML paper's remarks, and one unreviewed technical report; it is suggestive, not conclusive.
