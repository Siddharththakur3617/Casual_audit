# Statistical methods for testing edges and variable importance with false-positive control, and their use on learned models

Scope: the statistics behind the study's edge-scoring and pruning procedure. The procedure uses LOCO with small MLPs trained as one batched network from shared initial weights; shuffled-input decoys; 50 episode-level half-splits; a normalised Jacobian; a group-lasso neural-Granger cross-check; and candidate pruning rules (stability margin vs. the strongest decoy, per-split z-test with 2×median aggregation and BH, a minimum-relevant-effect test, a Jacobian threshold, and derandomised knockoff+). The chosen rule is a null-calibrated max-T: 20 row-shuffled-target repeats, Westfall–Young FWER, and p = (1+#)/(K+1).

World-model probing, causal-dynamics MBRL and causal representation learning are out of scope here.

Verification status: items marked (verified) were checked in this session through search results or abstract pages. Items marked (from prior knowledge) are standard references whose venue and ID I am confident of but did not re-open this session. The report writer may want to spot-check those.

---

## Q1. LOCO and its inference; correlated and collinear features; recommended remedies

### Takeaway
LOCO has a well-developed inference theory: Lei et al. 2018; Rinaldo, Wasserman & G'Sell 2019; Williamson et al. 2021 and 2023. That theory supports the study's use of held-out loss differences and sample splitting.

The literature agrees that LOCO is near zero for inputs that the other inputs can predict well. Verdinelli & Wasserman call this "sensible": the importance is not identifiable. The standard remedies are:
- group LOCO (test collinear blocks jointly);
- decorrelated LOCO;
- conditional, knockoff-based or model-X tests (CPI, HRT, CRT);
- Shapley-type sharing (SAGE, Shapley population VIM).

The study's VIF-scaled minimum effect is, in the linear case, close to Verdinelli & Wasserman's decorrelated LOCO. This is my inference, derived below.

### Cited Findings
**LOCO: origin and inference**
- LOCO was introduced as a by-product of distribution-free (split-conformal) predictive inference. The importance of variable j is the increase in held-out prediction error when j is removed and the model is refit. Inference uses sample splitting. Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, "Distribution-Free Predictive Inference for Regression", JASA 113 (2018) 1094–1111 (from prior knowledge) — [arXiv:1604.04173](https://arxiv.org/abs/1604.04173)
- Rinaldo, Wasserman & G'Sell give assumption-lean inference for LOCO-type parameters after data-driven model selection. They use sample splitting plus the bootstrap and allow the number of covariates to grow. Annals of Statistics 47(6) (2019) (from prior knowledge) — [arXiv:1611.05401](https://arxiv.org/abs/1611.05401)
- Williamson, Gilbert, Carone & Simon, "Nonparametric variable importance assessment using machine learning techniques", Biometrics 77(1) (2021) 9–22 (from prior knowledge). The parameter is a population R² difference between the full and reduced conditional mean, estimated with flexible learners. They note that the plug-in estimator's influence function degenerates at the null of zero importance. Valid tests therefore need sample splitting, with full and reduced models estimated on separate halves — [DOI 10.1111/biom.13392](https://doi.org/10.1111/biom.13392)
- Williamson, Gilbert, Simon & Carone, "A General Framework for Inference on Algorithm-Agnostic Variable Importance", JASA 118(543) (2023) 1645–1658 (verified). Importance is defined as a contrast between the oracle predictiveness of all features and of all features except the tested set. Any predictiveness measure can be used (R², deviance, AUC, etc.). The paper gives efficient estimators with valid CIs when ML is used, a test of zero importance, and the R package `vimp` — [arXiv:2004.03683](https://arxiv.org/abs/2004.03683); [PMC10652709](https://pmc.ncbi.nlm.nih.gov/articles/PMC10652709); [vimp](https://github.com/bdwilliamson/vimp)
- Shapley-based population importance with inference: Williamson & Feng, "Efficient nonparametric statistical inference on population feature importance using Shapley values", ICML 2020 (from prior knowledge) — [arXiv:2006.09481](https://arxiv.org/abs/2006.09481)
- Dai, Shen & Pan, "Significance Tests of Feature Relevance for a Black-Box Learner", IEEE TNNLS (2022) (verified). This is a LOCO-like test of whether a feature set matters to a deep network. There are two versions:
  - a one-split test (sample splitting plus data perturbation);
  - a two-split test.

  Both have combined versions that aggregate p-values over repeated sample splits. Repeated splitting is more stable but loses some power. The tests work on groups of features, and the Python package is `dnn-inference` — [arXiv:2103.04985](https://arxiv.org/abs/2103.04985); [PMC10915654](https://pmc.ncbi.nlm.nih.gov/articles/PMC10915654)
- Computational shortcut for reduced models in neural networks (verified): "LazyVI" approximates the reduced, feature-dropped network by linearising around the full network's parameters with a ridge penalty, instead of retraining. It gives an asymptotically normal VI estimate with CIs, and compares against dropout-based reduced models. Gao, Stevens, Raskutti & Willett, ICML 2022 — [PMLR v162](https://proceedings.mlr.press/v162/gao22h.html); [arXiv:2207.09097](https://arxiv.org/abs/2207.09097)

**Correlated and collinear features**
- Verdinelli & Wasserman, "Decorrelated Variable Importance", JMLR 25 (2024) 1–27 (verified). LOCO is a nonparametric partial R² and is distorted by correlation. They define decorrelated LOCO as "what LOCO would be if W and Z were independent": the squared difference between the full and reduced regression functions, integrated against the product of marginals and scaled by Var(W). It is hard to estimate nonparametrically, so semiparametric estimators are used — [JMLR](https://jmlr.org/papers/v25/22-0801.html); [arXiv:2111.10853](https://arxiv.org/abs/2111.10853)
- Verdinelli & Wasserman, "Feature Importance: A Closer Look at Shapley Values and LOCO", Statistical Science (2024) (verified preprint; venue from prior knowledge). Correcting for correlation is a trade-off, because removing its effect introduces other biases. They recommend a slightly modified LOCO. When W is perfectly correlated with other features its importance is not estimable, which the authors regard as sensible — [arXiv:2303.05981](https://arxiv.org/abs/2303.05981)
- Hooker, Mentch & Zhou, "Unrestricted permutation forces extrapolation", Statistics and Computing 31:82 (2021) (verified). With correlated features, permute-and-predict importance breaks dependencies and evaluates the model off the data manifold, over-weighting sparse regions. They recommend refit-based alternatives such as LOCO/drop-and-refit or conditional permutation. This supports the study's use of LOCO with refitting over plain permutation importance — [arXiv:1905.03151](https://arxiv.org/abs/1905.03151)
- Chamma, Engemann & Thirion, "Statistically Valid Variable Importance Assessment through Conditional Permutations", NeurIPS 2023 (verified):
  - standard permutation importance mislabels unimportant correlated variables;
  - Conditional Permutation Importance (CPI) controls Type-I error, including with deep networks;
  - a follow-up by the same group extends the approach with grouping and stacking and controls Type-I error with highly correlated groups.

  [arXiv:2309.07593](https://arxiv.org/abs/2309.07593); [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2023/file/d60e14c19cd6e0fc38556ad29ac8fbc9-Paper-Conference.pdf)
- Watson & Wright, "Testing Conditional Independence in Supervised Learning Algorithms", Machine Learning (2021) (verified). Conditional Predictive Impact (CPI) replaces the tested features with model-X knockoffs, which act as negative controls. It compares held-out loss with a paired t-test or Fisher exact test, for any learner and loss, and was tested with neural nets among others. Type-I error is controlled. R package `cpi` — [arXiv:1901.09917](https://arxiv.org/abs/1901.09917)
- Molnar, König, Bischl & Casalicchio, "Model-agnostic feature importance and effects with dependent features: a conditional subgroup approach", Data Mining and Knowledge Discovery (2023) (from prior knowledge). It permutes within subgroups defined by trees on the other features to approximate conditional permutation — [arXiv:2006.04628](https://arxiv.org/abs/2006.04628)
- Strobl et al., "Conditional variable importance for random forests", BMC Bioinformatics 9:307 (2008) is the origin of conditional permutation importance (from prior knowledge) — [DOI](https://doi.org/10.1186/1471-2105-9-307). Strobl et al., BMC Bioinformatics 8:25 (2007) documents bias in RF importance (from prior knowledge) — [DOI](https://doi.org/10.1186/1471-2105-8-25)
- Floodgate (Zhang & Janson, arXiv 2020, v5 2022) (verified). The target is the "mMSE gap", a model-free measure of how much Y depends on X given Z. Floodgate turns any fitted regression, including ML, into asymptotically valid lower confidence bounds on that gap. It is used on groups of variables, and Wang, Janson, Lei & Ramdas (ICML 2023) extend it to classification — [arXiv:2007.01283](https://arxiv.org/abs/2007.01283)
- SAGE (Covert, Lundberg & Lee, NeurIPS 2020) is Shapley-based global importance that shares credit among correlated features (from prior knowledge) — [arXiv:2004.00668](https://arxiv.org/abs/2004.00668)
- "True to the model or true to the data?" (Chen, Janizek, Lundberg & Lee, 2020) separates conditional (data) from interventional/marginal (model) attributions. This is relevant because the study's targets are a trained model's outputs (from prior knowledge) — [arXiv:2006.16234](https://arxiv.org/abs/2006.16234)
- There is a theoretical floor on conditional-independence testing. Shah & Peters, "The hardness of conditional independence testing and the generalised covariance measure", Annals of Statistics (2020) show that no CI test can have power against all alternatives while controlling size uniformly. Any valid edge test must therefore rely on assumptions, for example on the quality of the regressions (from prior knowledge) — [arXiv:1804.07203](https://arxiv.org/abs/1804.07203)

### Inferences
- **VIF-scaled minimum effect versus decorrelated LOCO.** In a linear model y = βⱼxⱼ + β₋ⱼᵀx₋ⱼ + ε, population LOCO (the unnormalised MSE increase) equals βⱼ²·Var(xⱼ | x₋ⱼ) = βⱼ²·Var(xⱼ)/VIFⱼ. Multiplying LOCO by VIFⱼ recovers βⱼ²·Var(xⱼ), the importance the variable would have if it were uncorrelated with the others. That is exactly what Verdinelli & Wasserman's decorrelated LOCO targets in the linear/additive case.

  The study's VIF-scaled sensitivity analysis can therefore be presented as a linear approximation to decorrelated LOCO, citing JMLR 2024. Caveat: the study uses log(MSE ratio) and nonlinear MLPs, so the correspondence holds only approximately. Verdinelli & Wasserman 2024 also argue that decorrelation adds its own biases.
- **For the collinear inputs (95% predictable from the others):**
  - The principled statistical answer is that the separate importance of each such input is not identifiable from observational data. Verdinelli & Wasserman say this explicitly.
  - Best practice is to report group LOCO for the collinear block, plus the individual edges marked "not separately identifiable". Group tests are available in Williamson 2023, Dai–Shen–Pan, Chamma et al. and floodgate.
  - Inflating the small individual LOCO values would not fix the identifiability problem.
  - In a simulator the variables can be intervened on, so the cleanest fix is interventional data that breaks the collinearity (randomised actions and states). This is my inference, not from a source.
- **Batched training from shared initial weights.** Training full and reduced MLPs from shared initial weights is a "common random numbers" variance reduction. It is close in spirit to LazyVI, which warm-starts or linearises the reduced model from the full model. The variance of the LOCO difference should fall, but the plug-in LOCO is still degenerate at the null.
  - Williamson et al. handle this by estimating the full and reduced predictiveness on different halves.
  - The study's per-split z-test may have badly calibrated variance at the null for this reason.
  - The null-calibrated permutation rule sidesteps the asymptotics, which is an argument in its favour.
- Plain permutation importance (Breiman-style) would be the wrong choice given the collinearity (Hooker et al. 2021; Chamma et al. 2023). The study's drop-and-refit LOCO is the recommended form.

### Gaps
- I did not find a source that defines a "VIF-scaled minimum relevant effect" for LOCO. The link to decorrelated LOCO is my derivation.
- I did not re-open Williamson et al. 2021 to quote the degeneracy discussion. It is from prior knowledge of the paper.
- Theory for LOCO when the target is a nearly deterministic function of the inputs (a trained model's output) is not treated in any source found. The noise term in the R²/MSE ratio is then approximation error, not irreducible noise.

---

## Q2. Error-control machinery: stability selection, p-value aggregation over splits, knockoffs, max-T permutation, exact permutation p-values, and decoys or pseudovariables

### Takeaway
Every component of the candidate rules has a primary reference with known guarantees. The guarantees are conditional, though:
- stability selection bounds expected false selections under exchangeability or r-concavity assumptions;
- quantile aggregation of split p-values (2×median) is valid under any dependence across splits;
- BH needs PRDS (otherwise use Benjamini–Yekutieli);
- knockoffs need a correct model for X and lose power with strongly dependent features;
- max-T over permutations controls FWER strongly only under subset pivotality.

The last point is the main soundness issue for the chosen rule. Permuting the target (row-shuffling) creates the complete null. Strong FWER control for individual edges, when other edges are real, needs subset pivotality. LOCO statistics from refit MLPs plausibly violate it.

### Cited Findings
**Stability selection and split aggregation**
- Stability selection (Meinshausen & Bühlmann, JRSS-B 72(4) (2010) 417–473) runs a selector on many subsamples and keeps variables with high selection frequency. It gives a bound on the expected number of false selections under an exchangeability assumption and a "better than random guessing" assumption (from prior knowledge) — [arXiv:0809.2932](https://arxiv.org/abs/0809.2932)
- Complementary pairs stability selection (Shah & Samworth, JRSS-B 75(1) (2013) 55–80) (verified) draws subsamples as complementary pairs. It bounds the number of low-selection-probability variables selected without the exchangeability assumption. The bounds sharpen under unimodality or r-concavity of the selection-proportion distribution. R package `stabs` — [arXiv:1105.5578](https://arxiv.org/abs/1105.5578)
- Meinshausen, Meier & Bühlmann, "p-Values for High-Dimensional Regression", JASA 104 (2009) 1671–1681 (from prior knowledge):
  - run repeated random sample splits, selecting on one half and testing on the other;
  - aggregate p-values with the γ-quantile rule Q(γ) = min{1, q_γ(p/γ)};
  - γ = 0.5 gives the "2 × median" rule the study uses;
  - FWER/FDR control holds under arbitrary dependence among the split p-values.

  [arXiv:0811.2177](https://arxiv.org/abs/0811.2177)
- Benjamini & Hochberg, JRSS-B 57 (1995) 289–300 (from prior knowledge) — [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). For arbitrary dependence the conservative version is Benjamini & Yekutieli, Annals of Statistics 29 (2001) (from prior knowledge) — [DOI](https://doi.org/10.1214/aos/1013699998)
- When the same data are resampled many times, the naive per-split variance underestimates the true variance of the CV estimate, and corrected resampled t-tests are needed. Nadeau & Bengio, "Inference for the Generalization Error", Machine Learning 52 (2003) 239–281 (from prior knowledge) — [DOI](https://doi.org/10.1023/A:1024068626366)

**Knockoffs**
- Fixed-X knockoffs with FDR control in finite samples, including the knockoff+ threshold: Barber & Candès, Annals of Statistics 43(5) (2015) 2055–2085 (from prior knowledge) — [arXiv:1404.5609](https://arxiv.org/abs/1404.5609)
- Model-X knockoffs, valid for any conditional model of Y|X when the distribution of X is known: Candès, Fan, Janson & Lv, JRSS-B 80(3) (2018) 551–577 (from prior knowledge) — [arXiv:1610.02351](https://arxiv.org/abs/1610.02351)
- Derandomizing knockoffs (Ren, Wei & Candès, JASA 118(542) (2023) 948–958) (verified) aggregates selections across many knockoff draws. This removes the randomness of a single draw while keeping Type-I error control (a PFER/k-FWER-type guarantee) — [arXiv:2012.02717](https://arxiv.org/abs/2012.02717)
- Ren & Barber, "Derandomised knockoffs: leveraging e-values for FDR control" (JRSS-B 2024; venue from prior knowledge) gives an FDR-controlling derandomisation (verified preprint) — [arXiv:2205.15461](https://arxiv.org/abs/2205.15461)
- Knockoffs with dependent features (from prior knowledge):
  - **The problem.** Under strong correlation, SDP/equicorrelated knockoffs end up almost identical to the originals, so power is low.
  - **Minimum-reconstructability knockoffs.** Spector & Janson, "Powerful knockoffs via minimizing reconstructability", Annals of Statistics (2022) — [arXiv:2011.14625](https://arxiv.org/abs/2011.14625)
  - **Group knockoffs.** Dai & Barber, ICML 2016 — [arXiv:1602.03589](https://arxiv.org/abs/1602.03589)
  - **Deep generative knockoffs for non-Gaussian X.** Romano, Sesia & Candès, JASA (2020) — [arXiv:1811.06687](https://arxiv.org/abs/1811.06687)
- Randomisation tests related to knockoffs (from prior knowledge):
  - the conditional randomization test (in Candès et al. 2018);
  - the distilled CRT (Liu, Katsevich, Janson & Ramdas, Biometrika 2022) — [arXiv:2006.03980](https://arxiv.org/abs/2006.03980);
  - the Holdout Randomization Test (Tansey et al., JCGS 31 (2022)) (verified). The model is fit once, and conditionally resampled features are evaluated on held-out data to give valid per-feature p-values and FDR control for any black box. In the authors' simulations it has more power than a knockoffs approach — [arXiv:1811.00645](https://arxiv.org/abs/1811.00645)

**Max-T permutation and exact p-values**
- Westfall & Young, *Resampling-Based Multiple Testing*, Wiley (1993) is the origin of single-step and step-down max-T/min-P resampling for FWER (from prior knowledge; book).
- Strong FWER control requires subset pivotality: the joint distribution of statistics for the true nulls must not depend on whether the other hypotheses are true. Under that condition, resampling under the complete null gives strong FWER control. It holds in the "one treatment, many outcomes" setting and for per-marker score tests, but not automatically in regressions with nuisance covariates. Subset pivotality is sufficient, not necessary (verified, via search summaries of):
  - Meinshausen, Maathuis & Bühlmann, "Asymptotic optimality of the Westfall–Young permutation procedure for multiple testing under dependence", Annals of Statistics (2011) — [arXiv:1106.2068](https://arxiv.org/abs/1106.2068)
  - "Permutation in genetic association studies with covariates…" — [arXiv:1612.07010](https://arxiv.org/abs/1612.07010)
- Romano & Wolf, "Exact and approximate stepdown methods for multiple hypothesis testing", JASA 100 (2005) 94–108 (from prior knowledge). Step-down resampling with strong FWER control under weaker conditions, generating the null by resampling centred statistics rather than permuting the response — [DOI](https://doi.org/10.1198/016214504000000539)
- Winkler, Ridgway, Webster, Smith & Nichols, "Permutation inference for the general linear model", NeuroImage 92 (2014) 381–397 (verified). If a nuisance effect is present, the data are not exchangeable even under the null, so you cannot simply permute Y to test a partial effect. Freedman–Lane, which permutes residuals of the reduced model, performs well for error-rate control and power. The paper also covers exchangeability blocks for repeated measures or longitudinal data. This is the standard neuroimaging reference for permutation FWER with max-T — [PMC4010955](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4010955/)
- Nichols & Holmes, "Nonparametric permutation tests for functional neuroimaging: a primer", Human Brain Mapping 15 (2002) 1–25 is the max-statistic (max-T) FWER in neuroimaging (from prior knowledge) — [DOI](https://doi.org/10.1002/hbm.1058)
- Phipson & Smyth, "Permutation P-values Should Never Be Zero", Statistical Applications in Genetics and Molecular Biology 9(1) (2010) (from prior knowledge). With random permutations, p = (b+1)/(m+1) is exact, whereas b/m is anti-conservative. The study's formula matches — [arXiv:1603.05766](https://arxiv.org/abs/1603.05766)

**Decoys, pseudovariables and response-permutation nulls for importance (the closest precedents to the study's decoys and null repeats)**
- Wu, Boos & Stefanski, "Controlling Variable Selection by the Addition of Pseudovariables", JASA 102(477) (2007) 235–243 (verified):
  - add known-null pseudovariables, run selection, and monitor the fraction of pseudovariables selected;
  - use bootstrap-like replicates to estimate and control the false selection rate;
  - pseudovariables are defined as unrelated to both the response and the real predictors;
  - a later extension builds them by permuting rows of X.

  [IDEAS/RePEc](https://ideas.repec.org/a/bes/jnlasa/v102y2007p235-243.html)
- Boruta (Kursa & Rudnicki, Journal of Statistical Software 36(11) (2010)) adds shuffled "shadow" copies of every feature, compares each real feature with the maximum shadow importance over repeated runs, and tests the hit counts with a binomial test. This is very close to the study's "margin vs. strongest decoy over ≥70% of splits" rule (from prior knowledge) — [DOI](https://doi.org/10.18637/jss.v036.i11)
- PIMP (Altmann, Toloşi, Sander & Lengauer, "Permutation importance: a corrected feature importance measure", Bioinformatics 26 (2010)) (verified) repeatedly permutes the outcome vector to build a null distribution of each feature's importance and reports a p-value. This is the direct precedent for the study's row-shuffled-target null repeats — [PubMed 20385727](https://pubmed.ncbi.nlm.nih.gov/20385727/)
- Target–decoy FDR estimation in proteomics (Elias & Gygi, Nature Methods 4 (2007)) is the same negative-control logic from another field (from prior knowledge) — [DOI](https://doi.org/10.1038/nmeth1019)

### Inferences
- **The chosen rule (row-shuffled target, max-T, (1+b)/(K+1)) is PIMP-style response permutation combined with Westfall–Young max-T.** It gives exact level-α control of the global null ("the model's output depends on no input"), which is weak FWER control. For individual edges, strong FWER needs subset pivotality. Here the null distribution of a truly irrelevant input's LOCO, when other inputs carry strong signal, plausibly differs from its distribution when nothing carries signal. The reasons:
  - residual MSE is far smaller under the alternative;
  - log-ratio LOCO divides refit/optimisation noise by that small MSE;
  - collinearity with real causes changes how the reduced model refits.

  The procedure could therefore be anti-conservative or conservative for individual edges. The neuroimaging remedy is Freedman–Lane, which permutes residuals of the reduced model (Winkler et al. 2014). The model-X remedy is to resample the tested input conditionally (CRT/HRT/CPI) instead of permuting Y. This is my reasoning, grounded in the subset-pivotality literature.
- **Resolution floor.** With K = 20 null repeats, the smallest attainable p is 1/21 ≈ 0.048. At α = 0.05 an edge is kept only if its statistic beats all 20 null maxima. This gives no room for a stricter α and makes the decision very sensitive to the most extreme null repeat. It is valid (Phipson & Smyth) but low in resolution. Using K ≥ 99 or ≥ 199 is conventional in neuroimaging; K = 1000+ is common in Nichols & Holmes-style analyses (inference, not a direct quote).
- **Exchangeability with time-series episodes.** Row-shuffling targets across time steps within and across episodes assumes rows are exchangeable under the null. Autocorrelated inputs and outputs violate this; it is the spurious-regression problem. Winkler et al.'s exchangeability blocks, permuting whole episodes or using block or circular shifts, are the standard fix. The study clusters episodes for splitting, but the null repeats shuffle rows. (Inference.)
- **Decoys versus knockoffs.** Shuffled decoys are Wu–Boos–Stefanski pseudovariables or Boruta shadows. They are independent of everything, so they calibrate pure fitting noise. They do not mimic a real null input that is correlated with true causes; knockoffs or CRT draws do. A strongest-decoy margin is therefore a reasonable test for overfitting noise but does not control false edges caused by collinearity (inference, from Candès et al. 2018 and Watson & Wright 2021).
- **The 2×median + BH rule.** Quantile aggregation is valid under arbitrary dependence across splits, so the 2×median step is sound (Meinshausen et al. 2009). The remaining weak points are:
  - the per-split z-test variance, given LOCO's null degeneracy (Williamson) and resampling effects (Nadeau & Bengio);
  - BH's PRDS assumption across edges that share a target.

  Benjamini–Yekutieli is the safe alternative.
- **Derandomised knockoff+ on collinear dynamics inputs.** Low power is expected, as Spector & Janson describe. It is also only as valid as the knockoff model for the joint distribution of states and actions, which for cart-pole/acrobot states is non-Gaussian and deterministic-dynamics-induced (inference).

### Gaps
- I did not find a paper that analyses subset pivotality specifically for refit-based LOCO statistics under response permutation. The concern above is reasoned, not cited.
- I did not open the original Westfall & Young book; it is cited from prior knowledge.

---

## Q3. Use on trained neural networks, causal discovery, graph recovery and neural Granger causality with formal error control

### Takeaway
Error-controlled feature testing has been applied to trained neural networks many times:
- DeepPINK;
- HRT;
- CPI (Watson & Wright; Chamma et al.);
- Dai–Shen–Pan;
- LazyVI;
- knockoff-based DNN interaction detection (DeepROCK).

For time-series causal graphs there is:
- knockoff-counterfactual Granger with DeepAR (Ahmad et al. 2021);
- debiased-lasso FDR for linear VAR Granger networks (JASA 2025);
- doubly robust deep-learning Granger tests (2025);
- PCMCI-style CI testing.

Neural Granger causality itself (Tank et al.; GVAR; Jacobian Granger) generally selects edges with a penalty or a stability threshold, not a formal FWER/FDR guarantee. I found no prior work that applies LOCO + decoys + split repetition + max-T to a trained world model's predicted dynamics to extract a causal graph.

### Cited Findings
- **DeepPINK** (Lu, Fan, Lv & Noble, NeurIPS 2018) builds knockoffs into a deep network's architecture (a pairwise-coupling layer) for FDR-controlled feature selection (verified) — [arXiv:1809.01185](https://arxiv.org/abs/1809.01185)
- **DeepROCK** gives error-controlled (knockoff-based) detection of feature interactions in deep networks (verified) — [arXiv:2309.15319](https://arxiv.org/abs/2309.15319)
- **Knockoff-based FDR screening for deep networks** is a 2026 preprint (verified existence only) — [arXiv:2606.04404](https://arxiv.org/abs/2606.04404)
- **HRT** (Tansey et al., JCGS 2022) gives per-feature p-values for an already-trained black box, with case studies replacing heuristic importance rankings (verified) — [arXiv:1811.00645](https://arxiv.org/abs/1811.00645)
- **CPI** (Watson & Wright 2021) tests features of trained neural nets with knockoff negative controls (verified) — [arXiv:1901.09917](https://arxiv.org/abs/1901.09917). The conditional-permutation CPI of Chamma et al. (NeurIPS 2023) has Type-I error control with DNNs (verified) — [arXiv:2309.07593](https://arxiv.org/abs/2309.07593)
- **Dai–Shen–Pan** (TNNLS 2022) give LOCO-type tests for DNNs with p-values aggregated over repeated splits (verified) — [arXiv:2103.04985](https://arxiv.org/abs/2103.04985)
- **LazyVI** (ICML 2022) gives VI with CIs for large neural networks, applied to climate forecasting (verified) — [arXiv:2207.09097](https://arxiv.org/abs/2207.09097)
- **Neural Granger causality** (Tank, Covert, Foti, Shojaie & Fox, IEEE TPAMI 2021) puts group-sparsity penalties on the first-layer weights of MLP/LSTM per-target models. A series is Granger non-causal if all its outgoing first-layer weights are zero. There is no formal error control; edges come from the penalty path (verified via search) — [arXiv:1802.05842](https://arxiv.org/abs/1802.05842)
- **GVAR** (Marcinkevičs & Vogt, ICLR 2021) uses self-explaining networks for Granger causality. Edges are chosen by a stability criterion that compares fits on the original and time-reversed series. That is a stability heuristic, not FWER/FDR control (from prior knowledge) — [arXiv:2101.07600](https://arxiv.org/abs/2101.07600)
- **Jacobian Regularizer-based Neural Granger Causality** (JRNGC, ICML 2024) uses the input Jacobian as the importance measure for Granger edges, with a variable-selection procedure. It is the closest precedent for the study's normalised Jacobian effect size (verified) — [arXiv:2405.08779](https://arxiv.org/abs/2405.08779)
- **Knockoff counterfactuals for nonlinear Granger** (Ahmad, Shadaydeh & Denzler, 2021) uses knockoff samples as interventions, with DeepAR forecasters, to detect nonlinear Granger causality. The authors argue this limits false discoveries and compare FDR with PCMCI. The search results do not show a formal FDR guarantee (verified) — [arXiv:2109.10817](https://arxiv.org/abs/2109.10817). A related paper uses model invariance under knockoff interventions — [arXiv:2207.04055](https://arxiv.org/abs/2207.04055)
- **Network Granger causality in large VARs** (JASA 2025) gives FDR-controlling multiple-testing procedures (debiased lasso and a bootstrap version) for linear VAR Granger graphs (verified) — [T&F](https://www.tandfonline.com/doi/full/10.1080/01621459.2025.2450836)
- **A deep-learning doubly robust test for Granger causality** (2025) builds a neural-network-based test statistic with a parametric rate (verified, abstract-level only) — [arXiv:2509.15798](https://arxiv.org/abs/2509.15798)
- **Nested full/reduced neural models for Granger testing** compare a network with and without series i — a LOCO-style Granger test (verified, abstract-level only):
  - Sultan et al. — [arXiv:2208.03703](https://arxiv.org/abs/2208.03703)
  - "On the ability of deep neural networks to learn Granger causality" (2025) — [arXiv:2506.20347](https://arxiv.org/abs/2506.20347)
- **PCMCI** (Runge et al., Science Advances 2019) does time-series causal discovery with conditional-independence tests (including nonparametric CMI tests with permutation nulls) and optional FDR correction (from prior knowledge) — [DOI](https://doi.org/10.1126/sciadv.aau4996)

### Inferences
- The individual pieces have been applied to trained neural networks:
  - LOCO-type tests on DNNs (Dai–Shen–Pan, LazyVI, Williamson's `vimp` with neural learners);
  - knockoff and CRT-style tests on DNNs (DeepPINK, CPI, HRT);
  - response-permutation nulls for importance (PIMP);
  - shuffled-shadow decoys with repeated runs (Boruta).
- What seems new in the study's setup:
  - **Targets.** The targets are a trained world model's read-out predictions, not observed outcomes. Inputs are true simulator variables, so the procedure is a statistically tested distillation of the model's input–output dependency graph.
  - **Combined error control.** It combines decoy-based negative controls, episode-level repeated half-splits and a max-T permutation FWER over all edges of the graph.
  - **Rule selection.** The pruning rule was chosen on pre-specified validation graphs before seeing the result.

  I found no paper that does this combination. The novelty is in assembling and validating known components, not in a new test.
- Of the precedents, the neural-Granger literature (Tank, GVAR, JRNGC) is the one the study's graph output most resembles. The study's error-controlled pruning is stronger than those papers' penalty or stability thresholds. This is a fair, citable claim of improvement, with the subset-pivotality caveat from Q2.

### Gaps
- My searches were limited, so absence is not proof. I found no paper that tests a trained world model's (or any learned simulator's) dependency graph with formal FWER/FDR control.
- I could not confirm whether arXiv 2506.20347 or 2509.15798 give formal error control over a whole graph; I saw abstract-level information only.

---

## Q4. Recommended practice for validating a graph-extraction procedure with ground truth, noise-matched data and null controls

### Takeaway
Recommended practice is to check a procedure on:
- data with known ground truth;
- data where the null is known to hold (shuffled or permuted);
- a randomised or untrained model, as a "sanity check".

Benchmarks should not reward procedures for exploiting artefacts of how the data were simulated, such as varsortability or scale. The study's design follows these principles, with a known simulator graph, noise-matched data, a shuffled null, an untrained model, and a rule chosen before looking at the trained model. The main remaining risks are:
- the validation graphs share structural features (collinearity pattern, scale) that make some rules look better than they would on the trained model;
- selecting a rule among six on a single ground-truth graph is itself a mild multiplicity or overfitting step.

### Cited Findings
- Reisach, Seiler & Weichwald, "Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game", NeurIPS 2021 (verified). In simulated additive-noise benchmarks, marginal variance often increases along the causal order ("varsortability"). Continuous structure learners exploit this, and a trivial sort-by-variance baseline matches them. Standardising the data removes the advantage. They recommend reporting varsortability and caution against generic benchmarks with independently drawn parameters — [arXiv:2102.13647](https://arxiv.org/abs/2102.13647)
- Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018 (from prior knowledge). The model-parameter randomization test asks whether an attribution method's output changes when the trained weights are replaced by random weights. Methods that do not change are not explaining the model. This is the direct precedent for the study's untrained-model control — [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)
- Gentzel, Garant & Jensen, "The Case for Evaluating Causal Models Using Interventional Measures and Empirical Data", NeurIPS 2019 argue for evaluating causal discovery with interventional ground truth and realistic data, not only structural metrics on synthetic DAGs (from prior knowledge) — [arXiv:1910.05387](https://arxiv.org/abs/1910.05387)
- Pseudovariables, shadows and target-decoy all use known nulls to estimate realised false-selection rates, the same logic as a shuffled-null graph (verified for Wu–Boos–Stefanski, [IDEAS](https://ideas.repec.org/a/bes/jnlasa/v102y2007p235-243.html), and PIMP, [PubMed](https://pubmed.ncbi.nlm.nih.gov/20385727/); Boruta from prior knowledge, [DOI](https://doi.org/10.18637/jss.v036.i11))
- Fixing the analysis before seeing the target result is in the spirit of permutation-based neuroimaging pipelines, where the max-statistic threshold is set by the permutation distribution and not by the observed map (Nichols & Holmes 2002, from prior knowledge — [DOI](https://doi.org/10.1002/hbm.1058))

### Inferences
- **Coverage of the validation set.** The four validation graphs cover two things:
  - power and recovery: true data, and noise-matched data that mimics the world model's error level;
  - size: the shuffled null, and the untrained model, which has structure but no learned dynamics.
- **The untrained-model control.** It is especially informative because an untrained encoder–predictor still produces outputs that depend on inputs. A sound rule should find some dependence there, but not the simulator's graph. Reporting the edge-wise false-positive rate there separately from the shuffled null would make the distinction visible. (Inference.)
- **Suggested additions, following the cited practice (inferences):**
  - Report realised FWER/FDP across many independent null replicates (not one shuffled-null graph), with Monte Carlo CIs.
  - Report power per edge type, especially for the collinear inputs.
  - Check robustness to standardisation or scale (the Reisach lesson).
  - Include a "real signal elsewhere + null edge" synthetic case. This directly tests the subset-pivotality concern from Q2: plant known null inputs correlated with true causes and check the rule's false-positive rate on them.

### Gaps
- I found no published protocol specifically for "noise-matched" validation of graph extraction from learned models. The noise-matched control appears to be the study's own design; the closest analogues are the sanity-check and benchmark-critique papers above.

---

## Q5. Novelty and soundness of the full combination

### Takeaway
Each component has strong precedent:

| Component | Precedent |
|---|---|
| LOCO | Lei 2018; Williamson 2021/23 |
| Decoys | Wu–Boos–Stefanski 2007; Boruta 2010 |
| Split repetition and p-value aggregation | Meinshausen et al. 2009; stability selection; Dai–Shen–Pan |
| Response-permutation null for importance | PIMP 2010 |
| Max-T FWER | Westfall–Young 1993; neuroimaging |
| Exact p-values | Phipson & Smyth 2010 |
| Knockoffs | Barber–Candès; Candès et al.; Ren–Wei–Candès |

I found no prior work combining them to extract and test a causal graph from a trained world model's predictions. The procedure is careful about overfitting noise (decoys, held-out loss, splits, exact permutation p-values).

Its weaker points, all with known remedies in the literature:
- **Subset pivotality.** Max-T with response permutation is only guaranteed under subset pivotality; Freedman–Lane or CRT/HRT/CPI-style conditional nulls fix this.
- **Row-level shuffling** of autocorrelated episodes; block or episode permutation fixes this.
- **Coarse p-value floor** with K = 20 (p_min = 1/21); more null repeats fix this.
- **LOCO non-identifiability for collinear inputs**; group LOCO, decorrelated LOCO, SAGE/Shapley or interventional data fix this.

### Cited Findings
- Subset pivotality is required for strong FWER with complete-null resampling — [arXiv:1106.2068](https://arxiv.org/abs/1106.2068); [arXiv:1612.07010](https://arxiv.org/abs/1612.07010)
- Freedman–Lane, exchangeability with nuisance effects, and exchangeability blocks — [Winkler et al. 2014, PMC4010955](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4010955/)
- Response-permutation null for importance (PIMP) — [PubMed 20385727](https://pubmed.ncbi.nlm.nih.gov/20385727/)
- Conditional nulls for individual features in black boxes:
  - HRT — [arXiv:1811.00645](https://arxiv.org/abs/1811.00645)
  - CPI (knockoffs) — [arXiv:1901.09917](https://arxiv.org/abs/1901.09917)
  - CPI (conditional permutation) — [arXiv:2309.07593](https://arxiv.org/abs/2309.07593)
- LOCO under correlation, and decorrelated LOCO — [arXiv:2111.10853](https://arxiv.org/abs/2111.10853); [arXiv:2303.05981](https://arxiv.org/abs/2303.05981)
- Group-level importance inference:
  - Williamson et al. 2023 — [arXiv:2004.03683](https://arxiv.org/abs/2004.03683)
  - floodgate — [arXiv:2007.01283](https://arxiv.org/abs/2007.01283)
  - Dai–Shen–Pan — [arXiv:2103.04985](https://arxiv.org/abs/2103.04985)
- Exact permutation p-values — [arXiv:1603.05766](https://arxiv.org/abs/1603.05766)

### Inferences
- **How to describe the contribution.** It can honestly be described as "a pre-registered, null-calibrated combination of established variable-importance tests, applied to a learned world model". It should not be called a new test.
- **Concrete, low-cost strengthening steps** (inferences):
  1. Increase K for the null repeats.
  2. Permute at the episode or block level.
  3. Add a Freedman–Lane-style or conditional (CRT/HRT) null for the final edge set, or show empirically, with planted correlated nulls, that the row-shuffled max-T keeps FWER when other edges are real.
  4. Report group LOCO for the collinear block, noting that individual edges are not identifiable. Present the VIF-scaled analysis as an approximation to decorrelated LOCO.
  5. Use Benjamini–Yekutieli instead of BH if the BH rule is reported.
- **Defensible choices to keep:**
  - Choosing the rule on validation graphs before seeing the trained model's score. This is good practice, analogous to pre-specifying the max-T threshold.
  - The decoy and untrained-model controls, which echo Boruta, pseudovariables and Adebayo's randomisation test.

### Gaps
- There is no published empirical comparison of exactly these six pruning rules on a common benchmark, so their relative ranking cannot be grounded in the literature.
- Formal theory for max-T with refit-LOCO statistics under response permutation was not found.
