# Inferring Opinion Dynamics from Noisy Social Data with Particle-Based Bayesian Methods — Project Plan

Nov 8, 2026 · @Guilherme

## Goal and status

The project builds a hidden-state algorithm that learns opinion dynamics from limited, noisy data: latent opinions evolve on a network, people only see each other's expressions, and researchers only see noisy labels of those expressions. Parameters are inferred with a particle-filter likelihood, with the hidden trajectories integrated out.

- **Current phase:** Phase 1, scalar hidden opinions on simulated data.
- **Next action:** coverage study over repeated datasets to test whether process noise $\sigma_p$ and expression noise $\sigma_e$ are separated correctly (see 1.3), and to choose the default input mode.
- **Repository:** config loader, DeGroot model, simulator, joint and per-agent Kalman and particle filters, and check scripts in place.
- **Thesis contributions** (as stated to the supervisor, 2026-11-18):
    1. Inference under noisy dynamics: random shocks to opinions that the network does not explain (news, outside events), and an estimated noise between what people think and the score that measures it.
    2. Model selection (headline): which of DeGroot, FJ and bounded confidence the data follow more closely, with uncertainty on the choice (see 1.7).
    3. Heterogeneous agents: stubborn agents and per-agent confidence bounds, with per-agent uncertainty (see 1.5).
    4. Cost of ignoring noise and an identifiability map (see 1.2 findings and 1.3).
- **Schedule:** model and inference to Feb 2027; simulation studies, model selection and baselines Mar–Jun 2027; first paper submitted July 2027 (noisy dynamics, model selection, benchmark); real data Aug–Nov 2027; writing Dec 2027–Feb 2028; buffer to the June 2028 deadline. Heterogeneous agents follow the July paper.

## Phase 1 — Scalar hidden opinions vs. classical models

Phase 1 tests whether the algorithm recovers known parameters when data come from the classical models plus a realistic noise channel.

### 1.1 Simulated data

Each agent passes through opinion → expression → observation. Agents update only from neighbours' expressions, never from hidden opinions.

$$
x_{i,t} = \lambda\, x_{i,t-1} + (1-\lambda)\sum_j W_{ij}\, s_{j,t-1} + \varepsilon_{i,t}, \qquad s_{i,t} = x_{i,t} + \xi_{i,t}
$$

$$
Y_{i,t} = k \iff c_{k-1} < s_{i,t} + e_{i,t} \le c_k, \qquad k = 1,\dots,5
$$

- Noise: process $\varepsilon \sim \mathcal N(0, \sigma_p^2)$, expression $\xi \sim \mathcal N(0, \sigma_e^2)$, measurement $e \sim \mathcal N(0, \sigma_m^2)$.
- Initial opinions: $x_0 \sim \mathcal N(\mu_0, \sigma_0^2 I)$, so the linear models are exactly linear-Gaussian and the Kalman filter is a correctly specified reference.
- No clipping: in the continuous stage $x$, $s$ and $Y$ live in $\mathbb R$; in the ordinal stage boundedness comes from the outer thresholds ($c_0=-\infty$, $c_5=+\infty$); bounded hidden opinions, if needed, via $\mathrm{logit}\, x$.
- Baseline noise is moderate ($\sigma_e = 0.05$, $\sigma_m = 0.1$, total observation noise about the size of the opinions' own variation); high- and low-noise regimes are separate configs in the recovery study.
- Model order: DeGroot → FJ (linear) → bounded confidence (nonlinear).
- Observation switch: continuous $Y = s + e$, or ordinal with 5 levels (ordered probit).
- Parameters to recover: $\lambda$ (or model equivalents), $\sigma_p$, $\sigma_e$. Keep $\sigma_m$ and the thresholds $c_k$ fixed, assuming a labeled validation set.

### 1.2 Inference

- The particle filter computes the likelihood $p(Y \mid \psi)$ and filtered opinions as a by-product.
- Parameters: grid or optimizer first, then IF2 to locate the high-likelihood region, then PMMH for the joint posterior $p(\lambda, \sigma_p, \sigma_e \mid Y)$: all parameters estimated together, credible intervals and correlations, and opinion trajectories with parameter uncertainty integrated out.
- PMMH reference: for linear models, run MCMC with the exact Kalman likelihood to get a reference posterior; PMMH with the particle filter must reproduce it. Tune $N$ so that the sd of $\log\hat p(Y \mid \psi)$ is about 1–1.5 (standard PMMH), or so that the sd of the log-likelihood *ratio* between current and proposed points is about 1 (correlated PMMH).
- Correlated PMMH (implement from the start): all random numbers used by the particle filter are one Gaussian array $u$ (initial draws, process noise, `filtered_var` input draws, and resampling uniforms obtained as $\Phi(z)$ of Gaussians). Each iteration proposes $\psi'$ together with $u' = \rho\, u + \sqrt{1-\rho^2}\,\varepsilon$, $\varepsilon \sim \mathcal N(0, I)$, $\rho \approx 0.99$–$0.999$, and accepts both or neither. Likelihood estimates at $\psi$ and $\psi'$ then share most of their noise, so the ratio is far less noisy and far fewer particles are needed. Resampling breaks the correlation unless particles are sorted first; in the per-agent filter each state is scalar, so sorting each column by $x$ before systematic resampling is enough. The filter therefore takes pre-drawn arrays instead of an `rng`; standard PMMH is the special case $\rho = 0$, and both are validated against the Kalman-MCMC reference.
- PMMH is only as exact as the likelihood it uses: run it on the per-agent filter only after the input bias is fixed.
- In an agent's own emission, expression and measurement noise integrate out in closed form, with total variance $\sigma_e^2 + \sigma_m^2$.
- Neighbours' expressions are hidden inputs: (a) per-agent filter with plug-in inputs from $Y$, fast but biased; (b) joint filter over all agents on a small network, exact reference.
- Finding (2026-11-10): with raw $Y$ as inputs, the per-agent filter overestimates $\lambda$ through attenuation, and grows more confident as it gets more wrong. Mean $\hat\lambda$ over 20 datasets (true $0.3$): $0.32, 0.40, 0.57, 0.76$ for $\sigma_m = 0.02, 0.05, 0.1, 0.2$, while the joint Kalman estimate stays near $0.3$. At the baseline $\sigma_m = 0.1$ the plug-in profile puts the truth more than 60 log-likelihood units below its peak.
- Fixes, in order of preference: (1) corrected inputs, i.e. filtered estimates of $s_j$ in place of $Y_j$, computed causally in the same pass (filtered, not smoothed, so an agent's own later posts never leak into its inputs), with the input variance optionally added to the process noise; (2) neighbourhood (block) filters that also track neighbours' states; (3) simulation-based bias correction.
- Corrected inputs in detail. After the update at $t$, agent $j$'s filter has opinion mean $m_{j,t}$ and variance $P_{j,t}$; with $\kappa = \sigma_e^2/(\sigma_e^2+\sigma_m^2)$ the estimated expression and its variance are

$$
\hat s_{j,t} = (1-\kappa)\, m_{j,t} + \kappa\, Y_{j,t}, \qquad v_{j,t} = (1-\kappa)^2 P_{j,t} + (1-\kappa)\,\sigma_e^2 ,
$$

  and agent $i$ reads $\sum_j W_{ij}\hat s_{j,t}$ at step $t+1$. When $\sigma_m \to 0$, $\kappa \to 1$ and $\hat s \to Y$, so all modes coincide (built-in check).
- Input uncertainty (optional, mode `filtered_var`): $\hat s$ is still uncertain, and treating it as exact keeps the likelihood overconfident. In the Kalman version the input variance is added to the prediction,

$$
\mathrm{Var}(x_{i,t+1} \mid \cdot) = \lambda^2 P_{i,t} + \sigma_p^2 + (1-\lambda)^2 \sum_j W_{ij}^2\, v_{j,t} ;
$$

  in the particle version each particle draws its own neighbour expressions $\hat s_t + \sqrt{v_t}\,\eta$, $\eta \sim \mathcal N(0, I)$, which reuses `model.step` unchanged and carries over to nonlinear models such as bounded confidence.
- Input modes compared: `observed` ($Y_{j,t-1}$; biased, overconfident), `filtered` ($\hat s_{j,t-1}$; should remove most of the bias), `filtered_var` ($\hat s_{j,t-1}$ plus its uncertainty; should also bring the profile width close to the joint Kalman one). Any remaining gap is the cost of dropping the cross-agent covariances that the joint Kalman filter keeps.
- Finding (2026-11-10, corrected inputs; 20 datasets per $\sigma_m$, other parameters at truth): up to the baseline noise both corrected modes match the exact joint Kalman estimator in bias and profile width. At $\sigma_m = 0.1$: mean $\hat\lambda$ is $0.27$ (joint), $0.57$ (`observed`), $0.28$ (`filtered`), $0.25$ (`filtered_var`); 2-unit profile widths $0.160$, $0.08$, $0.178$, $0.164$. At $\sigma_m = 0.2$ the per-agent approximation breaks down: `filtered` stays centred ($0.31$) but very uncertain (sd $0.24$, width $0.36$ vs. $0.31$ exact); `filtered_var` is biased low ($0.11$ vs. $0.24$ exact) and too narrow ($0.21$). Hypothesis, untested: the input-variance term $(1-\lambda)^2 \sum_j W_{ij}^2 v_j$ grows as $\lambda$ falls and absorbs scatter, and it treats input errors as independent over time and across agents when they are correlated.
- The per-agent particle filter with `filtered_var` follows its Kalman profile but jitters by about $\pm 3$ log-likelihood units near the peak at $N = 500$: too noisy for standard PMMH, which motivates correlated PMMH.
- Steps without a post: propagate particles, skip weighting and resampling.
- Scalability, only if cost becomes a problem on large networks: case-control subsampling, taken from Sewell & Chen (2016, *Latent space models for dynamic networks with weighted edges*). They cut a latent-network likelihood from order $Tn^2$ to order $Tn$ terms by summing the positive edges exactly and only a random subsample of each actor's zero edges, rescaled by the ratio of all zeros to subsampled zeros. Check before use: with a sparse $W$ the per-agent likelihood is already linear in the number of neighbours, so the trick helps only if the observation sums over all pairs or the network is dense; and the subsample must stay fixed inside one PMMH run (or share the pre-drawn random numbers $u$), otherwise it adds noise that breaks the correlation between $\hat p(Y\mid\psi)$ and $\hat p(Y\mid\psi')$. Measure the cost first (see the 2026-11-09 timing).

### 1.3 Evaluation

- [x] Correctness: particle-filter log-likelihood matches the joint Kalman filter (linear model, continuous $Y$)
- [x] PMMH vs. Kalman-MCMC reference posterior (linear model): correlated PMMH ($N = 100$, $\rho = 0.99$) reproduces exact MH on the same per-agent likelihood (`check_pmmh.py`, means within 0.7 Monte Carlo s.e.)
- [ ] Coverage study, separating process noise from expression noise. Motivation (notebook 02, one dataset): the per-agent posteriors get the total noise $\tau^2 = \sigma_p^2 + \sigma_e^2$ right ($0.0049$ vs. $0.0047$ exact, truth $0.005$) but split it wrongly ($\sigma_p \approx 0.06$, $\sigma_e \approx 0.035$–$0.04$ vs. exact $0.045$, $0.052$; truth $0.05$ each). Design: 50 datasets on the baseline network ($\sigma_m = 0.1$ fixed); exact MH on the per-agent Kalman likelihood with `filtered` and `filtered_var` inputs for all 50, and on the joint Kalman likelihood for a subset of about 20 (cost); for $\lambda$, $\sigma_p$, $\sigma_e$, $\tau^2$ and the share $f = \sigma_e^2/\tau^2$, report the share of datasets whose 95% interval covers the truth and the mean bias. Expected: $\tau^2$ well covered, $f$ biased in the per-agent model. Outcome decides the default input mode and whether block filters are needed to separate $\sigma_p$ from $\sigma_e$
- [ ] Parameter recovery vs. number of steps $T$, number of agents $n$ and noise level: bias, interval coverage, smallest usable $T$
- [ ] Cost of approximations: discrete observation and plug-in inputs, version (a) vs. (b). Raw and corrected plug-in inputs measured with other parameters at truth (see Findings in 1.2); still to measure under joint estimation with PMMH and for the discrete observation
- [ ] Identifiability: $\sigma_e$ vs. influence parameters; multimodal posteriors under bounded confidence
- [ ] Robustness to a misspecified network: simulate on the true $W$, fit with a perturbed one (a fraction of edges rewired or missing); measure how the estimates of $\lambda$, $\sigma_p$ and $\sigma_e$ degrade as the perturbation grows
- [ ] Robustness to a misspecified measurement layer: simulate with one $\sigma_m$ (and thresholds), fit with $\sigma_m$ fixed at a wrong value; measure the bias in $\lambda$, $\sigma_p$, $\sigma_e$
- [ ] Simulation-based calibration of the whole pipeline (Talts et al. 2018): rank statistics of the true parameters within their posteriors over many prior draws
- [ ] Cost of ignoring noise (contribution 4): the labels-as-opinions fit of 1.6 against PMMH, across noise levels
- [ ] Identifiability map (contribution 4): which parameters are recoverable under which series length, network size, noise level and distance from consensus, and which only in combination

### 1.4 Realism extensions

- [ ] Heterogeneous posting rates; agents update only when exposed to new posts
- [ ] Several posts per time step (time steps as windows); weight = product over posts
- [ ] Posting rate that depends on opinion (informative silence), modeled in the likelihood

### 1.5 Heterogeneous agents

Detect agent-level differences, such as who is stubborn or each agent's confidence range. Starts after corrected neighbour inputs and PMMH are in place, and before Phase 2: Beta beliefs bring their own stubbornness through certainty $\kappa_i$, which would be confounded with heterogeneous $\lambda_i$.

- Models: hierarchical, e.g. $\mathrm{logit}\,\lambda_i \sim \mathcal N(\mu_\lambda, \tau_\lambda^2)$ with partial pooling; or agent types $z_i \in \{\text{stubborn}, \text{flexible}\}$ with posterior $P(z_i = 1 \mid Y)$.
- Inference: Gibbs-style updates alternating between population parameters and each agent's own parameters (plain PMMH does not scale to $n$ extra parameters).
- Expected difficulties: $\lambda_i$ is identified only when agent $i$'s neighbours disagree with it, so there is little information near consensus; agents with few neighbours or posts stay uncertain; $\epsilon_i$ is informative only when neighbours come near the boundary.

Stages:

- [ ] DeGroot with per-agent $\lambda_i$ (exact Kalman reference still available)
- [ ] FJ with per-agent attachment to the initial opinion (zealots; still linear)
- [ ] Bounded confidence with per-agent range $\epsilon_i$

Evaluation:

- [ ] Recovery of the population spread $\tau_\lambda$
- [ ] Correlation between estimated and true $\lambda_i$
- [ ] Detection of stubborn agents: AUC, and share of the top-$k$ flagged agents that are truly stubborn
- [ ] Calibration of per-agent credible intervals
- [ ] All of the above as functions of $T$, $n$ and the noise level

### 1.6 Baselines and comparison (matches proposal §4.5)

Three external baselines, each compared only on what both methods return. The two inference methods of Kolic et al. (PNAS Nexus 2026, *Comparing data assimilation and likelihood-based inference on latent state estimation in agent-based models*) are run in their own setting with their public code; neural posterior estimation is run on my simulations. Literature map: Theme 1 (Kolic), Theme 11 and the related-literature tab (NPE).

| Method | Needs to run | Returns | Compared on |
| --- | --- | --- | --- |
| This thesis | Noisy labels, network, calibrated measurement layer, priors on the parameters | Posterior of the parameters and of the hidden-opinion trajectories; probabilities of the three models | Reference for all comparisons |
| B1 Likelihood-based inference (Kolic et al. 2026) | Observed data, network, model with its parameters given | One trajectory per agent, from the estimated initial opinions; no uncertainty | State error, at agent level and for the opinion distribution |
| B2 Ensemble Kalman filter (Kolic et al. 2026) | Same as B1 | Estimate of each hidden opinion at each time step, updated as data arrive | State error, at agent level and for the opinion distribution |
| B3 Neural posterior estimation (Papamakarios & Murray 2016; Dyer et al. 2024) | Simulator, prior on the parameters, many simulated datasets, graph neural network summary | Posterior of the parameters; with a classifier, probabilities of the three models; no hidden opinions | Parameter bias and 95% coverage; model-selection accuracy |

Comparison rules:

- Against B1 and B2, which take the parameters as given: state error only, on their benchmark. In that setting my method also estimates the confidence bound $\epsilon$, which they treat as known; report this alongside, and repeat their wrong-$\epsilon$ experiment.
- Against B3: parameter bias and coverage of 95% intervals over repeated datasets, and model choice (how often the correct model is chosen, from the 1.7 experiment).
- For every method: computing time. Where a method gives forecasts: log predictive score and calibration on held-out periods, agent-level and aggregate forecasts side by side.
- Why these: Kolic et al. (May 2026) is the most recent published comparison for recovering hidden opinions from behavioural data; NPE is the strongest Bayesian alternative without a likelihood, benchmarked on agent-based models (Dyer et al. 2024) but tested so far only with fully observed opinions (Dyer et al. 2022).

**Kolic benchmark (B1, B2).** Deterministic, synchronous bounded confidence; $N = 100$ on a complete graph; binary interaction indicators for all 4,950 pairs; fit on steps 0–250 of 1,000; $\epsilon \in \{0.2, 0.3\}$; $\mu = 10^{-4}$; 10 seeds. Code and data on Zenodo (doi:10.5281/zenodo.17153274). Replicate their Fig. 1 first.

- The particle filter needs two changes for this setting: a soft observation link $P(y_{ij} = 1) = \sigma\big(\rho\,(\epsilon - |x_i - x_j|)\big)$, because a 0/1 step function gives particles weight 0 or 1; and a small process noise, because deterministic dynamics never rediversify resampled particles.
- Check the sign of the published logit, printed as $|x_i - x_j| - \epsilon$, against their code.
- State error: their $E$ (mean absolute error per agent) and $E_{\mathrm{sort}}$ (opinion distribution), with one mirror reflection $x \to 1 - x$ for the whole vector, not one per agent as in their printed $E_{\mathrm{symm}}$.
- Every pair is observed at every step, so the joint filter will collapse; use the per-agent filter with corrected inputs.
- Run their code in its own environment (PyTorch); only its outputs enter `checks/check_baselines.py`.

**Neural posterior estimation (B3).** `sbi`'s `NPE_C` (automatic posterior transformation, Greenberg, Nonnenmacher & Macke 2019), on datasets simulated from the prior.

- Summary network: a recurrent graph neural network over the label matrix and $W$, trained jointly with the density estimator, as in Dyer et al. (2022). Hand-picked summaries (per-agent label means and lag-1 autocorrelations, correlation between an agent's change and its neighbours' mean label, label distribution per step, share of agents whose label changed) as a cheaper fallback and ablation.
- Model selection: a classifier on the same summaries returns a probability for each of the three models (compare with Chen, Mira & Onnela 2020).
- Same simulation budget reported for NPE and PMMH; NPE is amortized, so also report its cost per new dataset after training.
- Check its calibration with simulation-based calibration, the same as the main method. Expect it to lose accuracy as $n$ grows, because the summaries discard agent-level information that the likelihood keeps.
- Optional newer estimator: flow-matching posterior estimation (Wildberger et al. 2023).

**Internal references (not baselines).** Used inside the validation of 1.3, not reported as competing methods:

- Oracle: the particle filter (or Kalman filter) at the true $\psi$, as an upper bound.
- Labels treated as opinions: the same dynamics fitted by least squares directly to the labels, $Y_{i,t} \approx \lambda Y_{i,t-1} + (1-\lambda)\sum_j W_{ij} Y_{j,t-1}$ (ordinal labels at their level midpoints). This measures the cost of ignoring noise (contribution 4) and is also a simple alternative on real data.

**Dropped from the comparison (2026-11-18).** The whole-path MAP extension of likelihood-based inference and an ensemble Kalman filter built on my own model (ordinal observation, localization along $W$) are no longer baselines: the proposal compares with Kolic et al. in their own setting. Keep them as optional extra analyses only if time allows.

Stages:

- [ ] Kolic benchmark: replicate one seed of their Fig. 1 with their code; add the particle filter with $\epsilon$ known; then PMMH with $\epsilon$ unknown, including the wrong-$\epsilon$ test
- [ ] NPE on continuous DeGroot: parameter bias and coverage against PMMH, as functions of $n$ and $\sigma_m$
- [ ] NPE on FJ and bounded confidence; NPE model classifier in the 1.7 experiment
- [ ] Computing time for all methods on the same datasets
- [ ] Phase 3: forecasts on held-out periods for the methods that give them

### 1.7 Model selection (headline contribution)

The method should tell which rule the data follow more closely, DeGroot, FJ or bounded confidence, instead of assuming one. Part of the July 2027 paper.

- Scores per model $M$: (a) the marginal likelihood $p(Y \mid M)$, estimated by SMC² or by importance sampling over the PMMH posterior with the particle-filter likelihood; (b) leave-future-out predictive score (Bürkner, Gabry & Vehtari 2020). Posterior model probabilities with equal prior weights.
- Experiment: simulate datasets from each of the three models on the same network, over a grid of $T$, $n$ and $\sigma_m$; fit all three to each; report the confusion matrix of the chosen model and the calibration of the model probabilities.
- Expected difficulty: the models are nested or nearly so. DeGroot is FJ with no attachment to the initial opinion, and bounded confidence with a large bound behaves like DeGroot. Near consensus, or when the confidence bound is never reached, the data cannot tell them apart; the posterior should then stay undecided, and the experiment maps where it does.
- Baseline for this task: the NPE model classifier of B3.

Stages:

- [ ] DeGroot vs FJ (both linear; exact Kalman evidence available as reference)
- [ ] Add bounded confidence
- [ ] Map of decidable vs undecidable regimes (distance from consensus, $T$, noise)
- [ ] Phase 3: posterior model probabilities on the Reddit data

## Phase 2 — Beta hidden opinions

Phase 2 replaces the scalar opinion with a belief distribution $\text{Beta}(a, b)$, with direction $m = a/(a+b)$ and certainty $\kappa = a+b$.

- Expression is sampled from the agent's own belief, so uncertain agents post more erratically.
- Influence through expressed signals: pseudo-count updating with forgetting, where $\delta$ is memory and $\eta$ is social learning strength.

$$
\boldsymbol\alpha_{i,t} = \delta\,\boldsymbol\alpha_{i,t-1} + \eta\sum_j W_{ij}\,\mathbf e(s_{j,t-1})
$$

- Certainty settles at $\kappa^* = \eta/(1-\delta)$; classical DeGroot with self-weight $\delta$ is the limiting case.
- Process noise in $(\mathrm{logit}\, m, \log\kappa)$ coordinates to keep $a, b > 0$.
- Alternative variant to compare: logarithmic pool (agents share full beliefs), $\boldsymbol\alpha_t = W\boldsymbol\alpha_{t-1}$.
- Truth models: stochastic opinion-dynamics models.

## Phase 3 — Real data

Phase 3 asks which opinion-dynamics model best explains real data and whether the hidden-state model predicts better than simpler alternatives.

1. Calibrate the measurement layer on a small hand-labeled subset of posts.
2. Fit DeGroot, FJ and bounded confidence; compare by predictive performance on held-out time periods.
3. Baselines: no change (last label persists); population frequencies or random guessing; the same dynamics fitted to the labels without the hidden layer; neural posterior estimation where it gives forecasts (see 1.6). The Kolic et al. methods are compared on their benchmark only.
4. Metrics: log predictive score and calibration of predictive intervals, not only accuracy.

Dataset: the Reddit climate-change dataset of Antelmi, Spagnuolo & Aiello (2026, *Online Social Networks and Media* 53, 100350; data on Zenodo, record 15225705; code github.com/alessant/GrootSim).

- Six subreddits active on the 2021–2022 California wildfires (r/collapse, r/news, r/AskReddit, r/politics, r/bayarea, r/California), 1 July 2020 – 31 December 2022, collected with Pushshift and filtered by climate keywords; bots and one-time posters without replies removed.
- 54,923 users, 6,251 submissions, 363,350 comments; 14,673 users (26.7%) have at least two scored posts. Few users cross subreddits (40 post in all six).
- Each record is (user $u$ replied to user $v$, text, time), so the reply network $W$ is built directly; the authors use one-day windows and found one-week windows predict worse.
- Stance: GPT-3.5 Turbo score in $[0, 1]$ toward "Climate change is a real concern" (0 strongly against, 0.5 neutral, 1 strongly in favour; unrelated comments marked NI). Validated on SPINOS: correct positive/negative direction in 90% of cases. The accuracy of the continuous score is not measured, which is why $\sigma_m$ must be calibrated on hand labels.
- The authors call the scores a synthetic ground truth, not true beliefs, and their own diffusion model predicts stability much better than change. Both support the measurement layer and give a concrete target.
- To decide: time step (start with one day, test longer), minimum posts per user, and whether to use the continuous score directly or its five ordinal levels.

## Future work — Unknown network layer

Not part of Phases 1–3, where $W$ is given (and its misspecification is only stress-tested in 1.3). If the graph $G$ is not available and must be inferred from interactions, Sewell & Chen (2016) offer a template for that layer: each actor gets a latent social position $X_{it}$ that follows a Gaussian random walk, and the weighted interaction counts $y_{ijt}$ (Poisson, or Tobit for non-negative reals) depend on the distance between positions and on each actor's reach. The inferred closeness could define $W_t$ for the opinion dynamics.

- Cost: one more hidden layer, so a larger state and a harder joint posterior.
- Open points: how to turn latent distances into influence weights $W_{ij}$; whether positions and opinions can be told apart when both drift (homophily would confound them); positions are identified only up to rotation and translation, so only relative geometry can feed $W$.
- Where it sits in the literature map: Theme 3, page *Latent-space networks · Sewell & Chen 2016*.

## Future work — Modular architecture

Later goals add new kinds of hidden quantity: agent roles, unknown or dynamic links, and parameters that change over time. The idea comes from Lenti, Silvestri & De Francisci Morales (IEEE TCSS 2026, *Variational inference of parameters in opinion dynamics models*, in the literature map). They extend one base model in one direction per rule category (structural, interaction, update, graph) and validate each variant separately. Here the same idea becomes a modular algorithm: the current chain (opinion → expression → label on a network $W$) is the base, and each goal is an extension module on top of it. Only the structural module is inside the thesis (Phase 1.5), plus perhaps one simple temporal module; the rest is future work.

Each module declares four things:

1. Its hidden variables and their scope: global ($\lambda$), per agent ($\lambda_i$, $z_i$), per link ($W_{ij}$) or per time ($\theta_t$).
2. Their prior.
3. Where it enters the base model: dynamics, network or observation.
4. Its own update step in a blocked sampler (PMMH within Gibbs, or particle Gibbs), which cycles through the modules. One joint variational distribution, as in the paper, does not fit variables this different in kind and size.

| Goal | Module | Hidden variables | Update step | Main risk |
| --- | --- | --- | --- | --- |
| Role detection (stubborn agents, leaders) | Structural | $z_i \in \{1, \dots, K\}$ or $\mathrm{logit}\,\lambda_i \sim \mathcal N(\mu_\lambda, \tau_\lambda^2)$ | Gibbs on $z_i$ or $\lambda_i$ given the sampled paths; hyperparameters separately | Weak identification near consensus and for agents with few posts (see 1.5) |
| Unknown or dynamic links | Graph | Static $W$, or $W_t$ through latent positions (see Unknown network layer) | Own block; latent positions keep it to order $nd$ instead of $n^2$ | Homophily: drifting positions and drifting opinions are hard to tell apart |
| Non-stationary dynamics | Temporal | $\theta_t$ as a random walk (e.g. on $\mathrm{logit}\,\lambda_t$), or change points $\tau_1, \tau_2, \dots$ | $\theta_t$ added to the filter's state, or SMC² and change-point moves | A drifting $\lambda_t$ and a large $\sigma_p$ look alike |
| Richer observation | Observation | Posting rates, informative silence (see 1.4) | Inside the likelihood | Silence that depends on opinion is not missing at random |

Validation rule: one module at a time, then pairs, then combinations. The same pattern can come from different modules. An agent who barely moves toward its neighbours may be stubborn (high $\lambda_i$), weakly linked (small $W_{ij}$), or living through a period when everyone was stubborn (high $\lambda_t$).

1. Fit each module alone on the base model, with the recovery and coverage studies of 1.3.
2. Run pairwise confusion tests: simulate from module A, fit A and B together, and check that B stays at its null value (uniform $\lambda_i$, fixed $W$, constant $\theta$).
3. Combine modules only after that, and report which combinations the data can tell apart, as a result in its own right.

Rules:

- No chained estimates (roles estimated and fixed, then links estimated on top). Estimate jointly, or carry the first stage's uncertainty forward. Fixing $\sigma_m$ from hand labels in Phase 3 is the one deliberate exception, with a sensitivity check over plausible values of $\sigma_m$.
- Some modules change the filter, not only the parameters. A time-varying $\theta_t$ enlarges the filtered state. A dynamic $W_t$ changes the neighbour inputs of the per-agent filter, where the plug-in bias of 1.2 appeared. Each such module re-runs the filter checks (joint Kalman reference for linear models, joint particle filter on small networks) before any parameter study.
- Per-agent and per-link modules add $n$ or $n^2$ unknowns, so blocked updates are the default, not plain PMMH.

Interfaces to fix now, cheaply, so later modules plug in without rewrites:

- One dynamics signature, `step(x_prev, s_hat_prev, W_t, theta_t, noise)`, with $W_t = W$ and $\theta_t = \theta$ constant for now.
- Parameters tagged with their scope (`global`, `agent`, `link`, `time`) in the config, so priors and update steps can be assigned per scope.
- Filters and samplers take the model's components (dynamics, expression, observation, network) as arguments instead of hard-coding them.

Order: structural module (Phase 1.5) → a single change point in $\lambda$ (simplest temporal module) → random-walk $\theta_t$ → static unknown $W$ → dynamic $W_t$. Each module brings its own check script and its own section in `notebooks/01_sanity_checks.ipynb`.

## Code structure and conventions

The repository starts small and grows one verified step at a time.

| File | Role | State |
| --- | --- | --- |
| `configs/degroot_base.yaml` | Parameter values for one run | Filled |
| `configs/inference_base.yaml` | Priors, MCMC and PMMH settings for parameter inference | Filled |
| `src/config.py` | Loads YAML into strict dataclasses | Implemented |
| `src/model.py` | Network, dynamics and expression layers | DeGroot implemented |
| `src/observation.py` | Observation layer: simulation of $Y$, its likelihood, and expression estimates from $(x, Y)$ shared by the filters | Continuous observation implemented |
| `src/neighbour_inputs.py` | Neighbour-input modes of the per-agent filters (`observed`, `filtered`, `filtered_var`) | Implemented |
| `src/simulate.py` | Ground truth $x$, $s$, $Y$ and posting mask $P$ | Implemented (Gaussian x0, save/load) |
| `src/kalman.py` | Exact Kalman log-likelihood (linear-Gaussian reference) | Joint and per-agent Kalman log-likelihood; per-agent with observed / filtered / filtered_var neighbour inputs |
| `src/inference.py` | Pseudo-marginal MH: exact MH on Kalman likelihoods, correlated PMMH on the particle filter | Implemented |
| `src/particle_filter.py` | Log-likelihood estimator | Joint and per-agent filters (continuous observation); per-agent with observed / filtered / filtered_var neighbour inputs; noise-driven per-agent filter with sorted resampling (for correlated PMMH) |
| `checks/check_joint_filter.py` | Joint particle filter vs. joint Kalman filter | PF vs Kalman comparison |
| `checks/common.py` | Shared helpers for the check scripts (paths, base config, $\lambda$ profiles, profile width) | Implemented |
| `checks/plotting.py` | Shared figure helpers for the check scripts (style, $\lambda$-profile figure, log ticks, saving) | Implemented |
| `checks/check_per_agent_filter.py` | Per-agent particle filter vs. per-agent Kalman filter | Implemented |
| `checks/check_plugin_bias.py` | Plug-in bias: per-agent vs. joint Kalman estimates of $\lambda$ | Implemented |
| `checks/check_correlated_pf.py` | Unbiasedness, smoothness and correlation payoff of the noise-driven filter | Implemented |
| `checks/check_pmmh.py` | Posteriors of $(\lambda, \sigma_p, \sigma_e)$: exact MH on joint and per-agent Kalman vs. correlated and standard PMMH | Implemented |
| `notebooks/01_sanity_checks.ipynb` | Sanity checks of the model layer (classical limit, noise, reproducibility) | DeGroot, simulator and Kalman checks passing |
| `notebooks/02_simulate_and_filter.ipynb` | Narrative of the method so far: data-generating process, filtering, likelihood landscape, plug-in bias, posterior inference | Filled |
| `src/baselines/kolic_benchmark.py` | Runs the public code of Kolic et al. (likelihood-based inference B1, ensemble Kalman filter B2) on their benchmark and loads their outputs | Planned |
| `src/baselines/labels_as_opinions.py` | Least-squares fit of the dynamics directly to the labels (internal reference, cost of ignoring noise) | Planned |
| `checks/check_baselines.py` | Main method against B1–B3 on the quantities each returns (Table in 1.6) | Planned |
| `src/baselines/npe.py` | Neural posterior estimation baseline B3 (`sbi`, PyTorch): simulator wrapper, summary statistics, training | Planned |
| `src/model_selection.py` | Marginal-likelihood and leave-future-out scores per model; posterior model probabilities | Planned |
| `checks/check_model_selection.py` | Confusion matrix of the chosen model across DeGroot, FJ and bounded confidence (1.7) | Planned |

- Pass `rng = np.random.default_rng(seed)` explicitly; no global randomness.
- Separate random streams for network and data (`SeedSequence(seed).spawn(2)`), so many datasets can be drawn on the same network.
- Missing posts are `NaN` in $Y$ (where $P$ is false); filters skip them.
- Every simulated dataset is saved as `.npz` with its config next to it.
- Performance: keep the code vectorized NumPy; vectorize hot loops (e.g. per-agent resampling) first, then parallelize independent runs across CPU cores (`joblib`/`multiprocessing`); consider JAX on GPU only if a measured run is still too slow (most likely with real data in Phase 3).
- Dynamics chosen by a dispatch dictionary; unsupported models raise an error.
- One config file per experiment; save the config next to every result.
- One check script per question in `checks/`; each writes its figures and config to `results/checks/<name>/`.
- Split `model.py` into dynamics and network files when FJ arrives (the observation layer is already in `observation.py`).
- Every new dynamics model gets its own section in `notebooks/01_sanity_checks.ipynb` (classical limit, noise behaviour, reproducibility) before it is used anywhere else.
- Keep the dynamics signature `step(x_prev, s_hat_prev, W_t, theta_t, noise)` and scope-tagged parameters (`global`, `agent`, `link`, `time`) from now on, so future modules plug in (see Future work — Modular architecture).
- The Kolic et al. code (PyTorch) runs in its own environment; only its outputs enter the repository's checks. JAX is no longer needed unless the optional whole-path MAP analysis is done.
- The NPE baseline uses the `sbi` package (PyTorch) in `src/baselines/npe.py` only; keep PyTorch out of the core code.
- Config still to add: `thresholds`.

## Decisions and open questions

Decisions are logged newest first; add a row whenever the plan changes.

| Date | Decision |
| --- | --- |
| 2026-11-18 | Validation and baselines aligned with the revised proposal (§4.4–4.5): three external baselines, B1 likelihood-based inference and B2 ensemble Kalman filter run with Kolic et al.'s code on their benchmark and compared on state error only, B3 neural posterior estimation compared on parameter bias, coverage and model choice; oracle and labels-as-opinions kept as internal references; whole-path MAP and an EnKF on my own model dropped (optional only); simulation-based calibration and misspecified-measurement-layer test added to 1.3 |
| 2026-11-18 | B5 (now B3) uses a graph-neural-network summary of the label matrix (Dyer et al. 2022) instead of hand-picked summaries, which stay as an ablation; cited as NPE (Papamakarios & Murray 2016) applied to ABMs (Dyer et al. 2024) |
| 2026-11-18 | After supervisor feedback: four stated contributions, with model selection among DeGroot, FJ and bounded confidence as a headline (new 1.7, in the July 2027 paper); neural simulation-based inference (NPE) added as baseline B5, SINN and graph diffusion networks cited only; Reddit climate dataset of Antelmi et al. (2026) fixed for Phase 3; heterogeneous agents moved after the July paper |
| 2026-11-17 | Thesis title set to "Inferring Opinion Dynamics from Noisy Social Data with Particle-Based Bayesian Methods" |
| 2026-11-16 | Future work: modular architecture after Lenti, Silvestri & De Francisci Morales (2026). The current chain is the base; roles, links, time-varying parameters and richer observation become extension modules, each with its own hidden variables, prior and update step in a blocked sampler. Validated one module at a time, then pairwise confusion tests, then combinations; no chained estimates. Interfaces (dynamics signature with $W_t$, $\theta_t$; scope-tagged parameters) fixed now; inside the thesis only the structural module and possibly one change point |
| 2026-11-15 | Added 1.6 Baselines: oracle, no hidden layer, path MAP (LBI extended from Kolic et al. 2026 to the whole path, since process noise makes $x_0$ insufficient), ensemble Kalman filter (only for ordinal and nonlinear stages; the exact Kalman filter covers linear-Gaussian), and a replication of the Kolic benchmark as the first external check; all scored on parameters, states, forecasts at agent and aggregate level, and cost |
| 2026-11-13 | From Sewell & Chen (2016): case-control subsampling noted in 1.2 as a scalability option, to be used only if a measured run is too slow and only after checking it fits a sparse $W$ and correlated PMMH; an inferred network layer (latent social positions) recorded as future work, outside Phases 1–3 |
| 2026-11-12 | Next step is a coverage study over repeated datasets: per-agent posteriors recover $\sigma_p^2 + \sigma_e^2$ but appear to misallocate it between process and expression noise |
| 2026-11-10 | Corrected neighbour inputs fix the per-agent filter up to the baseline noise ($\hat\lambda \approx 0.28$ vs. $0.57$ with raw labels at $\sigma_m = 0.1$); $\sigma_m = 0.2$ recorded as a known limitation. PMMH will be the correlated version from the start, with sorted per-agent resampling and pre-drawn random numbers |
| 2026-11-10 | Shared code gets one home: observation formulas in `src/observation.py` (split from `model.py`), input modes in `src/neighbour_inputs.py`, check helpers in `checks/common.py` |
| 2026-11-10 | Added Phase 1.5, heterogeneous agents (per-agent $\lambda_i$, FJ zealots, per-agent $\epsilon_i$), after corrected inputs and PMMH and before Phase 2; PMMH validated against Kalman-likelihood MCMC |
| 2026-11-10 | Raw plug-in inputs make the per-agent filter unusable for $\lambda$ at realistic noise ($\hat\lambda \approx 0.57$ vs. true $0.3$ at $\sigma_m = 0.1$, with overconfident profiles); next step is corrected neighbour inputs, benchmarked with `check_plugin_bias.py` |
| 2026-11-09 | No GPU for now: a per-agent likelihood evaluation takes ~0.2 s on CPU ($n=50$, $T=100$, $N=500$); parallelize across runs before considering JAX/GPU |
| 2026-11-09 | Added robustness experiment: fit with a perturbed network to measure sensitivity to network measurement error |
| 2026-11-08 | No clipping of opinions or observations: it breaks the Kalman reference and biases a Gaussian likelihood; the ordinal observation handles bounds |
| 2026-11-08 | Baseline noise lowered to $\sigma_e = 0.05$, $\sigma_m = 0.1$ (was 0.1, 0.2, where a single observation was ~85% noise); noise level becomes an experimental axis |
| 2026-11-08 | Simulator: Gaussian initial opinions, separate RNG streams for network and data, missing posts as `NaN`, datasets saved with their config |
| 2026-11-08 | `01_sanity_checks.ipynb` is the shared sanity-check notebook for all dynamics models; DeGroot checks pass |
| 2026-11-08 | Formulas written in LaTeX (`$…$` inline, `$$…$$` display) instead of code blocks |
| 2026-11-08 | Posting mask $P$ built in from the start (all ones in Phase 1.1–1.3) |
| 2026-11-08 | Expression is scalar; only the observation is discrete (5 levels, ordered probit) |
| 2026-11-08 | First version uses scalar hidden opinions; Beta beliefs deferred to Phase 2 |
| 2026-11-08 | Agents update only from neighbours' expressions, never hidden opinions |
| 2026-11-06 | Likelihood via particle filter; start with simulation validation |

Open questions:

- [ ] Why does `filtered_var` become biased low and too narrow at $\sigma_m = 0.2$, and does joint estimation of $\sigma_p$, $\sigma_e$ change the picture?
- [ ] Which input mode (`filtered` or `filtered_var`) gives correct posterior coverage?
- [ ] Can the per-agent filter separate process noise from expression noise, or does it only identify their sum? Hypothesis: with $\lambda = 0.3$ an agent's own process noise fades quickly, so within one agent's series both noises look nearly white; the network propagation that separates them is mostly lost when neighbours' states are replaced by point estimates
- [ ] Can $\sigma_e$ and $\sigma_m$ be recovered jointly, or must $\sigma_m$ stay fixed?
- [ ] Time step size for real data
- [x] Real dataset for Phase 3: Antelmi et al. (2026) Reddit climate dataset
