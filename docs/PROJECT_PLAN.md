# Learning Opinion Dynamics from Noisy Data — Project Plan

Nov 8, 2026 · @Guilherme

## Goal and status

The project builds a hidden-state algorithm that learns opinion dynamics from limited, noisy data: latent opinions evolve on a network, people only see each other's expressions, and researchers only see noisy labels of those expressions. Parameters are inferred with a particle-filter likelihood, with the hidden trajectories integrated out.

- **Current phase:** Phase 1, scalar hidden opinions on simulated data.
- **Next action:** add corrected neighbour inputs to the per-agent filter: replace raw labels $Y_{j,t-1}$ by the filtered estimate $\hat s_{j,t-1} = \mathbb E[s_{j,t-1} \mid Y_{0:t-1}]$ computed by the neighbours' own filters in the same pass, optionally adding the input uncertainty to the process noise; measure the remaining bias with `checks/check_plugin_bias.py`.
- **Repository:** config loader, DeGroot model, simulator, joint and per-agent Kalman and particle filters, and check scripts in place.

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
- PMMH reference: for linear models, run MCMC with the exact Kalman likelihood to get a reference posterior; PMMH with the particle filter must reproduce it. Tune $N$ so that the sd of $\log\hat p(Y \mid \psi)$ is about 1–1.5; correlated PMMH later if more efficiency is needed.
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
- Steps without a post: propagate particles, skip weighting and resampling.

### 1.3 Evaluation

- [x] Correctness: particle-filter log-likelihood matches the joint Kalman filter (linear model, continuous $Y$)
- [ ] PMMH vs. Kalman-MCMC reference posterior (linear model): same posterior means, intervals and correlations
- [ ] Parameter recovery vs. number of steps $T$, number of agents $n$ and noise level: bias, interval coverage, smallest usable $T$
- [ ] Cost of approximations: discrete observation and plug-in inputs, version (a) vs. (b). Raw plug-in inputs measured (see Finding in 1.2); corrected inputs still to measure
- [ ] Identifiability: $\sigma_e$ vs. influence parameters; multimodal posteriors under bounded confidence
- [ ] Robustness to a misspecified network: simulate on the true $W$, fit with a perturbed one (a fraction of edges rewired or missing); measure how the estimates of $\lambda$, $\sigma_p$ and $\sigma_e$ degrade as the perturbation grows

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
3. Baselines: no change (last label persists); population frequencies or random guessing; the same dynamics fitted without the hidden layer.
4. Metrics: log predictive score and calibration of predictive intervals, not only accuracy.

Open question: which dataset.

## Code structure and conventions

The repository starts small and grows one verified step at a time.

| File | Role | State |
| --- | --- | --- |
| `configs/degroot_base.yaml` | Parameter values for one run | Filled |
| `src/config.py` | Loads YAML into strict dataclasses | Implemented |
| `src/model.py` | Network, dynamics and expression layers | DeGroot implemented |
| `src/observation.py` | Observation layer: simulation of $Y$, its likelihood, and expression estimates from $(x, Y)$ shared by the filters | Continuous observation implemented |
| `src/neighbour_inputs.py` | Neighbour-input modes of the per-agent filters (`observed`, `filtered`, `filtered_var`) | Implemented |
| `src/simulate.py` | Ground truth $x$, $s$, $Y$ and posting mask $P$ | Implemented (Gaussian x0, save/load) |
| `src/kalman.py` | Exact Kalman log-likelihood (linear-Gaussian reference) | Joint and per-agent Kalman log-likelihood; per-agent with observed / filtered / filtered_var neighbour inputs |
| `src/particle_filter.py` | Log-likelihood estimator | Joint and per-agent filters (continuous observation); per-agent with observed / filtered / filtered_var neighbour inputs |
| `checks/check_joint_filter.py` | Joint particle filter vs. joint Kalman filter | PF vs Kalman comparison |
| `checks/common.py` | Shared helpers for the check scripts (paths, base config, $\lambda$ profiles, profile width) | Implemented |
| `checks/check_per_agent_filter.py` | Per-agent particle filter vs. per-agent Kalman filter | Implemented |
| `checks/check_plugin_bias.py` | Plug-in bias: per-agent vs. joint Kalman estimates of $\lambda$ | Implemented |
| `notebooks/01_sanity_checks.ipynb` | Sanity checks of the model layer (classical limit, noise, reproducibility) | DeGroot, simulator and Kalman checks passing |
| `notebooks/02_simulate_and_filter.ipynb` | Simulation and filtering exploration and plots | Empty |

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
- Config still to add: `thresholds`.

## Decisions and open questions

Decisions are logged newest first; add a row whenever the plan changes.

| Date | Decision |
| --- | --- |
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

- [ ] Do corrected neighbour inputs remove the attenuation bias in $\lambda$, and does the per-agent likelihood stay overconfident?
- [ ] Can $\sigma_e$ and $\sigma_m$ be recovered jointly, or must $\sigma_m$ stay fixed?
- [ ] Time step size for real data
- [ ] Real dataset for Phase 3
