# Learning Opinion Dynamics from Noisy Data — Project Plan

Nov 8, 2026 · @Guilherme

## Goal and status

The project builds a hidden-state algorithm that learns opinion dynamics from limited, noisy data: latent opinions evolve on a network, people only see each other's expressions, and researchers only see noisy labels of those expressions. Parameters are inferred with a particle-filter likelihood, with the hidden trajectories integrated out.

- **Current phase:** Phase 1, scalar hidden opinions on simulated data.
- **Next action:** implement the joint Kalman filter as the exact likelihood reference for DeGroot with continuous observations, then the particle filter, and compare them in `checks/check_kalman.py`.
- **Repository:** config loader, DeGroot model, simulator and sanity checks in place.

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
- Parameters: grid or optimizer first, then IF2, then PMMH.
- In an agent's own emission, expression and measurement noise integrate out in closed form, with total variance $\sigma_e^2 + \sigma_m^2$.
- Neighbours' expressions are hidden inputs: (a) per-agent filter with plug-in inputs from $Y$, fast but biased; (b) joint filter over all agents on a small network, exact reference.
- Steps without a post: propagate particles, skip weighting and resampling.

### 1.3 Evaluation

- [ ] Correctness: particle-filter log-likelihood matches the joint Kalman filter (linear model, continuous $Y$)
- [ ] Parameter recovery vs. number of steps $T$, number of agents $n$ and noise level: bias, interval coverage, smallest usable $T$
- [ ] Cost of approximations: discrete observation and plug-in inputs, version (a) vs. (b)
- [ ] Identifiability: $\sigma_e$ vs. influence parameters; multimodal posteriors under bounded confidence
- [ ] Robustness to a misspecified network: simulate on the true $W$, fit with a perturbed one (a fraction of edges rewired or missing); measure how the estimates of $\lambda$, $\sigma_p$ and $\sigma_e$ degrade as the perturbation grows

### 1.4 Realism extensions

- [ ] Heterogeneous posting rates; agents update only when exposed to new posts
- [ ] Several posts per time step (time steps as windows); weight = product over posts
- [ ] Posting rate that depends on opinion (informative silence), modeled in the likelihood

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
| `src/model.py` | Network, dynamics, expression and observation layers | DeGroot implemented (continuous observation) |
| `src/simulate.py` | Ground truth $x$, $s$, $Y$ and posting mask $P$ | Implemented (Gaussian x0, save/load) |
| `src/particle_filter.py` | Log-likelihood estimator | Empty |
| `checks/check_kalman.py` | Particle filter vs. joint Kalman filter | Empty |
| `notebooks/01_sanity_checks.ipynb` | Sanity checks of the model layer (classical limit, noise, reproducibility) | DeGroot and simulator checks passing |
| `notebooks/02_simulate_and_filter.ipynb` | Simulation and filtering exploration and plots | Empty |

- Pass `rng = np.random.default_rng(seed)` explicitly; no global randomness.
- Separate random streams for network and data (`SeedSequence(seed).spawn(2)`), so many datasets can be drawn on the same network.
- Missing posts are `NaN` in $Y$ (where $P$ is false); filters skip them.
- Every simulated dataset is saved as `.npz` with its config next to it.
- Dynamics chosen by a dispatch dictionary; unsupported models raise an error.
- One config file per experiment; save the config next to every result.
- Split `model.py` into dynamics, observation and network files when FJ arrives.
- Every new dynamics model gets its own section in `notebooks/01_sanity_checks.ipynb` (classical limit, noise behaviour, reproducibility) before it is used anywhere else.
- Config still to add: `thresholds`.

## Decisions and open questions

Decisions are logged newest first; add a row whenever the plan changes.

| Date | Decision |
| --- | --- |
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

- [ ] Plug-in vs. corrected neighbour inputs: how large is the attenuation bias?
- [ ] Can $\sigma_e$ and $\sigma_m$ be recovered jointly, or must $\sigma_m$ stay fixed?
- [ ] Time step size for real data
- [ ] Real dataset for Phase 3
