# Learning Opinion Dynamics from Noisy Data — Project Plan

Nov 8, 2026 · @Guilherme

## Goal and status

The project builds a hidden-state algorithm that learns opinion dynamics from limited, noisy data: latent opinions evolve on a network, people only see each other's expressions, and researchers only see noisy labels of those expressions. Parameters are inferred with a particle-filter likelihood, with the hidden trajectories integrated out.

- **Current phase:** Phase 1, scalar hidden opinions on simulated data.
- **Next action:** implement DeGroot with continuous observations and check the particle-filter log-likelihood against the joint Kalman filter.
- **Repository:** project skeleton and config loader in place.

## Phase 1 — Scalar hidden opinions vs. classical models

Phase 1 tests whether the algorithm recovers known parameters when data come from the classical models plus a realistic noise channel.

### 1.1 Simulated data

Each agent passes through opinion → expression → observation. Agents update only from neighbours' expressions, never from hidden opinions.

```latex
x_{i,t} = \lambda\, x_{i,t-1} + (1-\lambda)\sum_j W_{ij}\, s_{j,t-1} + \varepsilon_{i,t}, \qquad s_{i,t} = x_{i,t} + \xi_{i,t}
```

```latex
Y_{i,t} = k \iff c_{k-1} < s_{i,t} + e_{i,t} \le c_k, \qquad k = 1,\dots,5
```

- Noise: process ε \~ N(0, σp²), expression ξ \~ N(0, σe²), measurement e \~ N(0, σm²).
- Model order: DeGroot → FJ (linear) → bounded confidence (nonlinear).
- Observation switch: continuous Y = s + e, or ordinal with 5 levels (ordered probit).
- Parameters to recover: λ (or model equivalents), σp, σe. Keep σm and thresholds c fixed, assuming a labeled validation set.

### 1.2 Inference

- The particle filter computes the likelihood p(Y | ψ) and filtered opinions as a by-product.
- Parameters: grid or optimizer first, then IF2, then PMMH.
- In an agent's own emission, expression and measurement noise integrate out in closed form, with total variance σe² + σm².
- Neighbours' expressions are hidden inputs: (a) per-agent filter with plug-in inputs from Y, fast but biased; (b) joint filter over all agents on a small network, exact reference.
- Steps without a post: propagate particles, skip weighting and resampling.

### 1.3 Evaluation

- [ ] Correctness: particle-filter log-likelihood matches the joint Kalman filter (linear model, continuous Y)
- [ ] Parameter recovery vs. number of steps T, number of agents n and noise level: bias, interval coverage, smallest usable T
- [ ] Cost of approximations: discrete observation and plug-in inputs, version (a) vs. (b)
- [ ] Identifiability: σe vs. influence parameters; multimodal posteriors under bounded confidence

### 1.4 Realism extensions

- [ ] Heterogeneous posting rates; agents update only when exposed to new posts
- [ ] Several posts per time step (time steps as windows); weight = product over posts
- [ ] Posting rate that depends on opinion (informative silence), modeled in the likelihood

## Phase 2 — Beta hidden opinions

Phase 2 replaces the scalar opinion with a belief distribution, Beta(a, b), with direction m = a/(a+b) and certainty κ = a+b.

- Expression is sampled from the agent's own belief, so uncertain agents post more erratically.
- Influence through expressed signals: pseudo-count updating with forgetting, where δ is memory and η is social learning strength.

```latex
\boldsymbol\alpha_{i,t} = \delta\,\boldsymbol\alpha_{i,t-1} + \eta\sum_j W_{ij}\,\mathbf e(s_{j,t-1})
```

- Certainty settles at κ\* = η/(1−δ); classical DeGroot with self-weight δ is the limiting case.
- Process noise in (logit m, log κ) coordinates to keep a, b > 0.
- Alternative variant to compare: logarithmic pool (agents share full beliefs), α\_t = W α\_{t−1}.
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
| configs/degroot\_base.yaml | Parameter values for one run | Filled |
| src/config.py | Loads YAML into strict dataclasses | Implemented |
| src/model.py | Network, dynamics, expression and observation layers | Empty |
| src/simulate.py | Ground truth x, s, Y and posting mask P | Empty |
| src/particle\_filter.py | Log-likelihood estimator | Empty |
| checks/check\_kalman.py | Particle filter vs. joint Kalman filter | Empty |
| notebooks/01\_simulate\_and\_filter.ipynb | Exploration and plots | Empty |

- Pass rng = np.random.default\_rng(seed) explicitly; no global randomness.
- Dynamics chosen by a dispatch dictionary; unsupported models raise an error.
- One config file per experiment; save the config next to every result.
- Split model.py into dynamics, observation and network files when FJ arrives.
- Config still to add: expression\_noise\_sd, measurement\_noise\_sd, thresholds, observation (continuous | ordinal); rename stubbornness to self\_weight for DeGroot.

## Decisions and open questions

Decisions are logged newest first; add a row whenever the plan changes.

| Date | Decision |
| --- | --- |
| 2026-11-08 | Posting mask P built in from the start (all ones in Phase 1.1–1.3) |
| 2026-11-08 | Expression is scalar; only the observation is discrete (5 levels, ordered probit) |
| 2026-11-08 | First version uses scalar hidden opinions; Beta beliefs deferred to Phase 2 |
| 2026-11-08 | Agents update only from neighbours' expressions, never hidden opinions |
| 2026-11-06 | Likelihood via particle filter; start with simulation validation |

Open questions:

- [ ] Plug-in vs. corrected neighbour inputs: how large is the attenuation bias?
- [ ] Can σe and σm be recovered jointly, or must σm stay fixed?
- [ ] Time step size for real data
- [ ] Real dataset for Phase 3
